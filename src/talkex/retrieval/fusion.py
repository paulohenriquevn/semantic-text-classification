"""Score fusion strategies for hybrid retrieval.

Pure functions that combine lexical and semantic retrieval hits into
a unified ranked list. No state, no indexes — just math on hit lists.

Supported strategies:
    RRF (Reciprocal Rank Fusion):
        score = Σ 1/(k + rank_i) for each system that found the document.
        Robust when lexical and semantic scores have different scales.
        Does not require score normalization.

    LINEAR (Weighted Linear Combination):
        score = w * semantic_score + (1 - w) * lexical_score
        Requires min-max normalization of raw scores to [0, 1] before
        blending. Use with caution — sensitive to score distributions.

Design decisions:
    - Deduplication by object_id happens BEFORE fusion.
    - Component scores (lexical_score, semantic_score) are PRESERVED
      in the fused hit for observability — never discarded.
    - Ordering contract enforced: score descending, tie-break by
      object_id ascending, rank 1-based.
"""

from __future__ import annotations

from talkex.retrieval.models import RetrievalHit


def reciprocal_rank_fusion(
    lexical_hits: list[RetrievalHit],
    semantic_hits: list[RetrievalHit],
    k: int = 60,
) -> list[RetrievalHit]:
    """Fuse lexical and semantic hits using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1/(k + rank_i) for each list containing d.

    Hits are deduplicated by object_id. Component scores from both
    lists are merged into the final hit.

    Args:
        lexical_hits: Hits from BM25, ordered by lexical_score descending.
        semantic_hits: Hits from ANN, ordered by semantic_score descending.
        k: RRF smoothing constant (default 60). Higher values reduce
            the impact of top ranks.

    Returns:
        Fused hits sorted by RRF score descending, tie-break by
        object_id ascending. Rank is 1-based.
    """
    merged: dict[str, _MergedHit] = {}

    for hit in lexical_hits:
        entry = merged.setdefault(
            hit.object_id,
            _MergedHit(object_id=hit.object_id, object_type=hit.object_type),
        )
        entry.rrf_score += 1.0 / (k + hit.rank)
        entry.lexical_score = hit.lexical_score
        entry.metadata.update(hit.metadata)

    for hit in semantic_hits:
        entry = merged.setdefault(
            hit.object_id,
            _MergedHit(object_id=hit.object_id, object_type=hit.object_type),
        )
        entry.rrf_score += 1.0 / (k + hit.rank)
        entry.semantic_score = hit.semantic_score
        entry.metadata.update(hit.metadata)

    return _to_ranked_hits(merged)


def linear_fusion(
    lexical_hits: list[RetrievalHit],
    semantic_hits: list[RetrievalHit],
    semantic_weight: float = 0.5,
) -> list[RetrievalHit]:
    """Fuse lexical and semantic hits using weighted linear combination.

    Applies min-max normalization to raw scores within each list,
    then combines: score = w * sem_norm + (1 - w) * lex_norm.

    For hits appearing in only one list, the missing component
    contributes 0.0 to the weighted sum.

    Args:
        lexical_hits: Hits from BM25.
        semantic_hits: Hits from ANN.
        semantic_weight: Weight for the semantic component in [0, 1].
            Lexical weight is (1 - semantic_weight).

    Returns:
        Fused hits sorted by linear score descending, tie-break by
        object_id ascending. Rank is 1-based.
    """
    lex_norm = _min_max_normalize(lexical_hits)
    sem_norm = _min_max_normalize(semantic_hits)
    lex_weight = 1.0 - semantic_weight

    merged: dict[str, _MergedHit] = {}

    for hit, norm_score in zip(lexical_hits, lex_norm, strict=True):
        entry = merged.setdefault(
            hit.object_id,
            _MergedHit(object_id=hit.object_id, object_type=hit.object_type),
        )
        entry.lex_normalized = norm_score
        entry.lexical_score = hit.lexical_score
        entry.metadata.update(hit.metadata)

    for hit, norm_score in zip(semantic_hits, sem_norm, strict=True):
        entry = merged.setdefault(
            hit.object_id,
            _MergedHit(object_id=hit.object_id, object_type=hit.object_type),
        )
        entry.sem_normalized = norm_score
        entry.semantic_score = hit.semantic_score
        entry.metadata.update(hit.metadata)

    for entry in merged.values():
        entry.rrf_score = semantic_weight * entry.sem_normalized + lex_weight * entry.lex_normalized

    return _to_ranked_hits(merged)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _MergedHit:
    """Mutable accumulator for merging hits from multiple sources."""

    __slots__ = (
        "lex_normalized",
        "lexical_score",
        "metadata",
        "object_id",
        "object_type",
        "rrf_score",
        "sem_normalized",
        "semantic_score",
    )

    def __init__(self, object_id: str, object_type: str) -> None:
        self.object_id = object_id
        self.object_type = object_type
        self.rrf_score: float = 0.0
        self.lexical_score: float | None = None
        self.semantic_score: float | None = None
        self.lex_normalized: float = 0.0
        self.sem_normalized: float = 0.0
        self.metadata: dict[str, object] = {}


def _to_ranked_hits(merged: dict[str, _MergedHit]) -> list[RetrievalHit]:
    """Convert merged entries to sorted, ranked RetrievalHits."""
    entries = sorted(
        merged.values(),
        key=lambda e: (-e.rrf_score, e.object_id),
    )
    return [
        RetrievalHit(
            object_id=e.object_id,
            object_type=e.object_type,
            score=round(e.rrf_score, 6),
            lexical_score=e.lexical_score,
            semantic_score=e.semantic_score,
            rank=rank,
            metadata=dict(e.metadata),
        )
        for rank, e in enumerate(entries, start=1)
    ]


def _min_max_normalize(hits: list[RetrievalHit]) -> list[float]:
    """Min-max normalize scores to [0, 1].

    If all scores are identical (or list is empty), returns all 1.0
    (each hit is equally the best candidate in its set).

    Args:
        hits: Hits with score values.

    Returns:
        List of normalized scores, same order as input.
    """
    if not hits:
        return []
    scores = [h.score for h in hits]
    min_s = min(scores)
    max_s = max(scores)
    span = max_s - min_s
    if span == 0.0:
        return [1.0] * len(hits)
    return [(s - min_s) / span for s in scores]
