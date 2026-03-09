"""Retrieval data types — query, hit, and result objects.

These are the payloads exchanged between retrieval components:
queries flow in, results flow out. They are NOT domain entities —
they are pipeline-internal data objects for the retrieval stage.

RetrievalQuery defines what to search for and how.
RetrievalHit represents a single matched document with provenance.
RetrievalResult is the envelope carrying hits and retrieval metadata.

Score semantics (stable contract):
    score:          float   — always present, final score after fusion/rerank
    lexical_score:  float | None — None means BM25 was NOT consulted for
                    this hit. 0.0 means BM25 ran and returned zero.
                    NEVER use 0.0 to represent absence.
    semantic_score: float | None — None means ANN was NOT consulted for
                    this hit. 0.0 means ANN ran and returned zero.
                    NEVER use 0.0 to represent absence.

Ordering contract (stable):
    RetrievalResult.hits MUST be sorted by score descending.
    Tie-break: deterministic by object_id ascending (lexicographic).
    rank is 1-based (1 = best hit), derived from final ordering.

See ADR-002 for the frozen/strict design decision.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QueryType(StrEnum):
    """Type of retrieval query.

    LEXICAL: BM25 only.
    SEMANTIC: ANN vector search only.
    HYBRID: Both lexical and semantic, fused.
    """

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class RetrievalMode(StrEnum):
    """Actual retrieval mode used (may differ from requested due to degradation).

    Reports which indexes were actually consulted. When a requested
    mode degrades (e.g., HYBRID → LEXICAL_ONLY because ANN is unavailable),
    this field records what actually happened.
    """

    HYBRID = "hybrid"
    LEXICAL_ONLY = "lexical_only"
    SEMANTIC_ONLY = "semantic_only"


@dataclass(frozen=True)
class RetrievalFilter:
    """Structural filter for retrieval queries.

    All fields are optional — only non-None fields are applied.

    Args:
        channel: Filter by communication channel.
        queue: Filter by service queue.
        product: Filter by product or service.
        region: Filter by geographic region.
        date_from: ISO 8601 date string for range start.
        date_to: ISO 8601 date string for range end.
        object_type: Filter by object granularity.
    """

    channel: str | None = None
    queue: str | None = None
    product: str | None = None
    region: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    object_type: str | None = None


@dataclass(frozen=True)
class RetrievalQuery:
    """A retrieval query specifying what to search for.

    Args:
        query_text: The text to search for.
        top_k: Maximum number of results to return.
        query_type: Desired retrieval mode (may degrade).
        filters: Optional structural filters.
        metadata: Additional query context for routing or logging.
    """

    query_text: str
    top_k: int = 10
    query_type: QueryType = QueryType.HYBRID
    filters: RetrievalFilter | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    """A single retrieval result with provenance.

    Carries both the final fused score and the individual component
    scores for observability and debugging.

    Score semantics:
        - ``score`` is always present — the final fused/reranked score.
        - ``lexical_score`` is None when BM25 was NOT consulted for this
          hit (e.g., semantic-only mode or degradation). A value of 0.0
          means BM25 ran and returned zero — NOT absence.
        - ``semantic_score`` is None when ANN was NOT consulted. Same
          0.0 vs None distinction applies.

    Ordering:
        Hits in a RetrievalResult are sorted by ``score`` descending,
        with tie-break by ``object_id`` ascending (lexicographic).
        ``rank`` is 1-based (1 = best hit).

    Args:
        object_id: ID of the matched object.
        object_type: Granularity level of the matched object.
        score: Final score after fusion and/or reranking. Always present.
        lexical_score: BM25 score. None if BM25 was not consulted.
        semantic_score: Vector similarity score. None if ANN was not
            consulted.
        rank: Position in the final result list (1-based, 1 = best).
        metadata: Additional hit metadata (text snippet, source, etc.).
    """

    object_id: str
    object_type: str
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None
    rank: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """Envelope for retrieval results.

    Carries the hits and metadata about the retrieval execution,
    including the actual mode used (which may differ from requested
    due to graceful degradation).

    Ordering contract:
        ``hits`` MUST be sorted by ``score`` descending. Ties are
        broken by ``object_id`` ascending (lexicographic). Each hit's
        ``rank`` field reflects this final ordering (1-based).

    Args:
        hits: Ordered list of retrieval hits (by descending score,
            tie-break by object_id ascending).
        total_candidates: Total candidates before final top-k cutoff.
        mode: Actual retrieval mode used (may differ from requested
            due to graceful degradation).
        stats: Operational statistics (timing, index sizes, etc.).
    """

    hits: list[RetrievalHit]
    total_candidates: int = 0
    mode: RetrievalMode = RetrievalMode.HYBRID
    stats: dict[str, Any] = field(default_factory=dict)
