"""Hybrid retriever — orchestrates lexical + semantic search with fusion.

Implements the HybridRetriever protocol by composing:
    1. LexicalIndex (BM25 search)
    2. VectorIndex (ANN/exact vector search)
    3. EmbeddingGenerator (query text → query vector)
    4. Fusion strategy (RRF or LINEAR)
    5. Optional Reranker (deferred — not invoked in this foundation)

The retriever routes by QueryType:
    - LEXICAL: BM25 only → mode=LEXICAL_ONLY
    - SEMANTIC: vector only → mode=SEMANTIC_ONLY
    - HYBRID: both → fuse → mode=HYBRID

Graceful degradation: if a component is None (not provided), the
retriever degrades to the available component and reports the actual
mode in RetrievalResult.mode.

Design decisions:
    - Query embedding uses the SAME generator that indexed the documents,
      ensuring vectors live in the same embedding space.
    - Fusion is delegated to pure functions in fusion.py.
    - Stats capture candidate counts, timing, and strategy for
      observability.
    - Reranker slot exists but is not invoked until Sprint 3.4+.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from talkex.models.enums import ObjectType
from talkex.models.types import EmbeddingId
from talkex.retrieval.config import (
    FusionStrategy,
    HybridRetrievalConfig,
)
from talkex.retrieval.fusion import (
    linear_fusion,
    reciprocal_rank_fusion,
)
from talkex.retrieval.models import (
    QueryType,
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

# Protocol-compatible types — use structural subtyping, no imports
# from protocols.py to avoid circular dependencies. The type checker
# verifies structural compatibility at usage sites.


class _LexicalIndex(Protocol):
    def search(self, query_text: str, top_k: int = 10) -> list[RetrievalHit]: ...


class _VectorIndex(Protocol):
    def search_by_vector(self, vector: list[float], top_k: int = 10) -> list[RetrievalHit]: ...


class _EmbeddingGenerator(Protocol):
    def generate(self, batch: EmbeddingBatch) -> list[Any]: ...


class _Reranker(Protocol):
    def rerank(self, query_text: str, hits: list[RetrievalHit]) -> list[RetrievalHit]: ...


@dataclass
class SimpleHybridRetriever:
    """Hybrid retriever combining lexical and semantic search.

    Satisfies the HybridRetriever protocol. Routes queries by type,
    applies fusion, and reports actual retrieval mode.

    Args:
        lexical_index: BM25 index for lexical search. None to disable.
        vector_index: Vector index for semantic search. None to disable.
        embedding_generator: Generator for converting query text to
            query vector. Required when vector_index is provided.
        reranker: Optional cross-encoder reranker. Not invoked in
            this foundation — slot reserved for future use.
        config: Hybrid retrieval configuration.
    """

    lexical_index: _LexicalIndex | None = None
    vector_index: _VectorIndex | None = None
    embedding_generator: _EmbeddingGenerator | None = None
    reranker: _Reranker | None = None
    config: HybridRetrievalConfig = field(default_factory=HybridRetrievalConfig)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Execute a retrieval query with mode routing and fusion.

        Args:
            query: Retrieval query with text, type, and parameters.

        Returns:
            RetrievalResult with ordered hits and execution metadata.
        """
        start = time.monotonic()
        stats: dict[str, Any] = {}

        requested_type = query.query_type
        top_k = query.top_k or self.config.final_top_k

        lexical_hits: list[RetrievalHit] = []
        semantic_hits: list[RetrievalHit] = []
        mode: RetrievalMode

        should_lex = requested_type in (QueryType.LEXICAL, QueryType.HYBRID)
        should_sem = requested_type in (QueryType.SEMANTIC, QueryType.HYBRID)

        can_lex = self.lexical_index is not None
        can_sem = self.vector_index is not None and self.embedding_generator is not None

        # Execute lexical search
        if should_lex and can_lex:
            assert self.lexical_index is not None
            lex_start = time.monotonic()
            lexical_hits = self.lexical_index.search(query.query_text, top_k=self.config.lexical_top_k)
            stats["lexical_ms"] = round((time.monotonic() - lex_start) * 1000, 2)
            stats["lexical_candidates"] = len(lexical_hits)

        # Execute semantic search
        if should_sem and can_sem:
            assert self.vector_index is not None
            assert self.embedding_generator is not None
            sem_start = time.monotonic()
            query_vector = self._embed_query(query.query_text)
            semantic_hits = self.vector_index.search_by_vector(query_vector, top_k=self.config.vector_top_k)
            stats["semantic_ms"] = round((time.monotonic() - sem_start) * 1000, 2)
            stats["semantic_candidates"] = len(semantic_hits)

        # Determine actual mode
        has_lex = len(lexical_hits) > 0 or (should_lex and can_lex)
        has_sem = len(semantic_hits) > 0 or (should_sem and can_sem)

        if has_lex and has_sem:
            mode = RetrievalMode.HYBRID
        elif has_lex:
            mode = RetrievalMode.LEXICAL_ONLY
        elif has_sem:
            mode = RetrievalMode.SEMANTIC_ONLY
        else:
            mode = RetrievalMode.HYBRID  # No results, default

        # Fuse results
        if lexical_hits and semantic_hits:
            fused = self._fuse(lexical_hits, semantic_hits)
            stats["fusion_strategy"] = self.config.fusion_strategy.value
        elif lexical_hits:
            fused = self._passthrough_ranked(lexical_hits)
        elif semantic_hits:
            fused = self._passthrough_ranked(semantic_hits)
        else:
            fused = []

        total_candidates = len(fused)
        stats["union_candidates"] = total_candidates

        # Apply final top_k cutoff
        final_hits = fused[:top_k]
        # Re-rank after cutoff
        final_hits = self._assign_ranks(final_hits)

        stats["retrieval_mode"] = mode.value
        elapsed = (time.monotonic() - start) * 1000
        stats["hybrid_latency_ms"] = round(elapsed, 2)

        logger.debug(
            "HybridRetriever: mode=%s, lex=%d, sem=%d, fused=%d, final=%d, ms=%.2f",
            mode.value,
            len(lexical_hits),
            len(semantic_hits),
            total_candidates,
            len(final_hits),
            elapsed,
        )

        return RetrievalResult(
            hits=final_hits,
            total_candidates=total_candidates,
            mode=mode,
            stats=stats,
        )

    def _embed_query(self, query_text: str) -> list[float]:
        """Convert query text to a vector using the embedding generator.

        Args:
            query_text: Raw query text.

        Returns:
            Query vector as list[float].
        """
        assert self.embedding_generator is not None
        inp = EmbeddingInput(
            embedding_id=EmbeddingId("query_temp"),
            object_type=ObjectType.CONTEXT_WINDOW,
            object_id="query",
            text=query_text,
        )
        batch = EmbeddingBatch(items=[inp])
        records = self.embedding_generator.generate(batch)
        result: list[float] = records[0].vector
        return result

    def _fuse(
        self,
        lexical_hits: list[RetrievalHit],
        semantic_hits: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Apply configured fusion strategy.

        Args:
            lexical_hits: BM25 results.
            semantic_hits: Vector search results.

        Returns:
            Fused and sorted hits.
        """
        if self.config.fusion_strategy == FusionStrategy.RRF:
            return reciprocal_rank_fusion(lexical_hits, semantic_hits, k=self.config.rrf_k)
        elif self.config.fusion_strategy == FusionStrategy.LINEAR:
            return linear_fusion(
                lexical_hits,
                semantic_hits,
                semantic_weight=self.config.fusion_weight,
            )
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.config.fusion_strategy}")

    @staticmethod
    def _passthrough_ranked(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Re-sort single-source hits to enforce ordering contract.

        Args:
            hits: Hits from a single source.

        Returns:
            Hits sorted by score descending, object_id ascending.
        """
        sorted_hits = sorted(hits, key=lambda h: (-h.score, h.object_id))
        return [
            RetrievalHit(
                object_id=h.object_id,
                object_type=h.object_type,
                score=h.score,
                lexical_score=h.lexical_score,
                semantic_score=h.semantic_score,
                rank=rank,
                metadata=dict(h.metadata),
            )
            for rank, h in enumerate(sorted_hits, start=1)
        ]

    @staticmethod
    def _assign_ranks(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Reassign 1-based ranks to an already-sorted hit list.

        Args:
            hits: Sorted hits needing rank update.

        Returns:
            New hits with correct rank values.
        """
        return [
            RetrievalHit(
                object_id=h.object_id,
                object_type=h.object_type,
                score=h.score,
                lexical_score=h.lexical_score,
                semantic_score=h.semantic_score,
                rank=rank,
                metadata=dict(h.metadata),
            )
            for rank, h in enumerate(hits, start=1)
        ]
