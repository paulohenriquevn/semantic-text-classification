"""Unit tests for hybrid retriever.

Tests cover: mode routing (LEXICAL/SEMANTIC/HYBRID), graceful degradation,
deduplication, RRF fusion, ordering contract, score preservation, stats,
top_k enforcement, and protocol compliance.
"""

from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
from semantic_conversation_engine.embeddings.generator import NullEmbeddingGenerator
from semantic_conversation_engine.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from semantic_conversation_engine.models.embedding_record import EmbeddingRecord
from semantic_conversation_engine.models.enums import ObjectType, PoolingStrategy
from semantic_conversation_engine.models.types import EmbeddingId
from semantic_conversation_engine.retrieval.bm25 import InMemoryBM25Index
from semantic_conversation_engine.retrieval.config import (
    FusionStrategy,
    HybridRetrievalConfig,
    VectorIndexConfig,
)
from semantic_conversation_engine.retrieval.hybrid import SimpleHybridRetriever
from semantic_conversation_engine.retrieval.models import (
    QueryType,
    RetrievalMode,
    RetrievalQuery,
)
from semantic_conversation_engine.retrieval.vector_index import InMemoryVectorIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_config() -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        model_name="null-test",
        model_version="1.0",
        pooling_strategy=PoolingStrategy.MEAN,
    )


def _generator() -> NullEmbeddingGenerator:
    return NullEmbeddingGenerator(model_config=_model_config(), dimensions=8)


def _embed(gen: NullEmbeddingGenerator, text: str, obj_id: str) -> EmbeddingRecord:
    batch = EmbeddingBatch(
        items=[
            EmbeddingInput(
                embedding_id=EmbeddingId(f"emb_{obj_id}"),
                object_type=ObjectType.CONTEXT_WINDOW,
                object_id=obj_id,
                text=text,
            )
        ]
    )
    return gen.generate(batch)[0]


def _populated_indexes() -> tuple[InMemoryBM25Index, InMemoryVectorIndex, NullEmbeddingGenerator]:
    """Create BM25 + vector indexes with 3 documents."""
    gen = _generator()
    docs = [
        ("w0", "billing issue with credit card payment"),
        ("w1", "cancel subscription immediately please"),
        ("w2", "billing charge duplicate refund request"),
    ]

    bm25 = InMemoryBM25Index()
    bm25.index([{"doc_id": did, "text": text} for did, text in docs])

    vector = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=8))
    records = [_embed(gen, text, did) for did, text in docs]
    vector.upsert(records)

    return bm25, vector, gen


# ---------------------------------------------------------------------------
# Hybrid mode
# ---------------------------------------------------------------------------


class TestHybridMode:
    def test_hybrid_query_uses_both_indexes(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.mode == RetrievalMode.HYBRID
        assert len(result.hits) >= 1

    def test_hybrid_hits_have_fused_score(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        for hit in result.hits:
            assert hit.score > 0.0

    def test_hybrid_deduplicates_by_object_id(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        ids = [h.object_id for h in result.hits]
        assert len(ids) == len(set(ids))  # No duplicates


# ---------------------------------------------------------------------------
# Lexical-only mode
# ---------------------------------------------------------------------------


class TestLexicalOnlyMode:
    def test_lexical_query_skips_vector(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        query = RetrievalQuery(query_text="billing", query_type=QueryType.LEXICAL)
        result = retriever.retrieve(query)
        assert result.mode == RetrievalMode.LEXICAL_ONLY
        assert "semantic_ms" not in result.stats

    def test_lexical_hits_have_lexical_score(self) -> None:
        bm25, _, _ = _populated_indexes()
        retriever = SimpleHybridRetriever(lexical_index=bm25)
        query = RetrievalQuery(query_text="billing", query_type=QueryType.LEXICAL)
        result = retriever.retrieve(query)
        for hit in result.hits:
            assert hit.lexical_score is not None


# ---------------------------------------------------------------------------
# Semantic-only mode
# ---------------------------------------------------------------------------


class TestSemanticOnlyMode:
    def test_semantic_query_skips_lexical(self) -> None:
        _, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            vector_index=vector,
            embedding_generator=gen,
        )
        query = RetrievalQuery(query_text="billing", query_type=QueryType.SEMANTIC)
        result = retriever.retrieve(query)
        assert result.mode == RetrievalMode.SEMANTIC_ONLY
        assert "lexical_ms" not in result.stats

    def test_semantic_hits_have_semantic_score(self) -> None:
        _, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            vector_index=vector,
            embedding_generator=gen,
        )
        query = RetrievalQuery(query_text="billing", query_type=QueryType.SEMANTIC)
        result = retriever.retrieve(query)
        for hit in result.hits:
            assert hit.semantic_score is not None


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_hybrid_degrades_to_lexical_when_no_vector(self) -> None:
        bm25, _, _ = _populated_indexes()
        retriever = SimpleHybridRetriever(lexical_index=bm25)
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.mode == RetrievalMode.LEXICAL_ONLY

    def test_hybrid_degrades_to_semantic_when_no_lexical(self) -> None:
        _, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.mode == RetrievalMode.SEMANTIC_ONLY

    def test_no_indexes_returns_empty(self) -> None:
        retriever = SimpleHybridRetriever()
        result = retriever.retrieve(RetrievalQuery(query_text="anything"))
        assert result.hits == []

    def test_vector_without_generator_degrades(self) -> None:
        _, vector, _ = _populated_indexes()
        retriever = SimpleHybridRetriever(vector_index=vector)
        # No embedding_generator → can't embed query → no semantic search
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.hits == []


# ---------------------------------------------------------------------------
# Ordering contract
# ---------------------------------------------------------------------------


class TestHybridOrderingContract:
    def test_scores_descending(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_rank_is_one_based(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        for i, hit in enumerate(result.hits):
            assert hit.rank == i + 1


# ---------------------------------------------------------------------------
# Score preservation
# ---------------------------------------------------------------------------


class TestHybridScorePreservation:
    def test_hybrid_hit_from_both_has_both_scores(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        # At least one hit should have appeared in both indexes
        hits_with_both = [h for h in result.hits if h.lexical_score is not None and h.semantic_score is not None]
        assert len(hits_with_both) >= 1


# ---------------------------------------------------------------------------
# Top-k enforcement
# ---------------------------------------------------------------------------


class TestHybridTopK:
    def test_final_top_k_limits_results(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing", top_k=1))
        assert len(result.hits) <= 1

    def test_total_candidates_before_cutoff(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing", top_k=1))
        assert result.total_candidates >= len(result.hits)


# ---------------------------------------------------------------------------
# Stats / observability
# ---------------------------------------------------------------------------


class TestHybridStats:
    def test_hybrid_stats_contain_candidate_counts(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert "lexical_candidates" in result.stats
        assert "semantic_candidates" in result.stats
        assert "union_candidates" in result.stats

    def test_hybrid_stats_contain_latency(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert "hybrid_latency_ms" in result.stats
        assert result.stats["hybrid_latency_ms"] >= 0.0

    def test_hybrid_stats_contain_fusion_strategy(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.stats["fusion_strategy"] == "rrf"

    def test_hybrid_stats_contain_retrieval_mode(self) -> None:
        bm25, vector, gen = _populated_indexes()
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.stats["retrieval_mode"] == "hybrid"


# ---------------------------------------------------------------------------
# Config: fusion strategy
# ---------------------------------------------------------------------------


class TestHybridFusionConfig:
    def test_linear_fusion_strategy(self) -> None:
        bm25, vector, gen = _populated_indexes()
        config = HybridRetrievalConfig(
            fusion_strategy=FusionStrategy.LINEAR,
            fusion_weight=0.7,
        )
        retriever = SimpleHybridRetriever(
            lexical_index=bm25,
            vector_index=vector,
            embedding_generator=gen,
            config=config,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.stats["fusion_strategy"] == "linear"
        assert len(result.hits) >= 1


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestHybridReexport:
    def test_importable_from_retrieval_package(self) -> None:
        from semantic_conversation_engine.retrieval import (
            SimpleHybridRetriever as SHR,
        )

        assert SHR is SimpleHybridRetriever
