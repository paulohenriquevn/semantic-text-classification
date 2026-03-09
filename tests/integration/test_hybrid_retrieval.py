"""Integration tests for hybrid retrieval pipeline.

End-to-end tests: raw text → TextProcessingPipeline → NullEmbeddingGenerator →
BM25 + Vector indexes → SimpleHybridRetriever.

Validates that the full pipeline produces correct hybrid search results
with fused scores, mode reporting, and stats.
"""

from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
from semantic_conversation_engine.embeddings.generator import NullEmbeddingGenerator
from semantic_conversation_engine.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.enums import Channel, ObjectType, PoolingStrategy
from semantic_conversation_engine.models.types import ConversationId, EmbeddingId
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.retrieval.bm25 import InMemoryBM25Index
from semantic_conversation_engine.retrieval.builders import (
    context_windows_to_lexical_docs,
)
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
from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT = (
    "Customer: I need help with my billing issue\n"
    "Agent: I'd be happy to help you with billing\n"
    "Customer: My credit card was charged twice\n"
    "Agent: Let me check your account for duplicate charges\n"
    "Customer: Also I want to cancel my subscription\n"
    "Agent: I understand, let me process the cancellation"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pipeline(
    fusion_strategy: FusionStrategy = FusionStrategy.RRF,
    fusion_weight: float = 0.5,
) -> SimpleHybridRetriever:
    """Build the full pipeline from raw text to indexed hybrid retriever."""
    segmenter = TurnSegmenter()
    builder = SlidingWindowBuilder()
    pipeline = TextProcessingPipeline(segmenter=segmenter, context_builder=builder)
    result = pipeline.run(
        TranscriptInput(
            conversation_id=ConversationId("conv_hybrid"),
            channel=Channel.CHAT,
            raw_text=_TRANSCRIPT,
            source_format=SourceFormat.LABELED,
        ),
        context_config=ContextWindowConfig(window_size=2, stride=1),
    )
    windows = result.windows

    # Generate embeddings
    gen = NullEmbeddingGenerator(
        model_config=EmbeddingModelConfig(
            model_name="null-test",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        ),
        dimensions=8,
    )
    items = [
        EmbeddingInput(
            embedding_id=EmbeddingId(f"emb_{w.window_id}"),
            object_type=ObjectType.CONTEXT_WINDOW,
            object_id=w.window_id,
            text=w.window_text,
        )
        for w in windows
    ]
    records = gen.generate(EmbeddingBatch(items=items))

    # Index both BM25 and vector
    bm25 = InMemoryBM25Index()
    bm25.index(context_windows_to_lexical_docs(windows))

    vector = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=8))
    vector.upsert(records)

    # Create retriever
    config = HybridRetrievalConfig(
        fusion_strategy=fusion_strategy,
        fusion_weight=fusion_weight,
    )
    return SimpleHybridRetriever(
        lexical_index=bm25,
        vector_index=vector,
        embedding_generator=gen,
        config=config,
    )


# ---------------------------------------------------------------------------
# End-to-end hybrid retrieval tests
# ---------------------------------------------------------------------------


class TestHybridRetrievalPipeline:
    def test_hybrid_search_returns_results(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing issue"))
        assert len(result.hits) >= 1
        assert result.mode == RetrievalMode.HYBRID

    def test_hybrid_hits_have_fused_scores(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        for hit in result.hits:
            assert hit.score > 0.0

    def test_hybrid_deduplicates_across_sources(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        ids = [h.object_id for h in result.hits]
        assert len(ids) == len(set(ids))

    def test_hybrid_ordering_contract(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_ranks_are_one_based(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        for i, hit in enumerate(result.hits):
            assert hit.rank == i + 1

    def test_hybrid_stats_completeness(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert "lexical_candidates" in result.stats
        assert "semantic_candidates" in result.stats
        assert "union_candidates" in result.stats
        assert "fusion_strategy" in result.stats
        assert "retrieval_mode" in result.stats
        assert "hybrid_latency_ms" in result.stats
        assert result.stats["retrieval_mode"] == "hybrid"


class TestHybridRetrievalModes:
    def test_lexical_only_mode_in_pipeline(self) -> None:
        retriever = _build_pipeline()
        query = RetrievalQuery(query_text="billing", query_type=QueryType.LEXICAL)
        result = retriever.retrieve(query)
        assert result.mode == RetrievalMode.LEXICAL_ONLY
        assert "semantic_ms" not in result.stats
        for hit in result.hits:
            assert hit.lexical_score is not None

    def test_semantic_only_mode_in_pipeline(self) -> None:
        retriever = _build_pipeline()
        query = RetrievalQuery(query_text="billing", query_type=QueryType.SEMANTIC)
        result = retriever.retrieve(query)
        assert result.mode == RetrievalMode.SEMANTIC_ONLY
        assert "lexical_ms" not in result.stats
        for hit in result.hits:
            assert hit.semantic_score is not None


class TestHybridRetrievalFusionStrategies:
    def test_rrf_is_default_strategy(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.stats["fusion_strategy"] == "rrf"

    def test_linear_fusion_strategy_in_pipeline(self) -> None:
        retriever = _build_pipeline(
            fusion_strategy=FusionStrategy.LINEAR,
            fusion_weight=0.6,
        )
        result = retriever.retrieve(RetrievalQuery(query_text="billing"))
        assert result.stats["fusion_strategy"] == "linear"
        assert len(result.hits) >= 1


class TestHybridRetrievalTopK:
    def test_top_k_limits_final_results(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing", top_k=2))
        assert len(result.hits) <= 2

    def test_total_candidates_exceeds_final_hits(self) -> None:
        retriever = _build_pipeline()
        result = retriever.retrieve(RetrievalQuery(query_text="billing", top_k=1))
        assert result.total_candidates >= len(result.hits)
