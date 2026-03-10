"""Integration tests for index pipeline: ContextWindow → Embedding → Index → Search.

Uses real TurnSegmenter + SlidingWindowBuilder + NullEmbeddingGenerator
to produce embeddings, then indexes and searches with both BM25 and
vector indexes. Validates the full data flow from raw text to retrieval hits.
"""

from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig
from talkex.embeddings.config import EmbeddingModelConfig
from talkex.embeddings.generator import NullEmbeddingGenerator
from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel, ObjectType, PoolingStrategy
from talkex.models.types import ConversationId, EmbeddingId
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.retrieval.bm25 import InMemoryBM25Index
from talkex.retrieval.builders import (
    context_windows_to_lexical_docs,
)
from talkex.retrieval.config import VectorIndexConfig
from talkex.retrieval.vector_index import InMemoryVectorIndex
from talkex.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT = (
    "Customer: I need help with my billing issue\n"
    "Agent: I'd be happy to help you with billing\n"
    "Customer: My credit card was charged twice\n"
    "Agent: Let me check your account for duplicate charges\n"
    "Customer: Also I want to cancel my subscription\n"
    "Agent: I understand, let me process the cancellation"
)


def _run_pipeline() -> tuple[list, list]:
    """Run text processing pipeline and return (turns, windows)."""
    segmenter = TurnSegmenter()
    builder = SlidingWindowBuilder()
    pipeline = TextProcessingPipeline(segmenter=segmenter, context_builder=builder)
    result = pipeline.run(
        TranscriptInput(
            conversation_id=ConversationId("conv_1"),
            channel=Channel.CHAT,
            raw_text=_TRANSCRIPT,
            source_format=SourceFormat.LABELED,
        ),
        context_config=ContextWindowConfig(window_size=3, stride=2),
    )
    return result.turns, result.windows


def _generate_embeddings(windows: list) -> list:
    """Generate embeddings for context windows using NullEmbeddingGenerator."""
    gen = NullEmbeddingGenerator(
        model_config=EmbeddingModelConfig(
            model_name="null-test",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        ),
        dimensions=16,
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
    batch = EmbeddingBatch(items=items)
    return gen.generate(batch)


# ---------------------------------------------------------------------------
# BM25 index integration
# ---------------------------------------------------------------------------


class TestBM25IndexPipeline:
    def test_index_and_search_context_windows(self) -> None:
        _, windows = _run_pipeline()
        docs = context_windows_to_lexical_docs(windows)
        idx = InMemoryBM25Index()
        idx.index(docs)
        assert idx.document_count == len(windows)

        hits = idx.search("billing", top_k=5)
        assert len(hits) >= 1
        assert all(h.lexical_score is not None for h in hits)
        assert all(h.semantic_score is None for h in hits)

    def test_billing_ranked_above_cancellation(self) -> None:
        _, windows = _run_pipeline()
        docs = context_windows_to_lexical_docs(windows)
        idx = InMemoryBM25Index()
        idx.index(docs)
        hits = idx.search("billing credit card charges")
        assert len(hits) >= 1
        # First hit should contain billing-related text
        top_doc_id = hits[0].object_id
        top_window = next(w for w in windows if w.window_id == top_doc_id)
        assert "billing" in top_window.window_text.lower() or "credit" in top_window.window_text.lower()

    def test_cancellation_query(self) -> None:
        _, windows = _run_pipeline()
        docs = context_windows_to_lexical_docs(windows)
        idx = InMemoryBM25Index()
        idx.index(docs)
        hits = idx.search("cancel subscription")
        assert len(hits) >= 1

    def test_hit_object_ids_match_windows(self) -> None:
        _, windows = _run_pipeline()
        window_ids = {w.window_id for w in windows}
        docs = context_windows_to_lexical_docs(windows)
        idx = InMemoryBM25Index()
        idx.index(docs)
        hits = idx.search("billing")
        for hit in hits:
            assert hit.object_id in window_ids


# ---------------------------------------------------------------------------
# Vector index integration
# ---------------------------------------------------------------------------


class TestVectorIndexPipeline:
    def test_index_and_search_embeddings(self) -> None:
        _, windows = _run_pipeline()
        records = _generate_embeddings(windows)
        idx = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=16))
        idx.upsert(records)
        assert idx.vector_count == len(windows)

        query_vector = records[0].vector
        hits = idx.search_by_vector(query_vector, top_k=5)
        assert len(hits) >= 1
        # Self-search should rank the source document first
        assert hits[0].object_id == records[0].source_id

    def test_all_hits_have_semantic_score(self) -> None:
        _, windows = _run_pipeline()
        records = _generate_embeddings(windows)
        idx = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=16))
        idx.upsert(records)
        hits = idx.search_by_vector(records[0].vector)
        assert all(h.semantic_score is not None for h in hits)
        assert all(h.lexical_score is None for h in hits)

    def test_hit_object_ids_match_records(self) -> None:
        _, windows = _run_pipeline()
        records = _generate_embeddings(windows)
        source_ids = {r.source_id for r in records}
        idx = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=16))
        idx.upsert(records)
        hits = idx.search_by_vector(records[0].vector)
        for hit in hits:
            assert hit.object_id in source_ids

    def test_deterministic_results(self) -> None:
        _, windows = _run_pipeline()
        records = _generate_embeddings(windows)

        idx1 = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=16))
        idx1.upsert(records)
        hits1 = idx1.search_by_vector(records[0].vector)

        idx2 = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=16))
        idx2.upsert(records)
        hits2 = idx2.search_by_vector(records[0].vector)

        assert [h.object_id for h in hits1] == [h.object_id for h in hits2]
        assert [h.score for h in hits1] == [h.score for h in hits2]


# ---------------------------------------------------------------------------
# Save/load integration
# ---------------------------------------------------------------------------


class TestVectorIndexPersistence:
    def test_save_load_preserves_search_results(self, tmp_path: object) -> None:
        _, windows = _run_pipeline()
        records = _generate_embeddings(windows)

        idx = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=16))
        idx.upsert(records)
        hits_before = idx.search_by_vector(records[0].vector)
        idx.save(str(tmp_path))

        idx2 = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=16))
        idx2.load(str(tmp_path))
        hits_after = idx2.search_by_vector(records[0].vector)

        assert [h.object_id for h in hits_before] == [h.object_id for h in hits_after]


# ---------------------------------------------------------------------------
# Both indexes on same data
# ---------------------------------------------------------------------------


class TestDualIndexPipeline:
    def test_both_indexes_return_valid_hits(self) -> None:
        _, windows = _run_pipeline()
        records = _generate_embeddings(windows)

        bm25 = InMemoryBM25Index()
        bm25.index(context_windows_to_lexical_docs(windows))

        vector = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=16))
        vector.upsert(records)

        lex_hits = bm25.search("billing")
        sem_hits = vector.search_by_vector(records[0].vector)

        assert len(lex_hits) >= 1
        assert len(sem_hits) >= 1
        # Both should return valid window IDs
        window_ids = {w.window_id for w in windows}
        for h in lex_hits:
            assert h.object_id in window_ids
        for h in sem_hits:
            assert h.object_id in window_ids
