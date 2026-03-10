"""Unit tests for embedding, retrieval, and classification protocol compliance.

Tests verify that stub implementations satisfying the protocol signatures
work correctly with the type system. This validates DIP — concrete
implementations need no inheritance from Protocol classes.
"""

from talkex.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import ObjectType, PoolingStrategy
from talkex.models.types import EmbeddingId
from talkex.pipeline.protocols import (
    Classifier,
    EmbeddingGenerator,
    HybridRetriever,
    LexicalIndex,
    Reranker,
    VectorIndex,
)
from talkex.retrieval.models import (
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
)

# ---------------------------------------------------------------------------
# Stubs (pure duck typing — no Protocol inheritance)
# ---------------------------------------------------------------------------


class StubEmbeddingGenerator:
    def generate(self, batch: EmbeddingBatch) -> list[EmbeddingRecord]:
        return [
            EmbeddingRecord(
                embedding_id=EmbeddingId(f"emb_{i}"),
                source_id=item.object_id,
                source_type=item.object_type,
                model_name="test-model",
                model_version="1.0",
                pooling_strategy=PoolingStrategy.MEAN,
                dimensions=3,
                vector=[0.1, 0.2, 0.3],
            )
            for i, item in enumerate(batch.items)
        ]


class StubVectorIndex:
    def __init__(self) -> None:
        self.records: list[EmbeddingRecord] = []

    def upsert(self, records: list[EmbeddingRecord]) -> None:
        self.records.extend(records)

    def search_by_vector(self, vector: list[float], top_k: int = 10) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                object_id=r.source_id,
                object_type=r.source_type.value,
                score=0.9,
                semantic_score=0.9,
                rank=i + 1,
            )
            for i, r in enumerate(self.records[:top_k])
        ]


class StubLexicalIndex:
    def __init__(self) -> None:
        self.docs: list[dict[str, object]] = []

    def index(self, documents: list[dict[str, object]]) -> None:
        self.docs.extend(documents)

    def search(self, query_text: str, top_k: int = 10) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                object_id=str(d.get("doc_id", "")),
                object_type="context_window",
                score=0.7,
                lexical_score=0.7,
                rank=i + 1,
            )
            for i, d in enumerate(self.docs[:top_k])
        ]


class StubHybridRetriever:
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        return RetrievalResult(
            hits=[
                RetrievalHit(
                    object_id="win_0",
                    object_type="context_window",
                    score=0.85,
                    lexical_score=0.7,
                    semantic_score=0.9,
                    rank=1,
                )
            ],
            total_candidates=1,
            mode=RetrievalMode.HYBRID,
        )


class StubReranker:
    def rerank(self, query_text: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        return hits


class StubClassifier:
    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        return [
            ClassificationResult(
                source_id=inp.source_id,
                source_type=inp.source_type,
                label_scores=[
                    LabelScore(
                        label="billing",
                        score=0.85,
                        confidence=0.82,
                    )
                ],
                model_name="stub-classifier",
                model_version="1.0",
            )
            for inp in inputs
        ]


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestEmbeddingGeneratorProtocol:
    def test_stub_generates_records(self) -> None:
        gen = StubEmbeddingGenerator()
        batch = EmbeddingBatch(
            items=[
                EmbeddingInput(
                    embedding_id=EmbeddingId("emb_001"),
                    object_type=ObjectType.CONTEXT_WINDOW,
                    object_id="win_0",
                    text="hello",
                )
            ]
        )
        records = gen.generate(batch)
        assert len(records) == 1
        assert records[0].source_id == "win_0"
        assert records[0].dimensions == 3

    def test_stub_preserves_batch_order(self) -> None:
        gen = StubEmbeddingGenerator()
        batch = EmbeddingBatch(
            items=[
                EmbeddingInput(
                    embedding_id=EmbeddingId("emb_a"),
                    object_type=ObjectType.TURN,
                    object_id="turn_0",
                    text="first",
                ),
                EmbeddingInput(
                    embedding_id=EmbeddingId("emb_b"),
                    object_type=ObjectType.TURN,
                    object_id="turn_1",
                    text="second",
                ),
            ]
        )
        records = gen.generate(batch)
        assert [r.source_id for r in records] == ["turn_0", "turn_1"]


class TestVectorIndexProtocol:
    def test_upsert_and_search(self) -> None:
        idx = StubVectorIndex()
        record = EmbeddingRecord(
            embedding_id=EmbeddingId("emb_001"),
            source_id="win_0",
            source_type=ObjectType.CONTEXT_WINDOW,
            model_name="test",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
            dimensions=3,
            vector=[0.1, 0.2, 0.3],
        )
        idx.upsert([record])
        hits = idx.search_by_vector([0.1, 0.2, 0.3], top_k=5)
        assert len(hits) == 1
        assert hits[0].object_id == "win_0"


class TestLexicalIndexProtocol:
    def test_index_and_search(self) -> None:
        idx = StubLexicalIndex()
        idx.index([{"doc_id": "win_0", "text": "billing issue"}])
        hits = idx.search("billing", top_k=5)
        assert len(hits) == 1
        assert hits[0].object_id == "win_0"


class TestHybridRetrieverProtocol:
    def test_retrieve_returns_result(self) -> None:
        retriever = StubHybridRetriever()
        query = RetrievalQuery(query_text="cancel subscription")
        result = retriever.retrieve(query)
        assert len(result.hits) == 1
        assert result.mode == RetrievalMode.HYBRID

    def test_hit_has_provenance_scores(self) -> None:
        retriever = StubHybridRetriever()
        result = retriever.retrieve(RetrievalQuery(query_text="test"))
        hit = result.hits[0]
        assert hit.lexical_score is not None
        assert hit.semantic_score is not None


class TestRerankerProtocol:
    def test_rerank_returns_hits(self) -> None:
        reranker = StubReranker()
        hits = [
            RetrievalHit(object_id="w0", object_type="t", score=0.5, rank=0),
            RetrievalHit(object_id="w1", object_type="t", score=0.3, rank=1),
        ]
        result = reranker.rerank("query", hits)
        assert len(result) == 2


class TestClassifierProtocol:
    def test_stub_classifies_single_input(self) -> None:
        classifier = StubClassifier()
        inputs = [
            ClassificationInput(
                source_id="win_0",
                source_type="context_window",
                text="billing issue with credit card",
            )
        ]
        results = classifier.classify(inputs)
        assert len(results) == 1
        assert results[0].source_id == "win_0"
        assert results[0].top_label == "billing"

    def test_stub_preserves_input_order(self) -> None:
        classifier = StubClassifier()
        inputs = [
            ClassificationInput(
                source_id=f"win_{i}",
                source_type="context_window",
                text=f"text {i}",
            )
            for i in range(3)
        ]
        results = classifier.classify(inputs)
        assert [r.source_id for r in results] == ["win_0", "win_1", "win_2"]

    def test_stub_returns_label_scores(self) -> None:
        classifier = StubClassifier()
        inputs = [
            ClassificationInput(
                source_id="win_0",
                source_type="context_window",
                text="test",
            )
        ]
        results = classifier.classify(inputs)
        assert len(results[0].label_scores) == 1
        assert results[0].label_scores[0].score == 0.85


# ---------------------------------------------------------------------------
# Reexport from pipeline package
# ---------------------------------------------------------------------------


class TestProtocolReexport:
    def test_embedding_generator_from_pipeline(self) -> None:
        from talkex.pipeline import (
            EmbeddingGenerator as Imported,
        )

        assert Imported is EmbeddingGenerator

    def test_vector_index_from_pipeline(self) -> None:
        from talkex.pipeline import VectorIndex as Imported

        assert Imported is VectorIndex

    def test_lexical_index_from_pipeline(self) -> None:
        from talkex.pipeline import LexicalIndex as Imported

        assert Imported is LexicalIndex

    def test_hybrid_retriever_from_pipeline(self) -> None:
        from talkex.pipeline import HybridRetriever as Imported

        assert Imported is HybridRetriever

    def test_reranker_from_pipeline(self) -> None:
        from talkex.pipeline import Reranker as Imported

        assert Imported is Reranker

    def test_classifier_from_pipeline(self) -> None:
        from talkex.pipeline import Classifier as Imported

        assert Imported is Classifier
