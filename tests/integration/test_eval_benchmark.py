"""Integration tests for retrieval evaluation benchmark.

End-to-end: build pipeline → create evaluation dataset → run benchmark
across BM25, VECTOR, HYBRID_RRF strategies → compare results.
"""

from pathlib import Path

from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
from semantic_conversation_engine.embeddings.generator import NullEmbeddingGenerator
from semantic_conversation_engine.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from semantic_conversation_engine.evaluation.dataset import (
    EvaluationDataset,
    EvaluationExample,
    RelevanceJudgment,
)
from semantic_conversation_engine.evaluation.runner import BenchmarkRunner, RunConfig
from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.enums import Channel, ObjectType, PoolingStrategy
from semantic_conversation_engine.models.types import ConversationId, EmbeddingId
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.retrieval.bm25 import InMemoryBM25Index
from semantic_conversation_engine.retrieval.builders import (
    context_windows_to_lexical_docs,
)
from semantic_conversation_engine.retrieval.config import VectorIndexConfig
from semantic_conversation_engine.retrieval.hybrid import SimpleHybridRetriever
from semantic_conversation_engine.retrieval.vector_index import InMemoryVectorIndex
from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT = (
    "Customer: I have a billing issue with my credit card payment\n"
    "Agent: I can help you with that billing concern\n"
    "Customer: My card was charged twice for the same order\n"
    "Agent: Let me check your account for duplicate charges\n"
    "Customer: I also want to cancel my subscription\n"
    "Agent: I understand, let me process the cancellation\n"
    "Customer: When will I receive my refund?\n"
    "Agent: Refunds typically take 5 to 7 business days\n"
)


def _build_retrievers() -> tuple[dict[str, SimpleHybridRetriever], list]:
    """Build BM25-only, vector-only, and hybrid retrievers from raw text."""
    segmenter = TurnSegmenter()
    builder = SlidingWindowBuilder()
    pipeline = TextProcessingPipeline(segmenter=segmenter, context_builder=builder)
    result = pipeline.run(
        TranscriptInput(
            conversation_id=ConversationId("conv_eval"),
            channel=Channel.CHAT,
            raw_text=_TRANSCRIPT,
            source_format=SourceFormat.LABELED,
        ),
        context_config=ContextWindowConfig(window_size=2, stride=1),
    )
    windows = result.windows

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

    bm25 = InMemoryBM25Index()
    bm25.index(context_windows_to_lexical_docs(windows))

    vector = InMemoryVectorIndex(config=VectorIndexConfig(dimensions=8))
    vector.upsert(records)

    bm25_only = SimpleHybridRetriever(lexical_index=bm25)
    vector_only = SimpleHybridRetriever(vector_index=vector, embedding_generator=gen)
    hybrid = SimpleHybridRetriever(
        lexical_index=bm25,
        vector_index=vector,
        embedding_generator=gen,
    )

    return {"BM25": bm25_only, "VECTOR": vector_only, "HYBRID_RRF": hybrid}, windows


def _build_eval_dataset(windows: list) -> EvaluationDataset:
    """Create a simple evaluation dataset using actual window IDs."""
    # Use first two windows as relevant for "billing" query
    billing_relevant = [w.window_id for w in windows if "billing" in w.window_text.lower()]
    cancel_relevant = [w.window_id for w in windows if "cancel" in w.window_text.lower()]

    examples = []
    if billing_relevant:
        examples.append(
            EvaluationExample(
                query_id="q_billing",
                query_text="billing issue credit card",
                relevant_docs=[RelevanceJudgment(document_id=wid) for wid in billing_relevant],
            )
        )
    if cancel_relevant:
        examples.append(
            EvaluationExample(
                query_id="q_cancel",
                query_text="cancel subscription",
                relevant_docs=[RelevanceJudgment(document_id=wid) for wid in cancel_relevant],
            )
        )

    return EvaluationDataset(
        name="integration-test-eval",
        version="1.0",
        examples=examples,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestEvalBenchmarkPipeline:
    def test_single_retriever_evaluation(self) -> None:
        retrievers, windows = _build_retrievers()
        dataset = _build_eval_dataset(windows)
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[3, 5, 10]),
        )
        result = runner.evaluate(retrievers["BM25"], "BM25")
        assert result.method_name == "BM25"
        assert result.total_queries == len(dataset.examples)
        assert "mrr" in result.aggregated
        assert "recall@5" in result.aggregated

    def test_multi_retriever_comparison(self) -> None:
        retrievers, windows = _build_retrievers()
        dataset = _build_eval_dataset(windows)
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5, 10]),
        )
        report = runner.compare(retrievers)
        assert len(report.results) == 3
        names = {r.method_name for r in report.results}
        assert names == {"BM25", "VECTOR", "HYBRID_RRF"}

    def test_report_json_serializable(self) -> None:
        retrievers, windows = _build_retrievers()
        dataset = _build_eval_dataset(windows)
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5]),
        )
        report = runner.compare(retrievers)
        json_str = report.to_json()
        assert "BM25" in json_str
        assert "HYBRID_RRF" in json_str

    def test_report_csv_serializable(self) -> None:
        retrievers, windows = _build_retrievers()
        dataset = _build_eval_dataset(windows)
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5]),
        )
        report = runner.compare(retrievers)
        csv_str = report.to_csv()
        assert "BM25" in csv_str
        assert "recall@5" in csv_str

    def test_report_save_to_files(self, tmp_path: Path) -> None:
        retrievers, windows = _build_retrievers()
        dataset = _build_eval_dataset(windows)
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5]),
        )
        report = runner.compare(retrievers)
        report.save_json(tmp_path / "report.json")
        report.save_csv(tmp_path / "report.csv")
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.csv").exists()

    def test_dataset_save_and_reload(self, tmp_path: Path) -> None:
        _, windows = _build_retrievers()
        dataset = _build_eval_dataset(windows)
        path = tmp_path / "dataset.json"
        dataset.save(path)
        loaded = EvaluationDataset.load(path)
        assert loaded.name == dataset.name
        assert len(loaded.examples) == len(dataset.examples)

    def test_all_metrics_are_valid_floats(self) -> None:
        retrievers, windows = _build_retrievers()
        dataset = _build_eval_dataset(windows)
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5, 10]),
        )
        report = runner.compare(retrievers)
        for result in report.results:
            for key, value in result.aggregated.items():
                assert isinstance(value, float), f"{key} is not float: {value}"
                assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
