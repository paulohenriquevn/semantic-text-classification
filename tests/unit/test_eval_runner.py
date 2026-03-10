"""Unit tests for benchmark runner.

Tests cover: single retriever evaluation, multi-retriever comparison,
aggregation, RunConfig, and reexport.
"""

from dataclasses import dataclass, field

from talkex.evaluation.dataset import (
    EvaluationDataset,
    EvaluationExample,
    RelevanceJudgment,
)
from talkex.evaluation.runner import BenchmarkRunner, RunConfig
from talkex.retrieval.models import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
)

# ---------------------------------------------------------------------------
# Helpers — mock retriever
# ---------------------------------------------------------------------------


@dataclass
class _MockRetriever:
    """Returns pre-configured results keyed by query text."""

    results: dict[str, list[str]] = field(default_factory=dict)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        doc_ids = self.results.get(query.query_text, [])
        hits = [
            RetrievalHit(
                object_id=doc_id,
                object_type="context_window",
                score=1.0 / (i + 1),
                rank=i + 1,
            )
            for i, doc_id in enumerate(doc_ids)
        ]
        return RetrievalResult(hits=hits, total_candidates=len(hits))


def _make_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        name="test-eval",
        version="1.0",
        examples=[
            EvaluationExample(
                query_id="q1",
                query_text="billing",
                relevant_docs=[
                    RelevanceJudgment(document_id="d1"),
                    RelevanceJudgment(document_id="d2"),
                ],
            ),
            EvaluationExample(
                query_id="q2",
                query_text="cancel",
                relevant_docs=[
                    RelevanceJudgment(document_id="d3"),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_default_k_values(self) -> None:
        config = RunConfig()
        assert config.k_values == [1, 3, 5, 10]

    def test_custom_k_values(self) -> None:
        config = RunConfig(k_values=[5, 20])
        assert config.k_values == [5, 20]


# ---------------------------------------------------------------------------
# Single retriever evaluation
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerEvaluate:
    def test_evaluate_returns_method_result(self) -> None:
        dataset = _make_dataset()
        retriever = _MockRetriever(
            results={
                "billing": ["d1", "d2", "d5"],
                "cancel": ["d3", "d4"],
            }
        )
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[3, 5]),
        )
        result = runner.evaluate(retriever, "mock")
        assert result.method_name == "mock"
        assert result.total_queries == 2
        assert result.total_ms >= 0.0

    def test_evaluate_produces_query_metrics(self) -> None:
        dataset = _make_dataset()
        retriever = _MockRetriever(
            results={
                "billing": ["d1", "d2"],
                "cancel": ["d3"],
            }
        )
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5]),
        )
        result = runner.evaluate(retriever, "mock")
        assert len(result.query_metrics) == 2
        # q1: both relevant found → recall@5 = 1.0
        q1 = result.query_metrics[0]
        assert q1.query_id == "q1"
        assert q1.recall_at_k[5] == 1.0

    def test_evaluate_perfect_recall(self) -> None:
        dataset = _make_dataset()
        retriever = _MockRetriever(
            results={
                "billing": ["d1", "d2"],
                "cancel": ["d3"],
            }
        )
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5]),
        )
        result = runner.evaluate(retriever, "perfect")
        assert result.aggregated["recall@5"] == 1.0

    def test_evaluate_zero_recall(self) -> None:
        dataset = _make_dataset()
        retriever = _MockRetriever(results={})  # Returns nothing
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5]),
        )
        result = runner.evaluate(retriever, "empty")
        assert result.aggregated["recall@5"] == 0.0
        assert result.aggregated["mrr"] == 0.0

    def test_evaluate_aggregated_mrr(self) -> None:
        dataset = _make_dataset()
        retriever = _MockRetriever(
            results={
                "billing": ["d1", "d2"],  # d1 at rank 1 → RR = 1.0
                "cancel": ["d4", "d3"],  # d3 at rank 2 → RR = 0.5
            }
        )
        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5]),
        )
        result = runner.evaluate(retriever, "mock")
        # MRR = (1.0 + 0.5) / 2 = 0.75
        assert result.aggregated["mrr"] == 0.75


# ---------------------------------------------------------------------------
# Multi-retriever comparison
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerCompare:
    def test_compare_produces_report(self) -> None:
        dataset = _make_dataset()
        perfect = _MockRetriever(results={"billing": ["d1", "d2"], "cancel": ["d3"]})
        empty = _MockRetriever(results={})

        runner = BenchmarkRunner(
            dataset=dataset,
            config=RunConfig(k_values=[5]),
        )
        report = runner.compare({"perfect": perfect, "empty": empty})

        assert report.dataset_name == "test-eval"
        assert report.dataset_version == "1.0"
        assert len(report.results) == 2

        # Perfect retriever should have higher recall
        perfect_result = next(r for r in report.results if r.method_name == "perfect")
        empty_result = next(r for r in report.results if r.method_name == "empty")
        assert perfect_result.aggregated["recall@5"] > empty_result.aggregated["recall@5"]


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestRunnerReexport:
    def test_importable_from_evaluation_package(self) -> None:
        from talkex.evaluation import (
            BenchmarkRunner as BR,
        )
        from talkex.evaluation import (
            RunConfig as RC,
        )

        assert BR is BenchmarkRunner
        assert RC is RunConfig
