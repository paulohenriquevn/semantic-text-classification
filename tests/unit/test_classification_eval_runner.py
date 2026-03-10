"""Unit tests for ClassificationBenchmarkRunner.

Tests cover: single classifier evaluation, multi-classifier comparison,
per-example metrics, aggregation, config, reexport.
"""

from talkex.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from talkex.classification_eval.dataset import (
    ClassificationDataset,
    ClassificationExample,
    GroundTruthLabel,
)
from talkex.classification_eval.runner import (
    ClassificationBenchmarkRunner,
    ClassificationRunConfig,
)

# ---------------------------------------------------------------------------
# Stub classifier
# ---------------------------------------------------------------------------


class _StubClassifier:
    """Returns fixed predictions based on source_id mapping."""

    def __init__(self, predictions: dict[str, list[str]]) -> None:
        self._predictions = predictions

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        results = []
        for inp in inputs:
            pred_labels = self._predictions.get(inp.source_id, [])
            label_scores = [LabelScore(label=name, score=0.9, confidence=0.9, threshold=0.5) for name in pred_labels]
            results.append(
                ClassificationResult(
                    source_id=inp.source_id,
                    source_type=inp.source_type,
                    label_scores=label_scores,
                    model_name="stub",
                    model_version="1.0",
                )
            )
        return results


def _make_dataset() -> ClassificationDataset:
    return ClassificationDataset(
        name="test-cls-eval",
        version="1.0",
        examples=[
            ClassificationExample(
                example_id="ex_1",
                text="billing issue",
                ground_truth=[GroundTruthLabel(label="billing")],
            ),
            ClassificationExample(
                example_id="ex_2",
                text="cancel subscription",
                ground_truth=[GroundTruthLabel(label="cancel")],
            ),
            ClassificationExample(
                example_id="ex_3",
                text="refund request",
                ground_truth=[GroundTruthLabel(label="refund")],
            ),
        ],
        label_names=["billing", "cancel", "refund"],
    )


# ---------------------------------------------------------------------------
# Single classifier evaluation
# ---------------------------------------------------------------------------


class TestEvaluateSingle:
    def test_perfect_classifier(self) -> None:
        ds = _make_dataset()
        clf = _StubClassifier(
            {
                "ex_1": ["billing"],
                "ex_2": ["cancel"],
                "ex_3": ["refund"],
            }
        )
        runner = ClassificationBenchmarkRunner(dataset=ds)
        result = runner.evaluate(clf, "perfect")

        assert result.method_name == "perfect"
        assert result.total_examples == 3
        assert result.aggregated["mean_f1"] == 1.0
        assert result.aggregated["micro_f1"] == 1.0
        assert result.aggregated["macro_f1"] == 1.0

    def test_zero_classifier(self) -> None:
        ds = _make_dataset()
        clf = _StubClassifier({})  # Predicts nothing
        runner = ClassificationBenchmarkRunner(dataset=ds)
        result = runner.evaluate(clf, "zero")

        assert result.aggregated["mean_f1"] == 0.0
        assert result.aggregated["micro_f1"] == 0.0
        assert result.aggregated["macro_f1"] == 0.0

    def test_partial_classifier(self) -> None:
        ds = _make_dataset()
        clf = _StubClassifier(
            {
                "ex_1": ["billing"],
                "ex_2": ["billing"],  # Wrong prediction
                "ex_3": ["refund"],
            }
        )
        runner = ClassificationBenchmarkRunner(dataset=ds)
        result = runner.evaluate(clf, "partial")

        assert result.total_examples == 3
        # ex_1: F1=1.0, ex_2: F1=0.0, ex_3: F1=1.0 → mean_f1 ≈ 0.6667
        assert 0.6 < result.aggregated["mean_f1"] < 0.7

    def test_per_example_metrics_populated(self) -> None:
        ds = _make_dataset()
        clf = _StubClassifier({"ex_1": ["billing"]})
        runner = ClassificationBenchmarkRunner(dataset=ds)
        result = runner.evaluate(clf, "test")

        assert len(result.example_metrics) == 3
        # ex_1 should have perfect metrics
        em1 = result.example_metrics[0]
        assert em1.example_id == "ex_1"
        assert em1.precision == 1.0
        assert em1.recall == 1.0
        assert em1.f1 == 1.0

    def test_per_label_metrics_populated(self) -> None:
        ds = _make_dataset()
        clf = _StubClassifier({"ex_1": ["billing"], "ex_2": ["cancel"], "ex_3": ["refund"]})
        runner = ClassificationBenchmarkRunner(dataset=ds)
        result = runner.evaluate(clf, "test")

        assert "billing" in result.per_label
        assert result.per_label["billing"]["f1"] == 1.0
        assert result.per_label["billing"]["support"] == 1.0

    def test_timing_measured(self) -> None:
        ds = _make_dataset()
        clf = _StubClassifier({"ex_1": ["billing"]})
        runner = ClassificationBenchmarkRunner(dataset=ds)
        result = runner.evaluate(clf, "test")
        assert result.total_ms >= 0.0


# ---------------------------------------------------------------------------
# Multi-classifier comparison
# ---------------------------------------------------------------------------


class TestCompare:
    def test_compare_two_classifiers(self) -> None:
        ds = _make_dataset()
        clf_a = _StubClassifier({"ex_1": ["billing"], "ex_2": ["cancel"], "ex_3": ["refund"]})
        clf_b = _StubClassifier({})

        runner = ClassificationBenchmarkRunner(dataset=ds)
        report = runner.compare({"perfect": clf_a, "empty": clf_b})

        assert report.dataset_name == "test-cls-eval"
        assert report.dataset_version == "1.0"
        assert len(report.results) == 2
        assert report.label_names == ["billing", "cancel", "refund"]

    def test_compare_preserves_method_names(self) -> None:
        ds = _make_dataset()
        clf = _StubClassifier({"ex_1": ["billing"]})
        runner = ClassificationBenchmarkRunner(dataset=ds)
        report = runner.compare({"method_a": clf, "method_b": clf})

        names = {r.method_name for r in report.results}
        assert names == {"method_a", "method_b"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_default_config(self) -> None:
        cfg = ClassificationRunConfig()
        assert cfg.threshold_override is None

    def test_threshold_override(self) -> None:
        ds = _make_dataset()
        # Stub returns score=0.9 with threshold=0.5 → all positive
        # But with threshold_override=0.95, score=0.9 is below → no positives
        clf = _StubClassifier({"ex_1": ["billing"]})
        runner = ClassificationBenchmarkRunner(
            dataset=ds,
            config=ClassificationRunConfig(threshold_override=0.95),
        )
        result = runner.evaluate(clf, "high_threshold")
        # With threshold=0.95, score=0.9 should NOT be positive
        assert result.aggregated["mean_f1"] == 0.0


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestRunnerReexport:
    def test_importable_from_package(self) -> None:
        from talkex.classification_eval import (
            ClassificationBenchmarkRunner as CBR,
        )
        from talkex.classification_eval import (
            ClassificationRunConfig as CRC,
        )

        assert CBR is ClassificationBenchmarkRunner
        assert CRC is ClassificationRunConfig
