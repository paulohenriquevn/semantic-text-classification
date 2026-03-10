"""Unit tests for ClassificationExperimentReport, ClassificationMethodResult, ExampleMetrics.

Tests cover: construction, JSON serialization, CSV serialization,
file I/O, empty results, immutability, reexport.
"""

import csv
import io
import json

from talkex.classification_eval.report import (
    ClassificationExperimentReport,
    ClassificationMethodResult,
    ExampleMetrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example_metrics() -> ExampleMetrics:
    return ExampleMetrics(
        example_id="ex_1",
        predicted_labels=["billing"],
        ground_truth_labels=["billing"],
        precision=1.0,
        recall=1.0,
        f1=1.0,
    )


def _make_method_result(
    method_name: str = "test-method",
) -> ClassificationMethodResult:
    return ClassificationMethodResult(
        method_name=method_name,
        example_metrics=[_make_example_metrics()],
        aggregated={
            "mean_precision": 1.0,
            "mean_recall": 1.0,
            "mean_f1": 1.0,
            "micro_f1": 1.0,
            "macro_f1": 1.0,
        },
        per_label={
            "billing": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1.0},
        },
        total_examples=1,
        total_ms=5.0,
    )


def _make_report() -> ClassificationExperimentReport:
    return ClassificationExperimentReport(
        dataset_name="test-dataset",
        dataset_version="1.0",
        label_names=["billing", "cancel"],
        results=[
            _make_method_result("similarity"),
            _make_method_result("logistic"),
        ],
    )


# ---------------------------------------------------------------------------
# ExampleMetrics
# ---------------------------------------------------------------------------


class TestExampleMetrics:
    def test_construction(self) -> None:
        em = _make_example_metrics()
        assert em.example_id == "ex_1"
        assert em.predicted_labels == ["billing"]
        assert em.ground_truth_labels == ["billing"]
        assert em.f1 == 1.0

    def test_frozen(self) -> None:
        em = _make_example_metrics()
        try:
            em.f1 = 0.5  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ClassificationMethodResult
# ---------------------------------------------------------------------------


class TestClassificationMethodResult:
    def test_construction(self) -> None:
        mr = _make_method_result()
        assert mr.method_name == "test-method"
        assert mr.total_examples == 1
        assert mr.total_ms == 5.0
        assert mr.aggregated["micro_f1"] == 1.0

    def test_frozen(self) -> None:
        mr = _make_method_result()
        try:
            mr.method_name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestReportJSON:
    def test_to_json_is_valid(self) -> None:
        report = _make_report()
        data = json.loads(report.to_json())
        assert data["dataset_name"] == "test-dataset"
        assert data["dataset_version"] == "1.0"
        assert len(data["results"]) == 2

    def test_json_includes_aggregated(self) -> None:
        report = _make_report()
        data = json.loads(report.to_json())
        assert data["results"][0]["aggregated"]["micro_f1"] == 1.0

    def test_json_includes_per_label(self) -> None:
        report = _make_report()
        data = json.loads(report.to_json())
        assert "billing" in data["results"][0]["per_label"]

    def test_json_includes_method_name(self) -> None:
        report = _make_report()
        data = json.loads(report.to_json())
        names = [r["method_name"] for r in data["results"]]
        assert "similarity" in names
        assert "logistic" in names

    def test_save_json(self, tmp_path) -> None:
        report = _make_report()
        path = tmp_path / "report.json"
        report.save_json(path)
        data = json.loads(path.read_text())
        assert data["dataset_name"] == "test-dataset"


# ---------------------------------------------------------------------------
# CSV serialization
# ---------------------------------------------------------------------------


class TestReportCSV:
    def test_to_csv_has_header(self) -> None:
        report = _make_report()
        csv_str = report.to_csv()
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header[0] == "method"
        assert "total_examples" in header
        assert "total_ms" in header

    def test_to_csv_has_rows(self) -> None:
        report = _make_report()
        csv_str = report.to_csv()
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 methods

    def test_to_csv_method_names(self) -> None:
        report = _make_report()
        csv_str = report.to_csv()
        reader = csv.reader(io.StringIO(csv_str))
        next(reader)  # skip header
        methods = [row[0] for row in reader]
        assert "similarity" in methods
        assert "logistic" in methods

    def test_to_csv_empty_results(self) -> None:
        report = ClassificationExperimentReport(
            dataset_name="test",
            dataset_version="1.0",
            label_names=[],
            results=[],
        )
        assert report.to_csv() == ""

    def test_save_csv(self, tmp_path) -> None:
        report = _make_report()
        path = tmp_path / "report.csv"
        report.save_csv(path)
        content = path.read_text()
        assert "similarity" in content


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestReportReexport:
    def test_importable_from_package(self) -> None:
        from talkex.classification_eval import (
            ClassificationExperimentReport as CER,
        )
        from talkex.classification_eval import (
            ClassificationMethodResult as CMR,
        )
        from talkex.classification_eval import (
            ExampleMetrics as EM,
        )

        assert CER is ClassificationExperimentReport
        assert CMR is ClassificationMethodResult
        assert EM is ExampleMetrics
