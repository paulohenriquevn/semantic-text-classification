"""Unit tests for experiment reporting.

Tests cover: QueryMetrics, MethodResult, ExperimentReport JSON/CSV
serialization, and reexport.
"""

import csv
import io
import json
from pathlib import Path

from talkex.evaluation.report import (
    ExperimentReport,
    MethodResult,
    QueryMetrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_query_metrics(query_id: str = "q1") -> QueryMetrics:
    return QueryMetrics(
        query_id=query_id,
        recall_at_k={5: 0.8, 10: 1.0},
        precision_at_k={5: 0.4, 10: 0.3},
        ndcg_at_k={5: 0.75, 10: 0.85},
        reciprocal_rank=0.5,
        retrieved_count=10,
        relevant_count=3,
    )


def _make_method_result(
    method_name: str = "BM25",
) -> MethodResult:
    return MethodResult(
        method_name=method_name,
        query_metrics=[_make_query_metrics("q1"), _make_query_metrics("q2")],
        aggregated={
            "mrr": 0.75,
            "recall@5": 0.8,
            "recall@10": 1.0,
            "precision@5": 0.4,
            "ndcg@5": 0.75,
        },
        total_queries=2,
        total_ms=12.5,
    )


def _make_report() -> ExperimentReport:
    return ExperimentReport(
        dataset_name="billing-eval",
        dataset_version="1.0",
        k_values=[5, 10],
        results=[
            _make_method_result("BM25"),
            _make_method_result("HYBRID_RRF"),
        ],
    )


# ---------------------------------------------------------------------------
# QueryMetrics
# ---------------------------------------------------------------------------


class TestQueryMetrics:
    def test_creates_with_all_fields(self) -> None:
        qm = _make_query_metrics()
        assert qm.query_id == "q1"
        assert qm.recall_at_k[5] == 0.8
        assert qm.reciprocal_rank == 0.5
        assert qm.retrieved_count == 10
        assert qm.relevant_count == 3

    def test_is_frozen(self) -> None:
        qm = _make_query_metrics()
        try:
            qm.query_id = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# MethodResult
# ---------------------------------------------------------------------------


class TestMethodResult:
    def test_creates_with_all_fields(self) -> None:
        mr = _make_method_result()
        assert mr.method_name == "BM25"
        assert mr.total_queries == 2
        assert mr.total_ms == 12.5
        assert len(mr.query_metrics) == 2
        assert mr.aggregated["mrr"] == 0.75

    def test_is_frozen(self) -> None:
        mr = _make_method_result()
        try:
            mr.method_name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ExperimentReport — JSON
# ---------------------------------------------------------------------------


class TestExperimentReportJSON:
    def test_to_json_produces_valid_json(self) -> None:
        report = _make_report()
        raw = report.to_json()
        data = json.loads(raw)
        assert data["dataset_name"] == "billing-eval"
        assert len(data["results"]) == 2

    def test_to_json_includes_aggregated_metrics(self) -> None:
        report = _make_report()
        data = json.loads(report.to_json())
        bm25 = data["results"][0]
        assert bm25["method_name"] == "BM25"
        assert bm25["aggregated"]["mrr"] == 0.75
        assert bm25["total_queries"] == 2

    def test_save_json_creates_file(self, tmp_path: Path) -> None:
        report = _make_report()
        path = tmp_path / "report.json"
        report.save_json(path)
        data = json.loads(path.read_text())
        assert data["dataset_name"] == "billing-eval"


# ---------------------------------------------------------------------------
# ExperimentReport — CSV
# ---------------------------------------------------------------------------


class TestExperimentReportCSV:
    def test_to_csv_produces_valid_csv(self) -> None:
        report = _make_report()
        raw = report.to_csv()
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 methods
        assert rows[0][0] == "method"
        assert rows[1][0] == "BM25"
        assert rows[2][0] == "HYBRID_RRF"

    def test_csv_header_matches_aggregated_keys(self) -> None:
        report = _make_report()
        raw = report.to_csv()
        reader = csv.reader(io.StringIO(raw))
        header = next(reader)
        assert "mrr" in header
        assert "recall@5" in header
        assert "total_queries" in header
        assert "total_ms" in header

    def test_save_csv_creates_file(self, tmp_path: Path) -> None:
        report = _make_report()
        path = tmp_path / "report.csv"
        report.save_csv(path)
        content = path.read_text()
        assert "BM25" in content
        assert "HYBRID_RRF" in content

    def test_empty_results_produces_empty_csv(self) -> None:
        report = ExperimentReport(
            dataset_name="empty",
            dataset_version="1.0",
            k_values=[5],
            results=[],
        )
        assert report.to_csv() == ""


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestReportReexport:
    def test_importable_from_evaluation_package(self) -> None:
        from talkex.evaluation import (
            ExperimentReport as ER,
        )
        from talkex.evaluation import (
            MethodResult as MR,
        )

        assert ER is ExperimentReport
        assert MR is MethodResult
