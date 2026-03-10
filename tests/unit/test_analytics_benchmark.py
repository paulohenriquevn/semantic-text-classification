"""Unit tests for analytics benchmark runner — query scenario comparison.

Tests cover: AnalyticsBenchmarkRunner (run_scenario, compare),
QueryScenarioResult, AnalyticsBenchmarkReport (JSON/CSV export),
empty scenarios, and reexports.
"""

import json
from datetime import UTC, datetime, timedelta

from talkex.analytics.aggregators import SimpleAnalyticsEngine
from talkex.analytics.benchmark import (
    AnalyticsBenchmarkConfig,
    AnalyticsBenchmarkReport,
    AnalyticsBenchmarkRunner,
)
from talkex.analytics.config import (
    AggregationWindow,
    AnalyticsLevel,
    MetricType,
)
from talkex.analytics.models import (
    AnalyticsEvent,
    AnalyticsQuery,
)

_BASE = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)


def _make_event(
    event_id: str,
    *,
    timestamp: datetime = _BASE,
    metric_type: MetricType = MetricType.CLASSIFICATION,
    value: float = 0.8,
    label: str = "billing",
    matched: bool = True,
    metadata: dict[str, object] | None = None,
) -> AnalyticsEvent:
    return AnalyticsEvent(
        event_id=event_id,
        event_type="prediction",
        source_id="win_001",
        source_type="context_window",
        timestamp=timestamp,
        metric_type=metric_type,
        value=value,
        label=label,
        matched=matched,
        metadata=metadata or {},
    )


def _make_engine() -> SimpleAnalyticsEngine:
    events = [
        _make_event("e1", label="billing", metadata={"channel": "voice"}),
        _make_event("e2", label="support", metadata={"channel": "chat"}),
        _make_event("e3", label="billing", metadata={"channel": "voice"}, matched=False),
        _make_event("e4", label="support", metadata={"channel": "voice"}, timestamp=_BASE + timedelta(days=1)),
        _make_event(
            "e5",
            label="billing",
            metadata={"channel": "chat"},
            metric_type=MetricType.RULE,
            timestamp=_BASE + timedelta(days=1),
        ),
    ]
    return SimpleAnalyticsEngine(events=events)


# ---------------------------------------------------------------------------
# run_scenario
# ---------------------------------------------------------------------------


class TestRunScenario:
    def test_grouped_scenario(self) -> None:
        runner = AnalyticsBenchmarkRunner(engine=_make_engine())
        query = AnalyticsQuery(query_id="q1", level=AnalyticsLevel.SYSTEM, group_by="label")
        result = runner.run_scenario(query, "by_label")

        assert result.scenario_name == "by_label"
        assert result.group_count > 0
        assert result.trend_point_count == 0
        assert result.generation_time_ms >= 0
        assert result.query_params["query_type"] == "grouped"
        assert result.query_params["group_by"] == "label"

    def test_trend_scenario(self) -> None:
        runner = AnalyticsBenchmarkRunner(engine=_make_engine())
        query = AnalyticsQuery(query_id="q2", level=AnalyticsLevel.SYSTEM, window=AggregationWindow.DAILY)
        result = runner.run_scenario(query, "daily_match_rate", metric_name="match_rate")

        assert result.scenario_name == "daily_match_rate"
        assert result.trend_point_count > 0
        assert result.group_count == 0
        assert result.query_params["query_type"] == "trend"
        assert result.query_params["metric_name"] == "match_rate"

    def test_frozen_result(self) -> None:
        runner = AnalyticsBenchmarkRunner(engine=_make_engine())
        query = AnalyticsQuery(query_id="q3", level=AnalyticsLevel.SYSTEM, group_by="label")
        result = runner.run_scenario(query, "test")
        try:
            result.scenario_name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_compare_grouped_scenarios(self) -> None:
        runner = AnalyticsBenchmarkRunner(engine=_make_engine())
        report = runner.compare(
            {
                "by_label": AnalyticsQuery(query_id="q1", level=AnalyticsLevel.SYSTEM, group_by="label"),
                "by_channel": AnalyticsQuery(query_id="q2", level=AnalyticsLevel.SYSTEM, group_by="channel"),
            }
        )

        assert len(report.results) == 2
        names = {r.scenario_name for r in report.results}
        assert names == {"by_label", "by_channel"}
        assert report.total_events == 5
        assert report.total_ms > 0

    def test_compare_with_trend_scenarios(self) -> None:
        runner = AnalyticsBenchmarkRunner(engine=_make_engine())
        report = runner.compare(
            {"by_label": AnalyticsQuery(query_id="q3", level=AnalyticsLevel.SYSTEM, group_by="label")},
            trend_scenarios={
                "daily_match": (
                    AnalyticsQuery(query_id="q4", level=AnalyticsLevel.SYSTEM, window=AggregationWindow.DAILY),
                    "match_rate",
                ),
            },
        )

        assert len(report.results) == 2
        assert report.aggregated["report_count"] == 2

    def test_compare_empty_scenarios(self) -> None:
        runner = AnalyticsBenchmarkRunner(engine=_make_engine())
        report = runner.compare({})
        assert len(report.results) == 0

    def test_aggregated_metrics_present(self) -> None:
        runner = AnalyticsBenchmarkRunner(engine=_make_engine())
        report = runner.compare(
            {
                "by_label": AnalyticsQuery(query_id="q5", level=AnalyticsLevel.SYSTEM, group_by="label"),
            }
        )
        assert "report_count" in report.aggregated
        assert "avg_generation_time_ms" in report.aggregated
        assert "empty_report_rate" in report.aggregated

    def test_custom_config(self) -> None:
        config = AnalyticsBenchmarkConfig(experiment_name="custom_exp", experiment_version="2.0")
        runner = AnalyticsBenchmarkRunner(engine=_make_engine(), config=config)
        report = runner.compare(
            {
                "test": AnalyticsQuery(query_id="q6", level=AnalyticsLevel.SYSTEM, group_by="label"),
            }
        )
        assert report.experiment_name == "custom_exp"
        assert report.experiment_version == "2.0"


# ---------------------------------------------------------------------------
# AnalyticsBenchmarkReport serialization
# ---------------------------------------------------------------------------


class TestBenchmarkReportSerialization:
    def _make_report(self) -> AnalyticsBenchmarkReport:
        runner = AnalyticsBenchmarkRunner(engine=_make_engine())
        return runner.compare(
            {
                "by_label": AnalyticsQuery(query_id="q1", level=AnalyticsLevel.SYSTEM, group_by="label"),
                "by_channel": AnalyticsQuery(query_id="q2", level=AnalyticsLevel.SYSTEM, group_by="channel"),
            }
        )

    def test_to_json(self) -> None:
        report = self._make_report()
        json_str = report.to_json()
        data = json.loads(json_str)

        assert data["experiment_name"] == "analytics_benchmark"
        assert len(data["results"]) == 2
        assert "aggregated" in data
        assert "total_events" in data

    def test_to_json_roundtrip(self) -> None:
        report = self._make_report()
        data1 = json.loads(report.to_json())
        data2 = json.loads(json.dumps(data1, indent=2, ensure_ascii=False))
        assert data1 == data2

    def test_to_csv(self) -> None:
        report = self._make_report()
        csv_str = report.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 scenarios
        assert "scenario_name" in lines[0]

    def test_to_csv_empty(self) -> None:
        report = AnalyticsBenchmarkReport(
            experiment_name="empty",
            experiment_version="1.0",
        )
        assert report.to_csv() == ""

    def test_save_json(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        report = self._make_report()
        path = tmp_path / "benchmark.json"
        report.save_json(path)
        data = json.loads(path.read_text())
        assert data["experiment_name"] == "analytics_benchmark"

    def test_save_csv(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        report = self._make_report()
        path = tmp_path / "benchmark.csv"
        report.save_csv(path)
        content = path.read_text()
        assert "by_label" in content

    def test_frozen(self) -> None:
        report = self._make_report()
        try:
            report.experiment_name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestBenchmarkReexport:
    def test_importable_from_analytics_package(self) -> None:
        from talkex.analytics import (
            AnalyticsBenchmarkConfig,
            AnalyticsBenchmarkReport,
            AnalyticsBenchmarkRunner,
            QueryScenarioResult,
        )

        assert AnalyticsBenchmarkConfig is not None
        assert AnalyticsBenchmarkReport is not None
        assert AnalyticsBenchmarkRunner is not None
        assert QueryScenarioResult is not None
