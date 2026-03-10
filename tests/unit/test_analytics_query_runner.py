"""Unit tests for analytics query runner — structured report generation.

Tests cover: AnalyticsQueryRunner (run_grouped, run_trend, run_composite),
execution metadata, section structure, composite queries, empty inputs,
and reexport.
"""

import json
from datetime import UTC, datetime, timedelta

from talkex.analytics.aggregators import SimpleAnalyticsEngine
from talkex.analytics.config import (
    AggregationWindow,
    AnalyticsLevel,
    MetricType,
)
from talkex.analytics.models import (
    AnalyticsEvent,
    AnalyticsQuery,
)
from talkex.analytics.query_runner import AnalyticsQueryRunner

_BASE = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)


def _make_event(
    event_id: str = "evt_001",
    *,
    timestamp: datetime = _BASE,
    metric_type: MetricType = MetricType.CLASSIFICATION,
    value: float = 0.8,
    label: str | None = "billing",
    matched: bool | None = True,
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
    """Create an engine with a diverse set of events for testing."""
    events = [
        _make_event("e1", label="billing", value=0.9, matched=True, metadata={"channel": "voice"}),
        _make_event("e2", label="billing", value=0.7, matched=True, metadata={"channel": "chat"}),
        _make_event("e3", label="support", value=0.6, matched=False, metadata={"channel": "voice"}),
        _make_event(
            "e4",
            label="support",
            value=0.8,
            matched=True,
            metadata={"channel": "voice"},
            timestamp=_BASE + timedelta(days=1),
        ),
        _make_event(
            "e5",
            label="billing",
            value=0.5,
            matched=False,
            metadata={"channel": "chat"},
            metric_type=MetricType.RULE,
            timestamp=_BASE + timedelta(days=1),
        ),
    ]
    return SimpleAnalyticsEngine(events=events)


# ---------------------------------------------------------------------------
# run_grouped
# ---------------------------------------------------------------------------


class TestRunGrouped:
    def test_grouped_by_label(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        query = AnalyticsQuery(
            query_id="q1",
            level=AnalyticsLevel.SYSTEM,
            group_by="label",
        )
        report = runner.run_grouped(query)

        assert report.report_name == "Grouped: label"
        assert len(report.sections) == 1
        assert report.sections[0].name == "Grouped Results"

        # Should have grouped metrics for billing and support
        gm_values = {gm.group_value for gm in report.sections[0].grouped_metrics}
        assert "billing" in gm_values
        assert "support" in gm_values

    def test_grouped_with_custom_section_name(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        query = AnalyticsQuery(
            query_id="q2",
            level=AnalyticsLevel.SYSTEM,
            group_by="channel",
        )
        report = runner.run_grouped(
            query,
            section_name="By Channel",
            section_description="Grouped by communication channel",
        )
        assert report.sections[0].name == "By Channel"
        assert report.sections[0].description == "Grouped by communication channel"

    def test_grouped_metadata_includes_query_info(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        query = AnalyticsQuery(
            query_id="q3",
            level=AnalyticsLevel.CONVERSATION,
            group_by="label",
            metric_types=[MetricType.CLASSIFICATION],
        )
        report = runner.run_grouped(query)

        assert report.metadata["query_id"] == "q3"
        assert report.metadata["level"] == "conversation"
        assert report.metadata["group_by"] == "label"
        assert "generation_time_ms" in report.metadata
        assert report.metadata["total_events"] == 5
        assert "metric_types" in report.metadata

    def test_grouped_with_filters(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        query = AnalyticsQuery(
            query_id="q4",
            level=AnalyticsLevel.SYSTEM,
            group_by="label",
            filters={"channel": "voice"},
        )
        report = runner.run_grouped(query)

        # Only voice events: e1(billing), e3(support), e4(support)
        total_events = sum(
            gm.event_count for gm in report.sections[0].grouped_metrics if gm.metric_name == "event_count"
        )
        # billing=1 event, support=2 events = 3 total
        assert total_events == 3

    def test_grouped_empty_result(self) -> None:
        engine = SimpleAnalyticsEngine()
        runner = AnalyticsQueryRunner(engine)
        query = AnalyticsQuery(
            query_id="q5",
            level=AnalyticsLevel.SYSTEM,
            group_by="label",
        )
        report = runner.run_grouped(query)
        assert len(report.sections) == 1
        assert report.sections[0].grouped_metrics == []

    def test_grouped_default_group_by_metric_type(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        query = AnalyticsQuery(
            query_id="q6",
            level=AnalyticsLevel.SYSTEM,
        )
        report = runner.run_grouped(query)
        assert report.report_name == "Grouped: metric_type"


# ---------------------------------------------------------------------------
# run_trend
# ---------------------------------------------------------------------------


class TestRunTrend:
    def test_trend_match_rate(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        query = AnalyticsQuery(
            query_id="qt1",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.DAILY,
        )
        report = runner.run_trend(query, "match_rate")

        assert report.report_name == "Trend: match_rate"
        assert len(report.sections) == 1
        assert len(report.sections[0].trend_series) == 1

        series = report.sections[0].trend_series[0]
        assert series.metric_name == "match_rate"
        assert series.window == "daily"
        assert len(series.points) == 2  # 2 days of data

    def test_trend_with_custom_section(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        query = AnalyticsQuery(
            query_id="qt2",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.DAILY,
        )
        report = runner.run_trend(
            query,
            "avg_value",
            section_name="Score Trend",
            section_description="Average score over time",
        )
        assert report.sections[0].name == "Score Trend"

    def test_trend_metadata(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        query = AnalyticsQuery(
            query_id="qt3",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.DAILY,
        )
        report = runner.run_trend(query, "event_count")

        assert report.metadata["query_id"] == "qt3"
        assert report.metadata["metric_name"] == "event_count"
        assert "generation_time_ms" in report.metadata

    def test_trend_empty(self) -> None:
        engine = SimpleAnalyticsEngine()
        runner = AnalyticsQueryRunner(engine)
        query = AnalyticsQuery(
            query_id="qt4",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.DAILY,
        )
        report = runner.run_trend(query, "match_rate")
        assert report.sections[0].trend_series[0].points == []


# ---------------------------------------------------------------------------
# run_composite
# ---------------------------------------------------------------------------


class TestRunComposite:
    def test_composite_grouped_only(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        queries = [
            (
                AnalyticsQuery(query_id="cq1", level=AnalyticsLevel.SYSTEM, group_by="label"),
                "By Label",
                "Classification labels",
            ),
            (
                AnalyticsQuery(query_id="cq2", level=AnalyticsLevel.SYSTEM, group_by="channel"),
                "By Channel",
                "Communication channels",
            ),
        ]
        report = runner.run_composite(queries, report_name="Multi-Dimension Report")

        assert report.report_name == "Multi-Dimension Report"
        assert len(report.sections) == 2
        assert report.sections[0].name == "By Label"
        assert report.sections[1].name == "By Channel"
        assert report.metadata["grouped_sections"] == 2
        assert report.metadata["trend_sections"] == 0

    def test_composite_with_trends(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        grouped_queries = [
            (
                AnalyticsQuery(query_id="cq3", level=AnalyticsLevel.SYSTEM, group_by="label"),
                "By Label",
                "",
            ),
        ]
        trend_queries = [
            (
                AnalyticsQuery(query_id="cqt1", level=AnalyticsLevel.SYSTEM, window=AggregationWindow.DAILY),
                "match_rate",
                "Match Rate Trend",
                "Daily match rate",
            ),
        ]
        report = runner.run_composite(
            grouped_queries,
            trend_queries=trend_queries,
            report_name="Full Report",
        )

        assert len(report.sections) == 2
        assert report.sections[0].name == "By Label"
        assert report.sections[1].name == "Match Rate Trend"
        assert report.metadata["grouped_sections"] == 1
        assert report.metadata["trend_sections"] == 1

    def test_composite_metadata(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        report = runner.run_composite(
            [],
            trend_queries=[],
            report_name="Empty Composite",
        )
        assert report.metadata["grouped_sections"] == 0
        assert report.metadata["trend_sections"] == 0
        assert "generation_time_ms" in report.metadata

    def test_composite_json_serializable(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        queries = [
            (
                AnalyticsQuery(query_id="cq4", level=AnalyticsLevel.SYSTEM, group_by="label"),
                "By Label",
                "",
            ),
        ]
        trend_queries = [
            (
                AnalyticsQuery(query_id="cqt2", level=AnalyticsLevel.SYSTEM, window=AggregationWindow.DAILY),
                "match_rate",
                "Trends",
                "",
            ),
        ]
        report = runner.run_composite(queries, trend_queries=trend_queries)
        json_str = report.to_json()
        data = json.loads(json_str)
        assert len(data["sections"]) == 2

    def test_composite_csv_serializable(self) -> None:
        runner = AnalyticsQueryRunner(_make_engine())
        queries = [
            (
                AnalyticsQuery(query_id="cq5", level=AnalyticsLevel.SYSTEM, group_by="label"),
                "By Label",
                "",
            ),
        ]
        report = runner.run_composite(queries)
        csv_str = report.to_csv()
        assert "By Label" in csv_str


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestQueryRunnerReexport:
    def test_importable_from_analytics_package(self) -> None:
        from talkex.analytics import AnalyticsQueryRunner

        assert AnalyticsQueryRunner is not None
