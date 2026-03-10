"""Unit tests for analytics operational metrics — pure functions over reports.

Tests cover: avg_generation_time_ms, avg_result_count, empty_report_rate,
avg_group_count, avg_trend_point_count, total_events_considered,
compute_analytics_metrics, and empty inputs.
"""

from talkex.analytics.metrics import (
    avg_generation_time_ms,
    avg_group_count,
    avg_result_count,
    avg_trend_point_count,
    compute_analytics_metrics,
    empty_report_rate,
    total_events_considered,
)
from talkex.analytics.report import (
    AnalyticsReport,
    AnalyticsSection,
    GroupedMetric,
    TrendSeries,
)


def _make_report(
    *,
    groups: int = 0,
    trend_points: int = 0,
    generation_time_ms: float = 1.0,
    total_events: int = 100,
) -> AnalyticsReport:
    grouped_metrics = [
        GroupedMetric(
            group_key="label",
            group_value=f"label_{i}",
            metric_name="match_rate",
            value=0.5,
            event_count=10,
        )
        for i in range(groups)
    ]
    points = [{"timestamp": f"2026-03-{10 + i}T00:00:00+00:00", "value": 0.5, "count": 10} for i in range(trend_points)]
    sections = []
    if groups > 0:
        sections.append(AnalyticsSection(name="Groups", grouped_metrics=grouped_metrics))
    if trend_points > 0:
        series = TrendSeries(metric_name="match_rate", window="daily", points=points)
        sections.append(AnalyticsSection(name="Trend", trend_series=[series]))

    return AnalyticsReport(
        report_name="Test",
        generated_at="2026-03-10T12:00:00+00:00",
        sections=sections,
        metadata={
            "generation_time_ms": generation_time_ms,
            "total_events": total_events,
        },
    )


class TestAvgGenerationTimeMs:
    def test_computes_average(self) -> None:
        reports = [
            _make_report(generation_time_ms=2.0),
            _make_report(generation_time_ms=4.0),
        ]
        assert avg_generation_time_ms(reports) == 3.0

    def test_empty_returns_zero(self) -> None:
        assert avg_generation_time_ms([]) == 0.0


class TestAvgResultCount:
    def test_counts_groups_and_points(self) -> None:
        reports = [
            _make_report(groups=3, trend_points=2),
            _make_report(groups=1, trend_points=0),
        ]
        # (3+2) + (1+0) = 6, avg = 3.0
        assert avg_result_count(reports) == 3.0

    def test_empty_returns_zero(self) -> None:
        assert avg_result_count([]) == 0.0


class TestEmptyReportRate:
    def test_some_empty(self) -> None:
        reports = [
            _make_report(groups=3),
            _make_report(),  # empty
            _make_report(trend_points=2),
            _make_report(),  # empty
        ]
        assert empty_report_rate(reports) == 0.5

    def test_none_empty(self) -> None:
        reports = [_make_report(groups=1), _make_report(trend_points=1)]
        assert empty_report_rate(reports) == 0.0

    def test_all_empty(self) -> None:
        reports = [_make_report(), _make_report()]
        assert empty_report_rate(reports) == 1.0

    def test_empty_list_returns_zero(self) -> None:
        assert empty_report_rate([]) == 0.0


class TestAvgGroupCount:
    def test_computes_average(self) -> None:
        reports = [_make_report(groups=4), _make_report(groups=6)]
        assert avg_group_count(reports) == 5.0

    def test_empty_returns_zero(self) -> None:
        assert avg_group_count([]) == 0.0


class TestAvgTrendPointCount:
    def test_computes_average(self) -> None:
        reports = [_make_report(trend_points=3), _make_report(trend_points=7)]
        assert avg_trend_point_count(reports) == 5.0

    def test_empty_returns_zero(self) -> None:
        assert avg_trend_point_count([]) == 0.0


class TestTotalEventsConsidered:
    def test_sums_events(self) -> None:
        reports = [
            _make_report(total_events=100),
            _make_report(total_events=200),
        ]
        assert total_events_considered(reports) == 300

    def test_empty_returns_zero(self) -> None:
        assert total_events_considered([]) == 0


class TestComputeAnalyticsMetrics:
    def test_returns_all_metrics(self) -> None:
        reports = [
            _make_report(groups=3, trend_points=2, generation_time_ms=1.5, total_events=50),
            _make_report(groups=1, generation_time_ms=2.5, total_events=50),
        ]
        metrics = compute_analytics_metrics(reports)

        assert metrics["report_count"] == 2
        assert "avg_generation_time_ms" in metrics
        assert "avg_result_count" in metrics
        assert "empty_report_rate" in metrics
        assert "avg_group_count" in metrics
        assert "avg_trend_point_count" in metrics
        assert metrics["total_events_considered"] == 100

    def test_empty_reports(self) -> None:
        metrics = compute_analytics_metrics([])
        assert metrics["report_count"] == 0
        assert metrics["avg_generation_time_ms"] == 0.0


class TestMetricsReexport:
    def test_importable_from_analytics_package(self) -> None:
        from talkex.analytics import compute_analytics_metrics

        assert compute_analytics_metrics is not None
