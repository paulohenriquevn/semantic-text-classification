"""Unit tests for analytics protocols — duck-typed structural subtyping.

Tests cover: AnalyticsAggregator and AnalyticsReporter protocol compliance
via stub implementations (no Protocol inheritance), and reexports from
the pipeline package.
"""

from datetime import UTC, datetime

from talkex.analytics.config import (
    AggregationWindow,
    AnalyticsLevel,
    MetricType,
)
from talkex.analytics.models import (
    AggregationResult,
    AnalyticsEvent,
    AnalyticsQuery,
    MetricValue,
    TrendPoint,
)
from talkex.pipeline.protocols import (
    AnalyticsAggregator,
    AnalyticsReporter,
)

_NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Stub implementations (no Protocol inheritance — duck typing)
# ---------------------------------------------------------------------------


class StubAggregator:
    """Duck-typed AnalyticsAggregator — no Protocol inheritance."""

    def aggregate(
        self,
        events: list[AnalyticsEvent],
        group_by: str,
    ) -> list[AggregationResult]:
        groups: dict[str, list[AnalyticsEvent]] = {}
        for event in events:
            key = event.metadata.get(group_by, "unknown")
            groups.setdefault(key, []).append(event)

        results = []
        for group_value, group_events in groups.items():
            avg_value = sum(e.value for e in group_events) / len(group_events)
            results.append(
                AggregationResult(
                    group_key=group_by,
                    group_value=group_value,
                    level=AnalyticsLevel.SYSTEM,
                    window=AggregationWindow.DAILY,
                    metrics=[
                        MetricValue(
                            name="avg_value",
                            value=avg_value,
                            metric_type=MetricType.CLASSIFICATION,
                            level=AnalyticsLevel.SYSTEM,
                            count=len(group_events),
                        ),
                    ],
                    event_count=len(group_events),
                )
            )
        return results


class StubReporter:
    """Duck-typed AnalyticsReporter — no Protocol inheritance."""

    def query(
        self,
        analytics_query: AnalyticsQuery,
    ) -> list[AggregationResult]:
        return [
            AggregationResult(
                group_key=analytics_query.group_by or "all",
                group_value="stub",
                level=analytics_query.level,
                window=analytics_query.window,
                metrics=[],
                event_count=0,
            ),
        ]

    def trend(
        self,
        analytics_query: AnalyticsQuery,
        metric_name: str,
    ) -> list[TrendPoint]:
        return [
            TrendPoint(
                timestamp=_NOW,
                window=analytics_query.window,
                metric_name=metric_name,
                value=0.5,
                count=10,
            ),
        ]


# ---------------------------------------------------------------------------
# AnalyticsAggregator protocol compliance
# ---------------------------------------------------------------------------


class TestAnalyticsAggregatorProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        aggregator: AnalyticsAggregator = StubAggregator()
        assert hasattr(aggregator, "aggregate")

    def test_aggregate_groups_events(self) -> None:
        aggregator = StubAggregator()
        events = [
            AnalyticsEvent(
                event_id="e1",
                event_type="prediction",
                source_id="w1",
                source_type="context_window",
                timestamp=_NOW,
                metric_type=MetricType.CLASSIFICATION,
                value=0.8,
                metadata={"channel": "voice"},
            ),
            AnalyticsEvent(
                event_id="e2",
                event_type="prediction",
                source_id="w2",
                source_type="context_window",
                timestamp=_NOW,
                metric_type=MetricType.CLASSIFICATION,
                value=0.6,
                metadata={"channel": "voice"},
            ),
            AnalyticsEvent(
                event_id="e3",
                event_type="prediction",
                source_id="w3",
                source_type="context_window",
                timestamp=_NOW,
                metric_type=MetricType.CLASSIFICATION,
                value=0.9,
                metadata={"channel": "chat"},
            ),
        ]
        results = aggregator.aggregate(events, "channel")
        assert len(results) == 2

        voice = next(r for r in results if r.group_value == "voice")
        assert voice.event_count == 2
        assert voice.metrics[0].value == 0.7  # avg(0.8, 0.6)

    def test_aggregate_empty(self) -> None:
        aggregator = StubAggregator()
        results = aggregator.aggregate([], "channel")
        assert results == []


# ---------------------------------------------------------------------------
# AnalyticsReporter protocol compliance
# ---------------------------------------------------------------------------


class TestAnalyticsReporterProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        reporter: AnalyticsReporter = StubReporter()
        assert hasattr(reporter, "query")
        assert hasattr(reporter, "trend")

    def test_query_returns_aggregation_results(self) -> None:
        reporter = StubReporter()
        query = AnalyticsQuery(
            query_id="q1",
            level=AnalyticsLevel.CONVERSATION,
            group_by="channel",
        )
        results = reporter.query(query)
        assert len(results) == 1
        assert results[0].group_key == "channel"
        assert results[0].level == AnalyticsLevel.CONVERSATION

    def test_trend_returns_trend_points(self) -> None:
        reporter = StubReporter()
        query = AnalyticsQuery(
            query_id="q2",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.WEEKLY,
        )
        points = reporter.trend(query, "match_rate")
        assert len(points) == 1
        assert points[0].metric_name == "match_rate"
        assert points[0].window == AggregationWindow.WEEKLY


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestProtocolReexport:
    def test_importable_from_pipeline_package(self) -> None:
        from talkex.pipeline import (
            AnalyticsAggregator,
            AnalyticsReporter,
        )

        assert AnalyticsAggregator is not None
        assert AnalyticsReporter is not None
