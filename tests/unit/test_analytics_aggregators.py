"""Unit tests for analytics aggregators — filtering, grouping, temporal bucketing.

Tests cover: aggregate_by_dimension, aggregate_temporal, filter_events,
SimpleAnalyticsEngine (query, trend, compute_stats), edge cases (empty inputs),
and reexports.
"""

from datetime import UTC, datetime, timedelta

from talkex.analytics.aggregators import (
    SimpleAnalyticsEngine,
    aggregate_by_dimension,
    aggregate_temporal,
    filter_events,
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
    event_id: str = "evt_001",
    *,
    event_type: str = "prediction",
    source_id: str = "win_001",
    source_type: str = "context_window",
    timestamp: datetime = _BASE,
    metric_type: MetricType = MetricType.CLASSIFICATION,
    value: float = 0.8,
    label: str | None = "billing",
    matched: bool | None = True,
    metadata: dict[str, object] | None = None,
) -> AnalyticsEvent:
    return AnalyticsEvent(
        event_id=event_id,
        event_type=event_type,
        source_id=source_id,
        source_type=source_type,
        timestamp=timestamp,
        metric_type=metric_type,
        value=value,
        label=label,
        matched=matched,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# filter_events
# ---------------------------------------------------------------------------


class TestFilterEvents:
    def test_no_filters_returns_all(self) -> None:
        events = [_make_event("e1"), _make_event("e2")]
        assert len(filter_events(events)) == 2

    def test_filter_by_metric_type(self) -> None:
        events = [
            _make_event("e1", metric_type=MetricType.CLASSIFICATION),
            _make_event("e2", metric_type=MetricType.RULE),
            _make_event("e3", metric_type=MetricType.CLASSIFICATION),
        ]
        result = filter_events(events, metric_types=[MetricType.CLASSIFICATION])
        assert len(result) == 2
        assert all(e.metric_type == MetricType.CLASSIFICATION for e in result)

    def test_filter_by_time_range(self) -> None:
        t1 = _BASE
        t2 = _BASE + timedelta(hours=2)
        t3 = _BASE + timedelta(hours=5)
        events = [
            _make_event("e1", timestamp=t1),
            _make_event("e2", timestamp=t2),
            _make_event("e3", timestamp=t3),
        ]
        result = filter_events(
            events,
            start_time=_BASE + timedelta(hours=1),
            end_time=_BASE + timedelta(hours=4),
        )
        assert len(result) == 1
        assert result[0].event_id == "e2"

    def test_filter_by_metadata(self) -> None:
        events = [
            _make_event("e1", metadata={"channel": "voice"}),
            _make_event("e2", metadata={"channel": "chat"}),
            _make_event("e3", metadata={"channel": "voice"}),
        ]
        result = filter_events(events, filters={"channel": "voice"})
        assert len(result) == 2

    def test_filter_by_event_attribute(self) -> None:
        events = [
            _make_event("e1", label="billing"),
            _make_event("e2", label="support"),
        ]
        result = filter_events(events, filters={"label": "billing"})
        assert len(result) == 1
        assert result[0].label == "billing"

    def test_combined_filters(self) -> None:
        events = [
            _make_event("e1", metric_type=MetricType.CLASSIFICATION, metadata={"channel": "voice"}),
            _make_event("e2", metric_type=MetricType.RULE, metadata={"channel": "voice"}),
            _make_event("e3", metric_type=MetricType.CLASSIFICATION, metadata={"channel": "chat"}),
        ]
        result = filter_events(
            events,
            metric_types=[MetricType.CLASSIFICATION],
            filters={"channel": "voice"},
        )
        assert len(result) == 1
        assert result[0].event_id == "e1"

    def test_empty_events_returns_empty(self) -> None:
        assert filter_events([]) == []


# ---------------------------------------------------------------------------
# aggregate_by_dimension
# ---------------------------------------------------------------------------


class TestAggregateByDimension:
    def test_groups_by_metadata_key(self) -> None:
        events = [
            _make_event("e1", metadata={"channel": "voice"}),
            _make_event("e2", metadata={"channel": "chat"}),
            _make_event("e3", metadata={"channel": "voice"}),
        ]
        results = aggregate_by_dimension(events, "channel")
        assert len(results) == 2
        # Sorted by group_value
        assert results[0].group_value == "chat"
        assert results[0].event_count == 1
        assert results[1].group_value == "voice"
        assert results[1].event_count == 2

    def test_groups_by_event_attribute(self) -> None:
        events = [
            _make_event("e1", label="billing"),
            _make_event("e2", label="support"),
            _make_event("e3", label="billing"),
        ]
        results = aggregate_by_dimension(events, "label")
        assert len(results) == 2
        assert results[0].group_value == "billing"
        assert results[0].event_count == 2

    def test_unknown_dimension_fallback(self) -> None:
        events = [_make_event("e1")]
        results = aggregate_by_dimension(events, "nonexistent_dim")
        assert len(results) == 1
        assert results[0].group_value == "unknown"

    def test_metrics_computed_per_group(self) -> None:
        events = [
            _make_event("e1", value=0.8, matched=True),
            _make_event("e2", value=0.6, matched=False),
        ]
        results = aggregate_by_dimension(events, "event_type")
        assert len(results) == 1
        metrics = {m.name: m.value for m in results[0].metrics}
        assert metrics["event_count"] == 2.0
        assert metrics["avg_value"] == 0.7
        assert metrics["match_rate"] == 0.5

    def test_respects_level_and_window(self) -> None:
        events = [_make_event("e1")]
        results = aggregate_by_dimension(
            events,
            "event_type",
            level=AnalyticsLevel.CONVERSATION,
            window=AggregationWindow.WEEKLY,
        )
        assert results[0].level == AnalyticsLevel.CONVERSATION
        assert results[0].window == AggregationWindow.WEEKLY

    def test_empty_events_returns_empty(self) -> None:
        assert aggregate_by_dimension([], "anything") == []

    def test_sorted_by_group_value(self) -> None:
        events = [
            _make_event("e1", metadata={"region": "c"}),
            _make_event("e2", metadata={"region": "a"}),
            _make_event("e3", metadata={"region": "b"}),
        ]
        results = aggregate_by_dimension(events, "region")
        values = [r.group_value for r in results]
        assert values == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# aggregate_temporal
# ---------------------------------------------------------------------------


class TestAggregateTemporal:
    def test_daily_bucketing(self) -> None:
        events = [
            _make_event("e1", timestamp=datetime(2026, 3, 10, 8, 0, tzinfo=UTC), value=0.8),
            _make_event("e2", timestamp=datetime(2026, 3, 10, 14, 0, tzinfo=UTC), value=0.6),
            _make_event("e3", timestamp=datetime(2026, 3, 11, 9, 0, tzinfo=UTC), value=0.9),
        ]
        points = aggregate_temporal(events, "avg_value", AggregationWindow.DAILY)
        assert len(points) == 2
        assert points[0].timestamp == datetime(2026, 3, 10, 0, 0, tzinfo=UTC)
        assert points[0].count == 2
        assert points[0].value == 0.7  # (0.8 + 0.6) / 2
        assert points[1].count == 1

    def test_hourly_bucketing(self) -> None:
        events = [
            _make_event("e1", timestamp=datetime(2026, 3, 10, 8, 15, tzinfo=UTC)),
            _make_event("e2", timestamp=datetime(2026, 3, 10, 8, 45, tzinfo=UTC)),
            _make_event("e3", timestamp=datetime(2026, 3, 10, 9, 30, tzinfo=UTC)),
        ]
        points = aggregate_temporal(events, "event_count", AggregationWindow.HOURLY)
        assert len(points) == 2
        assert points[0].timestamp == datetime(2026, 3, 10, 8, 0, tzinfo=UTC)
        assert points[0].count == 2

    def test_weekly_bucketing_truncates_to_monday(self) -> None:
        # March 10, 2026 is a Tuesday
        events = [
            _make_event("e1", timestamp=datetime(2026, 3, 10, 12, 0, tzinfo=UTC)),
            _make_event("e2", timestamp=datetime(2026, 3, 12, 12, 0, tzinfo=UTC)),
        ]
        points = aggregate_temporal(events, "event_count", AggregationWindow.WEEKLY)
        assert len(points) == 1
        # Should truncate to Monday March 9
        assert points[0].timestamp == datetime(2026, 3, 9, 0, 0, tzinfo=UTC)

    def test_monthly_bucketing(self) -> None:
        events = [
            _make_event("e1", timestamp=datetime(2026, 3, 10, 12, 0, tzinfo=UTC)),
            _make_event("e2", timestamp=datetime(2026, 3, 25, 12, 0, tzinfo=UTC)),
            _make_event("e3", timestamp=datetime(2026, 4, 5, 12, 0, tzinfo=UTC)),
        ]
        points = aggregate_temporal(events, "event_count", AggregationWindow.MONTHLY)
        assert len(points) == 2
        assert points[0].timestamp == datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        assert points[0].count == 2
        assert points[1].timestamp == datetime(2026, 4, 1, 0, 0, tzinfo=UTC)

    def test_match_rate_metric(self) -> None:
        events = [
            _make_event("e1", matched=True),
            _make_event("e2", matched=False),
            _make_event("e3", matched=True),
            _make_event("e4", matched=True),
        ]
        points = aggregate_temporal(events, "match_rate", AggregationWindow.DAILY)
        assert len(points) == 1
        assert points[0].value == 0.75

    def test_ordered_ascending_by_timestamp(self) -> None:
        events = [
            _make_event("e1", timestamp=datetime(2026, 3, 12, 12, 0, tzinfo=UTC)),
            _make_event("e2", timestamp=datetime(2026, 3, 10, 12, 0, tzinfo=UTC)),
            _make_event("e3", timestamp=datetime(2026, 3, 11, 12, 0, tzinfo=UTC)),
        ]
        points = aggregate_temporal(events, "event_count", AggregationWindow.DAILY)
        timestamps = [p.timestamp for p in points]
        assert timestamps == sorted(timestamps)

    def test_empty_events_returns_empty(self) -> None:
        assert aggregate_temporal([], "avg_value", AggregationWindow.DAILY) == []


# ---------------------------------------------------------------------------
# SimpleAnalyticsEngine
# ---------------------------------------------------------------------------


class TestSimpleAnalyticsEngine:
    def test_construction_empty(self) -> None:
        engine = SimpleAnalyticsEngine()
        assert engine.event_count == 0

    def test_construction_with_events(self) -> None:
        events = [_make_event("e1"), _make_event("e2")]
        engine = SimpleAnalyticsEngine(events=events)
        assert engine.event_count == 2

    def test_add_events(self) -> None:
        engine = SimpleAnalyticsEngine()
        engine.add_events([_make_event("e1")])
        engine.add_events([_make_event("e2"), _make_event("e3")])
        assert engine.event_count == 3

    def test_aggregate_delegates_to_pure_function(self) -> None:
        events = [
            _make_event("e1", label="billing"),
            _make_event("e2", label="support"),
        ]
        engine = SimpleAnalyticsEngine()
        results = engine.aggregate(events, "label")
        assert len(results) == 2

    def test_query_filters_and_groups(self) -> None:
        engine = SimpleAnalyticsEngine(
            events=[
                _make_event("e1", metric_type=MetricType.CLASSIFICATION, metadata={"channel": "voice"}),
                _make_event("e2", metric_type=MetricType.RULE, metadata={"channel": "voice"}),
                _make_event("e3", metric_type=MetricType.CLASSIFICATION, metadata={"channel": "chat"}),
            ]
        )
        query = AnalyticsQuery(
            query_id="q1",
            level=AnalyticsLevel.SYSTEM,
            metric_types=[MetricType.CLASSIFICATION],
            group_by="channel",
        )
        results = engine.query(query)
        assert len(results) == 2
        values = {r.group_value for r in results}
        assert values == {"voice", "chat"}

    def test_query_respects_max_results(self) -> None:
        engine = SimpleAnalyticsEngine(
            events=[
                _make_event("e1", metadata={"region": "a"}),
                _make_event("e2", metadata={"region": "b"}),
                _make_event("e3", metadata={"region": "c"}),
            ]
        )
        query = AnalyticsQuery(
            query_id="q2",
            level=AnalyticsLevel.SYSTEM,
            group_by="region",
            max_results=2,
        )
        results = engine.query(query)
        assert len(results) == 2

    def test_query_defaults_group_by_to_metric_type(self) -> None:
        engine = SimpleAnalyticsEngine(
            events=[
                _make_event("e1", metric_type=MetricType.CLASSIFICATION),
                _make_event("e2", metric_type=MetricType.RULE),
            ]
        )
        query = AnalyticsQuery(query_id="q3", level=AnalyticsLevel.SYSTEM)
        results = engine.query(query)
        assert len(results) == 2

    def test_query_with_time_range(self) -> None:
        t1 = _BASE
        t2 = _BASE + timedelta(hours=5)
        engine = SimpleAnalyticsEngine(
            events=[
                _make_event("e1", timestamp=t1),
                _make_event("e2", timestamp=t2),
            ]
        )
        query = AnalyticsQuery(
            query_id="q4",
            level=AnalyticsLevel.SYSTEM,
            start_time=_BASE + timedelta(hours=3),
        )
        results = engine.query(query)
        assert len(results) == 1

    def test_trend_computes_time_series(self) -> None:
        engine = SimpleAnalyticsEngine(
            events=[
                _make_event("e1", timestamp=datetime(2026, 3, 10, 8, 0, tzinfo=UTC), matched=True),
                _make_event("e2", timestamp=datetime(2026, 3, 10, 14, 0, tzinfo=UTC), matched=False),
                _make_event("e3", timestamp=datetime(2026, 3, 11, 9, 0, tzinfo=UTC), matched=True),
            ]
        )
        query = AnalyticsQuery(
            query_id="q5",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.DAILY,
        )
        points = engine.trend(query, "match_rate")
        assert len(points) == 2
        assert points[0].value == 0.5  # 1/2 on day 1
        assert points[1].value == 1.0  # 1/1 on day 2

    def test_trend_respects_max_results(self) -> None:
        events = [_make_event(f"e{i}", timestamp=_BASE + timedelta(days=i)) for i in range(5)]
        engine = SimpleAnalyticsEngine(events=events)
        query = AnalyticsQuery(
            query_id="q6",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.DAILY,
            max_results=3,
        )
        points = engine.trend(query, "event_count")
        assert len(points) == 3

    def test_compute_stats(self) -> None:
        engine = SimpleAnalyticsEngine(
            events=[
                _make_event("e1", value=0.8, matched=True, metric_type=MetricType.CLASSIFICATION),
                _make_event("e2", value=0.6, matched=False, metric_type=MetricType.CLASSIFICATION),
                _make_event("e3", value=0.9, matched=True, metric_type=MetricType.RULE),
            ]
        )
        stats = engine.compute_stats()
        assert stats["total_events"] == 3
        assert stats["matched_events"] == 2
        assert stats["match_rate"] == round(2 / 3, 4)
        assert stats["avg_value"] == round((0.8 + 0.6 + 0.9) / 3, 4)
        assert stats["metric_type_distribution"]["classification"] == 2
        assert stats["metric_type_distribution"]["rule"] == 1
        assert "computation_time_ms" in stats

    def test_compute_stats_empty(self) -> None:
        engine = SimpleAnalyticsEngine()
        stats = engine.compute_stats()
        assert stats["total_events"] == 0
        assert stats["match_rate"] == 0.0
        assert stats["avg_value"] == 0.0


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestAggregatorsReexport:
    def test_importable_from_analytics_package(self) -> None:
        from talkex.analytics import (
            SimpleAnalyticsEngine,
            aggregate_by_dimension,
            aggregate_temporal,
            filter_events,
        )

        assert SimpleAnalyticsEngine is not None
        assert aggregate_by_dimension is not None
        assert aggregate_temporal is not None
        assert filter_events is not None
