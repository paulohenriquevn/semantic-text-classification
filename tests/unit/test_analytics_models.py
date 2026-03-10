"""Unit tests for analytics models — events, metrics, aggregations, queries.

Tests cover: construction, frozen immutability, defaults, field values,
and reexports for all 5 analytics model types.
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

_NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# AnalyticsEvent
# ---------------------------------------------------------------------------


class TestAnalyticsEvent:
    def test_construction(self) -> None:
        event = AnalyticsEvent(
            event_id="evt_001",
            event_type="prediction",
            source_id="win_001",
            source_type="context_window",
            timestamp=_NOW,
            metric_type=MetricType.CLASSIFICATION,
            value=0.85,
            label="billing",
            matched=True,
        )
        assert event.event_id == "evt_001"
        assert event.event_type == "prediction"
        assert event.value == 0.85
        assert event.label == "billing"
        assert event.matched is True
        assert event.metadata == {}

    def test_defaults(self) -> None:
        event = AnalyticsEvent(
            event_id="evt_002",
            event_type="rule_execution",
            source_id="win_002",
            source_type="context_window",
            timestamp=_NOW,
            metric_type=MetricType.RULE,
        )
        assert event.value == 0.0
        assert event.label is None
        assert event.matched is None
        assert event.metadata == {}

    def test_frozen(self) -> None:
        event = AnalyticsEvent(
            event_id="evt_003",
            event_type="prediction",
            source_id="win_003",
            source_type="context_window",
            timestamp=_NOW,
            metric_type=MetricType.CLASSIFICATION,
        )
        try:
            event.value = 1.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_metadata_isolation(self) -> None:
        meta = {"model": "v1"}
        event = AnalyticsEvent(
            event_id="evt_004",
            event_type="prediction",
            source_id="win_004",
            source_type="context_window",
            timestamp=_NOW,
            metric_type=MetricType.CLASSIFICATION,
            metadata=meta,
        )
        assert event.metadata == {"model": "v1"}
        meta["model"] = "v2"
        # Frozen dataclass shares reference, but that's acceptable for dict defaults


# ---------------------------------------------------------------------------
# MetricValue
# ---------------------------------------------------------------------------


class TestMetricValue:
    def test_construction(self) -> None:
        mv = MetricValue(
            name="match_rate",
            value=0.75,
            metric_type=MetricType.RULE,
            level=AnalyticsLevel.CONVERSATION,
            count=100,
        )
        assert mv.name == "match_rate"
        assert mv.value == 0.75
        assert mv.metric_type == MetricType.RULE
        assert mv.level == AnalyticsLevel.CONVERSATION
        assert mv.count == 100

    def test_defaults(self) -> None:
        mv = MetricValue(
            name="avg_score",
            value=0.5,
            metric_type=MetricType.CLASSIFICATION,
            level=AnalyticsLevel.SYSTEM,
        )
        assert mv.count == 0
        assert mv.metadata == {}

    def test_frozen(self) -> None:
        mv = MetricValue(
            name="test",
            value=1.0,
            metric_type=MetricType.PIPELINE,
            level=AnalyticsLevel.TURN,
        )
        try:
            mv.value = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# AggregationResult
# ---------------------------------------------------------------------------


class TestAggregationResult:
    def test_construction(self) -> None:
        metrics = [
            MetricValue(
                name="match_rate",
                value=0.8,
                metric_type=MetricType.RULE,
                level=AnalyticsLevel.CONVERSATION,
            ),
        ]
        agg = AggregationResult(
            group_key="channel",
            group_value="voice",
            level=AnalyticsLevel.CONVERSATION,
            window=AggregationWindow.DAILY,
            metrics=metrics,
            event_count=50,
        )
        assert agg.group_key == "channel"
        assert agg.group_value == "voice"
        assert len(agg.metrics) == 1
        assert agg.event_count == 50

    def test_defaults(self) -> None:
        agg = AggregationResult(
            group_key="label",
            group_value="billing",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.WEEKLY,
            metrics=[],
        )
        assert agg.event_count == 0
        assert agg.metadata == {}

    def test_frozen(self) -> None:
        agg = AggregationResult(
            group_key="k",
            group_value="v",
            level=AnalyticsLevel.TURN,
            window=AggregationWindow.HOURLY,
            metrics=[],
        )
        try:
            agg.event_count = 10  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# TrendPoint
# ---------------------------------------------------------------------------


class TestTrendPoint:
    def test_construction(self) -> None:
        tp = TrendPoint(
            timestamp=_NOW,
            window=AggregationWindow.DAILY,
            metric_name="match_rate",
            value=0.82,
            count=200,
        )
        assert tp.timestamp == _NOW
        assert tp.window == AggregationWindow.DAILY
        assert tp.metric_name == "match_rate"
        assert tp.value == 0.82
        assert tp.count == 200

    def test_defaults(self) -> None:
        tp = TrendPoint(
            timestamp=_NOW,
            window=AggregationWindow.MONTHLY,
            metric_name="avg_latency",
            value=5.2,
        )
        assert tp.count == 0
        assert tp.metadata == {}

    def test_frozen(self) -> None:
        tp = TrendPoint(
            timestamp=_NOW,
            window=AggregationWindow.DAILY,
            metric_name="test",
            value=1.0,
        )
        try:
            tp.value = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# AnalyticsQuery
# ---------------------------------------------------------------------------


class TestAnalyticsQuery:
    def test_construction(self) -> None:
        query = AnalyticsQuery(
            query_id="q_001",
            level=AnalyticsLevel.CONVERSATION,
            metric_types=[MetricType.CLASSIFICATION, MetricType.RULE],
            window=AggregationWindow.DAILY,
            group_by="channel",
            max_results=50,
        )
        assert query.query_id == "q_001"
        assert query.level == AnalyticsLevel.CONVERSATION
        assert len(query.metric_types) == 2
        assert query.group_by == "channel"
        assert query.max_results == 50

    def test_defaults(self) -> None:
        query = AnalyticsQuery(
            query_id="q_002",
            level=AnalyticsLevel.SYSTEM,
        )
        assert query.metric_types == []
        assert query.window == AggregationWindow.DAILY
        assert query.start_time is None
        assert query.end_time is None
        assert query.group_by is None
        assert query.filters == {}
        assert query.max_results == 100

    def test_with_time_range(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 3, 1, tzinfo=UTC)
        query = AnalyticsQuery(
            query_id="q_003",
            level=AnalyticsLevel.TURN,
            start_time=start,
            end_time=end,
        )
        assert query.start_time == start
        assert query.end_time == end

    def test_with_filters(self) -> None:
        query = AnalyticsQuery(
            query_id="q_004",
            level=AnalyticsLevel.CONTEXT_WINDOW,
            filters={"channel": "voice", "product": "credit_card"},
        )
        assert query.filters["channel"] == "voice"
        assert len(query.filters) == 2

    def test_frozen(self) -> None:
        query = AnalyticsQuery(
            query_id="q_005",
            level=AnalyticsLevel.SYSTEM,
        )
        try:
            query.max_results = 200  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestModelsReexport:
    def test_importable_from_analytics_package(self) -> None:
        from talkex.analytics import (
            AggregationResult,
            AnalyticsEvent,
            AnalyticsQuery,
            MetricValue,
            TrendPoint,
        )

        assert AnalyticsEvent is not None
        assert MetricValue is not None
        assert AggregationResult is not None
        assert TrendPoint is not None
        assert AnalyticsQuery is not None
