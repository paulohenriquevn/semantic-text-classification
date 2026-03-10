"""Analytics aggregators — concrete implementations of AnalyticsAggregator.

Provides pure-function aggregation over AnalyticsEvent lists, producing
AggregationResult objects grouped by configurable dimensions.

Also provides SimpleAnalyticsEngine — a concrete implementation of both
AnalyticsAggregator and AnalyticsReporter protocols, executing queries
with filtering, grouping, and temporal bucketing.

Aggregation pipeline:
    events → filter → group_by → compute metrics → AggregationResult

Temporal bucketing:
    events → filter → bucket by window → compute per-bucket metrics → TrendPoint
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from semantic_conversation_engine.analytics.config import (
    AggregationWindow,
    AnalyticsLevel,
    MetricType,
)
from semantic_conversation_engine.analytics.models import (
    AggregationResult,
    AnalyticsEvent,
    AnalyticsQuery,
    MetricValue,
    TrendPoint,
)

# ---------------------------------------------------------------------------
# Pure aggregation functions
# ---------------------------------------------------------------------------


def aggregate_by_dimension(
    events: list[AnalyticsEvent],
    group_by: str,
    *,
    level: AnalyticsLevel = AnalyticsLevel.SYSTEM,
    window: AggregationWindow = AggregationWindow.DAILY,
) -> list[AggregationResult]:
    """Group events by a metadata dimension and compute metrics per group.

    Events are grouped by looking up ``group_by`` in event.metadata first,
    then falling back to event attributes (event_type, source_type, label,
    metric_type).

    Args:
        events: Events to aggregate.
        group_by: Dimension key for grouping.
        level: Aggregation level for the result.
        window: Temporal window for the result.

    Returns:
        List of AggregationResult, one per distinct group value.
        Sorted by group_value for determinism.
    """
    if not events:
        return []

    groups: dict[str, list[AnalyticsEvent]] = {}
    for event in events:
        key = _resolve_group_key(event, group_by)
        groups.setdefault(key, []).append(event)

    results = []
    for group_value in sorted(groups):
        group_events = groups[group_value]
        metrics = _compute_group_metrics(group_events, level)
        results.append(
            AggregationResult(
                group_key=group_by,
                group_value=group_value,
                level=level,
                window=window,
                metrics=metrics,
                event_count=len(group_events),
            )
        )
    return results


def aggregate_temporal(
    events: list[AnalyticsEvent],
    metric_name: str,
    window: AggregationWindow,
) -> list[TrendPoint]:
    """Bucket events by temporal window and compute a metric per bucket.

    Args:
        events: Events to bucket.
        metric_name: Name of the metric to compute (e.g., "avg_value", "match_rate").
        window: Temporal bucket size.

    Returns:
        Ordered list of TrendPoint objects (ascending by timestamp).
    """
    if not events:
        return []

    buckets: dict[datetime, list[AnalyticsEvent]] = {}
    for event in events:
        bucket_ts = _truncate_timestamp(event.timestamp, window)
        buckets.setdefault(bucket_ts, []).append(event)

    points = []
    for ts in sorted(buckets):
        bucket_events = buckets[ts]
        value = _compute_metric_value(bucket_events, metric_name)
        points.append(
            TrendPoint(
                timestamp=ts,
                window=window,
                metric_name=metric_name,
                value=value,
                count=len(bucket_events),
            )
        )
    return points


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_events(
    events: list[AnalyticsEvent],
    *,
    metric_types: list[MetricType] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    filters: dict[str, str] | None = None,
) -> list[AnalyticsEvent]:
    """Filter events by metric type, time range, and metadata key-values.

    Args:
        events: Events to filter.
        metric_types: If provided, only include events with matching metric_type.
        start_time: If provided, exclude events before this time.
        end_time: If provided, exclude events after this time.
        filters: If provided, only include events where metadata matches.

    Returns:
        Filtered list of events (order preserved).
    """
    result = events

    if metric_types:
        type_set = set(metric_types)
        result = [e for e in result if e.metric_type in type_set]

    if start_time is not None:
        result = [e for e in result if e.timestamp >= start_time]

    if end_time is not None:
        result = [e for e in result if e.timestamp <= end_time]

    if filters:
        result = [e for e in result if _matches_filters(e, filters)]

    return result


# ---------------------------------------------------------------------------
# SimpleAnalyticsEngine — concrete aggregator + reporter
# ---------------------------------------------------------------------------


class SimpleAnalyticsEngine:
    """Concrete analytics engine implementing AnalyticsAggregator and AnalyticsReporter.

    Satisfies both protocols via structural subtyping. Operates over
    in-memory event lists with filtering, grouping, and temporal bucketing.

    Usage::

        engine = SimpleAnalyticsEngine(events=event_list)
        results = engine.aggregate(events, "channel")
        trend = engine.trend(query, "match_rate")
    """

    def __init__(self, events: list[AnalyticsEvent] | None = None) -> None:
        self._events: list[AnalyticsEvent] = list(events) if events else []

    def add_events(self, events: list[AnalyticsEvent]) -> None:
        """Add events to the engine's internal store.

        Args:
            events: Events to add.
        """
        self._events.extend(events)

    @property
    def event_count(self) -> int:
        """Total number of stored events."""
        return len(self._events)

    def aggregate(
        self,
        events: list[AnalyticsEvent],
        group_by: str,
    ) -> list[AggregationResult]:
        """Aggregate events by dimension (AnalyticsAggregator protocol).

        Args:
            events: Events to aggregate.
            group_by: Dimension to group by.

        Returns:
            List of AggregationResult, one per group.
        """
        return aggregate_by_dimension(events, group_by)

    def query(
        self,
        analytics_query: AnalyticsQuery,
    ) -> list[AggregationResult]:
        """Execute an analytics query (AnalyticsReporter protocol).

        Filters stored events, then groups by the query's group_by dimension.

        Args:
            analytics_query: Typed query with filters, level, and grouping.

        Returns:
            List of AggregationResult matching the query.
        """
        filtered = filter_events(
            self._events,
            metric_types=analytics_query.metric_types or None,
            start_time=analytics_query.start_time,
            end_time=analytics_query.end_time,
            filters=analytics_query.filters or None,
        )

        group_by = analytics_query.group_by or "metric_type"

        results = aggregate_by_dimension(
            filtered,
            group_by,
            level=analytics_query.level,
            window=analytics_query.window,
        )

        return results[: analytics_query.max_results]

    def trend(
        self,
        analytics_query: AnalyticsQuery,
        metric_name: str,
    ) -> list[TrendPoint]:
        """Compute a time-series trend (AnalyticsReporter protocol).

        Filters stored events, then buckets by the query's window.

        Args:
            analytics_query: Query specifying time range and filters.
            metric_name: Metric to track over time.

        Returns:
            Ordered list of TrendPoint objects.
        """
        filtered = filter_events(
            self._events,
            metric_types=analytics_query.metric_types or None,
            start_time=analytics_query.start_time,
            end_time=analytics_query.end_time,
            filters=analytics_query.filters or None,
        )

        points = aggregate_temporal(filtered, metric_name, analytics_query.window)
        return points[: analytics_query.max_results]

    def compute_stats(self) -> dict[str, Any]:
        """Compute operational statistics about stored events.

        Returns:
            Dictionary with event counts, type distribution, and coverage.
        """
        start = time.monotonic()

        type_dist: dict[str, int] = {}
        matched_count = 0
        total_value = 0.0

        for event in self._events:
            type_dist[event.metric_type.value] = type_dist.get(event.metric_type.value, 0) + 1
            if event.matched:
                matched_count += 1
            total_value += event.value

        n = len(self._events)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        return {
            "total_events": n,
            "matched_events": matched_count,
            "match_rate": round(matched_count / n, 4) if n > 0 else 0.0,
            "avg_value": round(total_value / n, 4) if n > 0 else 0.0,
            "metric_type_distribution": type_dist,
            "computation_time_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_group_key(event: AnalyticsEvent, group_by: str) -> str:
    """Resolve the group key value from event metadata or attributes.

    Lookup order:
        1. event.metadata[group_by]
        2. getattr(event, group_by) (for built-in fields like label, source_type)
        3. "unknown" fallback
    """
    if group_by in event.metadata:
        return str(event.metadata[group_by])

    val = getattr(event, group_by, None)
    if val is not None:
        return str(val)

    return "unknown"


def _compute_group_metrics(
    events: list[AnalyticsEvent],
    level: AnalyticsLevel,
) -> list[MetricValue]:
    """Compute standard metrics for a group of events."""
    n = len(events)
    if n == 0:
        return []

    total_value = sum(e.value for e in events)
    matched = sum(1 for e in events if e.matched)

    # Determine dominant metric type
    type_counts: dict[MetricType, int] = {}
    for e in events:
        type_counts[e.metric_type] = type_counts.get(e.metric_type, 0) + 1
    dominant_type = max(type_counts, key=lambda t: type_counts[t])

    return [
        MetricValue(
            name="event_count",
            value=float(n),
            metric_type=dominant_type,
            level=level,
            count=n,
        ),
        MetricValue(
            name="avg_value",
            value=round(total_value / n, 4),
            metric_type=dominant_type,
            level=level,
            count=n,
        ),
        MetricValue(
            name="match_rate",
            value=round(matched / n, 4),
            metric_type=dominant_type,
            level=level,
            count=n,
        ),
    ]


def _compute_metric_value(events: list[AnalyticsEvent], metric_name: str) -> float:
    """Compute a single metric value for a list of events."""
    n = len(events)
    if n == 0:
        return 0.0

    if metric_name == "match_rate":
        matched = sum(1 for e in events if e.matched)
        return round(matched / n, 4)

    if metric_name == "event_count":
        return float(n)

    # Default: avg_value
    total = sum(e.value for e in events)
    return round(total / n, 4)


def _truncate_timestamp(ts: datetime, window: AggregationWindow) -> datetime:
    """Truncate a timestamp to the start of its temporal bucket."""
    if window == AggregationWindow.HOURLY:
        return ts.replace(minute=0, second=0, microsecond=0)

    if window == AggregationWindow.DAILY:
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)

    if window == AggregationWindow.WEEKLY:
        # Truncate to Monday of the week
        weekday = ts.weekday()
        from datetime import timedelta

        monday = ts - timedelta(days=weekday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    # MONTHLY
    return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _matches_filters(event: AnalyticsEvent, filters: dict[str, str]) -> bool:
    """Check if an event's metadata matches all filter key-value pairs."""
    for key, value in filters.items():
        # Check metadata first, then event attributes
        if key in event.metadata:
            if str(event.metadata[key]) != value:
                return False
        else:
            attr_val = getattr(event, key, None)
            if attr_val is None or str(attr_val) != value:
                return False
    return True
