"""Analytics internal models — events, metrics, aggregations, queries.

Pipeline-internal data objects for the analytics subsystem.
These are analytical views over domain entities, NOT domain entities themselves.
The analytics layer consumes stabilized artifacts (Prediction, RuleExecution,
Conversation metadata) and produces structured analytical outputs.

Model hierarchy:
    AnalyticsEvent    — atomic analytical observation (a prediction, a rule firing)
    MetricValue       — named numeric metric with type and level
    AggregationResult — grouped metrics by dimension (channel, label, time)
    TrendPoint        — single data point in a time-series
    AnalyticsQuery    — typed query for analytical data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from talkex.analytics.config import (
    AggregationWindow,
    AnalyticsLevel,
    MetricType,
)


@dataclass(frozen=True)
class AnalyticsEvent:
    """Atomic analytical observation from the pipeline.

    Represents a single event worth tracking: a classification prediction,
    a rule execution, a retrieval query, or a pipeline completion.

    Args:
        event_id: Unique identifier for this event.
        event_type: Type of event (e.g., "prediction", "rule_execution").
        source_id: ID of the originating object (turn/window/conversation).
        source_type: Granularity level of the source.
        timestamp: When the event occurred.
        metric_type: Pipeline stage that produced this event.
        value: Primary numeric value of the event (score, latency, etc.).
        label: Associated label, if any (predicted label, rule name).
        matched: Whether the event represents a positive match.
        metadata: Additional event context.
    """

    event_id: str
    event_type: str
    source_id: str
    source_type: str
    timestamp: datetime
    metric_type: MetricType
    value: float = 0.0
    label: str | None = None
    matched: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricValue:
    """Named numeric metric with type and level annotation.

    A computed metric derived from one or more events. Carries enough
    context to be grouped, filtered, and compared.

    Args:
        name: Metric name (e.g., "match_rate", "avg_score", "p95_latency").
        value: Numeric value of the metric.
        metric_type: Pipeline stage category.
        level: Aggregation granularity.
        count: Number of observations used to compute this metric.
        metadata: Additional metric context (model_version, config, etc.).
    """

    name: str
    value: float
    metric_type: MetricType
    level: AnalyticsLevel
    count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregationResult:
    """Grouped metrics by one or more dimensions.

    Represents the output of an aggregation operation: metrics computed
    over a group of events sharing a common dimension value (channel,
    label, time window, etc.).

    Args:
        group_key: The dimension name (e.g., "channel", "label", "rule_name").
        group_value: The dimension value (e.g., "voice", "billing", "fraud_rule").
        level: Aggregation granularity.
        window: Temporal grouping used.
        metrics: Computed metrics for this group.
        event_count: Total number of events in this group.
        metadata: Additional aggregation context.
    """

    group_key: str
    group_value: str
    level: AnalyticsLevel
    window: AggregationWindow
    metrics: list[MetricValue]
    event_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrendPoint:
    """Single data point in a time-series trend.

    Represents a metric value at a specific point in time, used for
    building time-series visualizations and drift detection.

    Args:
        timestamp: Start of the time bucket.
        window: Temporal bucket size.
        metric_name: Name of the tracked metric.
        value: Metric value for this time bucket.
        count: Number of observations in this bucket.
        metadata: Additional point context.
    """

    timestamp: datetime
    window: AggregationWindow
    metric_name: str
    value: float
    count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyticsQuery:
    """Typed query for analytical data.

    Specifies what analytics data to retrieve or compute, with filters
    for level, metric type, time range, and grouping.

    Args:
        query_id: Unique identifier for this query.
        level: Granularity level to query at.
        metric_types: Which pipeline stages to include.
        window: Temporal aggregation window.
        start_time: Start of time range filter. None means no lower bound.
        end_time: End of time range filter. None means no upper bound.
        group_by: Dimension to group results by (e.g., "channel", "label").
        filters: Additional key-value filters for narrowing results.
        max_results: Maximum number of results to return.
    """

    query_id: str
    level: AnalyticsLevel
    metric_types: list[MetricType] = field(default_factory=list)
    window: AggregationWindow = AggregationWindow.DAILY
    start_time: datetime | None = None
    end_time: datetime | None = None
    group_by: str | None = None
    filters: dict[str, str] = field(default_factory=dict)
    max_results: int = 100
