"""Analytics subsystem configuration — enums and frozen config for analytical operations.

Defines the vocabulary for analytics: what levels of aggregation exist,
what time windows are available, and what types of metrics can be computed.

Enums:
    AnalyticsLevel    — granularity of analytical observation
    AggregationWindow — temporal grouping for trend analysis
    MetricType        — category of computed metric

Config:
    AnalyticsConfig   — frozen pydantic config governing analytics behavior
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsLevel(StrEnum):
    """Granularity level for analytics observations.

    Determines the resolution of analytical events and aggregations.
    Aligned with ObjectType (turn, context_window, conversation) plus
    system-wide aggregate.
    """

    TURN = "turn"
    CONTEXT_WINDOW = "context_window"
    CONVERSATION = "conversation"
    SYSTEM = "system"


class AggregationWindow(StrEnum):
    """Temporal window for grouping analytical data points.

    Controls the time bucket size for trend analysis and
    time-series aggregation.
    """

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class MetricType(StrEnum):
    """Category of computed metric.

    Classifies metrics by their origin in the pipeline,
    enabling filtered queries and grouped reporting.
    """

    CLASSIFICATION = "classification"
    RULE = "rule"
    RETRIEVAL = "retrieval"
    PIPELINE = "pipeline"


class AnalyticsConfig(BaseModel):
    """Configuration for the analytics subsystem.

    Controls default aggregation behavior, metric computation, and
    output settings.

    Args:
        default_level: Default analytics granularity.
        default_window: Default temporal aggregation window.
        max_trend_points: Maximum number of trend points per query.
        include_metadata: Whether to include source metadata in events.
        enabled_metric_types: Which metric types to compute.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    default_level: AnalyticsLevel = AnalyticsLevel.CONVERSATION
    default_window: AggregationWindow = AggregationWindow.DAILY
    max_trend_points: int = Field(default=100, ge=1)
    include_metadata: bool = True
    enabled_metric_types: list[MetricType] = Field(
        default_factory=lambda: list(MetricType),
    )
