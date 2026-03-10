"""Analytics subsystem for the Semantic Conversation Intelligence Engine.

Provides structured analytical views over pipeline outputs: events from
classification and rule evaluation, computed metrics, temporal aggregations,
and typed queries. Consumes stabilized domain artifacts (Prediction,
RuleExecution, Conversation metadata) and produces auditable analytical data.
"""

from semantic_conversation_engine.analytics.aggregators import (
    SimpleAnalyticsEngine,
    aggregate_by_dimension,
    aggregate_temporal,
    filter_events,
)
from semantic_conversation_engine.analytics.benchmark import (
    AnalyticsBenchmarkConfig,
    AnalyticsBenchmarkReport,
    AnalyticsBenchmarkRunner,
    QueryScenarioResult,
)
from semantic_conversation_engine.analytics.builders import (
    prediction_to_event,
    rule_execution_to_event,
)
from semantic_conversation_engine.analytics.config import (
    AggregationWindow,
    AnalyticsConfig,
    AnalyticsLevel,
    MetricType,
)
from semantic_conversation_engine.analytics.metrics import (
    compute_analytics_metrics,
)
from semantic_conversation_engine.analytics.models import (
    AggregationResult,
    AnalyticsEvent,
    AnalyticsQuery,
    MetricValue,
    TrendPoint,
)
from semantic_conversation_engine.analytics.query_runner import (
    AnalyticsQueryRunner,
)
from semantic_conversation_engine.analytics.report import (
    AnalyticsReport,
    AnalyticsSection,
    GroupedMetric,
    TrendSeries,
)

__all__ = [
    "AggregationResult",
    "AggregationWindow",
    "AnalyticsBenchmarkConfig",
    "AnalyticsBenchmarkReport",
    "AnalyticsBenchmarkRunner",
    "AnalyticsConfig",
    "AnalyticsEvent",
    "AnalyticsLevel",
    "AnalyticsQuery",
    "AnalyticsQueryRunner",
    "AnalyticsReport",
    "AnalyticsSection",
    "GroupedMetric",
    "MetricType",
    "MetricValue",
    "QueryScenarioResult",
    "SimpleAnalyticsEngine",
    "TrendPoint",
    "TrendSeries",
    "aggregate_by_dimension",
    "aggregate_temporal",
    "compute_analytics_metrics",
    "filter_events",
    "prediction_to_event",
    "rule_execution_to_event",
]
