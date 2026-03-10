"""Event builders — transform domain artifacts into AnalyticsEvent objects.

Pure functions that convert stabilized pipeline outputs (Prediction,
RuleExecution) into AnalyticsEvent objects. This decouples the aggregation
layer from the specifics of each domain entity.

Each builder produces a single AnalyticsEvent from a single domain artifact,
preserving lineage via source_id and metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from semantic_conversation_engine.analytics.config import MetricType
from semantic_conversation_engine.analytics.models import AnalyticsEvent

if TYPE_CHECKING:
    from semantic_conversation_engine.models.prediction import Prediction
    from semantic_conversation_engine.models.rule_execution import RuleExecution


def prediction_to_event(
    prediction: Prediction,
    *,
    event_id: str,
    timestamp: datetime,
) -> AnalyticsEvent:
    """Convert a Prediction to an AnalyticsEvent.

    Args:
        prediction: Domain Prediction entity.
        event_id: Unique event identifier.
        timestamp: When the prediction was produced.

    Returns:
        AnalyticsEvent with classification metric type.
    """
    return AnalyticsEvent(
        event_id=event_id,
        event_type="prediction",
        source_id=prediction.source_id,
        source_type=prediction.source_type.value,
        timestamp=timestamp,
        metric_type=MetricType.CLASSIFICATION,
        value=prediction.score,
        label=prediction.label,
        matched=prediction.is_above_threshold,
        metadata={
            "prediction_id": str(prediction.prediction_id),
            "confidence": prediction.confidence,
            "threshold": prediction.threshold,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
        },
    )


def rule_execution_to_event(
    execution: RuleExecution,
    *,
    event_id: str,
    timestamp: datetime,
) -> AnalyticsEvent:
    """Convert a RuleExecution to an AnalyticsEvent.

    Args:
        execution: Domain RuleExecution entity.
        event_id: Unique event identifier.
        timestamp: When the rule was evaluated.

    Returns:
        AnalyticsEvent with rule metric type.
    """
    return AnalyticsEvent(
        event_id=event_id,
        event_type="rule_execution",
        source_id=execution.source_id,
        source_type=execution.source_type.value,
        timestamp=timestamp,
        metric_type=MetricType.RULE,
        value=execution.score,
        label=execution.rule_name,
        matched=execution.matched,
        metadata={
            "rule_id": str(execution.rule_id),
            "rule_name": execution.rule_name,
            "execution_time_ms": execution.execution_time_ms,
            "evidence_count": len(execution.evidence),
        },
    )
