"""Unit tests for analytics builders — domain artifact to AnalyticsEvent conversion.

Tests cover: prediction_to_event, rule_execution_to_event, metadata preservation,
field mapping, and TYPE_CHECKING import boundary.
"""

from datetime import UTC, datetime
from uuid import uuid4

from talkex.analytics.builders import (
    prediction_to_event,
    rule_execution_to_event,
)
from talkex.analytics.config import MetricType
from talkex.models.enums import ObjectType
from talkex.models.prediction import Prediction
from talkex.models.rule_execution import RuleExecution

_NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)


def _make_prediction(**overrides: object) -> Prediction:
    defaults: dict[str, object] = {
        "prediction_id": str(uuid4()),
        "source_id": "win_001",
        "source_type": ObjectType.CONTEXT_WINDOW,
        "label": "billing",
        "score": 0.85,
        "confidence": 0.9,
        "threshold": 0.5,
        "model_name": "similarity_v1",
        "model_version": "1.0.0",
    }
    defaults.update(overrides)
    return Prediction(**defaults)  # type: ignore[arg-type]


def _make_rule_execution(**overrides: object) -> RuleExecution:
    defaults: dict[str, object] = {
        "rule_id": str(uuid4()),
        "rule_name": "escalation_risk",
        "source_id": "win_002",
        "source_type": ObjectType.CONTEXT_WINDOW,
        "score": 0.75,
        "matched": True,
        "evidence": [
            {
                "predicate_type": "lexical",
                "field": "text",
                "operator": "contains",
                "expected": "cancel",
                "actual": "I want to cancel",
                "matched": True,
                "score": 1.0,
            },
        ],
        "execution_time_ms": 3.5,
    }
    defaults.update(overrides)
    return RuleExecution(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# prediction_to_event
# ---------------------------------------------------------------------------


class TestPredictionToEvent:
    def test_maps_basic_fields(self) -> None:
        pred = _make_prediction(source_id="win_010", label="billing", score=0.85)
        event = prediction_to_event(pred, event_id="evt_001", timestamp=_NOW)

        assert event.event_id == "evt_001"
        assert event.event_type == "prediction"
        assert event.source_id == "win_010"
        assert event.source_type == "context_window"
        assert event.timestamp == _NOW
        assert event.metric_type == MetricType.CLASSIFICATION
        assert event.value == 0.85
        assert event.label == "billing"

    def test_maps_matched_from_is_above_threshold(self) -> None:
        pred_above = _make_prediction(score=0.8, threshold=0.5)
        event_above = prediction_to_event(pred_above, event_id="e1", timestamp=_NOW)
        assert event_above.matched is True

        pred_below = _make_prediction(score=0.3, threshold=0.5)
        event_below = prediction_to_event(pred_below, event_id="e2", timestamp=_NOW)
        assert event_below.matched is False

    def test_preserves_metadata_lineage(self) -> None:
        pred = _make_prediction(
            model_name="e5_base",
            model_version="2.1.0",
            confidence=0.92,
            threshold=0.6,
        )
        event = prediction_to_event(pred, event_id="evt_meta", timestamp=_NOW)

        assert event.metadata["model_name"] == "e5_base"
        assert event.metadata["model_version"] == "2.1.0"
        assert event.metadata["confidence"] == 0.92
        assert event.metadata["threshold"] == 0.6
        assert "prediction_id" in event.metadata

    def test_preserves_prediction_id_in_metadata(self) -> None:
        pred = _make_prediction(prediction_id="pred_abc123")
        event = prediction_to_event(pred, event_id="evt_pid", timestamp=_NOW)
        assert event.metadata["prediction_id"] == "pred_abc123"

    def test_uses_provided_event_id_and_timestamp(self) -> None:
        ts = datetime(2026, 1, 15, 8, 30, 0, tzinfo=UTC)
        pred = _make_prediction()
        event = prediction_to_event(pred, event_id="custom_id", timestamp=ts)
        assert event.event_id == "custom_id"
        assert event.timestamp == ts


# ---------------------------------------------------------------------------
# rule_execution_to_event
# ---------------------------------------------------------------------------


class TestRuleExecutionToEvent:
    def test_maps_basic_fields(self) -> None:
        exe = _make_rule_execution(
            source_id="win_020",
            rule_name="fraud_check",
            score=0.9,
            matched=True,
        )
        event = rule_execution_to_event(exe, event_id="evt_r01", timestamp=_NOW)

        assert event.event_id == "evt_r01"
        assert event.event_type == "rule_execution"
        assert event.source_id == "win_020"
        assert event.source_type == "context_window"
        assert event.timestamp == _NOW
        assert event.metric_type == MetricType.RULE
        assert event.value == 0.9
        assert event.label == "fraud_check"
        assert event.matched is True

    def test_maps_unmatched_execution(self) -> None:
        exe = _make_rule_execution(matched=False, evidence=[], score=0.2)
        event = rule_execution_to_event(exe, event_id="evt_r02", timestamp=_NOW)
        assert event.matched is False
        assert event.value == 0.2

    def test_preserves_metadata_lineage(self) -> None:
        exe = _make_rule_execution(
            rule_id="rule_xyz",
            rule_name="compliance_check",
            execution_time_ms=5.2,
        )
        event = rule_execution_to_event(exe, event_id="evt_rmeta", timestamp=_NOW)

        assert event.metadata["rule_id"] == "rule_xyz"
        assert event.metadata["rule_name"] == "compliance_check"
        assert event.metadata["execution_time_ms"] == 5.2
        assert event.metadata["evidence_count"] == 1  # one evidence item in default

    def test_evidence_count_reflects_evidence_length(self) -> None:
        exe = _make_rule_execution(
            evidence=[
                {
                    "predicate_type": "lexical",
                    "field": "text",
                    "operator": "contains",
                    "expected": "a",
                    "actual": "a b",
                    "matched": True,
                    "score": 1.0,
                },
                {
                    "predicate_type": "lexical",
                    "field": "text",
                    "operator": "contains",
                    "expected": "b",
                    "actual": "a b",
                    "matched": True,
                    "score": 1.0,
                },
            ],
        )
        event = rule_execution_to_event(exe, event_id="evt_ec", timestamp=_NOW)
        assert event.metadata["evidence_count"] == 2

    def test_uses_provided_event_id_and_timestamp(self) -> None:
        ts = datetime(2026, 2, 20, 14, 0, 0, tzinfo=UTC)
        exe = _make_rule_execution()
        event = rule_execution_to_event(exe, event_id="custom_rule_id", timestamp=ts)
        assert event.event_id == "custom_rule_id"
        assert event.timestamp == ts


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestBuildersReexport:
    def test_importable_from_analytics_package(self) -> None:
        from talkex.analytics import (
            prediction_to_event,
            rule_execution_to_event,
        )

        assert prediction_to_event is not None
        assert rule_execution_to_event is not None
