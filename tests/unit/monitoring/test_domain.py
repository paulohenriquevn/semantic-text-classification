"""Unit tests for monitoring domain models (T1.2)."""

import pytest
from pydantic import ValidationError

from talkex.models.rule_execution import EvidenceItem
from talkex.models.types import ConversationId
from talkex.monitoring.domain.models import Alert, AlertId, SessionState


def _evidence() -> EvidenceItem:
    return EvidenceItem(predicate_type="contains_any", matched_text="cancelar", score=1.0, threshold=0.5)


class TestAlert:
    def test_carries_evidence_and_is_frozen(self) -> None:
        alert = Alert(
            alert_id=AlertId("alert_1"),
            conversation_id=ConversationId("conv_1"),
            window_id="win_1",
            rule_name="cancellation_risk",
            evidence=[_evidence()],
        )
        assert alert.evidence[0]["matched_text"] == "cancelar"
        with pytest.raises(ValidationError):
            alert.rule_name = "other"  # type: ignore[misc]

    def test_evidence_defaults_empty(self) -> None:
        alert = Alert(
            alert_id=AlertId("alert_2"),
            conversation_id=ConversationId("conv_1"),
            window_id="win_1",
            rule_name="r",
        )
        assert alert.evidence == []


class TestSessionState:
    def test_lifecycle_values_present(self) -> None:
        assert SessionState.INITIALIZING == "initializing"
        assert SessionState.LISTENING == "listening"
        assert SessionState.CLOSING == "closing"
