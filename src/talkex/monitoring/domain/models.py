"""Monitoring domain models (frozen/strict, per ADR-002).

`Alert` is the evidence-backed domain event emitted when a critical DSL rule fires on
a context window (blueprint D2/D4). `SessionState` is the explicit session state machine
(blueprint D2, mirroring livekit agent_session.py).
"""

from __future__ import annotations

from enum import StrEnum
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field

from talkex.models.rule_execution import EvidenceItem
from talkex.models.types import ConversationId

AlertId = NewType("AlertId", str)


class SessionState(StrEnum):
    """Explicit lifecycle of a live monitoring session (blueprint D2)."""

    INITIALIZING = "initializing"
    LISTENING = "listening"
    CLOSING = "closing"


class Alert(BaseModel):
    """An evidence-backed alert raised when a critical rule matches a window.

    Attributes:
        alert_id: Unique identifier (format: alert_<uuid4>).
        conversation_id: The conversation the alert belongs to.
        window_id: The context window that triggered the match.
        rule_name: Name of the DSL rule that fired.
        evidence: Per-predicate evidence items (reuses the rules-engine EvidenceItem).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    alert_id: AlertId
    conversation_id: ConversationId
    window_id: str
    rule_name: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
