"""Core domain models for the TalkEx — Conversation Intelligence Engine.

This package contains the pydantic data models that define the contracts
between pipeline stages: Conversation, Turn, ContextWindow, EmbeddingRecord,
Prediction, and RuleExecution.

All models are immutable (frozen) and strictly typed (see ADR-002).
"""

from talkex.models.context_window import ContextWindow
from talkex.models.conversation import Conversation
from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import (
    Channel,
    ObjectType,
    PoolingStrategy,
    SpeakerRole,
)
from talkex.models.prediction import Prediction
from talkex.models.rule_execution import EvidenceItem, RuleExecution
from talkex.models.turn import Turn
from talkex.models.types import (
    ConversationId,
    EmbeddingId,
    PredictionId,
    RuleId,
    Score,
    TurnId,
    Vector,
    WindowId,
)

__all__ = [
    "Channel",
    "ContextWindow",
    "Conversation",
    "ConversationId",
    "EmbeddingId",
    "EmbeddingRecord",
    "EvidenceItem",
    "ObjectType",
    "PoolingStrategy",
    "Prediction",
    "PredictionId",
    "RuleExecution",
    "RuleId",
    "Score",
    "SpeakerRole",
    "Turn",
    "TurnId",
    "Vector",
    "WindowId",
]
