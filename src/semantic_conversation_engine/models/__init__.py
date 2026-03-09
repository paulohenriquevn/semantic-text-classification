"""Core domain models for the Semantic Conversation Intelligence Engine.

This package contains the pydantic data models that define the contracts
between pipeline stages: Conversation, Turn, ContextWindow, EmbeddingRecord,
Prediction, and RuleExecution.

All models are immutable (frozen) and strictly typed (see ADR-002).
"""

from semantic_conversation_engine.models.context_window import ContextWindow
from semantic_conversation_engine.models.conversation import Conversation
from semantic_conversation_engine.models.embedding_record import EmbeddingRecord
from semantic_conversation_engine.models.enums import (
    Channel,
    ObjectType,
    PoolingStrategy,
    SpeakerRole,
)
from semantic_conversation_engine.models.prediction import Prediction
from semantic_conversation_engine.models.rule_execution import EvidenceItem, RuleExecution
from semantic_conversation_engine.models.turn import Turn
from semantic_conversation_engine.models.types import (
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
