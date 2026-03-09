"""Domain enums for the Semantic Conversation Intelligence Engine.

Enums enforce finite valid values across all pipeline stages, preventing
stringly-typed bugs. All enums use str mixin for JSON-safe serialization.
"""

from enum import StrEnum


class SpeakerRole(StrEnum):
    """Role of the speaker within a conversation turn.

    Used in Turn attribution, context window role-aware views,
    and structural predicates in the rule engine.
    """

    CUSTOMER = "customer"
    AGENT = "agent"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class Channel(StrEnum):
    """Communication channel from which a conversation originates.

    Used in Conversation metadata, structural predicates,
    and business filters in hybrid retrieval.
    """

    VOICE = "voice"
    CHAT = "chat"
    EMAIL = "email"
    TICKET = "ticket"


class ObjectType(StrEnum):
    """Granularity level of an object in the pipeline.

    Used in EmbeddingRecord and Prediction to identify whether
    the associated object is a turn, context window, or full conversation.
    """

    TURN = "turn"
    CONTEXT_WINDOW = "context_window"
    CONVERSATION = "conversation"


class PoolingStrategy(StrEnum):
    """Strategy for aggregating token embeddings into a single vector.

    Mean pooling is the baseline. Attention pooling is recommended
    for long context windows and conversations (see KB_Complementar §18.6).
    """

    MEAN = "mean"
    MAX = "max"
    ATTENTION = "attention"
