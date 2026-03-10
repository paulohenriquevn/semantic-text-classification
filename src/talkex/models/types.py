"""Domain type primitives for the TalkEx — Conversation Intelligence Engine.

NewType wrappers provide static type safety for domain IDs, preventing
accidental confusion between conversation IDs, turn IDs, etc. at type-check
time (mypy enforces these boundaries).

Type aliases (Score, Vector) are documentation aids — they do NOT enforce
constraints at runtime. Runtime validation happens in pydantic model validators.

ID format convention:
    conv_<uuid4>   — ConversationId
    turn_<uuid4>   — TurnId
    win_<uuid4>    — WindowId
    emb_<uuid4>    — EmbeddingId
    pred_<uuid4>   — PredictionId
    rule_<uuid4>   — RuleId
"""

from typing import NewType

from numpy import ndarray

ConversationId = NewType("ConversationId", str)
"""Unique identifier for a conversation. Format: conv_<uuid4>."""

TurnId = NewType("TurnId", str)
"""Unique identifier for a turn within a conversation. Format: turn_<uuid4>."""

WindowId = NewType("WindowId", str)
"""Unique identifier for a context window. Format: win_<uuid4>."""

EmbeddingId = NewType("EmbeddingId", str)
"""Unique identifier for an embedding record. Format: emb_<uuid4>."""

PredictionId = NewType("PredictionId", str)
"""Unique identifier for a classification prediction. Format: pred_<uuid4>."""

RuleId = NewType("RuleId", str)
"""Unique identifier for a rule in the semantic rule engine. Format: rule_<uuid4>."""

Score = float
"""Alias for float representing a score in [0.0, 1.0].

This is a documentation aid only — it does NOT enforce range at runtime.
Runtime range validation is handled by pydantic field_validators in models.
"""

Vector = ndarray
"""Alias for numpy.ndarray representing a dense embedding vector.

This is a boundary type for computation, NOT for storage. Data models
use list[float] for serialization (see ADR-003). Conversion to ndarray
happens at the boundary of consuming modules (embeddings, retrieval, rules).
"""
