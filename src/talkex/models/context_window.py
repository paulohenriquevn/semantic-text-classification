"""ContextWindow model — a sliding window of adjacent turns for contextual analysis.

A ContextWindow captures multi-turn dependencies (disambiguation, intent shifts,
objection patterns) by grouping N adjacent turns from a conversation. It is the
primary unit for contextual classification and semantic rule evaluation.

See PRD §11 for the data model specification.
See ADR-002 for the frozen/strict design decision.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from talkex.models.types import ConversationId, TurnId, WindowId


class ContextWindow(BaseModel):
    """A sliding window of adjacent turns within a conversation.

    Args:
        window_id: Unique identifier. Format: win_<uuid4>.
        conversation_id: Reference to the parent conversation. Format: conv_<uuid4>.
        turn_ids: Ordered list of turn IDs in this window. Must not be empty.
        window_text: Concatenated text of all turns in window (for search/embeddings).
        start_index: Index of first turn in conversation (0-based).
        end_index: Index of last turn in conversation (0-based).
        window_size: Number of turns in this window (for audit/reproducibility).
        stride: Stride value used to generate this window (for reproducibility).
        metadata: Additional window-level metadata (speaker counts, features, etc.).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    window_id: WindowId
    conversation_id: ConversationId
    turn_ids: list[TurnId]
    window_text: str
    start_index: int
    end_index: int
    window_size: int
    stride: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("window_id")
    @classmethod
    def window_id_must_not_be_empty(cls, v: WindowId) -> WindowId:
        """Reject empty or whitespace-only window IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("window_id must not be empty or whitespace-only")
        return v

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_must_not_be_empty(cls, v: ConversationId) -> ConversationId:
        """Reject empty or whitespace-only conversation IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("conversation_id must not be empty or whitespace-only")
        return v

    @field_validator("window_text")
    @classmethod
    def window_text_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only window text.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("window_text must not be empty or whitespace-only")
        return v

    @field_validator("turn_ids")
    @classmethod
    def turn_ids_must_not_be_empty(cls, v: list[TurnId]) -> list[TurnId]:
        """A context window must contain at least one turn."""
        if not v:
            raise ValueError("turn_ids must contain at least one turn")
        return v

    @field_validator("window_size")
    @classmethod
    def window_size_must_be_positive(cls, v: int) -> int:
        """Window size represents turn count — must be at least 1."""
        if v < 1:
            raise ValueError("window_size must be at least 1")
        return v

    @field_validator("stride")
    @classmethod
    def stride_must_be_positive(cls, v: int) -> int:
        """Stride represents the step between windows — must be at least 1."""
        if v < 1:
            raise ValueError("stride must be at least 1")
        return v

    @field_validator("start_index")
    @classmethod
    def start_index_must_be_non_negative(cls, v: int) -> int:
        """Indices are 0-based positions — they cannot be negative."""
        if v < 0:
            raise ValueError("start_index must be non-negative")
        return v

    @model_validator(mode="after")
    def cross_field_invariants(self) -> "ContextWindow":
        """Validate cross-field invariants.

        - end_index >= start_index
        - window_size == len(turn_ids)
        """
        if self.end_index < self.start_index:
            raise ValueError(f"end_index ({self.end_index}) must not precede start_index ({self.start_index})")
        if self.window_size != len(self.turn_ids):
            raise ValueError(f"window_size ({self.window_size}) must equal len(turn_ids) ({len(self.turn_ids)})")
        return self
