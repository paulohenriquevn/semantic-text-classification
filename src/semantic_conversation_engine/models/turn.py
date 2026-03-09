"""Turn model — the finest-grained unit of a conversation.

A Turn represents an individual utterance within a conversation, attributed
to a speaker (customer, agent, system). Turns are the building blocks for
context windows and the finest granularity for embedding, classification,
and rule evaluation.

See PRD §11 for the data model specification.
See ADR-002 for the frozen/strict design decision.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from semantic_conversation_engine.models.enums import SpeakerRole
from semantic_conversation_engine.models.types import ConversationId, TurnId


class Turn(BaseModel):
    """An individual utterance within a conversation.

    Args:
        turn_id: Unique identifier. Format: turn_<uuid4>.
        conversation_id: Reference to the parent conversation. Format: conv_<uuid4>.
        speaker: Who spoke (customer, agent, system, unknown).
        raw_text: Original unmodified text as transcribed or input.
        start_offset: Start position in conversation (characters or milliseconds).
        end_offset: End position in conversation (characters or milliseconds).
        normalized_text: Lightly normalized text. None if normalization not yet applied.
        metadata: Additional turn-level metadata (ASR confidence, language, etc.).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    turn_id: TurnId
    conversation_id: ConversationId
    speaker: SpeakerRole
    raw_text: str
    start_offset: int
    end_offset: int
    normalized_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("turn_id")
    @classmethod
    def turn_id_must_not_be_empty(cls, v: TurnId) -> TurnId:
        """Reject empty or whitespace-only turn IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("turn_id must not be empty or whitespace-only")
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

    @field_validator("raw_text")
    @classmethod
    def raw_text_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only raw text.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("raw_text must not be empty or whitespace-only")
        return v

    @field_validator("start_offset")
    @classmethod
    def start_offset_must_be_non_negative(cls, v: int) -> int:
        """Offsets represent positions — they cannot be negative."""
        if v < 0:
            raise ValueError("start_offset must be non-negative")
        return v

    @model_validator(mode="after")
    def end_offset_must_exceed_start_offset(self) -> "Turn":
        """Ensure end_offset > start_offset.

        A turn must span at least one unit. Zero-length spans are degenerate
        and would contaminate segmentation, context windows, and indexing.
        """
        if self.end_offset <= self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be strictly greater than start_offset ({self.start_offset})"
            )
        return self
