"""Conversation model — the atomic unit of the Semantic Conversation Intelligence Engine.

A Conversation represents a complete interaction between customer and agent,
originating from voice (ASR transcription), chat, email, or tickets. It carries
channel, temporal, and operational metadata used throughout the pipeline for
filtering, classification, rule evaluation, and analytics.

See PRD §11 for the data model specification.
See ADR-002 for the frozen/strict design decision.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from semantic_conversation_engine.models.enums import Channel
from semantic_conversation_engine.models.types import ConversationId


class Conversation(BaseModel):
    """A complete interaction between customer and agent.

    Args:
        conversation_id: Unique identifier. Format: conv_<uuid4>.
        channel: Communication channel (voice, chat, email, ticket).
        start_time: When the conversation began.
        end_time: When the conversation ended. None if still in progress.
        customer_id: Customer identifier, when permitted by privacy policy.
        product: Product or service related to this conversation.
        queue: Service queue that handled this conversation.
        region: Geographic region of the conversation.
        metadata: Additional operational metadata (CRM data, tags, etc.).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    conversation_id: ConversationId
    channel: Channel
    start_time: datetime
    end_time: datetime | None = None
    customer_id: str | None = None
    product: str | None = None
    queue: str | None = None
    region: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

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

    @model_validator(mode="after")
    def end_time_must_not_precede_start_time(self) -> "Conversation":
        """Ensure end_time >= start_time when both are present."""
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time.isoformat()}) must not precede start_time ({self.start_time.isoformat()})"
            )
        return self
