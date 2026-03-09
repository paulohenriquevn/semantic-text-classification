"""Boundary objects for embedding generation.

EmbeddingInput and EmbeddingBatch are boundary objects — they represent
requests for embedding generation arriving from pipeline stages.
They are NOT domain entities like EmbeddingRecord; they are the payload
that the EmbeddingGenerator transforms INTO EmbeddingRecords.

EmbeddingInput carries the text to embed, the source object identity
(type + id), and optional metadata for routing and observability.

EmbeddingBatch groups multiple inputs for efficient batch generation.

See ADR-002 for the frozen/strict design decision.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from semantic_conversation_engine.models.enums import Channel, ObjectType
from semantic_conversation_engine.models.types import EmbeddingId


class EmbeddingInput(BaseModel):
    """A single embedding generation request.

    Represents one text-to-vector transformation request. The source
    identity (object_type + object_id) links the generated EmbeddingRecord
    back to the originating Turn, ContextWindow, or Conversation.

    Args:
        embedding_id: Pre-assigned ID for the generated EmbeddingRecord.
        object_type: Granularity level of the source object.
        object_id: ID of the source object (turn_id, window_id, etc.).
        text: The text to embed. Must not be empty.
        metadata: Additional context for routing or observability.
        language: ISO 639-1 language code. None if unknown.
        channel: Communication channel of the source conversation.
            None if not applicable.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    embedding_id: EmbeddingId
    object_type: ObjectType
    object_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    language: str | None = None
    channel: Channel | None = None

    @field_validator("embedding_id")
    @classmethod
    def embedding_id_must_not_be_empty(cls, v: EmbeddingId) -> EmbeddingId:
        """Validates only — does NOT normalize."""
        if not v.strip():
            raise ValueError("embedding_id must not be empty or whitespace-only")
        return v

    @field_validator("object_id")
    @classmethod
    def object_id_must_not_be_empty(cls, v: str) -> str:
        """Validates only — does NOT normalize."""
        if not v.strip():
            raise ValueError("object_id must not be empty or whitespace-only")
        return v

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        """Validates only — does NOT normalize."""
        if not v.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return v


class EmbeddingBatch(BaseModel):
    """A batch of embedding generation requests.

    Groups multiple EmbeddingInput objects for efficient batch processing.
    The batch itself is validated to be non-empty — generating embeddings
    for zero inputs is a no-op that should be caught early.

    Args:
        items: Ordered list of embedding inputs. Must not be empty.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    items: list[EmbeddingInput]

    @field_validator("items")
    @classmethod
    def items_must_not_be_empty(cls, v: list[EmbeddingInput]) -> list[EmbeddingInput]:
        """A batch must contain at least one item."""
        if not v:
            raise ValueError("items must not be empty")
        return v
