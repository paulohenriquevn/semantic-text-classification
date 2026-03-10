"""Input contracts for the text processing pipeline.

TranscriptInput is a boundary object — it represents raw data arriving
from external sources (API handlers, file parsers, message queues) before
any domain processing. It is NOT a domain entity like Conversation; it is
the payload that the pipeline transforms INTO domain entities.

See ADR-002 for the frozen/strict design decision.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from talkex.ingestion.enums import SourceFormat
from talkex.models.enums import Channel
from talkex.models.types import ConversationId


class SpeakerHint(BaseModel):
    """A hint about speaker identity for a section of text.

    Used when the source system provides partial speaker attribution
    (e.g. diarization output, CRM metadata about participants).

    Args:
        speaker_label: The label used in the transcript (e.g. 'Speaker 1', 'CUSTOMER').
        role: The role to map this label to (e.g. 'customer', 'agent').
    """

    model_config = ConfigDict(frozen=True, strict=True)

    speaker_label: str
    role: str

    @field_validator("speaker_label")
    @classmethod
    def speaker_label_must_not_be_empty(cls, v: str) -> str:
        """Validates only — does NOT normalize. The original value is preserved."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("speaker_label must not be empty or whitespace-only")
        return v

    @field_validator("role")
    @classmethod
    def role_must_not_be_empty(cls, v: str) -> str:
        """Validates only — does NOT normalize. The original value is preserved."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("role must not be empty or whitespace-only")
        return v


class QualitySignals(BaseModel):
    """Quality indicators from the upstream source (e.g. ASR system).

    All fields are optional because different sources provide
    different quality information. None means "unknown".

    Args:
        asr_confidence: Overall ASR confidence in [0.0, 1.0]. None if not from voice.
        language_code: ISO 639-1 language code (e.g. 'pt', 'en'). None if unknown.
        audio_duration_ms: Duration of the source audio in milliseconds. None if not voice.
        word_error_rate: Estimated WER in [0.0, 1.0]. None if unknown.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    asr_confidence: float | None = None
    language_code: str | None = None
    audio_duration_ms: int | None = None
    word_error_rate: float | None = None

    @field_validator("asr_confidence")
    @classmethod
    def asr_confidence_must_be_in_unit_range(cls, v: float | None) -> float | None:
        """ASR confidence must be in [0.0, 1.0] when present."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError(f"asr_confidence must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("word_error_rate")
    @classmethod
    def word_error_rate_must_be_in_unit_range(cls, v: float | None) -> float | None:
        """Word error rate must be in [0.0, 1.0] when present."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError(f"word_error_rate must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("audio_duration_ms")
    @classmethod
    def audio_duration_must_be_positive(cls, v: int | None) -> int | None:
        """Audio duration must be positive when present."""
        if v is not None and v <= 0:
            raise ValueError(f"audio_duration_ms must be positive, got {v}")
        return v


class TranscriptInput(BaseModel):
    """Raw transcript input for the text processing pipeline.

    This is a boundary object, NOT a domain entity. It represents
    unprocessed data arriving from external systems. The pipeline
    transforms it into Conversation, Turn, and ContextWindow instances.

    Args:
        conversation_id: Unique identifier for the conversation.
        channel: Communication channel (voice, chat, email, ticket).
        raw_text: The complete raw transcript text.
        source_format: How the transcript is structured.
        speaker_hints: Optional hints for speaker identification.
        quality_signals: Optional quality indicators from upstream.
        metadata: Additional source metadata.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    conversation_id: ConversationId
    channel: Channel
    raw_text: str
    source_format: SourceFormat
    speaker_hints: list[SpeakerHint] = Field(default_factory=list)
    quality_signals: QualitySignals | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_must_not_be_empty(cls, v: ConversationId) -> ConversationId:
        """Validates only — does NOT normalize. The original value is preserved."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("conversation_id must not be empty or whitespace-only")
        return v

    @field_validator("raw_text")
    @classmethod
    def raw_text_must_not_be_empty(cls, v: str) -> str:
        """Validates only — does NOT normalize. The original value is preserved."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("raw_text must not be empty or whitespace-only")
        return v
