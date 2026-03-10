"""Unit tests for TranscriptInput, SpeakerHint, QualitySignals, and SourceFormat.

Tests cover: construction, validation, strict mode, immutability,
serialization, boundary deserialization, and reexport.
"""

from typing import Any

import pytest

from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import (
    QualitySignals,
    SpeakerHint,
    TranscriptInput,
)
from talkex.models.enums import Channel
from talkex.models.types import ConversationId

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SAMPLE_TRANSCRIPT = (
    "CUSTOMER: I need help with my broadband connection\n"
    "AGENT: Sure, let me look into that for you\n"
    "CUSTOMER: It has been down since yesterday"
)


def _make_transcript_input(**overrides: object) -> TranscriptInput:
    """Factory with sensible defaults."""
    defaults: dict[str, Any] = {
        "conversation_id": ConversationId("conv_abc123"),
        "channel": Channel.VOICE,
        "raw_text": _SAMPLE_TRANSCRIPT,
        "source_format": SourceFormat.LABELED,
    }
    defaults.update(overrides)
    return TranscriptInput(**defaults)


# ---------------------------------------------------------------------------
# SourceFormat enum
# ---------------------------------------------------------------------------


class TestSourceFormat:
    def test_has_expected_members(self) -> None:
        assert set(SourceFormat) == {SourceFormat.LABELED, SourceFormat.MULTILINE, SourceFormat.PLAIN}

    def test_string_values(self) -> None:
        assert str(SourceFormat.LABELED) == "labeled"
        assert str(SourceFormat.MULTILINE) == "multiline"
        assert str(SourceFormat.PLAIN) == "plain"

    def test_is_str_subclass(self) -> None:
        assert isinstance(SourceFormat.LABELED, str)


# ---------------------------------------------------------------------------
# SpeakerHint
# ---------------------------------------------------------------------------


class TestSpeakerHint:
    def test_creates_valid_hint(self) -> None:
        hint = SpeakerHint(speaker_label="Speaker 1", role="customer")
        assert hint.speaker_label == "Speaker 1"
        assert hint.role == "customer"

    def test_rejects_empty_speaker_label(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            SpeakerHint(speaker_label="", role="customer")

    def test_rejects_empty_role(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            SpeakerHint(speaker_label="Speaker 1", role="")

    def test_is_frozen(self) -> None:
        hint = SpeakerHint(speaker_label="Speaker 1", role="customer")
        with pytest.raises(ValueError, match="frozen"):
            hint.speaker_label = "Speaker 2"


# ---------------------------------------------------------------------------
# QualitySignals
# ---------------------------------------------------------------------------


class TestQualitySignals:
    def test_creates_with_no_signals(self) -> None:
        qs = QualitySignals()
        assert qs.asr_confidence is None
        assert qs.language_code is None
        assert qs.audio_duration_ms is None
        assert qs.word_error_rate is None

    def test_creates_with_all_signals(self) -> None:
        qs = QualitySignals(
            asr_confidence=0.92,
            language_code="pt",
            audio_duration_ms=45000,
            word_error_rate=0.08,
        )
        assert qs.asr_confidence == 0.92
        assert qs.language_code == "pt"
        assert qs.audio_duration_ms == 45000
        assert qs.word_error_rate == 0.08

    def test_rejects_asr_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="asr_confidence must be in"):
            QualitySignals(asr_confidence=1.5)

    def test_rejects_negative_asr_confidence(self) -> None:
        with pytest.raises(ValueError, match="asr_confidence must be in"):
            QualitySignals(asr_confidence=-0.1)

    def test_rejects_word_error_rate_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="word_error_rate must be in"):
            QualitySignals(word_error_rate=1.5)

    def test_rejects_non_positive_audio_duration(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            QualitySignals(audio_duration_ms=0)

    def test_is_frozen(self) -> None:
        qs = QualitySignals(asr_confidence=0.9)
        with pytest.raises(ValueError, match="frozen"):
            qs.asr_confidence = 0.5


# ---------------------------------------------------------------------------
# TranscriptInput — construction
# ---------------------------------------------------------------------------


class TestTranscriptInputConstruction:
    def test_creates_with_required_fields_only(self) -> None:
        ti = _make_transcript_input()
        assert ti.conversation_id == "conv_abc123"
        assert ti.channel == Channel.VOICE
        assert ti.raw_text == _SAMPLE_TRANSCRIPT
        assert ti.source_format == SourceFormat.LABELED
        assert ti.speaker_hints == []
        assert ti.quality_signals is None
        assert ti.metadata == {}

    def test_creates_with_all_fields(self) -> None:
        hints = [SpeakerHint(speaker_label="Speaker 1", role="customer")]
        signals = QualitySignals(asr_confidence=0.9, language_code="pt")
        ti = _make_transcript_input(
            speaker_hints=hints,
            quality_signals=signals,
            metadata={"source": "crm"},
        )
        assert len(ti.speaker_hints) == 1
        assert ti.quality_signals is not None
        assert ti.quality_signals.asr_confidence == 0.9
        assert ti.metadata == {"source": "crm"}

    def test_speaker_hints_defaults_to_empty_list(self) -> None:
        ti = _make_transcript_input()
        assert ti.speaker_hints == []
        assert isinstance(ti.speaker_hints, list)

    def test_metadata_defaults_to_empty_dict(self) -> None:
        ti = _make_transcript_input()
        assert ti.metadata == {}
        assert isinstance(ti.metadata, dict)


# ---------------------------------------------------------------------------
# TranscriptInput — validation
# ---------------------------------------------------------------------------


class TestTranscriptInputValidation:
    def test_rejects_empty_conversation_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_transcript_input(conversation_id=ConversationId(""))

    def test_rejects_whitespace_only_conversation_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_transcript_input(conversation_id=ConversationId("   "))

    def test_rejects_empty_raw_text(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_transcript_input(raw_text="")

    def test_rejects_whitespace_only_raw_text(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_transcript_input(raw_text="   ")

    def test_preserves_raw_text_without_normalizing(self) -> None:
        padded = "  hello world  "
        ti = _make_transcript_input(raw_text=padded)
        assert ti.raw_text == "  hello world  "


# ---------------------------------------------------------------------------
# TranscriptInput — strict mode
# ---------------------------------------------------------------------------


class TestTranscriptInputStrictMode:
    def test_rejects_string_channel_coercion(self) -> None:
        with pytest.raises(ValueError):
            _make_transcript_input(channel="voice")

    def test_rejects_string_source_format_coercion(self) -> None:
        with pytest.raises(ValueError):
            _make_transcript_input(source_format="labeled")

    def test_rejects_int_for_conversation_id(self) -> None:
        with pytest.raises(ValueError):
            _make_transcript_input(conversation_id=12345)


# ---------------------------------------------------------------------------
# TranscriptInput — immutability
# ---------------------------------------------------------------------------


class TestTranscriptInputImmutability:
    def test_cannot_assign_to_field(self) -> None:
        ti = _make_transcript_input()
        with pytest.raises(ValueError, match="frozen"):
            ti.raw_text = "new text"

    def test_cannot_assign_to_metadata(self) -> None:
        ti = _make_transcript_input()
        with pytest.raises(ValueError, match="frozen"):
            ti.metadata = {"new": "data"}


# ---------------------------------------------------------------------------
# TranscriptInput — serialization
# ---------------------------------------------------------------------------


class TestTranscriptInputSerializationInMemory:
    def test_model_dump_produces_dict(self) -> None:
        ti = _make_transcript_input()
        data = ti.model_dump()
        assert isinstance(data, dict)
        assert data["conversation_id"] == "conv_abc123"
        assert data["source_format"] == "labeled"

    def test_enums_serialize_as_values(self) -> None:
        ti = _make_transcript_input()
        data = ti.model_dump()
        assert data["channel"] == "voice"
        assert data["source_format"] == "labeled"


class TestTranscriptInputBoundaryDeserialization:
    def test_reconstructs_from_model_dump(self) -> None:
        hints = [SpeakerHint(speaker_label="S1", role="customer")]
        signals = QualitySignals(asr_confidence=0.9)
        ti = _make_transcript_input(
            speaker_hints=hints,
            quality_signals=signals,
            metadata={"key": "value"},
        )
        data = ti.model_dump()
        restored = TranscriptInput.model_validate(data, strict=False)
        assert restored == ti

    def test_reconstructs_from_json_mode_dump(self) -> None:
        ti = _make_transcript_input(metadata={"nested": {"deep": True}})
        json_data = ti.model_dump(mode="json")
        restored = TranscriptInput.model_validate(json_data, strict=False)
        assert restored == ti


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestIngestionReexport:
    def test_transcript_input_importable_from_ingestion(self) -> None:
        from talkex.ingestion import TranscriptInput as Imported

        assert Imported is TranscriptInput

    def test_source_format_importable_from_ingestion(self) -> None:
        from talkex.ingestion import SourceFormat as Imported

        assert Imported is SourceFormat

    def test_quality_signals_importable_from_ingestion(self) -> None:
        from talkex.ingestion import QualitySignals as Imported

        assert Imported is QualitySignals

    def test_speaker_hint_importable_from_ingestion(self) -> None:
        from talkex.ingestion import SpeakerHint as Imported

        assert Imported is SpeakerHint
