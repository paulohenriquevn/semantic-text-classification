"""Unit tests for TurnSegmenter — end-to-end segmentation pipeline.

Tests cover: determinism, no empty turns, end_offset > start_offset,
configurable merge, UNKNOWN fallback, lexical features in metadata,
speaker hints, format routing, and reexport.
"""

from typing import Any

from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import (
    SpeakerHint,
    TranscriptInput,
)
from talkex.models.enums import Channel, SpeakerRole
from talkex.models.types import ConversationId
from talkex.segmentation.config import SegmentationConfig
from talkex.segmentation.segmenter import TurnSegmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABELED_TRANSCRIPT = (
    "CUSTOMER: I need help with my broadband connection\n"
    "AGENT: Sure, let me look into that for you\n"
    "CUSTOMER: It has been down since yesterday"
)

_MULTILINE_TRANSCRIPT = "First line\nSecond line\nThird line"


def _make_input(**overrides: Any) -> TranscriptInput:
    defaults: dict[str, Any] = {
        "conversation_id": ConversationId("conv_test123"),
        "channel": Channel.VOICE,
        "raw_text": _LABELED_TRANSCRIPT,
        "source_format": SourceFormat.LABELED,
    }
    defaults.update(overrides)
    return TranscriptInput(**defaults)


def _segmenter() -> TurnSegmenter:
    return TurnSegmenter()


# ---------------------------------------------------------------------------
# Basic segmentation
# ---------------------------------------------------------------------------


class TestBasicSegmentation:
    def test_labeled_transcript_produces_correct_turn_count(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        assert len(turns) == 3

    def test_assigns_correct_speakers(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        assert turns[0].speaker == SpeakerRole.CUSTOMER
        assert turns[1].speaker == SpeakerRole.AGENT
        assert turns[2].speaker == SpeakerRole.CUSTOMER

    def test_extracts_text_content(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        assert "broadband" in turns[0].raw_text
        assert "look into" in turns[1].raw_text

    def test_all_turns_have_conversation_id(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        for turn in turns:
            assert turn.conversation_id == "conv_test123"

    def test_turn_ids_follow_deterministic_pattern(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        for i, turn in enumerate(turns):
            assert turn.turn_id == f"conv_test123_turn_{i}"

    def test_all_turn_ids_are_unique(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        ids = [turn.turn_id for turn in turns]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Full determinism (same input → byte-identical output)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_produces_identical_output(self) -> None:
        """Same input + same config → byte-identical turns, including IDs."""
        inp = _make_input()
        config = SegmentationConfig()
        turns_a = _segmenter().segment(inp, config)
        turns_b = _segmenter().segment(inp, config)

        assert len(turns_a) == len(turns_b)
        for a, b in zip(turns_a, turns_b, strict=True):
            assert a.turn_id == b.turn_id
            assert a.speaker == b.speaker
            assert a.raw_text == b.raw_text
            assert a.start_offset == b.start_offset
            assert a.end_offset == b.end_offset
            assert a.normalized_text == b.normalized_text
            assert a.metadata == b.metadata

    def test_turn_ids_are_reproducible_across_instances(self) -> None:
        """Different TurnSegmenter instances produce identical IDs."""
        inp = _make_input()
        config = SegmentationConfig()
        turns_a = TurnSegmenter().segment(inp, config)
        turns_b = TurnSegmenter().segment(inp, config)
        assert [t.turn_id for t in turns_a] == [t.turn_id for t in turns_b]


# ---------------------------------------------------------------------------
# No empty turns
# ---------------------------------------------------------------------------


class TestNoEmptyTurns:
    def test_no_turn_has_empty_raw_text(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        for turn in turns:
            assert turn.raw_text.strip() != ""

    def test_no_turn_has_empty_normalized_text(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        for turn in turns:
            assert turn.normalized_text is not None
            assert turn.normalized_text.strip() != ""


# ---------------------------------------------------------------------------
# end_offset > start_offset
# ---------------------------------------------------------------------------


class TestOffsetInvariant:
    def test_all_turns_have_end_offset_greater_than_start(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        for turn in turns:
            assert turn.end_offset > turn.start_offset

    def test_multiline_offsets_valid(self) -> None:
        inp = _make_input(raw_text=_MULTILINE_TRANSCRIPT, source_format=SourceFormat.MULTILINE)
        turns = _segmenter().segment(inp, SegmentationConfig())
        for turn in turns:
            assert turn.end_offset > turn.start_offset

    def test_plain_offsets_valid(self) -> None:
        inp = _make_input(raw_text="Hello world", source_format=SourceFormat.PLAIN)
        turns = _segmenter().segment(inp, SegmentationConfig())
        for turn in turns:
            assert turn.end_offset > turn.start_offset


# ---------------------------------------------------------------------------
# Configurable merge
# ---------------------------------------------------------------------------


class TestConfigurableMerge:
    def test_merge_enabled_combines_consecutive_same_speaker(self) -> None:
        text = "CUSTOMER: hello\nCUSTOMER: world\nAGENT: hi"
        inp = _make_input(raw_text=text)
        config = SegmentationConfig(merge_consecutive_same_speaker=True)
        turns = _segmenter().segment(inp, config)
        assert len(turns) == 2
        assert turns[0].speaker == SpeakerRole.CUSTOMER
        assert turns[1].speaker == SpeakerRole.AGENT

    def test_merge_disabled_preserves_all_turns(self) -> None:
        text = "CUSTOMER: hello\nCUSTOMER: world\nAGENT: hi"
        inp = _make_input(raw_text=text)
        config = SegmentationConfig(merge_consecutive_same_speaker=False)
        turns = _segmenter().segment(inp, config)
        assert len(turns) == 3

    def test_merged_turn_contains_both_texts(self) -> None:
        text = "CUSTOMER: hello\nCUSTOMER: world"
        inp = _make_input(raw_text=text)
        config = SegmentationConfig(merge_consecutive_same_speaker=True)
        turns = _segmenter().segment(inp, config)
        assert "hello" in turns[0].raw_text
        assert "world" in turns[0].raw_text


# ---------------------------------------------------------------------------
# UNKNOWN fallback
# ---------------------------------------------------------------------------


class TestUnknownFallback:
    def test_multiline_all_unknown(self) -> None:
        inp = _make_input(raw_text=_MULTILINE_TRANSCRIPT, source_format=SourceFormat.MULTILINE)
        turns = _segmenter().segment(inp, SegmentationConfig())
        assert all(t.speaker == SpeakerRole.UNKNOWN for t in turns)

    def test_plain_is_unknown(self) -> None:
        inp = _make_input(raw_text="Some text", source_format=SourceFormat.PLAIN)
        turns = _segmenter().segment(inp, SegmentationConfig())
        assert turns[0].speaker == SpeakerRole.UNKNOWN

    def test_unrecognized_label_is_unknown(self) -> None:
        config = SegmentationConfig(speaker_label_pattern=r"^(NARRATOR)\s*:")
        inp = _make_input(raw_text="NARRATOR: Once upon a time")
        turns = _segmenter().segment(inp, config)
        assert turns[0].speaker == SpeakerRole.UNKNOWN


# ---------------------------------------------------------------------------
# Lexical features in metadata
# ---------------------------------------------------------------------------


class TestLexicalFeatures:
    def test_metadata_contains_expected_keys(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        expected_keys = {
            "char_count",
            "word_count",
            "has_question",
            "line_count",
            "avg_word_length",
        }
        for turn in turns:
            assert set(turn.metadata.keys()) == expected_keys

    def test_word_count_is_positive(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        for turn in turns:
            assert turn.metadata["word_count"] > 0

    def test_char_count_matches_normalized_text(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        for turn in turns:
            assert turn.normalized_text is not None
            assert turn.metadata["char_count"] == len(turn.normalized_text)


# ---------------------------------------------------------------------------
# Speaker hints
# ---------------------------------------------------------------------------


class TestSpeakerHints:
    def test_hints_remap_custom_labels(self) -> None:
        config = SegmentationConfig(speaker_label_pattern=r"^(Speaker 1|Speaker 2)\s*:")
        text = "Speaker 1: Hello\nSpeaker 2: Hi there"
        hints = [
            SpeakerHint(speaker_label="Speaker 1", role="customer"),
            SpeakerHint(speaker_label="Speaker 2", role="agent"),
        ]
        inp = _make_input(raw_text=text, source_format=SourceFormat.LABELED, speaker_hints=hints)
        turns = _segmenter().segment(inp, config)
        assert turns[0].speaker == SpeakerRole.CUSTOMER
        assert turns[1].speaker == SpeakerRole.AGENT

    def test_unrecognized_hint_role_falls_back_to_unknown(self) -> None:
        config = SegmentationConfig(speaker_label_pattern=r"^(Speaker 1)\s*:")
        hints = [SpeakerHint(speaker_label="Speaker 1", role="narrator")]
        inp = _make_input(
            raw_text="Speaker 1: Hello",
            source_format=SourceFormat.LABELED,
            speaker_hints=hints,
        )
        turns = _segmenter().segment(inp, config)
        assert turns[0].speaker == SpeakerRole.UNKNOWN


# ---------------------------------------------------------------------------
# min_turn_chars filtering
# ---------------------------------------------------------------------------


class TestMinTurnCharsFiltering:
    def test_filters_short_turns_after_normalization(self) -> None:
        text = "CUSTOMER: hi\nAGENT: I can certainly help you with that"
        inp = _make_input(raw_text=text)
        config = SegmentationConfig(min_turn_chars=5)
        turns = _segmenter().segment(inp, config)
        # "hi" is 2 chars, should be filtered out
        assert len(turns) == 1
        assert turns[0].speaker == SpeakerRole.AGENT

    def test_keeps_turns_at_exact_threshold(self) -> None:
        text = "CUSTOMER: hello\nAGENT: world"
        inp = _make_input(raw_text=text)
        config = SegmentationConfig(min_turn_chars=5)
        turns = _segmenter().segment(inp, config)
        assert len(turns) == 2


# ---------------------------------------------------------------------------
# Normalization applied
# ---------------------------------------------------------------------------


class TestNormalizationApplied:
    def test_normalized_text_is_populated(self) -> None:
        turns = _segmenter().segment(_make_input(), SegmentationConfig())
        for turn in turns:
            assert turn.normalized_text is not None

    def test_unicode_normalization_applied_to_normalized_text(self) -> None:
        # Use fullwidth characters in the text
        text = "CUSTOMER: \uff28ello"
        inp = _make_input(raw_text=text)
        config = SegmentationConfig(normalize_unicode=True)
        turns = _segmenter().segment(inp, config)
        assert turns[0].normalized_text is not None
        assert "H" in turns[0].normalized_text  # fullwidth H -> H

    def test_nfkc_preserves_raw_text_intact(self) -> None:
        """NFKC normalization ONLY affects normalized_text.

        raw_text must be preserved exactly as parsed from the input,
        without any Unicode normalization. This is critical for
        auditability — the original text is always recoverable.
        """
        fullwidth_text = "\uff28\uff45\uff4c\uff4c\uff4f"  # fullwidth "Hello"
        text = f"CUSTOMER: {fullwidth_text}"
        inp = _make_input(raw_text=text)
        config = SegmentationConfig(normalize_unicode=True)
        turns = _segmenter().segment(inp, config)

        # raw_text preserves the original fullwidth characters
        assert fullwidth_text in turns[0].raw_text

        # normalized_text has NFKC-normalized ASCII characters
        assert turns[0].normalized_text is not None
        assert "Hello" in turns[0].normalized_text
        assert fullwidth_text not in turns[0].normalized_text


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestSegmenterReexport:
    def test_importable_from_segmentation_package(self) -> None:
        from talkex.segmentation import TurnSegmenter as Imported

        assert Imported is TurnSegmenter
