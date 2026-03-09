"""Unit tests for turn parsing.

Tests cover: labeled parsing (basic, multi-turn, no labels, speaker map,
unknown labels), multiline parsing, plain parsing, parse_transcript routing,
offset correctness, and RawTurn construction.
"""

import pytest

from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.models.enums import SpeakerRole
from semantic_conversation_engine.segmentation.config import SegmentationConfig
from semantic_conversation_engine.segmentation.parsing import (
    RawTurn,
    parse_transcript,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = SegmentationConfig()

_LABELED_TRANSCRIPT = (
    "CUSTOMER: I need help with my broadband\n"
    "AGENT: Sure, let me look into that\n"
    "CUSTOMER: It has been down since yesterday"
)

_MULTILINE_TRANSCRIPT = "First line\nSecond line\nThird line"


# ---------------------------------------------------------------------------
# Labeled format — basic parsing
# ---------------------------------------------------------------------------


class TestParseLabeledBasic:
    def test_parses_two_speakers(self) -> None:
        turns = parse_transcript(_LABELED_TRANSCRIPT, SourceFormat.LABELED, _DEFAULT_CONFIG)
        assert len(turns) == 3

    def test_assigns_correct_speakers(self) -> None:
        turns = parse_transcript(_LABELED_TRANSCRIPT, SourceFormat.LABELED, _DEFAULT_CONFIG)
        assert turns[0].speaker == SpeakerRole.CUSTOMER
        assert turns[1].speaker == SpeakerRole.AGENT
        assert turns[2].speaker == SpeakerRole.CUSTOMER

    def test_extracts_text_without_speaker_label(self) -> None:
        turns = parse_transcript(_LABELED_TRANSCRIPT, SourceFormat.LABELED, _DEFAULT_CONFIG)
        assert turns[0].text == "I need help with my broadband"
        assert turns[1].text == "Sure, let me look into that"

    def test_offsets_are_non_negative(self) -> None:
        turns = parse_transcript(_LABELED_TRANSCRIPT, SourceFormat.LABELED, _DEFAULT_CONFIG)
        for turn in turns:
            assert turn.start_offset >= 0
            assert turn.end_offset > turn.start_offset

    def test_offsets_partition_the_text(self) -> None:
        turns = parse_transcript(_LABELED_TRANSCRIPT, SourceFormat.LABELED, _DEFAULT_CONFIG)
        # First turn starts at 0
        assert turns[0].start_offset == 0
        # Each turn's start_offset equals the previous turn's end_offset
        for i in range(1, len(turns)):
            assert turns[i].start_offset == turns[i - 1].end_offset
        # Last turn's end_offset equals the text length
        assert turns[-1].end_offset == len(_LABELED_TRANSCRIPT)


class TestParseLabeledEdgeCases:
    def test_no_labels_returns_single_unknown_turn(self) -> None:
        turns = parse_transcript("just some plain text", SourceFormat.LABELED, _DEFAULT_CONFIG)
        assert len(turns) == 1
        assert turns[0].speaker == SpeakerRole.UNKNOWN
        assert turns[0].text == "just some plain text"

    def test_empty_text_between_labels_is_skipped(self) -> None:
        text = "CUSTOMER:\nAGENT: hello"
        turns = parse_transcript(text, SourceFormat.LABELED, _DEFAULT_CONFIG)
        # CUSTOMER turn has no text, should be skipped
        assert len(turns) == 1
        assert turns[0].speaker == SpeakerRole.AGENT

    def test_system_speaker_recognized(self) -> None:
        text = "SYSTEM: Welcome to the service"
        turns = parse_transcript(text, SourceFormat.LABELED, _DEFAULT_CONFIG)
        assert turns[0].speaker == SpeakerRole.SYSTEM

    def test_unknown_speaker_recognized(self) -> None:
        text = "UNKNOWN: Something was said"
        turns = parse_transcript(text, SourceFormat.LABELED, _DEFAULT_CONFIG)
        assert turns[0].speaker == SpeakerRole.UNKNOWN


# ---------------------------------------------------------------------------
# Labeled format — speaker map (from hints)
# ---------------------------------------------------------------------------


class TestParseLabeledSpeakerMap:
    def test_custom_label_mapped_via_speaker_map(self) -> None:
        config = SegmentationConfig(speaker_label_pattern=r"^(Speaker 1|Speaker 2)\s*:")
        text = "Speaker 1: Hello\nSpeaker 2: Hi there"
        speaker_map = {
            "Speaker 1": SpeakerRole.CUSTOMER,
            "Speaker 2": SpeakerRole.AGENT,
        }
        turns = parse_transcript(text, SourceFormat.LABELED, config, speaker_map)
        assert turns[0].speaker == SpeakerRole.CUSTOMER
        assert turns[1].speaker == SpeakerRole.AGENT

    def test_unrecognized_label_falls_back_to_unknown(self) -> None:
        config = SegmentationConfig(speaker_label_pattern=r"^(NARRATOR|CROWD)\s*:")
        text = "NARRATOR: Once upon a time"
        turns = parse_transcript(text, SourceFormat.LABELED, config)
        assert turns[0].speaker == SpeakerRole.UNKNOWN

    def test_speaker_map_case_insensitive(self) -> None:
        config = SegmentationConfig(speaker_label_pattern=r"^(speaker1|speaker2)\s*:")
        text = "speaker1: Hello"
        speaker_map = {"SPEAKER1": SpeakerRole.CUSTOMER}
        turns = parse_transcript(text, SourceFormat.LABELED, config, speaker_map)
        assert turns[0].speaker == SpeakerRole.CUSTOMER


# ---------------------------------------------------------------------------
# Multiline format
# ---------------------------------------------------------------------------


class TestParseMultiline:
    def test_each_line_becomes_a_turn(self) -> None:
        turns = parse_transcript(_MULTILINE_TRANSCRIPT, SourceFormat.MULTILINE, _DEFAULT_CONFIG)
        assert len(turns) == 3

    def test_all_speakers_are_unknown(self) -> None:
        turns = parse_transcript(_MULTILINE_TRANSCRIPT, SourceFormat.MULTILINE, _DEFAULT_CONFIG)
        assert all(t.speaker == SpeakerRole.UNKNOWN for t in turns)

    def test_empty_lines_are_skipped(self) -> None:
        text = "line1\n\n\nline2"
        turns = parse_transcript(text, SourceFormat.MULTILINE, _DEFAULT_CONFIG)
        assert len(turns) == 2
        assert turns[0].text == "line1"
        assert turns[1].text == "line2"

    def test_strips_whitespace_from_lines(self) -> None:
        text = "  hello  \n  world  "
        turns = parse_transcript(text, SourceFormat.MULTILINE, _DEFAULT_CONFIG)
        assert turns[0].text == "hello"
        assert turns[1].text == "world"

    def test_offsets_are_correct(self) -> None:
        text = "hello\nworld"
        turns = parse_transcript(text, SourceFormat.MULTILINE, _DEFAULT_CONFIG)
        assert turns[0].start_offset == 0
        assert turns[0].end_offset == 5
        assert turns[1].start_offset == 6
        assert turns[1].end_offset == 11


# ---------------------------------------------------------------------------
# Plain format
# ---------------------------------------------------------------------------


class TestParsePlain:
    def test_single_turn_with_unknown_speaker(self) -> None:
        turns = parse_transcript("Just some text here", SourceFormat.PLAIN, _DEFAULT_CONFIG)
        assert len(turns) == 1
        assert turns[0].speaker == SpeakerRole.UNKNOWN
        assert turns[0].text == "Just some text here"

    def test_strips_leading_trailing_whitespace(self) -> None:
        turns = parse_transcript("   hello world   ", SourceFormat.PLAIN, _DEFAULT_CONFIG)
        assert turns[0].text == "hello world"
        assert turns[0].start_offset == 3
        assert turns[0].end_offset == 14

    def test_whitespace_only_returns_empty(self) -> None:
        turns = parse_transcript("   \n   ", SourceFormat.PLAIN, _DEFAULT_CONFIG)
        assert turns == []


# ---------------------------------------------------------------------------
# parse_transcript routing
# ---------------------------------------------------------------------------


class TestParseTranscriptRouting:
    def test_routes_to_labeled(self) -> None:
        turns = parse_transcript("CUSTOMER: hello", SourceFormat.LABELED, _DEFAULT_CONFIG)
        assert turns[0].speaker == SpeakerRole.CUSTOMER

    def test_routes_to_multiline(self) -> None:
        turns = parse_transcript("line1\nline2", SourceFormat.MULTILINE, _DEFAULT_CONFIG)
        assert len(turns) == 2

    def test_routes_to_plain(self) -> None:
        turns = parse_transcript("hello", SourceFormat.PLAIN, _DEFAULT_CONFIG)
        assert len(turns) == 1
        assert turns[0].speaker == SpeakerRole.UNKNOWN

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Unknown source format"):
            parse_transcript("text", "invalid", _DEFAULT_CONFIG)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RawTurn construction
# ---------------------------------------------------------------------------


class TestRawTurn:
    def test_is_frozen(self) -> None:
        rt = RawTurn(speaker=SpeakerRole.CUSTOMER, text="hello", start_offset=0, end_offset=5)
        with pytest.raises(AttributeError):
            rt.text = "modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = RawTurn(speaker=SpeakerRole.CUSTOMER, text="hello", start_offset=0, end_offset=5)
        b = RawTurn(speaker=SpeakerRole.CUSTOMER, text="hello", start_offset=0, end_offset=5)
        assert a == b
