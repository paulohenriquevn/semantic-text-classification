"""Unit tests for merge and filter heuristics.

Tests cover: consecutive same-speaker merging, alternating speakers,
triple merge, text concatenation, offset extension, short turn filtering,
and edge cases (empty input, single turn).
"""

from talkex.models.enums import SpeakerRole
from talkex.segmentation.merging import (
    filter_short_turns,
    merge_consecutive_same_speaker,
)
from talkex.segmentation.parsing import RawTurn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rt(speaker: SpeakerRole, text: str, start: int, end: int) -> RawTurn:
    return RawTurn(speaker=speaker, text=text, start_offset=start, end_offset=end)


# ---------------------------------------------------------------------------
# merge_consecutive_same_speaker
# ---------------------------------------------------------------------------


class TestMergeConsecutiveSameSpeaker:
    def test_empty_input_returns_empty(self) -> None:
        assert merge_consecutive_same_speaker([]) == []

    def test_single_turn_returns_unchanged(self) -> None:
        turns = [_rt(SpeakerRole.CUSTOMER, "hello", 0, 5)]
        result = merge_consecutive_same_speaker(turns)
        assert len(result) == 1
        assert result[0] == turns[0]

    def test_alternating_speakers_no_merge(self) -> None:
        turns = [
            _rt(SpeakerRole.CUSTOMER, "hello", 0, 5),
            _rt(SpeakerRole.AGENT, "hi", 5, 7),
            _rt(SpeakerRole.CUSTOMER, "thanks", 7, 13),
        ]
        result = merge_consecutive_same_speaker(turns)
        assert len(result) == 3

    def test_consecutive_same_speaker_merges(self) -> None:
        turns = [
            _rt(SpeakerRole.CUSTOMER, "hello", 0, 5),
            _rt(SpeakerRole.CUSTOMER, "world", 5, 10),
        ]
        result = merge_consecutive_same_speaker(turns)
        assert len(result) == 1
        assert result[0].speaker == SpeakerRole.CUSTOMER

    def test_merged_text_joined_with_newline(self) -> None:
        turns = [
            _rt(SpeakerRole.CUSTOMER, "hello", 0, 5),
            _rt(SpeakerRole.CUSTOMER, "world", 5, 10),
        ]
        result = merge_consecutive_same_speaker(turns)
        assert result[0].text == "hello\nworld"

    def test_merged_offsets_span_all_segments(self) -> None:
        turns = [
            _rt(SpeakerRole.CUSTOMER, "hello", 0, 5),
            _rt(SpeakerRole.CUSTOMER, "world", 6, 11),
        ]
        result = merge_consecutive_same_speaker(turns)
        assert result[0].start_offset == 0
        assert result[0].end_offset == 11

    def test_triple_consecutive_merges_into_one(self) -> None:
        turns = [
            _rt(SpeakerRole.AGENT, "a", 0, 1),
            _rt(SpeakerRole.AGENT, "b", 1, 2),
            _rt(SpeakerRole.AGENT, "c", 2, 3),
        ]
        result = merge_consecutive_same_speaker(turns)
        assert len(result) == 1
        assert result[0].text == "a\nb\nc"
        assert result[0].start_offset == 0
        assert result[0].end_offset == 3

    def test_mixed_pattern_merges_correctly(self) -> None:
        turns = [
            _rt(SpeakerRole.CUSTOMER, "a", 0, 1),
            _rt(SpeakerRole.CUSTOMER, "b", 1, 2),
            _rt(SpeakerRole.AGENT, "c", 2, 3),
            _rt(SpeakerRole.AGENT, "d", 3, 4),
            _rt(SpeakerRole.CUSTOMER, "e", 4, 5),
        ]
        result = merge_consecutive_same_speaker(turns)
        assert len(result) == 3
        assert result[0].text == "a\nb"
        assert result[1].text == "c\nd"
        assert result[2].text == "e"


# ---------------------------------------------------------------------------
# filter_short_turns
# ---------------------------------------------------------------------------


class TestFilterShortTurns:
    def test_empty_input_returns_empty(self) -> None:
        assert filter_short_turns([], min_chars=5) == []

    def test_removes_turns_below_threshold(self) -> None:
        turns = [
            _rt(SpeakerRole.CUSTOMER, "hi", 0, 2),
            _rt(SpeakerRole.AGENT, "hello world", 2, 13),
        ]
        result = filter_short_turns(turns, min_chars=5)
        assert len(result) == 1
        assert result[0].text == "hello world"

    def test_keeps_turns_at_exact_threshold(self) -> None:
        turns = [_rt(SpeakerRole.CUSTOMER, "hello", 0, 5)]
        result = filter_short_turns(turns, min_chars=5)
        assert len(result) == 1

    def test_removes_all_when_none_meet_threshold(self) -> None:
        turns = [
            _rt(SpeakerRole.CUSTOMER, "a", 0, 1),
            _rt(SpeakerRole.AGENT, "bb", 1, 3),
        ]
        result = filter_short_turns(turns, min_chars=10)
        assert result == []

    def test_preserves_order(self) -> None:
        turns = [
            _rt(SpeakerRole.CUSTOMER, "hello world", 0, 11),
            _rt(SpeakerRole.AGENT, "x", 11, 12),
            _rt(SpeakerRole.CUSTOMER, "goodbye world", 12, 25),
        ]
        result = filter_short_turns(turns, min_chars=5)
        assert len(result) == 2
        assert result[0].text == "hello world"
        assert result[1].text == "goodbye world"
