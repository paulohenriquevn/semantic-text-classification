"""Unit tests for sliding window generation.

Tests cover: basic windowing, stride behavior, partial tail handling,
min_window_size constraint, edge cases (empty, single turn, fewer
turns than window_size), and WindowSlice construction.
"""

from typing import Any

import pytest

from talkex.context.config import ContextWindowConfig
from talkex.context.windowing import (
    WindowSlice,
    generate_window_slices,
)
from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turn(index: int, speaker: SpeakerRole = SpeakerRole.CUSTOMER) -> Turn:
    """Create a minimal Turn for windowing tests."""
    return Turn(
        turn_id=TurnId(f"conv_t_turn_{index}"),
        conversation_id=ConversationId("conv_t"),
        speaker=speaker,
        raw_text=f"Turn {index} text",
        start_offset=index * 20,
        end_offset=index * 20 + 15,
    )


def _turns(n: int) -> list[Turn]:
    """Create n turns with alternating speakers."""
    speakers = [SpeakerRole.CUSTOMER, SpeakerRole.AGENT]
    return [_turn(i, speakers[i % 2]) for i in range(n)]


def _config(**overrides: Any) -> ContextWindowConfig:
    defaults: dict[str, Any] = {
        "window_size": 3,
        "stride": 2,
        "min_window_size": 1,
        "include_partial_tail": True,
    }
    defaults.update(overrides)
    return ContextWindowConfig(**defaults)


# ---------------------------------------------------------------------------
# Basic windowing
# ---------------------------------------------------------------------------


class TestBasicWindowing:
    def test_generates_correct_number_of_windows(self) -> None:
        # 6 turns, window=3, stride=2 → starts at 0,2,4 → 3 windows
        slices = generate_window_slices(_turns(6), _config())
        assert len(slices) == 3

    def test_first_window_starts_at_zero(self) -> None:
        slices = generate_window_slices(_turns(6), _config())
        assert slices[0].start_index == 0

    def test_window_contains_correct_turns(self) -> None:
        turns = _turns(6)
        slices = generate_window_slices(turns, _config())
        assert slices[0].turns == tuple(turns[0:3])
        assert slices[1].turns == tuple(turns[2:5])
        assert slices[2].turns == tuple(turns[4:6])

    def test_end_index_is_inclusive(self) -> None:
        slices = generate_window_slices(_turns(6), _config())
        assert slices[0].end_index == 2  # turns 0,1,2
        assert slices[1].end_index == 4  # turns 2,3,4


# ---------------------------------------------------------------------------
# Stride behavior
# ---------------------------------------------------------------------------


class TestStrideBehavior:
    def test_stride_one_produces_maximum_overlap(self) -> None:
        # 5 turns, window=3, stride=1 → starts at 0,1,2 → 3 full + possible partials
        slices = generate_window_slices(_turns(5), _config(stride=1))
        # starts: 0(3), 1(3), 2(3), 3(2), 4(1)
        assert len(slices) == 5

    def test_stride_equals_window_produces_no_overlap(self) -> None:
        # 6 turns, window=3, stride=3 → starts at 0,3 → 2 windows, no overlap
        slices = generate_window_slices(_turns(6), _config(stride=3))
        assert len(slices) == 2
        assert slices[0].start_index == 0
        assert slices[1].start_index == 3

    def test_stride_greater_than_window_creates_gaps(self) -> None:
        # 10 turns, window=3, stride=5 → starts at 0,5 → 2 full windows
        slices = generate_window_slices(_turns(10), _config(stride=5))
        assert len(slices) == 2
        assert slices[0].end_index == 2
        assert slices[1].start_index == 5


# ---------------------------------------------------------------------------
# Partial tail handling
# ---------------------------------------------------------------------------


class TestPartialTailHandling:
    def test_includes_partial_tail_when_enabled(self) -> None:
        # 5 turns, window=3, stride=2 → starts at 0,2,4
        # Window at 4: turns[4:5] = 1 turn (partial)
        slices = generate_window_slices(_turns(5), _config(include_partial_tail=True))
        assert len(slices) == 3
        assert len(slices[2].turns) == 1

    def test_excludes_partial_tail_when_disabled(self) -> None:
        # 5 turns, window=3, stride=2 → starts at 0,2
        # Window at 4 would be partial, excluded
        slices = generate_window_slices(_turns(5), _config(include_partial_tail=False))
        assert len(slices) == 2
        # Both windows are full size
        assert all(len(s.turns) == 3 for s in slices)

    def test_partial_tail_respects_min_window_size(self) -> None:
        # 7 turns, window=5, stride=3 → starts at 0,3,6
        # Window at 6: turns[6:7] = 1 turn, min_window_size=2 → skip
        slices = generate_window_slices(
            _turns(7),
            _config(
                window_size=5,
                stride=3,
                min_window_size=2,
                include_partial_tail=True,
            ),
        )
        # Window at 0: 5 turns, window at 3: 4 turns (partial but >= 2)
        # Window at 6: 1 turn < 2 → skipped
        assert len(slices) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWindowingEdgeCases:
    def test_empty_turns_returns_empty(self) -> None:
        assert generate_window_slices([], _config()) == []

    def test_single_turn(self) -> None:
        slices = generate_window_slices(_turns(1), _config())
        assert len(slices) == 1
        assert len(slices[0].turns) == 1
        assert slices[0].start_index == 0
        assert slices[0].end_index == 0

    def test_turns_equal_to_window_size_stride_gte_window(self) -> None:
        # 3 turns, window=3, stride=3 → exactly one full window
        slices = generate_window_slices(_turns(3), _config(stride=3))
        assert len(slices) == 1
        assert len(slices[0].turns) == 3

    def test_turns_equal_to_window_size_with_overlap(self) -> None:
        # 3 turns, window=3, stride=2 → 1 full + 1 partial
        slices = generate_window_slices(_turns(3), _config(stride=2))
        assert len(slices) == 2
        assert len(slices[0].turns) == 3
        assert len(slices[1].turns) == 1

    def test_fewer_turns_than_window_with_partial_tail(self) -> None:
        # 2 turns, window=5 → one partial window
        slices = generate_window_slices(_turns(2), _config(window_size=5, include_partial_tail=True))
        assert len(slices) == 1
        assert len(slices[0].turns) == 2

    def test_fewer_turns_than_window_without_partial_tail(self) -> None:
        # 2 turns, window=5 → no full windows possible
        slices = generate_window_slices(_turns(2), _config(window_size=5, include_partial_tail=False))
        assert slices == []

    def test_fewer_turns_than_min_window_size(self) -> None:
        # 1 turn, min_window_size=3 → too small even with partial tail
        slices = generate_window_slices(
            _turns(1),
            _config(window_size=5, min_window_size=3, include_partial_tail=True),
        )
        assert slices == []


# ---------------------------------------------------------------------------
# WindowSlice construction
# ---------------------------------------------------------------------------


class TestWindowSlice:
    def test_is_frozen(self) -> None:
        ws = WindowSlice(start_index=0, end_index=2, turns=tuple(_turns(3)))
        with pytest.raises(AttributeError):
            ws.start_index = 5  # type: ignore[misc]

    def test_equality(self) -> None:
        turns = tuple(_turns(3))
        a = WindowSlice(start_index=0, end_index=2, turns=turns)
        b = WindowSlice(start_index=0, end_index=2, turns=turns)
        assert a == b
