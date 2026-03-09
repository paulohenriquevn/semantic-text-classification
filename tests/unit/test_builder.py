"""Unit tests for SlidingWindowBuilder — end-to-end context window construction.

Tests cover: determinism, window_size/stride/tail behavior, rendering
stability, customer_text/agent_text correctness, operational metrics,
ContextWindow invariant compliance, and reexport.
"""

from datetime import UTC, datetime
from typing import Any

from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.models.conversation import Conversation
from semantic_conversation_engine.models.enums import Channel, SpeakerRole
from semantic_conversation_engine.models.turn import Turn
from semantic_conversation_engine.models.types import ConversationId, TurnId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONV = Conversation(
    conversation_id=ConversationId("conv_b"),
    channel=Channel.VOICE,
    start_time=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
)


def _turn(
    index: int,
    speaker: SpeakerRole = SpeakerRole.CUSTOMER,
    text: str | None = None,
) -> Turn:
    return Turn(
        turn_id=TurnId(f"conv_b_turn_{index}"),
        conversation_id=ConversationId("conv_b"),
        speaker=speaker,
        raw_text=text or f"Raw turn {index}",
        start_offset=index * 20,
        end_offset=index * 20 + 15,
        normalized_text=text or f"Turn {index} text",
    )


def _turns(n: int) -> list[Turn]:
    speakers = [SpeakerRole.CUSTOMER, SpeakerRole.AGENT]
    return [_turn(i, speakers[i % 2]) for i in range(n)]


def _config(**overrides: Any) -> ContextWindowConfig:
    defaults: dict[str, Any] = {
        "window_size": 3,
        "stride": 2,
        "min_window_size": 1,
        "include_partial_tail": True,
        "render_speaker_labels": True,
        "render_turn_delimiter": "\n",
    }
    defaults.update(overrides)
    return ContextWindowConfig(**defaults)


def _builder() -> SlidingWindowBuilder:
    return SlidingWindowBuilder()


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestBasicConstruction:
    def test_produces_correct_window_count(self) -> None:
        # 6 turns, window=3, stride=2 → 3 windows
        windows = _builder().build(_CONV, _turns(6), _config())
        assert len(windows) == 3

    def test_window_ids_are_deterministic(self) -> None:
        windows = _builder().build(_CONV, _turns(6), _config())
        assert windows[0].window_id == "conv_b_win_0"
        assert windows[1].window_id == "conv_b_win_1"
        assert windows[2].window_id == "conv_b_win_2"

    def test_conversation_id_propagated(self) -> None:
        windows = _builder().build(_CONV, _turns(4), _config())
        for w in windows:
            assert w.conversation_id == "conv_b"

    def test_turn_ids_match_input_turns(self) -> None:
        turns = _turns(6)
        windows = _builder().build(_CONV, turns, _config())
        # First window: turns 0,1,2
        assert windows[0].turn_ids == [t.turn_id for t in turns[0:3]]

    def test_empty_turns_produces_empty(self) -> None:
        windows = _builder().build(_CONV, [], _config())
        assert windows == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestBuilderDeterminism:
    def test_same_input_produces_identical_output(self) -> None:
        turns = _turns(6)
        config = _config()
        a = _builder().build(_CONV, turns, config)
        b = _builder().build(_CONV, turns, config)

        assert len(a) == len(b)
        for wa, wb in zip(a, b, strict=True):
            assert wa.window_id == wb.window_id
            assert wa.turn_ids == wb.turn_ids
            assert wa.window_text == wb.window_text
            assert wa.start_index == wb.start_index
            assert wa.end_index == wb.end_index
            assert wa.metadata == wb.metadata

    def test_different_builder_instances_same_output(self) -> None:
        turns = _turns(4)
        config = _config()
        a = SlidingWindowBuilder().build(_CONV, turns, config)
        b = SlidingWindowBuilder().build(_CONV, turns, config)
        assert [w.window_id for w in a] == [w.window_id for w in b]


# ---------------------------------------------------------------------------
# window_size, stride, tail behavior
# ---------------------------------------------------------------------------


class TestWindowSizeStrideTail:
    def test_window_size_matches_turn_count(self) -> None:
        windows = _builder().build(_CONV, _turns(6), _config())
        assert windows[0].window_size == 3
        assert windows[1].window_size == 3

    def test_partial_tail_has_fewer_turns(self) -> None:
        # 5 turns, window=3, stride=2 → window at 4 has 1 turn
        windows = _builder().build(_CONV, _turns(5), _config())
        assert windows[-1].window_size == 1

    def test_no_partial_tail_when_disabled(self) -> None:
        windows = _builder().build(_CONV, _turns(5), _config(include_partial_tail=False))
        assert all(w.window_size == 3 for w in windows)

    def test_stride_recorded_in_window(self) -> None:
        windows = _builder().build(_CONV, _turns(4), _config(stride=2))
        for w in windows:
            assert w.stride == 2

    def test_start_end_indices_correct(self) -> None:
        windows = _builder().build(_CONV, _turns(6), _config())
        assert windows[0].start_index == 0
        assert windows[0].end_index == 2
        assert windows[1].start_index == 2
        assert windows[1].end_index == 4


# ---------------------------------------------------------------------------
# Rendering stability
# ---------------------------------------------------------------------------


class TestRenderingStability:
    def test_window_text_includes_speaker_labels(self) -> None:
        turns = [_turn(0, SpeakerRole.CUSTOMER, "hello")]
        windows = _builder().build(_CONV, turns, _config(render_speaker_labels=True))
        assert "[CUSTOMER]" in windows[0].window_text

    def test_window_text_excludes_labels_when_disabled(self) -> None:
        turns = [_turn(0, SpeakerRole.CUSTOMER, "hello")]
        windows = _builder().build(_CONV, turns, _config(render_speaker_labels=False))
        assert "[CUSTOMER]" not in windows[0].window_text
        assert "hello" in windows[0].window_text

    def test_window_text_uses_custom_delimiter(self) -> None:
        turns = [
            _turn(0, SpeakerRole.CUSTOMER, "a"),
            _turn(1, SpeakerRole.AGENT, "b"),
        ]
        windows = _builder().build(
            _CONV,
            turns,
            _config(render_speaker_labels=False, render_turn_delimiter=" | "),
        )
        assert windows[0].window_text == "a | b"

    def test_window_text_prefers_normalized_text(self) -> None:
        turn = Turn(
            turn_id=TurnId("conv_b_turn_0"),
            conversation_id=ConversationId("conv_b"),
            speaker=SpeakerRole.CUSTOMER,
            raw_text="RAW",
            start_offset=0,
            end_offset=3,
            normalized_text="NORMALIZED",
        )
        windows = _builder().build(_CONV, [turn], _config(render_speaker_labels=False))
        assert windows[0].window_text == "NORMALIZED"


# ---------------------------------------------------------------------------
# customer_text and agent_text
# ---------------------------------------------------------------------------


class TestRoleAwareText:
    def test_customer_text_correct(self) -> None:
        turns = [
            _turn(0, SpeakerRole.CUSTOMER, "customer hello"),
            _turn(1, SpeakerRole.AGENT, "agent hello"),
            _turn(2, SpeakerRole.CUSTOMER, "customer bye"),
        ]
        windows = _builder().build(_CONV, turns, _config())
        assert windows[0].metadata["role_views"]["customer_text"] == "customer hello\ncustomer bye"

    def test_agent_text_correct(self) -> None:
        turns = [
            _turn(0, SpeakerRole.CUSTOMER, "customer hello"),
            _turn(1, SpeakerRole.AGENT, "agent hello"),
        ]
        windows = _builder().build(_CONV, turns, _config())
        assert windows[0].metadata["role_views"]["agent_text"] == "agent hello"

    def test_customer_text_empty_when_no_customer(self) -> None:
        turns = [_turn(0, SpeakerRole.AGENT, "only agent")]
        windows = _builder().build(_CONV, turns, _config())
        assert windows[0].metadata["role_views"]["customer_text"] == ""

    def test_agent_text_empty_when_no_agent(self) -> None:
        turns = [_turn(0, SpeakerRole.CUSTOMER, "only customer")]
        windows = _builder().build(_CONV, turns, _config())
        assert windows[0].metadata["role_views"]["agent_text"] == ""

    def test_role_text_no_speaker_labels(self) -> None:
        turns = [_turn(0, SpeakerRole.CUSTOMER, "hello")]
        windows = _builder().build(_CONV, turns, _config(render_speaker_labels=True))
        # Role text should NOT have labels even if window_text does
        assert "[CUSTOMER]" not in windows[0].metadata["role_views"]["customer_text"]
        assert windows[0].metadata["role_views"]["customer_text"] == "hello"


# ---------------------------------------------------------------------------
# Operational metrics
# ---------------------------------------------------------------------------


class TestOperationalMetrics:
    def test_total_chars_matches_window_text(self) -> None:
        turns = _turns(3)
        windows = _builder().build(_CONV, turns, _config())
        for w in windows:
            assert w.metadata["total_chars"] == len(w.window_text)

    def test_speaker_distribution_present(self) -> None:
        windows = _builder().build(_CONV, _turns(4), _config())
        assert "speakers" in windows[0].metadata
        dist = windows[0].metadata["speakers"]["distribution"]
        assert isinstance(dist, dict)

    def test_has_customer_and_has_agent(self) -> None:
        turns = [
            _turn(0, SpeakerRole.CUSTOMER, "c"),
            _turn(1, SpeakerRole.AGENT, "a"),
        ]
        windows = _builder().build(_CONV, turns, _config())
        assert windows[0].metadata["speakers"]["has_customer"] is True
        assert windows[0].metadata["speakers"]["has_agent"] is True

    def test_metadata_has_all_expected_keys(self) -> None:
        windows = _builder().build(_CONV, _turns(3), _config())
        expected = {
            "total_chars",
            "total_words",
            "role_views",
            "speakers",
        }
        assert set(windows[0].metadata.keys()) == expected


# ---------------------------------------------------------------------------
# ContextWindow invariant compliance
# ---------------------------------------------------------------------------


class TestContextWindowInvariants:
    def test_window_size_equals_len_turn_ids(self) -> None:
        windows = _builder().build(_CONV, _turns(6), _config())
        for w in windows:
            assert w.window_size == len(w.turn_ids)

    def test_end_index_gte_start_index(self) -> None:
        windows = _builder().build(_CONV, _turns(6), _config())
        for w in windows:
            assert w.end_index >= w.start_index

    def test_window_text_not_empty(self) -> None:
        windows = _builder().build(_CONV, _turns(4), _config())
        for w in windows:
            assert w.window_text.strip() != ""

    def test_turn_ids_not_empty(self) -> None:
        windows = _builder().build(_CONV, _turns(4), _config())
        for w in windows:
            assert len(w.turn_ids) > 0

    def test_stride_positive(self) -> None:
        windows = _builder().build(_CONV, _turns(4), _config())
        for w in windows:
            assert w.stride >= 1


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestBuilderReexport:
    def test_importable_from_context_package(self) -> None:
        from semantic_conversation_engine.context import (
            SlidingWindowBuilder as Imported,
        )

        assert Imported is SlidingWindowBuilder
