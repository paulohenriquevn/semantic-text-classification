"""Unit tests for window text rendering and role-aware view extraction.

Tests cover: speaker label rendering, delimiter handling, normalized_text
preference, role-aware extraction, empty role text, and determinism.
"""

from talkex.context.config import ContextWindowConfig
from talkex.context.rendering import (
    extract_role_text,
    render_window_text,
)
from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turn(
    index: int,
    speaker: SpeakerRole = SpeakerRole.CUSTOMER,
    raw: str = "raw text",
    normalized: str | None = "normalized text",
) -> Turn:
    return Turn(
        turn_id=TurnId(f"conv_r_turn_{index}"),
        conversation_id=ConversationId("conv_r"),
        speaker=speaker,
        raw_text=raw,
        start_offset=index * 10,
        end_offset=index * 10 + 8,
        normalized_text=normalized,
    )


# ---------------------------------------------------------------------------
# render_window_text — speaker labels
# ---------------------------------------------------------------------------


class TestRenderWindowTextLabels:
    def test_includes_speaker_labels_when_enabled(self) -> None:
        turns = (
            _turn(0, SpeakerRole.CUSTOMER, normalized="hello"),
            _turn(1, SpeakerRole.AGENT, normalized="hi there"),
        )
        config = ContextWindowConfig(render_speaker_labels=True)
        result = render_window_text(turns, config)
        assert result == "[CUSTOMER] hello\n[AGENT] hi there"

    def test_excludes_speaker_labels_when_disabled(self) -> None:
        turns = (
            _turn(0, SpeakerRole.CUSTOMER, normalized="hello"),
            _turn(1, SpeakerRole.AGENT, normalized="hi there"),
        )
        config = ContextWindowConfig(render_speaker_labels=False)
        result = render_window_text(turns, config)
        assert result == "hello\nhi there"

    def test_label_uses_uppercase_role_value(self) -> None:
        turns = (_turn(0, SpeakerRole.SYSTEM, normalized="welcome"),)
        config = ContextWindowConfig(render_speaker_labels=True)
        result = render_window_text(turns, config)
        assert result == "[SYSTEM] welcome"

    def test_unknown_speaker_label(self) -> None:
        turns = (_turn(0, SpeakerRole.UNKNOWN, normalized="something"),)
        config = ContextWindowConfig(render_speaker_labels=True)
        result = render_window_text(turns, config)
        assert result == "[UNKNOWN] something"


# ---------------------------------------------------------------------------
# render_window_text — delimiter
# ---------------------------------------------------------------------------


class TestRenderWindowTextDelimiter:
    def test_default_newline_delimiter(self) -> None:
        turns = (
            _turn(0, normalized="a"),
            _turn(1, normalized="b"),
        )
        config = ContextWindowConfig(render_speaker_labels=False)
        assert "\n" in render_window_text(turns, config)

    def test_custom_delimiter(self) -> None:
        turns = (
            _turn(0, normalized="a"),
            _turn(1, normalized="b"),
        )
        config = ContextWindowConfig(render_speaker_labels=False, render_turn_delimiter=" | ")
        result = render_window_text(turns, config)
        assert result == "a | b"

    def test_single_turn_no_delimiter(self) -> None:
        turns = (_turn(0, normalized="hello"),)
        config = ContextWindowConfig(render_speaker_labels=False)
        result = render_window_text(turns, config)
        assert result == "hello"


# ---------------------------------------------------------------------------
# render_window_text — text selection
# ---------------------------------------------------------------------------


class TestRenderWindowTextSelection:
    def test_prefers_normalized_text(self) -> None:
        turns = (_turn(0, raw="RAW", normalized="NORMALIZED"),)
        config = ContextWindowConfig(render_speaker_labels=False)
        result = render_window_text(turns, config)
        assert result == "NORMALIZED"

    def test_falls_back_to_raw_text(self) -> None:
        turns = (_turn(0, raw="RAW", normalized=None),)
        config = ContextWindowConfig(render_speaker_labels=False)
        result = render_window_text(turns, config)
        assert result == "RAW"


# ---------------------------------------------------------------------------
# extract_role_text
# ---------------------------------------------------------------------------


class TestExtractRoleText:
    def test_extracts_customer_text_only(self) -> None:
        turns = (
            _turn(0, SpeakerRole.CUSTOMER, normalized="customer says"),
            _turn(1, SpeakerRole.AGENT, normalized="agent says"),
            _turn(2, SpeakerRole.CUSTOMER, normalized="customer again"),
        )
        result = extract_role_text(turns, SpeakerRole.CUSTOMER, "\n")
        assert result == "customer says\ncustomer again"

    def test_extracts_agent_text_only(self) -> None:
        turns = (
            _turn(0, SpeakerRole.CUSTOMER, normalized="customer says"),
            _turn(1, SpeakerRole.AGENT, normalized="agent says"),
        )
        result = extract_role_text(turns, SpeakerRole.AGENT, "\n")
        assert result == "agent says"

    def test_returns_empty_when_no_matching_role(self) -> None:
        turns = (_turn(0, SpeakerRole.CUSTOMER, normalized="hello"),)
        result = extract_role_text(turns, SpeakerRole.AGENT, "\n")
        assert result == ""

    def test_uses_custom_delimiter(self) -> None:
        turns = (
            _turn(0, SpeakerRole.CUSTOMER, normalized="a"),
            _turn(1, SpeakerRole.CUSTOMER, normalized="b"),
        )
        result = extract_role_text(turns, SpeakerRole.CUSTOMER, " | ")
        assert result == "a | b"

    def test_no_speaker_labels_in_role_text(self) -> None:
        turns = (_turn(0, SpeakerRole.CUSTOMER, normalized="hello"),)
        result = extract_role_text(turns, SpeakerRole.CUSTOMER, "\n")
        assert "[CUSTOMER]" not in result
        assert result == "hello"

    def test_prefers_normalized_text(self) -> None:
        turns = (_turn(0, SpeakerRole.CUSTOMER, raw="RAW", normalized="NORM"),)
        result = extract_role_text(turns, SpeakerRole.CUSTOMER, "\n")
        assert result == "NORM"

    def test_falls_back_to_raw_text(self) -> None:
        turns = (_turn(0, SpeakerRole.CUSTOMER, raw="RAW", normalized=None),)
        result = extract_role_text(turns, SpeakerRole.CUSTOMER, "\n")
        assert result == "RAW"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestRenderingDeterminism:
    def test_same_input_produces_same_output(self) -> None:
        turns = (
            _turn(0, SpeakerRole.CUSTOMER, normalized="hello"),
            _turn(1, SpeakerRole.AGENT, normalized="hi"),
        )
        config = ContextWindowConfig()
        a = render_window_text(turns, config)
        b = render_window_text(turns, config)
        assert a == b
