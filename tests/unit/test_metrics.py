"""Unit tests for window operational metrics.

Tests cover: total_chars, total_words, speaker distribution (namespaced),
has_customer/has_agent, customer/agent turn counts, role texts,
edge cases (single speaker, all unknown), and build-level coverage metrics.
"""

from semantic_conversation_engine.context.metrics import (
    compute_build_coverage,
    compute_window_metrics,
)
from semantic_conversation_engine.models.context_window import ContextWindow
from semantic_conversation_engine.models.enums import SpeakerRole
from semantic_conversation_engine.models.turn import Turn
from semantic_conversation_engine.models.types import ConversationId, TurnId, WindowId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turn(index: int, speaker: SpeakerRole = SpeakerRole.CUSTOMER) -> Turn:
    return Turn(
        turn_id=TurnId(f"conv_m_turn_{index}"),
        conversation_id=ConversationId("conv_m"),
        speaker=speaker,
        raw_text=f"Turn {index}",
        start_offset=index * 10,
        end_offset=index * 10 + 6,
        normalized_text=f"turn {index}",
    )


def _window(
    index: int,
    start: int,
    end: int,
) -> ContextWindow:
    """Create a minimal ContextWindow for coverage tests."""
    return ContextWindow(
        window_id=WindowId(f"conv_m_win_{index}"),
        conversation_id=ConversationId("conv_m"),
        turn_ids=[TurnId(f"conv_m_turn_{i}") for i in range(start, end + 1)],
        window_text="text",
        start_index=start,
        end_index=end,
        window_size=end - start + 1,
        stride=2,
    )


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------


class TestBasicMetrics:
    def test_total_chars(self) -> None:
        m = compute_window_metrics(
            turns=(_turn(0),),
            window_text="hello world",
            customer_text="hello world",
            agent_text="",
        )
        assert m["total_chars"] == 11

    def test_total_words(self) -> None:
        m = compute_window_metrics(
            turns=(_turn(0),),
            window_text="hello world foo",
            customer_text="",
            agent_text="",
        )
        assert m["total_words"] == 3


# ---------------------------------------------------------------------------
# Speaker distribution (namespaced under "speakers")
# ---------------------------------------------------------------------------


class TestSpeakerDistribution:
    def test_counts_per_role(self) -> None:
        turns = (
            _turn(0, SpeakerRole.CUSTOMER),
            _turn(1, SpeakerRole.AGENT),
            _turn(2, SpeakerRole.CUSTOMER),
        )
        m = compute_window_metrics(turns, "text", "", "")
        assert m["speakers"]["distribution"] == {"customer": 2, "agent": 1}

    def test_single_speaker(self) -> None:
        turns = (_turn(0, SpeakerRole.AGENT), _turn(1, SpeakerRole.AGENT))
        m = compute_window_metrics(turns, "text", "", "")
        assert m["speakers"]["distribution"] == {"agent": 2}

    def test_includes_unknown_speaker(self) -> None:
        turns = (_turn(0, SpeakerRole.UNKNOWN),)
        m = compute_window_metrics(turns, "text", "", "")
        assert m["speakers"]["distribution"] == {"unknown": 1}


# ---------------------------------------------------------------------------
# Boolean flags and counts (namespaced under "speakers")
# ---------------------------------------------------------------------------


class TestBooleanFlagsAndCounts:
    def test_has_customer_true(self) -> None:
        turns = (_turn(0, SpeakerRole.CUSTOMER),)
        m = compute_window_metrics(turns, "text", "text", "")
        assert m["speakers"]["has_customer"] is True
        assert m["speakers"]["customer_turn_count"] == 1

    def test_has_customer_false(self) -> None:
        turns = (_turn(0, SpeakerRole.AGENT),)
        m = compute_window_metrics(turns, "text", "", "text")
        assert m["speakers"]["has_customer"] is False
        assert m["speakers"]["customer_turn_count"] == 0

    def test_has_agent_true(self) -> None:
        turns = (_turn(0, SpeakerRole.AGENT),)
        m = compute_window_metrics(turns, "text", "", "text")
        assert m["speakers"]["has_agent"] is True
        assert m["speakers"]["agent_turn_count"] == 1

    def test_has_agent_false(self) -> None:
        turns = (_turn(0, SpeakerRole.CUSTOMER),)
        m = compute_window_metrics(turns, "text", "text", "")
        assert m["speakers"]["has_agent"] is False
        assert m["speakers"]["agent_turn_count"] == 0

    def test_mixed_speakers_counts(self) -> None:
        turns = (
            _turn(0, SpeakerRole.CUSTOMER),
            _turn(1, SpeakerRole.AGENT),
            _turn(2, SpeakerRole.AGENT),
            _turn(3, SpeakerRole.CUSTOMER),
        )
        m = compute_window_metrics(turns, "text", "ct", "at")
        assert m["speakers"]["customer_turn_count"] == 2
        assert m["speakers"]["agent_turn_count"] == 2


# ---------------------------------------------------------------------------
# Role texts in metrics (namespaced under "role_views")
# ---------------------------------------------------------------------------


class TestRoleTextsInMetrics:
    def test_customer_text_preserved(self) -> None:
        m = compute_window_metrics(
            turns=(_turn(0, SpeakerRole.CUSTOMER),),
            window_text="full",
            customer_text="customer only",
            agent_text="",
        )
        assert m["role_views"]["customer_text"] == "customer only"

    def test_agent_text_preserved(self) -> None:
        m = compute_window_metrics(
            turns=(_turn(0, SpeakerRole.AGENT),),
            window_text="full",
            customer_text="",
            agent_text="agent only",
        )
        assert m["role_views"]["agent_text"] == "agent only"


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


class TestMetricsStructure:
    def test_returns_all_expected_top_level_keys(self) -> None:
        m = compute_window_metrics((_turn(0),), "text", "ct", "at")
        expected_keys = {"total_chars", "total_words", "role_views", "speakers"}
        assert set(m.keys()) == expected_keys

    def test_role_views_has_expected_keys(self) -> None:
        m = compute_window_metrics((_turn(0),), "text", "ct", "at")
        assert set(m["role_views"].keys()) == {"customer_text", "agent_text"}

    def test_speakers_has_expected_keys(self) -> None:
        m = compute_window_metrics((_turn(0),), "text", "ct", "at")
        expected = {
            "distribution",
            "has_customer",
            "has_agent",
            "customer_turn_count",
            "agent_turn_count",
        }
        assert set(m["speakers"].keys()) == expected


# ---------------------------------------------------------------------------
# Build-level coverage metrics
# ---------------------------------------------------------------------------


class TestBuildCoverage:
    def test_empty_input_returns_zeroed_metrics(self) -> None:
        result = compute_build_coverage(total_turns=0, windows=[])
        assert result["total_windows"] == 0
        assert result["total_turns"] == 0
        assert result["unique_turns_covered"] == 0
        assert result["orphan_turn_count"] == 0
        assert result["coverage_ratio"] == 0.0
        assert result["multi_window_turns"] == 0

    def test_full_coverage_no_overlap(self) -> None:
        # 6 turns, 2 windows covering [0,1,2] and [3,4,5]
        windows = [_window(0, 0, 2), _window(1, 3, 5)]
        result = compute_build_coverage(total_turns=6, windows=windows)
        assert result["total_windows"] == 2
        assert result["unique_turns_covered"] == 6
        assert result["orphan_turn_count"] == 0
        assert result["coverage_ratio"] == 1.0
        assert result["multi_window_turns"] == 0

    def test_overlap_produces_multi_window_turns(self) -> None:
        # Windows [0,1,2] and [2,3,4] — turn 2 appears in both
        windows = [_window(0, 0, 2), _window(1, 2, 4)]
        result = compute_build_coverage(total_turns=5, windows=windows)
        assert result["unique_turns_covered"] == 5
        assert result["multi_window_turns"] == 1  # turn 2

    def test_gaps_produce_orphan_turns(self) -> None:
        # 10 turns, windows cover [0,1,2] and [5,6,7] — turns 3,4,8,9 orphaned
        windows = [_window(0, 0, 2), _window(1, 5, 7)]
        result = compute_build_coverage(total_turns=10, windows=windows)
        assert result["unique_turns_covered"] == 6
        assert result["orphan_turn_count"] == 4
        assert result["coverage_ratio"] == 0.6

    def test_single_window_single_turn(self) -> None:
        windows = [_window(0, 0, 0)]
        result = compute_build_coverage(total_turns=1, windows=windows)
        assert result["total_windows"] == 1
        assert result["unique_turns_covered"] == 1
        assert result["coverage_ratio"] == 1.0
        assert result["multi_window_turns"] == 0

    def test_coverage_ratio_is_float(self) -> None:
        windows = [_window(0, 0, 2)]
        result = compute_build_coverage(total_turns=5, windows=windows)
        assert isinstance(result["coverage_ratio"], float)
        assert result["coverage_ratio"] == 3 / 5
