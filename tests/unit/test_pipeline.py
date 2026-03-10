"""Unit tests for TextProcessingPipeline orchestrator.

Tests cover: happy path, zero-turn handling, zero-window handling,
orphan turn warnings, coverage metrics propagation, stats collection,
default config fallback, determinism, and DIP compliance.

Uses stub implementations of Segmenter and ContextBuilder protocols
to test the orchestrator in isolation.
"""

from typing import Any

import pytest

from talkex.context.config import ContextWindowConfig
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.context_window import ContextWindow
from talkex.models.conversation import Conversation
from talkex.models.enums import Channel, SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId, WindowId
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.pipeline.result import PipelineWarning
from talkex.segmentation.config import SegmentationConfig

# ---------------------------------------------------------------------------
# Stubs (protocol-conforming, no inheritance needed)
# ---------------------------------------------------------------------------


class StubSegmenter:
    """Configurable stub for the Segmenter protocol."""

    def __init__(self, turns: list[Turn] | None = None) -> None:
        self._turns = turns or []
        self.call_count = 0
        self.last_config: SegmentationConfig | None = None

    def segment(self, transcript: TranscriptInput, config: SegmentationConfig) -> list[Turn]:
        self.call_count += 1
        self.last_config = config
        return self._turns


class StubContextBuilder:
    """Configurable stub for the ContextBuilder protocol."""

    def __init__(self, windows: list[ContextWindow] | None = None) -> None:
        self._windows = windows or []
        self.call_count = 0
        self.last_config: ContextWindowConfig | None = None

    def build(
        self,
        conversation: Conversation,
        turns: list[Turn],
        config: ContextWindowConfig,
    ) -> list[ContextWindow]:
        self.call_count += 1
        self.last_config = config
        return self._windows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transcript(**overrides: Any) -> TranscriptInput:
    defaults: dict[str, Any] = {
        "conversation_id": ConversationId("conv_p"),
        "channel": Channel.CHAT,
        "raw_text": "CUSTOMER: hello\nAGENT: hi there",
        "source_format": SourceFormat.LABELED,
    }
    defaults.update(overrides)
    return TranscriptInput(**defaults)


def _turn(index: int, speaker: SpeakerRole = SpeakerRole.CUSTOMER) -> Turn:
    return Turn(
        turn_id=TurnId(f"conv_p_turn_{index}"),
        conversation_id=ConversationId("conv_p"),
        speaker=speaker,
        raw_text=f"Turn {index}",
        start_offset=index * 20,
        end_offset=index * 20 + 10,
        normalized_text=f"turn {index}",
    )


def _window(index: int, start: int, end: int) -> ContextWindow:
    return ContextWindow(
        window_id=WindowId(f"conv_p_win_{index}"),
        conversation_id=ConversationId("conv_p"),
        turn_ids=[TurnId(f"conv_p_turn_{i}") for i in range(start, end + 1)],
        window_text=f"window {index} text",
        start_index=start,
        end_index=end,
        window_size=end - start + 1,
        stride=2,
    )


def _pipeline(
    turns: list[Turn] | None = None,
    windows: list[ContextWindow] | None = None,
) -> tuple[TextProcessingPipeline, StubSegmenter, StubContextBuilder]:
    seg = StubSegmenter(turns)
    ctx = StubContextBuilder(windows)
    return TextProcessingPipeline(seg, ctx), seg, ctx


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_produces_pipeline_result(self) -> None:
        turns = [_turn(0, SpeakerRole.CUSTOMER), _turn(1, SpeakerRole.AGENT)]
        windows = [_window(0, 0, 1)]
        pipe, _, _ = _pipeline(turns, windows)
        result = pipe.run(_transcript())
        assert result.conversation is not None
        assert result.turns == turns
        assert result.windows == windows

    def test_conversation_has_correct_id(self) -> None:
        pipe, _, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        result = pipe.run(_transcript())
        assert result.conversation.conversation_id == "conv_p"

    def test_conversation_has_correct_channel(self) -> None:
        pipe, _, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        result = pipe.run(_transcript())
        assert result.conversation.channel == Channel.CHAT

    def test_no_warnings_on_normal_run(self) -> None:
        turns = [_turn(0), _turn(1)]
        windows = [_window(0, 0, 1)]
        pipe, _, _ = _pipeline(turns, windows)
        result = pipe.run(_transcript())
        assert result.warnings == []

    def test_coverage_present(self) -> None:
        turns = [_turn(0), _turn(1)]
        windows = [_window(0, 0, 1)]
        pipe, _, _ = _pipeline(turns, windows)
        result = pipe.run(_transcript())
        assert result.coverage["total_turns"] == 2
        assert result.coverage["unique_turns_covered"] == 2
        assert result.coverage["coverage_ratio"] == 1.0


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_warns_on_zero_turns(self) -> None:
        pipe, _, _ = _pipeline(turns=[], windows=[])
        result = pipe.run(_transcript())
        codes = [w.code for w in result.warnings]
        assert "NO_TURNS_PRODUCED" in codes

    def test_warns_on_zero_windows_with_turns(self) -> None:
        pipe, _, _ = _pipeline(turns=[_turn(0)], windows=[])
        result = pipe.run(_transcript())
        codes = [w.code for w in result.warnings]
        assert "NO_WINDOWS_PRODUCED" in codes

    def test_no_window_warning_when_zero_turns(self) -> None:
        """Zero turns → zero windows is expected, not a separate warning."""
        pipe, _, _ = _pipeline(turns=[], windows=[])
        result = pipe.run(_transcript())
        codes = [w.code for w in result.warnings]
        assert "NO_WINDOWS_PRODUCED" not in codes

    def test_warns_on_orphan_turns(self) -> None:
        # 4 turns but window only covers [0,1]
        turns = [_turn(i) for i in range(4)]
        windows = [_window(0, 0, 1)]
        pipe, _, _ = _pipeline(turns, windows)
        result = pipe.run(_transcript())
        codes = [w.code for w in result.warnings]
        assert "ORPHAN_TURNS_DETECTED" in codes

    def test_orphan_warning_has_context(self) -> None:
        turns = [_turn(i) for i in range(4)]
        windows = [_window(0, 0, 1)]
        pipe, _, _ = _pipeline(turns, windows)
        result = pipe.run(_transcript())
        orphan_warning = next(w for w in result.warnings if w.code == "ORPHAN_TURNS_DETECTED")
        assert orphan_warning.context["orphan_count"] == 2
        assert orphan_warning.context["conversation_id"] == "conv_p"

    def test_warning_is_pipeline_warning_type(self) -> None:
        pipe, _, _ = _pipeline(turns=[], windows=[])
        result = pipe.run(_transcript())
        for w in result.warnings:
            assert isinstance(w, PipelineWarning)

    def test_no_orphan_warning_when_full_coverage(self) -> None:
        turns = [_turn(0), _turn(1)]
        windows = [_window(0, 0, 1)]
        pipe, _, _ = _pipeline(turns, windows)
        result = pipe.run(_transcript())
        codes = [w.code for w in result.warnings]
        assert "ORPHAN_TURNS_DETECTED" not in codes


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_has_turn_count(self) -> None:
        turns = [_turn(0), _turn(1), _turn(2)]
        pipe, _, _ = _pipeline(turns, [_window(0, 0, 2)])
        result = pipe.run(_transcript())
        assert result.stats["turn_count"] == 3

    def test_stats_has_window_count(self) -> None:
        pipe, _, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        result = pipe.run(_transcript())
        assert result.stats["window_count"] == 1

    def test_stats_has_timing(self) -> None:
        pipe, _, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        result = pipe.run(_transcript())
        assert "segmentation_ms" in result.stats
        assert "context_build_ms" in result.stats
        assert "pipeline_ms" in result.stats

    def test_timing_values_are_non_negative(self) -> None:
        pipe, _, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        result = pipe.run(_transcript())
        assert result.stats["segmentation_ms"] >= 0
        assert result.stats["context_build_ms"] >= 0
        assert result.stats["pipeline_ms"] >= 0


# ---------------------------------------------------------------------------
# Config propagation
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    def test_uses_default_segmentation_config(self) -> None:
        pipe, seg, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        pipe.run(_transcript())
        assert seg.last_config is not None
        assert seg.last_config.normalize_unicode is True  # default

    def test_uses_custom_segmentation_config(self) -> None:
        pipe, seg, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        custom = SegmentationConfig(normalize_unicode=False)
        pipe.run(_transcript(), segmentation_config=custom)
        assert seg.last_config is not None
        assert seg.last_config.normalize_unicode is False

    def test_uses_default_context_config(self) -> None:
        pipe, _, ctx = _pipeline([_turn(0)], [_window(0, 0, 0)])
        pipe.run(_transcript())
        assert ctx.last_config is not None
        assert ctx.last_config.window_size == 5  # default

    def test_uses_custom_context_config(self) -> None:
        pipe, _, ctx = _pipeline([_turn(0)], [_window(0, 0, 0)])
        custom = ContextWindowConfig(window_size=10)
        pipe.run(_transcript(), context_config=custom)
        assert ctx.last_config is not None
        assert ctx.last_config.window_size == 10


# ---------------------------------------------------------------------------
# DIP compliance — protocol stubs work without inheritance
# ---------------------------------------------------------------------------


class TestDIPCompliance:
    def test_segmenter_called_once(self) -> None:
        pipe, seg, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        pipe.run(_transcript())
        assert seg.call_count == 1

    def test_context_builder_called_once(self) -> None:
        pipe, _, ctx = _pipeline([_turn(0)], [_window(0, 0, 0)])
        pipe.run(_transcript())
        assert ctx.call_count == 1

    def test_pipeline_accepts_any_protocol_conforming_objects(self) -> None:
        """Stubs have no inheritance from Protocol classes — pure duck typing."""
        seg = StubSegmenter([_turn(0)])
        ctx = StubContextBuilder([_window(0, 0, 0)])
        pipe = TextProcessingPipeline(seg, ctx)
        result = pipe.run(_transcript())
        assert len(result.turns) == 1


# ---------------------------------------------------------------------------
# Coverage metrics integration
# ---------------------------------------------------------------------------


class TestCoverageIntegration:
    def test_coverage_reflects_actual_windows(self) -> None:
        turns = [_turn(i) for i in range(6)]
        windows = [_window(0, 0, 2), _window(1, 2, 4)]
        pipe, _, _ = _pipeline(turns, windows)
        result = pipe.run(_transcript())
        assert result.coverage["total_windows"] == 2
        assert result.coverage["unique_turns_covered"] == 5
        assert result.coverage["orphan_turn_count"] == 1  # turn 5

    def test_empty_pipeline_coverage(self) -> None:
        pipe, _, _ = _pipeline([], [])
        result = pipe.run(_transcript())
        assert result.coverage["total_turns"] == 0
        assert result.coverage["coverage_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Result immutability
# ---------------------------------------------------------------------------


class TestResultImmutability:
    def test_result_is_frozen(self) -> None:
        pipe, _, _ = _pipeline([_turn(0)], [_window(0, 0, 0)])
        result = pipe.run(_transcript())
        with pytest.raises(AttributeError):
            result.turns = []  # type: ignore[misc]

    def test_warning_is_frozen(self) -> None:
        pipe, _, _ = _pipeline([], [])
        result = pipe.run(_transcript())
        assert len(result.warnings) > 0
        with pytest.raises(AttributeError):
            result.warnings[0].code = "MODIFIED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestPipelineReexport:
    def test_importable_from_pipeline_package(self) -> None:
        from talkex.pipeline import (
            TextProcessingPipeline as Imported,
        )

        assert Imported is TextProcessingPipeline

    def test_result_importable_from_pipeline_package(self) -> None:
        from talkex.pipeline import (
            PipelineResult as Imported,
        )
        from talkex.pipeline.result import (
            PipelineResult as Direct,
        )

        assert Imported is Direct

    def test_warning_importable_from_pipeline_package(self) -> None:
        from talkex.pipeline import (
            PipelineWarning as Imported,
        )
        from talkex.pipeline.result import (
            PipelineWarning as Direct,
        )

        assert Imported is Direct
