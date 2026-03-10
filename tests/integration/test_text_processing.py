"""Integration tests for the text processing pipeline.

Tests the full pipeline end-to-end with real TurnSegmenter and
SlidingWindowBuilder implementations. No mocks — validates that
all pipeline stages compose correctly and produce valid output.
"""

from typing import Any

from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel
from talkex.models.types import ConversationId
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.segmentation.config import SegmentationConfig
from talkex.segmentation.segmenter import TurnSegmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipeline() -> TextProcessingPipeline:
    return TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    )


def _transcript(**overrides: Any) -> TranscriptInput:
    defaults: dict[str, Any] = {
        "conversation_id": ConversationId("conv_int"),
        "channel": Channel.CHAT,
        "raw_text": (
            "CUSTOMER: I need help with my account\n"
            "AGENT: Sure, I can help you with that\n"
            "CUSTOMER: My balance seems wrong\n"
            "AGENT: Let me check your account\n"
            "CUSTOMER: Thank you\n"
            "AGENT: Your balance has been corrected"
        ),
        "source_format": SourceFormat.LABELED,
    }
    defaults.update(overrides)
    return TranscriptInput(**defaults)


# ---------------------------------------------------------------------------
# End-to-end flow
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    def test_produces_turns_and_windows(self) -> None:
        result = _pipeline().run(_transcript())
        assert len(result.turns) > 0
        assert len(result.windows) > 0

    def test_conversation_id_propagated_through_pipeline(self) -> None:
        result = _pipeline().run(_transcript())
        assert result.conversation.conversation_id == "conv_int"
        for turn in result.turns:
            assert turn.conversation_id == "conv_int"
        for window in result.windows:
            assert window.conversation_id == "conv_int"

    def test_turn_ids_are_deterministic(self) -> None:
        a = _pipeline().run(_transcript())
        b = _pipeline().run(_transcript())
        assert [t.turn_id for t in a.turns] == [t.turn_id for t in b.turns]

    def test_window_ids_are_deterministic(self) -> None:
        a = _pipeline().run(_transcript())
        b = _pipeline().run(_transcript())
        assert [w.window_id for w in a.windows] == [w.window_id for w in b.windows]

    def test_window_turn_ids_reference_actual_turns(self) -> None:
        result = _pipeline().run(_transcript())
        turn_id_set = {t.turn_id for t in result.turns}
        for window in result.windows:
            for tid in window.turn_ids:
                assert tid in turn_id_set


# ---------------------------------------------------------------------------
# Segmentation correctness
# ---------------------------------------------------------------------------


class TestSegmentationCorrectness:
    def test_labeled_transcript_parses_speakers(self) -> None:
        result = _pipeline().run(_transcript())
        speakers = {t.speaker.value for t in result.turns}
        assert "customer" in speakers
        assert "agent" in speakers

    def test_turns_have_normalized_text(self) -> None:
        result = _pipeline().run(_transcript())
        for turn in result.turns:
            assert turn.normalized_text is not None
            assert len(turn.normalized_text) > 0

    def test_turns_have_valid_offsets(self) -> None:
        result = _pipeline().run(_transcript())
        for turn in result.turns:
            assert turn.end_offset > turn.start_offset


# ---------------------------------------------------------------------------
# Context window correctness
# ---------------------------------------------------------------------------


class TestContextWindowCorrectness:
    def test_windows_have_non_empty_text(self) -> None:
        result = _pipeline().run(_transcript())
        for window in result.windows:
            assert window.window_text.strip() != ""

    def test_windows_have_role_views(self) -> None:
        result = _pipeline().run(_transcript())
        for window in result.windows:
            assert "role_views" in window.metadata
            assert "customer_text" in window.metadata["role_views"]
            assert "agent_text" in window.metadata["role_views"]

    def test_windows_have_speaker_metrics(self) -> None:
        result = _pipeline().run(_transcript())
        for window in result.windows:
            assert "speakers" in window.metadata
            assert "distribution" in window.metadata["speakers"]

    def test_window_size_equals_turn_ids_length(self) -> None:
        result = _pipeline().run(_transcript())
        for window in result.windows:
            assert window.window_size == len(window.turn_ids)


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------


class TestCoverageMetrics:
    def test_full_coverage_with_default_config(self) -> None:
        result = _pipeline().run(_transcript())
        # Default stride=2, window_size=5, include_partial_tail=True
        # should cover all turns
        assert result.coverage["coverage_ratio"] == 1.0
        assert result.coverage["orphan_turn_count"] == 0

    def test_coverage_reflects_turn_count(self) -> None:
        result = _pipeline().run(_transcript())
        assert result.coverage["total_turns"] == len(result.turns)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestIntegrationWarnings:
    def test_no_warnings_on_healthy_input(self) -> None:
        result = _pipeline().run(_transcript())
        assert result.warnings == []

    def test_warns_on_short_text_producing_no_turns(self) -> None:
        transcript = _transcript(
            raw_text="x",
            source_format=SourceFormat.PLAIN,
        )
        config = SegmentationConfig(min_turn_chars=10)
        result = _pipeline().run(transcript, segmentation_config=config)
        codes = [w.code for w in result.warnings]
        assert "NO_TURNS_PRODUCED" in codes


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestIntegrationStats:
    def test_stats_populated(self) -> None:
        result = _pipeline().run(_transcript())
        assert result.stats["turn_count"] > 0
        assert result.stats["window_count"] > 0
        assert result.stats["pipeline_ms"] >= 0

    def test_pipeline_ms_gte_sum_of_stages(self) -> None:
        result = _pipeline().run(_transcript())
        stage_sum = result.stats["segmentation_ms"] + result.stats["context_build_ms"]
        assert result.stats["pipeline_ms"] >= stage_sum


# ---------------------------------------------------------------------------
# Config customization
# ---------------------------------------------------------------------------


class TestConfigCustomization:
    def test_custom_window_size(self) -> None:
        ctx_config = ContextWindowConfig(window_size=2, stride=1)
        result = _pipeline().run(_transcript(), context_config=ctx_config)
        for window in result.windows:
            assert window.window_size <= 2

    def test_custom_segmentation_disables_merge(self) -> None:
        seg_config = SegmentationConfig(merge_consecutive_same_speaker=False)
        result = _pipeline().run(_transcript(), segmentation_config=seg_config)
        assert len(result.turns) >= 6  # 6 labeled lines, no merging


# ---------------------------------------------------------------------------
# Multiline and plain formats
# ---------------------------------------------------------------------------


class TestAlternateFormats:
    def test_multiline_format(self) -> None:
        transcript = _transcript(
            raw_text="First line\nSecond line\nThird line",
            source_format=SourceFormat.MULTILINE,
        )
        result = _pipeline().run(transcript)
        assert len(result.turns) > 0
        assert len(result.windows) > 0

    def test_plain_format(self) -> None:
        transcript = _transcript(
            raw_text="This is a single block of text with no formatting.",
            source_format=SourceFormat.PLAIN,
        )
        result = _pipeline().run(transcript)
        assert len(result.turns) == 1
        assert len(result.windows) == 1
