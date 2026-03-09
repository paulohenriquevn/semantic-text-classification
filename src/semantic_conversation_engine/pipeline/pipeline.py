"""TextProcessingPipeline — end-to-end orchestrator for text processing.

Composes the Segmenter and ContextBuilder protocols into a single
pipeline execution. Transforms a TranscriptInput boundary object into
a PipelineResult containing domain entities, coverage metrics, structured
warnings, and operational statistics.

The pipeline flow:

    TranscriptInput
        ↓
    Segmenter.segment() → list[Turn]
        ↓
    ContextBuilder.build() → list[ContextWindow]
        ↓
    compute_build_coverage() → coverage metrics
        ↓
    PipelineResult

Dependencies are injected via constructor (DIP). The pipeline accepts
any object satisfying Segmenter and ContextBuilder protocols, not
concrete implementations.
"""

import time
from datetime import UTC, datetime

from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.context.metrics import compute_build_coverage
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.conversation import Conversation
from semantic_conversation_engine.pipeline.protocols import ContextBuilder, Segmenter
from semantic_conversation_engine.pipeline.result import PipelineResult, PipelineWarning
from semantic_conversation_engine.segmentation.config import SegmentationConfig


class TextProcessingPipeline:
    """End-to-end text processing pipeline.

    Orchestrates segmentation and context window construction from raw
    transcript input. Dependencies are injected — the pipeline depends
    on protocols, not concrete implementations.

    Args:
        segmenter: Any object implementing the Segmenter protocol.
        context_builder: Any object implementing the ContextBuilder protocol.
    """

    def __init__(
        self,
        segmenter: Segmenter,
        context_builder: ContextBuilder,
    ) -> None:
        self._segmenter = segmenter
        self._context_builder = context_builder

    def run(
        self,
        transcript: TranscriptInput,
        segmentation_config: SegmentationConfig | None = None,
        context_config: ContextWindowConfig | None = None,
    ) -> PipelineResult:
        """Execute the full text processing pipeline.

        Transforms a TranscriptInput into a PipelineResult containing
        the constructed Conversation, segmented Turns, context Windows,
        coverage metrics, warnings, and stats.

        Args:
            transcript: Raw transcript input from the ingestion boundary.
            segmentation_config: Segmentation configuration. Uses defaults
                if None.
            context_config: Context window configuration. Uses defaults
                if None.

        Returns:
            PipelineResult with all pipeline outputs.

        Raises:
            PipelineError: If a pipeline stage encounters an
                irrecoverable error.
        """
        seg_config = segmentation_config or SegmentationConfig()
        ctx_config = context_config or ContextWindowConfig()

        warnings: list[PipelineWarning] = []
        stats: dict[str, object] = {}

        pipeline_start = time.monotonic()

        # --- Stage 1: Build Conversation domain entity ---
        conversation = Conversation(
            conversation_id=transcript.conversation_id,
            channel=transcript.channel,
            start_time=datetime.now(tz=UTC),
        )

        # --- Stage 2: Segmentation ---
        seg_start = time.monotonic()
        turns = self._segmenter.segment(transcript, seg_config)
        stats["segmentation_ms"] = _elapsed_ms(seg_start)
        stats["turn_count"] = len(turns)

        if len(turns) == 0:
            warnings.append(
                PipelineWarning(
                    code="NO_TURNS_PRODUCED",
                    message="Segmentation produced zero turns from the transcript.",
                    context={"conversation_id": transcript.conversation_id},
                )
            )

        # --- Stage 3: Context window construction ---
        ctx_start = time.monotonic()
        windows = self._context_builder.build(conversation, turns, ctx_config)
        stats["context_build_ms"] = _elapsed_ms(ctx_start)
        stats["window_count"] = len(windows)

        if len(turns) > 0 and len(windows) == 0:
            warnings.append(
                PipelineWarning(
                    code="NO_WINDOWS_PRODUCED",
                    message=(
                        "Context builder produced zero windows despite "
                        f"{len(turns)} turns being available. Check min_window_size "
                        "and include_partial_tail configuration."
                    ),
                    context={
                        "conversation_id": transcript.conversation_id,
                        "turn_count": len(turns),
                        "window_size": ctx_config.window_size,
                        "min_window_size": ctx_config.min_window_size,
                    },
                )
            )

        # --- Stage 4: Coverage metrics ---
        coverage = compute_build_coverage(len(turns), windows)

        if coverage["orphan_turn_count"] > 0:
            warnings.append(
                PipelineWarning(
                    code="ORPHAN_TURNS_DETECTED",
                    message=(
                        f"{coverage['orphan_turn_count']} turn(s) are not covered "
                        "by any context window. Consider reducing stride or "
                        "increasing window_size."
                    ),
                    context={
                        "conversation_id": transcript.conversation_id,
                        "orphan_count": coverage["orphan_turn_count"],
                        "coverage_ratio": coverage["coverage_ratio"],
                    },
                )
            )

        stats["pipeline_ms"] = _elapsed_ms(pipeline_start)

        return PipelineResult(
            conversation=conversation,
            turns=turns,
            windows=windows,
            coverage=coverage,
            warnings=warnings,
            stats=stats,
        )


def _elapsed_ms(start: float) -> float:
    """Compute elapsed milliseconds since a monotonic start time."""
    return round((time.monotonic() - start) * 1000, 2)
