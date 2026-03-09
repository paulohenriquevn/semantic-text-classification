"""Pipeline execution result — envelope for text processing output.

PipelineResult is NOT a domain entity. It is an execution envelope that
carries the outputs of a complete pipeline run: the constructed Conversation,
segmented Turns, context Windows, build-level coverage metrics, structured
warnings, and operational statistics.

Warnings are non-fatal observations about the pipeline run. They do not
halt execution but inform operators of edge conditions:
    - No turns produced after segmentation
    - Partial turn coverage (orphan turns)
    - Empty windows list
    - Configuration-related observations

Stats capture timing and counts for operational monitoring.
"""

from dataclasses import dataclass, field
from typing import Any

from semantic_conversation_engine.models.context_window import ContextWindow
from semantic_conversation_engine.models.conversation import Conversation
from semantic_conversation_engine.models.turn import Turn


@dataclass(frozen=True)
class PipelineWarning:
    """A structured, non-fatal observation from a pipeline run.

    Args:
        code: Machine-readable warning code (e.g. 'NO_TURNS_PRODUCED').
        message: Human-readable description.
        context: Optional structured data for diagnosis.
    """

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """Execution envelope for the text processing pipeline.

    Carries all outputs from a complete pipeline run. Immutable — once
    produced, the result cannot be modified.

    Args:
        conversation: The Conversation domain entity constructed from input.
        turns: Ordered list of segmented Turn objects.
        windows: Ordered list of ContextWindow objects.
        coverage: Build-level coverage metrics from compute_build_coverage.
        warnings: Non-fatal observations about the pipeline run.
        stats: Operational statistics (timing, counts).
    """

    conversation: Conversation
    turns: list[Turn]
    windows: list[ContextWindow]
    coverage: dict[str, Any]
    warnings: list[PipelineWarning] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
