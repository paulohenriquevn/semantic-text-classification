"""SystemPipeline — end-to-end orchestrator composing all pipeline stages.

Integrates the six subsystems into a single execution flow:

    TranscriptInput
        ↓
    Stage 1: Text Processing (Segmenter → ContextBuilder)
        ↓
    Stage 2: Embedding Generation (EmbeddingGenerator)
        ↓
    Stage 3: Index Building (VectorIndex + LexicalIndex)
        ↓
    Stage 4: Classification (Classifier → Predictions)
        ↓
    Stage 5: Rule Evaluation (RuleCompiler + RuleEvaluator → RuleExecutions)
        ↓
    Stage 6: Analytics Event Collection
        ↓
    SystemPipelineResult

Each stage is optional — the pipeline degrades gracefully when components
are not provided. Per-stage timing, warnings, and error isolation are
built in. Dependencies are injected via constructor (DIP).

This is a thin composition layer. The real work happens in existing
orchestrators (TextProcessingPipeline, ClassificationOrchestrator) and
concrete implementations of each protocol.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from talkex.classification.orchestrator import (
    ClassificationBatchResult,
    ClassificationOrchestrator,
)
from talkex.context.config import ContextWindowConfig
from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from talkex.models.context_window import ContextWindow
from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import ObjectType
from talkex.models.rule_execution import RuleExecution
from talkex.models.types import EmbeddingId
from talkex.pipeline.manifest import (
    PipelineRunManifest,
    compute_config_fingerprint,
)
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.pipeline.result import PipelineResult, PipelineWarning
from talkex.rules.config import RuleEngineConfig
from talkex.rules.models import RuleDefinition, RuleEvaluationInput
from talkex.segmentation.config import SegmentationConfig

if TYPE_CHECKING:
    from talkex.analytics.models import AnalyticsEvent
    from talkex.pipeline.protocols import (
        Classifier,
        EmbeddingGenerator,
        LexicalIndex,
        RuleEvaluator,
        VectorIndex,
    )


@dataclass(frozen=True)
class StageResult:
    """Execution result for a single pipeline stage.

    Args:
        name: Stage name (e.g. 'text_processing', 'embedding').
        elapsed_ms: Wall-clock time for the stage.
        item_count: Number of items produced by the stage.
        skipped: Whether the stage was skipped (missing component).
        error: Error message if the stage failed, None otherwise.
    """

    name: str
    elapsed_ms: float
    item_count: int = 0
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True)
class SystemPipelineResult:
    """Execution envelope for the full system pipeline.

    Carries all outputs from a complete end-to-end run. Each stage
    populates its section; skipped stages produce empty lists.

    The ``manifest`` field provides artifact lineage: a unique run_id,
    timestamp, component versions, and configuration fingerprint that
    tie every produced artifact back to this execution.

    Args:
        pipeline_result: Text processing result (conversation, turns, windows).
        embeddings: Generated embedding records.
        classification: Classification batch result with predictions.
        rule_executions: Rule evaluation results mapped to domain entities.
        analytics_events: Analytics events generated from predictions and rules.
        stages: Per-stage execution results for observability.
        warnings: Non-fatal observations from any stage.
        stats: Aggregated operational statistics.
        manifest: Execution identity and version manifest for this run.
    """

    pipeline_result: PipelineResult
    embeddings: list[EmbeddingRecord] = field(default_factory=list)
    classification: ClassificationBatchResult | None = None
    rule_executions: list[RuleExecution] = field(default_factory=list)
    analytics_events: list[AnalyticsEvent] = field(default_factory=list)
    stages: list[StageResult] = field(default_factory=list)
    warnings: list[PipelineWarning] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    manifest: PipelineRunManifest | None = None


def _elapsed_ms(start: float) -> float:
    """Compute elapsed milliseconds since a monotonic start time."""
    return round((time.monotonic() - start) * 1000, 2)


def _build_embedding_inputs(
    windows: list[ContextWindow],
) -> list[EmbeddingInput]:
    """Build EmbeddingInput objects from context windows.

    Args:
        windows: Context windows to embed.

    Returns:
        List of EmbeddingInput objects, one per window.
    """
    inputs: list[EmbeddingInput] = []
    for window in windows:
        inputs.append(
            EmbeddingInput(
                embedding_id=EmbeddingId(f"emb_{uuid.uuid4().hex[:12]}"),
                object_type=ObjectType.CONTEXT_WINDOW,
                object_id=window.window_id,
                text=window.window_text,
                metadata={
                    "conversation_id": window.conversation_id,
                    "start_index": window.start_index,
                    "end_index": window.end_index,
                },
            )
        )
    return inputs


def _build_rule_evaluation_input(
    window: ContextWindow,
    features: dict[str, float],
) -> RuleEvaluationInput:
    """Build a RuleEvaluationInput from a context window.

    Args:
        window: The context window to evaluate rules against.
        features: Pre-extracted feature dict for the window.

    Returns:
        RuleEvaluationInput ready for rule evaluation.
    """
    return RuleEvaluationInput(
        source_id=window.window_id,
        source_type="context_window",
        text=window.window_text,
        features=features,
        metadata=window.metadata,
    )


class SystemPipeline:
    """End-to-end system orchestrator composing all pipeline stages.

    Composes text processing, embedding generation, index building,
    classification, rule evaluation, and analytics into a single
    auditable execution flow.

    All dependencies are optional except ``text_pipeline``. Missing
    components cause their stages to be skipped with a warning.

    Args:
        text_pipeline: Text processing pipeline (required).
        embedding_generator: Embedding generator (optional).
        vector_index: Vector index for semantic search (optional).
        lexical_index: Lexical index for BM25 search (optional).
        classifier: Classifier for supervised classification (optional).
        rule_evaluator: Rule evaluator for DSL rules (optional).
    """

    def __init__(
        self,
        text_pipeline: TextProcessingPipeline,
        *,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_index: VectorIndex | None = None,
        lexical_index: LexicalIndex | None = None,
        classifier: Classifier | None = None,
        rule_evaluator: RuleEvaluator | None = None,
    ) -> None:
        self._text_pipeline = text_pipeline
        self._embedding_generator = embedding_generator
        self._vector_index = vector_index
        self._lexical_index = lexical_index
        self._classifier = classifier
        self._rule_evaluator = rule_evaluator

    def _collect_component_versions(self) -> dict[str, str]:
        """Collect version information from all injected components.

        Inspects each component for ``model_name``, ``model_version``,
        or ``__class__.__name__`` to build the version manifest.

        Returns:
            Mapping of component name to version string.
        """
        versions: dict[str, str] = {}
        versions["text_pipeline"] = type(self._text_pipeline).__name__

        if self._embedding_generator is not None:
            gen = self._embedding_generator
            model_config = getattr(gen, "model_config", None)
            if model_config is not None:
                name = getattr(model_config, "model_name", "unknown")
                version = getattr(model_config, "model_version", "unknown")
                versions["embedding_model"] = f"{name}/{version}"
            else:
                versions["embedding_model"] = type(gen).__name__

        if self._vector_index is not None:
            versions["vector_index"] = type(self._vector_index).__name__

        if self._lexical_index is not None:
            versions["lexical_index"] = type(self._lexical_index).__name__

        if self._classifier is not None:
            versions["classifier"] = type(self._classifier).__name__

        if self._rule_evaluator is not None:
            versions["rule_evaluator"] = type(self._rule_evaluator).__name__

        return versions

    def run(
        self,
        transcript: Any,
        *,
        segmentation_config: SegmentationConfig | None = None,
        context_config: ContextWindowConfig | None = None,
        rules: list[RuleDefinition] | None = None,
        rule_config: RuleEngineConfig | None = None,
        embeddings_for_classification: dict[str, list[float]] | None = None,
    ) -> SystemPipelineResult:
        """Execute the full system pipeline.

        Runs all stages sequentially. Each stage is timed independently.
        Stages with missing components are skipped with a warning.

        Args:
            transcript: TranscriptInput boundary object.
            segmentation_config: Segmentation configuration. Uses defaults
                if None.
            context_config: Context window configuration. Uses defaults
                if None.
            rules: Compiled rule definitions to evaluate. Skips rule
                evaluation if None or empty.
            rule_config: Rule engine configuration. Uses defaults if None.
            embeddings_for_classification: Optional pre-computed embeddings
                keyed by window_id for classification. If None and
                embedding_generator is available, generated embeddings
                are used instead.

        Returns:
            SystemPipelineResult with all stage outputs and observability.
        """
        pipeline_start = time.monotonic()
        stage_results: list[StageResult] = []
        all_warnings: list[PipelineWarning] = []
        all_stats: dict[str, Any] = {}

        # --- Build manifest ---
        component_versions = self._collect_component_versions()
        config_dict: dict[str, Any] = {
            "segmentation_config": repr(segmentation_config),
            "context_config": repr(context_config),
            "rule_count": len(rules) if rules else 0,
            "rule_config": repr(rule_config),
            "components": sorted(component_versions.keys()),
        }
        manifest = PipelineRunManifest.create(
            component_versions=component_versions,
            config_fingerprint=compute_config_fingerprint(config_dict),
        )

        # --- Stage 1: Text Processing ---
        stage_start = time.monotonic()
        pipeline_result = self._text_pipeline.run(
            transcript,
            segmentation_config=segmentation_config,
            context_config=context_config,
        )
        stage_results.append(
            StageResult(
                name="text_processing",
                elapsed_ms=_elapsed_ms(stage_start),
                item_count=len(pipeline_result.windows),
            )
        )
        all_warnings.extend(pipeline_result.warnings)
        all_stats["text_processing"] = pipeline_result.stats
        windows = pipeline_result.windows

        # --- Stage 2: Embedding Generation ---
        embedding_records: list[EmbeddingRecord] = []
        if self._embedding_generator is not None and len(windows) > 0:
            stage_start = time.monotonic()
            embedding_inputs = _build_embedding_inputs(windows)
            batch = EmbeddingBatch(items=embedding_inputs)
            embedding_records = self._embedding_generator.generate(batch)
            stage_results.append(
                StageResult(
                    name="embedding",
                    elapsed_ms=_elapsed_ms(stage_start),
                    item_count=len(embedding_records),
                )
            )
            all_stats["embedding_count"] = len(embedding_records)
        else:
            reason = "no windows" if len(windows) == 0 else "no embedding_generator"
            stage_results.append(StageResult(name="embedding", elapsed_ms=0.0, skipped=True))
            if len(windows) > 0:
                all_warnings.append(
                    PipelineWarning(
                        code="EMBEDDING_STAGE_SKIPPED",
                        message=f"Embedding stage skipped: {reason}.",
                    )
                )

        # --- Stage 3: Index Building ---
        index_built = False
        if len(embedding_records) > 0 or len(windows) > 0:
            stage_start = time.monotonic()
            items_indexed = 0

            if self._vector_index is not None and len(embedding_records) > 0:
                self._vector_index.upsert(embedding_records)
                items_indexed += len(embedding_records)

            if self._lexical_index is not None and len(windows) > 0:
                from talkex.retrieval.builders import (
                    context_windows_to_lexical_docs,
                )

                docs = context_windows_to_lexical_docs(windows)
                self._lexical_index.index(docs)
                items_indexed += len(docs)

            if items_indexed > 0:
                index_built = True
                stage_results.append(
                    StageResult(
                        name="indexing",
                        elapsed_ms=_elapsed_ms(stage_start),
                        item_count=items_indexed,
                    )
                )
                all_stats["items_indexed"] = items_indexed
            else:
                stage_results.append(StageResult(name="indexing", elapsed_ms=0.0, skipped=True))
        else:
            stage_results.append(StageResult(name="indexing", elapsed_ms=0.0, skipped=True))

        # --- Stage 4: Classification ---
        classification_result: ClassificationBatchResult | None = None
        if self._classifier is not None and len(windows) > 0:
            stage_start = time.monotonic()
            orchestrator = ClassificationOrchestrator(classifier=self._classifier)

            # Use provided embeddings, or build from generated records
            emb_map = embeddings_for_classification
            if emb_map is None and len(embedding_records) > 0:
                emb_map = {rec.source_id: rec.vector for rec in embedding_records}

            classification_result = orchestrator.classify_windows(windows, embeddings=emb_map)
            stage_results.append(
                StageResult(
                    name="classification",
                    elapsed_ms=_elapsed_ms(stage_start),
                    item_count=len(classification_result.predictions),
                )
            )
            all_stats["classification"] = classification_result.stats
        else:
            stage_results.append(StageResult(name="classification", elapsed_ms=0.0, skipped=True))
            if len(windows) > 0 and self._classifier is None:
                all_warnings.append(
                    PipelineWarning(
                        code="CLASSIFICATION_STAGE_SKIPPED",
                        message="Classification stage skipped: no classifier provided.",
                    )
                )

        # --- Stage 5: Rule Evaluation ---
        rule_executions: list[RuleExecution] = []
        if self._rule_evaluator is not None and rules and len(windows) > 0:
            from talkex.classification.features import (
                extract_lexical_features,
            )
            from talkex.rules.evaluator import (
                map_to_rule_execution,
            )

            stage_start = time.monotonic()
            effective_config = rule_config or RuleEngineConfig()

            for window in windows:
                lexical = extract_lexical_features(window.window_text)
                rule_input = _build_rule_evaluation_input(window, lexical.features)
                results = self._rule_evaluator.evaluate(rules, rule_input, effective_config)
                for result in results:
                    rule_executions.append(map_to_rule_execution(result))

            stage_results.append(
                StageResult(
                    name="rule_evaluation",
                    elapsed_ms=_elapsed_ms(stage_start),
                    item_count=len(rule_executions),
                )
            )
            all_stats["rule_execution_count"] = len(rule_executions)
        else:
            stage_results.append(StageResult(name="rule_evaluation", elapsed_ms=0.0, skipped=True))

        # --- Stage 6: Analytics Event Collection ---
        analytics_events: list[AnalyticsEvent] = []
        predictions = classification_result.predictions if classification_result else []
        if len(predictions) > 0 or len(rule_executions) > 0:
            from talkex.analytics.builders import (
                prediction_to_event,
                rule_execution_to_event,
            )

            stage_start = time.monotonic()
            now = datetime.now(tz=UTC)
            event_counter = 0

            for pred in predictions:
                event_counter += 1
                analytics_events.append(
                    prediction_to_event(
                        pred,
                        event_id=f"evt_{event_counter:04d}",
                        timestamp=now,
                    )
                )

            for execution in rule_executions:
                event_counter += 1
                analytics_events.append(
                    rule_execution_to_event(
                        execution,
                        event_id=f"evt_{event_counter:04d}",
                        timestamp=now,
                    )
                )

            stage_results.append(
                StageResult(
                    name="analytics",
                    elapsed_ms=_elapsed_ms(stage_start),
                    item_count=len(analytics_events),
                )
            )
            all_stats["analytics_event_count"] = len(analytics_events)
        else:
            stage_results.append(StageResult(name="analytics", elapsed_ms=0.0, skipped=True))

        # --- Aggregated stats ---
        all_stats["total_pipeline_ms"] = _elapsed_ms(pipeline_start)
        all_stats["stages_executed"] = sum(1 for s in stage_results if not s.skipped)
        all_stats["stages_skipped"] = sum(1 for s in stage_results if s.skipped)
        all_stats["index_built"] = index_built

        return SystemPipelineResult(
            pipeline_result=pipeline_result,
            embeddings=embedding_records,
            classification=classification_result,
            rule_executions=rule_executions,
            analytics_events=analytics_events,
            stages=stage_results,
            warnings=all_warnings,
            stats=all_stats,
            manifest=manifest,
        )
