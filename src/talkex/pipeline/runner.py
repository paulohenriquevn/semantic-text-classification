"""Operational pipeline runner — executes pipelines and persists outputs.

Bridges the gap between the SystemPipeline orchestrator and operational
use (CLI, batch jobs, scripts). Handles:

    - Pipeline construction from PipelineConfig
    - Transcript loading from files
    - Output persistence (manifest, results, reports)
    - Structured logging of execution progress

This module keeps CLI logic separate from pipeline logic (SRP).

Usage::

    runner = PipelineRunner(config=PipelineConfig())
    summary = runner.run_file("transcript.txt", channel="voice")
    runner.save_outputs(summary, output_dir="output/")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from talkex.context.builder import SlidingWindowBuilder
from talkex.embeddings.generator import NullEmbeddingGenerator
from talkex.embeddings.preprocessing import PreprocessingConfig
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel
from talkex.pipeline.config import PipelineConfig
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.pipeline.system_pipeline import (
    SystemPipeline,
    SystemPipelineResult,
)
from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.segmentation.segmenter import TurnSegmenter

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RunSummary:
    """Summary of a pipeline execution for reporting.

    Args:
        run_id: Unique pipeline run identifier.
        input_path: Path to the input transcript.
        output_dir: Directory where outputs were saved.
        total_ms: Total execution time in milliseconds.
        turns_count: Number of turns produced.
        windows_count: Number of context windows produced.
        embeddings_count: Number of embeddings generated.
        predictions_count: Number of predictions made.
        rule_executions_count: Number of rule evaluations.
        stages_executed: Number of pipeline stages executed.
        stages_skipped: Number of pipeline stages skipped.
        result: The full SystemPipelineResult.
    """

    run_id: str
    input_path: str
    output_dir: str
    total_ms: float
    turns_count: int
    windows_count: int
    embeddings_count: int
    predictions_count: int
    rule_executions_count: int
    stages_executed: int
    stages_skipped: int
    result: SystemPipelineResult


@dataclass
class PipelineRunner:
    """Operational runner for the SystemPipeline.

    Constructs a configured pipeline from PipelineConfig and provides
    high-level operations: run from file, persist outputs, generate
    summaries.

    Args:
        config: Unified pipeline configuration.
    """

    config: PipelineConfig = field(default_factory=PipelineConfig)

    def build_pipeline(
        self,
        *,
        enable_embeddings: bool = True,
        enable_rules: bool = True,
    ) -> SystemPipeline:
        """Build a SystemPipeline from the current configuration.

        Args:
            enable_embeddings: Whether to include the embedding generator.
            enable_rules: Whether to include the rule evaluator.

        Returns:
            Configured SystemPipeline ready for execution.
        """
        text_pipeline = TextProcessingPipeline(
            segmenter=TurnSegmenter(),
            context_builder=SlidingWindowBuilder(),
        )

        embedding_generator = None
        if enable_embeddings and self.config.embedding.model is not None:
            embedding_generator = NullEmbeddingGenerator(
                model_config=self.config.embedding.model,
                preprocessing_config=PreprocessingConfig(),
                dimensions=self.config.embedding.dimensions,
            )

        rule_evaluator = None
        if enable_rules:
            rule_evaluator = SimpleRuleEvaluator()

        return SystemPipeline(
            text_pipeline=text_pipeline,
            embedding_generator=embedding_generator,
            rule_evaluator=rule_evaluator,
        )

    def run_file(
        self,
        input_path: str | Path,
        *,
        channel: str = "voice",
        source_format: str = "labeled",
        conversation_id: str | None = None,
        rules_text: list[str] | None = None,
        enable_embeddings: bool = True,
        enable_rules: bool = True,
    ) -> RunSummary:
        """Execute the pipeline on a transcript file.

        Args:
            input_path: Path to the transcript file.
            channel: Communication channel (voice, chat, email).
            source_format: Transcript format (labeled, plain).
            conversation_id: Optional conversation ID. Auto-generated
                from filename if not provided.
            rules_text: Optional list of DSL rule strings to evaluate.
            enable_embeddings: Whether to run the embedding stage.
            enable_rules: Whether to run the rule evaluation stage.

        Returns:
            RunSummary with execution results and metadata.

        Raises:
            FileNotFoundError: If the transcript file does not exist.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")

        raw_text = path.read_text()
        conv_id = conversation_id or f"conv_{path.stem}"

        channel_enum = Channel(channel.lower())
        format_enum = SourceFormat(source_format.lower())

        transcript = TranscriptInput(
            conversation_id=conv_id,
            raw_text=raw_text,
            source_format=format_enum,
            channel=channel_enum,
        )

        # Compile rules if provided
        compiled_rules = None
        if rules_text and enable_rules:
            compiler = SimpleRuleCompiler()
            compiled_rules = []
            for i, rule_dsl in enumerate(rules_text):
                rule_name = f"rule_{i}"
                rule_label = f"label_{i}"
                compiled_rules.append(compiler.compile(rule_dsl, rule_name, rule_label))

        pipeline = self.build_pipeline(
            enable_embeddings=enable_embeddings,
            enable_rules=enable_rules,
        )

        logger.info(
            "pipeline_run_start",
            input_path=str(path),
            channel=channel,
            format=source_format,
            conversation_id=conv_id,
        )

        start = time.monotonic()
        result = pipeline.run(
            transcript,
            segmentation_config=self.config.segmentation,
            context_config=self.config.context,
            rules=compiled_rules,
            rule_config=self.config.rules if compiled_rules else None,
        )
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        run_id = result.manifest.run_id if result.manifest else "unknown"
        predictions_count = len(result.classification.predictions) if result.classification else 0

        logger.info(
            "pipeline_run_complete",
            run_id=run_id,
            total_ms=elapsed_ms,
            turns=len(result.pipeline_result.turns),
            windows=len(result.pipeline_result.windows),
            embeddings=len(result.embeddings),
            predictions=predictions_count,
            rule_executions=len(result.rule_executions),
        )

        return RunSummary(
            run_id=run_id,
            input_path=str(path),
            output_dir=self.config.output_dir,
            total_ms=elapsed_ms,
            turns_count=len(result.pipeline_result.turns),
            windows_count=len(result.pipeline_result.windows),
            embeddings_count=len(result.embeddings),
            predictions_count=predictions_count,
            rule_executions_count=len(result.rule_executions),
            stages_executed=sum(1 for s in result.stages if not s.skipped),
            stages_skipped=sum(1 for s in result.stages if s.skipped),
            result=result,
        )

    @staticmethod
    def save_outputs(summary: RunSummary, output_dir: str | Path | None = None) -> Path:
        """Persist pipeline outputs to disk.

        Creates a structured output directory with:
            - manifest.json — execution identity and versions
            - summary.json — run summary with counts and timing

        Args:
            summary: Pipeline run summary to persist.
            output_dir: Base output directory. Uses summary.output_dir
                if not provided.

        Returns:
            Path to the run-specific output directory.
        """
        base = Path(output_dir or summary.output_dir)
        run_dir = base / summary.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save manifest
        if summary.result.manifest:
            manifest_path = run_dir / "manifest.json"
            summary.result.manifest.save_json(manifest_path)

        # Save summary
        summary_data: dict[str, Any] = {
            "run_id": summary.run_id,
            "input_path": summary.input_path,
            "total_ms": summary.total_ms,
            "turns_count": summary.turns_count,
            "windows_count": summary.windows_count,
            "embeddings_count": summary.embeddings_count,
            "predictions_count": summary.predictions_count,
            "rule_executions_count": summary.rule_executions_count,
            "stages_executed": summary.stages_executed,
            "stages_skipped": summary.stages_skipped,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False))

        return run_dir
