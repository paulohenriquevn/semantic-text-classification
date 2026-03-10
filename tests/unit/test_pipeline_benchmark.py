"""Unit tests for system pipeline benchmark runner.

Tests cover: SystemBenchmarkRunner (run_scenario, compare),
ScenarioResult, SystemBenchmarkReport (JSON/CSV export),
empty scenarios, custom config, and reexports.
"""

from __future__ import annotations

import json

from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig
from talkex.embeddings.config import EmbeddingModelConfig
from talkex.embeddings.generator import NullEmbeddingGenerator
from talkex.embeddings.preprocessing import PreprocessingConfig
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel
from talkex.pipeline.benchmark import (
    SystemBenchmarkConfig,
    SystemBenchmarkReport,
    SystemBenchmarkRunner,
)
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.pipeline.system_pipeline import (
    SystemPipeline,
    SystemPipelineResult,
)
from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT = TranscriptInput(
    conversation_id="conv_bench",
    raw_text="""\
CUSTOMER: I have a billing issue.
AGENT: I can help you with that.
CUSTOMER: I was charged twice.
AGENT: Let me look into that.
""",
    source_format=SourceFormat.LABELED,
    channel=Channel.VOICE,
)

_CONTEXT_CONFIG = ContextWindowConfig(window_size=3, stride=2)


def _make_text_pipeline() -> TextProcessingPipeline:
    return TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    )


def _make_embedding_generator() -> NullEmbeddingGenerator:
    return NullEmbeddingGenerator(
        model_config=EmbeddingModelConfig(model_name="null-bench", model_version="1.0"),
        preprocessing_config=PreprocessingConfig(),
        dimensions=64,
    )


def _run_full() -> SystemPipelineResult:
    """Run full pipeline with embeddings and rules."""
    compiler = SimpleRuleCompiler()
    rules = [compiler.compile('keyword("billing")', "rule_billing", "billing_issue")]

    pipeline = SystemPipeline(
        text_pipeline=_make_text_pipeline(),
        embedding_generator=_make_embedding_generator(),
        rule_evaluator=SimpleRuleEvaluator(),
    )
    return pipeline.run(
        _TRANSCRIPT,
        context_config=_CONTEXT_CONFIG,
        rules=rules,
    )


def _run_text_only() -> SystemPipelineResult:
    """Run text processing only."""
    pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
    return pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)


def _run_with_embeddings() -> SystemPipelineResult:
    """Run with embeddings but no rules."""
    pipeline = SystemPipeline(
        text_pipeline=_make_text_pipeline(),
        embedding_generator=_make_embedding_generator(),
    )
    return pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)


# ---------------------------------------------------------------------------
# run_scenario
# ---------------------------------------------------------------------------


class TestRunScenario:
    def test_runs_scenario(self) -> None:
        runner = SystemBenchmarkRunner()
        result = runner.run_scenario(_run_full, "full_pipeline")

        assert result.scenario_name == "full_pipeline"
        assert result.total_ms >= 0
        assert result.stages_executed > 0
        assert "text_processing" in result.stage_latencies
        assert result.result is not None

    def test_scenario_with_params(self) -> None:
        runner = SystemBenchmarkRunner()
        result = runner.run_scenario(
            _run_text_only,
            "text_only",
            scenario_params={"mode": "text_only", "embeddings": "disabled"},
        )

        assert result.scenario_params["mode"] == "text_only"
        assert result.stages_skipped > 0

    def test_frozen_result(self) -> None:
        runner = SystemBenchmarkRunner()
        result = runner.run_scenario(_run_text_only, "test")
        try:
            result.scenario_name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_compare_scenarios(self) -> None:
        runner = SystemBenchmarkRunner()
        report = runner.compare(
            {
                "full": _run_full,
                "text_only": _run_text_only,
            }
        )

        assert len(report.results) == 2
        names = {r.scenario_name for r in report.results}
        assert names == {"full", "text_only"}
        assert report.total_runs == 2
        assert report.total_ms > 0

    def test_compare_with_params(self) -> None:
        runner = SystemBenchmarkRunner()
        report = runner.compare(
            {
                "full": _run_full,
                "text_only": _run_text_only,
            },
            scenario_params={
                "full": {"embeddings": "enabled", "rules": "enabled"},
                "text_only": {"embeddings": "disabled", "rules": "disabled"},
            },
        )

        full_result = next(r for r in report.results if r.scenario_name == "full")
        assert full_result.scenario_params["embeddings"] == "enabled"

    def test_compare_empty(self) -> None:
        runner = SystemBenchmarkRunner()
        report = runner.compare({})
        assert len(report.results) == 0
        assert report.total_runs == 0

    def test_aggregated_metrics_present(self) -> None:
        runner = SystemBenchmarkRunner()
        report = runner.compare({"full": _run_full})

        assert "run_count" in report.aggregated
        assert "avg_total_pipeline_ms" in report.aggregated
        assert "total_artifacts" in report.aggregated

    def test_custom_config(self) -> None:
        config = SystemBenchmarkConfig(experiment_name="custom_exp", experiment_version="2.0")
        runner = SystemBenchmarkRunner(config=config)
        report = runner.compare({"test": _run_text_only})

        assert report.experiment_name == "custom_exp"
        assert report.experiment_version == "2.0"


# ---------------------------------------------------------------------------
# SystemBenchmarkReport serialization
# ---------------------------------------------------------------------------


class TestBenchmarkReportSerialization:
    def _make_report(self) -> SystemBenchmarkReport:
        runner = SystemBenchmarkRunner()
        return runner.compare(
            {
                "full": _run_full,
                "text_only": _run_text_only,
            }
        )

    def test_to_json(self) -> None:
        report = self._make_report()
        json_str = report.to_json()
        data = json.loads(json_str)

        assert data["experiment_name"] == "system_pipeline_benchmark"
        assert len(data["results"]) == 2
        assert "aggregated" in data
        assert "total_runs" in data

    def test_to_json_roundtrip(self) -> None:
        report = self._make_report()
        data1 = json.loads(report.to_json())
        data2 = json.loads(json.dumps(data1, indent=2, ensure_ascii=False))
        assert data1 == data2

    def test_to_csv(self) -> None:
        report = self._make_report()
        csv_str = report.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 scenarios
        assert "scenario_name" in lines[0]
        assert "text_processing_ms" in lines[0]

    def test_to_csv_empty(self) -> None:
        report = SystemBenchmarkReport(
            experiment_name="empty",
            experiment_version="1.0",
        )
        assert report.to_csv() == ""

    def test_save_json(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        report = self._make_report()
        path = tmp_path / "benchmark.json"
        report.save_json(path)
        data = json.loads(path.read_text())
        assert data["experiment_name"] == "system_pipeline_benchmark"

    def test_save_csv(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        report = self._make_report()
        path = tmp_path / "benchmark.csv"
        report.save_csv(path)
        content = path.read_text()
        assert "full" in content
        assert "text_only" in content

    def test_frozen(self) -> None:
        report = self._make_report()
        try:
            report.experiment_name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestBenchmarkReexport:
    def test_importable_from_pipeline_package(self) -> None:
        from talkex.pipeline import (
            ScenarioResult,
            SystemBenchmarkConfig,
            SystemBenchmarkReport,
            SystemBenchmarkRunner,
            compute_pipeline_metrics,
        )

        assert ScenarioResult is not None
        assert SystemBenchmarkConfig is not None
        assert SystemBenchmarkReport is not None
        assert SystemBenchmarkRunner is not None
        assert compute_pipeline_metrics is not None
