"""Unit tests for the operational pipeline runner.

Tests cover: PipelineRunner (build_pipeline, run_file, save_outputs),
RunSummary structure, output persistence, and reexports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from talkex.pipeline.config import PipelineConfig
from talkex.pipeline.runner import PipelineRunner

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_SAMPLE_TRANSCRIPT = _FIXTURES_DIR / "sample_transcript.txt"


# ---------------------------------------------------------------------------
# PipelineRunner — build_pipeline
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    def test_builds_text_only(self) -> None:
        runner = PipelineRunner()
        pipeline = runner.build_pipeline(enable_embeddings=False, enable_rules=False)
        assert pipeline._text_pipeline is not None
        assert pipeline._embedding_generator is None
        assert pipeline._rule_evaluator is None

    def test_builds_with_rules(self) -> None:
        runner = PipelineRunner()
        pipeline = runner.build_pipeline(enable_embeddings=False, enable_rules=True)
        assert pipeline._rule_evaluator is not None

    def test_builds_with_embedding_config(self) -> None:
        from talkex.embeddings.config import EmbeddingModelConfig
        from talkex.pipeline.config import EmbeddingConfig

        config = PipelineConfig(
            embedding=EmbeddingConfig(
                model=EmbeddingModelConfig(model_name="test", model_version="1.0"),
                dimensions=64,
            ),
        )
        runner = PipelineRunner(config=config)
        pipeline = runner.build_pipeline(enable_embeddings=True)
        assert pipeline._embedding_generator is not None


# ---------------------------------------------------------------------------
# PipelineRunner — run_file
# ---------------------------------------------------------------------------


class TestRunFile:
    def test_runs_sample_transcript(self) -> None:
        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            enable_embeddings=False,
            enable_rules=False,
        )
        assert summary.turns_count == 6
        assert summary.windows_count > 0
        assert summary.run_id.startswith("run_")
        assert summary.total_ms >= 0

    def test_file_not_found(self) -> None:
        runner = PipelineRunner()
        with pytest.raises(FileNotFoundError, match="not found"):
            runner.run_file("/nonexistent/file.txt")

    def test_with_rules(self) -> None:
        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            rules_text=['keyword("billing")'],
            enable_embeddings=False,
        )
        assert summary.rule_executions_count > 0

    def test_custom_conversation_id(self) -> None:
        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            conversation_id="conv_custom_123",
            enable_embeddings=False,
            enable_rules=False,
        )
        assert summary.result.pipeline_result.conversation.conversation_id == "conv_custom_123"

    def test_run_summary_has_result(self) -> None:
        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            enable_embeddings=False,
            enable_rules=False,
        )
        assert summary.result is not None
        assert summary.result.manifest is not None


# ---------------------------------------------------------------------------
# PipelineRunner — save_outputs
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    def test_saves_manifest_and_summary(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            enable_embeddings=False,
            enable_rules=False,
        )

        run_dir = PipelineRunner.save_outputs(summary, tmp_path)

        assert run_dir.exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "summary.json").exists()

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["run_id"] == summary.run_id

        summary_data = json.loads((run_dir / "summary.json").read_text())
        assert summary_data["turns_count"] == summary.turns_count
        assert summary_data["run_id"] == summary.run_id

    def test_output_dir_created(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            enable_embeddings=False,
            enable_rules=False,
        )

        nested = tmp_path / "deep" / "nested"
        run_dir = PipelineRunner.save_outputs(summary, nested)
        assert run_dir.exists()


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestRunnerReexport:
    def test_importable_from_pipeline_package(self) -> None:
        from talkex.pipeline import PipelineRunner, RunSummary

        assert PipelineRunner is not None
        assert RunSummary is not None
