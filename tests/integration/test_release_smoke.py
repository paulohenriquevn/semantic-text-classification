"""Release smoke tests — verify the system is installable and executable.

These tests validate the end-to-end operational flow that a user would
experience after installing the package. They cover:

    - Package version accessible
    - CLI entrypoint registered and functional
    - Pipeline execution from file
    - Benchmark execution
    - Config export and validation
    - Output persistence
    - Examples runnable
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from semantic_conversation_engine import __version__

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_SAMPLE_TRANSCRIPT = _FIXTURES_DIR / "sample_transcript.txt"


class TestPackageInstallation:
    """Verify the package is properly installed."""

    def test_version_accessible(self) -> None:
        assert __version__ == "0.1.0"

    def test_package_importable(self) -> None:
        import semantic_conversation_engine

        assert hasattr(semantic_conversation_engine, "__version__")

    def test_pipeline_package_importable(self) -> None:
        from semantic_conversation_engine.pipeline import (
            PipelineConfig,
            PipelineRunManifest,
            PipelineRunner,
            SystemBenchmarkRunner,
            SystemPipeline,
            SystemPipelineResult,
        )

        assert all(
            [
                PipelineConfig,
                PipelineRunManifest,
                PipelineRunner,
                SystemBenchmarkRunner,
                SystemPipeline,
                SystemPipelineResult,
            ]
        )


class TestCliEntrypoint:
    """Verify the CLI is registered and functional."""

    def test_sce_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "semantic_conversation_engine", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_sce_config_defaults(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "semantic_conversation_engine", "config"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["context"]["window_size"] == 5


class TestEndToEndExecution:
    """Verify full pipeline execution from file."""

    def test_pipeline_run_from_file(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)

        from semantic_conversation_engine.pipeline.runner import PipelineRunner

        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            rules_text=['keyword("billing")'],
            enable_embeddings=False,
        )

        assert summary.turns_count == 6
        assert summary.windows_count > 0
        assert summary.rule_executions_count > 0

        run_dir = PipelineRunner.save_outputs(summary, tmp_path)
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "summary.json").exists()

    def test_benchmark_run(self) -> None:
        from semantic_conversation_engine.pipeline.benchmark import SystemBenchmarkRunner
        from semantic_conversation_engine.pipeline.runner import PipelineRunner

        def _text_only() -> object:
            return PipelineRunner().run_file(_SAMPLE_TRANSCRIPT, enable_embeddings=False, enable_rules=False).result

        runner = SystemBenchmarkRunner()
        report = runner.compare({"text_only": _text_only})  # type: ignore[dict-item]

        assert report.total_runs == 1
        assert report.results[0].stages_executed >= 1

    def test_config_roundtrip(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)

        from semantic_conversation_engine.pipeline.config import PipelineConfig

        # Export
        config = PipelineConfig()
        path = tmp_path / "config.json"
        config.save_json(path)

        # Re-load
        config2 = PipelineConfig.from_json(path)
        assert config2.context.window_size == config.context.window_size
        assert config2.segmentation.normalize_unicode == config.segmentation.normalize_unicode


class TestExamplesRunnable:
    """Verify example scripts execute without errors."""

    @pytest.mark.parametrize("example", ["run_pipeline.py", "benchmark_pipeline.py"])
    def test_example_runs(self, example: str) -> None:
        example_path = Path(__file__).parent.parent.parent / "examples" / example
        if not example_path.exists():
            pytest.skip(f"Example {example} not found")

        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Example {example} failed: {result.stderr}"
