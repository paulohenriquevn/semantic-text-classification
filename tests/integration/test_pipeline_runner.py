"""Integration tests for the operational pipeline runner.

End-to-end: config loading → pipeline construction → transcript execution
→ output persistence → manifest/summary serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
from semantic_conversation_engine.pipeline.config import EmbeddingConfig, PipelineConfig
from semantic_conversation_engine.pipeline.runner import PipelineRunner

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_SAMPLE_TRANSCRIPT = _FIXTURES_DIR / "sample_transcript.txt"


class TestPipelineRunnerIntegration:
    """End-to-end pipeline runner integration tests."""

    def test_full_run_with_config(self, tmp_path: object) -> None:
        """Full pipeline run with custom config, outputs, and manifest."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)

        config = PipelineConfig(
            context={"window_size": 3, "stride": 2},  # type: ignore[arg-type]
            embedding=EmbeddingConfig(
                model=EmbeddingModelConfig(model_name="null-test", model_version="1.0"),
                dimensions=32,
            ),
            output_dir=str(tmp_path),
        )
        runner = PipelineRunner(config=config)
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            rules_text=['keyword("billing")', 'keyword("cancel")'],
        )

        assert summary.turns_count == 6
        assert summary.windows_count > 0
        assert summary.embeddings_count > 0
        assert summary.rule_executions_count > 0

        # Persist and verify
        run_dir = PipelineRunner.save_outputs(summary, tmp_path)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["run_id"] == summary.run_id
        assert manifest["component_versions"]["embedding_model"] == "null-test/1.0"

    def test_text_only_run(self) -> None:
        """Minimal pipeline run with text processing only."""
        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            enable_embeddings=False,
            enable_rules=False,
        )

        assert summary.turns_count == 6
        assert summary.embeddings_count == 0
        assert summary.predictions_count == 0
        assert summary.rule_executions_count == 0
        assert summary.stages_skipped > 0

    def test_config_from_json_file(self, tmp_path: object) -> None:
        """Load config from JSON file and run pipeline."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)

        config_data = {
            "context": {"window_size": 4, "stride": 2},
            "output_dir": str(tmp_path),
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        config = PipelineConfig.from_json(config_path)
        runner = PipelineRunner(config=config)
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            enable_embeddings=False,
            enable_rules=False,
        )

        assert summary.turns_count == 6
        assert summary.windows_count > 0

    def test_multiple_runs_unique_ids(self) -> None:
        """Multiple runs produce unique run_ids and timestamps."""
        runner = PipelineRunner()

        s1 = runner.run_file(_SAMPLE_TRANSCRIPT, enable_embeddings=False, enable_rules=False)
        s2 = runner.run_file(_SAMPLE_TRANSCRIPT, enable_embeddings=False, enable_rules=False)

        assert s1.run_id != s2.run_id
        assert s1.result.manifest is not None
        assert s2.result.manifest is not None
        assert s1.result.manifest.timestamp != s2.result.manifest.timestamp

    def test_output_directory_structure(self, tmp_path: object) -> None:
        """Output directory follows conventions: {output_dir}/{run_id}/."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)

        runner = PipelineRunner()
        summary = runner.run_file(
            _SAMPLE_TRANSCRIPT,
            enable_embeddings=False,
            enable_rules=False,
        )

        run_dir = PipelineRunner.save_outputs(summary, tmp_path)

        # Verify structure: {tmp_path}/{run_id}/
        assert run_dir.parent == tmp_path
        assert run_dir.name == summary.run_id

        # Verify contents
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "summary.json").exists()

        summary_data = json.loads((run_dir / "summary.json").read_text())
        assert summary_data["turns_count"] == 6
        assert summary_data["stages_executed"] > 0
