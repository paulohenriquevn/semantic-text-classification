"""Smoke tests for the pipeline CLI entrypoint.

Tests cover: CLI group, version command, config export/validate,
run command execution, and __main__.py module entry.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from semantic_conversation_engine.pipeline.cli import cli

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_SAMPLE_TRANSCRIPT = _FIXTURES_DIR / "sample_transcript.txt"


# ---------------------------------------------------------------------------
# CLI — version
# ---------------------------------------------------------------------------


class TestCliVersion:
    def test_version_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "semantic-conversation-engine" in result.output
        assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# CLI — config
# ---------------------------------------------------------------------------


class TestCliConfig:
    def test_config_defaults(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["context"]["window_size"] == 5

    def test_config_export(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        out = tmp_path / "template.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--export", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "context" in data

    def test_config_validate_valid(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        path = tmp_path / "valid.json"
        path.write_text('{"context": {"window_size": 3}}')
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--validate", str(path)])
        assert result.exit_code == 0
        assert "Config valid" in result.output

    def test_config_validate_invalid(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        path = tmp_path / "invalid.json"
        path.write_text("{bad json")
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--validate", str(path)])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# CLI — run
# ---------------------------------------------------------------------------


class TestCliRun:
    def test_run_transcript(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(_SAMPLE_TRANSCRIPT),
                "--no-embeddings",
                "--no-rules",
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert "Run ID:" in result.output
        assert "Turns:" in result.output
        assert "Output:" in result.output

    def test_run_with_rules(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(_SAMPLE_TRANSCRIPT),
                "--no-embeddings",
                "--rule",
                'keyword("billing")',
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert "Rule executions:" in result.output

    def test_run_creates_output_files(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(_SAMPLE_TRANSCRIPT),
                "--no-embeddings",
                "--no-rules",
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0

        # Find the run directory
        run_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "manifest.json").exists()
        assert (run_dirs[0] / "summary.json").exists()

    def test_run_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "/nonexistent/file.txt"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# __main__.py
# ---------------------------------------------------------------------------


class TestMainModule:
    def test_main_module_exists(self) -> None:
        """The __main__.py module exists and contains main."""
        from pathlib import Path

        main_path = Path(__file__).parent.parent.parent / "src" / "semantic_conversation_engine" / "__main__.py"
        assert main_path.exists()
        content = main_path.read_text()
        assert "from semantic_conversation_engine.pipeline.cli import main" in content
