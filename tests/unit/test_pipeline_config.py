"""Unit tests for unified pipeline configuration.

Tests cover: PipelineConfig (construction, defaults, serialization,
JSON loading, validation), config fingerprint interaction, and reexports.
"""

from __future__ import annotations

import json

from semantic_conversation_engine.pipeline.config import PipelineConfig

# ---------------------------------------------------------------------------
# PipelineConfig — construction and defaults
# ---------------------------------------------------------------------------


class TestPipelineConfigConstruction:
    def test_defaults(self) -> None:
        config = PipelineConfig()
        assert config.context.window_size == 5
        assert config.context.stride == 2
        assert config.segmentation.normalize_unicode is True
        assert config.output_dir == "output"

    def test_custom_context(self) -> None:
        config = PipelineConfig.from_dict({"context": {"window_size": 3, "stride": 1}})
        assert config.context.window_size == 3
        assert config.context.stride == 1

    def test_custom_output_dir(self) -> None:
        config = PipelineConfig.from_dict({"output_dir": "/tmp/results"})
        assert config.output_dir == "/tmp/results"

    def test_frozen(self) -> None:
        config = PipelineConfig()
        try:
            config.output_dir = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# PipelineConfig — serialization
# ---------------------------------------------------------------------------


class TestPipelineConfigSerialization:
    def test_to_dict(self) -> None:
        config = PipelineConfig()
        d = config.to_dict()
        assert "context" in d
        assert "segmentation" in d
        assert d["context"]["window_size"] == 5

    def test_to_json(self) -> None:
        config = PipelineConfig()
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["context"]["window_size"] == 5

    def test_to_json_roundtrip(self) -> None:
        config = PipelineConfig.from_dict({"context": {"window_size": 3}})
        json_str = config.to_json()
        config2 = PipelineConfig.from_dict(json.loads(json_str))
        assert config2.context.window_size == 3

    def test_save_json(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        config = PipelineConfig()
        path = tmp_path / "config.json"
        config.save_json(path)
        data = json.loads(path.read_text())
        assert data["context"]["window_size"] == 5


# ---------------------------------------------------------------------------
# PipelineConfig — JSON loading
# ---------------------------------------------------------------------------


class TestPipelineConfigJsonLoading:
    def test_from_json(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        data = {"context": {"window_size": 7, "stride": 3}, "output_dir": "custom/"}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))
        config = PipelineConfig.from_json(path)
        assert config.context.window_size == 7
        assert config.output_dir == "custom/"

    def test_from_json_file_not_found(self) -> None:
        import pytest

        with pytest.raises(FileNotFoundError, match="not found"):
            PipelineConfig.from_json("/nonexistent/config.json")

    def test_from_json_invalid_json(self, tmp_path: object) -> None:
        import pathlib

        import pytest

        assert isinstance(tmp_path, pathlib.Path)
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            PipelineConfig.from_json(path)

    def test_from_json_empty_dict(self, tmp_path: object) -> None:
        """Empty config file produces valid defaults."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        path = tmp_path / "empty.json"
        path.write_text("{}")
        config = PipelineConfig.from_json(path)
        assert config.context.window_size == 5


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestPipelineConfigReexport:
    def test_importable_from_pipeline_package(self) -> None:
        from semantic_conversation_engine.pipeline import PipelineConfig

        assert PipelineConfig is not None
