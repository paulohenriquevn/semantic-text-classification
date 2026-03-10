"""Unified pipeline configuration — loads and validates all stage configs.

Provides a single entry point for configuring the full SystemPipeline from
a JSON file, a dictionary, or individual config objects. Each stage's
config is optional and uses its own defaults when omitted.

Usage::

    # From JSON file
    config = PipelineConfig.from_json("pipeline_config.json")

    # From dict (e.g. parsed YAML or environment)
    config = PipelineConfig.from_dict({"context": {"window_size": 3}})

    # Programmatic
    config = PipelineConfig(
        context=ContextWindowConfig(window_size=3, stride=2),
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from talkex.context.config import ContextWindowConfig
from talkex.embeddings.config import EmbeddingModelConfig
from talkex.rules.config import RuleEngineConfig
from talkex.segmentation.config import SegmentationConfig


class EmbeddingConfig(BaseModel):
    """Embedding stage configuration for the unified pipeline config.

    Args:
        model: Embedding model configuration.
        dimensions: Vector dimensionality for null/test generators.
    """

    model_config = ConfigDict(frozen=True)

    model: EmbeddingModelConfig | None = None
    dimensions: int = 384


class PipelineConfig(BaseModel):
    """Unified configuration for the full SystemPipeline.

    Aggregates all stage-level configs into a single loadable object.
    Each section is optional — omitted sections use their stage defaults.

    Args:
        segmentation: Turn segmentation configuration.
        context: Context window builder configuration.
        embedding: Embedding generation configuration.
        rules: Rule engine configuration.
        output_dir: Base directory for pipeline outputs.
    """

    model_config = ConfigDict(frozen=True)

    segmentation: SegmentationConfig = SegmentationConfig()  # type: ignore[call-arg]
    context: ContextWindowConfig = ContextWindowConfig()  # type: ignore[call-arg]
    embedding: EmbeddingConfig = EmbeddingConfig()
    rules: RuleEngineConfig = RuleEngineConfig()
    output_dir: str = "output"

    @staticmethod
    def from_json(path: str | Path) -> PipelineConfig:
        """Load pipeline configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Validated PipelineConfig.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the JSON is malformed or validation fails.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            data = json.loads(config_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}") from e

        return PipelineConfig.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelineConfig:
        """Load pipeline configuration from a dictionary.

        Uses non-strict validation to accept JSON-serialized values
        (e.g. enum strings like "all" instead of RuleEvaluationMode.ALL).
        This follows the ADR-002 boundary deserialization pattern.

        Args:
            data: Configuration dictionary with optional stage sections.

        Returns:
            Validated PipelineConfig.
        """
        return PipelineConfig.model_validate(data, strict=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize configuration to formatted JSON string.

        Returns:
            JSON string with indentation.
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save_json(self, path: str | Path) -> None:
        """Save configuration as JSON file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_json())
