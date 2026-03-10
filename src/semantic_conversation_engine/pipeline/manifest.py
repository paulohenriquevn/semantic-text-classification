"""Pipeline run manifest — execution identity and version tracking.

A PipelineRunManifest captures the complete execution context for a single
SystemPipeline.run() invocation: unique run_id, timestamp, component versions,
and configuration fingerprint.

This enables:
    - **Artifact lineage**: every artifact can be traced back to its pipeline run
    - **Reproducibility**: version manifest records exactly which models/rules ran
    - **Auditability**: config fingerprint detects configuration drift across runs

The manifest is attached to SystemPipelineResult and propagated to artifact
metadata during pipeline execution.

Usage::

    manifest = PipelineRunManifest.create(
        component_versions={"embedding_model": "e5-base-v2/1.0"},
        config_fingerprint="sha256:abc123...",
    )
    # manifest.run_id → "run_a1b2c3d4e5f6"
    # manifest.timestamp → datetime(2025, ...)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineRunManifest:
    """Execution identity and version manifest for a pipeline run.

    Args:
        run_id: Unique identifier for this pipeline execution.
            Format: run_<hex12>.
        timestamp: When the pipeline run started (UTC).
        component_versions: Mapping of component name to version string.
            Records which embedding model, classifier, rule set, etc.
            were active during this run.
        config_fingerprint: Hash of the effective pipeline configuration.
            Enables detecting configuration drift between runs.
        pipeline_version: Version of the pipeline code itself.
        metadata: Additional run-level context (dataset_id, experiment, etc.).
    """

    run_id: str
    timestamp: datetime
    component_versions: dict[str, str] = field(default_factory=dict)
    config_fingerprint: str = ""
    pipeline_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        *,
        component_versions: dict[str, str] | None = None,
        config_fingerprint: str = "",
        pipeline_version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> PipelineRunManifest:
        """Create a new manifest with auto-generated run_id and timestamp.

        Factory method that assigns a unique run_id and captures the
        current UTC timestamp.

        Args:
            component_versions: Mapping of component name to version string.
            config_fingerprint: Hash of the effective pipeline configuration.
            pipeline_version: Version of the pipeline code.
            metadata: Additional run-level context.

        Returns:
            A new PipelineRunManifest with unique identity.
        """
        return PipelineRunManifest(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(tz=UTC),
            component_versions=component_versions or {},
            config_fingerprint=config_fingerprint,
            pipeline_version=pipeline_version,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a plain dictionary.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "component_versions": dict(self.component_versions),
            "config_fingerprint": self.config_fingerprint,
            "pipeline_version": self.pipeline_version,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize manifest to JSON string.

        Returns:
            Formatted JSON string.
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save_json(self, path: str | Path) -> None:
        """Save manifest as JSON file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_json())


def compute_config_fingerprint(config: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 fingerprint of a configuration dict.

    Produces a stable hash by sorting keys and using consistent
    JSON serialization. Two equivalent configurations always produce
    the same fingerprint.

    Args:
        config: Configuration dictionary to fingerprint.

    Returns:
        SHA-256 hex digest prefixed with 'sha256:'.
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"sha256:{digest}"
