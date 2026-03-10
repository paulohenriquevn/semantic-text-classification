"""Unit tests for pipeline run manifest and artifact lineage.

Tests cover: PipelineRunManifest (creation, serialization, immutability),
compute_config_fingerprint (determinism, sensitivity), lineage propagation
via SystemPipelineResult.manifest, and reexports.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from semantic_conversation_engine.pipeline.manifest import (
    PipelineRunManifest,
    compute_config_fingerprint,
)

# ---------------------------------------------------------------------------
# PipelineRunManifest — construction
# ---------------------------------------------------------------------------


class TestPipelineRunManifestConstruction:
    def test_create_generates_unique_run_id(self) -> None:
        m1 = PipelineRunManifest.create()
        m2 = PipelineRunManifest.create()
        assert m1.run_id != m2.run_id

    def test_create_run_id_format(self) -> None:
        m = PipelineRunManifest.create()
        assert m.run_id.startswith("run_")
        assert len(m.run_id) == 16  # "run_" + 12 hex chars

    def test_create_sets_utc_timestamp(self) -> None:
        before = datetime.now(tz=UTC)
        m = PipelineRunManifest.create()
        after = datetime.now(tz=UTC)
        assert before <= m.timestamp <= after

    def test_create_with_component_versions(self) -> None:
        versions = {"embedding_model": "e5-base-v2/1.0", "classifier": "LogisticClassifier"}
        m = PipelineRunManifest.create(component_versions=versions)
        assert m.component_versions == versions

    def test_create_with_config_fingerprint(self) -> None:
        m = PipelineRunManifest.create(config_fingerprint="sha256:abc123")
        assert m.config_fingerprint == "sha256:abc123"

    def test_create_with_metadata(self) -> None:
        m = PipelineRunManifest.create(metadata={"dataset_id": "ds_001"})
        assert m.metadata["dataset_id"] == "ds_001"

    def test_create_defaults(self) -> None:
        m = PipelineRunManifest.create()
        assert m.component_versions == {}
        assert m.config_fingerprint == ""
        assert m.pipeline_version == "1.0"
        assert m.metadata == {}

    def test_direct_construction(self) -> None:
        ts = datetime(2025, 6, 1, tzinfo=UTC)
        m = PipelineRunManifest(
            run_id="run_test123456",
            timestamp=ts,
            component_versions={"embedding_model": "test/1.0"},
            config_fingerprint="sha256:abc",
            pipeline_version="2.0",
            metadata={"key": "value"},
        )
        assert m.run_id == "run_test123456"
        assert m.timestamp == ts
        assert m.pipeline_version == "2.0"


# ---------------------------------------------------------------------------
# PipelineRunManifest — immutability
# ---------------------------------------------------------------------------


class TestPipelineRunManifestImmutability:
    def test_frozen(self) -> None:
        m = PipelineRunManifest.create()
        try:
            m.run_id = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# PipelineRunManifest — serialization
# ---------------------------------------------------------------------------


class TestPipelineRunManifestSerialization:
    def test_to_dict(self) -> None:
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        m = PipelineRunManifest(
            run_id="run_aabbccddee00",
            timestamp=ts,
            component_versions={"embedding_model": "null/1.0"},
            config_fingerprint="sha256:abc",
        )
        d = m.to_dict()
        assert d["run_id"] == "run_aabbccddee00"
        assert d["timestamp"] == "2025-06-01T12:00:00+00:00"
        assert d["component_versions"]["embedding_model"] == "null/1.0"
        assert d["config_fingerprint"] == "sha256:abc"

    def test_to_json(self) -> None:
        m = PipelineRunManifest.create(
            component_versions={"classifier": "stub"},
        )
        json_str = m.to_json()
        data = json.loads(json_str)
        assert data["run_id"] == m.run_id
        assert "classifier" in data["component_versions"]

    def test_to_json_roundtrip(self) -> None:
        m = PipelineRunManifest.create(
            component_versions={"a": "1", "b": "2"},
            metadata={"x": "y"},
        )
        data1 = json.loads(m.to_json())
        data2 = json.loads(json.dumps(data1, indent=2, ensure_ascii=False))
        assert data1 == data2

    def test_save_json(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        m = PipelineRunManifest.create()
        path = tmp_path / "manifest.json"
        m.save_json(path)
        data = json.loads(path.read_text())
        assert data["run_id"] == m.run_id


# ---------------------------------------------------------------------------
# compute_config_fingerprint
# ---------------------------------------------------------------------------


class TestComputeConfigFingerprint:
    def test_deterministic(self) -> None:
        config = {"window_size": 3, "stride": 2, "model": "e5"}
        fp1 = compute_config_fingerprint(config)
        fp2 = compute_config_fingerprint(config)
        assert fp1 == fp2

    def test_sha256_prefix(self) -> None:
        fp = compute_config_fingerprint({"key": "value"})
        assert fp.startswith("sha256:")

    def test_key_order_independent(self) -> None:
        fp1 = compute_config_fingerprint({"a": 1, "b": 2})
        fp2 = compute_config_fingerprint({"b": 2, "a": 1})
        assert fp1 == fp2

    def test_sensitive_to_value_changes(self) -> None:
        fp1 = compute_config_fingerprint({"window_size": 3})
        fp2 = compute_config_fingerprint({"window_size": 5})
        assert fp1 != fp2

    def test_sensitive_to_key_changes(self) -> None:
        fp1 = compute_config_fingerprint({"window_size": 3})
        fp2 = compute_config_fingerprint({"stride": 3})
        assert fp1 != fp2

    def test_empty_config(self) -> None:
        fp = compute_config_fingerprint({})
        assert fp.startswith("sha256:")
        assert len(fp) > len("sha256:")


# ---------------------------------------------------------------------------
# Manifest integration with SystemPipelineResult
# ---------------------------------------------------------------------------


class TestManifestInSystemPipelineResult:
    def test_result_has_manifest_field(self) -> None:
        """SystemPipelineResult accepts a manifest field."""
        from semantic_conversation_engine.context.builder import SlidingWindowBuilder
        from semantic_conversation_engine.context.config import ContextWindowConfig
        from semantic_conversation_engine.ingestion.enums import SourceFormat
        from semantic_conversation_engine.ingestion.inputs import TranscriptInput
        from semantic_conversation_engine.models.enums import Channel
        from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
        from semantic_conversation_engine.pipeline.system_pipeline import SystemPipeline
        from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

        transcript = TranscriptInput(
            conversation_id="conv_manifest",
            raw_text="CUSTOMER: Hello\nAGENT: Hi there",
            source_format=SourceFormat.LABELED,
            channel=Channel.VOICE,
        )
        pipeline = SystemPipeline(
            text_pipeline=TextProcessingPipeline(
                segmenter=TurnSegmenter(),
                context_builder=SlidingWindowBuilder(),
            ),
        )
        result = pipeline.run(transcript, context_config=ContextWindowConfig(window_size=2, stride=1))

        assert result.manifest is not None
        assert result.manifest.run_id.startswith("run_")
        assert result.manifest.timestamp is not None
        assert "text_pipeline" in result.manifest.component_versions

    def test_manifest_tracks_embedding_model(self) -> None:
        """Manifest records embedding model version when generator is provided."""
        from semantic_conversation_engine.context.builder import SlidingWindowBuilder
        from semantic_conversation_engine.context.config import ContextWindowConfig
        from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
        from semantic_conversation_engine.embeddings.generator import NullEmbeddingGenerator
        from semantic_conversation_engine.embeddings.preprocessing import PreprocessingConfig
        from semantic_conversation_engine.ingestion.enums import SourceFormat
        from semantic_conversation_engine.ingestion.inputs import TranscriptInput
        from semantic_conversation_engine.models.enums import Channel
        from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
        from semantic_conversation_engine.pipeline.system_pipeline import SystemPipeline
        from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

        transcript = TranscriptInput(
            conversation_id="conv_emb_manifest",
            raw_text="CUSTOMER: Test\nAGENT: Response",
            source_format=SourceFormat.LABELED,
            channel=Channel.VOICE,
        )
        pipeline = SystemPipeline(
            text_pipeline=TextProcessingPipeline(
                segmenter=TurnSegmenter(),
                context_builder=SlidingWindowBuilder(),
            ),
            embedding_generator=NullEmbeddingGenerator(
                model_config=EmbeddingModelConfig(model_name="e5-test", model_version="2.0"),
                preprocessing_config=PreprocessingConfig(),
                dimensions=64,
            ),
        )
        result = pipeline.run(transcript, context_config=ContextWindowConfig(window_size=2, stride=1))

        assert result.manifest is not None
        assert result.manifest.component_versions["embedding_model"] == "e5-test/2.0"

    def test_manifest_config_fingerprint_changes_with_config(self) -> None:
        """Different configs produce different fingerprints."""
        from semantic_conversation_engine.context.builder import SlidingWindowBuilder
        from semantic_conversation_engine.context.config import ContextWindowConfig
        from semantic_conversation_engine.ingestion.enums import SourceFormat
        from semantic_conversation_engine.ingestion.inputs import TranscriptInput
        from semantic_conversation_engine.models.enums import Channel
        from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
        from semantic_conversation_engine.pipeline.system_pipeline import SystemPipeline
        from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

        transcript = TranscriptInput(
            conversation_id="conv_fp",
            raw_text="CUSTOMER: Test\nAGENT: Response",
            source_format=SourceFormat.LABELED,
            channel=Channel.VOICE,
        )
        pipeline = SystemPipeline(
            text_pipeline=TextProcessingPipeline(
                segmenter=TurnSegmenter(),
                context_builder=SlidingWindowBuilder(),
            ),
        )

        r1 = pipeline.run(transcript, context_config=ContextWindowConfig(window_size=2, stride=1))
        r2 = pipeline.run(transcript, context_config=ContextWindowConfig(window_size=3, stride=1))

        assert r1.manifest is not None
        assert r2.manifest is not None
        assert r1.manifest.config_fingerprint != r2.manifest.config_fingerprint

    def test_each_run_gets_unique_run_id(self) -> None:
        """Two pipeline runs produce different run_ids."""
        from semantic_conversation_engine.context.builder import SlidingWindowBuilder
        from semantic_conversation_engine.context.config import ContextWindowConfig
        from semantic_conversation_engine.ingestion.enums import SourceFormat
        from semantic_conversation_engine.ingestion.inputs import TranscriptInput
        from semantic_conversation_engine.models.enums import Channel
        from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
        from semantic_conversation_engine.pipeline.system_pipeline import SystemPipeline
        from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

        transcript = TranscriptInput(
            conversation_id="conv_unique",
            raw_text="CUSTOMER: Test\nAGENT: Response",
            source_format=SourceFormat.LABELED,
            channel=Channel.VOICE,
        )
        pipeline = SystemPipeline(
            text_pipeline=TextProcessingPipeline(
                segmenter=TurnSegmenter(),
                context_builder=SlidingWindowBuilder(),
            ),
        )

        r1 = pipeline.run(transcript, context_config=ContextWindowConfig(window_size=2, stride=1))
        r2 = pipeline.run(transcript, context_config=ContextWindowConfig(window_size=2, stride=1))

        assert r1.manifest is not None
        assert r2.manifest is not None
        assert r1.manifest.run_id != r2.manifest.run_id


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestManifestReexport:
    def test_importable_from_pipeline_package(self) -> None:
        from semantic_conversation_engine.pipeline import (
            PipelineRunManifest,
            compute_config_fingerprint,
        )

        assert PipelineRunManifest is not None
        assert compute_config_fingerprint is not None
