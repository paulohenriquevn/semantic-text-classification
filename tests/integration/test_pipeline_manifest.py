"""Integration tests for pipeline run manifest and artifact lineage.

End-to-end: full pipeline → manifest creation → lineage tracking
→ version recording → benchmark compatibility.

Verifies that manifest is consistently produced across different
pipeline configurations and integrates with the benchmark runner.
"""

from __future__ import annotations

import json

from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
from semantic_conversation_engine.embeddings.generator import NullEmbeddingGenerator
from semantic_conversation_engine.embeddings.preprocessing import PreprocessingConfig
from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.enums import Channel
from semantic_conversation_engine.pipeline.benchmark import SystemBenchmarkRunner
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.pipeline.system_pipeline import (
    SystemPipeline,
    SystemPipelineResult,
)
from semantic_conversation_engine.rules.compiler import SimpleRuleCompiler
from semantic_conversation_engine.rules.evaluator import SimpleRuleEvaluator
from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT = TranscriptInput(
    conversation_id="conv_manifest_integ",
    raw_text="""\
CUSTOMER: I have a billing issue with my credit card.
AGENT: I can help you with that. What is the issue?
CUSTOMER: I was charged twice for the same order.
AGENT: Let me look into that for you right away.
""",
    source_format=SourceFormat.LABELED,
    channel=Channel.VOICE,
)

_CONTEXT_CONFIG = ContextWindowConfig(window_size=3, stride=2)

_COMPILER = SimpleRuleCompiler()
_RULES = [
    _COMPILER.compile('keyword("billing")', "rule_billing", "billing_issue"),
]


def _make_text_pipeline() -> TextProcessingPipeline:
    return TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    )


def _make_emb_gen() -> NullEmbeddingGenerator:
    return NullEmbeddingGenerator(
        model_config=EmbeddingModelConfig(model_name="null-manifest", model_version="1.0"),
        preprocessing_config=PreprocessingConfig(),
        dimensions=64,
    )


def _scenario_full() -> SystemPipelineResult:
    pipeline = SystemPipeline(
        text_pipeline=_make_text_pipeline(),
        embedding_generator=_make_emb_gen(),
        rule_evaluator=SimpleRuleEvaluator(),
    )
    return pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG, rules=_RULES)


def _scenario_text_only() -> SystemPipelineResult:
    pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
    return pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)


class TestPipelineManifestIntegration:
    """End-to-end manifest and lineage integration tests."""

    def test_full_pipeline_has_manifest(self) -> None:
        """Full pipeline run produces a manifest with component versions."""
        result = _scenario_full()

        assert result.manifest is not None
        assert result.manifest.run_id.startswith("run_")
        assert "text_pipeline" in result.manifest.component_versions
        assert "embedding_model" in result.manifest.component_versions
        assert result.manifest.component_versions["embedding_model"] == "null-manifest/1.0"
        assert "rule_evaluator" in result.manifest.component_versions

    def test_text_only_pipeline_has_manifest(self) -> None:
        """Text-only pipeline produces manifest with minimal components."""
        result = _scenario_text_only()

        assert result.manifest is not None
        assert "text_pipeline" in result.manifest.component_versions
        assert "embedding_model" not in result.manifest.component_versions
        assert "rule_evaluator" not in result.manifest.component_versions

    def test_different_configs_produce_different_fingerprints(self) -> None:
        """Config fingerprint changes when pipeline configuration differs."""
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())

        r1 = pipeline.run(_TRANSCRIPT, context_config=ContextWindowConfig(window_size=2, stride=1))
        r2 = pipeline.run(_TRANSCRIPT, context_config=ContextWindowConfig(window_size=3, stride=2))

        assert r1.manifest is not None
        assert r2.manifest is not None
        assert r1.manifest.config_fingerprint != r2.manifest.config_fingerprint

    def test_same_config_produces_same_fingerprint(self) -> None:
        """Same configuration produces identical fingerprints across runs."""
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())

        r1 = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)
        r2 = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        assert r1.manifest is not None
        assert r2.manifest is not None
        assert r1.manifest.config_fingerprint == r2.manifest.config_fingerprint
        # But run_ids differ
        assert r1.manifest.run_id != r2.manifest.run_id

    def test_manifest_serializable_in_benchmark_report(self) -> None:
        """Manifest data is accessible through benchmark scenario results."""
        runner = SystemBenchmarkRunner()
        report = runner.compare(
            {
                "full": _scenario_full,
                "text_only": _scenario_text_only,
            }
        )

        for scenario_result in report.results:
            assert scenario_result.result is not None
            assert scenario_result.result.manifest is not None
            assert scenario_result.result.manifest.run_id.startswith("run_")

    def test_manifest_json_serialization(self) -> None:
        """Manifest serializes to valid JSON with all fields."""
        result = _scenario_full()
        assert result.manifest is not None

        json_str = result.manifest.to_json()
        data = json.loads(json_str)

        assert data["run_id"] == result.manifest.run_id
        assert "timestamp" in data
        assert "component_versions" in data
        assert "config_fingerprint" in data
        assert data["component_versions"]["embedding_model"] == "null-manifest/1.0"

    def test_manifest_across_benchmark_scenarios(self) -> None:
        """Each benchmark scenario has its own unique manifest."""
        runner = SystemBenchmarkRunner()
        report = runner.compare(
            {
                "full": _scenario_full,
                "text_only": _scenario_text_only,
            }
        )

        run_ids = set()
        for scenario_result in report.results:
            assert scenario_result.result is not None
            assert scenario_result.result.manifest is not None
            run_ids.add(scenario_result.result.manifest.run_id)

        # Each scenario has a unique run_id
        assert len(run_ids) == 2
