"""Integration tests for system pipeline benchmark.

End-to-end: full pipeline → benchmark runner → scenario comparison
→ metrics aggregation → JSON/CSV serialization.

Compares full vs partial pipeline configurations with real
implementations (NullEmbeddingGenerator, InMemoryBM25Index,
SimpleRuleCompiler/Evaluator).
"""

import json

from semantic_conversation_engine.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
from semantic_conversation_engine.embeddings.generator import NullEmbeddingGenerator
from semantic_conversation_engine.embeddings.preprocessing import PreprocessingConfig
from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.enums import Channel
from semantic_conversation_engine.pipeline.benchmark import (
    SystemBenchmarkRunner,
)
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.pipeline.system_pipeline import (
    SystemPipeline,
    SystemPipelineResult,
)
from semantic_conversation_engine.retrieval.bm25 import InMemoryBM25Index
from semantic_conversation_engine.retrieval.config import VectorIndexConfig
from semantic_conversation_engine.retrieval.vector_index import InMemoryVectorIndex
from semantic_conversation_engine.rules.compiler import SimpleRuleCompiler
from semantic_conversation_engine.rules.config import (
    EvidencePolicy,
    RuleEngineConfig,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)
from semantic_conversation_engine.rules.evaluator import SimpleRuleEvaluator
from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT = TranscriptInput(
    conversation_id="conv_bench_integration",
    raw_text="""\
CUSTOMER: I have a billing issue with my credit card.
AGENT: I can help you with that. What is the issue?
CUSTOMER: I was charged twice for the same order. Can I get a refund?
AGENT: Let me look into that for you right away.
CUSTOMER: Also, I want to cancel my subscription.
AGENT: I understand. Let me process both requests.
""",
    source_format=SourceFormat.LABELED,
    channel=Channel.VOICE,
)

_CONTEXT_CONFIG = ContextWindowConfig(window_size=3, stride=2)

_COMPILER = SimpleRuleCompiler()
_RULES = [
    _COMPILER.compile('keyword("billing")', "rule_billing", "billing_issue"),
    _COMPILER.compile('keyword("cancel")', "rule_cancel", "cancel_intent"),
    _COMPILER.compile('keyword("refund")', "rule_refund", "refund_request"),
]

_RULE_CONFIG = RuleEngineConfig(
    evaluation_mode=RuleEvaluationMode.ALL,
    evidence_policy=EvidencePolicy.ALWAYS,
    short_circuit_policy=ShortCircuitPolicy.DECLARATION,
)


class _StubClassifier:
    """Stub classifier for benchmark integration tests."""

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        for inp in inputs:
            scores = []
            if "billing" in inp.text.lower():
                scores.append(LabelScore(label="billing", score=0.9, confidence=0.9, threshold=0.5))
            if not scores:
                scores.append(LabelScore(label="other", score=0.3, confidence=0.3, threshold=0.5))
            results.append(
                ClassificationResult(
                    source_id=inp.source_id,
                    source_type=inp.source_type,
                    label_scores=scores,
                    model_name="stub",
                    model_version="1.0",
                )
            )
        return results


def _make_text_pipeline() -> TextProcessingPipeline:
    return TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    )


def _make_emb_gen() -> NullEmbeddingGenerator:
    return NullEmbeddingGenerator(
        model_config=EmbeddingModelConfig(model_name="null-bench", model_version="1.0"),
        preprocessing_config=PreprocessingConfig(),
        dimensions=64,
    )


def _scenario_full() -> SystemPipelineResult:
    """Full pipeline: text + embeddings + indexes + classification + rules + analytics."""
    pipeline = SystemPipeline(
        text_pipeline=_make_text_pipeline(),
        embedding_generator=_make_emb_gen(),
        vector_index=InMemoryVectorIndex(config=VectorIndexConfig(dimensions=64)),
        lexical_index=InMemoryBM25Index(),
        classifier=_StubClassifier(),
        rule_evaluator=SimpleRuleEvaluator(),
    )
    return pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG, rules=_RULES, rule_config=_RULE_CONFIG)


def _scenario_no_rules() -> SystemPipelineResult:
    """Pipeline without rules."""
    pipeline = SystemPipeline(
        text_pipeline=_make_text_pipeline(),
        embedding_generator=_make_emb_gen(),
        classifier=_StubClassifier(),
    )
    return pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)


def _scenario_text_only() -> SystemPipelineResult:
    """Text processing only."""
    pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
    return pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)


def _scenario_with_embeddings() -> SystemPipelineResult:
    """Text + embeddings only."""
    pipeline = SystemPipeline(
        text_pipeline=_make_text_pipeline(),
        embedding_generator=_make_emb_gen(),
    )
    return pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)


class TestSystemBenchmarkIntegration:
    """End-to-end benchmark integration tests."""

    def test_compare_full_vs_partial(self) -> None:
        """Benchmark comparison between full and partial pipeline configurations."""
        runner = SystemBenchmarkRunner()
        report = runner.compare(
            {
                "full": _scenario_full,
                "no_rules": _scenario_no_rules,
                "text_only": _scenario_text_only,
                "with_embeddings": _scenario_with_embeddings,
            },
            scenario_params={
                "full": {"embeddings": "yes", "rules": "yes", "indexes": "yes", "classification": "yes"},
                "no_rules": {"embeddings": "yes", "rules": "no", "indexes": "no", "classification": "yes"},
                "text_only": {"embeddings": "no", "rules": "no", "indexes": "no", "classification": "no"},
                "with_embeddings": {"embeddings": "yes", "rules": "no", "indexes": "no", "classification": "no"},
            },
        )

        assert len(report.results) == 4
        assert report.total_runs == 4

        # Full pipeline should execute more stages
        full = next(r for r in report.results if r.scenario_name == "full")
        text_only = next(r for r in report.results if r.scenario_name == "text_only")
        assert full.stages_executed > text_only.stages_executed
        assert full.stages_skipped < text_only.stages_skipped

    def test_aggregated_metrics_across_scenarios(self) -> None:
        """Aggregated metrics reflect cross-scenario averages."""
        runner = SystemBenchmarkRunner()
        report = runner.compare(
            {
                "full": _scenario_full,
                "text_only": _scenario_text_only,
            }
        )

        agg = report.aggregated
        assert agg["run_count"] == 2
        assert agg["avg_total_pipeline_ms"] > 0
        assert "text_processing" in agg["avg_stage_latency_ms"]
        assert agg["total_artifacts"]["turns"] > 0

    def test_per_stage_latencies_available(self) -> None:
        """Each scenario has per-stage latency breakdowns."""
        runner = SystemBenchmarkRunner()
        report = runner.compare({"full": _scenario_full})

        full = report.results[0]
        assert "text_processing" in full.stage_latencies
        assert "embedding" in full.stage_latencies
        assert all(v >= 0 for v in full.stage_latencies.values())

    def test_json_serialization(self) -> None:
        """Benchmark report serializes to valid JSON."""
        runner = SystemBenchmarkRunner()
        report = runner.compare(
            {
                "full": _scenario_full,
                "text_only": _scenario_text_only,
            }
        )

        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["experiment_name"] == "system_pipeline_benchmark"
        assert len(data["results"]) == 2
        assert "aggregated" in data

    def test_csv_serialization(self) -> None:
        """Benchmark report serializes to valid CSV."""
        runner = SystemBenchmarkRunner()
        report = runner.compare(
            {
                "full": _scenario_full,
                "text_only": _scenario_text_only,
            }
        )

        csv_str = report.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 scenarios
        assert "text_processing_ms" in lines[0]

    def test_scenario_stage_item_counts(self) -> None:
        """Per-scenario item counts reflect actual pipeline outputs."""
        runner = SystemBenchmarkRunner()
        report = runner.compare({"full": _scenario_full})

        full = report.results[0]
        assert full.stage_item_counts.get("text_processing", 0) > 0
        assert full.stage_item_counts.get("embedding", 0) > 0

    def test_deterministic_benchmark(self) -> None:
        """Two benchmark runs produce consistent structure."""
        runner = SystemBenchmarkRunner()
        r1 = runner.compare({"full": _scenario_full})
        r2 = runner.compare({"full": _scenario_full})

        assert len(r1.results) == len(r2.results)
        assert r1.results[0].stages_executed == r2.results[0].stages_executed
        # Stage item counts should be identical for deterministic pipeline
        assert r1.results[0].stage_item_counts == r2.results[0].stage_item_counts
