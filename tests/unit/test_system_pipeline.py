"""Unit tests for SystemPipeline — end-to-end system orchestrator.

Tests cover: stage execution, graceful degradation, per-stage timing,
warnings, embedding-to-classification wiring, rule evaluation, analytics
event generation, and the SystemPipelineResult envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

from talkex.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig
from talkex.embeddings.config import EmbeddingModelConfig
from talkex.embeddings.generator import NullEmbeddingGenerator
from talkex.embeddings.preprocessing import PreprocessingConfig
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.pipeline.system_pipeline import (
    StageResult,
    SystemPipeline,
    SystemPipelineResult,
    _build_embedding_inputs,
)
from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.config import (
    EvidencePolicy,
    RuleEngineConfig,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT = TranscriptInput(
    conversation_id="conv_test_system",
    raw_text="""\
CUSTOMER: I have a billing issue.
AGENT: I can help you with that.
CUSTOMER: I was charged twice for the same order.
AGENT: Let me look into that right away.
""",
    source_format=SourceFormat.LABELED,
    channel=Channel.VOICE,
)

_CONTEXT_CONFIG = ContextWindowConfig(window_size=3, stride=2)


def _make_text_pipeline() -> TextProcessingPipeline:
    return TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    )


def _make_embedding_generator() -> NullEmbeddingGenerator:
    model_config = EmbeddingModelConfig(
        model_name="null-test",
        model_version="1.0",
    )
    return NullEmbeddingGenerator(
        model_config=model_config,
        preprocessing_config=PreprocessingConfig(),
        dimensions=64,
    )


@dataclass
class StubClassifier:
    """Stub classifier returning a fixed label for every input."""

    label: str = "billing_issue"
    score: float = 0.85
    threshold: float = 0.5
    model_name: str = "stub_classifier"
    model_version: str = "1.0"

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        for inp in inputs:
            results.append(
                ClassificationResult(
                    source_id=inp.source_id,
                    source_type=inp.source_type,
                    label_scores=[
                        LabelScore(
                            label=self.label,
                            score=self.score,
                            confidence=self.score,
                            threshold=self.threshold,
                        )
                    ],
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Stage execution tests
# ---------------------------------------------------------------------------


class TestTextProcessingOnly:
    """Tests with only the text processing stage."""

    def test_runs_text_processing_stage(self) -> None:
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        assert isinstance(result, SystemPipelineResult)
        assert len(result.pipeline_result.turns) == 4
        assert len(result.pipeline_result.windows) > 0

    def test_stage_results_present(self) -> None:
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        stage_names = [s.name for s in result.stages]
        assert "text_processing" in stage_names
        assert "embedding" in stage_names
        assert "classification" in stage_names
        assert "rule_evaluation" in stage_names
        assert "analytics" in stage_names

    def test_skipped_stages_marked(self) -> None:
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        text_stage = next(s for s in result.stages if s.name == "text_processing")
        emb_stage = next(s for s in result.stages if s.name == "embedding")

        assert not text_stage.skipped
        assert emb_stage.skipped

    def test_stats_include_totals(self) -> None:
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        assert result.stats["total_pipeline_ms"] > 0
        assert result.stats["stages_executed"] >= 1
        assert result.stats["stages_skipped"] >= 1

    def test_minimal_transcript_runs(self) -> None:
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
        minimal_transcript = TranscriptInput(
            conversation_id="conv_minimal",
            raw_text="just some plain text",
            source_format=SourceFormat.PLAIN,
            channel=Channel.CHAT,
        )
        result = pipeline.run(minimal_transcript)

        # Plain format produces at least one turn
        assert len(result.pipeline_result.turns) >= 1


# ---------------------------------------------------------------------------
# Embedding stage tests
# ---------------------------------------------------------------------------


class TestEmbeddingStage:
    def test_generates_embeddings(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            embedding_generator=_make_embedding_generator(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        assert len(result.embeddings) > 0
        assert len(result.embeddings) == len(result.pipeline_result.windows)

    def test_embedding_stage_not_skipped(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            embedding_generator=_make_embedding_generator(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        emb_stage = next(s for s in result.stages if s.name == "embedding")
        assert not emb_stage.skipped
        assert emb_stage.elapsed_ms >= 0
        assert emb_stage.item_count > 0

    def test_embedding_records_have_correct_dimensions(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            embedding_generator=_make_embedding_generator(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        for rec in result.embeddings:
            assert rec.dimensions == 64
            assert len(rec.vector) == 64


# ---------------------------------------------------------------------------
# Classification stage tests
# ---------------------------------------------------------------------------


class TestClassificationStage:
    def test_classifies_windows(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            classifier=StubClassifier(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        assert result.classification is not None
        assert len(result.classification.predictions) > 0

    def test_classification_with_embeddings(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            embedding_generator=_make_embedding_generator(),
            classifier=StubClassifier(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        assert result.classification is not None
        assert len(result.embeddings) > 0
        assert len(result.classification.predictions) > 0

    def test_classification_stage_not_skipped(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            classifier=StubClassifier(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        cls_stage = next(s for s in result.stages if s.name == "classification")
        assert not cls_stage.skipped
        assert cls_stage.item_count > 0


# ---------------------------------------------------------------------------
# Rule evaluation stage tests
# ---------------------------------------------------------------------------


class TestRuleEvaluationStage:
    def test_evaluates_rules(self) -> None:
        compiler = SimpleRuleCompiler()
        rules = [
            compiler.compile('keyword("billing")', "rule_billing", "billing_issue"),
            compiler.compile('keyword("charged")', "rule_charged", "double_charge"),
        ]

        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            rule_evaluator=SimpleRuleEvaluator(),
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=_CONTEXT_CONFIG,
            rules=rules,
        )

        assert len(result.rule_executions) > 0

    def test_rule_stage_not_skipped_with_rules(self) -> None:
        compiler = SimpleRuleCompiler()
        rules = [
            compiler.compile('keyword("billing")', "rule_billing", "billing_issue"),
        ]

        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            rule_evaluator=SimpleRuleEvaluator(),
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=_CONTEXT_CONFIG,
            rules=rules,
        )

        rule_stage = next(s for s in result.stages if s.name == "rule_evaluation")
        assert not rule_stage.skipped
        assert rule_stage.item_count > 0

    def test_rule_stage_skipped_without_rules(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            rule_evaluator=SimpleRuleEvaluator(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        rule_stage = next(s for s in result.stages if s.name == "rule_evaluation")
        assert rule_stage.skipped

    def test_rule_config_passed_through(self) -> None:
        compiler = SimpleRuleCompiler()
        rules = [
            compiler.compile('keyword("billing")', "rule_billing", "billing_issue"),
        ]
        config = RuleEngineConfig(
            evaluation_mode=RuleEvaluationMode.ALL,
            evidence_policy=EvidencePolicy.ALWAYS,
            short_circuit_policy=ShortCircuitPolicy.DECLARATION,
        )

        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            rule_evaluator=SimpleRuleEvaluator(),
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=_CONTEXT_CONFIG,
            rules=rules,
            rule_config=config,
        )

        assert len(result.rule_executions) > 0


# ---------------------------------------------------------------------------
# Analytics event collection tests
# ---------------------------------------------------------------------------


class TestAnalyticsEventCollection:
    def test_collects_events_from_predictions(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            classifier=StubClassifier(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        assert len(result.analytics_events) > 0
        prediction_events = [e for e in result.analytics_events if e.event_type == "prediction"]
        assert len(prediction_events) == len(result.classification.predictions)

    def test_collects_events_from_rule_executions(self) -> None:
        compiler = SimpleRuleCompiler()
        rules = [
            compiler.compile('keyword("billing")', "rule_billing", "billing_issue"),
        ]

        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            rule_evaluator=SimpleRuleEvaluator(),
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=_CONTEXT_CONFIG,
            rules=rules,
        )

        rule_events = [e for e in result.analytics_events if e.event_type == "rule_execution"]
        assert len(rule_events) == len(result.rule_executions)

    def test_collects_events_from_both(self) -> None:
        compiler = SimpleRuleCompiler()
        rules = [
            compiler.compile('keyword("billing")', "rule_billing", "billing_issue"),
        ]

        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            classifier=StubClassifier(),
            rule_evaluator=SimpleRuleEvaluator(),
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=_CONTEXT_CONFIG,
            rules=rules,
        )

        total_expected = len(result.classification.predictions) + len(result.rule_executions)
        assert len(result.analytics_events) == total_expected

    def test_analytics_stage_skipped_without_outputs(self) -> None:
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        analytics_stage = next(s for s in result.stages if s.name == "analytics")
        assert analytics_stage.skipped


# ---------------------------------------------------------------------------
# Full pipeline integration (unit-level with stubs)
# ---------------------------------------------------------------------------


class TestFullPipelineComposition:
    def test_all_stages_execute(self) -> None:
        compiler = SimpleRuleCompiler()
        rules = [
            compiler.compile('keyword("billing")', "rule_billing", "billing_issue"),
        ]

        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            embedding_generator=_make_embedding_generator(),
            classifier=StubClassifier(),
            rule_evaluator=SimpleRuleEvaluator(),
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=_CONTEXT_CONFIG,
            rules=rules,
        )

        # All stages should have run
        executed = [s for s in result.stages if not s.skipped]
        # text_processing, embedding, classification, rule_evaluation, analytics
        # indexing is skipped because no vector_index or lexical_index provided
        assert len(executed) >= 5

        # All outputs populated
        assert len(result.pipeline_result.windows) > 0
        assert len(result.embeddings) > 0
        assert result.classification is not None
        assert len(result.rule_executions) > 0
        assert len(result.analytics_events) > 0

    def test_result_is_frozen(self) -> None:
        pipeline = SystemPipeline(text_pipeline=_make_text_pipeline())
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        try:
            result.embeddings = []  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_stage_result_is_frozen(self) -> None:
        stage = StageResult(name="test", elapsed_ms=1.0)
        try:
            stage.name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Indexing stage tests
# ---------------------------------------------------------------------------


class TestIndexingStage:
    def test_indexes_embeddings_in_vector_index(self) -> None:
        from talkex.retrieval.config import VectorIndexConfig
        from talkex.retrieval.vector_index import (
            InMemoryVectorIndex,
        )

        vector_config = VectorIndexConfig(dimensions=64)
        vector_index = InMemoryVectorIndex(config=vector_config)

        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            embedding_generator=_make_embedding_generator(),
            vector_index=vector_index,
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        idx_stage = next(s for s in result.stages if s.name == "indexing")
        assert not idx_stage.skipped
        assert vector_index.vector_count > 0

    def test_indexes_windows_in_lexical_index(self) -> None:
        from talkex.retrieval.bm25 import InMemoryBM25Index

        lexical_index = InMemoryBM25Index()

        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            lexical_index=lexical_index,
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        idx_stage = next(s for s in result.stages if s.name == "indexing")
        assert not idx_stage.skipped
        assert lexical_index.document_count > 0

    def test_indexing_skipped_without_indexes(self) -> None:
        pipeline = SystemPipeline(
            text_pipeline=_make_text_pipeline(),
            embedding_generator=_make_embedding_generator(),
        )
        result = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        idx_stage = next(s for s in result.stages if s.name == "indexing")
        assert idx_stage.skipped


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestBuildEmbeddingInputs:
    def test_builds_inputs_from_windows(self) -> None:
        pipeline = _make_text_pipeline()
        pr = pipeline.run(_TRANSCRIPT, context_config=_CONTEXT_CONFIG)

        inputs = _build_embedding_inputs(pr.windows)

        assert len(inputs) == len(pr.windows)
        for inp, window in zip(inputs, pr.windows, strict=False):
            assert inp.object_id == window.window_id
            assert inp.text == window.window_text
            assert inp.object_type.value == "context_window"

    def test_empty_windows_produces_empty_inputs(self) -> None:
        inputs = _build_embedding_inputs([])
        assert inputs == []


# ---------------------------------------------------------------------------
# Reexport tests
# ---------------------------------------------------------------------------


class TestReexport:
    def test_importable_from_pipeline_package(self) -> None:
        from talkex.pipeline import (
            StageResult,
            SystemPipeline,
            SystemPipelineResult,
        )

        assert StageResult is not None
        assert SystemPipeline is not None
        assert SystemPipelineResult is not None
