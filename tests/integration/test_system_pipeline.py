"""Integration tests for SystemPipeline — full end-to-end flow.

Exercises the complete pipeline: raw transcript → text processing → embedding
→ dual index building → classification → rule evaluation → analytics events.

All stages use real implementations (NullEmbeddingGenerator, InMemoryBM25Index,
InMemoryVectorIndex, StubClassifier, SimpleRuleCompiler/Evaluator).
"""

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
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.pipeline.system_pipeline import (
    SystemPipeline,
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
    conversation_id="conv_integration_system",
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


class StubClassifier:
    """Stub classifier for integration testing."""

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        for inp in inputs:
            scores = []
            text_lower = inp.text.lower()
            if "billing" in text_lower:
                scores.append(LabelScore(label="billing_issue", score=0.9, confidence=0.9, threshold=0.5))
            if "cancel" in text_lower:
                scores.append(LabelScore(label="cancellation", score=0.85, confidence=0.85, threshold=0.5))
            if "refund" in text_lower:
                scores.append(LabelScore(label="refund_request", score=0.8, confidence=0.8, threshold=0.5))
            if not scores:
                scores.append(LabelScore(label="general", score=0.3, confidence=0.3, threshold=0.5))
            results.append(
                ClassificationResult(
                    source_id=inp.source_id,
                    source_type=inp.source_type,
                    label_scores=scores,
                    model_name="integration_stub",
                    model_version="1.0",
                )
            )
        return results


def _build_full_pipeline() -> tuple:
    """Build all components for a full system pipeline."""
    text_pipeline = TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    )
    embedding_generator = NullEmbeddingGenerator(
        model_config=EmbeddingModelConfig(
            model_name="null-integration",
            model_version="1.0",
        ),
        preprocessing_config=PreprocessingConfig(),
        dimensions=64,
    )
    vector_index = InMemoryVectorIndex(
        config=VectorIndexConfig(dimensions=64),
    )
    lexical_index = InMemoryBM25Index()
    classifier = StubClassifier()
    rule_evaluator = SimpleRuleEvaluator()

    compiler = SimpleRuleCompiler()
    rules = [
        compiler.compile('keyword("billing")', "rule_billing", "billing_issue"),
        compiler.compile('keyword("cancel")', "rule_cancel", "cancel_intent"),
        compiler.compile('keyword("refund")', "rule_refund", "refund_request"),
    ]

    rule_config = RuleEngineConfig(
        evaluation_mode=RuleEvaluationMode.ALL,
        evidence_policy=EvidencePolicy.ALWAYS,
        short_circuit_policy=ShortCircuitPolicy.DECLARATION,
    )

    return (
        text_pipeline,
        embedding_generator,
        vector_index,
        lexical_index,
        classifier,
        rule_evaluator,
        rules,
        rule_config,
    )


class TestSystemPipelineEndToEnd:
    """Full end-to-end integration tests."""

    def test_all_stages_produce_output(self) -> None:
        """Every stage in the pipeline produces non-empty output."""
        (
            text_pipeline,
            emb_gen,
            vec_idx,
            lex_idx,
            classifier,
            rule_eval,
            rules,
            rule_config,
        ) = _build_full_pipeline()

        pipeline = SystemPipeline(
            text_pipeline=text_pipeline,
            embedding_generator=emb_gen,
            vector_index=vec_idx,
            lexical_index=lex_idx,
            classifier=classifier,
            rule_evaluator=rule_eval,
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=ContextWindowConfig(window_size=3, stride=2),
            rules=rules,
            rule_config=rule_config,
        )

        # Text processing
        assert len(result.pipeline_result.turns) == 6
        assert len(result.pipeline_result.windows) > 0

        # Embeddings
        assert len(result.embeddings) == len(result.pipeline_result.windows)

        # Indexes
        assert vec_idx.vector_count == len(result.embeddings)
        assert lex_idx.document_count == len(result.pipeline_result.windows)

        # Classification
        assert result.classification is not None
        assert len(result.classification.predictions) > 0

        # Rules
        assert len(result.rule_executions) > 0

        # Analytics
        assert len(result.analytics_events) > 0

    def test_stages_are_timed(self) -> None:
        """Every executed stage has timing information."""
        (
            text_pipeline,
            emb_gen,
            vec_idx,
            lex_idx,
            classifier,
            rule_eval,
            rules,
            rule_config,
        ) = _build_full_pipeline()

        pipeline = SystemPipeline(
            text_pipeline=text_pipeline,
            embedding_generator=emb_gen,
            vector_index=vec_idx,
            lexical_index=lex_idx,
            classifier=classifier,
            rule_evaluator=rule_eval,
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=ContextWindowConfig(window_size=3, stride=2),
            rules=rules,
            rule_config=rule_config,
        )

        executed = [s for s in result.stages if not s.skipped]
        assert len(executed) == 6  # all 6 stages
        for stage in executed:
            assert stage.elapsed_ms >= 0

    def test_aggregated_stats(self) -> None:
        """Aggregated stats reflect actual execution."""
        (
            text_pipeline,
            emb_gen,
            vec_idx,
            lex_idx,
            classifier,
            rule_eval,
            rules,
            rule_config,
        ) = _build_full_pipeline()

        pipeline = SystemPipeline(
            text_pipeline=text_pipeline,
            embedding_generator=emb_gen,
            vector_index=vec_idx,
            lexical_index=lex_idx,
            classifier=classifier,
            rule_evaluator=rule_eval,
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=ContextWindowConfig(window_size=3, stride=2),
            rules=rules,
            rule_config=rule_config,
        )

        assert result.stats["total_pipeline_ms"] > 0
        assert result.stats["stages_executed"] == 6
        assert result.stats["stages_skipped"] == 0
        assert result.stats["index_built"] is True
        assert result.stats["embedding_count"] > 0
        assert result.stats["rule_execution_count"] > 0
        assert result.stats["analytics_event_count"] > 0

    def test_analytics_events_have_correct_types(self) -> None:
        """Analytics events correctly distinguish predictions and rules."""
        (
            text_pipeline,
            emb_gen,
            _vec_idx,
            _lex_idx,
            classifier,
            rule_eval,
            rules,
            rule_config,
        ) = _build_full_pipeline()

        pipeline = SystemPipeline(
            text_pipeline=text_pipeline,
            embedding_generator=emb_gen,
            classifier=classifier,
            rule_evaluator=rule_eval,
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=ContextWindowConfig(window_size=3, stride=2),
            rules=rules,
            rule_config=rule_config,
        )

        pred_events = [e for e in result.analytics_events if e.event_type == "prediction"]
        rule_events = [e for e in result.analytics_events if e.event_type == "rule_execution"]

        assert len(pred_events) == len(result.classification.predictions)
        assert len(rule_events) == len(result.rule_executions)

    def test_indexes_searchable_after_pipeline(self) -> None:
        """Indexes built during pipeline execution are immediately searchable."""
        (
            text_pipeline,
            emb_gen,
            vec_idx,
            lex_idx,
            _classifier,
            _rule_eval,
            _rules,
            _rule_config,
        ) = _build_full_pipeline()

        pipeline = SystemPipeline(
            text_pipeline=text_pipeline,
            embedding_generator=emb_gen,
            vector_index=vec_idx,
            lexical_index=lex_idx,
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=ContextWindowConfig(window_size=3, stride=2),
        )

        # Lexical search should find billing-related windows
        lexical_hits = lex_idx.search("billing issue", top_k=5)
        assert len(lexical_hits) > 0

        # Vector search should return results
        query_vector = result.embeddings[0].vector
        vector_hits = vec_idx.search_by_vector(query_vector, top_k=5)
        assert len(vector_hits) > 0

    def test_deterministic_output(self) -> None:
        """Two runs with same input produce same structure."""
        (
            text_pipeline,
            emb_gen,
            _,
            _,
            classifier,
            rule_eval,
            rules,
            rule_config,
        ) = _build_full_pipeline()

        pipeline = SystemPipeline(
            text_pipeline=text_pipeline,
            embedding_generator=emb_gen,
            classifier=classifier,
            rule_evaluator=rule_eval,
        )
        config = ContextWindowConfig(window_size=3, stride=2)

        r1 = pipeline.run(_TRANSCRIPT, context_config=config, rules=rules, rule_config=rule_config)
        r2 = pipeline.run(_TRANSCRIPT, context_config=config, rules=rules, rule_config=rule_config)

        assert len(r1.pipeline_result.turns) == len(r2.pipeline_result.turns)
        assert len(r1.pipeline_result.windows) == len(r2.pipeline_result.windows)
        assert len(r1.embeddings) == len(r2.embeddings)
        assert len(r1.rule_executions) == len(r2.rule_executions)

    def test_partial_pipeline_without_classification(self) -> None:
        """Pipeline runs successfully without a classifier."""
        (
            text_pipeline,
            emb_gen,
            vec_idx,
            lex_idx,
            _,
            rule_eval,
            rules,
            rule_config,
        ) = _build_full_pipeline()

        pipeline = SystemPipeline(
            text_pipeline=text_pipeline,
            embedding_generator=emb_gen,
            vector_index=vec_idx,
            lexical_index=lex_idx,
            rule_evaluator=rule_eval,
        )
        result = pipeline.run(
            _TRANSCRIPT,
            context_config=ContextWindowConfig(window_size=3, stride=2),
            rules=rules,
            rule_config=rule_config,
        )

        assert result.classification is None
        assert len(result.embeddings) > 0
        assert len(result.rule_executions) > 0
        # Analytics events only from rules
        assert all(e.event_type == "rule_execution" for e in result.analytics_events)
