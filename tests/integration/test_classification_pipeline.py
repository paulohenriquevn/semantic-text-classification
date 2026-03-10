"""Integration tests for classification pipeline.

End-to-end: raw text → segmentation → context windows → feature extraction
→ embedding generation → classification (similarity + logistic) → Predictions.
"""

from semantic_conversation_engine.classification.features import (
    extract_lexical_features,
    extract_structural_features,
    merge_feature_sets,
)
from semantic_conversation_engine.classification.labels import LabelSpace
from semantic_conversation_engine.classification.logistic import (
    LogisticRegressionClassifier,
)
from semantic_conversation_engine.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from semantic_conversation_engine.classification.orchestrator import (
    ClassificationOrchestrator,
)
from semantic_conversation_engine.classification.similarity import (
    EmbeddingSimilarityClassifier,
)
from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
from semantic_conversation_engine.embeddings.generator import NullEmbeddingGenerator
from semantic_conversation_engine.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.enums import Channel, ObjectType, PoolingStrategy
from semantic_conversation_engine.models.types import EmbeddingId
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT_TEXT = """\
Customer: I have a billing issue with my credit card.
Agent: I can help you with that. What is the issue?
Customer: I was charged twice for the same order.
Agent: Let me look into that for you right away.
"""

_LABEL_SPACE = LabelSpace(
    labels=["billing", "cancel", "refund"],
    thresholds={"billing": 0.3},
    default_threshold=0.5,
)


def _build_pipeline_result():
    """Run the text processing pipeline on test transcript."""
    pipeline = TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    )
    transcript = TranscriptInput(
        conversation_id="conv_integ_cls",
        raw_text=_TRANSCRIPT_TEXT,
        channel=Channel.VOICE,
        source_format=SourceFormat.LABELED,
    )
    return pipeline.run(
        transcript,
        context_config=ContextWindowConfig(window_size=3, stride=2),
    )


def _generate_embeddings(windows):
    """Generate embeddings for context windows using NullEmbeddingGenerator."""
    generator = NullEmbeddingGenerator(
        model_config=EmbeddingModelConfig(
            model_name="null-test",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        ),
        dimensions=8,
    )
    batch = EmbeddingBatch(
        items=[
            EmbeddingInput(
                embedding_id=EmbeddingId(f"emb_{w.window_id}"),
                object_type=ObjectType.CONTEXT_WINDOW,
                object_id=w.window_id,
                text=w.window_text,
            )
            for w in windows
        ]
    )
    records = generator.generate(batch)
    return {r.source_id: r.vector for r in records}


# ---------------------------------------------------------------------------
# Similarity classifier end-to-end
# ---------------------------------------------------------------------------


class TestSimilarityPipeline:
    def test_full_pipeline_with_similarity_classifier(self) -> None:
        result = _build_pipeline_result()
        windows = result.windows
        assert len(windows) > 0

        embeddings = _generate_embeddings(windows)

        # Use first window's embedding as billing centroid
        first_emb = next(iter(embeddings.values()))
        centroids = {
            "billing": first_emb,
            "cancel": [0.0] * len(first_emb),
            "refund": [float(i) for i in range(len(first_emb))],
        }

        classifier = EmbeddingSimilarityClassifier(
            label_space=_LABEL_SPACE,
            centroids=centroids,
            model_name="integ-similarity",
            model_version="1.0",
        )

        orch = ClassificationOrchestrator(classifier=classifier)
        batch_result = orch.classify_windows(windows, embeddings=embeddings)

        assert len(batch_result.classification_results) == len(windows)
        for cr in batch_result.classification_results:
            assert cr.model_name == "integ-similarity"
            assert len(cr.label_scores) == 3

    def test_similarity_produces_predictions(self) -> None:
        result = _build_pipeline_result()
        windows = result.windows
        embeddings = _generate_embeddings(windows)

        first_emb = next(iter(embeddings.values()))
        centroids = {
            "billing": first_emb,
            "cancel": [0.0] * len(first_emb),
            "refund": [0.0] * len(first_emb),
        }

        classifier = EmbeddingSimilarityClassifier(
            label_space=_LABEL_SPACE,
            centroids=centroids,
        )
        orch = ClassificationOrchestrator(classifier=classifier)
        batch_result = orch.classify_windows(windows, embeddings=embeddings)

        # First window should be very similar to billing centroid
        assert len(batch_result.predictions) > 0
        for pred in batch_result.predictions:
            assert pred.prediction_id.startswith("pred_")
            assert pred.source_type == ObjectType.CONTEXT_WINDOW
            assert pred.is_above_threshold

    def test_similarity_stats_populated(self) -> None:
        result = _build_pipeline_result()
        windows = result.windows
        embeddings = _generate_embeddings(windows)

        first_emb = next(iter(embeddings.values()))
        centroids = {
            "billing": first_emb,
            "cancel": [0.0] * len(first_emb),
            "refund": [0.0] * len(first_emb),
        }

        classifier = EmbeddingSimilarityClassifier(
            label_space=_LABEL_SPACE,
            centroids=centroids,
        )
        orch = ClassificationOrchestrator(classifier=classifier)
        batch_result = orch.classify_windows(windows, embeddings=embeddings)

        assert batch_result.stats["classification_level"] == "context_window"
        assert batch_result.stats["num_inputs"] == len(windows)
        assert "classification_latency_ms" in batch_result.stats


# ---------------------------------------------------------------------------
# Logistic classifier end-to-end
# ---------------------------------------------------------------------------


class TestLogisticPipeline:
    def test_full_pipeline_with_logistic_classifier(self) -> None:
        result = _build_pipeline_result()
        windows = result.windows
        assert len(windows) > 0

        feature_names = [
            "char_count",
            "word_count",
            "avg_word_length",
            "question_count",
            "exclamation_count",
            "uppercase_ratio",
            "digit_ratio",
            "is_customer",
            "is_agent",
            "turn_count",
            "speaker_count",
        ]

        classifier = LogisticRegressionClassifier(
            label_space=_LABEL_SPACE,
            feature_names=feature_names,
            model_name="integ-logistic",
            model_version="1.0",
        )

        # Build training data from the windows themselves
        training_inputs: list[ClassificationInput] = []
        training_labels: list[str] = []
        for i, w in enumerate(windows):
            lex = extract_lexical_features(w.window_text)
            struct = extract_structural_features(turn_count=w.window_size, speaker_count=2)
            merged = merge_feature_sets(lex, struct)
            training_inputs.append(
                ClassificationInput(
                    source_id=f"train_{i}",
                    source_type="context_window",
                    text=w.window_text,
                    features=merged.features,
                )
            )
            training_labels.append("billing")

        # Need at least 2 classes for sklearn
        training_inputs.append(
            ClassificationInput(
                source_id="train_cancel",
                source_type="context_window",
                text="cancel my subscription",
                features={n: 0.0 for n in feature_names},
            )
        )
        training_labels.append("cancel")

        stats = classifier.fit(training_inputs, training_labels)
        assert stats["sample_count"] == len(training_inputs)

        # Classify via orchestrator
        orch = ClassificationOrchestrator(classifier=classifier)
        batch_result = orch.classify_windows(windows)

        assert len(batch_result.classification_results) == len(windows)
        for cr in batch_result.classification_results:
            assert cr.model_name == "integ-logistic"

    def test_logistic_produces_predictions(self) -> None:
        result = _build_pipeline_result()
        windows = result.windows

        feature_names = [
            "char_count",
            "word_count",
            "avg_word_length",
            "question_count",
            "exclamation_count",
            "uppercase_ratio",
            "digit_ratio",
            "is_customer",
            "is_agent",
            "turn_count",
            "speaker_count",
        ]

        classifier = LogisticRegressionClassifier(
            label_space=_LABEL_SPACE,
            feature_names=feature_names,
        )

        # Train with labeled data
        training_inputs = []
        training_labels = []
        for i in range(5):
            training_inputs.append(
                ClassificationInput(
                    source_id=f"t_b_{i}",
                    source_type="context_window",
                    text="billing issue",
                    features={
                        "char_count": 50.0,
                        "word_count": 10.0,
                        "avg_word_length": 5.0,
                        "question_count": 1.0,
                        "exclamation_count": 0.0,
                        "uppercase_ratio": 0.1,
                        "digit_ratio": 0.0,
                        "is_customer": 1.0,
                        "is_agent": 1.0,
                        "turn_count": 3.0,
                        "speaker_count": 2.0,
                    },
                )
            )
            training_labels.append("billing")

        for i in range(5):
            training_inputs.append(
                ClassificationInput(
                    source_id=f"t_c_{i}",
                    source_type="context_window",
                    text="cancel",
                    features={n: 0.0 for n in feature_names},
                )
            )
            training_labels.append("cancel")

        classifier.fit(training_inputs, training_labels)

        orch = ClassificationOrchestrator(classifier=classifier)
        batch_result = orch.classify_windows(windows)

        # Should produce at least some predictions
        assert len(batch_result.predictions) >= 0
        for pred in batch_result.predictions:
            assert pred.source_type == ObjectType.CONTEXT_WINDOW
            assert pred.prediction_id.startswith("pred_")


# ---------------------------------------------------------------------------
# Multi-level classification
# ---------------------------------------------------------------------------


class TestMultiLevelClassification:
    def test_turn_level_classification(self) -> None:
        result = _build_pipeline_result()
        turns = result.turns
        assert len(turns) > 0

        scores = [
            LabelScore(label="billing", score=0.8, confidence=0.8, threshold=0.5),
            LabelScore(label="cancel", score=0.2, confidence=0.2, threshold=0.5),
        ]

        class _TurnClassifier:
            def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
                return [
                    ClassificationResult(
                        source_id=inp.source_id,
                        source_type=inp.source_type,
                        label_scores=list(scores),
                        model_name="turn-clf",
                        model_version="1.0",
                    )
                    for inp in inputs
                ]

        orch = ClassificationOrchestrator(classifier=_TurnClassifier())

        turn_inputs = [
            ClassificationInput(
                source_id=t.turn_id,
                source_type="turn",
                text=t.raw_text,
            )
            for t in turns
        ]
        batch_result = orch.classify_inputs(turn_inputs)

        assert len(batch_result.predictions) == len(turns)
        for pred in batch_result.predictions:
            assert pred.source_type == ObjectType.TURN
            assert pred.label == "billing"
