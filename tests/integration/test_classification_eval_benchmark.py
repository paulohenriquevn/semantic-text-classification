"""Integration tests for classification evaluation benchmark.

End-to-end: raw text → segmentation → context windows → embedding generation
→ classification (similarity + logistic) → evaluation dataset → benchmark
runner comparison → JSON/CSV export.
"""

import json

from talkex.classification.features import (
    extract_lexical_features,
    extract_structural_features,
    merge_feature_sets,
)
from talkex.classification.labels import LabelSpace
from talkex.classification.logistic import (
    LogisticRegressionClassifier,
)
from talkex.classification.models import (
    ClassificationInput,
)
from talkex.classification.similarity import (
    EmbeddingSimilarityClassifier,
)
from talkex.classification_eval.dataset import (
    ClassificationDataset,
    ClassificationExample,
    GroundTruthLabel,
)
from talkex.classification_eval.runner import (
    ClassificationBenchmarkRunner,
)
from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig
from talkex.embeddings.config import EmbeddingModelConfig
from talkex.embeddings.generator import NullEmbeddingGenerator
from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel, ObjectType, PoolingStrategy
from talkex.models.types import EmbeddingId
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.segmentation.segmenter import TurnSegmenter

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
        conversation_id="conv_eval",
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
            model_name="null-eval",
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


def _build_evaluation_dataset(
    windows,
    embeddings: dict[str, list[float]] | None = None,
    features_map: dict[str, dict[str, float]] | None = None,
) -> ClassificationDataset:
    """Build an evaluation dataset from pipeline windows."""
    examples = []
    for w in windows:
        embedding = embeddings.get(w.window_id) if embeddings else None
        features = features_map.get(w.window_id, {}) if features_map else {}
        # All billing-related in this test transcript
        examples.append(
            ClassificationExample(
                example_id=w.window_id,
                text=w.window_text,
                ground_truth=[GroundTruthLabel(label="billing")],
                source_type="context_window",
                embedding=embedding,
                features=features,
                metadata={"conversation_id": w.conversation_id},
            )
        )
    return ClassificationDataset(
        name="billing-eval",
        version="1.0",
        examples=examples,
        label_names=["billing", "cancel", "refund"],
        description="Evaluation dataset for billing classification",
    )


def _build_features_map(windows) -> dict[str, dict[str, float]]:
    """Extract features for each window, keyed by window_id."""
    features_map: dict[str, dict[str, float]] = {}
    for w in windows:
        lex = extract_lexical_features(w.window_text)
        struct = extract_structural_features(turn_count=w.window_size, speaker_count=2)
        merged = merge_feature_sets(lex, struct)
        features_map[w.window_id] = merged.features
    return features_map


def _build_similarity_classifier(embeddings):
    """Build a similarity classifier using first embedding as billing centroid."""
    first_emb = next(iter(embeddings.values()))
    centroids = {
        "billing": first_emb,
        "cancel": [0.0] * len(first_emb),
        "refund": [0.0] * len(first_emb),
    }
    return EmbeddingSimilarityClassifier(
        label_space=_LABEL_SPACE,
        centroids=centroids,
        model_name="eval-similarity",
        model_version="1.0",
    )


def _build_logistic_classifier(windows):
    """Build and train a logistic classifier from window features."""
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
        model_name="eval-logistic",
        model_version="1.0",
    )

    # Build training data
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

    classifier.fit(training_inputs, training_labels)
    return classifier


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestClassificationEvalBenchmark:
    def test_full_pipeline_to_evaluation(self) -> None:
        """End-to-end: pipeline → classifiers → evaluation → comparison."""
        result = _build_pipeline_result()
        windows = result.windows
        assert len(windows) > 0

        embeddings = _generate_embeddings(windows)
        features_map = _build_features_map(windows)
        dataset = _build_evaluation_dataset(windows, embeddings=embeddings, features_map=features_map)

        sim_clf = _build_similarity_classifier(embeddings)
        log_clf = _build_logistic_classifier(windows)

        runner = ClassificationBenchmarkRunner(dataset=dataset)
        report = runner.compare(
            {
                "similarity": sim_clf,
                "logistic": log_clf,
            }
        )

        assert report.dataset_name == "billing-eval"
        assert len(report.results) == 2

        # Both should have metrics
        for method_result in report.results:
            assert method_result.total_examples == len(windows)
            assert "mean_f1" in method_result.aggregated
            assert "micro_f1" in method_result.aggregated
            assert "macro_f1" in method_result.aggregated

    def test_metric_ranges(self) -> None:
        """All metrics should be in [0.0, 1.0]."""
        result = _build_pipeline_result()
        windows = result.windows
        embeddings = _generate_embeddings(windows)
        dataset = _build_evaluation_dataset(windows, embeddings=embeddings)

        sim_clf = _build_similarity_classifier(embeddings)
        runner = ClassificationBenchmarkRunner(dataset=dataset)
        method_result = runner.evaluate(sim_clf, "similarity")

        for key, value in method_result.aggregated.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"

        for em in method_result.example_metrics:
            assert 0.0 <= em.precision <= 1.0
            assert 0.0 <= em.recall <= 1.0
            assert 0.0 <= em.f1 <= 1.0

    def test_per_label_metrics_complete(self) -> None:
        """Per-label metrics should cover all labels in the dataset."""
        result = _build_pipeline_result()
        windows = result.windows
        embeddings = _generate_embeddings(windows)
        dataset = _build_evaluation_dataset(windows, embeddings=embeddings)

        sim_clf = _build_similarity_classifier(embeddings)
        runner = ClassificationBenchmarkRunner(dataset=dataset)
        method_result = runner.evaluate(sim_clf, "similarity")

        assert set(method_result.per_label.keys()) == {"billing", "cancel", "refund"}
        for _label, metrics in method_result.per_label.items():
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "support" in metrics

    def test_json_export(self, tmp_path) -> None:
        """JSON export produces valid, parseable JSON."""
        result = _build_pipeline_result()
        windows = result.windows
        embeddings = _generate_embeddings(windows)
        features_map = _build_features_map(windows)
        dataset = _build_evaluation_dataset(windows, embeddings=embeddings, features_map=features_map)

        sim_clf = _build_similarity_classifier(embeddings)
        log_clf = _build_logistic_classifier(windows)

        runner = ClassificationBenchmarkRunner(dataset=dataset)
        report = runner.compare({"similarity": sim_clf, "logistic": log_clf})

        path = tmp_path / "report.json"
        report.save_json(path)
        data = json.loads(path.read_text())
        assert data["dataset_name"] == "billing-eval"
        assert len(data["results"]) == 2

    def test_csv_export(self, tmp_path) -> None:
        """CSV export produces parseable CSV with correct structure."""
        result = _build_pipeline_result()
        windows = result.windows
        embeddings = _generate_embeddings(windows)
        features_map = _build_features_map(windows)
        dataset = _build_evaluation_dataset(windows, embeddings=embeddings, features_map=features_map)

        sim_clf = _build_similarity_classifier(embeddings)
        log_clf = _build_logistic_classifier(windows)

        runner = ClassificationBenchmarkRunner(dataset=dataset)
        report = runner.compare({"similarity": sim_clf, "logistic": log_clf})

        path = tmp_path / "report.csv"
        report.save_csv(path)
        content = path.read_text()
        assert "similarity" in content
        assert "logistic" in content

    def test_dataset_json_round_trip(self, tmp_path) -> None:
        """Evaluation dataset survives JSON save/load round-trip."""
        result = _build_pipeline_result()
        windows = result.windows
        dataset = _build_evaluation_dataset(windows)

        path = tmp_path / "dataset.json"
        dataset.save(path)
        loaded = ClassificationDataset.load(path)

        assert loaded.name == dataset.name
        assert loaded.version == dataset.version
        assert len(loaded.examples) == len(dataset.examples)
        assert loaded.label_names == dataset.label_names

        # Verify ground truth preserved
        for orig, loaded_ex in zip(dataset.examples, loaded.examples, strict=True):
            assert orig.label_set == loaded_ex.label_set

    def test_similarity_has_billing_recall(self) -> None:
        """Similarity classifier should detect billing (uses billing centroid)."""
        result = _build_pipeline_result()
        windows = result.windows
        embeddings = _generate_embeddings(windows)
        dataset = _build_evaluation_dataset(windows, embeddings=embeddings)

        sim_clf = _build_similarity_classifier(embeddings)
        runner = ClassificationBenchmarkRunner(dataset=dataset)
        method_result = runner.evaluate(sim_clf, "similarity")

        # Billing recall should be > 0 since we use the first window's
        # embedding as the billing centroid
        assert method_result.per_label["billing"]["recall"] > 0.0
