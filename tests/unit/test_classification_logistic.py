"""Unit tests for LogisticRegressionClassifier.

Tests cover: construction, training, classification, multi-label,
thresholds, error handling, and reexport.
"""

import pytest

from talkex.classification.labels import LabelSpace
from talkex.classification.logistic import (
    LogisticRegressionClassifier,
)
from talkex.classification.models import ClassificationInput
from talkex.exceptions import ModelError


def _make_label_space() -> LabelSpace:
    return LabelSpace(
        labels=["billing", "cancel", "refund"],
        default_threshold=0.5,
    )


def _make_training_data() -> tuple[list[ClassificationInput], list[str]]:
    """Create simple linearly separable training data."""
    inputs: list[ClassificationInput] = []
    labels: list[str] = []

    # billing: high word_count, low question_count
    for i in range(10):
        inputs.append(
            ClassificationInput(
                source_id=f"train_b_{i}",
                source_type="context_window",
                text="billing",
                features={"word_count": 10.0 + i, "question_count": 0.0},
            )
        )
        labels.append("billing")

    # cancel: low word_count, high question_count
    for i in range(10):
        inputs.append(
            ClassificationInput(
                source_id=f"train_c_{i}",
                source_type="context_window",
                text="cancel",
                features={"word_count": 1.0, "question_count": 5.0 + i},
            )
        )
        labels.append("cancel")

    # refund: medium word_count, medium question_count
    for i in range(10):
        inputs.append(
            ClassificationInput(
                source_id=f"train_r_{i}",
                source_type="context_window",
                text="refund",
                features={"word_count": 5.0, "question_count": 2.0 + i},
            )
        )
        labels.append("refund")

    return inputs, labels


FEATURE_NAMES = ["word_count", "question_count"]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLogisticConstruction:
    def test_creates_successfully(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        assert cls.is_fitted is False

    def test_custom_model_name(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
            model_name="custom-lr",
            model_version="3.0",
        )
        assert cls.label_space.size == 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestLogisticTraining:
    def test_fit_returns_stats(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        stats = cls.fit(inputs, labels)
        assert stats["sample_count"] == 30
        assert stats["unique_labels"] == 3
        assert stats["feature_count"] == 2
        assert "training_time_ms" in stats

    def test_fit_sets_is_fitted(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)
        assert cls.is_fitted is True

    def test_rejects_empty_inputs(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        with pytest.raises(ModelError, match="non-empty"):
            cls.fit([], [])

    def test_rejects_mismatched_lengths(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, _ = _make_training_data()
        with pytest.raises(ModelError, match="mismatch"):
            cls.fit(inputs, ["billing"])

    def test_rejects_unknown_labels(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs = [
            ClassificationInput(
                source_id="t0",
                source_type="context_window",
                text="test",
                features={"word_count": 1.0, "question_count": 0.0},
            )
        ]
        with pytest.raises(ModelError, match="Unknown labels"):
            cls.fit(inputs, ["unknown_label"])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestLogisticClassification:
    def test_classify_returns_results(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)

        test_input = ClassificationInput(
            source_id="test_0",
            source_type="context_window",
            text="test",
            features={"word_count": 15.0, "question_count": 0.0},
        )
        results = cls.classify([test_input])
        assert len(results) == 1
        assert results[0].source_id == "test_0"

    def test_high_word_count_predicts_billing(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)

        test_input = ClassificationInput(
            source_id="test_0",
            source_type="context_window",
            text="test",
            features={"word_count": 20.0, "question_count": 0.0},
        )
        results = cls.classify([test_input])
        assert results[0].top_label == "billing"

    def test_high_question_count_predicts_cancel(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)

        test_input = ClassificationInput(
            source_id="test_0",
            source_type="context_window",
            text="test",
            features={"word_count": 1.0, "question_count": 15.0},
        )
        results = cls.classify([test_input])
        assert results[0].top_label == "cancel"

    def test_classify_batch(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)

        test_inputs = [
            ClassificationInput(
                source_id=f"t_{i}",
                source_type="context_window",
                text="test",
                features={"word_count": float(10 + i), "question_count": 0.0},
            )
            for i in range(5)
        ]
        results = cls.classify(test_inputs)
        assert len(results) == 5
        assert all(r.source_id == f"t_{i}" for i, r in enumerate(results))

    def test_label_scores_sorted_descending(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)

        test_input = ClassificationInput(
            source_id="t0",
            source_type="context_window",
            text="test",
            features={"word_count": 10.0, "question_count": 0.0},
        )
        results = cls.classify([test_input])
        scores = [ls.score for ls in results[0].label_scores]
        assert scores == sorted(scores, reverse=True)

    def test_stats_populated(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)

        test_input = ClassificationInput(
            source_id="t0",
            source_type="context_window",
            text="test",
            features={"word_count": 10.0, "question_count": 0.0},
        )
        results = cls.classify([test_input])
        assert "latency_ms" in results[0].stats
        assert results[0].stats["num_labels"] == 3

    def test_model_name_and_version_in_result(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
            model_name="my-lr",
            model_version="2.0",
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)

        test_input = ClassificationInput(
            source_id="t0",
            source_type="context_window",
            text="test",
            features={"word_count": 10.0, "question_count": 0.0},
        )
        results = cls.classify([test_input])
        assert results[0].model_name == "my-lr"
        assert results[0].model_version == "2.0"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestLogisticErrors:
    def test_classify_before_fit_raises(self) -> None:
        cls = LogisticRegressionClassifier(
            label_space=_make_label_space(),
            feature_names=FEATURE_NAMES,
        )
        with pytest.raises(ModelError, match="fitted"):
            cls.classify(
                [
                    ClassificationInput(
                        source_id="t0",
                        source_type="context_window",
                        text="test",
                        features={"word_count": 1.0},
                    )
                ]
            )


# ---------------------------------------------------------------------------
# Per-label thresholds
# ---------------------------------------------------------------------------


class TestLogisticThresholds:
    def test_per_label_threshold_applied_to_label_scores(self) -> None:
        ls = LabelSpace(
            labels=["billing", "cancel", "refund"],
            thresholds={"billing": 0.8},
            default_threshold=0.5,
        )
        cls = LogisticRegressionClassifier(
            label_space=ls,
            feature_names=FEATURE_NAMES,
        )
        inputs, labels = _make_training_data()
        cls.fit(inputs, labels)

        test_input = ClassificationInput(
            source_id="t0",
            source_type="context_window",
            text="test",
            features={"word_count": 20.0, "question_count": 0.0},
        )
        results = cls.classify([test_input])
        # Verify that each label_score has the correct threshold from label_space
        threshold_map = {ls.label: ls.threshold for ls in results[0].label_scores}
        assert threshold_map["billing"] == 0.8
        assert threshold_map["cancel"] == 0.5
        assert threshold_map["refund"] == 0.5


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestLogisticReexport:
    def test_importable_from_classification_package(self) -> None:
        from talkex.classification import (
            LogisticRegressionClassifier as LRC,
        )

        assert LRC is LogisticRegressionClassifier
