"""Unit tests for classification data types.

Tests cover: ClassificationInput, LabelScore, ClassificationResult
construction, properties, score semantics, and reexport.
"""

from semantic_conversation_engine.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)


# ---------------------------------------------------------------------------
# ClassificationInput
# ---------------------------------------------------------------------------


class TestClassificationInput:
    def test_creates_with_required_fields(self) -> None:
        inp = ClassificationInput(
            source_id="win_1",
            source_type="context_window",
            text="billing issue with credit card",
        )
        assert inp.source_id == "win_1"
        assert inp.source_type == "context_window"
        assert inp.text == "billing issue with credit card"
        assert inp.embedding is None
        assert inp.features == {}
        assert inp.metadata == {}

    def test_creates_with_embedding(self) -> None:
        inp = ClassificationInput(
            source_id="win_1",
            source_type="context_window",
            text="billing",
            embedding=[0.1, 0.2, 0.3],
        )
        assert inp.embedding == [0.1, 0.2, 0.3]

    def test_creates_with_features(self) -> None:
        inp = ClassificationInput(
            source_id="win_1",
            source_type="context_window",
            text="billing",
            features={"word_count": 5.0, "char_count": 30.0},
        )
        assert inp.features["word_count"] == 5.0

    def test_is_frozen(self) -> None:
        inp = ClassificationInput(
            source_id="win_1", source_type="context_window", text="test"
        )
        try:
            inp.source_id = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# LabelScore
# ---------------------------------------------------------------------------


class TestLabelScore:
    def test_creates_with_required_fields(self) -> None:
        ls = LabelScore(label="billing", score=0.85, confidence=0.82)
        assert ls.label == "billing"
        assert ls.score == 0.85
        assert ls.confidence == 0.82
        assert ls.threshold == 0.5

    def test_creates_with_custom_threshold(self) -> None:
        ls = LabelScore(label="billing", score=0.4, confidence=0.4, threshold=0.3)
        assert ls.threshold == 0.3

    def test_is_positive_above_threshold(self) -> None:
        ls = LabelScore(label="billing", score=0.7, confidence=0.7, threshold=0.5)
        assert ls.is_positive is True

    def test_is_positive_at_threshold(self) -> None:
        ls = LabelScore(label="billing", score=0.5, confidence=0.5, threshold=0.5)
        assert ls.is_positive is True

    def test_is_not_positive_below_threshold(self) -> None:
        ls = LabelScore(label="billing", score=0.3, confidence=0.3, threshold=0.5)
        assert ls.is_positive is False

    def test_is_frozen(self) -> None:
        ls = LabelScore(label="billing", score=0.5, confidence=0.5)
        try:
            ls.label = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------


class TestClassificationResult:
    def test_creates_with_required_fields(self) -> None:
        result = ClassificationResult(
            source_id="win_1",
            source_type="context_window",
            label_scores=[
                LabelScore(label="billing", score=0.8, confidence=0.8),
                LabelScore(label="cancel", score=0.3, confidence=0.3),
            ],
            model_name="intent-v1",
            model_version="1.0",
        )
        assert result.source_id == "win_1"
        assert len(result.label_scores) == 2

    def test_predicted_labels_filters_by_threshold(self) -> None:
        result = ClassificationResult(
            source_id="win_1",
            source_type="context_window",
            label_scores=[
                LabelScore(label="billing", score=0.8, confidence=0.8, threshold=0.5),
                LabelScore(label="cancel", score=0.3, confidence=0.3, threshold=0.5),
                LabelScore(label="refund", score=0.6, confidence=0.6, threshold=0.5),
            ],
            model_name="intent-v1",
            model_version="1.0",
        )
        predicted = result.predicted_labels
        assert "billing" in predicted
        assert "refund" in predicted
        assert "cancel" not in predicted

    def test_top_label_returns_highest_score(self) -> None:
        result = ClassificationResult(
            source_id="win_1",
            source_type="context_window",
            label_scores=[
                LabelScore(label="billing", score=0.9, confidence=0.9),
                LabelScore(label="cancel", score=0.3, confidence=0.3),
            ],
            model_name="intent-v1",
            model_version="1.0",
        )
        assert result.top_label == "billing"

    def test_top_score_returns_highest_score_value(self) -> None:
        result = ClassificationResult(
            source_id="win_1",
            source_type="context_window",
            label_scores=[
                LabelScore(label="billing", score=0.9, confidence=0.9),
            ],
            model_name="intent-v1",
            model_version="1.0",
        )
        assert result.top_score == 0.9

    def test_top_label_none_when_empty(self) -> None:
        result = ClassificationResult(
            source_id="win_1",
            source_type="context_window",
            label_scores=[],
            model_name="intent-v1",
            model_version="1.0",
        )
        assert result.top_label is None
        assert result.top_score is None

    def test_stats_default_empty(self) -> None:
        result = ClassificationResult(
            source_id="win_1",
            source_type="context_window",
            label_scores=[],
            model_name="intent-v1",
            model_version="1.0",
        )
        assert result.stats == {}

    def test_is_frozen(self) -> None:
        result = ClassificationResult(
            source_id="win_1",
            source_type="context_window",
            label_scores=[],
            model_name="intent-v1",
            model_version="1.0",
        )
        try:
            result.source_id = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestClassificationModelsReexport:
    def test_importable_from_classification_package(self) -> None:
        from semantic_conversation_engine.classification import (
            ClassificationInput as CI,
            ClassificationResult as CR,
            LabelScore as LS,
        )

        assert CI is ClassificationInput
        assert CR is ClassificationResult
        assert LS is LabelScore
