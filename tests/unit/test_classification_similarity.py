"""Unit tests for EmbeddingSimilarityClassifier.

Tests cover: construction, classification, multi-label, thresholds,
error handling, determinism, and reexport.
"""

import pytest

from semantic_conversation_engine.classification.labels import LabelSpace
from semantic_conversation_engine.classification.models import ClassificationInput
from semantic_conversation_engine.classification.similarity import (
    EmbeddingSimilarityClassifier,
)
from semantic_conversation_engine.exceptions import ModelError


def _make_label_space() -> LabelSpace:
    return LabelSpace(
        labels=["billing", "cancel", "refund"],
        default_threshold=0.5,
    )


def _make_centroids() -> dict[str, list[float]]:
    return {
        "billing": [1.0, 0.0, 0.0],
        "cancel": [0.0, 1.0, 0.0],
        "refund": [0.0, 0.0, 1.0],
    }


def _make_input(
    embedding: list[float],
    source_id: str = "win_0",
) -> ClassificationInput:
    return ClassificationInput(
        source_id=source_id,
        source_type="context_window",
        text="test text",
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSimilarityClassifierConstruction:
    def test_creates_successfully(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        assert cls.label_space.size == 3

    def test_rejects_missing_centroid(self) -> None:
        with pytest.raises(ModelError, match="Missing centroids"):
            EmbeddingSimilarityClassifier(
                label_space=_make_label_space(),
                centroids={"billing": [1.0, 0.0, 0.0]},  # missing cancel, refund
            )

    def test_rejects_inconsistent_dimensions(self) -> None:
        with pytest.raises(ModelError, match="same dimensionality"):
            EmbeddingSimilarityClassifier(
                label_space=_make_label_space(),
                centroids={
                    "billing": [1.0, 0.0],
                    "cancel": [0.0, 1.0, 0.0],
                    "refund": [0.0, 0.0, 1.0],
                },
            )

    def test_custom_model_name_and_version(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
            model_name="my-sim",
            model_version="2.0",
        )
        results = cls.classify([_make_input([1.0, 0.0, 0.0])])
        assert results[0].model_name == "my-sim"
        assert results[0].model_version == "2.0"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestSimilarityClassification:
    def test_closest_label_gets_highest_score(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        results = cls.classify([_make_input([1.0, 0.0, 0.0])])
        assert results[0].top_label == "billing"

    def test_orthogonal_embedding_gets_zero_score(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        results = cls.classify([_make_input([1.0, 0.0, 0.0])])
        # cancel and refund centroids are orthogonal to [1,0,0]
        scores_by_label = {ls.label: ls.score for ls in results[0].label_scores}
        assert scores_by_label["cancel"] == 0.0
        assert scores_by_label["refund"] == 0.0

    def test_mixed_embedding_produces_partial_scores(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        # Equal mix of billing and cancel
        results = cls.classify([_make_input([1.0, 1.0, 0.0])])
        scores_by_label = {ls.label: ls.score for ls in results[0].label_scores}
        assert abs(scores_by_label["billing"] - scores_by_label["cancel"]) < 1e-6
        assert scores_by_label["billing"] > 0.0
        assert scores_by_label["refund"] == 0.0

    def test_batch_classification(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        inputs = [
            _make_input([1.0, 0.0, 0.0], source_id="w0"),
            _make_input([0.0, 1.0, 0.0], source_id="w1"),
            _make_input([0.0, 0.0, 1.0], source_id="w2"),
        ]
        results = cls.classify(inputs)
        assert len(results) == 3
        assert results[0].top_label == "billing"
        assert results[1].top_label == "cancel"
        assert results[2].top_label == "refund"

    def test_label_scores_sorted_descending(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        results = cls.classify([_make_input([1.0, 0.5, 0.0])])
        scores = [ls.score for ls in results[0].label_scores]
        assert scores == sorted(scores, reverse=True)

    def test_preserves_source_id(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        results = cls.classify([_make_input([1.0, 0.0, 0.0], source_id="my_win")])
        assert results[0].source_id == "my_win"
        assert results[0].source_type == "context_window"

    def test_stats_populated(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        results = cls.classify([_make_input([1.0, 0.0, 0.0])])
        assert "latency_ms" in results[0].stats
        assert results[0].stats["embedding_dim"] == 3
        assert results[0].stats["num_labels"] == 3


# ---------------------------------------------------------------------------
# Multi-label and thresholds
# ---------------------------------------------------------------------------


class TestSimilarityMultiLabel:
    def test_multi_label_with_low_threshold(self) -> None:
        ls = LabelSpace(
            labels=["billing", "cancel", "refund"],
            default_threshold=0.3,
        )
        cls = EmbeddingSimilarityClassifier(
            label_space=ls,
            centroids=_make_centroids(),
        )
        # Mix of billing and cancel — both should exceed 0.3
        results = cls.classify([_make_input([1.0, 1.0, 0.0])])
        predicted = results[0].predicted_labels
        assert "billing" in predicted
        assert "cancel" in predicted
        assert "refund" not in predicted

    def test_per_label_threshold(self) -> None:
        ls = LabelSpace(
            labels=["billing", "cancel", "refund"],
            thresholds={"billing": 0.9},
            default_threshold=0.3,
        )
        cls = EmbeddingSimilarityClassifier(
            label_space=ls,
            centroids=_make_centroids(),
        )
        # billing is close but below 0.9 threshold
        results = cls.classify([_make_input([1.0, 1.0, 0.0])])
        predicted = results[0].predicted_labels
        assert "billing" not in predicted
        assert "cancel" in predicted


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSimilarityErrors:
    def test_rejects_missing_embedding(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        inp = ClassificationInput(
            source_id="w0",
            source_type="context_window",
            text="test",
            embedding=None,
        )
        with pytest.raises(ModelError, match="no embedding"):
            cls.classify([inp])

    def test_rejects_wrong_embedding_dimension(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        with pytest.raises(ModelError, match="dimension mismatch"):
            cls.classify([_make_input([1.0, 0.0])])  # 2D instead of 3D


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestSimilarityDeterminism:
    def test_same_input_produces_same_output(self) -> None:
        cls = EmbeddingSimilarityClassifier(
            label_space=_make_label_space(),
            centroids=_make_centroids(),
        )
        inp = _make_input([0.5, 0.3, 0.8])
        r1 = cls.classify([inp])
        r2 = cls.classify([inp])
        assert r1[0].top_label == r2[0].top_label
        assert r1[0].top_score == r2[0].top_score


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestSimilarityClassifierReexport:
    def test_importable_from_classification_package(self) -> None:
        from semantic_conversation_engine.classification import (
            EmbeddingSimilarityClassifier as ESC,
        )

        assert ESC is EmbeddingSimilarityClassifier
