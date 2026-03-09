"""Unit tests for classifier serialization.

Tests cover: save/load round-trip for EmbeddingSimilarityClassifier,
error handling for missing files, and reexport.
"""

from pathlib import Path

import pytest

from semantic_conversation_engine.classification.labels import LabelSpace
from semantic_conversation_engine.classification.models import ClassificationInput
from semantic_conversation_engine.classification.serialization import (
    load_similarity_classifier,
    save_similarity_classifier,
)
from semantic_conversation_engine.classification.similarity import (
    EmbeddingSimilarityClassifier,
)
from semantic_conversation_engine.exceptions import ModelError


def _make_classifier() -> EmbeddingSimilarityClassifier:
    ls = LabelSpace(
        labels=["billing", "cancel"],
        thresholds={"billing": 0.7},
        default_threshold=0.5,
    )
    return EmbeddingSimilarityClassifier(
        label_space=ls,
        centroids={
            "billing": [1.0, 0.0, 0.0],
            "cancel": [0.0, 1.0, 0.0],
        },
        model_name="test-sim",
        model_version="1.0",
    )


# ---------------------------------------------------------------------------
# Save and load round-trip
# ---------------------------------------------------------------------------


class TestSimilaritySerialization:
    def test_save_creates_files(self, tmp_path: Path) -> None:
        cls = _make_classifier()
        save_similarity_classifier(cls, tmp_path / "model")
        assert (tmp_path / "model" / "metadata.json").exists()
        assert (tmp_path / "model" / "centroids.npy").exists()

    def test_round_trip_preserves_classification(self, tmp_path: Path) -> None:
        original = _make_classifier()
        save_similarity_classifier(original, tmp_path / "model")
        loaded = load_similarity_classifier(tmp_path / "model")

        inp = ClassificationInput(
            source_id="w0",
            source_type="context_window",
            text="test",
            embedding=[1.0, 0.0, 0.0],
        )
        original_results = original.classify([inp])
        loaded_results = loaded.classify([inp])

        assert original_results[0].top_label == loaded_results[0].top_label
        assert abs((original_results[0].top_score or 0) - (loaded_results[0].top_score or 0)) < 1e-6

    def test_round_trip_preserves_label_space(self, tmp_path: Path) -> None:
        original = _make_classifier()
        save_similarity_classifier(original, tmp_path / "model")
        loaded = load_similarity_classifier(tmp_path / "model")

        assert loaded.label_space.labels == original.label_space.labels
        assert loaded.label_space.thresholds == original.label_space.thresholds
        assert loaded.label_space.default_threshold == original.label_space.default_threshold

    def test_round_trip_preserves_thresholds(self, tmp_path: Path) -> None:
        original = _make_classifier()
        save_similarity_classifier(original, tmp_path / "model")
        loaded = load_similarity_classifier(tmp_path / "model")

        assert loaded.label_space.threshold_for("billing") == 0.7
        assert loaded.label_space.threshold_for("cancel") == 0.5

    def test_round_trip_preserves_all_label_scores(self, tmp_path: Path) -> None:
        original = _make_classifier()
        save_similarity_classifier(original, tmp_path / "model")
        loaded = load_similarity_classifier(tmp_path / "model")

        inp = ClassificationInput(
            source_id="w0",
            source_type="context_window",
            text="test",
            embedding=[0.7, 0.3, 0.0],
        )
        original_scores = {ls.label: ls.score for ls in original.classify([inp])[0].label_scores}
        loaded_scores = {ls.label: ls.score for ls in loaded.classify([inp])[0].label_scores}

        for label in original_scores:
            assert abs(original_scores[label] - loaded_scores[label]) < 1e-5


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSerializationErrors:
    def test_load_missing_metadata(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError, match="Missing metadata"):
            load_similarity_classifier(tmp_path / "nonexistent")

    def test_load_missing_centroids(self, tmp_path: Path) -> None:
        # Create metadata but no centroids
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text('{"classifier_type": "embedding_similarity"}')
        with pytest.raises(ModelError, match="Missing centroids"):
            load_similarity_classifier(model_dir)

    def test_load_wrong_classifier_type(self, tmp_path: Path) -> None:
        import json

        import numpy as np

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        meta = {"classifier_type": "random_forest", "labels": ["a"], "model_name": "x", "model_version": "1"}
        (model_dir / "metadata.json").write_text(json.dumps(meta))
        np.save(str(model_dir / "centroids.npy"), np.array([[1.0]]))
        with pytest.raises(ModelError, match="Expected classifier_type"):
            load_similarity_classifier(model_dir)


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestSerializationReexport:
    def test_save_importable_from_classification_package(self) -> None:
        from semantic_conversation_engine.classification import (
            save_similarity_classifier as save_fn,
        )

        assert save_fn is save_similarity_classifier

    def test_load_importable_from_classification_package(self) -> None:
        from semantic_conversation_engine.classification import (
            load_similarity_classifier as load_fn,
        )

        assert load_fn is load_similarity_classifier
