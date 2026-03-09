"""Classifier serialization — save and load trained classifiers.

Provides save/load functions for classifier state. Uses JSON for
metadata and numpy for array data, following the same pattern as
InMemoryVectorIndex persistence.

Each serialized classifier is stored as a directory containing:
    metadata.json  — label space, feature names, model identity
    model_data.npz — numpy arrays (centroids or sklearn coefficients)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from semantic_conversation_engine.classification.labels import LabelSpace
from semantic_conversation_engine.classification.similarity import (
    EmbeddingSimilarityClassifier,
)
from semantic_conversation_engine.exceptions import ModelError


def save_similarity_classifier(
    classifier: EmbeddingSimilarityClassifier,
    path: str | Path,
) -> None:
    """Save an EmbeddingSimilarityClassifier to disk.

    Creates a directory at path containing metadata.json and centroids.npy.

    Args:
        classifier: The classifier to save.
        path: Directory path for the saved model.

    Raises:
        ModelError: If saving fails.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)

    label_space = classifier.label_space
    metadata: dict[str, Any] = {
        "classifier_type": "embedding_similarity",
        "model_name": classifier._model_name,
        "model_version": classifier._model_version,
        "labels": label_space.labels,
        "thresholds": label_space.thresholds,
        "default_threshold": label_space.default_threshold,
        "embedding_dim": classifier._dim,
    }

    meta_path = dir_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    centroids_path = dir_path / "centroids.npy"
    np.save(str(centroids_path), classifier._centroid_matrix)


def load_similarity_classifier(
    path: str | Path,
) -> EmbeddingSimilarityClassifier:
    """Load an EmbeddingSimilarityClassifier from disk.

    Args:
        path: Directory path containing saved model files.

    Returns:
        Reconstructed classifier.

    Raises:
        ModelError: If files are missing, corrupted, or incompatible.
    """
    dir_path = Path(path)
    meta_path = dir_path / "metadata.json"
    centroids_path = dir_path / "centroids.npy"

    if not meta_path.exists():
        raise ModelError(
            f"Missing metadata.json in {dir_path}",
            context={"path": str(dir_path)},
        )
    if not centroids_path.exists():
        raise ModelError(
            f"Missing centroids.npy in {dir_path}",
            context={"path": str(dir_path)},
        )

    metadata = json.loads(meta_path.read_text())

    if metadata.get("classifier_type") != "embedding_similarity":
        raise ModelError(
            f"Expected classifier_type 'embedding_similarity', got '{metadata.get('classifier_type')}'",
            context={"path": str(dir_path)},
        )

    label_space = LabelSpace(
        labels=metadata["labels"],
        thresholds={k: v for k, v in metadata.get("thresholds", {}).items()},
        default_threshold=metadata.get("default_threshold", 0.5),
    )

    # Load centroids and reconstruct dict
    centroid_matrix = np.load(str(centroids_path)).astype(np.float32)
    # Denormalize is not needed — we pass raw centroids and let
    # the constructor re-normalize. But stored centroids are already
    # normalized, so we reconstruct from them directly.
    centroids: dict[str, list[float]] = {}
    for i, label in enumerate(label_space.labels):
        centroids[label] = centroid_matrix[i].tolist()

    return EmbeddingSimilarityClassifier(
        label_space=label_space,
        centroids=centroids,
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
    )
