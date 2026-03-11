"""MLP (Multi-Layer Perceptron) classifier for feature-based classification.

Wraps scikit-learn's MLPClassifier behind the Classifier protocol.
Suitable for dense feature vectors (embeddings), serving as a neural
baseline between logistic regression and more complex architectures.

scikit-learn is an optional dependency imported conditionally.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from talkex.classification.labels import LabelSpace
from talkex.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from talkex.exceptions import ModelError


def _import_sklearn_mlp() -> Any:
    """Conditionally import scikit-learn's MLPClassifier.

    Returns:
        The sklearn.neural_network module.

    Raises:
        ModelError: If scikit-learn is not installed.
    """
    try:
        from sklearn import neural_network

        return neural_network
    except ImportError:
        raise ModelError(
            "scikit-learn is required for MLPClassifier. Install it with: pip install scikit-learn"
        ) from None


class MLPClassifier:
    """Multi-layer perceptron classifier using feature vectors.

    Wraps sklearn MLPClassifier with a 2-hidden-layer architecture.
    Operates on feature vectors extracted from ClassificationInput.features dict.

    Implements the Classifier protocol.

    Args:
        label_space: Label space defining labels and thresholds.
        feature_names: Ordered feature names matching training data.
        model_name: Name for this classifier instance.
        model_version: Version string for reproducibility.
        hidden_layer_sizes: Tuple of hidden layer sizes. Default: (128, 64).
        sklearn_kwargs: Additional keyword arguments passed to
            sklearn.neural_network.MLPClassifier.
    """

    def __init__(
        self,
        *,
        label_space: LabelSpace,
        feature_names: list[str],
        model_name: str = "mlp",
        model_version: str = "1.0",
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        sklearn_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._label_space = label_space
        self._feature_names = list(feature_names)
        self._model_name = model_name
        self._model_version = model_version
        self._is_fitted = False

        neural_network = _import_sklearn_mlp()
        kwargs: dict[str, Any] = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1,
        }
        if sklearn_kwargs:
            kwargs.update(sklearn_kwargs)
        self._model: Any = neural_network.MLPClassifier(**kwargs)

    def fit(
        self,
        inputs: list[ClassificationInput],
        labels: list[str],
    ) -> dict[str, Any]:
        """Train the classifier on labeled data.

        Args:
            inputs: Training inputs with features populated.
            labels: Target label for each input. Must all be in label_space.

        Returns:
            Training stats dict with sample_count, unique_labels, etc.

        Raises:
            ModelError: If inputs/labels are empty, mismatched, or contain
                unknown labels.
        """
        if not inputs or not labels:
            raise ModelError("Training requires non-empty inputs and labels")
        if len(inputs) != len(labels):
            raise ModelError(f"Input/label count mismatch: {len(inputs)} inputs, {len(labels)} labels")
        unknown = set(labels) - set(self._label_space.labels)
        if unknown:
            raise ModelError(
                f"Unknown labels in training data: {sorted(unknown)}",
                context={"unknown_labels": sorted(unknown)},
            )

        t0 = time.perf_counter()
        x_matrix = self._build_feature_matrix(inputs)
        y_indices = np.array(
            [self._label_space.label_index(lbl) for lbl in labels],
            dtype=np.int64,
        )

        self._model.fit(x_matrix, y_indices)
        self._is_fitted = True

        return {
            "sample_count": len(inputs),
            "unique_labels": len(set(labels)),
            "feature_count": len(self._feature_names),
            "training_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        """Classify inputs using the fitted MLP model.

        Args:
            inputs: Classification inputs with features populated.

        Returns:
            List of ClassificationResult, one per input.

        Raises:
            ModelError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise ModelError("MLPClassifier must be fitted before classify()")

        t0 = time.perf_counter()
        x_matrix = self._build_feature_matrix(inputs)
        probabilities = self._model.predict_proba(x_matrix)

        sklearn_classes: list[int] = list(self._model.classes_)

        results: list[ClassificationResult] = []
        for i, inp in enumerate(inputs):
            label_scores = self._build_label_scores(probabilities[i], sklearn_classes)
            results.append(
                ClassificationResult(
                    source_id=inp.source_id,
                    source_type=inp.source_type,
                    label_scores=label_scores,
                    model_name=self._model_name,
                    model_version=self._model_version,
                    stats=self._build_stats(t0, len(inputs)),
                )
            )

        return results

    def _build_feature_matrix(self, inputs: list[ClassificationInput]) -> np.ndarray:
        """Build numpy feature matrix from inputs."""
        rows: list[list[float]] = []
        for inp in inputs:
            row = [inp.features.get(name, 0.0) for name in self._feature_names]
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def _build_label_scores(
        self,
        probas: np.ndarray,
        sklearn_classes: list[int],
    ) -> list[LabelScore]:
        """Build sorted LabelScore list from predicted probabilities."""
        scores: list[LabelScore] = []
        for label in self._label_space.labels:
            idx = self._label_space.label_index(label)
            if idx in sklearn_classes:
                proba_idx = sklearn_classes.index(idx)
                prob = float(probas[proba_idx])
            else:
                prob = 0.0
            threshold = self._label_space.threshold_for(label)
            scores.append(
                LabelScore(
                    label=label,
                    score=prob,
                    confidence=prob,
                    threshold=threshold,
                )
            )
        scores.sort(key=lambda ls: (-ls.score, ls.label))
        return scores

    def _build_stats(self, t0: float, batch_size: int) -> dict[str, Any]:
        """Build operational stats."""
        return {
            "classifier": self._model_name,
            "batch_size": batch_size,
            "feature_count": len(self._feature_names),
            "num_labels": self._label_space.size,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted

    @property
    def label_space(self) -> LabelSpace:
        """The label space used by this classifier."""
        return self._label_space
