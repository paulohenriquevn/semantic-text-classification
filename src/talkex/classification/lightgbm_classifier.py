"""LightGBM classifier for feature-based classification.

Wraps LightGBM's LGBMClassifier behind the Classifier protocol.
Excels at heterogeneous features (embeddings + lexical + structural),
making it ideal for multi-level conversation classification.

LightGBM is an optional dependency imported conditionally.
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


def _import_lightgbm() -> Any:
    """Conditionally import LightGBM.

    Returns:
        The lightgbm module.

    Raises:
        ModelError: If LightGBM is not installed.
    """
    try:
        import lightgbm as lgb

        return lgb
    except ImportError:
        raise ModelError("lightgbm is required for LightGBMClassifier. Install it with: pip install lightgbm") from None


class LightGBMClassifier:
    """Gradient boosting classifier using feature vectors.

    Wraps LightGBM LGBMClassifier. Operates on feature vectors
    extracted from ClassificationInput.features dict.

    Implements the Classifier protocol.

    Args:
        label_space: Label space defining labels and thresholds.
        feature_names: Ordered feature names matching training data.
        model_name: Name for this classifier instance.
        model_version: Version string for reproducibility.
        lgbm_kwargs: Additional keyword arguments passed to
            lightgbm.LGBMClassifier.
    """

    def __init__(
        self,
        *,
        label_space: LabelSpace,
        feature_names: list[str],
        model_name: str = "lightgbm",
        model_version: str = "1.0",
        lgbm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._label_space = label_space
        self._feature_names = list(feature_names)
        self._model_name = model_name
        self._model_version = model_version
        self._is_fitted = False

        lgb = _import_lightgbm()
        kwargs: dict[str, Any] = {
            "n_estimators": 100,
            "num_leaves": 31,
            "verbosity": -1,
            "random_state": 42,
        }
        if lgbm_kwargs:
            kwargs.update(lgbm_kwargs)
        self._model: Any = lgb.LGBMClassifier(**kwargs)

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
        """Classify inputs using the fitted LightGBM model.

        Args:
            inputs: Classification inputs with features populated.

        Returns:
            List of ClassificationResult, one per input.

        Raises:
            ModelError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise ModelError("LightGBMClassifier must be fitted before classify()")

        t0 = time.perf_counter()
        x_matrix = self._build_feature_matrix(inputs)
        probabilities = self._model.predict_proba(x_matrix)

        lgbm_classes: list[int] = list(self._model.classes_)

        results: list[ClassificationResult] = []
        for i, inp in enumerate(inputs):
            label_scores = self._build_label_scores(probabilities[i], lgbm_classes)
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
        lgbm_classes: list[int],
    ) -> list[LabelScore]:
        """Build sorted LabelScore list from predicted probabilities."""
        scores: list[LabelScore] = []
        for label in self._label_space.labels:
            idx = self._label_space.label_index(label)
            if idx in lgbm_classes:
                proba_idx = lgbm_classes.index(idx)
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
