"""Embedding similarity classifier.

Classifies inputs by computing cosine similarity between the input
embedding and pre-computed label centroids. Zero training required —
centroids are provided at construction time.

This is a strong baseline for intent classification where labeled
embeddings are available. It naturally supports multi-label: any
label whose similarity exceeds its threshold is predicted.

Algorithm:
    query_embedding → l2_normalize
    ↓
    cosine_similarity(query, centroid) for each label
    ↓
    LabelScore per label (sorted by score descending)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from talkex.classification.labels import LabelSpace
from talkex.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from talkex.exceptions import ModelError


class EmbeddingSimilarityClassifier:
    """Cosine similarity classifier using label centroids.

    Computes cosine similarity between input embeddings and
    pre-computed label centroid vectors. No training step needed —
    centroids are provided directly (e.g., from averaged labeled embeddings).

    Implements the Classifier protocol.

    Args:
        label_space: Label space defining labels and thresholds.
        centroids: Mapping from label name to centroid vector.
            Must contain an entry for every label in label_space.
        model_name: Name for this classifier instance.
        model_version: Version string for reproducibility.
    """

    def __init__(
        self,
        *,
        label_space: LabelSpace,
        centroids: dict[str, list[float]],
        model_name: str = "embedding-similarity",
        model_version: str = "1.0",
    ) -> None:
        missing = set(label_space.labels) - set(centroids)
        if missing:
            raise ModelError(
                f"Missing centroids for labels: {sorted(missing)}",
                context={"missing_labels": sorted(missing)},
            )
        self._label_space = label_space
        self._model_name = model_name
        self._model_version = model_version

        # Build centroid matrix: (num_labels, embedding_dim)
        dims = {len(v) for v in centroids.values()}
        if len(dims) != 1:
            raise ModelError(
                "All centroids must have the same dimensionality",
                context={"dimensions_found": sorted(dims)},
            )
        self._dim = dims.pop()
        self._centroid_matrix = np.array(
            [centroids[label] for label in label_space.labels],
            dtype=np.float32,
        )
        # Pre-normalize centroids for cosine similarity
        norms = np.linalg.norm(self._centroid_matrix, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-10)
        self._centroid_matrix = self._centroid_matrix / safe_norms

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        """Classify inputs by cosine similarity to label centroids.

        Args:
            inputs: Classification inputs. Each must have a non-None embedding.

        Returns:
            List of ClassificationResult, one per input.

        Raises:
            ModelError: If any input lacks an embedding or has wrong dimensions.
        """
        t0 = time.perf_counter()
        results: list[ClassificationResult] = []

        for inp in inputs:
            if inp.embedding is None:
                raise ModelError(
                    f"Input '{inp.source_id}' has no embedding",
                    context={"source_id": inp.source_id},
                )
            if len(inp.embedding) != self._dim:
                raise ModelError(
                    f"Embedding dimension mismatch: expected {self._dim}, got {len(inp.embedding)}",
                    context={
                        "source_id": inp.source_id,
                        "expected_dim": self._dim,
                        "actual_dim": len(inp.embedding),
                    },
                )

            query = np.array(inp.embedding, dtype=np.float32)
            query_norm = float(np.linalg.norm(query))
            safe_norm = max(query_norm, 1e-10)
            query_normalized = query / safe_norm

            # Cosine similarity: dot product of normalized vectors
            similarities: NDArray[np.float32] = self._centroid_matrix @ query_normalized

            label_scores = self._build_label_scores(similarities)

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

    def _build_label_scores(self, similarities: NDArray[np.float32]) -> list[LabelScore]:
        """Build sorted LabelScore list from raw similarities."""
        scores: list[LabelScore] = []
        for i, label in enumerate(self._label_space.labels):
            sim = float(similarities[i])
            # Clamp to [0, 1] — negative cosine treated as 0
            clamped = max(0.0, min(1.0, sim))
            threshold = self._label_space.threshold_for(label)
            scores.append(
                LabelScore(
                    label=label,
                    score=clamped,
                    confidence=clamped,
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
            "embedding_dim": self._dim,
            "num_labels": self._label_space.size,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    @property
    def label_space(self) -> LabelSpace:
        """The label space used by this classifier."""
        return self._label_space
