"""Classification data types — feature vectors, label scores, and results.

Pipeline-internal data objects for the classification stage.
These are NOT domain entities — they are payloads exchanged between
classification components.

ClassificationInput packages text with pre-computed features for a classifier.
LabelScore carries a label prediction with its score.
ClassificationResult is the envelope carrying predictions and execution metadata.

Score semantics (stable contract):
    score:      float in [0.0, 1.0] — classifier output (probability or similarity)
    confidence: float in [0.0, 1.0] — calibrated confidence (may equal score if uncalibrated)
    threshold:  float in [0.0, 1.0] — decision boundary for positive prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClassificationInput:
    """Input to a classifier — text with pre-computed features.

    Args:
        source_id: ID of the object being classified (turn_id, window_id, etc.).
        source_type: Granularity level (turn, context_window, conversation).
        text: Text content to classify.
        embedding: Pre-computed embedding vector. None if not available.
        features: Additional pre-computed features (lexical, structural, etc.).
        metadata: Additional context for classification.
    """

    source_id: str
    source_type: str
    text: str
    embedding: list[float] | None = None
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelScore:
    """A single label prediction with score and threshold.

    Args:
        label: Predicted label name.
        score: Classifier output in [0.0, 1.0].
        confidence: Calibrated confidence in [0.0, 1.0].
        threshold: Decision threshold for this label.
    """

    label: str
    score: float
    confidence: float
    threshold: float = 0.5

    @property
    def is_positive(self) -> bool:
        """Whether this prediction meets the threshold."""
        return self.score >= self.threshold


@dataclass(frozen=True)
class ClassificationResult:
    """Envelope for classification results on a single input.

    Carries all label scores (not just positive ones) for observability.
    The primary predictions are filtered by threshold.

    Args:
        source_id: ID of the classified object.
        source_type: Granularity level of the classified object.
        label_scores: All label scores, sorted by score descending.
        model_name: Name of the classifier that produced these scores.
        model_version: Version of the classifier.
        stats: Operational statistics (timing, feature counts, etc.).
    """

    source_id: str
    source_type: str
    label_scores: list[LabelScore]
    model_name: str
    model_version: str
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def predicted_labels(self) -> list[str]:
        """Labels that meet their respective thresholds."""
        return [ls.label for ls in self.label_scores if ls.is_positive]

    @property
    def top_label(self) -> str | None:
        """Highest-scoring label, or None if no scores."""
        if not self.label_scores:
            return None
        return self.label_scores[0].label

    @property
    def top_score(self) -> float | None:
        """Score of the highest-scoring label, or None if no scores."""
        if not self.label_scores:
            return None
        return self.label_scores[0].score
