"""Prediction model — a classification result with evidence and confidence.

A Prediction represents the output of a classifier applied to a text object
(turn, context window, or conversation). It carries the predicted label,
score, confidence, and threshold, enabling downstream stages to make
informed decisions and produce auditable results.

See PRD §11 for the data model specification.
See ADR-002 for the frozen/strict design decision.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from talkex.models.enums import ObjectType
from talkex.models.types import PredictionId


class Prediction(BaseModel):
    """A classification prediction with score, confidence, and threshold.

    Args:
        prediction_id: Unique identifier. Format: pred_<uuid4>.
        source_id: ID of the classified object (turn, window, or conversation).
        source_type: Granularity level of the classified object.
        label: Predicted classification label.
        score: Raw classifier output in [0.0, 1.0].
        confidence: Calibrated confidence in [0.0, 1.0].
        threshold: Decision threshold in [0.0, 1.0].
        model_name: Name of the classifier that produced this prediction.
        model_version: Version string of the classifier.
        metadata: Additional prediction metadata (feature importance, etc.).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    prediction_id: PredictionId
    source_id: str
    source_type: ObjectType
    label: str
    score: float
    confidence: float
    threshold: float
    model_name: str
    model_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prediction_id")
    @classmethod
    def prediction_id_must_not_be_empty(cls, v: PredictionId) -> PredictionId:
        """Reject empty or whitespace-only prediction IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("prediction_id must not be empty or whitespace-only")
        return v

    @field_validator("source_id")
    @classmethod
    def source_id_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only source IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("source_id must not be empty or whitespace-only")
        return v

    @field_validator("label")
    @classmethod
    def label_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only labels.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("label must not be empty or whitespace-only")
        return v

    @field_validator("score")
    @classmethod
    def score_must_be_in_unit_range(cls, v: float) -> float:
        """Score must be in [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_unit_range(cls, v: float) -> float:
        """Confidence must be in [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("threshold")
    @classmethod
    def threshold_must_be_in_unit_range(cls, v: float) -> float:
        """Threshold must be in [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("model_name")
    @classmethod
    def model_name_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only model names.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("model_name must not be empty or whitespace-only")
        return v

    @field_validator("model_version")
    @classmethod
    def model_version_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only model versions.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("model_version must not be empty or whitespace-only")
        return v

    @property
    def is_above_threshold(self) -> bool:
        """Whether the score meets or exceeds the decision threshold.

        This is a derived property, NOT a stored field. It is always
        consistent with the current score and threshold values.
        """
        return self.score >= self.threshold
