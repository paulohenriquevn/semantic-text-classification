"""Classification configuration — model identity and runtime parameters.

Defines configuration objects for classifier identity, feature extraction,
and classification runtime behavior. All configs are frozen pydantic models
for immutability and validation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClassificationLevel(StrEnum):
    """Granularity level at which classification operates.

    TURN: classify individual turns.
    WINDOW: classify context windows (primary use case).
    CONVERSATION: classify entire conversations.
    """

    TURN = "turn"
    WINDOW = "window"
    CONVERSATION = "conversation"


class ClassificationMode(StrEnum):
    """Classification output mode.

    SINGLE_LABEL: exactly one label per input (argmax).
    MULTI_LABEL: zero or more labels per input (threshold-based).
    """

    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"


class ClassifierConfig(BaseModel):
    """Configuration identifying a classifier and its behavior.

    Args:
        model_name: Unique name for this classifier.
        model_version: Version string for reproducibility.
        classification_mode: Single-label or multi-label.
        classification_level: Turn, window, or conversation level.
        default_threshold: Decision threshold for positive predictions.
        labels: Ordered list of known labels. Index corresponds to
            model output position.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    model_name: str
    model_version: str
    classification_mode: ClassificationMode = ClassificationMode.SINGLE_LABEL
    classification_level: ClassificationLevel = ClassificationLevel.WINDOW
    default_threshold: float = 0.5
    labels: list[str] = Field(default_factory=list)

    @field_validator("model_name")
    @classmethod
    def model_name_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only model names."""
        if not v.strip():
            raise ValueError("model_name must not be empty or whitespace-only")
        return v

    @field_validator("model_version")
    @classmethod
    def model_version_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only model versions."""
        if not v.strip():
            raise ValueError("model_version must not be empty or whitespace-only")
        return v

    @field_validator("default_threshold")
    @classmethod
    def threshold_must_be_in_unit_range(cls, v: float) -> float:
        """Threshold must be in [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"default_threshold must be in [0.0, 1.0], got {v}")
        return v
