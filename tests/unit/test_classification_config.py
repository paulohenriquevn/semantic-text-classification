"""Unit tests for classification configuration.

Tests cover: ClassifierConfig construction, validation, enums,
immutability, and reexport.
"""

import pytest

from semantic_conversation_engine.classification.config import (
    ClassificationLevel,
    ClassificationMode,
    ClassifierConfig,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestClassificationEnums:
    def test_classification_level_values(self) -> None:
        assert ClassificationLevel.TURN == "turn"
        assert ClassificationLevel.WINDOW == "window"
        assert ClassificationLevel.CONVERSATION == "conversation"

    def test_classification_mode_values(self) -> None:
        assert ClassificationMode.SINGLE_LABEL == "single_label"
        assert ClassificationMode.MULTI_LABEL == "multi_label"


# ---------------------------------------------------------------------------
# ClassifierConfig construction
# ---------------------------------------------------------------------------


class TestClassifierConfigConstruction:
    def test_creates_with_required_fields(self) -> None:
        config = ClassifierConfig(
            model_name="intent-v1",
            model_version="1.0",
        )
        assert config.model_name == "intent-v1"
        assert config.model_version == "1.0"
        assert config.classification_mode == ClassificationMode.SINGLE_LABEL
        assert config.classification_level == ClassificationLevel.WINDOW
        assert config.default_threshold == 0.5
        assert config.labels == []

    def test_creates_with_all_fields(self) -> None:
        config = ClassifierConfig(
            model_name="topic-v2",
            model_version="2.1",
            classification_mode=ClassificationMode.MULTI_LABEL,
            classification_level=ClassificationLevel.CONVERSATION,
            default_threshold=0.3,
            labels=["billing", "cancel", "refund"],
        )
        assert config.classification_mode == ClassificationMode.MULTI_LABEL
        assert config.classification_level == ClassificationLevel.CONVERSATION
        assert config.default_threshold == 0.3
        assert config.labels == ["billing", "cancel", "refund"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestClassifierConfigValidation:
    def test_rejects_empty_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            ClassifierConfig(model_name="", model_version="1.0")

    def test_rejects_whitespace_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            ClassifierConfig(model_name="  ", model_version="1.0")

    def test_rejects_empty_model_version(self) -> None:
        with pytest.raises(ValueError, match="model_version"):
            ClassifierConfig(model_name="test", model_version="")

    def test_rejects_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError, match="default_threshold"):
            ClassifierConfig(
                model_name="test", model_version="1.0", default_threshold=-0.1
            )

    def test_rejects_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="default_threshold"):
            ClassifierConfig(
                model_name="test", model_version="1.0", default_threshold=1.1
            )

    def test_accepts_threshold_at_boundaries(self) -> None:
        c0 = ClassifierConfig(
            model_name="test", model_version="1.0", default_threshold=0.0
        )
        c1 = ClassifierConfig(
            model_name="test", model_version="1.0", default_threshold=1.0
        )
        assert c0.default_threshold == 0.0
        assert c1.default_threshold == 1.0


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestClassifierConfigImmutability:
    def test_is_frozen(self) -> None:
        config = ClassifierConfig(model_name="test", model_version="1.0")
        with pytest.raises(Exception):
            config.model_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestClassificationConfigReexport:
    def test_importable_from_classification_package(self) -> None:
        from semantic_conversation_engine.classification import (
            ClassificationLevel as CL,
            ClassificationMode as CM,
            ClassifierConfig as CC,
        )

        assert CL is ClassificationLevel
        assert CM is ClassificationMode
        assert CC is ClassifierConfig
