"""Unit tests for LabelSpace.

Tests cover: construction, validation, threshold lookup, label index,
immutability, and reexport.
"""

import pytest

from semantic_conversation_engine.classification.labels import LabelSpace

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLabelSpaceConstruction:
    def test_creates_with_labels(self) -> None:
        ls = LabelSpace(labels=["billing", "cancel", "refund"])
        assert ls.labels == ["billing", "cancel", "refund"]
        assert ls.thresholds == {}
        assert ls.default_threshold == 0.5

    def test_creates_with_per_label_thresholds(self) -> None:
        ls = LabelSpace(
            labels=["billing", "cancel"],
            thresholds={"billing": 0.7},
            default_threshold=0.5,
        )
        assert ls.thresholds == {"billing": 0.7}

    def test_size_property(self) -> None:
        ls = LabelSpace(labels=["a", "b", "c"])
        assert ls.size == 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestLabelSpaceValidation:
    def test_rejects_empty_labels(self) -> None:
        with pytest.raises(ValueError, match="at least one label"):
            LabelSpace(labels=[])

    def test_rejects_duplicate_labels(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            LabelSpace(labels=["billing", "billing"])

    def test_rejects_unknown_threshold_label(self) -> None:
        with pytest.raises(ValueError, match="unknown labels"):
            LabelSpace(
                labels=["billing"],
                thresholds={"cancel": 0.5},
            )

    def test_rejects_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError, match="Threshold"):
            LabelSpace(
                labels=["billing"],
                thresholds={"billing": -0.1},
            )

    def test_rejects_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="Threshold"):
            LabelSpace(
                labels=["billing"],
                thresholds={"billing": 1.1},
            )

    def test_rejects_default_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError, match="default_threshold"):
            LabelSpace(labels=["billing"], default_threshold=-0.1)

    def test_rejects_default_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="default_threshold"):
            LabelSpace(labels=["billing"], default_threshold=1.1)

    def test_accepts_boundary_thresholds(self) -> None:
        ls = LabelSpace(
            labels=["billing"],
            thresholds={"billing": 0.0},
            default_threshold=1.0,
        )
        assert ls.threshold_for("billing") == 0.0
        assert ls.default_threshold == 1.0


# ---------------------------------------------------------------------------
# Threshold lookup
# ---------------------------------------------------------------------------


class TestLabelSpaceThresholdLookup:
    def test_returns_per_label_threshold(self) -> None:
        ls = LabelSpace(
            labels=["billing", "cancel"],
            thresholds={"billing": 0.7},
        )
        assert ls.threshold_for("billing") == 0.7

    def test_returns_default_when_no_per_label(self) -> None:
        ls = LabelSpace(
            labels=["billing", "cancel"],
            thresholds={"billing": 0.7},
            default_threshold=0.3,
        )
        assert ls.threshold_for("cancel") == 0.3

    def test_raises_for_unknown_label(self) -> None:
        ls = LabelSpace(labels=["billing"])
        with pytest.raises(ValueError, match="Unknown label"):
            ls.threshold_for("cancel")


# ---------------------------------------------------------------------------
# Label index
# ---------------------------------------------------------------------------


class TestLabelSpaceLabelIndex:
    def test_returns_correct_index(self) -> None:
        ls = LabelSpace(labels=["billing", "cancel", "refund"])
        assert ls.label_index("billing") == 0
        assert ls.label_index("cancel") == 1
        assert ls.label_index("refund") == 2

    def test_raises_for_unknown_label(self) -> None:
        ls = LabelSpace(labels=["billing"])
        with pytest.raises(ValueError, match="Unknown label"):
            ls.label_index("cancel")


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestLabelSpaceImmutability:
    def test_is_frozen(self) -> None:
        ls = LabelSpace(labels=["billing"])
        try:
            ls.labels = ["other"]  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestLabelSpaceReexport:
    def test_importable_from_classification_package(self) -> None:
        from semantic_conversation_engine.classification import (
            LabelSpace as LS,
        )

        assert LS is LabelSpace
