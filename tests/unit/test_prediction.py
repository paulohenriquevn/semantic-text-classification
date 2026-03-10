"""Unit tests for the Prediction model — follows the Conversation golden template.

Tests cover: construction, validation (IDs, strings, score/confidence/threshold
ranges), is_above_threshold property, strict mode behavior, immutability,
serialization round-trip, and re-export.
"""

from typing import Any

import pytest

from talkex.models.enums import ObjectType
from talkex.models.prediction import Prediction
from talkex.models.types import PredictionId

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_prediction(**overrides: object) -> Prediction:
    """Factory with sensible defaults. Override any field via kwargs."""
    defaults: dict[str, Any] = {
        "prediction_id": PredictionId("pred_abc123"),
        "source_id": "turn_abc123",
        "source_type": ObjectType.TURN,
        "label": "billing_dispute",
        "score": 0.85,
        "confidence": 0.90,
        "threshold": 0.70,
        "model_name": "logistic_regression_v1",
        "model_version": "1.0.0",
    }
    defaults.update(overrides)
    return Prediction(**defaults)


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestPredictionConstruction:
    def test_creates_with_required_fields_only(self) -> None:
        pred = _make_prediction()
        assert pred.prediction_id == "pred_abc123"
        assert pred.source_id == "turn_abc123"
        assert pred.source_type == ObjectType.TURN
        assert pred.label == "billing_dispute"
        assert pred.score == 0.85
        assert pred.confidence == 0.90
        assert pred.threshold == 0.70
        assert pred.model_name == "logistic_regression_v1"
        assert pred.model_version == "1.0.0"
        assert pred.metadata == {}

    def test_creates_with_all_fields(self) -> None:
        pred = _make_prediction(
            metadata={"feature_importance": {"embedding_0": 0.3}},
        )
        assert pred.metadata == {"feature_importance": {"embedding_0": 0.3}}

    def test_metadata_defaults_to_empty_dict(self) -> None:
        pred = _make_prediction()
        assert pred.metadata == {}
        assert isinstance(pred.metadata, dict)


# ---------------------------------------------------------------------------
# Validation — IDs and strings
# ---------------------------------------------------------------------------


class TestPredictionIdValidation:
    def test_rejects_empty_prediction_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_prediction(prediction_id=PredictionId(""))

    def test_rejects_whitespace_only_prediction_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_prediction(prediction_id=PredictionId("   "))

    def test_preserves_prediction_id_without_normalizing(self) -> None:
        padded_id = PredictionId("  pred_123  ")
        pred = _make_prediction(prediction_id=padded_id)
        assert pred.prediction_id == "  pred_123  "

    def test_rejects_empty_source_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_prediction(source_id="")

    def test_rejects_empty_label(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_prediction(label="")

    def test_rejects_whitespace_only_label(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_prediction(label="   ")

    def test_rejects_empty_model_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_prediction(model_name="")

    def test_rejects_empty_model_version(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_prediction(model_version="")


# ---------------------------------------------------------------------------
# Validation — score, confidence, threshold ranges
# ---------------------------------------------------------------------------


class TestPredictionRangeValidation:
    # --- score ---
    def test_rejects_negative_score(self) -> None:
        with pytest.raises(ValueError, match="score must be in"):
            _make_prediction(score=-0.1)

    def test_rejects_score_above_one(self) -> None:
        with pytest.raises(ValueError, match="score must be in"):
            _make_prediction(score=1.1)

    def test_accepts_score_zero(self) -> None:
        pred = _make_prediction(score=0.0)
        assert pred.score == 0.0

    def test_accepts_score_one(self) -> None:
        pred = _make_prediction(score=1.0)
        assert pred.score == 1.0

    # --- confidence ---
    def test_rejects_negative_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            _make_prediction(confidence=-0.1)

    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            _make_prediction(confidence=1.1)

    def test_accepts_confidence_zero(self) -> None:
        pred = _make_prediction(confidence=0.0)
        assert pred.confidence == 0.0

    def test_accepts_confidence_one(self) -> None:
        pred = _make_prediction(confidence=1.0)
        assert pred.confidence == 1.0

    # --- threshold ---
    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            _make_prediction(threshold=-0.1)

    def test_rejects_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            _make_prediction(threshold=1.1)

    def test_accepts_threshold_zero(self) -> None:
        pred = _make_prediction(threshold=0.0)
        assert pred.threshold == 0.0

    def test_accepts_threshold_one(self) -> None:
        pred = _make_prediction(threshold=1.0)
        assert pred.threshold == 1.0


# ---------------------------------------------------------------------------
# Property — is_above_threshold
# ---------------------------------------------------------------------------


class TestPredictionIsAboveThreshold:
    def test_true_when_score_exceeds_threshold(self) -> None:
        pred = _make_prediction(score=0.85, threshold=0.70)
        assert pred.is_above_threshold is True

    def test_true_when_score_equals_threshold(self) -> None:
        pred = _make_prediction(score=0.70, threshold=0.70)
        assert pred.is_above_threshold is True

    def test_false_when_score_below_threshold(self) -> None:
        pred = _make_prediction(score=0.60, threshold=0.70)
        assert pred.is_above_threshold is False

    def test_boundary_zero_score_zero_threshold(self) -> None:
        pred = _make_prediction(score=0.0, threshold=0.0)
        assert pred.is_above_threshold is True

    def test_boundary_one_score_one_threshold(self) -> None:
        pred = _make_prediction(score=1.0, threshold=1.0)
        assert pred.is_above_threshold is True

    def test_not_in_model_dump(self) -> None:
        """is_above_threshold is a property, not a stored field."""
        pred = _make_prediction()
        data = pred.model_dump()
        assert "is_above_threshold" not in data


# ---------------------------------------------------------------------------
# Strict mode — no coercion
# ---------------------------------------------------------------------------


class TestPredictionStrictMode:
    def test_rejects_string_source_type_coercion(self) -> None:
        """strict=True means 'turn' (str) won't coerce to ObjectType.TURN."""
        with pytest.raises(ValueError):
            _make_prediction(source_type="turn")

    def test_accepts_int_for_float_fields(self) -> None:
        """strict=True still allows int → float (lossless widening in Pydantic v2)."""
        pred = _make_prediction(score=1)
        assert pred.score == 1.0
        assert isinstance(pred.score, int | float)

    def test_rejects_int_for_prediction_id(self) -> None:
        """strict=True means int won't coerce to str."""
        with pytest.raises(ValueError):
            _make_prediction(prediction_id=12345)


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestPredictionImmutability:
    def test_cannot_assign_to_field(self) -> None:
        pred = _make_prediction()
        with pytest.raises(ValueError, match="frozen"):
            pred.prediction_id = PredictionId("pred_new")

    def test_cannot_assign_to_metadata(self) -> None:
        pred = _make_prediction()
        with pytest.raises(ValueError, match="frozen"):
            pred.metadata = {"new": "data"}


# ---------------------------------------------------------------------------
# Serialization — in-memory (types preserved)
# ---------------------------------------------------------------------------


class TestPredictionSerializationInMemory:
    """In-memory serialization: model_dump() preserves Python types."""

    def test_model_dump_produces_dict(self) -> None:
        pred = _make_prediction()
        data = pred.model_dump()
        assert isinstance(data, dict)
        assert data["prediction_id"] == "pred_abc123"
        assert data["label"] == "billing_dispute"

    def test_enum_serializes_as_value(self) -> None:
        pred = _make_prediction(source_type=ObjectType.CONTEXT_WINDOW)
        data = pred.model_dump()
        assert data["source_type"] == "context_window"
        assert isinstance(data["source_type"], str)

    def test_model_dump_mode_json_produces_json_safe_types(self) -> None:
        pred = _make_prediction(metadata={"key": "value"})
        data = pred.model_dump(mode="json")
        assert isinstance(data["score"], float)
        assert isinstance(data["source_type"], str)


# ---------------------------------------------------------------------------
# Boundary deserialization (dict/JSON → model, uses parsing)
# ---------------------------------------------------------------------------


class TestPredictionBoundaryDeserialization:
    """Boundary parsing: reconstruct model from dict or JSON.

    At system boundaries (API handlers, file parsers), data arrives as plain
    dicts. Pydantic's model_validate with strict=False handles coercion.
    This is the ONLY place where strict=False is acceptable (ADR-002).
    """

    def test_reconstructs_from_model_dump(self) -> None:
        pred = _make_prediction(metadata={"key": "value"})
        data = pred.model_dump()
        restored = Prediction.model_validate(data, strict=False)
        assert restored == pred

    def test_preserves_scores_through_boundary(self) -> None:
        pred = _make_prediction(score=0.92, confidence=0.88, threshold=0.75)
        data = pred.model_dump()
        restored = Prediction.model_validate(data, strict=False)
        assert restored.score == 0.92
        assert restored.confidence == 0.88
        assert restored.threshold == 0.75

    def test_preserves_metadata_through_boundary(self) -> None:
        meta = {"features": ["emb_0", "emb_1"], "calibration_method": "platt"}
        pred = _make_prediction(metadata=meta)
        data = pred.model_dump()
        restored = Prediction.model_validate(data, strict=False)
        assert restored.metadata == meta

    def test_reconstructs_from_json_mode_dump(self) -> None:
        """Proves full JSON round-trip: model → JSON dict → model."""
        pred = _make_prediction(metadata={"nested": {"deep": True}})
        json_data = pred.model_dump(mode="json")
        restored = Prediction.model_validate(json_data, strict=False)
        assert restored == pred


# ---------------------------------------------------------------------------
# Re-export from models package
# ---------------------------------------------------------------------------


class TestPredictionReexport:
    def test_importable_from_models_package(self) -> None:
        from talkex.models import Prediction as Imported

        assert Imported is Prediction
