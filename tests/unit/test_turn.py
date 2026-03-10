"""Unit tests for the Turn model — follows the Conversation golden template.

Tests cover: construction, validation (IDs, text, offsets), cross-field validation,
strict mode behavior, immutability, serialization round-trip, and re-export.
"""

from typing import Any

import pytest

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_turn(**overrides: object) -> Turn:
    """Factory with sensible defaults. Override any field via kwargs."""
    defaults: dict[str, Any] = {
        "turn_id": TurnId("turn_abc123"),
        "conversation_id": ConversationId("conv_abc123"),
        "speaker": SpeakerRole.CUSTOMER,
        "raw_text": "I need help with my broadband connection",
        "start_offset": 0,
        "end_offset": 100,
    }
    defaults.update(overrides)
    return Turn(**defaults)


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestTurnConstruction:
    def test_creates_with_required_fields_only(self) -> None:
        turn = _make_turn()
        assert turn.turn_id == "turn_abc123"
        assert turn.conversation_id == "conv_abc123"
        assert turn.speaker == SpeakerRole.CUSTOMER
        assert turn.raw_text == "I need help with my broadband connection"
        assert turn.start_offset == 0
        assert turn.end_offset == 100
        assert turn.normalized_text is None
        assert turn.metadata == {}

    def test_creates_with_all_fields(self) -> None:
        turn = _make_turn(
            normalized_text="i need help with my broadband connection",
            metadata={"asr_confidence": 0.95, "language": "en"},
        )
        assert turn.normalized_text == "i need help with my broadband connection"
        assert turn.metadata == {"asr_confidence": 0.95, "language": "en"}

    def test_metadata_defaults_to_empty_dict(self) -> None:
        turn = _make_turn()
        assert turn.metadata == {}
        assert isinstance(turn.metadata, dict)


# ---------------------------------------------------------------------------
# Validation — IDs
# ---------------------------------------------------------------------------


class TestTurnIdValidation:
    def test_rejects_empty_turn_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_turn(turn_id=TurnId(""))

    def test_rejects_whitespace_only_turn_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_turn(turn_id=TurnId("   "))

    def test_accepts_valid_prefixed_turn_id(self) -> None:
        turn = _make_turn(turn_id=TurnId("turn_12345"))
        assert turn.turn_id == "turn_12345"

    def test_preserves_turn_id_without_normalizing(self) -> None:
        """Validator rejects invalid IDs but does NOT strip/normalize valid ones."""
        padded_id = TurnId("  turn_123  ")
        turn = _make_turn(turn_id=padded_id)
        assert turn.turn_id == "  turn_123  "

    def test_rejects_empty_conversation_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_turn(conversation_id=ConversationId(""))

    def test_rejects_whitespace_only_conversation_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_turn(conversation_id=ConversationId("   "))


# ---------------------------------------------------------------------------
# Validation — raw_text
# ---------------------------------------------------------------------------


class TestTurnTextValidation:
    def test_rejects_empty_raw_text(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_turn(raw_text="")

    def test_rejects_whitespace_only_raw_text(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_turn(raw_text="   ")

    def test_preserves_raw_text_without_normalizing(self) -> None:
        """Validator rejects invalid text but does NOT strip/normalize valid text."""
        padded_text = "  hello world  "
        turn = _make_turn(raw_text=padded_text)
        assert turn.raw_text == "  hello world  "


# ---------------------------------------------------------------------------
# Validation — offsets
# ---------------------------------------------------------------------------


class TestTurnOffsetValidation:
    def test_rejects_negative_start_offset(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _make_turn(start_offset=-1)

    def test_accepts_zero_start_offset(self) -> None:
        turn = _make_turn(start_offset=0)
        assert turn.start_offset == 0

    def test_rejects_end_offset_before_start_offset(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            _make_turn(start_offset=100, end_offset=50)

    def test_rejects_end_offset_equal_to_start_offset(self) -> None:
        """Zero-length spans are degenerate — a turn must span at least one unit."""
        with pytest.raises(ValueError, match="strictly greater"):
            _make_turn(start_offset=100, end_offset=100)

    def test_accepts_end_offset_after_start_offset(self) -> None:
        turn = _make_turn(start_offset=0, end_offset=200)
        assert turn.end_offset > turn.start_offset


# ---------------------------------------------------------------------------
# Strict mode — no coercion
# ---------------------------------------------------------------------------


class TestTurnStrictMode:
    def test_rejects_string_speaker_coercion(self) -> None:
        """strict=True means 'customer' (str) won't coerce to SpeakerRole.CUSTOMER."""
        with pytest.raises(ValueError):
            _make_turn(speaker="customer")

    def test_rejects_float_for_offset(self) -> None:
        """strict=True means float won't coerce to int."""
        with pytest.raises(ValueError):
            _make_turn(start_offset=0.5)

    def test_rejects_int_for_turn_id(self) -> None:
        """strict=True means int won't coerce to str."""
        with pytest.raises(ValueError):
            _make_turn(turn_id=12345)


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestTurnImmutability:
    def test_cannot_assign_to_field(self) -> None:
        turn = _make_turn()
        with pytest.raises(ValueError, match="frozen"):
            turn.turn_id = TurnId("turn_new")

    def test_cannot_assign_to_metadata(self) -> None:
        turn = _make_turn()
        with pytest.raises(ValueError, match="frozen"):
            turn.metadata = {"new": "data"}


# ---------------------------------------------------------------------------
# Serialization — in-memory (types preserved)
# ---------------------------------------------------------------------------


class TestTurnSerializationInMemory:
    """In-memory serialization: model_dump() preserves Python types."""

    def test_model_dump_produces_dict(self) -> None:
        turn = _make_turn()
        data = turn.model_dump()
        assert isinstance(data, dict)
        assert data["turn_id"] == "turn_abc123"
        assert data["speaker"] == "customer"

    def test_enum_serializes_as_value(self) -> None:
        """SpeakerRole must serialize as its string value, not the enum repr."""
        turn = _make_turn(speaker=SpeakerRole.AGENT)
        data = turn.model_dump()
        assert data["speaker"] == "agent"
        assert isinstance(data["speaker"], str)

    def test_model_dump_mode_json_produces_json_safe_types(self) -> None:
        """mode='json' produces JSON-serializable primitives."""
        turn = _make_turn(metadata={"key": "value"})
        data = turn.model_dump(mode="json")
        assert isinstance(data["speaker"], str)
        assert isinstance(data["start_offset"], int)


# ---------------------------------------------------------------------------
# Boundary deserialization (dict/JSON → model, uses parsing)
# ---------------------------------------------------------------------------


class TestTurnBoundaryDeserialization:
    """Boundary parsing: reconstruct model from dict or JSON.

    At system boundaries (API handlers, file parsers), data arrives as plain
    dicts where types are already demoted (enum→str). Pydantic's
    model_validate with strict=False handles this coercion at the boundary.
    This is the ONLY place where strict=False is acceptable (ADR-002).
    """

    def test_reconstructs_from_model_dump(self) -> None:
        turn = _make_turn(
            normalized_text="normalized",
            metadata={"key": "value"},
        )
        data = turn.model_dump()
        restored = Turn.model_validate(data, strict=False)
        assert restored == turn

    def test_preserves_offsets_through_boundary(self) -> None:
        turn = _make_turn(start_offset=50, end_offset=200)
        data = turn.model_dump()
        restored = Turn.model_validate(data, strict=False)
        assert restored.start_offset == 50
        assert restored.end_offset == 200

    def test_preserves_metadata_through_boundary(self) -> None:
        meta = {"asr_confidence": 0.95, "tags": ["urgent"], "language": "pt-BR"}
        turn = _make_turn(metadata=meta)
        data = turn.model_dump()
        restored = Turn.model_validate(data, strict=False)
        assert restored.metadata == meta

    def test_reconstructs_from_json_mode_dump(self) -> None:
        """Proves full JSON round-trip: model → JSON dict → model."""
        turn = _make_turn(
            normalized_text="hello",
            metadata={"nested": {"deep": True}},
        )
        json_data = turn.model_dump(mode="json")
        restored = Turn.model_validate(json_data, strict=False)
        assert restored == turn


# ---------------------------------------------------------------------------
# Re-export from models package
# ---------------------------------------------------------------------------


class TestTurnReexport:
    def test_importable_from_models_package(self) -> None:
        from talkex.models import Turn as Imported

        assert Imported is Turn
