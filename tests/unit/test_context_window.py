"""Unit tests for the ContextWindow model — follows the Conversation golden template.

Tests cover: construction, validation (IDs, text, turn_ids, indices, parameters),
cross-field validation, strict mode behavior, immutability, serialization round-trip,
and re-export.
"""

from typing import Any

import pytest

from talkex.models.context_window import ContextWindow
from talkex.models.types import ConversationId, TurnId, WindowId

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_context_window(**overrides: object) -> ContextWindow:
    """Factory with sensible defaults. Override any field via kwargs."""
    defaults: dict[str, Any] = {
        "window_id": WindowId("win_abc123"),
        "conversation_id": ConversationId("conv_abc123"),
        "turn_ids": [TurnId("turn_001"), TurnId("turn_002"), TurnId("turn_003")],
        "window_text": "Customer: I need help. Agent: Sure. Customer: My broadband is down.",
        "start_index": 0,
        "end_index": 2,
        "window_size": 3,
        "stride": 1,
    }
    defaults.update(overrides)
    return ContextWindow(**defaults)


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestContextWindowConstruction:
    def test_creates_with_required_fields_only(self) -> None:
        cw = _make_context_window()
        assert cw.window_id == "win_abc123"
        assert cw.conversation_id == "conv_abc123"
        assert len(cw.turn_ids) == 3
        assert cw.window_text == "Customer: I need help. Agent: Sure. Customer: My broadband is down."
        assert cw.start_index == 0
        assert cw.end_index == 2
        assert cw.window_size == 3
        assert cw.stride == 1
        assert cw.metadata == {}

    def test_creates_with_all_fields(self) -> None:
        cw = _make_context_window(
            metadata={"speaker_count": 2, "has_agent": True},
        )
        assert cw.metadata == {"speaker_count": 2, "has_agent": True}

    def test_metadata_defaults_to_empty_dict(self) -> None:
        cw = _make_context_window()
        assert cw.metadata == {}
        assert isinstance(cw.metadata, dict)


# ---------------------------------------------------------------------------
# Validation — IDs
# ---------------------------------------------------------------------------


class TestContextWindowIdValidation:
    def test_rejects_empty_window_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_context_window(window_id=WindowId(""))

    def test_rejects_whitespace_only_window_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_context_window(window_id=WindowId("   "))

    def test_accepts_valid_prefixed_window_id(self) -> None:
        cw = _make_context_window(window_id=WindowId("win_12345"))
        assert cw.window_id == "win_12345"

    def test_preserves_window_id_without_normalizing(self) -> None:
        """Validator rejects invalid IDs but does NOT strip/normalize valid ones."""
        padded_id = WindowId("  win_123  ")
        cw = _make_context_window(window_id=padded_id)
        assert cw.window_id == "  win_123  "

    def test_rejects_empty_conversation_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_context_window(conversation_id=ConversationId(""))

    def test_rejects_whitespace_only_conversation_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_context_window(conversation_id=ConversationId("   "))


# ---------------------------------------------------------------------------
# Validation — window_text
# ---------------------------------------------------------------------------


class TestContextWindowTextValidation:
    def test_rejects_empty_window_text(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_context_window(window_text="")

    def test_rejects_whitespace_only_window_text(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_context_window(window_text="   ")

    def test_preserves_window_text_without_normalizing(self) -> None:
        """Validator rejects invalid text but does NOT strip/normalize valid text."""
        padded_text = "  hello world  "
        cw = _make_context_window(window_text=padded_text)
        assert cw.window_text == "  hello world  "


# ---------------------------------------------------------------------------
# Validation — turn_ids
# ---------------------------------------------------------------------------


class TestContextWindowTurnIdsValidation:
    def test_rejects_empty_turn_ids(self) -> None:
        with pytest.raises(ValueError, match="at least one turn"):
            _make_context_window(turn_ids=[], window_size=0)

    def test_accepts_single_turn_id(self) -> None:
        cw = _make_context_window(
            turn_ids=[TurnId("turn_001")],
            start_index=0,
            end_index=0,
            window_size=1,
        )
        assert len(cw.turn_ids) == 1


# ---------------------------------------------------------------------------
# Validation — indices
# ---------------------------------------------------------------------------


class TestContextWindowIndexValidation:
    def test_rejects_negative_start_index(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _make_context_window(start_index=-1)

    def test_accepts_zero_start_index(self) -> None:
        cw = _make_context_window(start_index=0)
        assert cw.start_index == 0

    def test_rejects_end_index_before_start_index(self) -> None:
        with pytest.raises(ValueError, match="must not precede"):
            _make_context_window(start_index=5, end_index=2)

    def test_accepts_end_index_equal_to_start_index(self) -> None:
        cw = _make_context_window(
            turn_ids=[TurnId("turn_001")],
            start_index=3,
            end_index=3,
            window_size=1,
        )
        assert cw.start_index == cw.end_index


# ---------------------------------------------------------------------------
# Validation — window parameters
# ---------------------------------------------------------------------------


class TestContextWindowParameterValidation:
    def test_rejects_zero_window_size(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _make_context_window(window_size=0)

    def test_rejects_negative_window_size(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _make_context_window(window_size=-1)

    def test_rejects_zero_stride(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _make_context_window(stride=0)

    def test_rejects_negative_stride(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _make_context_window(stride=-1)

    def test_rejects_window_size_mismatch_with_turn_ids(self) -> None:
        """window_size must equal len(turn_ids) for consistency."""
        with pytest.raises(ValueError, match="must equal len"):
            _make_context_window(
                turn_ids=[TurnId("turn_001"), TurnId("turn_002")],
                window_size=5,
            )


# ---------------------------------------------------------------------------
# Strict mode — no coercion
# ---------------------------------------------------------------------------


class TestContextWindowStrictMode:
    def test_rejects_string_for_start_index(self) -> None:
        """strict=True means str won't coerce to int."""
        with pytest.raises(ValueError):
            _make_context_window(start_index="0")

    def test_rejects_float_for_window_size(self) -> None:
        """strict=True means float won't coerce to int."""
        with pytest.raises(ValueError):
            _make_context_window(window_size=3.0)

    def test_rejects_int_for_window_id(self) -> None:
        """strict=True means int won't coerce to str."""
        with pytest.raises(ValueError):
            _make_context_window(window_id=12345)


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestContextWindowImmutability:
    def test_cannot_assign_to_field(self) -> None:
        cw = _make_context_window()
        with pytest.raises(ValueError, match="frozen"):
            cw.window_id = WindowId("win_new")

    def test_cannot_assign_to_metadata(self) -> None:
        cw = _make_context_window()
        with pytest.raises(ValueError, match="frozen"):
            cw.metadata = {"new": "data"}


# ---------------------------------------------------------------------------
# Serialization — in-memory (types preserved)
# ---------------------------------------------------------------------------


class TestContextWindowSerializationInMemory:
    """In-memory serialization: model_dump() preserves Python types."""

    def test_model_dump_produces_dict(self) -> None:
        cw = _make_context_window()
        data = cw.model_dump()
        assert isinstance(data, dict)
        assert data["window_id"] == "win_abc123"
        assert data["window_size"] == 3

    def test_turn_ids_serialize_as_list_of_strings(self) -> None:
        """turn_ids must serialize as plain list of strings."""
        cw = _make_context_window()
        data = cw.model_dump()
        assert isinstance(data["turn_ids"], list)
        assert all(isinstance(tid, str) for tid in data["turn_ids"])

    def test_model_dump_mode_json_produces_json_safe_types(self) -> None:
        """mode='json' produces JSON-serializable primitives."""
        cw = _make_context_window(metadata={"key": "value"})
        data = cw.model_dump(mode="json")
        assert isinstance(data["window_size"], int)
        assert isinstance(data["window_id"], str)


# ---------------------------------------------------------------------------
# Boundary deserialization (dict/JSON → model, uses parsing)
# ---------------------------------------------------------------------------


class TestContextWindowBoundaryDeserialization:
    """Boundary parsing: reconstruct model from dict or JSON.

    At system boundaries (API handlers, file parsers), data arrives as plain
    dicts. Pydantic's model_validate with strict=False handles coercion.
    This is the ONLY place where strict=False is acceptable (ADR-002).
    """

    def test_reconstructs_from_model_dump(self) -> None:
        cw = _make_context_window(metadata={"key": "value"})
        data = cw.model_dump()
        restored = ContextWindow.model_validate(data, strict=False)
        assert restored == cw

    def test_preserves_turn_ids_through_boundary(self) -> None:
        turn_ids = [TurnId("turn_001"), TurnId("turn_002"), TurnId("turn_003")]
        cw = _make_context_window(turn_ids=turn_ids)
        data = cw.model_dump()
        restored = ContextWindow.model_validate(data, strict=False)
        assert restored.turn_ids == turn_ids

    def test_preserves_metadata_through_boundary(self) -> None:
        meta = {"speakers": ["customer", "agent"], "duration_ms": 45000}
        cw = _make_context_window(metadata=meta)
        data = cw.model_dump()
        restored = ContextWindow.model_validate(data, strict=False)
        assert restored.metadata == meta

    def test_reconstructs_from_json_mode_dump(self) -> None:
        """Proves full JSON round-trip: model → JSON dict → model."""
        cw = _make_context_window(
            metadata={"nested": {"deep": True}},
        )
        json_data = cw.model_dump(mode="json")
        restored = ContextWindow.model_validate(json_data, strict=False)
        assert restored == cw


# ---------------------------------------------------------------------------
# Re-export from models package
# ---------------------------------------------------------------------------


class TestContextWindowReexport:
    def test_importable_from_models_package(self) -> None:
        from talkex.models import ContextWindow as Imported

        assert Imported is ContextWindow
