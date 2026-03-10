"""Unit tests for the Conversation model — golden template for all core models.

Tests cover: construction, validation, immutability, serialization round-trip,
enum serialization in dumps, strict mode behavior, and re-export.
"""

from datetime import UTC, datetime

import pytest

from talkex.models.conversation import Conversation
from talkex.models.enums import Channel
from talkex.models.types import ConversationId

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_conversation(**overrides: object) -> Conversation:
    """Factory with sensible defaults. Override any field via kwargs."""
    defaults: dict = {
        "conversation_id": ConversationId("conv_abc123"),
        "channel": Channel.VOICE,
        "start_time": datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return Conversation(**defaults)


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestConversationConstruction:
    def test_creates_with_required_fields_only(self) -> None:
        conv = _make_conversation()
        assert conv.conversation_id == "conv_abc123"
        assert conv.channel == Channel.VOICE
        assert conv.end_time is None
        assert conv.customer_id is None
        assert conv.product is None
        assert conv.queue is None
        assert conv.region is None
        assert conv.metadata == {}

    def test_creates_with_all_fields(self) -> None:
        conv = _make_conversation(
            end_time=datetime(2026, 3, 1, 10, 30, 0, tzinfo=UTC),
            customer_id="cust_001",
            product="broadband",
            queue="retention",
            region="southeast",
            metadata={"source": "crm", "priority": 1},
        )
        assert conv.end_time == datetime(2026, 3, 1, 10, 30, 0, tzinfo=UTC)
        assert conv.customer_id == "cust_001"
        assert conv.product == "broadband"
        assert conv.queue == "retention"
        assert conv.region == "southeast"
        assert conv.metadata == {"source": "crm", "priority": 1}

    def test_metadata_defaults_to_empty_dict(self) -> None:
        conv = _make_conversation()
        assert conv.metadata == {}
        assert isinstance(conv.metadata, dict)


# ---------------------------------------------------------------------------
# Validation — ID
# ---------------------------------------------------------------------------


class TestConversationIdValidation:
    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_conversation(conversation_id=ConversationId(""))

    def test_rejects_whitespace_only_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_conversation(conversation_id=ConversationId("   "))

    def test_accepts_valid_prefixed_id(self) -> None:
        conv = _make_conversation(conversation_id=ConversationId("conv_12345"))
        assert conv.conversation_id == "conv_12345"

    def test_preserves_original_value_without_normalizing(self) -> None:
        """Validator rejects invalid IDs but does NOT strip/normalize valid ones."""
        padded_id = ConversationId("  conv_123  ")
        conv = _make_conversation(conversation_id=padded_id)
        assert conv.conversation_id == "  conv_123  "


# ---------------------------------------------------------------------------
# Validation — temporal
# ---------------------------------------------------------------------------


class TestConversationTemporalValidation:
    def test_rejects_end_time_before_start_time(self) -> None:
        with pytest.raises(ValueError, match="must not precede"):
            _make_conversation(
                start_time=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
                end_time=datetime(2026, 3, 1, 9, 0, 0, tzinfo=UTC),
            )

    def test_accepts_end_time_equal_to_start_time(self) -> None:
        t = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
        conv = _make_conversation(start_time=t, end_time=t)
        assert conv.start_time == conv.end_time

    def test_accepts_end_time_after_start_time(self) -> None:
        conv = _make_conversation(
            start_time=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
            end_time=datetime(2026, 3, 1, 10, 30, 0, tzinfo=UTC),
        )
        assert conv.end_time is not None
        assert conv.end_time > conv.start_time

    def test_accepts_none_end_time(self) -> None:
        conv = _make_conversation(end_time=None)
        assert conv.end_time is None


# ---------------------------------------------------------------------------
# Strict mode — no coercion
# ---------------------------------------------------------------------------


class TestConversationStrictMode:
    def test_rejects_string_channel_coercion(self) -> None:
        """strict=True means 'voice' (str) won't coerce to Channel.VOICE."""
        with pytest.raises(ValueError):
            _make_conversation(channel="voice")

    def test_rejects_string_datetime_coercion(self) -> None:
        """strict=True means ISO string won't coerce to datetime."""
        with pytest.raises(ValueError):
            _make_conversation(start_time="2026-03-01T10:00:00Z")

    def test_rejects_int_for_conversation_id(self) -> None:
        """strict=True means int won't coerce to str."""
        with pytest.raises(ValueError):
            _make_conversation(conversation_id=12345)


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestConversationImmutability:
    def test_cannot_assign_to_field(self) -> None:
        conv = _make_conversation()
        with pytest.raises(ValueError, match="frozen"):
            conv.conversation_id = ConversationId("conv_new")

    def test_cannot_assign_to_metadata(self) -> None:
        conv = _make_conversation()
        with pytest.raises(ValueError, match="frozen"):
            conv.metadata = {"new": "data"}


# ---------------------------------------------------------------------------
# Serialization — in-memory (types preserved)
# ---------------------------------------------------------------------------


class TestConversationSerializationInMemory:
    """In-memory serialization: model_dump() preserves Python types."""

    def test_model_dump_produces_dict(self) -> None:
        conv = _make_conversation()
        data = conv.model_dump()
        assert isinstance(data, dict)
        assert data["conversation_id"] == "conv_abc123"
        assert data["channel"] == "voice"

    def test_enum_serializes_as_value(self) -> None:
        """Channel must serialize as its string value, not the enum repr."""
        conv = _make_conversation(channel=Channel.CHAT)
        data = conv.model_dump()
        assert data["channel"] == "chat"
        assert isinstance(data["channel"], str)

    def test_model_dump_mode_json_produces_json_safe_types(self) -> None:
        """mode='json' produces JSON-serializable primitives (str, int, etc.)."""
        conv = _make_conversation(
            end_time=datetime(2026, 3, 1, 10, 30, 0, tzinfo=UTC),
            metadata={"key": "value"},
        )
        data = conv.model_dump(mode="json")
        assert isinstance(data["start_time"], str)
        assert isinstance(data["channel"], str)


# ---------------------------------------------------------------------------
# Boundary deserialization (dict/JSON → model, uses parsing)
# ---------------------------------------------------------------------------


class TestConversationBoundaryDeserialization:
    """Boundary parsing: reconstruct model from dict or JSON.

    At system boundaries (API handlers, file parsers), data arrives as plain
    dicts where types are already demoted (datetime→str, enum→str). Pydantic's
    model_validate with strict=False handles this coercion at the boundary.
    This is the ONLY place where strict=False is acceptable (ADR-002).
    """

    def test_reconstructs_from_model_dump(self) -> None:
        conv = _make_conversation(
            end_time=datetime(2026, 3, 1, 10, 30, 0, tzinfo=UTC),
            customer_id="cust_001",
            metadata={"key": "value"},
        )
        data = conv.model_dump()
        restored = Conversation.model_validate(data, strict=False)
        assert restored == conv

    def test_preserves_datetime_through_boundary(self) -> None:
        t_start = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
        t_end = datetime(2026, 3, 1, 10, 45, 0, tzinfo=UTC)
        conv = _make_conversation(start_time=t_start, end_time=t_end)
        data = conv.model_dump()
        restored = Conversation.model_validate(data, strict=False)
        assert restored.start_time == t_start
        assert restored.end_time == t_end

    def test_preserves_metadata_through_boundary(self) -> None:
        meta = {"source": "crm", "tags": ["vip", "churning"], "score": 0.82}
        conv = _make_conversation(metadata=meta)
        data = conv.model_dump()
        restored = Conversation.model_validate(data, strict=False)
        assert restored.metadata == meta

    def test_reconstructs_from_json_mode_dump(self) -> None:
        """Proves full JSON round-trip: model → JSON dict → model."""
        conv = _make_conversation(
            end_time=datetime(2026, 3, 1, 10, 30, 0, tzinfo=UTC),
            metadata={"nested": {"deep": True}},
        )
        json_data = conv.model_dump(mode="json")
        restored = Conversation.model_validate(json_data, strict=False)
        assert restored == conv


# ---------------------------------------------------------------------------
# Re-export from models package
# ---------------------------------------------------------------------------


class TestConversationReexport:
    def test_importable_from_models_package(self) -> None:
        from talkex.models import Conversation as Imported

        assert Imported is Conversation
