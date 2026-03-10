"""Unit tests for the EmbeddingRecord model — follows the Conversation golden template.

Tests cover: construction, validation (IDs, strings, dimensions, vector),
cross-field validation, strict mode behavior, immutability, serialization
round-trip, and re-export.
"""

from typing import Any

import pytest

from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import ObjectType, PoolingStrategy
from talkex.models.types import EmbeddingId

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SAMPLE_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5]


def _make_embedding_record(**overrides: object) -> EmbeddingRecord:
    """Factory with sensible defaults. Override any field via kwargs."""
    defaults: dict[str, Any] = {
        "embedding_id": EmbeddingId("emb_abc123"),
        "source_id": "turn_abc123",
        "source_type": ObjectType.TURN,
        "model_name": "e5-large-v2",
        "model_version": "1.0.0",
        "pooling_strategy": PoolingStrategy.MEAN,
        "dimensions": 5,
        "vector": list(_SAMPLE_VECTOR),
    }
    defaults.update(overrides)
    return EmbeddingRecord(**defaults)


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestEmbeddingRecordConstruction:
    def test_creates_with_all_fields(self) -> None:
        rec = _make_embedding_record()
        assert rec.embedding_id == "emb_abc123"
        assert rec.source_id == "turn_abc123"
        assert rec.source_type == ObjectType.TURN
        assert rec.model_name == "e5-large-v2"
        assert rec.model_version == "1.0.0"
        assert rec.pooling_strategy == PoolingStrategy.MEAN
        assert rec.dimensions == 5
        assert rec.vector == _SAMPLE_VECTOR

    def test_accepts_different_source_types(self) -> None:
        for source_type in ObjectType:
            rec = _make_embedding_record(source_type=source_type)
            assert rec.source_type == source_type

    def test_accepts_different_pooling_strategies(self) -> None:
        for strategy in PoolingStrategy:
            rec = _make_embedding_record(pooling_strategy=strategy)
            assert rec.pooling_strategy == strategy


# ---------------------------------------------------------------------------
# Validation — IDs and strings
# ---------------------------------------------------------------------------


class TestEmbeddingRecordIdValidation:
    def test_rejects_empty_embedding_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_embedding_record(embedding_id=EmbeddingId(""))

    def test_rejects_whitespace_only_embedding_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_embedding_record(embedding_id=EmbeddingId("   "))

    def test_preserves_embedding_id_without_normalizing(self) -> None:
        padded_id = EmbeddingId("  emb_123  ")
        rec = _make_embedding_record(embedding_id=padded_id)
        assert rec.embedding_id == "  emb_123  "

    def test_rejects_empty_source_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_embedding_record(source_id="")

    def test_rejects_empty_model_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_embedding_record(model_name="")

    def test_rejects_whitespace_only_model_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_embedding_record(model_name="   ")

    def test_rejects_empty_model_version(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_embedding_record(model_version="")

    def test_rejects_whitespace_only_model_version(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_embedding_record(model_version="   ")


# ---------------------------------------------------------------------------
# Validation — dimensions and vector
# ---------------------------------------------------------------------------


class TestEmbeddingRecordDimensionValidation:
    def test_rejects_zero_dimensions(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _make_embedding_record(dimensions=0, vector=[])

    def test_rejects_negative_dimensions(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _make_embedding_record(dimensions=-1, vector=[])

    def test_rejects_empty_vector(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_embedding_record(dimensions=1, vector=[])

    def test_rejects_dimensions_mismatch_with_vector(self) -> None:
        """dimensions must equal len(vector) — catches builder corruption."""
        with pytest.raises(ValueError, match="must equal len"):
            _make_embedding_record(dimensions=3, vector=[0.1, 0.2])

    def test_accepts_single_dimension_vector(self) -> None:
        rec = _make_embedding_record(dimensions=1, vector=[0.5])
        assert rec.dimensions == 1
        assert rec.vector == [0.5]

    def test_accepts_high_dimensional_vector(self) -> None:
        dim = 768
        vec = [0.01 * i for i in range(dim)]
        rec = _make_embedding_record(dimensions=dim, vector=vec)
        assert rec.dimensions == dim
        assert len(rec.vector) == dim


# ---------------------------------------------------------------------------
# Strict mode — no coercion
# ---------------------------------------------------------------------------


class TestEmbeddingRecordStrictMode:
    def test_rejects_string_source_type_coercion(self) -> None:
        """strict=True means 'turn' (str) won't coerce to ObjectType.TURN."""
        with pytest.raises(ValueError):
            _make_embedding_record(source_type="turn")

    def test_rejects_string_pooling_strategy_coercion(self) -> None:
        """strict=True means 'mean' (str) won't coerce to PoolingStrategy.MEAN."""
        with pytest.raises(ValueError):
            _make_embedding_record(pooling_strategy="mean")

    def test_rejects_float_for_dimensions(self) -> None:
        """strict=True means float won't coerce to int."""
        with pytest.raises(ValueError):
            _make_embedding_record(dimensions=5.0)

    def test_rejects_int_for_embedding_id(self) -> None:
        """strict=True means int won't coerce to str."""
        with pytest.raises(ValueError):
            _make_embedding_record(embedding_id=12345)


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestEmbeddingRecordImmutability:
    def test_cannot_assign_to_field(self) -> None:
        rec = _make_embedding_record()
        with pytest.raises(ValueError, match="frozen"):
            rec.embedding_id = EmbeddingId("emb_new")

    def test_cannot_assign_to_vector(self) -> None:
        rec = _make_embedding_record()
        with pytest.raises(ValueError, match="frozen"):
            rec.vector = [0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Serialization — in-memory (types preserved)
# ---------------------------------------------------------------------------


class TestEmbeddingRecordSerializationInMemory:
    """In-memory serialization: model_dump() preserves Python types."""

    def test_model_dump_produces_dict(self) -> None:
        rec = _make_embedding_record()
        data = rec.model_dump()
        assert isinstance(data, dict)
        assert data["embedding_id"] == "emb_abc123"
        assert data["dimensions"] == 5

    def test_enums_serialize_as_values(self) -> None:
        rec = _make_embedding_record()
        data = rec.model_dump()
        assert data["source_type"] == "turn"
        assert data["pooling_strategy"] == "mean"
        assert isinstance(data["source_type"], str)
        assert isinstance(data["pooling_strategy"], str)

    def test_vector_serializes_as_list_of_floats(self) -> None:
        rec = _make_embedding_record()
        data = rec.model_dump()
        assert isinstance(data["vector"], list)
        assert all(isinstance(v, float) for v in data["vector"])

    def test_model_dump_mode_json_produces_json_safe_types(self) -> None:
        rec = _make_embedding_record()
        data = rec.model_dump(mode="json")
        assert isinstance(data["source_type"], str)
        assert isinstance(data["dimensions"], int)


# ---------------------------------------------------------------------------
# Boundary deserialization (dict/JSON → model, uses parsing)
# ---------------------------------------------------------------------------


class TestEmbeddingRecordBoundaryDeserialization:
    """Boundary parsing: reconstruct model from dict or JSON.

    At system boundaries (API handlers, file parsers), data arrives as plain
    dicts. Pydantic's model_validate with strict=False handles coercion.
    This is the ONLY place where strict=False is acceptable (ADR-002).
    """

    def test_reconstructs_from_model_dump(self) -> None:
        rec = _make_embedding_record()
        data = rec.model_dump()
        restored = EmbeddingRecord.model_validate(data, strict=False)
        assert restored == rec

    def test_preserves_vector_through_boundary(self) -> None:
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        rec = _make_embedding_record(vector=vec, dimensions=5)
        data = rec.model_dump()
        restored = EmbeddingRecord.model_validate(data, strict=False)
        assert restored.vector == vec

    def test_reconstructs_from_json_mode_dump(self) -> None:
        """Proves full JSON round-trip: model → JSON dict → model."""
        rec = _make_embedding_record()
        json_data = rec.model_dump(mode="json")
        restored = EmbeddingRecord.model_validate(json_data, strict=False)
        assert restored == rec


# ---------------------------------------------------------------------------
# Re-export from models package
# ---------------------------------------------------------------------------


class TestEmbeddingRecordReexport:
    def test_importable_from_models_package(self) -> None:
        from talkex.models import EmbeddingRecord as Imported

        assert Imported is EmbeddingRecord
