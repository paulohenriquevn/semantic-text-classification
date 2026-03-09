"""Unit tests for EmbeddingInput and EmbeddingBatch boundary objects.

Tests cover: construction, field validation, strict mode,
immutability, serialization, and reexport.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from semantic_conversation_engine.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from semantic_conversation_engine.models.enums import Channel, ObjectType
from semantic_conversation_engine.models.types import EmbeddingId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _input(**overrides: Any) -> EmbeddingInput:
    defaults: dict[str, Any] = {
        "embedding_id": EmbeddingId("emb_001"),
        "object_type": ObjectType.CONTEXT_WINDOW,
        "object_id": "conv_a_win_0",
        "text": "customer needs help with billing",
    }
    defaults.update(overrides)
    return EmbeddingInput(**defaults)


# ---------------------------------------------------------------------------
# EmbeddingInput construction
# ---------------------------------------------------------------------------


class TestEmbeddingInputConstruction:
    def test_minimal_construction(self) -> None:
        inp = _input()
        assert inp.embedding_id == "emb_001"
        assert inp.object_type == ObjectType.CONTEXT_WINDOW
        assert inp.object_id == "conv_a_win_0"
        assert inp.text == "customer needs help with billing"

    def test_all_fields(self) -> None:
        inp = _input(
            language="pt",
            channel=Channel.VOICE,
            metadata={"source": "pipeline"},
        )
        assert inp.language == "pt"
        assert inp.channel == Channel.VOICE
        assert inp.metadata == {"source": "pipeline"}

    def test_defaults(self) -> None:
        inp = _input()
        assert inp.metadata == {}
        assert inp.language is None
        assert inp.channel is None


# ---------------------------------------------------------------------------
# EmbeddingInput field validation
# ---------------------------------------------------------------------------


class TestEmbeddingInputValidation:
    def test_rejects_empty_embedding_id(self) -> None:
        with pytest.raises(ValidationError, match="embedding_id"):
            _input(embedding_id=EmbeddingId(""))

    def test_rejects_whitespace_embedding_id(self) -> None:
        with pytest.raises(ValidationError, match="embedding_id"):
            _input(embedding_id=EmbeddingId("   "))

    def test_rejects_empty_object_id(self) -> None:
        with pytest.raises(ValidationError, match="object_id"):
            _input(object_id="")

    def test_rejects_whitespace_object_id(self) -> None:
        with pytest.raises(ValidationError, match="object_id"):
            _input(object_id="   ")

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError, match="text"):
            _input(text="")

    def test_rejects_whitespace_text(self) -> None:
        with pytest.raises(ValidationError, match="text"):
            _input(text="   ")

    def test_preserves_ids_without_normalizing(self) -> None:
        inp = _input(embedding_id=EmbeddingId("  emb_001  "))
        assert inp.embedding_id == "  emb_001  "


# ---------------------------------------------------------------------------
# EmbeddingInput strict mode
# ---------------------------------------------------------------------------


class TestEmbeddingInputStrictMode:
    def test_rejects_int_for_string(self) -> None:
        with pytest.raises(ValidationError):
            _input(text=123)

    def test_rejects_string_for_object_type(self) -> None:
        with pytest.raises(ValidationError):
            _input(object_type="invalid_type")


# ---------------------------------------------------------------------------
# EmbeddingInput immutability
# ---------------------------------------------------------------------------


class TestEmbeddingInputImmutability:
    def test_frozen(self) -> None:
        inp = _input()
        with pytest.raises(ValidationError):
            inp.text = "modified"


# ---------------------------------------------------------------------------
# EmbeddingBatch
# ---------------------------------------------------------------------------


class TestEmbeddingBatch:
    def test_construction(self) -> None:
        batch = EmbeddingBatch(items=[_input()])
        assert len(batch.items) == 1

    def test_multiple_items(self) -> None:
        items = [
            _input(embedding_id=EmbeddingId("emb_001"), object_id="win_0"),
            _input(embedding_id=EmbeddingId("emb_002"), object_id="win_1"),
        ]
        batch = EmbeddingBatch(items=items)
        assert len(batch.items) == 2

    def test_rejects_empty_items(self) -> None:
        with pytest.raises(ValidationError, match="items"):
            EmbeddingBatch(items=[])

    def test_frozen(self) -> None:
        batch = EmbeddingBatch(items=[_input()])
        with pytest.raises(ValidationError):
            batch.items = []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestEmbeddingInputSerialization:
    def test_model_dump(self) -> None:
        inp = _input()
        data = inp.model_dump()
        assert data["embedding_id"] == "emb_001"
        assert data["object_type"] == "context_window"
        assert data["text"] == "customer needs help with billing"

    def test_model_dump_json_mode(self) -> None:
        inp = _input(channel=Channel.VOICE)
        data = inp.model_dump(mode="json")
        assert data["channel"] == "voice"

    def test_batch_model_dump(self) -> None:
        batch = EmbeddingBatch(items=[_input()])
        data = batch.model_dump()
        assert len(data["items"]) == 1


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestEmbeddingInputReexport:
    def test_importable_from_embeddings_package(self) -> None:
        from semantic_conversation_engine.embeddings import (
            EmbeddingBatch as BatchImported,
        )
        from semantic_conversation_engine.embeddings import (
            EmbeddingInput as InputImported,
        )

        assert InputImported is EmbeddingInput
        assert BatchImported is EmbeddingBatch
