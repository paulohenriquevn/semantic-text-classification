"""Unit tests for embedding preprocessing.

Tests cover: PreprocessingConfig, prepare_embedding_text,
prepare_batch_texts, estimate_token_count, and edge cases.
"""

import pytest

from talkex.embeddings.inputs import EmbeddingInput
from talkex.embeddings.preprocessing import (
    PreprocessingConfig,
    estimate_token_count,
    prepare_batch_texts,
    prepare_embedding_text,
)
from talkex.models.enums import ObjectType
from talkex.models.types import EmbeddingId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inp(
    text: str = "hello world",
    object_type: ObjectType = ObjectType.CONTEXT_WINDOW,
) -> EmbeddingInput:
    return EmbeddingInput(
        embedding_id=EmbeddingId("emb_001"),
        object_type=object_type,
        object_id="obj_001",
        text=text,
    )


# ---------------------------------------------------------------------------
# PreprocessingConfig defaults
# ---------------------------------------------------------------------------


class TestPreprocessingConfigDefaults:
    def test_default_no_prefix(self) -> None:
        cfg = PreprocessingConfig()
        assert cfg.task_prefix == ""

    def test_default_no_object_type_prefix(self) -> None:
        cfg = PreprocessingConfig()
        assert cfg.include_object_type_prefix is False

    def test_frozen(self) -> None:
        cfg = PreprocessingConfig()
        with pytest.raises(AttributeError):
            cfg.task_prefix = "query: "  # type: ignore[misc]


# ---------------------------------------------------------------------------
# prepare_embedding_text — no prefixes
# ---------------------------------------------------------------------------


class TestPrepareEmbeddingTextNoPrefix:
    def test_returns_raw_text_when_no_config(self) -> None:
        result = prepare_embedding_text(_inp("billing issue"), PreprocessingConfig())
        assert result == "billing issue"

    def test_preserves_whitespace_in_text(self) -> None:
        result = prepare_embedding_text(_inp("  spaced  "), PreprocessingConfig())
        assert result == "  spaced  "


# ---------------------------------------------------------------------------
# prepare_embedding_text — task prefix
# ---------------------------------------------------------------------------


class TestPrepareEmbeddingTextTaskPrefix:
    def test_e5_passage_prefix(self) -> None:
        cfg = PreprocessingConfig(task_prefix="passage:")
        result = prepare_embedding_text(_inp("billing issue"), cfg)
        assert result == "passage: billing issue"

    def test_e5_query_prefix(self) -> None:
        cfg = PreprocessingConfig(task_prefix="query:")
        result = prepare_embedding_text(_inp("cancel subscription"), cfg)
        assert result == "query: cancel subscription"


# ---------------------------------------------------------------------------
# prepare_embedding_text — object type prefix
# ---------------------------------------------------------------------------


class TestPrepareEmbeddingTextObjectTypePrefix:
    def test_window_label(self) -> None:
        cfg = PreprocessingConfig(include_object_type_prefix=True)
        result = prepare_embedding_text(_inp("text", ObjectType.CONTEXT_WINDOW), cfg)
        assert result == "[WINDOW] text"

    def test_turn_label(self) -> None:
        cfg = PreprocessingConfig(include_object_type_prefix=True)
        result = prepare_embedding_text(_inp("text", ObjectType.TURN), cfg)
        assert result == "[TURN] text"

    def test_conversation_label(self) -> None:
        cfg = PreprocessingConfig(include_object_type_prefix=True)
        result = prepare_embedding_text(_inp("text", ObjectType.CONVERSATION), cfg)
        assert result == "[CONVERSATION] text"


# ---------------------------------------------------------------------------
# prepare_embedding_text — both prefixes combined
# ---------------------------------------------------------------------------


class TestPrepareEmbeddingTextCombined:
    def test_task_and_object_type_prefix(self) -> None:
        cfg = PreprocessingConfig(task_prefix="passage:", include_object_type_prefix=True)
        result = prepare_embedding_text(_inp("billing", ObjectType.CONTEXT_WINDOW), cfg)
        assert result == "passage: [WINDOW] billing"


# ---------------------------------------------------------------------------
# prepare_batch_texts
# ---------------------------------------------------------------------------


class TestPrepareBatchTexts:
    def test_batch_preserves_order(self) -> None:
        inputs = [_inp("first"), _inp("second"), _inp("third")]
        results = prepare_batch_texts(inputs, PreprocessingConfig())
        assert results == ["first", "second", "third"]

    def test_batch_applies_prefix(self) -> None:
        cfg = PreprocessingConfig(task_prefix="query:")
        inputs = [_inp("a"), _inp("b")]
        results = prepare_batch_texts(inputs, cfg)
        assert results == ["query: a", "query: b"]

    def test_empty_batch(self) -> None:
        results = prepare_batch_texts([], PreprocessingConfig())
        assert results == []


# ---------------------------------------------------------------------------
# estimate_token_count
# ---------------------------------------------------------------------------


class TestEstimateTokenCount:
    def test_short_text(self) -> None:
        assert estimate_token_count("hi") == 1  # 2 chars // 4 = 0 → max(1, 0)

    def test_known_length(self) -> None:
        # 20 chars → 20 // 4 = 5
        assert estimate_token_count("a" * 20) == 5

    def test_empty_text(self) -> None:
        assert estimate_token_count("") == 1  # max(1, 0)

    def test_minimum_is_one(self) -> None:
        assert estimate_token_count("x") >= 1


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestPreprocessingReexport:
    def test_importable_from_embeddings_package(self) -> None:
        from talkex.embeddings import (
            PreprocessingConfig as PC,
        )
        from talkex.embeddings import (
            prepare_batch_texts as pbt,
        )
        from talkex.embeddings import (
            prepare_embedding_text as pet,
        )

        assert PC is PreprocessingConfig
        assert pbt is prepare_batch_texts
        assert pet is prepare_embedding_text
