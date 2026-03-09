"""Unit tests for index document builders.

Tests cover: context_window_to_lexical_doc, context_windows_to_lexical_docs,
embedding_record_to_hit_metadata, and reexport.
"""

from semantic_conversation_engine.models.context_window import ContextWindow
from semantic_conversation_engine.models.embedding_record import EmbeddingRecord
from semantic_conversation_engine.models.enums import ObjectType, PoolingStrategy
from semantic_conversation_engine.models.types import (
    ConversationId,
    EmbeddingId,
    TurnId,
    WindowId,
)
from semantic_conversation_engine.retrieval.builders import (
    context_window_to_lexical_doc,
    context_windows_to_lexical_docs,
    embedding_record_to_hit_metadata,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window(
    win_id: str = "win_0",
    text: str = "hello world",
    conv_id: str = "conv_1",
) -> ContextWindow:
    return ContextWindow(
        window_id=WindowId(win_id),
        conversation_id=ConversationId(conv_id),
        turn_ids=[TurnId("t0"), TurnId("t1")],
        window_text=text,
        start_index=0,
        end_index=1,
        window_size=2,
        stride=1,
    )


def _record(emb_id: str = "emb_001") -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id=EmbeddingId(emb_id),
        source_id="win_0",
        source_type=ObjectType.CONTEXT_WINDOW,
        model_name="e5-base",
        model_version="2.0",
        pooling_strategy=PoolingStrategy.MEAN,
        dimensions=3,
        vector=[0.1, 0.2, 0.3],
    )


# ---------------------------------------------------------------------------
# context_window_to_lexical_doc
# ---------------------------------------------------------------------------


class TestContextWindowToLexicalDoc:
    def test_contains_doc_id(self) -> None:
        doc = context_window_to_lexical_doc(_window(win_id="win_42"))
        assert doc["doc_id"] == "win_42"

    def test_contains_text(self) -> None:
        doc = context_window_to_lexical_doc(_window(text="billing issue"))
        assert doc["text"] == "billing issue"

    def test_contains_object_type(self) -> None:
        doc = context_window_to_lexical_doc(_window())
        assert doc["object_type"] == "context_window"

    def test_contains_conversation_id(self) -> None:
        doc = context_window_to_lexical_doc(_window(conv_id="conv_99"))
        assert doc["conversation_id"] == "conv_99"

    def test_contains_positional_fields(self) -> None:
        doc = context_window_to_lexical_doc(_window())
        assert doc["start_index"] == 0
        assert doc["end_index"] == 1
        assert doc["window_size"] == 2


# ---------------------------------------------------------------------------
# context_windows_to_lexical_docs — batch
# ---------------------------------------------------------------------------


class TestContextWindowsToLexicalDocs:
    def test_batch_preserves_order(self) -> None:
        windows = [_window(win_id=f"win_{i}") for i in range(3)]
        docs = context_windows_to_lexical_docs(windows)
        assert [d["doc_id"] for d in docs] == ["win_0", "win_1", "win_2"]

    def test_empty_list(self) -> None:
        assert context_windows_to_lexical_docs([]) == []


# ---------------------------------------------------------------------------
# embedding_record_to_hit_metadata
# ---------------------------------------------------------------------------


class TestEmbeddingRecordToHitMetadata:
    def test_contains_model_name(self) -> None:
        meta = embedding_record_to_hit_metadata(_record())
        assert meta["model_name"] == "e5-base"

    def test_contains_model_version(self) -> None:
        meta = embedding_record_to_hit_metadata(_record())
        assert meta["model_version"] == "2.0"

    def test_contains_pooling_strategy(self) -> None:
        meta = embedding_record_to_hit_metadata(_record())
        assert meta["pooling_strategy"] == "mean"

    def test_contains_embedding_id(self) -> None:
        meta = embedding_record_to_hit_metadata(_record(emb_id="emb_xyz"))
        assert meta["embedding_id"] == "emb_xyz"


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestBuildersReexport:
    def test_importable_from_retrieval_package(self) -> None:
        from semantic_conversation_engine.retrieval import (
            context_window_to_lexical_doc as cw2ld,
        )
        from semantic_conversation_engine.retrieval import (
            context_windows_to_lexical_docs as cws2lds,
        )
        from semantic_conversation_engine.retrieval import (
            embedding_record_to_hit_metadata as er2hm,
        )

        assert cw2ld is context_window_to_lexical_doc
        assert cws2lds is context_windows_to_lexical_docs
        assert er2hm is embedding_record_to_hit_metadata
