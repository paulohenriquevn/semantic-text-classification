"""Index document builders — domain to index adaptation layer.

Pure functions converting domain objects (ContextWindow, EmbeddingRecord)
into the document/vector formats expected by LexicalIndex and VectorIndex.

This layer prevents coupling between indexes and domain entities.
Indexes receive plain dicts and vectors — they never import domain models.

Functions are intentionally simple: extract fields, build dict, done.
No transformation logic, no normalization, no feature engineering.
"""

from __future__ import annotations

from typing import Any

from talkex.models.context_window import ContextWindow
from talkex.models.embedding_record import EmbeddingRecord


def context_window_to_lexical_doc(window: ContextWindow) -> dict[str, object]:
    """Convert a ContextWindow into a lexical index document.

    Extracts the fields needed for BM25 indexing: doc_id, text,
    object_type, and conversation_id for filtering.

    Args:
        window: The context window to convert.

    Returns:
        Document dict compatible with LexicalIndex.index().
    """
    doc: dict[str, object] = {
        "doc_id": window.window_id,
        "text": window.window_text,
        "object_type": "context_window",
        "conversation_id": window.conversation_id,
        "start_index": window.start_index,
        "end_index": window.end_index,
        "window_size": window.window_size,
    }
    return doc


def context_windows_to_lexical_docs(
    windows: list[ContextWindow],
) -> list[dict[str, object]]:
    """Batch convert ContextWindows into lexical index documents.

    Args:
        windows: List of context windows.

    Returns:
        List of document dicts, same order as input.
    """
    return [context_window_to_lexical_doc(w) for w in windows]


def embedding_record_to_hit_metadata(record: EmbeddingRecord) -> dict[str, Any]:
    """Extract metadata from an EmbeddingRecord for RetrievalHit.

    Captures provenance information that may be needed for
    debugging, audit, or downstream processing.

    Args:
        record: The embedding record.

    Returns:
        Metadata dict with model name, version, and pooling strategy.
    """
    return {
        "model_name": record.model_name,
        "model_version": record.model_version,
        "pooling_strategy": record.pooling_strategy.value,
        "embedding_id": str(record.embedding_id),
    }
