"""Sentence-level highlighting for semantic search results.

Splits window text into sentences, embeds them in a single batch,
and finds the sentence most similar to the query vector using
vectorized cosine similarity.

Optimized for batching across multiple windows — all sentences from
all results are embedded in ONE model call, then matched via NumPy
matrix operations. Scales to millions of corpus records because
highlighting only runs on the top-K results (typically 10-20).
"""

from __future__ import annotations

import re

import numpy as np

from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from talkex.models.enums import ObjectType
from talkex.models.types import EmbeddingId

# Regex patterns compiled once at module level
_SPLIT_PATTERN = re.compile(r"\n|(?<=[.!?])\s+")
_SPEAKER_TAG_PATTERN = re.compile(r"\[[\w]+\]\s*")


def _split_sentences(text: str) -> list[tuple[str, str]]:
    """Split text into (original, cleaned) sentence pairs.

    Returns pairs where cleaned text has >5 chars after removing
    speaker tags. Returns empty list if fewer than 2 valid sentences.
    """
    sentences = [s.strip() for s in _SPLIT_PATTERN.split(text) if s.strip()]
    clean = [_SPEAKER_TAG_PATTERN.sub("", s).strip() for s in sentences]
    pairs = [(orig, c) for orig, c in zip(sentences, clean, strict=True) if len(c) > 5]
    return pairs if len(pairs) > 1 else []


def find_best_sentence(
    text: str,
    query_vector: list[float],
    embedding_generator: object,
) -> str | None:
    """Find the sentence in text most similar to the query vector.

    Single-window version — delegates to batch version internally.
    Kept for backward compatibility with category_service.

    Args:
        text: The full window text (may contain speaker tags and newlines).
        query_vector: The embedded query vector to compare against.
        embedding_generator: An embedding generator with a .generate() method.

    Returns:
        The best-matching original sentence, or None if text is too short.
    """
    results = find_best_sentences_batch([text], query_vector, embedding_generator)
    return results[0]


def find_best_sentences_batch(
    texts: list[str],
    query_vector: list[float],
    embedding_generator: object,
) -> list[str | None]:
    """Find the best-matching sentence for multiple windows in one batch.

    Collects all sentences from all windows, embeds them in a SINGLE
    model call, then uses vectorized cosine similarity to find the
    best match per window.

    This is O(1) model calls regardless of how many windows are passed,
    making it efficient for top-K result highlighting.

    Args:
        texts: List of window texts to highlight.
        query_vector: The embedded query vector to compare against.
        embedding_generator: An embedding generator with a .generate() method.

    Returns:
        List of best-matching sentences (same length as texts), None where
        text is too short to split meaningfully.
    """
    # Phase 1: Split all texts into sentences and track ownership
    window_sentence_pairs: list[list[tuple[str, str]]] = []
    all_items: list[EmbeddingInput] = []
    # Maps each embedding item index → (window_index, sentence_index_in_window)
    item_ownership: list[tuple[int, int]] = []

    for window_idx, text in enumerate(texts):
        pairs = _split_sentences(text)
        window_sentence_pairs.append(pairs)
        for sent_idx, (_orig, clean_text) in enumerate(pairs):
            all_items.append(
                EmbeddingInput(
                    embedding_id=EmbeddingId(f"emb_hl_{len(all_items)}"),
                    object_type=ObjectType.CONTEXT_WINDOW,
                    object_id=f"hl_{window_idx}_{sent_idx}",
                    text=clean_text,
                )
            )
            item_ownership.append((window_idx, sent_idx))

    # No sentences to embed — return all None
    if not all_items:
        return [None] * len(texts)

    # Phase 2: Single batch embedding call
    batch = EmbeddingBatch(items=all_items)
    records = embedding_generator.generate(batch)
    if not records:
        return [None] * len(texts)

    # Phase 3: Vectorized cosine similarity
    query_arr = np.array(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query_arr)
    if query_norm == 0:
        return [None] * len(texts)
    query_unit = query_arr / query_norm

    # Stack all sentence vectors into a matrix and normalize rows
    sentence_matrix = np.array([r.vector for r in records], dtype=np.float32)
    norms = np.linalg.norm(sentence_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    sentence_matrix /= norms

    # One matrix-vector multiply → all similarities at once
    similarities = sentence_matrix @ query_unit  # shape: (total_sentences,)

    # Phase 4: Find best sentence per window
    results: list[str | None] = [None] * len(texts)
    for window_idx, pairs in enumerate(window_sentence_pairs):
        if not pairs:
            continue

        # Find indices belonging to this window
        best_score = -1.0
        best_orig = None
        for item_idx, (w_idx, s_idx) in enumerate(item_ownership):
            if w_idx != window_idx:
                continue
            if item_idx < len(similarities) and similarities[item_idx] > best_score:
                best_score = similarities[item_idx]
                best_orig = pairs[s_idx][0]  # original sentence with tags

        results[window_idx] = best_orig

    return results
