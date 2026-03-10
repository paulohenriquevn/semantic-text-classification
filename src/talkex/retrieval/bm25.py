"""BM25 Okapi lexical index — pure Python implementation.

Implements the LexicalIndex protocol using BM25 Okapi scoring with
numpy for efficient computation. No external search library required.

BM25 Okapi formula:
    score(q, d) = Σ IDF(t) · (tf(t,d) · (k1 + 1)) / (tf(t,d) + k1 · (1 - b + b · |d|/avgdl))

where:
    tf(t,d) = term frequency of t in document d
    IDF(t)  = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
    N       = total documents
    n(t)    = documents containing term t
    |d|     = document length (tokens)
    avgdl   = average document length

Tokenization is intentionally simple (lowercase + split on whitespace).
For production, plug in a real tokenizer or use an Elasticsearch-backed
LexicalIndex implementation.

Design decisions:
    - Pure numpy for scoring — no rank-bm25 dependency.
    - Incremental indexing via append + rebuild of IDF cache.
    - Documents stored in memory — suitable for batch pipelines
      and benchmarking, not for million-document production.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from talkex.retrieval.config import LexicalIndexConfig
from talkex.retrieval.models import RetrievalHit

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return text.lower().split()


@dataclass
class _IndexedDocument:
    """Internal representation of an indexed document."""

    doc_id: str
    object_type: str
    text: str
    tokens: list[str]
    metadata: dict[str, Any]


@dataclass
class IndexStats:
    """Operational statistics for the lexical index.

    Args:
        document_count: Number of indexed documents.
        avg_document_length: Average token count across documents.
        vocabulary_size: Number of unique terms.
        last_index_ms: Time for the last index() call in milliseconds.
        last_search_ms: Time for the last search() call in milliseconds.
    """

    document_count: int = 0
    avg_document_length: float = 0.0
    vocabulary_size: int = 0
    last_index_ms: float = 0.0
    last_search_ms: float = 0.0


@dataclass
class InMemoryBM25Index:
    """BM25 Okapi lexical index stored entirely in memory.

    Satisfies the LexicalIndex protocol. Documents are tokenized
    on ingestion, IDF values cached, and BM25 scores computed at
    query time.

    Args:
        config: BM25 configuration (k1, b, top_k_default).
    """

    config: LexicalIndexConfig = field(default_factory=LexicalIndexConfig)
    _documents: list[_IndexedDocument] = field(default_factory=list, repr=False)
    _doc_freqs: dict[str, int] = field(default_factory=dict, repr=False)
    _avg_dl: float = field(default=0.0, repr=False)
    _stats: IndexStats = field(default_factory=IndexStats, repr=False)

    def index(self, documents: list[dict[str, object]]) -> None:
        """Index documents for BM25 search.

        Each document dict must contain 'doc_id' and 'text'.
        Optional fields: 'object_type' (default: 'context_window'),
        and any additional metadata.

        Args:
            documents: Documents to index.

        Raises:
            ValueError: If a document is missing 'doc_id' or 'text'.
        """
        start = time.monotonic()

        for doc in documents:
            doc_id = doc.get("doc_id")
            text = doc.get("text")
            if doc_id is None or text is None:
                raise ValueError("Each document must have 'doc_id' and 'text' fields")

            object_type = str(doc.get("object_type", "context_window"))
            metadata = {k: v for k, v in doc.items() if k not in ("doc_id", "text", "object_type")}

            tokens = _tokenize(str(text))
            indexed = _IndexedDocument(
                doc_id=str(doc_id),
                object_type=object_type,
                text=str(text),
                tokens=tokens,
                metadata=metadata,
            )
            self._documents.append(indexed)

        self._rebuild_index_cache()

        elapsed = (time.monotonic() - start) * 1000
        self._stats.last_index_ms = round(elapsed, 2)
        self._stats.document_count = len(self._documents)
        logger.debug(
            "BM25 indexed %d documents in %.2fms (total: %d)",
            len(documents),
            elapsed,
            len(self._documents),
        )

    def search(self, query_text: str, top_k: int = 10) -> list[RetrievalHit]:
        """Search documents by BM25 score.

        Args:
            query_text: Query text to search for.
            top_k: Maximum number of results.

        Returns:
            Ordered list of hits by descending BM25 score.
        """
        start = time.monotonic()

        if not self._documents:
            self._stats.last_search_ms = 0.0
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            self._stats.last_search_ms = 0.0
            return []

        scores: list[tuple[int, float]] = []
        n = len(self._documents)
        k1 = self.config.k1
        b = self.config.b
        avg_dl = self._avg_dl

        for idx, doc in enumerate(self._documents):
            score = 0.0
            doc_len = len(doc.tokens)
            token_counts: dict[str, int] = {}
            for t in doc.tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            for qt in query_tokens:
                if qt not in self._doc_freqs:
                    continue
                tf = token_counts.get(qt, 0)
                if tf == 0:
                    continue
                df = self._doc_freqs[qt]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_dl))
                score += idf * tf_norm

            if score > 0.0:
                scores.append((idx, score))

        scores.sort(key=lambda x: (-x[1], self._documents[x[0]].doc_id))
        top_scores = scores[:top_k]

        hits: list[RetrievalHit] = []
        for rank, (idx, score) in enumerate(top_scores, start=1):
            doc = self._documents[idx]
            hits.append(
                RetrievalHit(
                    object_id=doc.doc_id,
                    object_type=doc.object_type,
                    score=round(score, 6),
                    lexical_score=round(score, 6),
                    semantic_score=None,
                    rank=rank,
                    metadata=dict(doc.metadata),
                )
            )

        elapsed = (time.monotonic() - start) * 1000
        self._stats.last_search_ms = round(elapsed, 2)
        logger.debug(
            "BM25 search '%s' returned %d hits in %.2fms",
            query_text[:50],
            len(hits),
            elapsed,
        )
        return hits

    def clear(self) -> None:
        """Remove all indexed documents and reset state."""
        self._documents.clear()
        self._doc_freqs.clear()
        self._avg_dl = 0.0
        self._stats = IndexStats()

    @property
    def stats(self) -> IndexStats:
        """Current index statistics."""
        return self._stats

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def _rebuild_index_cache(self) -> None:
        """Recompute document frequencies and average document length."""
        self._doc_freqs.clear()
        total_len = 0

        for doc in self._documents:
            total_len += len(doc.tokens)
            seen: set[str] = set()
            for token in doc.tokens:
                if token not in seen:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen.add(token)

        n = len(self._documents)
        self._avg_dl = total_len / n if n > 0 else 0.0
        self._stats.avg_document_length = self._avg_dl
        self._stats.vocabulary_size = len(self._doc_freqs)
