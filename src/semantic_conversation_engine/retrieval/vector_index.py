"""In-memory vector index using numpy — flat exact search.

Implements the VectorIndex protocol using brute-force similarity
computation over a numpy matrix. Supports cosine, L2, and dot-product
distance metrics.

This is the foundation index for benchmarking and testing. Results
are exact (no approximation error from ANN), making it the ground
truth reference. FAISS-backed or Qdrant-backed implementations can
be added later as alternative VectorIndex implementations.

Design decisions:
    - Flat search (no ANN) — correctness over speed for foundation.
    - numpy-only — no faiss dependency required.
    - Vectors stored as float32 numpy matrix for efficient batch ops.
    - Cosine similarity via dot product on L2-normalized vectors.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from semantic_conversation_engine.models.embedding_record import EmbeddingRecord
from semantic_conversation_engine.retrieval.config import DistanceMetric, VectorIndexConfig
from semantic_conversation_engine.retrieval.models import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass
class VectorIndexStats:
    """Operational statistics for the vector index.

    Args:
        vector_count: Number of indexed vectors.
        dimensions: Dimensionality of vectors.
        metric: Distance metric in use.
        last_upsert_ms: Time for the last upsert() call.
        last_search_ms: Time for the last search() call.
    """

    vector_count: int = 0
    dimensions: int = 0
    metric: str = ""
    last_upsert_ms: float = 0.0
    last_search_ms: float = 0.0


@dataclass
class _IndexEntry:
    """Internal record linking a vector to its source identity."""

    object_id: str
    object_type: str
    embedding_id: str


@dataclass
class InMemoryVectorIndex:
    """Flat exact-search vector index using numpy.

    Satisfies the VectorIndex protocol. Stores vectors in a numpy
    matrix and performs brute-force similarity search.

    Args:
        config: Vector index configuration (metric, dimensions).
    """

    config: VectorIndexConfig
    _vectors: NDArray[np.float32] | None = field(default=None, repr=False)
    _entries: list[_IndexEntry] = field(default_factory=list, repr=False)
    _stats: VectorIndexStats = field(default_factory=VectorIndexStats, repr=False)

    def upsert(self, records: list[EmbeddingRecord]) -> None:
        """Insert or update embedding records in the index.

        If a record with the same source_id already exists, it is
        replaced. Otherwise, it is appended.

        Args:
            records: Embedding records with vectors to index.

        Raises:
            ValueError: If vector dimensions don't match index config.
        """
        start = time.monotonic()

        for record in records:
            if record.dimensions != self.config.dimensions:
                raise ValueError(
                    f"Vector dimensions mismatch: record has {record.dimensions}, "
                    f"index expects {self.config.dimensions}"
                )

        existing_ids = {e.object_id: idx for idx, e in enumerate(self._entries)}
        new_vectors: list[list[float]] = []
        new_entries: list[_IndexEntry] = []

        vectors_list: list[list[float]] = []
        if self._vectors is not None:
            vectors_list = self._vectors.tolist()

        entries_list = list(self._entries)

        for record in records:
            entry = _IndexEntry(
                object_id=record.source_id,
                object_type=record.source_type.value,
                embedding_id=str(record.embedding_id),
            )
            existing_idx = existing_ids.get(record.source_id)
            if existing_idx is not None:
                vectors_list[existing_idx] = record.vector
                entries_list[existing_idx] = entry
            else:
                new_vectors.append(record.vector)
                new_entries.append(entry)

        vectors_list.extend(new_vectors)
        entries_list.extend(new_entries)

        self._vectors = np.array(vectors_list, dtype=np.float32) if vectors_list else None
        self._entries = entries_list

        elapsed = (time.monotonic() - start) * 1000
        self._stats.last_upsert_ms = round(elapsed, 2)
        self._stats.vector_count = len(self._entries)
        self._stats.dimensions = self.config.dimensions
        self._stats.metric = self.config.metric.value
        logger.debug(
            "VectorIndex upserted %d records in %.2fms (total: %d)",
            len(records),
            elapsed,
            len(self._entries),
        )

    def search_by_vector(self, vector: list[float], top_k: int = 10) -> list[RetrievalHit]:
        """Search the index by vector similarity.

        Args:
            vector: Query vector. Must match the index dimensionality.
            top_k: Maximum number of results.

        Returns:
            Ordered list of hits by descending similarity score.

        Raises:
            ValueError: If query vector dimensions don't match index.
        """
        start = time.monotonic()

        if self._vectors is None or len(self._entries) == 0:
            self._stats.last_search_ms = 0.0
            return []

        if len(vector) != self.config.dimensions:
            raise ValueError(
                f"Query vector dimensions mismatch: got {len(vector)}, index expects {self.config.dimensions}"
            )

        query = np.array(vector, dtype=np.float32)
        scores = self._compute_scores(query)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: (-x[1], self._entries[x[0]].object_id))
        top_results = indexed_scores[:top_k]

        hits: list[RetrievalHit] = []
        for rank, (idx, score) in enumerate(top_results, start=1):
            entry = self._entries[idx]
            hits.append(
                RetrievalHit(
                    object_id=entry.object_id,
                    object_type=entry.object_type,
                    score=round(float(score), 6),
                    lexical_score=None,
                    semantic_score=round(float(score), 6),
                    rank=rank,
                )
            )

        elapsed = (time.monotonic() - start) * 1000
        self._stats.last_search_ms = round(elapsed, 2)
        logger.debug(
            "VectorIndex search returned %d hits in %.2fms",
            len(hits),
            elapsed,
        )
        return hits

    def clear(self) -> None:
        """Remove all indexed vectors and reset state."""
        self._vectors = None
        self._entries.clear()
        self._stats = VectorIndexStats()

    def save(self, path: str | Path) -> None:
        """Persist the index to disk.

        Saves vectors as .npy and metadata as .json in the given
        directory.

        Args:
            path: Directory path to save index files.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self._vectors is not None:
            np.save(dir_path / "vectors.npy", self._vectors)

        meta = {
            "config": {
                "dimensions": self.config.dimensions,
                "metric": self.config.metric.value,
                "index_type": self.config.index_type.value,
            },
            "entries": [
                {
                    "object_id": e.object_id,
                    "object_type": e.object_type,
                    "embedding_id": e.embedding_id,
                }
                for e in self._entries
            ],
        }
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(meta, f)

        logger.info("VectorIndex saved to %s (%d vectors)", dir_path, len(self._entries))

    def load(self, path: str | Path) -> None:
        """Load a previously saved index from disk.

        Args:
            path: Directory path containing index files.

        Raises:
            FileNotFoundError: If the index files don't exist.
            ValueError: If loaded dimensions don't match config.
        """
        dir_path = Path(path)
        vectors_path = dir_path / "vectors.npy"
        meta_path = dir_path / "metadata.json"

        with open(meta_path) as f:
            meta = json.load(f)

        saved_dims = meta["config"]["dimensions"]
        if saved_dims != self.config.dimensions:
            raise ValueError(
                f"Saved index dimensions ({saved_dims}) don't match config dimensions ({self.config.dimensions})"
            )

        self._entries = [
            _IndexEntry(
                object_id=e["object_id"],
                object_type=e["object_type"],
                embedding_id=e["embedding_id"],
            )
            for e in meta["entries"]
        ]

        if vectors_path.exists():
            self._vectors = np.load(vectors_path).astype(np.float32)
        else:
            self._vectors = None

        self._stats.vector_count = len(self._entries)
        self._stats.dimensions = self.config.dimensions
        self._stats.metric = self.config.metric.value
        logger.info("VectorIndex loaded from %s (%d vectors)", dir_path, len(self._entries))

    @property
    def stats(self) -> VectorIndexStats:
        """Current index statistics."""
        return self._stats

    @property
    def vector_count(self) -> int:
        """Number of indexed vectors."""
        return len(self._entries)

    def _compute_scores(self, query: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute similarity scores between query and all indexed vectors.

        Args:
            query: Query vector of shape (dim,).

        Returns:
            1D array of scores, one per indexed vector.
        """
        assert self._vectors is not None
        vectors = self._vectors

        if self.config.metric == DistanceMetric.COSINE:
            query_norm = np.linalg.norm(query)
            doc_norms = np.linalg.norm(vectors, axis=1)
            safe_query_norm = max(float(query_norm), 1e-10)
            safe_doc_norms = np.maximum(doc_norms, 1e-10)
            scores_arr = (vectors @ query) / (safe_doc_norms * safe_query_norm)
        elif self.config.metric == DistanceMetric.DOT_PRODUCT:
            scores_arr = vectors @ query
        elif self.config.metric == DistanceMetric.L2:
            diffs = vectors - query
            distances = np.linalg.norm(diffs, axis=1)
            scores_arr = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unsupported metric: {self.config.metric}")

        result: NDArray[np.float32] = scores_arr.astype(np.float32)
        return result
