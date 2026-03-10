"""Qdrant-backed vector index implementing the VectorIndex protocol.

Uses the qdrant-client library for ANN (Approximate Nearest Neighbor)
search. Supports both in-memory mode (`:memory:`) for development and
server mode (URL) for production.

Design decisions:
    - Implements VectorIndex protocol for drop-in replacement.
    - Uses qdrant_client local mode by default — no Docker needed.
    - Cosine metric stored as dot-product on normalized vectors (Qdrant convention).
    - Payload stores object_type and embedding_id for provenance.
    - Collection auto-created on first upsert if not exists.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from talkex.models.embedding_record import EmbeddingRecord
from talkex.retrieval.config import DistanceMetric, VectorIndexConfig
from talkex.retrieval.models import RetrievalHit

logger = logging.getLogger(__name__)

_METRIC_MAP = {
    DistanceMetric.COSINE: "Cosine",
    DistanceMetric.L2: "Euclid",
    DistanceMetric.DOT_PRODUCT: "Dot",
}


@dataclass
class QdrantVectorIndex:
    """ANN vector index backed by Qdrant.

    Satisfies the VectorIndex protocol. Uses qdrant-client for HNSW-based
    approximate nearest neighbor search.

    Args:
        config: Vector index configuration (metric, dimensions).
        collection_name: Qdrant collection name.
        location: Qdrant connection — ":memory:" for local, URL for server.
        path: Disk persistence path for local mode (None = in-memory only).
    """

    config: VectorIndexConfig
    collection_name: str = "talkex_vectors"
    location: str = ":memory:"
    path: str | None = None
    _client: object = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Lazily import and initialize the Qdrant client."""
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for QdrantVectorIndex. Install it with: pip install qdrant-client"
            ) from e

        if self.path:
            self._client = QdrantClient(path=self.path)
        else:
            self._client = QdrantClient(location=self.location)

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist yet."""
        if self._initialized:
            return

        from qdrant_client.models import Distance, VectorParams

        metric_str = _METRIC_MAP.get(self.config.metric)
        if metric_str is None:
            raise ValueError(f"Unsupported metric for Qdrant: {self.config.metric}")

        distance = Distance[metric_str.upper()]

        collections = self._client.get_collections().collections  # type: ignore[union-attr]
        existing_names = {c.name for c in collections}

        if self.collection_name not in existing_names:
            self._client.create_collection(  # type: ignore[union-attr]
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimensions,
                    distance=distance,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dims=%d, metric=%s)",
                self.collection_name,
                self.config.dimensions,
                metric_str,
            )

        self._initialized = True

    def upsert(self, records: list[EmbeddingRecord]) -> None:
        """Insert or update embedding records in the Qdrant collection.

        Args:
            records: Embedding records with vectors to index.

        Raises:
            ValueError: If vector dimensions don't match index config.
        """
        if not records:
            return

        start = time.monotonic()

        for record in records:
            if record.dimensions != self.config.dimensions:
                raise ValueError(
                    f"Vector dimensions mismatch: record has {record.dimensions}, "
                    f"index expects {self.config.dimensions}"
                )

        self._ensure_collection()

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=uuid.uuid5(uuid.NAMESPACE_DNS, record.source_id).hex,
                vector=record.vector,
                payload={
                    "object_id": record.source_id,
                    "object_type": record.source_type.value,
                    "embedding_id": str(record.embedding_id),
                },
            )
            for record in records
        ]

        self._client.upsert(  # type: ignore[union-attr]
            collection_name=self.collection_name,
            points=points,
        )

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "QdrantVectorIndex upserted %d records in %.2fms",
            len(records),
            elapsed,
        )

    def search_by_vector(self, vector: list[float], top_k: int = 10) -> list[RetrievalHit]:
        """Search the Qdrant collection by vector similarity.

        Args:
            vector: Query vector. Must match the index dimensionality.
            top_k: Maximum number of results.

        Returns:
            Ordered list of hits by descending similarity score.

        Raises:
            ValueError: If query vector dimensions don't match index.
        """
        start = time.monotonic()

        if len(vector) != self.config.dimensions:
            raise ValueError(
                f"Query vector dimensions mismatch: got {len(vector)}, index expects {self.config.dimensions}"
            )

        self._ensure_collection()

        results = self._client.query_points(  # type: ignore[union-attr]
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
        )

        hits: list[RetrievalHit] = []
        for rank, point in enumerate(results.points, start=1):
            payload = point.payload or {}
            score = float(point.score) if point.score is not None else 0.0
            hits.append(
                RetrievalHit(
                    object_id=payload.get("object_id", str(point.id)),
                    object_type=payload.get("object_type", "unknown"),
                    score=round(score, 6),
                    lexical_score=None,
                    semantic_score=round(score, 6),
                    rank=rank,
                )
            )

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "QdrantVectorIndex search returned %d hits in %.2fms",
            len(hits),
            elapsed,
        )
        return hits

    def clear(self) -> None:
        """Delete and recreate the collection."""
        collections = self._client.get_collections().collections  # type: ignore[union-attr]
        existing_names = {c.name for c in collections}

        if self.collection_name in existing_names:
            self._client.delete_collection(self.collection_name)  # type: ignore[union-attr]

        self._initialized = False
        logger.info("QdrantVectorIndex cleared collection '%s'", self.collection_name)

    @property
    def vector_count(self) -> int:
        """Number of indexed vectors."""
        if not self._initialized:
            return 0

        info = self._client.get_collection(self.collection_name)  # type: ignore[union-attr]
        return info.points_count or 0
