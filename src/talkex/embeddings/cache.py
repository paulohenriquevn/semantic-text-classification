"""In-memory embedding cache for deduplication and reuse.

Caches EmbeddingRecords by a stable composite key:
    (object_id, model_name, model_version, pooling_strategy)

This key uniquely identifies a vector — the same text embedded with the
same model and pooling produces the same result (determinism contract).

The cache is LRU-bounded to prevent unbounded memory growth. Cache hits
skip embedding generation entirely, returning the stored record.

Design decisions:
    - Uses functools-style LRU rather than a custom dict to keep
      implementation simple (KISS).
    - Cache is per-process, not distributed — suitable for batch
      pipelines and single-node inference.
    - Thread-safe via the GIL for dict operations; not designed for
      multi-process shared memory.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field

from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import PoolingStrategy

logger = logging.getLogger(__name__)

CacheKey = tuple[str, str, str, str]


def make_cache_key(
    object_id: str,
    model_name: str,
    model_version: str,
    pooling_strategy: PoolingStrategy,
) -> CacheKey:
    """Build a stable cache key from embedding identity fields.

    Args:
        object_id: Source object ID.
        model_name: Embedding model name.
        model_version: Embedding model version.
        pooling_strategy: Pooling strategy used.

    Returns:
        A 4-tuple suitable as a dict key.
    """
    return (object_id, model_name, model_version, pooling_strategy.value)


@dataclass
class EmbeddingCache:
    """LRU-bounded in-memory cache for EmbeddingRecords.

    Args:
        max_size: Maximum number of records to cache. When exceeded,
            the least recently used entry is evicted.
    """

    max_size: int = 10_000
    _store: OrderedDict[CacheKey, EmbeddingRecord] = field(default_factory=OrderedDict, repr=False)
    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)

    def get(self, key: CacheKey) -> EmbeddingRecord | None:
        """Look up a cached record and promote it to most-recent.

        Args:
            key: Cache key from make_cache_key().

        Returns:
            The cached EmbeddingRecord, or None on miss.
        """
        record = self._store.get(key)
        if record is not None:
            self._store.move_to_end(key)
            self._hits += 1
            return record
        self._misses += 1
        return None

    def put(self, key: CacheKey, record: EmbeddingRecord) -> None:
        """Store a record, evicting LRU if at capacity.

        Args:
            key: Cache key from make_cache_key().
            record: The EmbeddingRecord to cache.
        """
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = record
            return
        self._store[key] = record
        if len(self._store) > self.max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("Cache evicted key: %s", evicted_key)

    def contains(self, key: CacheKey) -> bool:
        """Check if a key exists without promoting it.

        Args:
            key: Cache key to check.

        Returns:
            True if the key is in the cache.
        """
        return key in self._store

    def clear(self) -> None:
        """Remove all entries and reset stats."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._store)

    @property
    def hits(self) -> int:
        """Total cache hits since creation or last clear."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses since creation or last clear."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0.0, 1.0].

        Returns 0.0 if no lookups have been performed.
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
