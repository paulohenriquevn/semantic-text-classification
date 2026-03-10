"""Unit tests for embedding cache.

Tests cover: cache get/put, LRU eviction, hit/miss counting, hit rate,
make_cache_key, clear, contains, and reexport.
"""

import pytest

from talkex.embeddings.cache import (
    EmbeddingCache,
    make_cache_key,
)
from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import ObjectType, PoolingStrategy
from talkex.models.types import EmbeddingId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    emb_id: str = "emb_001",
    source_id: str = "obj_001",
) -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id=EmbeddingId(emb_id),
        source_id=source_id,
        source_type=ObjectType.CONTEXT_WINDOW,
        model_name="test-model",
        model_version="1.0",
        pooling_strategy=PoolingStrategy.MEAN,
        dimensions=3,
        vector=[0.1, 0.2, 0.3],
    )


def _key(obj_id: str = "obj_001") -> tuple[str, str, str, str]:
    return make_cache_key(obj_id, "test-model", "1.0", PoolingStrategy.MEAN)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_produces_4_tuple(self) -> None:
        key = make_cache_key("obj_1", "model", "1.0", PoolingStrategy.MEAN)
        assert len(key) == 4

    def test_includes_all_fields(self) -> None:
        key = make_cache_key("obj_1", "e5-base", "2.0", PoolingStrategy.MAX)
        assert key == ("obj_1", "e5-base", "2.0", "max")

    def test_different_pooling_produces_different_key(self) -> None:
        k1 = make_cache_key("obj_1", "m", "1.0", PoolingStrategy.MEAN)
        k2 = make_cache_key("obj_1", "m", "1.0", PoolingStrategy.MAX)
        assert k1 != k2

    def test_different_version_produces_different_key(self) -> None:
        k1 = make_cache_key("obj_1", "m", "1.0", PoolingStrategy.MEAN)
        k2 = make_cache_key("obj_1", "m", "2.0", PoolingStrategy.MEAN)
        assert k1 != k2


# ---------------------------------------------------------------------------
# Basic get/put
# ---------------------------------------------------------------------------


class TestCacheBasicOperations:
    def test_miss_returns_none(self) -> None:
        cache = EmbeddingCache()
        assert cache.get(_key("nonexistent")) is None

    def test_put_then_get(self) -> None:
        cache = EmbeddingCache()
        key = _key()
        record = _record()
        cache.put(key, record)
        assert cache.get(key) is record

    def test_contains_after_put(self) -> None:
        cache = EmbeddingCache()
        key = _key()
        cache.put(key, _record())
        assert cache.contains(key) is True

    def test_not_contains_before_put(self) -> None:
        cache = EmbeddingCache()
        assert cache.contains(_key()) is False

    def test_size_increases_on_put(self) -> None:
        cache = EmbeddingCache()
        assert cache.size == 0
        cache.put(_key("a"), _record(source_id="a"))
        assert cache.size == 1
        cache.put(_key("b"), _record(source_id="b"))
        assert cache.size == 2

    def test_overwrite_same_key(self) -> None:
        cache = EmbeddingCache()
        key = _key()
        r1 = _record(emb_id="emb_1")
        r2 = _record(emb_id="emb_2")
        cache.put(key, r1)
        cache.put(key, r2)
        assert cache.get(key) is r2
        assert cache.size == 1


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestCacheLRUEviction:
    def test_evicts_oldest_when_full(self) -> None:
        cache = EmbeddingCache(max_size=2)
        cache.put(_key("a"), _record(source_id="a"))
        cache.put(_key("b"), _record(source_id="b"))
        cache.put(_key("c"), _record(source_id="c"))
        # "a" should be evicted
        assert cache.get(_key("a")) is None
        assert cache.get(_key("b")) is not None
        assert cache.get(_key("c")) is not None
        assert cache.size == 2

    def test_access_promotes_and_prevents_eviction(self) -> None:
        cache = EmbeddingCache(max_size=2)
        cache.put(_key("a"), _record(source_id="a"))
        cache.put(_key("b"), _record(source_id="b"))
        # Access "a" to promote it
        cache.get(_key("a"))
        # Now add "c" — "b" should be evicted (least recently used)
        cache.put(_key("c"), _record(source_id="c"))
        assert cache.get(_key("a")) is not None
        assert cache.get(_key("b")) is None
        assert cache.get(_key("c")) is not None

    def test_max_size_one(self) -> None:
        cache = EmbeddingCache(max_size=1)
        cache.put(_key("a"), _record(source_id="a"))
        cache.put(_key("b"), _record(source_id="b"))
        assert cache.size == 1
        assert cache.get(_key("a")) is None
        assert cache.get(_key("b")) is not None


# ---------------------------------------------------------------------------
# Hit/miss counting
# ---------------------------------------------------------------------------


class TestCacheHitMissCounting:
    def test_initial_counts_are_zero(self) -> None:
        cache = EmbeddingCache()
        assert cache.hits == 0
        assert cache.misses == 0

    def test_miss_increments_misses(self) -> None:
        cache = EmbeddingCache()
        cache.get(_key("nope"))
        assert cache.misses == 1
        assert cache.hits == 0

    def test_hit_increments_hits(self) -> None:
        cache = EmbeddingCache()
        key = _key()
        cache.put(key, _record())
        cache.get(key)
        assert cache.hits == 1
        assert cache.misses == 0

    def test_mixed_hits_and_misses(self) -> None:
        cache = EmbeddingCache()
        key = _key()
        cache.put(key, _record())
        cache.get(key)  # hit
        cache.get(key)  # hit
        cache.get(_key("x"))  # miss
        assert cache.hits == 2
        assert cache.misses == 1


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------


class TestCacheHitRate:
    def test_zero_lookups_returns_zero(self) -> None:
        cache = EmbeddingCache()
        assert cache.hit_rate == 0.0

    def test_all_hits(self) -> None:
        cache = EmbeddingCache()
        key = _key()
        cache.put(key, _record())
        cache.get(key)
        cache.get(key)
        assert cache.hit_rate == 1.0

    def test_all_misses(self) -> None:
        cache = EmbeddingCache()
        cache.get(_key("a"))
        cache.get(_key("b"))
        assert cache.hit_rate == 0.0

    def test_fifty_percent(self) -> None:
        cache = EmbeddingCache()
        key = _key()
        cache.put(key, _record())
        cache.get(key)  # hit
        cache.get(_key("x"))  # miss
        assert cache.hit_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestCacheClear:
    def test_clear_removes_all_entries(self) -> None:
        cache = EmbeddingCache()
        cache.put(_key("a"), _record(source_id="a"))
        cache.put(_key("b"), _record(source_id="b"))
        cache.clear()
        assert cache.size == 0

    def test_clear_resets_stats(self) -> None:
        cache = EmbeddingCache()
        key = _key()
        cache.put(key, _record())
        cache.get(key)
        cache.get(_key("x"))
        cache.clear()
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.hit_rate == 0.0


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestCacheReexport:
    def test_importable_from_embeddings_package(self) -> None:
        from talkex.embeddings import (
            EmbeddingCache as EC,
        )
        from talkex.embeddings import (
            make_cache_key as mck,
        )

        assert EC is EmbeddingCache
        assert mck is make_cache_key
