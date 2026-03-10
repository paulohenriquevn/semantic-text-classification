"""Unit tests for Qdrant vector index adapter.

Tests cover: upsert, search, ranking, dimension validation, clear,
vector_count, protocol compliance with VectorIndex, and re-export.
"""

import pytest

from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import ObjectType, PoolingStrategy
from talkex.models.types import EmbeddingId
from talkex.retrieval.config import DistanceMetric, VectorIndexConfig
from talkex.retrieval.qdrant import QdrantVectorIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(dims: int = 3, metric: DistanceMetric = DistanceMetric.COSINE) -> VectorIndexConfig:
    return VectorIndexConfig(dimensions=dims, metric=metric)


def _record(
    source_id: str = "win_0",
    vector: list[float] | None = None,
    dims: int = 3,
    emb_id: str = "emb_001",
) -> EmbeddingRecord:
    v = vector if vector is not None else [0.1, 0.2, 0.3][:dims]
    return EmbeddingRecord(
        embedding_id=EmbeddingId(emb_id),
        source_id=source_id,
        source_type=ObjectType.CONTEXT_WINDOW,
        model_name="test",
        model_version="1.0",
        pooling_strategy=PoolingStrategy.MEAN,
        dimensions=len(v),
        vector=v,
    )


def _make_index(dims: int = 3, metric: DistanceMetric = DistanceMetric.COSINE) -> QdrantVectorIndex:
    return QdrantVectorIndex(config=_config(dims, metric))


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------


class TestQdrantIndexEmpty:
    def test_search_returns_empty_on_new_collection(self) -> None:
        idx = _make_index()
        assert idx.search_by_vector([0.1, 0.2, 0.3]) == []

    def test_vector_count_is_zero_before_upsert(self) -> None:
        idx = _make_index()
        assert idx.vector_count == 0


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class TestQdrantIndexUpsert:
    def test_upsert_single_record(self) -> None:
        idx = _make_index()
        idx.upsert([_record()])
        assert idx.vector_count == 1

    def test_upsert_multiple_records(self) -> None:
        idx = _make_index()
        records = [
            _record(source_id="win_0", vector=[1.0, 0.0, 0.0], emb_id="emb_0"),
            _record(source_id="win_1", vector=[0.0, 1.0, 0.0], emb_id="emb_1"),
            _record(source_id="win_2", vector=[0.0, 0.0, 1.0], emb_id="emb_2"),
        ]
        idx.upsert(records)
        assert idx.vector_count == 3

    def test_upsert_replaces_existing_record(self) -> None:
        idx = _make_index()
        idx.upsert([_record(source_id="win_0", vector=[1.0, 0.0, 0.0])])
        idx.upsert([_record(source_id="win_0", vector=[0.0, 1.0, 0.0])])
        assert idx.vector_count == 1

    def test_upsert_rejects_dimension_mismatch(self) -> None:
        idx = _make_index(dims=3)
        bad_record = _record(source_id="win_x", vector=[1.0, 0.0], dims=2)
        with pytest.raises(ValueError, match="dimensions mismatch"):
            idx.upsert([bad_record])

    def test_upsert_empty_list_is_noop(self) -> None:
        idx = _make_index()
        idx.upsert([])
        assert idx.vector_count == 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestQdrantIndexSearch:
    def test_search_returns_correct_top_k(self) -> None:
        idx = _make_index()
        records = [
            _record(source_id=f"win_{i}", vector=[float(i == j) for j in range(3)], emb_id=f"emb_{i}") for i in range(3)
        ]
        idx.upsert(records)
        hits = idx.search_by_vector([1.0, 0.0, 0.0], top_k=2)
        assert len(hits) == 2

    def test_search_ranks_by_similarity(self) -> None:
        idx = _make_index()
        idx.upsert(
            [
                _record(source_id="win_exact", vector=[1.0, 0.0, 0.0], emb_id="emb_a"),
                _record(source_id="win_partial", vector=[0.7, 0.7, 0.0], emb_id="emb_b"),
                _record(source_id="win_ortho", vector=[0.0, 0.0, 1.0], emb_id="emb_c"),
            ]
        )
        hits = idx.search_by_vector([1.0, 0.0, 0.0], top_k=3)
        assert hits[0].object_id == "win_exact"
        assert hits[0].score > hits[1].score

    def test_search_hit_has_correct_fields(self) -> None:
        idx = _make_index()
        idx.upsert([_record(source_id="win_42", vector=[1.0, 0.0, 0.0])])
        hits = idx.search_by_vector([1.0, 0.0, 0.0], top_k=1)
        assert len(hits) == 1
        hit = hits[0]
        assert hit.object_id == "win_42"
        assert hit.object_type == "context_window"
        assert hit.rank == 1
        assert hit.semantic_score is not None
        assert hit.lexical_score is None

    def test_search_rejects_dimension_mismatch(self) -> None:
        idx = _make_index(dims=3)
        idx.upsert([_record()])
        with pytest.raises(ValueError, match="dimensions mismatch"):
            idx.search_by_vector([1.0, 0.0], top_k=1)

    def test_search_respects_top_k_limit(self) -> None:
        idx = _make_index()
        records = [_record(source_id=f"win_{i}", vector=[float(i + 1), 0.0, 0.0], emb_id=f"emb_{i}") for i in range(5)]
        idx.upsert(records)
        hits = idx.search_by_vector([1.0, 0.0, 0.0], top_k=3)
        assert len(hits) == 3


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestQdrantIndexClear:
    def test_clear_removes_all_vectors(self) -> None:
        idx = _make_index()
        idx.upsert([_record()])
        assert idx.vector_count == 1
        idx.clear()
        assert idx.vector_count == 0

    def test_clear_allows_reinsertion(self) -> None:
        idx = _make_index()
        idx.upsert([_record(source_id="win_a")])
        idx.clear()
        idx.upsert([_record(source_id="win_b")])
        assert idx.vector_count == 1


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestQdrantIndexProtocol:
    def test_has_upsert_method(self) -> None:
        idx = _make_index()
        assert hasattr(idx, "upsert")
        assert callable(idx.upsert)

    def test_has_search_by_vector_method(self) -> None:
        idx = _make_index()
        assert hasattr(idx, "search_by_vector")
        assert callable(idx.search_by_vector)

    def test_interchangeable_with_inmemory_index(self) -> None:
        """Both index types produce compatible RetrievalHit results."""
        from talkex.retrieval.vector_index import InMemoryVectorIndex

        record = _record(source_id="win_0", vector=[1.0, 0.0, 0.0])
        query = [1.0, 0.0, 0.0]
        config = _config()

        qdrant_idx = QdrantVectorIndex(config=config)
        qdrant_idx.upsert([record])
        qdrant_hits = qdrant_idx.search_by_vector(query, top_k=1)

        inmem_idx = InMemoryVectorIndex(config=config)
        inmem_idx.upsert([record])
        inmem_hits = inmem_idx.search_by_vector(query, top_k=1)

        assert len(qdrant_hits) == len(inmem_hits) == 1
        assert qdrant_hits[0].object_id == inmem_hits[0].object_id


# ---------------------------------------------------------------------------
# Re-export
# ---------------------------------------------------------------------------


class TestQdrantReexport:
    def test_reexported_from_retrieval_package(self) -> None:
        from talkex.retrieval import QdrantVectorIndex as Reexported

        assert Reexported is QdrantVectorIndex
