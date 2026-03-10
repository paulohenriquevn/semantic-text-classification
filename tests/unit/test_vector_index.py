"""Unit tests for in-memory vector index.

Tests cover: upsert, search, ranking, distance metrics, dimension
validation, save/load, clear, stats, and protocol compliance.
"""

import pytest

from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import ObjectType, PoolingStrategy
from talkex.models.types import EmbeddingId
from talkex.retrieval.config import DistanceMetric, VectorIndexConfig
from talkex.retrieval.vector_index import InMemoryVectorIndex

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


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------


class TestVectorIndexEmpty:
    def test_search_returns_empty(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        assert idx.search_by_vector([0.1, 0.2, 0.3]) == []

    def test_vector_count_is_zero(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        assert idx.vector_count == 0


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class TestVectorIndexUpsert:
    def test_upsert_increases_count(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert([_record("w0"), _record("w1", vector=[0.4, 0.5, 0.6], emb_id="e2")])
        assert idx.vector_count == 2

    def test_upsert_replaces_existing(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert([_record("w0", vector=[1.0, 0.0, 0.0])])
        idx.upsert([_record("w0", vector=[0.0, 1.0, 0.0])])
        assert idx.vector_count == 1
        hits = idx.search_by_vector([0.0, 1.0, 0.0], top_k=1)
        assert hits[0].object_id == "w0"

    def test_rejects_dimension_mismatch(self) -> None:
        idx = InMemoryVectorIndex(config=_config(dims=3))
        bad = _record("w0", vector=[1.0, 2.0], emb_id="e_bad")
        with pytest.raises(ValueError, match="dimensions mismatch"):
            idx.upsert([bad])

    def test_stats_updated(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert([_record()])
        assert idx.stats.vector_count == 1
        assert idx.stats.last_upsert_ms >= 0.0


# ---------------------------------------------------------------------------
# Search — cosine
# ---------------------------------------------------------------------------


class TestVectorIndexCosineSearch:
    def test_finds_most_similar(self) -> None:
        idx = InMemoryVectorIndex(config=_config(metric=DistanceMetric.COSINE))
        idx.upsert(
            [
                _record("similar", vector=[1.0, 0.0, 0.0]),
                _record("orthogonal", vector=[0.0, 1.0, 0.0], emb_id="e2"),
            ]
        )
        hits = idx.search_by_vector([1.0, 0.0, 0.0])
        assert hits[0].object_id == "similar"

    def test_scores_descending(self) -> None:
        idx = InMemoryVectorIndex(config=_config(metric=DistanceMetric.COSINE))
        idx.upsert(
            [
                _record("best", vector=[1.0, 0.0, 0.0]),
                _record("mid", vector=[0.7, 0.7, 0.0], emb_id="e2"),
                _record("worst", vector=[0.0, 0.0, 1.0], emb_id="e3"),
            ]
        )
        hits = idx.search_by_vector([1.0, 0.0, 0.0])
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Search — dot product
# ---------------------------------------------------------------------------


class TestVectorIndexDotProductSearch:
    def test_dot_product_ranking(self) -> None:
        idx = InMemoryVectorIndex(config=_config(metric=DistanceMetric.DOT_PRODUCT))
        idx.upsert(
            [
                _record("high", vector=[3.0, 0.0, 0.0]),
                _record("low", vector=[0.1, 0.0, 0.0], emb_id="e2"),
            ]
        )
        hits = idx.search_by_vector([1.0, 0.0, 0.0])
        assert hits[0].object_id == "high"


# ---------------------------------------------------------------------------
# Search — L2
# ---------------------------------------------------------------------------


class TestVectorIndexL2Search:
    def test_l2_finds_nearest(self) -> None:
        idx = InMemoryVectorIndex(config=_config(metric=DistanceMetric.L2))
        idx.upsert(
            [
                _record("near", vector=[1.0, 0.0, 0.0]),
                _record("far", vector=[10.0, 10.0, 10.0], emb_id="e2"),
            ]
        )
        hits = idx.search_by_vector([1.1, 0.0, 0.0])
        assert hits[0].object_id == "near"


# ---------------------------------------------------------------------------
# Search — ranking contract
# ---------------------------------------------------------------------------


class TestVectorIndexRankingContract:
    def test_rank_is_one_based(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert(
            [
                _record("a", vector=[1.0, 0.0, 0.0]),
                _record("b", vector=[0.0, 1.0, 0.0], emb_id="e2"),
            ]
        )
        hits = idx.search_by_vector([1.0, 0.0, 0.0])
        assert hits[0].rank == 1
        assert hits[1].rank == 2

    def test_respects_top_k(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        for i in range(10):
            v = [0.0, 0.0, 0.0]
            v[i % 3] = 1.0
            idx.upsert([_record(f"w{i}", vector=v, emb_id=f"e{i}")])
        hits = idx.search_by_vector([1.0, 0.0, 0.0], top_k=3)
        assert len(hits) <= 3

    def test_top_k_larger_than_corpus(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert([_record("w0")])
        hits = idx.search_by_vector([0.1, 0.2, 0.3], top_k=100)
        assert len(hits) == 1

    def test_tie_break_by_object_id(self) -> None:
        idx = InMemoryVectorIndex(config=_config(metric=DistanceMetric.COSINE))
        idx.upsert(
            [
                _record("b_win", vector=[1.0, 0.0, 0.0]),
                _record("a_win", vector=[1.0, 0.0, 0.0], emb_id="e2"),
            ]
        )
        hits = idx.search_by_vector([1.0, 0.0, 0.0])
        assert hits[0].object_id == "a_win"
        assert hits[1].object_id == "b_win"


# ---------------------------------------------------------------------------
# Score semantics
# ---------------------------------------------------------------------------


class TestVectorIndexScoreSemantics:
    def test_semantic_score_populated(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert([_record()])
        hits = idx.search_by_vector([0.1, 0.2, 0.3])
        assert hits[0].semantic_score is not None

    def test_lexical_score_is_none(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert([_record()])
        hits = idx.search_by_vector([0.1, 0.2, 0.3])
        assert hits[0].lexical_score is None


# ---------------------------------------------------------------------------
# Dimension validation on search
# ---------------------------------------------------------------------------


class TestVectorIndexSearchValidation:
    def test_rejects_query_dimension_mismatch(self) -> None:
        idx = InMemoryVectorIndex(config=_config(dims=3))
        idx.upsert([_record()])
        with pytest.raises(ValueError, match="dimensions mismatch"):
            idx.search_by_vector([1.0, 2.0])


# ---------------------------------------------------------------------------
# Save/Load
# ---------------------------------------------------------------------------


class TestVectorIndexSaveLoad:
    def test_save_and_load_preserves_results(self, tmp_path: object) -> None:
        path = str(tmp_path)
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert(
            [
                _record("w0", vector=[1.0, 0.0, 0.0]),
                _record("w1", vector=[0.0, 1.0, 0.0], emb_id="e2"),
            ]
        )
        idx.save(path)

        idx2 = InMemoryVectorIndex(config=_config())
        idx2.load(path)
        assert idx2.vector_count == 2
        hits = idx2.search_by_vector([1.0, 0.0, 0.0])
        assert hits[0].object_id == "w0"

    def test_load_rejects_dimension_mismatch(self, tmp_path: object) -> None:
        path = str(tmp_path)
        idx = InMemoryVectorIndex(config=_config(dims=3))
        idx.upsert([_record()])
        idx.save(path)

        idx2 = InMemoryVectorIndex(config=_config(dims=5))
        with pytest.raises(ValueError, match="dimensions"):
            idx2.load(path)


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestVectorIndexClear:
    def test_clear_removes_all(self) -> None:
        idx = InMemoryVectorIndex(config=_config())
        idx.upsert([_record()])
        idx.clear()
        assert idx.vector_count == 0
        assert idx.search_by_vector([0.1, 0.2, 0.3]) == []


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestVectorIndexReexport:
    def test_importable_from_retrieval_package(self) -> None:
        from talkex.retrieval import (
            InMemoryVectorIndex as IMVI,
        )

        assert IMVI is InMemoryVectorIndex
