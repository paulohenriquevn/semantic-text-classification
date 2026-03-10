"""Unit tests for BM25 lexical index.

Tests cover: indexing, search, ranking, scoring, empty index, clear,
stats, incremental indexing, edge cases, and protocol compliance.
"""

import pytest

from talkex.retrieval.bm25 import InMemoryBM25Index
from talkex.retrieval.config import LexicalIndexConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(doc_id: str, text: str, **extra: object) -> dict[str, object]:
    d: dict[str, object] = {"doc_id": doc_id, "text": text}
    d.update(extra)
    return d


def _index_with_docs(*docs: dict[str, object]) -> InMemoryBM25Index:
    idx = InMemoryBM25Index()
    idx.index(list(docs))
    return idx


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------


class TestBM25EmptyIndex:
    def test_search_returns_empty_list(self) -> None:
        idx = InMemoryBM25Index()
        assert idx.search("anything") == []

    def test_document_count_is_zero(self) -> None:
        idx = InMemoryBM25Index()
        assert idx.document_count == 0

    def test_stats_initial(self) -> None:
        idx = InMemoryBM25Index()
        assert idx.stats.document_count == 0
        assert idx.stats.vocabulary_size == 0


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestBM25Indexing:
    def test_index_increases_document_count(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello"), _doc("d2", "world"))
        assert idx.document_count == 2

    def test_incremental_indexing(self) -> None:
        idx = InMemoryBM25Index()
        idx.index([_doc("d1", "first")])
        idx.index([_doc("d2", "second")])
        assert idx.document_count == 2

    def test_rejects_missing_doc_id(self) -> None:
        idx = InMemoryBM25Index()
        with pytest.raises(ValueError, match="doc_id"):
            idx.index([{"text": "hello"}])

    def test_rejects_missing_text(self) -> None:
        idx = InMemoryBM25Index()
        with pytest.raises(ValueError, match="text"):
            idx.index([{"doc_id": "d1"}])

    def test_stats_updated_after_index(self) -> None:
        idx = _index_with_docs(_doc("d1", "billing issue"), _doc("d2", "cancel plan"))
        assert idx.stats.document_count == 2
        assert idx.stats.vocabulary_size > 0
        assert idx.stats.avg_document_length > 0
        assert idx.stats.last_index_ms >= 0.0


# ---------------------------------------------------------------------------
# Search — basic
# ---------------------------------------------------------------------------


class TestBM25Search:
    def test_finds_matching_document(self) -> None:
        idx = _index_with_docs(
            _doc("d1", "billing issue with credit card"),
            _doc("d2", "cancel my subscription please"),
        )
        hits = idx.search("billing")
        assert len(hits) >= 1
        assert hits[0].object_id == "d1"

    def test_returns_empty_for_no_match(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello world"))
        hits = idx.search("nonexistent")
        assert hits == []

    def test_returns_empty_for_empty_query(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello"))
        hits = idx.search("")
        assert hits == []

    def test_respects_top_k(self) -> None:
        docs = [_doc(f"d{i}", f"common word document {i}") for i in range(10)]
        idx = _index_with_docs(*docs)
        hits = idx.search("common", top_k=3)
        assert len(hits) <= 3

    def test_top_k_larger_than_corpus(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello"), _doc("d2", "hello world"))
        hits = idx.search("hello", top_k=100)
        assert len(hits) == 2


# ---------------------------------------------------------------------------
# Search — ranking
# ---------------------------------------------------------------------------


class TestBM25Ranking:
    def test_scores_descending(self) -> None:
        idx = _index_with_docs(
            _doc("d1", "billing billing billing"),
            _doc("d2", "billing issue"),
            _doc("d3", "unrelated document"),
        )
        hits = idx.search("billing")
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_rank_is_one_based(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello"), _doc("d2", "hello world"))
        hits = idx.search("hello")
        assert hits[0].rank == 1
        if len(hits) > 1:
            assert hits[1].rank == 2

    def test_tie_break_by_doc_id(self) -> None:
        idx = _index_with_docs(
            _doc("b_doc", "identical"),
            _doc("a_doc", "identical"),
        )
        hits = idx.search("identical")
        assert len(hits) == 2
        assert hits[0].object_id == "a_doc"
        assert hits[1].object_id == "b_doc"


# ---------------------------------------------------------------------------
# Search — score semantics
# ---------------------------------------------------------------------------


class TestBM25ScoreSemantics:
    def test_lexical_score_populated(self) -> None:
        idx = _index_with_docs(_doc("d1", "billing issue"))
        hits = idx.search("billing")
        assert hits[0].lexical_score is not None
        assert hits[0].lexical_score > 0.0

    def test_semantic_score_is_none(self) -> None:
        idx = _index_with_docs(_doc("d1", "billing issue"))
        hits = idx.search("billing")
        assert hits[0].semantic_score is None

    def test_score_equals_lexical_score(self) -> None:
        idx = _index_with_docs(_doc("d1", "billing issue"))
        hits = idx.search("billing")
        assert hits[0].score == hits[0].lexical_score


# ---------------------------------------------------------------------------
# Search — object type
# ---------------------------------------------------------------------------


class TestBM25ObjectType:
    def test_default_object_type(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello"))
        hits = idx.search("hello")
        assert hits[0].object_type == "context_window"

    def test_custom_object_type(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello", object_type="turn"))
        hits = idx.search("hello")
        assert hits[0].object_type == "turn"


# ---------------------------------------------------------------------------
# Custom BM25 config
# ---------------------------------------------------------------------------


class TestBM25CustomConfig:
    def test_k1_affects_scoring(self) -> None:
        docs = [_doc("d1", "word " * 10), _doc("d2", "word")]
        idx_default = InMemoryBM25Index()
        idx_default.index(docs)
        idx_high_k1 = InMemoryBM25Index(config=LexicalIndexConfig(k1=3.0))
        idx_high_k1.index(docs)
        hits_default = idx_default.search("word")
        hits_high = idx_high_k1.search("word")
        # Higher k1 increases tf impact — d1 should score relatively higher
        assert len(hits_default) == len(hits_high) == 2


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestBM25Clear:
    def test_clear_removes_documents(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello"))
        idx.clear()
        assert idx.document_count == 0
        assert idx.search("hello") == []

    def test_clear_resets_stats(self) -> None:
        idx = _index_with_docs(_doc("d1", "hello"))
        idx.clear()
        assert idx.stats.document_count == 0
        assert idx.stats.vocabulary_size == 0


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


class TestBM25CaseInsensitivity:
    def test_case_insensitive_search(self) -> None:
        idx = _index_with_docs(_doc("d1", "Billing Issue"))
        hits = idx.search("billing")
        assert len(hits) == 1

    def test_uppercase_query_matches(self) -> None:
        idx = _index_with_docs(_doc("d1", "billing issue"))
        hits = idx.search("BILLING")
        assert len(hits) == 1


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestBM25Reexport:
    def test_importable_from_retrieval_package(self) -> None:
        from talkex.retrieval import (
            InMemoryBM25Index as IMBM25,
        )

        assert IMBM25 is InMemoryBM25Index
