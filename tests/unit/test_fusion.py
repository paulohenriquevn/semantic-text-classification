"""Unit tests for score fusion strategies.

Tests cover: RRF fusion, LINEAR fusion, deduplication, score merging,
ordering contract, edge cases, and component score preservation.
"""

from semantic_conversation_engine.retrieval.fusion import (
    linear_fusion,
    reciprocal_rank_fusion,
)
from semantic_conversation_engine.retrieval.models import RetrievalHit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lex_hit(obj_id: str, score: float, rank: int, obj_type: str = "context_window") -> RetrievalHit:
    return RetrievalHit(
        object_id=obj_id,
        object_type=obj_type,
        score=score,
        lexical_score=score,
        semantic_score=None,
        rank=rank,
    )


def _sem_hit(obj_id: str, score: float, rank: int, obj_type: str = "context_window") -> RetrievalHit:
    return RetrievalHit(
        object_id=obj_id,
        object_type=obj_type,
        score=score,
        lexical_score=None,
        semantic_score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# RRF — basic
# ---------------------------------------------------------------------------


class TestRRFBasic:
    def test_disjoint_sets_combined(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        sem = [_sem_hit("b", 0.9, 1)]
        result = reciprocal_rank_fusion(lex, sem)
        ids = [h.object_id for h in result]
        assert "a" in ids
        assert "b" in ids

    def test_shared_hit_gets_higher_score(self) -> None:
        lex = [_lex_hit("a", 1.0, 1), _lex_hit("b", 0.5, 2)]
        sem = [_sem_hit("a", 0.9, 1), _sem_hit("c", 0.8, 2)]
        result = reciprocal_rank_fusion(lex, sem)
        # "a" appears in both → gets 2 RRF contributions
        scores = {h.object_id: h.score for h in result}
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_empty_both_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_empty_semantic_returns_lexical_only(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        result = reciprocal_rank_fusion(lex, [])
        assert len(result) == 1
        assert result[0].object_id == "a"

    def test_empty_lexical_returns_semantic_only(self) -> None:
        sem = [_sem_hit("b", 0.9, 1)]
        result = reciprocal_rank_fusion([], sem)
        assert len(result) == 1
        assert result[0].object_id == "b"


# ---------------------------------------------------------------------------
# RRF — ordering contract
# ---------------------------------------------------------------------------


class TestRRFOrderingContract:
    def test_scores_descending(self) -> None:
        lex = [_lex_hit("a", 1.0, 1), _lex_hit("b", 0.5, 2)]
        sem = [_sem_hit("c", 0.9, 1), _sem_hit("d", 0.4, 2)]
        result = reciprocal_rank_fusion(lex, sem)
        scores = [h.score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_is_one_based(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        sem = [_sem_hit("b", 0.9, 1)]
        result = reciprocal_rank_fusion(lex, sem)
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_tie_break_by_object_id(self) -> None:
        # Same rank in both lists → same RRF score
        lex = [_lex_hit("b_win", 1.0, 1)]
        sem = [_sem_hit("a_win", 0.9, 1)]
        result = reciprocal_rank_fusion(lex, sem, k=60)
        # Both have the same RRF score (1/(60+1)), so tie-break by ID
        assert result[0].object_id == "a_win"
        assert result[1].object_id == "b_win"


# ---------------------------------------------------------------------------
# RRF — score semantics
# ---------------------------------------------------------------------------


class TestRRFScoreSemantics:
    def test_lexical_only_hit_preserves_lexical_score(self) -> None:
        lex = [_lex_hit("a", 2.5, 1)]
        result = reciprocal_rank_fusion(lex, [])
        assert result[0].lexical_score == 2.5
        assert result[0].semantic_score is None

    def test_semantic_only_hit_preserves_semantic_score(self) -> None:
        sem = [_sem_hit("b", 0.95, 1)]
        result = reciprocal_rank_fusion([], sem)
        assert result[0].semantic_score == 0.95
        assert result[0].lexical_score is None

    def test_shared_hit_has_both_scores(self) -> None:
        lex = [_lex_hit("a", 2.5, 1)]
        sem = [_sem_hit("a", 0.95, 1)]
        result = reciprocal_rank_fusion(lex, sem)
        assert result[0].lexical_score == 2.5
        assert result[0].semantic_score == 0.95

    def test_k_parameter_affects_scores(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        r_low_k = reciprocal_rank_fusion(lex, [], k=1)
        r_high_k = reciprocal_rank_fusion(lex, [], k=100)
        # Lower k → higher RRF score for rank 1
        assert r_low_k[0].score > r_high_k[0].score


# ---------------------------------------------------------------------------
# RRF — deduplication
# ---------------------------------------------------------------------------


class TestRRFDeduplication:
    def test_same_id_in_both_lists_deduped(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        sem = [_sem_hit("a", 0.9, 1)]
        result = reciprocal_rank_fusion(lex, sem)
        assert len(result) == 1
        assert result[0].object_id == "a"

    def test_multiple_shared_ids_deduped(self) -> None:
        lex = [_lex_hit("a", 1.0, 1), _lex_hit("b", 0.8, 2)]
        sem = [_sem_hit("b", 0.9, 1), _sem_hit("a", 0.7, 2)]
        result = reciprocal_rank_fusion(lex, sem)
        ids = [h.object_id for h in result]
        assert len(ids) == 2
        assert set(ids) == {"a", "b"}


# ---------------------------------------------------------------------------
# LINEAR fusion — basic
# ---------------------------------------------------------------------------


class TestLinearFusionBasic:
    def test_fuses_two_lists(self) -> None:
        lex = [_lex_hit("a", 1.0, 1), _lex_hit("b", 0.5, 2)]
        sem = [_sem_hit("a", 0.9, 1), _sem_hit("c", 0.8, 2)]
        result = linear_fusion(lex, sem, semantic_weight=0.5)
        assert len(result) == 3
        ids = {h.object_id for h in result}
        assert ids == {"a", "b", "c"}

    def test_empty_both_returns_empty(self) -> None:
        result = linear_fusion([], [])
        assert result == []

    def test_semantic_weight_one_favors_semantic(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        sem = [_sem_hit("b", 0.9, 1)]
        result = linear_fusion(lex, sem, semantic_weight=1.0)
        # Only semantic contributes → b should rank higher
        scores = {h.object_id: h.score for h in result}
        assert scores["b"] > scores["a"]

    def test_semantic_weight_zero_favors_lexical(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        sem = [_sem_hit("b", 0.9, 1)]
        result = linear_fusion(lex, sem, semantic_weight=0.0)
        scores = {h.object_id: h.score for h in result}
        assert scores["a"] > scores["b"]


# ---------------------------------------------------------------------------
# LINEAR fusion — ordering
# ---------------------------------------------------------------------------


class TestLinearFusionOrdering:
    def test_scores_descending(self) -> None:
        lex = [_lex_hit("a", 1.0, 1), _lex_hit("b", 0.5, 2)]
        sem = [_sem_hit("c", 0.9, 1)]
        result = linear_fusion(lex, sem)
        scores = [h.score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_is_one_based(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        sem = [_sem_hit("b", 0.9, 1)]
        result = linear_fusion(lex, sem)
        ranks = [h.rank for h in result]
        assert ranks == [1, 2]


# ---------------------------------------------------------------------------
# LINEAR fusion — score semantics
# ---------------------------------------------------------------------------


class TestLinearFusionScoreSemantics:
    def test_preserves_component_scores(self) -> None:
        lex = [_lex_hit("a", 2.5, 1)]
        sem = [_sem_hit("a", 0.95, 1)]
        result = linear_fusion(lex, sem)
        assert result[0].lexical_score == 2.5
        assert result[0].semantic_score == 0.95

    def test_single_list_normalization(self) -> None:
        lex = [_lex_hit("a", 1.0, 1)]
        result = linear_fusion(lex, [], semantic_weight=0.5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestFusionReexport:
    def test_rrf_importable(self) -> None:
        from semantic_conversation_engine.retrieval import (
            reciprocal_rank_fusion as rrf,
        )

        assert rrf is reciprocal_rank_fusion

    def test_linear_importable(self) -> None:
        from semantic_conversation_engine.retrieval import (
            linear_fusion as lf,
        )

        assert lf is linear_fusion
