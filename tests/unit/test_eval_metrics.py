"""Unit tests for IR evaluation metrics.

Tests cover: recall@k, precision@k, reciprocal_rank (MRR component),
nDCG with binary and graded relevance, edge cases.
"""

import math

from talkex.evaluation.metrics import (
    ndcg,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_recall(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d4", "d5"}
        # 2 out of 4 relevant found in top-3
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_zero_recall(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d4", "d5"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_smaller_than_retrieved(self) -> None:
        retrieved = ["d1", "d2", "d3", "d4"]
        relevant = {"d1", "d3"}
        # Only top-2: d1 found → 1/2
        assert recall_at_k(retrieved, relevant, k=2) == 0.5

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k(["d1", "d2"], set(), k=2) == 0.0

    def test_empty_retrieved_returns_zero(self) -> None:
        assert recall_at_k([], {"d1"}, k=5) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        retrieved = ["d1", "d2"]
        relevant = {"d1", "d2", "d3"}
        # 2 out of 3 found, even though k=10
        result = recall_at_k(retrieved, relevant, k=10)
        assert abs(result - 2.0 / 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_precision(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1"}
        # 1 out of 3 in top-3
        result = precision_at_k(retrieved, relevant, k=3)
        assert abs(result - 1.0 / 3.0) < 1e-9

    def test_zero_precision(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d4", "d5"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_zero_returns_zero(self) -> None:
        assert precision_at_k(["d1"], {"d1"}, k=0) == 0.0

    def test_k_one(self) -> None:
        retrieved = ["d1", "d2"]
        relevant = {"d1"}
        assert precision_at_k(retrieved, relevant, k=1) == 1.0

    def test_k_larger_than_retrieved(self) -> None:
        retrieved = ["d1"]
        relevant = {"d1"}
        # Only 1 doc retrieved, k=5 → precision = 1/5
        assert precision_at_k(retrieved, relevant, k=5) == 0.2


# ---------------------------------------------------------------------------
# Reciprocal Rank (MRR component)
# ---------------------------------------------------------------------------


class TestReciprocalRank:
    def test_first_result_relevant(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1"}
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_result_relevant(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d2"}
        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_result_relevant(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d3"}
        result = reciprocal_rank(retrieved, relevant)
        assert abs(result - 1.0 / 3.0) < 1e-9

    def test_no_relevant_found(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d4"}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_retrieved_returns_zero(self) -> None:
        assert reciprocal_rank([], {"d1"}) == 0.0

    def test_multiple_relevant_returns_first(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d2", "d3"}
        # First relevant is d2 at rank 2
        assert reciprocal_rank(retrieved, relevant) == 0.5


# ---------------------------------------------------------------------------
# nDCG
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking_binary(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevance_map = {"d1": 1, "d2": 1, "d3": 1}
        # Perfect order → nDCG = 1.0
        assert ndcg(retrieved, relevance_map, k=3) == 1.0

    def test_perfect_ranking_graded(self) -> None:
        # Ideal order: d1 (rel=3), d2 (rel=2), d3 (rel=1)
        retrieved = ["d1", "d2", "d3"]
        relevance_map = {"d1": 3, "d2": 2, "d3": 1}
        result = ndcg(retrieved, relevance_map, k=3)
        assert abs(result - 1.0) < 1e-9

    def test_reversed_graded_ranking(self) -> None:
        # Worst order: d3 (rel=1), d2 (rel=2), d1 (rel=3)
        retrieved = ["d3", "d2", "d1"]
        relevance_map = {"d1": 3, "d2": 2, "d3": 1}
        result = ndcg(retrieved, relevance_map, k=3)
        assert result < 1.0
        assert result > 0.0

    def test_no_relevant_docs_returns_zero(self) -> None:
        retrieved = ["d1", "d2"]
        assert ndcg(retrieved, {}, k=2) == 0.0

    def test_all_irrelevant_results_returns_zero(self) -> None:
        retrieved = ["d1", "d2"]
        relevance_map = {"d3": 1}
        assert ndcg(retrieved, relevance_map, k=2) == 0.0

    def test_single_relevant_at_top(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevance_map = {"d1": 1}
        assert ndcg(retrieved, relevance_map, k=3) == 1.0

    def test_single_relevant_not_at_top(self) -> None:
        retrieved = ["d2", "d1", "d3"]
        relevance_map = {"d1": 1}
        result = ndcg(retrieved, relevance_map, k=3)
        # d1 at position 2, ideal is position 1 → nDCG < 1
        assert result < 1.0
        assert result > 0.0

    def test_k_limits_evaluation(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevance_map = {"d3": 1}
        # k=2 → d3 not seen → nDCG = 0
        assert ndcg(retrieved, relevance_map, k=2) == 0.0

    def test_ndcg_manual_calculation(self) -> None:
        # Manual calculation for verification:
        # Retrieved: [d2, d1], relevant: d1(3), d2(2)
        # DCG = (2^2-1)/log2(2) + (2^3-1)/log2(3) = 3/1 + 7/1.585 = 3 + 4.416 = 7.416
        # Ideal: [d1, d2]
        # IDCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) = 7/1 + 3/1.585 = 7 + 1.893 = 8.893
        # nDCG = 7.416 / 8.893 ≈ 0.834
        retrieved = ["d2", "d1"]
        relevance_map = {"d1": 3, "d2": 2}
        result = ndcg(retrieved, relevance_map, k=2)

        dcg = (2**2 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3)
        idcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
        expected = dcg / idcg
        assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestMetricsReexport:
    def test_importable_from_evaluation_package(self) -> None:
        from talkex.evaluation import (
            ndcg as _ndcg,
        )
        from talkex.evaluation import (
            precision_at_k as _pak,
        )
        from talkex.evaluation import (
            recall_at_k as _rak,
        )
        from talkex.evaluation import (
            reciprocal_rank as _rr,
        )

        assert _rak is recall_at_k
        assert _pak is precision_at_k
        assert _rr is reciprocal_rank
        assert _ndcg is ndcg
