"""IR evaluation metrics — pure functions over ranked result lists.

All functions take a list of retrieved document IDs (in rank order)
and a set/map of relevant documents, returning a single float score.

Supported metrics:
    recall_at_k:      fraction of relevant docs found in top-K
    precision_at_k:   fraction of top-K results that are relevant
    reciprocal_rank:  1 / rank of the first relevant document (MRR component)
    ndcg:             normalized discounted cumulative gain (position-sensitive,
                      supports graded relevance)

All functions are deterministic and stateless.
"""

from __future__ import annotations

import math


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of relevant documents retrieved in the top-K results.

    Args:
        retrieved: Document IDs in rank order (best first).
        relevant: Set of known relevant document IDs.
        k: Cutoff position.

    Returns:
        Recall@K in [0.0, 1.0]. Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of top-K results that are relevant.

    Args:
        retrieved: Document IDs in rank order (best first).
        relevant: Set of known relevant document IDs.
        k: Cutoff position.

    Returns:
        Precision@K in [0.0, 1.0]. Returns 0.0 if k is 0.
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def reciprocal_rank(
    retrieved: list[str],
    relevant: set[str],
) -> float:
    """Reciprocal of the rank of the first relevant document.

    This is the per-query component of Mean Reciprocal Rank (MRR).
    MRR = mean of reciprocal_rank across all queries.

    Args:
        retrieved: Document IDs in rank order (best first).
        relevant: Set of known relevant document IDs.

    Returns:
        1/rank of first relevant doc, or 0.0 if none found.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(
    retrieved: list[str],
    relevance_map: dict[str, int],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at position K.

    Measures ranking quality with position-sensitive discounting.
    Supports graded relevance (relevance > 1 for highly relevant docs).

    DCG@K  = Σ (2^rel_i - 1) / log2(i + 2)  for i in [0, K)
    IDCG@K = DCG@K computed on the ideal (best possible) ranking
    nDCG@K = DCG@K / IDCG@K

    Args:
        retrieved: Document IDs in rank order (best first).
        relevance_map: Mapping from document_id to relevance grade.
            Documents not in the map have relevance 0.
        k: Cutoff position.

    Returns:
        nDCG@K in [0.0, 1.0]. Returns 0.0 if no relevant docs exist.
    """
    if not relevance_map:
        return 0.0

    # Actual DCG from the retrieved ranking
    dcg = _dcg(retrieved[:k], relevance_map)

    # Ideal DCG: sort all relevant docs by relevance descending
    ideal_order = sorted(relevance_map.keys(), key=lambda d: -relevance_map[d])
    idcg = _dcg(ideal_order[:k], relevance_map)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _dcg(
    doc_ids: list[str],
    relevance_map: dict[str, int],
) -> float:
    """Compute Discounted Cumulative Gain for a ranked list.

    Args:
        doc_ids: Document IDs in rank order.
        relevance_map: Mapping from document_id to relevance grade.

    Returns:
        DCG score.
    """
    total = 0.0
    for i, doc_id in enumerate(doc_ids):
        rel = relevance_map.get(doc_id, 0)
        if rel > 0:
            total += (2**rel - 1) / math.log2(i + 2)
    return total
