"""Pooling strategies for aggregating token embeddings into a single vector.

Embedding models produce per-token vectors (shape: [seq_len, dim]).
Pooling reduces these to a single vector (shape: [dim]) suitable for
indexing, retrieval, and classification.

All functions are pure — they take numpy arrays and return numpy arrays.
No model state, no side effects.

Supported strategies:
    MEAN: Average of all token vectors. Good general-purpose default.
    MAX:  Element-wise maximum across tokens. Captures salient features.

ATTENTION pooling is defined in the PoolingStrategy enum but not implemented
here — it requires model-specific attention weights and will be added when
a concrete use case justifies the complexity (YAGNI).
"""

import numpy as np
from numpy.typing import NDArray

from semantic_conversation_engine.models.enums import PoolingStrategy


def mean_pool(token_embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    """Average all token vectors into a single embedding.

    Args:
        token_embeddings: 2D array of shape (seq_len, dim).
            Must have at least one token.

    Returns:
        1D array of shape (dim,).

    Raises:
        ValueError: If input is empty (zero tokens).
    """
    if token_embeddings.size == 0:
        raise ValueError("Cannot pool empty token embeddings")
    result: NDArray[np.float32] = np.mean(token_embeddings, axis=0).astype(np.float32)
    return result


def max_pool(token_embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    """Element-wise maximum across all token vectors.

    Args:
        token_embeddings: 2D array of shape (seq_len, dim).
            Must have at least one token.

    Returns:
        1D array of shape (dim,).

    Raises:
        ValueError: If input is empty (zero tokens).
    """
    if token_embeddings.size == 0:
        raise ValueError("Cannot pool empty token embeddings")
    result: NDArray[np.float32] = np.max(token_embeddings, axis=0).astype(np.float32)
    return result


def l2_normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalize a vector to unit length.

    Required when using dot-product indexes with cosine similarity
    semantics. A zero vector is returned as-is (no division by zero).

    Args:
        vector: 1D array of shape (dim,).

    Returns:
        Unit-length 1D array of shape (dim,).
    """
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


_POOL_FN = {
    PoolingStrategy.MEAN: mean_pool,
    PoolingStrategy.MAX: max_pool,
}


def apply_pooling(
    token_embeddings: NDArray[np.float32],
    strategy: PoolingStrategy,
    normalize: bool = False,
) -> NDArray[np.float32]:
    """Apply a pooling strategy and optional L2 normalization.

    Dispatches to the correct pooling function based on the strategy
    enum, then optionally normalizes the result.

    Args:
        token_embeddings: 2D array of shape (seq_len, dim).
        strategy: Which pooling strategy to use.
        normalize: Whether to L2-normalize the output vector.

    Returns:
        1D array of shape (dim,).

    Raises:
        ValueError: If strategy is not supported (e.g., ATTENTION).
        ValueError: If input is empty.
    """
    pool_fn = _POOL_FN.get(strategy)
    if pool_fn is None:
        raise ValueError(
            f"Pooling strategy {strategy.value!r} is not supported. Supported: {[s.value for s in _POOL_FN]}"
        )
    pooled = pool_fn(token_embeddings)
    if normalize:
        pooled = l2_normalize(pooled)
    return pooled
