"""Unit tests for pooling strategies.

Tests cover: mean_pool, max_pool, l2_normalize, apply_pooling dispatch,
edge cases (empty input, zero vector), and normalization.
"""

import numpy as np
import pytest

from talkex.embeddings.pooling import (
    apply_pooling,
    l2_normalize,
    max_pool,
    mean_pool,
)
from talkex.models.enums import PoolingStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokens(rows: list[list[float]]) -> np.ndarray:
    """Build a (seq_len, dim) float32 array from nested lists."""
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# mean_pool
# ---------------------------------------------------------------------------


class TestMeanPool:
    def test_single_token(self) -> None:
        result = mean_pool(_tokens([[1.0, 2.0, 3.0]]))
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_two_tokens(self) -> None:
        result = mean_pool(_tokens([[2.0, 4.0], [6.0, 8.0]]))
        np.testing.assert_array_almost_equal(result, [4.0, 6.0])

    def test_output_is_1d(self) -> None:
        result = mean_pool(_tokens([[1.0, 2.0], [3.0, 4.0]]))
        assert result.ndim == 1

    def test_output_dtype_is_float32(self) -> None:
        result = mean_pool(_tokens([[1.0, 2.0]]))
        assert result.dtype == np.float32

    def test_rejects_empty_input(self) -> None:
        empty = np.array([], dtype=np.float32).reshape(0, 3)
        with pytest.raises(ValueError, match="empty"):
            mean_pool(empty)


# ---------------------------------------------------------------------------
# max_pool
# ---------------------------------------------------------------------------


class TestMaxPool:
    def test_single_token(self) -> None:
        result = max_pool(_tokens([[1.0, 2.0, 3.0]]))
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_two_tokens_takes_max(self) -> None:
        result = max_pool(_tokens([[1.0, 8.0], [5.0, 2.0]]))
        np.testing.assert_array_almost_equal(result, [5.0, 8.0])

    def test_with_negatives(self) -> None:
        result = max_pool(_tokens([[-1.0, -2.0], [-3.0, -0.5]]))
        np.testing.assert_array_almost_equal(result, [-1.0, -0.5])

    def test_output_is_1d(self) -> None:
        result = max_pool(_tokens([[1.0, 2.0]]))
        assert result.ndim == 1

    def test_rejects_empty_input(self) -> None:
        empty = np.array([], dtype=np.float32).reshape(0, 3)
        with pytest.raises(ValueError, match="empty"):
            max_pool(empty)


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_unit_vector_unchanged(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = l2_normalize(v)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_normalizes_to_unit_length(self) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = l2_normalize(v)
        np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)

    def test_preserves_direction(self) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = l2_normalize(v)
        np.testing.assert_array_almost_equal(result, [0.6, 0.8])

    def test_zero_vector_returns_zero(self) -> None:
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = l2_normalize(v)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_output_dtype_is_float32(self) -> None:
        v = np.array([1.0, 2.0], dtype=np.float32)
        result = l2_normalize(v)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# apply_pooling dispatch
# ---------------------------------------------------------------------------


class TestApplyPooling:
    def test_dispatches_mean(self) -> None:
        tokens = _tokens([[2.0, 4.0], [6.0, 8.0]])
        result = apply_pooling(tokens, PoolingStrategy.MEAN)
        np.testing.assert_array_almost_equal(result, [4.0, 6.0])

    def test_dispatches_max(self) -> None:
        tokens = _tokens([[1.0, 8.0], [5.0, 2.0]])
        result = apply_pooling(tokens, PoolingStrategy.MAX)
        np.testing.assert_array_almost_equal(result, [5.0, 8.0])

    def test_with_normalization(self) -> None:
        tokens = _tokens([[3.0, 4.0]])
        result = apply_pooling(tokens, PoolingStrategy.MEAN, normalize=True)
        np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)

    def test_without_normalization(self) -> None:
        tokens = _tokens([[3.0, 4.0]])
        result = apply_pooling(tokens, PoolingStrategy.MEAN, normalize=False)
        np.testing.assert_array_almost_equal(result, [3.0, 4.0])

    def test_rejects_unsupported_strategy(self) -> None:
        tokens = _tokens([[1.0, 2.0]])
        with pytest.raises(ValueError, match="not supported"):
            apply_pooling(tokens, PoolingStrategy.ATTENTION)


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestPoolingReexport:
    def test_importable_from_embeddings_package(self) -> None:
        from talkex.embeddings import (
            apply_pooling as ap,
        )
        from talkex.embeddings import (
            l2_normalize as ln,
        )
        from talkex.embeddings import (
            max_pool as mxp,
        )
        from talkex.embeddings import (
            mean_pool as mp,
        )

        assert ap is apply_pooling
        assert ln is l2_normalize
        assert mxp is max_pool
        assert mp is mean_pool
