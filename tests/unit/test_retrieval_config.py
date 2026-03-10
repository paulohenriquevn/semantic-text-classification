"""Unit tests for retrieval config objects and data types.

Tests cover: LexicalIndexConfig, VectorIndexConfig, HybridRetrievalConfig,
retrieval enums, RetrievalFilter, RetrievalQuery, RetrievalHit,
RetrievalResult, field validation, strict mode, immutability, and reexport.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from talkex.retrieval.config import (
    DistanceMetric,
    FusionStrategy,
    HybridRetrievalConfig,
    IndexType,
    LexicalIndexConfig,
    VectorIndexConfig,
)
from talkex.retrieval.models import (
    QueryType,
    RetrievalFilter,
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lexical_config(**overrides: Any) -> LexicalIndexConfig:
    defaults: dict[str, Any] = {}
    defaults.update(overrides)
    return LexicalIndexConfig(**defaults)


def _vector_config(**overrides: Any) -> VectorIndexConfig:
    defaults: dict[str, Any] = {"dimensions": 384}
    defaults.update(overrides)
    return VectorIndexConfig(**defaults)


def _hybrid_config(**overrides: Any) -> HybridRetrievalConfig:
    defaults: dict[str, Any] = {}
    defaults.update(overrides)
    return HybridRetrievalConfig(**defaults)


# ---------------------------------------------------------------------------
# LexicalIndexConfig
# ---------------------------------------------------------------------------


class TestLexicalIndexConfig:
    def test_defaults(self) -> None:
        cfg = _lexical_config()
        assert cfg.k1 == 1.5
        assert cfg.b == 0.75
        assert cfg.top_k_default == 25

    def test_custom_values(self) -> None:
        cfg = _lexical_config(k1=2.0, b=0.5, top_k_default=50)
        assert cfg.k1 == 2.0
        assert cfg.b == 0.5
        assert cfg.top_k_default == 50

    def test_rejects_negative_k1(self) -> None:
        with pytest.raises(ValidationError, match="k1"):
            _lexical_config(k1=-0.1)

    def test_rejects_b_below_zero(self) -> None:
        with pytest.raises(ValidationError, match="b"):
            _lexical_config(b=-0.1)

    def test_rejects_b_above_one(self) -> None:
        with pytest.raises(ValidationError, match="b"):
            _lexical_config(b=1.1)

    def test_rejects_zero_top_k(self) -> None:
        with pytest.raises(ValidationError, match="top_k_default"):
            _lexical_config(top_k_default=0)

    def test_frozen(self) -> None:
        cfg = _lexical_config()
        with pytest.raises(ValidationError):
            cfg.k1 = 2.0


# ---------------------------------------------------------------------------
# VectorIndexConfig
# ---------------------------------------------------------------------------


class TestVectorIndexConfig:
    def test_minimal_construction(self) -> None:
        cfg = _vector_config()
        assert cfg.dimensions == 384
        assert cfg.metric == DistanceMetric.COSINE

    def test_defaults(self) -> None:
        cfg = _vector_config()
        assert cfg.index_type == IndexType.FLAT
        assert cfg.train_required is False
        assert cfg.top_k_default == 25

    def test_rejects_zero_dimensions(self) -> None:
        with pytest.raises(ValidationError, match="dimensions"):
            _vector_config(dimensions=0)

    def test_rejects_negative_dimensions(self) -> None:
        with pytest.raises(ValidationError, match="dimensions"):
            _vector_config(dimensions=-1)

    def test_rejects_zero_top_k(self) -> None:
        with pytest.raises(ValidationError, match="top_k_default"):
            _vector_config(top_k_default=0)

    def test_ivf_requires_training(self) -> None:
        with pytest.raises(ValidationError, match="IVF_FLAT"):
            _vector_config(index_type=IndexType.IVF_FLAT, train_required=False)

    def test_ivf_with_training_accepted(self) -> None:
        cfg = _vector_config(index_type=IndexType.IVF_FLAT, train_required=True)
        assert cfg.train_required is True

    def test_hnsw_config(self) -> None:
        cfg = _vector_config(index_type=IndexType.HNSW, metric=DistanceMetric.L2)
        assert cfg.index_type == IndexType.HNSW
        assert cfg.metric == DistanceMetric.L2

    def test_frozen(self) -> None:
        cfg = _vector_config()
        with pytest.raises(ValidationError):
            cfg.dimensions = 768


# ---------------------------------------------------------------------------
# HybridRetrievalConfig
# ---------------------------------------------------------------------------


class TestHybridRetrievalConfig:
    def test_defaults(self) -> None:
        cfg = _hybrid_config()
        assert cfg.lexical_top_k == 25
        assert cfg.vector_top_k == 25
        assert cfg.fusion_strategy == FusionStrategy.RRF
        assert cfg.fusion_weight == 0.5
        assert cfg.rrf_k == 60
        assert cfg.reranker_enabled is False
        assert cfg.filters_enabled is True
        assert cfg.final_top_k == 10

    def test_custom_values(self) -> None:
        cfg = _hybrid_config(
            lexical_top_k=50,
            vector_top_k=50,
            fusion_strategy=FusionStrategy.LINEAR,
            fusion_weight=0.7,
            reranker_enabled=True,
            final_top_k=20,
        )
        assert cfg.fusion_strategy == FusionStrategy.LINEAR
        assert cfg.fusion_weight == 0.7
        assert cfg.reranker_enabled is True

    def test_rejects_zero_lexical_top_k(self) -> None:
        with pytest.raises(ValidationError, match="lexical_top_k"):
            _hybrid_config(lexical_top_k=0)

    def test_rejects_zero_vector_top_k(self) -> None:
        with pytest.raises(ValidationError, match="vector_top_k"):
            _hybrid_config(vector_top_k=0)

    def test_rejects_fusion_weight_below_zero(self) -> None:
        with pytest.raises(ValidationError, match="fusion_weight"):
            _hybrid_config(fusion_weight=-0.1)

    def test_rejects_fusion_weight_above_one(self) -> None:
        with pytest.raises(ValidationError, match="fusion_weight"):
            _hybrid_config(fusion_weight=1.1)

    def test_rejects_zero_rrf_k(self) -> None:
        with pytest.raises(ValidationError, match="rrf_k"):
            _hybrid_config(rrf_k=0)

    def test_rejects_zero_final_top_k(self) -> None:
        with pytest.raises(ValidationError, match="final_top_k"):
            _hybrid_config(final_top_k=0)

    def test_frozen(self) -> None:
        cfg = _hybrid_config()
        with pytest.raises(ValidationError):
            cfg.fusion_strategy = FusionStrategy.LINEAR


# ---------------------------------------------------------------------------
# Retrieval enums
# ---------------------------------------------------------------------------


class TestRetrievalEnums:
    def test_distance_metrics(self) -> None:
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.L2.value == "l2"
        assert DistanceMetric.DOT_PRODUCT.value == "dot_product"

    def test_index_types(self) -> None:
        assert IndexType.FLAT.value == "flat"
        assert IndexType.HNSW.value == "hnsw"
        assert IndexType.IVF_FLAT.value == "ivf_flat"

    def test_fusion_strategies(self) -> None:
        assert FusionStrategy.RRF.value == "rrf"
        assert FusionStrategy.LINEAR.value == "linear"

    def test_query_types(self) -> None:
        assert QueryType.LEXICAL.value == "lexical"
        assert QueryType.SEMANTIC.value == "semantic"
        assert QueryType.HYBRID.value == "hybrid"

    def test_retrieval_modes(self) -> None:
        assert RetrievalMode.HYBRID.value == "hybrid"
        assert RetrievalMode.LEXICAL_ONLY.value == "lexical_only"
        assert RetrievalMode.SEMANTIC_ONLY.value == "semantic_only"


# ---------------------------------------------------------------------------
# Retrieval data types
# ---------------------------------------------------------------------------


class TestRetrievalFilter:
    def test_empty_filter(self) -> None:
        f = RetrievalFilter()
        assert f.channel is None
        assert f.queue is None

    def test_partial_filter(self) -> None:
        f = RetrievalFilter(channel="voice", product="credit_card")
        assert f.channel == "voice"
        assert f.product == "credit_card"
        assert f.queue is None

    def test_frozen(self) -> None:
        f = RetrievalFilter(channel="voice")
        with pytest.raises(AttributeError):
            f.channel = "chat"  # type: ignore[misc]


class TestRetrievalQuery:
    def test_minimal_construction(self) -> None:
        q = RetrievalQuery(query_text="billing issue")
        assert q.query_text == "billing issue"
        assert q.top_k == 10
        assert q.query_type == QueryType.HYBRID

    def test_with_filters(self) -> None:
        f = RetrievalFilter(channel="voice")
        q = RetrievalQuery(query_text="cancel", filters=f)
        assert q.filters is not None
        assert q.filters.channel == "voice"

    def test_frozen(self) -> None:
        q = RetrievalQuery(query_text="test")
        with pytest.raises(AttributeError):
            q.query_text = "modified"  # type: ignore[misc]


class TestRetrievalHit:
    def test_construction(self) -> None:
        hit = RetrievalHit(
            object_id="win_0",
            object_type="context_window",
            score=0.85,
            lexical_score=0.6,
            semantic_score=0.9,
            rank=1,
        )
        assert hit.score == 0.85
        assert hit.lexical_score == 0.6

    def test_defaults(self) -> None:
        hit = RetrievalHit(object_id="win_0", object_type="context_window", score=0.5)
        assert hit.lexical_score is None
        assert hit.semantic_score is None
        assert hit.rank == 1
        assert hit.metadata == {}

    def test_rank_is_one_based(self) -> None:
        hit = RetrievalHit(object_id="w", object_type="t", score=0.9)
        assert hit.rank == 1  # default is 1 (best hit)

    def test_frozen(self) -> None:
        hit = RetrievalHit(object_id="w", object_type="t", score=0.5)
        with pytest.raises(AttributeError):
            hit.score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Score semantics contract
# ---------------------------------------------------------------------------


class TestScoreSemanticsContract:
    """Validates the None vs 0.0 contract for component scores.

    None = component was NOT consulted (absence of information).
    0.0  = component ran and returned zero (real score).
    These must never be conflated.
    """

    def test_none_means_not_consulted(self) -> None:
        hit = RetrievalHit(object_id="w", object_type="t", score=0.5)
        assert hit.lexical_score is None  # BM25 was NOT consulted
        assert hit.semantic_score is None  # ANN was NOT consulted

    def test_zero_means_real_score(self) -> None:
        hit = RetrievalHit(
            object_id="w",
            object_type="t",
            score=0.3,
            lexical_score=0.0,
            semantic_score=0.0,
        )
        assert hit.lexical_score == 0.0  # BM25 ran, returned zero
        assert hit.semantic_score == 0.0  # ANN ran, returned zero
        # Critically: these are NOT the same as None
        assert hit.lexical_score is not None
        assert hit.semantic_score is not None

    def test_lexical_only_hit(self) -> None:
        """Hit from BM25 only — semantic_score must be None."""
        hit = RetrievalHit(
            object_id="w",
            object_type="t",
            score=0.7,
            lexical_score=0.7,
            semantic_score=None,
        )
        assert hit.lexical_score == 0.7
        assert hit.semantic_score is None

    def test_semantic_only_hit(self) -> None:
        """Hit from ANN only — lexical_score must be None."""
        hit = RetrievalHit(
            object_id="w",
            object_type="t",
            score=0.9,
            lexical_score=None,
            semantic_score=0.9,
        )
        assert hit.lexical_score is None
        assert hit.semantic_score == 0.9

    def test_hybrid_hit_both_present(self) -> None:
        """Hit from both BM25 and ANN — both scores present."""
        hit = RetrievalHit(
            object_id="w",
            object_type="t",
            score=0.85,
            lexical_score=0.6,
            semantic_score=0.9,
        )
        assert hit.lexical_score is not None
        assert hit.semantic_score is not None


class TestRetrievalResult:
    def test_construction(self) -> None:
        hits = [RetrievalHit(object_id="w0", object_type="t", score=0.9)]
        result = RetrievalResult(hits=hits, total_candidates=5)
        assert len(result.hits) == 1
        assert result.total_candidates == 5

    def test_defaults(self) -> None:
        result = RetrievalResult(hits=[])
        assert result.total_candidates == 0
        assert result.mode == RetrievalMode.HYBRID
        assert result.stats == {}

    def test_mode_degradation(self) -> None:
        result = RetrievalResult(hits=[], mode=RetrievalMode.LEXICAL_ONLY)
        assert result.mode == RetrievalMode.LEXICAL_ONLY

    def test_frozen(self) -> None:
        result = RetrievalResult(hits=[])
        with pytest.raises(AttributeError):
            result.mode = RetrievalMode.SEMANTIC_ONLY  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestRetrievalReexport:
    def test_configs_importable_from_retrieval_package(self) -> None:
        from talkex.retrieval import (
            HybridRetrievalConfig as HRC,
        )
        from talkex.retrieval import (
            LexicalIndexConfig as LIC,
        )
        from talkex.retrieval import (
            VectorIndexConfig as VIC,
        )

        assert HRC is HybridRetrievalConfig
        assert LIC is LexicalIndexConfig
        assert VIC is VectorIndexConfig

    def test_enums_importable_from_retrieval_package(self) -> None:
        from talkex.retrieval import (
            DistanceMetric as DM,
        )
        from talkex.retrieval import (
            FusionStrategy as FS,
        )
        from talkex.retrieval import (
            IndexType as IT,
        )

        assert DM is DistanceMetric
        assert FS is FusionStrategy
        assert IT is IndexType

    def test_models_importable_from_retrieval_package(self) -> None:
        from talkex.retrieval import (
            RetrievalHit as RH,
        )
        from talkex.retrieval import (
            RetrievalQuery as RQ,
        )
        from talkex.retrieval import (
            RetrievalResult as RR,
        )

        assert RH is RetrievalHit
        assert RQ is RetrievalQuery
        assert RR is RetrievalResult
