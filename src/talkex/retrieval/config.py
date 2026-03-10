"""Configuration for lexical index, vector index, and hybrid retrieval.

Three config objects cover the retrieval stack:
- LexicalIndexConfig: BM25 indexing and search parameters.
- VectorIndexConfig: ANN index construction and search parameters.
- HybridRetrievalConfig: Score fusion, reranking, and filter behavior.

All are frozen, typed configuration objects with conservative defaults.
See ADR-002 for the frozen/strict design decision.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class DistanceMetric(StrEnum):
    """Distance metric for vector similarity search."""

    COSINE = "cosine"
    L2 = "l2"
    DOT_PRODUCT = "dot_product"


class IndexType(StrEnum):
    """Vector index algorithm.

    FLAT for exact search (small datasets, benchmarking).
    HNSW for approximate nearest neighbors (production).
    IVF_FLAT for partitioned exact search (medium datasets).
    """

    FLAT = "flat"
    HNSW = "hnsw"
    IVF_FLAT = "ivf_flat"


class FusionStrategy(StrEnum):
    """Score fusion strategy for hybrid retrieval.

    RRF (Reciprocal Rank Fusion) is robust when lexical and semantic
    scores have different scales. LINEAR uses weighted combination
    of normalized scores.
    """

    RRF = "rrf"
    LINEAR = "linear"


class LexicalIndexConfig(BaseModel):
    """Configuration for the BM25 lexical index.

    Args:
        k1: BM25 term frequency saturation parameter.
            Higher values increase the impact of term frequency.
        b: BM25 document length normalization parameter.
            0.0 = no normalization, 1.0 = full normalization.
        top_k_default: Default number of results to return.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    k1: float = 1.5
    b: float = 0.75
    top_k_default: int = 25

    @field_validator("k1")
    @classmethod
    def k1_must_be_non_negative(cls, v: float) -> float:
        """BM25 k1 must be non-negative."""
        if v < 0:
            raise ValueError(f"k1 must be non-negative, got {v}")
        return v

    @field_validator("b")
    @classmethod
    def b_must_be_in_unit_range(cls, v: float) -> float:
        """BM25 b must be in [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"b must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("top_k_default")
    @classmethod
    def top_k_must_be_positive(cls, v: int) -> int:
        """Top-k must be at least 1."""
        if v < 1:
            raise ValueError(f"top_k_default must be at least 1, got {v}")
        return v


class VectorIndexConfig(BaseModel):
    """Configuration for the ANN vector index.

    Args:
        metric: Distance metric for similarity search.
        index_type: Index algorithm. FLAT for exact, HNSW for approximate.
        dimensions: Expected dimensionality of vectors. Must match the
            embedding model output.
        train_required: Whether the index needs a training step before
            insertion (true for IVF variants).
        top_k_default: Default number of results to return.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.FLAT
    dimensions: int
    train_required: bool = False
    top_k_default: int = 25

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_be_positive(cls, v: int) -> int:
        """Vector dimensions must be at least 1."""
        if v < 1:
            raise ValueError(f"dimensions must be at least 1, got {v}")
        return v

    @field_validator("top_k_default")
    @classmethod
    def top_k_must_be_positive(cls, v: int) -> int:
        """Top-k must be at least 1."""
        if v < 1:
            raise ValueError(f"top_k_default must be at least 1, got {v}")
        return v

    @model_validator(mode="after")
    def ivf_requires_training(self) -> "VectorIndexConfig":
        """IVF indexes require a training step."""
        if self.index_type == IndexType.IVF_FLAT and not self.train_required:
            raise ValueError("IVF_FLAT index_type requires train_required=True")
        return self


class HybridRetrievalConfig(BaseModel):
    """Configuration for hybrid retrieval (lexical + semantic).

    Controls how BM25 and ANN results are combined, whether reranking
    is enabled, and whether structural filters are applied.

    Args:
        lexical_top_k: Number of candidates from BM25.
        vector_top_k: Number of candidates from ANN.
        fusion_strategy: How to combine lexical and semantic scores.
        fusion_weight: Weight for the semantic score in LINEAR fusion.
            Lexical weight is (1 - fusion_weight). Ignored for RRF.
        rrf_k: RRF smoothing constant. Higher values reduce the
            impact of high ranks. Ignored for LINEAR fusion.
        reranker_enabled: Whether to apply cross-encoder reranking
            after fusion.
        filters_enabled: Whether to apply structural filters
            (channel, queue, product, region, date range).
        final_top_k: Maximum results after fusion and filtering.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    lexical_top_k: int = 25
    vector_top_k: int = 25
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    fusion_weight: float = 0.5
    rrf_k: int = 60
    reranker_enabled: bool = False
    filters_enabled: bool = True
    final_top_k: int = 10

    @field_validator("lexical_top_k")
    @classmethod
    def lexical_top_k_must_be_positive(cls, v: int) -> int:
        """Lexical candidate count must be at least 1."""
        if v < 1:
            raise ValueError(f"lexical_top_k must be at least 1, got {v}")
        return v

    @field_validator("vector_top_k")
    @classmethod
    def vector_top_k_must_be_positive(cls, v: int) -> int:
        """Vector candidate count must be at least 1."""
        if v < 1:
            raise ValueError(f"vector_top_k must be at least 1, got {v}")
        return v

    @field_validator("fusion_weight")
    @classmethod
    def fusion_weight_must_be_in_unit_range(cls, v: float) -> float:
        """Fusion weight must be in [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"fusion_weight must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("rrf_k")
    @classmethod
    def rrf_k_must_be_positive(cls, v: int) -> int:
        """RRF smoothing constant must be at least 1."""
        if v < 1:
            raise ValueError(f"rrf_k must be at least 1, got {v}")
        return v

    @field_validator("final_top_k")
    @classmethod
    def final_top_k_must_be_positive(cls, v: int) -> int:
        """Final result count must be at least 1."""
        if v < 1:
            raise ValueError(f"final_top_k must be at least 1, got {v}")
        return v
