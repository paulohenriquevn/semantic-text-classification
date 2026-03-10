"""Configuration for the embedding generation stage.

EmbeddingModelConfig defines model identity and generation parameters.
EmbeddingRuntimeConfig controls batch processing behavior.

Both are frozen, typed configuration objects with conservative defaults.
See ADR-002 for the frozen/strict design decision.
"""

from pydantic import BaseModel, ConfigDict, field_validator

from talkex.models.enums import PoolingStrategy


class EmbeddingModelConfig(BaseModel):
    """Configuration for the embedding model.

    Identifies the model and controls how vectors are generated.
    Each config uniquely determines the embedding space — changing
    any field produces incompatible vectors.

    Args:
        model_name: HuggingFace model name or path
            (e.g. 'intfloat/e5-base-v2', 'BAAI/bge-small-en-v1.5').
        model_version: Semantic version of the model weights.
        pooling_strategy: Token aggregation strategy.
        normalize_vectors: Whether to L2-normalize output vectors.
            Required for cosine similarity with dot-product indexes.
        max_length: Maximum token length. Texts longer than this
            are truncated by the tokenizer.
        batch_size: Number of texts per forward pass.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    model_name: str
    model_version: str
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    normalize_vectors: bool = True
    max_length: int = 512
    batch_size: int = 32

    @field_validator("model_name")
    @classmethod
    def model_name_must_not_be_empty(cls, v: str) -> str:
        """Validates only — does NOT normalize."""
        if not v.strip():
            raise ValueError("model_name must not be empty or whitespace-only")
        return v

    @field_validator("model_version")
    @classmethod
    def model_version_must_not_be_empty(cls, v: str) -> str:
        """Validates only — does NOT normalize."""
        if not v.strip():
            raise ValueError("model_version must not be empty or whitespace-only")
        return v

    @field_validator("max_length")
    @classmethod
    def max_length_must_be_positive(cls, v: int) -> int:
        """Token limit must be at least 1."""
        if v < 1:
            raise ValueError(f"max_length must be at least 1, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def batch_size_must_be_positive(cls, v: int) -> int:
        """Batch size must be at least 1."""
        if v < 1:
            raise ValueError(f"batch_size must be at least 1, got {v}")
        return v


class EmbeddingRuntimeConfig(BaseModel):
    """Runtime configuration for embedding generation.

    Controls operational behavior separate from model identity.
    Changing runtime config does NOT change the embedding space.

    Args:
        enable_cache: Whether to cache embeddings by
            (object_id, model_name, model_version, pooling_strategy).
        max_retries: Maximum retries on transient generation failures.
        timeout_seconds: Timeout per batch in seconds.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    enable_cache: bool = True
    max_retries: int = 3
    timeout_seconds: float = 30.0

    @field_validator("max_retries")
    @classmethod
    def max_retries_must_be_non_negative(cls, v: int) -> int:
        """Retries must be 0 or more."""
        if v < 0:
            raise ValueError(f"max_retries must be non-negative, got {v}")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def timeout_must_be_positive(cls, v: float) -> float:
        """Timeout must be positive."""
        if v <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {v}")
        return v
