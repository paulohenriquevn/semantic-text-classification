"""EmbeddingRecord model — a versioned vector representation of text.

An EmbeddingRecord associates a dense vector with a specific object (turn,
context window, or conversation), versioned by model name, model version,
and pooling strategy. Vectors are stored as list[float] for serialization;
conversion to ndarray happens at computation boundaries (see ADR-003).

See PRD §11 for the data model specification.
See ADR-002 for the frozen/strict design decision.
See ADR-003 for the list[float] vs ndarray decision.
"""

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from talkex.models.enums import ObjectType, PoolingStrategy
from talkex.models.types import EmbeddingId


class EmbeddingRecord(BaseModel):
    """A versioned dense vector representation of a text object.

    Args:
        embedding_id: Unique identifier. Format: emb_<uuid4>.
        source_id: ID of the object this embedding represents (turn, window, or conversation).
        source_type: Granularity level of the source object.
        model_name: Name of the embedding model (e.g. 'e5-large-v2', 'bge-base-en').
        model_version: Version string of the embedding model.
        pooling_strategy: Strategy used to aggregate token embeddings.
        dimensions: Dimensionality of the vector. Must equal len(vector).
        vector: Dense embedding vector stored as list[float] (see ADR-003).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    embedding_id: EmbeddingId
    source_id: str
    source_type: ObjectType
    model_name: str
    model_version: str
    pooling_strategy: PoolingStrategy
    dimensions: int
    vector: list[float]

    @field_validator("embedding_id")
    @classmethod
    def embedding_id_must_not_be_empty(cls, v: EmbeddingId) -> EmbeddingId:
        """Reject empty or whitespace-only embedding IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("embedding_id must not be empty or whitespace-only")
        return v

    @field_validator("source_id")
    @classmethod
    def source_id_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only source IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("source_id must not be empty or whitespace-only")
        return v

    @field_validator("model_name")
    @classmethod
    def model_name_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only model names.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("model_name must not be empty or whitespace-only")
        return v

    @field_validator("model_version")
    @classmethod
    def model_version_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only model versions.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("model_version must not be empty or whitespace-only")
        return v

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_be_positive(cls, v: int) -> int:
        """Embedding vectors must have at least one dimension."""
        if v < 1:
            raise ValueError("dimensions must be at least 1")
        return v

    @field_validator("vector")
    @classmethod
    def vector_must_not_be_empty(cls, v: list[float]) -> list[float]:
        """An embedding vector must contain at least one element."""
        if not v:
            raise ValueError("vector must not be empty")
        return v

    @model_validator(mode="after")
    def dimensions_must_equal_vector_length(self) -> "EmbeddingRecord":
        """Ensure dimensions == len(vector) for consistency.

        This invariant catches mismatches between declared dimensionality
        and actual vector length, which would corrupt retrieval and scoring.
        """
        if self.dimensions != len(self.vector):
            raise ValueError(f"dimensions ({self.dimensions}) must equal len(vector) ({len(self.vector)})")
        return self
