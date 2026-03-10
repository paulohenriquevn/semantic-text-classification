"""Embedding generator implementations.

Provides concrete implementations of the EmbeddingGenerator protocol:

    NullEmbeddingGenerator:
        Produces deterministic fake vectors using hash-based seeds.
        No ML dependencies required. Used for testing the full pipeline
        without loading real models.

    SentenceTransformerGenerator:
        Real implementation using sentence-transformers. Conditional
        import — fails gracefully with a clear error if the library
        is not installed.

Both implementations produce EmbeddingRecord objects with complete
provenance (model name, version, pooling strategy, dimensions).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from talkex.embeddings.config import EmbeddingModelConfig
from talkex.embeddings.inputs import EmbeddingBatch
from talkex.embeddings.pooling import apply_pooling, l2_normalize
from talkex.embeddings.preprocessing import (
    PreprocessingConfig,
    prepare_batch_texts,
)
from talkex.models.embedding_record import EmbeddingRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationStats:
    """Operational statistics for a single generate() call.

    Args:
        batch_size: Number of items in the batch.
        total_ms: Wall-clock time for the full generate() call.
        model_name: Model used for generation.
    """

    batch_size: int
    total_ms: float
    model_name: str


@dataclass
class NullEmbeddingGenerator:
    """Deterministic fake embedding generator for testing.

    Produces vectors derived from a hash of the input text, making
    outputs reproducible across runs for the same input. The vector
    dimensionality is configurable (default 384, matching common
    small models).

    This generator satisfies the EmbeddingGenerator protocol without
    any ML dependencies.

    Args:
        model_config: Embedding model configuration.
        preprocessing_config: Text preprocessing configuration.
        dimensions: Dimensionality of generated vectors.
    """

    model_config: EmbeddingModelConfig
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    dimensions: int = 384
    _stats: list[GenerationStats] = field(default_factory=list, repr=False)

    def generate(self, batch: EmbeddingBatch) -> list[EmbeddingRecord]:
        """Generate deterministic fake embeddings for a batch.

        Each vector is derived from the SHA-256 hash of the preprocessed
        text, ensuring reproducibility.

        Args:
            batch: Batch of embedding inputs.

        Returns:
            List of EmbeddingRecords, one per input, in the same order.
        """
        start = time.monotonic()
        texts = prepare_batch_texts(list(batch.items), self.preprocessing_config)
        records: list[EmbeddingRecord] = []

        for inp, text in zip(batch.items, texts, strict=True):
            vector = self._deterministic_vector(text)
            records.append(
                EmbeddingRecord(
                    embedding_id=inp.embedding_id,
                    source_id=inp.object_id,
                    source_type=inp.object_type,
                    model_name=self.model_config.model_name,
                    model_version=self.model_config.model_version,
                    pooling_strategy=self.model_config.pooling_strategy,
                    dimensions=self.dimensions,
                    vector=vector,
                )
            )

        elapsed = (time.monotonic() - start) * 1000
        stats = GenerationStats(
            batch_size=len(batch.items),
            total_ms=round(elapsed, 2),
            model_name=self.model_config.model_name,
        )
        self._stats.append(stats)
        logger.debug(
            "NullEmbeddingGenerator: batch_size=%d, total_ms=%.2f",
            stats.batch_size,
            stats.total_ms,
        )
        return records

    def _deterministic_vector(self, text: str) -> list[float]:
        """Generate a reproducible vector from text via SHA-256 seed.

        Args:
            text: Preprocessed text to hash.

        Returns:
            List of floats with length == self.dimensions.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed))
        raw = rng.standard_normal(self.dimensions).astype(np.float32)
        if self.model_config.normalize_vectors:
            raw = l2_normalize(raw)
        result: list[float] = raw.tolist()
        return result

    @property
    def stats(self) -> list[GenerationStats]:
        """Access generation statistics for observability."""
        return list(self._stats)


@dataclass
class SentenceTransformerGenerator:
    """Real embedding generator using sentence-transformers.

    Loads a HuggingFace model and generates embeddings via batch
    encoding. Applies preprocessing, pooling, and optional L2
    normalization.

    Requires the ``sentence-transformers`` package to be installed.
    Raises ImportError at construction time if unavailable.

    Args:
        model_config: Embedding model configuration.
        preprocessing_config: Text preprocessing configuration.
    """

    model_config: EmbeddingModelConfig
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    _stats: list[GenerationStats] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Load the model at construction time."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerGenerator. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(
            self.model_config.model_name,
            trust_remote_code=False,
        )

    def generate(self, batch: EmbeddingBatch) -> list[EmbeddingRecord]:
        """Generate embeddings using the sentence-transformers model.

        Args:
            batch: Batch of embedding inputs.

        Returns:
            List of EmbeddingRecords, one per input, in the same order.
        """
        start = time.monotonic()
        texts = prepare_batch_texts(list(batch.items), self.preprocessing_config)

        raw_embeddings = self._model.encode(
            texts,
            batch_size=self.model_config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        records: list[EmbeddingRecord] = []
        for inp, token_emb in zip(batch.items, raw_embeddings, strict=True):
            token_emb_2d = np.atleast_2d(token_emb).astype(np.float32)
            pooled = apply_pooling(
                token_emb_2d,
                strategy=self.model_config.pooling_strategy,
                normalize=self.model_config.normalize_vectors,
            )
            records.append(
                EmbeddingRecord(
                    embedding_id=inp.embedding_id,
                    source_id=inp.object_id,
                    source_type=inp.object_type,
                    model_name=self.model_config.model_name,
                    model_version=self.model_config.model_version,
                    pooling_strategy=self.model_config.pooling_strategy,
                    dimensions=len(pooled),
                    vector=pooled.tolist(),
                )
            )

        elapsed = (time.monotonic() - start) * 1000
        stats = GenerationStats(
            batch_size=len(batch.items),
            total_ms=round(elapsed, 2),
            model_name=self.model_config.model_name,
        )
        self._stats.append(stats)
        logger.info(
            "SentenceTransformerGenerator: batch_size=%d, total_ms=%.2f, model=%s",
            stats.batch_size,
            stats.total_ms,
            stats.model_name,
        )
        return records

    @property
    def stats(self) -> list[GenerationStats]:
        """Access generation statistics for observability."""
        return list(self._stats)
