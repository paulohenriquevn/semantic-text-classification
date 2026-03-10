"""Multi-level embedding generation for the TalkEx — Conversation Intelligence Engine.

Generates versioned dense vector representations at turn, context window,
conversation, and role-aware levels. Supports multiple embedding models
(E5, BGE, Instructor) and pooling strategies (mean, attention).
"""

from talkex.embeddings.cache import (
    EmbeddingCache,
    make_cache_key,
)
from talkex.embeddings.config import (
    EmbeddingModelConfig,
    EmbeddingRuntimeConfig,
)
from talkex.embeddings.generator import (
    GenerationStats,
    NullEmbeddingGenerator,
)
from talkex.embeddings.inputs import (
    EmbeddingBatch,
    EmbeddingInput,
)
from talkex.embeddings.pooling import (
    apply_pooling,
    l2_normalize,
    max_pool,
    mean_pool,
)
from talkex.embeddings.preprocessing import (
    PreprocessingConfig,
    prepare_batch_texts,
    prepare_embedding_text,
)

__all__ = [
    "EmbeddingBatch",
    "EmbeddingCache",
    "EmbeddingInput",
    "EmbeddingModelConfig",
    "EmbeddingRuntimeConfig",
    "GenerationStats",
    "NullEmbeddingGenerator",
    "PreprocessingConfig",
    "apply_pooling",
    "l2_normalize",
    "make_cache_key",
    "max_pool",
    "mean_pool",
    "prepare_batch_texts",
    "prepare_embedding_text",
]
