"""Multi-level embedding generation for the Semantic Conversation Intelligence Engine.

Generates versioned dense vector representations at turn, context window,
conversation, and role-aware levels. Supports multiple embedding models
(E5, BGE, Instructor) and pooling strategies (mean, attention).
"""

from semantic_conversation_engine.embeddings.cache import (
    EmbeddingCache,
    make_cache_key,
)
from semantic_conversation_engine.embeddings.config import (
    EmbeddingModelConfig,
    EmbeddingRuntimeConfig,
)
from semantic_conversation_engine.embeddings.generator import (
    GenerationStats,
    NullEmbeddingGenerator,
)
from semantic_conversation_engine.embeddings.inputs import (
    EmbeddingBatch,
    EmbeddingInput,
)
from semantic_conversation_engine.embeddings.pooling import (
    apply_pooling,
    l2_normalize,
    max_pool,
    mean_pool,
)
from semantic_conversation_engine.embeddings.preprocessing import (
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
