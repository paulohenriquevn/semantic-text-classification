"""Hybrid retrieval combining lexical (BM25) and semantic (ANN vector) search.

Pipeline: BM25 top-K + ANN top-K -> union -> score fusion / reciprocal rank
fusion -> optional cross-encoder rerank -> business filters.
"""

from semantic_conversation_engine.retrieval.bm25 import InMemoryBM25Index
from semantic_conversation_engine.retrieval.builders import (
    context_window_to_lexical_doc,
    context_windows_to_lexical_docs,
    embedding_record_to_hit_metadata,
)
from semantic_conversation_engine.retrieval.config import (
    DistanceMetric,
    FusionStrategy,
    HybridRetrievalConfig,
    IndexType,
    LexicalIndexConfig,
    VectorIndexConfig,
)
from semantic_conversation_engine.retrieval.fusion import (
    linear_fusion,
    reciprocal_rank_fusion,
)
from semantic_conversation_engine.retrieval.hybrid import SimpleHybridRetriever
from semantic_conversation_engine.retrieval.models import (
    QueryType,
    RetrievalFilter,
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
)
from semantic_conversation_engine.retrieval.vector_index import InMemoryVectorIndex

__all__ = [
    "DistanceMetric",
    "FusionStrategy",
    "HybridRetrievalConfig",
    "InMemoryBM25Index",
    "InMemoryVectorIndex",
    "IndexType",
    "LexicalIndexConfig",
    "QueryType",
    "RetrievalFilter",
    "RetrievalHit",
    "RetrievalMode",
    "RetrievalQuery",
    "RetrievalResult",
    "SimpleHybridRetriever",
    "VectorIndexConfig",
    "context_window_to_lexical_doc",
    "context_windows_to_lexical_docs",
    "embedding_record_to_hit_metadata",
    "linear_fusion",
    "reciprocal_rank_fusion",
]
