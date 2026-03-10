"""Hybrid retrieval combining lexical (BM25) and semantic (ANN vector) search.

Pipeline: BM25 top-K + ANN top-K -> union -> score fusion / reciprocal rank
fusion -> optional cross-encoder rerank -> business filters.
"""

from talkex.retrieval.bm25 import InMemoryBM25Index
from talkex.retrieval.builders import (
    context_window_to_lexical_doc,
    context_windows_to_lexical_docs,
    embedding_record_to_hit_metadata,
)
from talkex.retrieval.config import (
    DistanceMetric,
    FusionStrategy,
    HybridRetrievalConfig,
    IndexType,
    LexicalIndexConfig,
    VectorIndexConfig,
)
from talkex.retrieval.fusion import (
    linear_fusion,
    reciprocal_rank_fusion,
)
from talkex.retrieval.hybrid import SimpleHybridRetriever
from talkex.retrieval.models import (
    QueryType,
    RetrievalFilter,
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
)
from talkex.retrieval.qdrant import QdrantVectorIndex
from talkex.retrieval.vector_index import InMemoryVectorIndex

__all__ = [
    "DistanceMetric",
    "FusionStrategy",
    "HybridRetrievalConfig",
    "InMemoryBM25Index",
    "InMemoryVectorIndex",
    "IndexType",
    "LexicalIndexConfig",
    "QdrantVectorIndex",
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
