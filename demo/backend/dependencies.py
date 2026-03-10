"""Dependency injection container for the demo backend.

Wires TalkEx engine components together and provides FastAPI
dependency functions for the router layer.
"""

from __future__ import annotations

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path

from demo.backend.config import DemoConfig
from demo.backend.services.category_service import CategoryService
from demo.backend.services.conversation_store import ConversationStore
from demo.backend.services.search_service import SearchService

from talkex.embeddings.config import EmbeddingModelConfig
from talkex.embeddings.generator import NullEmbeddingGenerator
from talkex.models.enums import PoolingStrategy
from talkex.retrieval.config import DistanceMetric, HybridRetrievalConfig, VectorIndexConfig
from talkex.retrieval.hybrid import SimpleHybridRetriever
from talkex.retrieval.qdrant import QdrantVectorIndex

logger = logging.getLogger(__name__)


@lru_cache
def get_config() -> DemoConfig:
    """Get demo configuration (singleton)."""
    return DemoConfig()


@lru_cache
def get_store() -> ConversationStore:
    """Get conversation store (singleton, loaded once)."""
    config = get_config()
    store = ConversationStore()
    store.load(config.index_dir)
    return store


@lru_cache
def get_manifest() -> dict:
    """Get the build manifest (singleton)."""
    config = get_config()
    manifest_path = Path(config.index_dir) / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


@lru_cache
def _get_qdrant_index() -> QdrantVectorIndex:
    """Get shared Qdrant vector index (singleton).

    Qdrant local mode uses an exclusive file lock, so only one client
    instance can access the storage directory at a time.
    """
    config = get_config()
    manifest = get_manifest()
    index_path = Path(config.index_dir)

    dims = manifest.get("dimensions", 384)
    vec_config = VectorIndexConfig(dimensions=dims, metric=DistanceMetric.COSINE)
    qdrant_index = QdrantVectorIndex(
        config=vec_config,
        collection_name="talkex_demo",
        path=str(index_path / "qdrant_storage"),
    )
    logger.info("Qdrant index loaded from %s", index_path / "qdrant_storage")
    return qdrant_index


@lru_cache
def _get_embedding_generator() -> NullEmbeddingGenerator:
    """Get shared embedding generator (singleton)."""
    manifest = get_manifest()
    dims = manifest.get("dimensions", 384)
    model_name = manifest.get("embedding_model", "null")

    if model_name == "null":
        emb_config = EmbeddingModelConfig(
            model_name="null",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        )
        return NullEmbeddingGenerator(model_config=emb_config, dimensions=dims)

    try:
        from talkex.embeddings.generator import SentenceTransformerGenerator

        emb_config = EmbeddingModelConfig(
            model_name=model_name,
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        )
        return SentenceTransformerGenerator(model_config=emb_config)
    except ImportError:
        logger.warning("sentence-transformers not available, using NullEmbeddingGenerator")
        emb_config = EmbeddingModelConfig(
            model_name="null",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        )
        return NullEmbeddingGenerator(model_config=emb_config, dimensions=dims)


@lru_cache
def get_search_service() -> SearchService:
    """Get search service (singleton, wired with retriever + store)."""
    config = get_config()
    index_path = Path(config.index_dir)

    # Load BM25 index
    bm25_path = index_path / "bm25_index.pkl"
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)
    logger.info("BM25 index loaded from %s", bm25_path)

    # Wire retriever with shared Qdrant + embedding generator
    retriever = SimpleHybridRetriever(
        lexical_index=bm25_index,
        vector_index=_get_qdrant_index(),
        embedding_generator=_get_embedding_generator(),
        config=HybridRetrievalConfig(),
    )

    return SearchService(retriever=retriever, store=get_store())


@lru_cache
def get_category_service() -> CategoryService:
    """Get category service (singleton, wired with shared embedding infrastructure)."""
    config = get_config()
    persist_path = Path(config.index_dir) / "categories.json"

    return CategoryService(
        store=get_store(),
        persist_path=persist_path,
        embedding_generator=_get_embedding_generator(),
        vector_index=_get_qdrant_index(),
    )
