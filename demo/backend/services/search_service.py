"""Search service — orchestrates hybrid retrieval for the demo API.

Bridges the FastAPI layer with the TalkEx hybrid retrieval engine.
Handles query embedding, BM25 + vector search, score fusion, and
result enrichment with conversation metadata.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from demo.backend.schemas.api_models import SearchFilters, SearchHit, SearchResponse
from demo.backend.services.conversation_store import ConversationStore

from talkex.retrieval.hybrid import SimpleHybridRetriever
from talkex.retrieval.models import QueryType, RetrievalQuery

logger = logging.getLogger(__name__)


@dataclass
class SearchService:
    """Hybrid search service backed by TalkEx retrieval engine.

    Args:
        retriever: Configured SimpleHybridRetriever with both indexes.
        store: ConversationStore for metadata enrichment.
    """

    retriever: SimpleHybridRetriever
    store: ConversationStore

    def search(
        self,
        query_text: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 20,
    ) -> SearchResponse:
        """Execute hybrid search and return enriched results.

        Args:
            query_text: Natural language search query.
            filters: Optional filters (speaker, domain, topic).
            top_k: Maximum results to return.

        Returns:
            SearchResponse with ranked, enriched hits and latency.
        """
        start = time.monotonic()

        retrieval_query = RetrievalQuery(
            query_text=query_text,
            top_k=top_k,
            query_type=QueryType.HYBRID,
        )

        result = self.retriever.retrieve(retrieval_query)

        # Enrich hits with metadata from store
        hits: list[SearchHit] = []
        for hit in result.hits:
            window_meta = self.store.get_window(hit.object_id)
            if window_meta is None:
                continue

            # Apply filters
            if filters:
                if filters.domain and window_meta.get("domain") != filters.domain:
                    continue
                if filters.topic and window_meta.get("topic") != filters.topic:
                    continue

            hits.append(
                SearchHit(
                    window_id=hit.object_id,
                    conversation_id=window_meta.get("conversation_id", ""),
                    text=window_meta.get("window_text", ""),
                    lexical_score=hit.lexical_score,
                    semantic_score=hit.semantic_score,
                    score=hit.score,
                    rank=len(hits) + 1,
                    domain=window_meta.get("domain", ""),
                    topic=window_meta.get("topic", ""),
                )
            )

            if len(hits) >= top_k:
                break

        elapsed_ms = (time.monotonic() - start) * 1000

        return SearchResponse(
            results=hits,
            total_candidates=result.total_candidates,
            query=query_text,
            latency_ms=round(elapsed_ms, 2),
        )
