"""Search service — orchestrates hybrid retrieval for the demo API.

Bridges the FastAPI layer with the TalkEx hybrid retrieval engine.
Handles query embedding, BM25 + vector search, score fusion, and
result enrichment with conversation metadata and sentence highlighting.

Performance: sentence highlighting uses a single batch embedding call
for all top-K results, making it O(1) model calls regardless of K.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from demo.backend.schemas.api_models import SearchFilters, SearchHit, SearchResponse
from demo.backend.services.conversation_store import ConversationStore
from demo.backend.services.text_highlighting import find_best_sentences_batch

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

        # Collect hits and their texts for batch highlighting
        pre_hits: list[tuple[dict, float | None, float | None, float, str]] = []
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

            pre_hits.append(
                (
                    window_meta,
                    hit.lexical_score,
                    hit.semantic_score,
                    hit.score,
                    hit.object_id,
                )
            )

            if len(pre_hits) >= top_k:
                break

        # Batch highlight: embed query once, embed all sentences in one call
        texts = [meta.get("window_text", "") for meta, *_ in pre_hits]
        matched_texts: list[str | None] = [None] * len(texts)

        gen = self.retriever.embedding_generator
        if gen is not None and texts:
            # Reuse the retriever's embedding generator to embed the query
            # (the retriever already embedded it for ANN search, but that
            # vector isn't exposed — this is the only extra embedding call)
            query_vector = self.retriever._embed_query(query_text)
            matched_texts = find_best_sentences_batch(texts, query_vector, gen)

        # Build final hits
        hits: list[SearchHit] = []
        for i, (meta, lex_score, sem_score, score, obj_id) in enumerate(pre_hits):
            hits.append(
                SearchHit(
                    window_id=obj_id,
                    conversation_id=meta.get("conversation_id", ""),
                    text=texts[i],
                    lexical_score=lex_score,
                    semantic_score=sem_score,
                    score=score,
                    rank=i + 1,
                    domain=meta.get("domain", ""),
                    topic=meta.get("topic", ""),
                    matched_text=matched_texts[i],
                )
            )

        elapsed_ms = (time.monotonic() - start) * 1000

        return SearchResponse(
            results=hits,
            total_candidates=result.total_candidates,
            query=query_text,
            latency_ms=round(elapsed_ms, 2),
        )
