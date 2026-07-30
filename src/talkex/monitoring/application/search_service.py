"""M5 hybrid-search application service (blueprint D1).

Orchestrates the online QA search: embed the query, fetch lexical + semantic candidates from the
`TurnSearchPort`, fuse them with the shipped `reciprocal_rank_fusion` (DRY — no re-implementation),
and return evidence-carrying `SearchHit`s. Fusion lives here (application), candidate SQL lives in
the infrastructure adapter (DIP) — so this service is unit-testable with a fake port + embedder.
"""

from __future__ import annotations

from datetime import datetime

from talkex.monitoring.domain.ports import TurnEmbedder, TurnSearchPort
from talkex.monitoring.domain.search import SearchHit, SearchQuery
from talkex.retrieval.fusion import reciprocal_rank_fusion
from talkex.retrieval.models import RetrievalHit


class SearchService:
    """Hybrid (BM25-adjacent ⊕ ANN) search over the retention window."""

    def __init__(self, search_port: TurnSearchPort, embedder: TurnEmbedder) -> None:
        self._port = search_port
        self._embedder = embedder

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        vector = self._embedder.embed(query.query_text)
        lexical = await self._port.lexical_candidates(query.query_text, query.top_k, query.window_days, query.criteria)
        semantic = await self._port.semantic_candidates(vector, query.top_k, query.window_days, query.criteria)
        fused = reciprocal_rank_fusion(lexical, semantic)
        return [_to_search_hit(hit) for hit in fused[: query.top_k]]


def _to_search_hit(hit: RetrievalHit) -> SearchHit:
    meta = hit.metadata
    created_at = meta["created_at"]
    return SearchHit(
        turn_id=hit.object_id,
        conversation_id=str(meta["conversation_id"]),
        raw_text=str(meta["raw_text"]),
        created_at=created_at if isinstance(created_at, datetime) else datetime.fromisoformat(str(created_at)),
        score=hit.score,
        lexical_score=hit.lexical_score,
        semantic_score=hit.semantic_score,
    )
