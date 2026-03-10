"""Analytics API router — GET /analytics/summary."""

from __future__ import annotations

from demo.backend.dependencies import get_manifest, get_store
from demo.backend.schemas.api_models import AnalyticsSummary, DomainCount
from demo.backend.services.conversation_store import ConversationStore
from fastapi import APIRouter, Depends

router = APIRouter(tags=["analytics"])


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def get_analytics_summary(
    store: ConversationStore = Depends(get_store),  # noqa: B008
    manifest: dict = Depends(get_manifest),  # noqa: B008
) -> AnalyticsSummary:
    """Get summary analytics across all indexed conversations."""
    domain_counts = store.get_domain_counts()
    domains = [DomainCount(domain=d, count=c) for d, c in sorted(domain_counts.items(), key=lambda x: -x[1])]

    return AnalyticsSummary(
        total_conversations=store.conversation_count,
        total_windows=store.window_count,
        total_embeddings=manifest.get("embeddings", 0),
        avg_asr_confidence=round(store.get_avg_asr_confidence(), 4),
        avg_turns_per_conversation=round(store.get_avg_turns_per_conversation(), 1),
        domains=domains,
        embedding_model=manifest.get("embedding_model", ""),
        index_dimensions=manifest.get("dimensions", 0),
    )
