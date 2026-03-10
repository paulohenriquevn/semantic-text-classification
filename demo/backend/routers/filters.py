"""Filters API router — GET /filters."""

from __future__ import annotations

from demo.backend.dependencies import get_store
from demo.backend.schemas.api_models import FiltersResponse
from demo.backend.services.conversation_store import ConversationStore
from fastapi import APIRouter, Depends

router = APIRouter(tags=["filters"])


@router.get("/filters", response_model=FiltersResponse)
def get_filters(
    store: ConversationStore = Depends(get_store),  # noqa: B008
) -> FiltersResponse:
    """Get available filter values for search."""
    return FiltersResponse(
        domains=store.domains,
        topics=store.topics,
        speakers=["customer", "agent"],
    )
