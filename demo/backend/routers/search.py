"""Search API router — POST /search."""

from __future__ import annotations

from demo.backend.dependencies import get_search_service
from demo.backend.schemas.api_models import SearchRequest, SearchResponse
from demo.backend.services.search_service import SearchService
from fastapi import APIRouter, Depends

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),  # noqa: B008
) -> SearchResponse:
    """Hybrid search across indexed conversations.

    Combines BM25 lexical search with vector semantic search,
    applies score fusion, and returns ranked results enriched
    with conversation metadata.
    """
    return service.search(
        request.query,
        filters=request.filters,
        top_k=request.top_k,
    )
