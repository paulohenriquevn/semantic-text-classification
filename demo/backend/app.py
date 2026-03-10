"""TalkEx Demo — FastAPI application.

Conversation intelligence search engine demo with hybrid retrieval.

Usage:
    uvicorn demo.backend.app:app --reload
    # or
    python -m demo.backend.app
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from demo.backend.dependencies import get_config, get_manifest, get_store
from demo.backend.routers import analytics, categories, conversations, filters, search
from demo.backend.schemas.api_models import HealthResponse
from demo.backend.services.conversation_store import ConversationStore
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from talkex import __version__

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load indexes and stores on server startup."""
    store = get_store()
    manifest = get_manifest()

    logger.info("─" * 60)
    logger.info("TalkEx Demo server started")
    logger.info("  Conversations: %d", store.conversation_count)
    logger.info("  Windows:       %d", store.window_count)
    logger.info("  Embeddings:    %d", manifest.get("embeddings", 0))
    logger.info("  Model:         %s", manifest.get("embedding_model", "n/a"))
    logger.info("  Dimensions:    %d", manifest.get("dimensions", 0))
    logger.info("─" * 60)
    yield


app = FastAPI(
    title="TalkEx Demo",
    description="Conversation Intelligence Search Engine — hybrid BM25 + vector search over call center transcripts",
    version=__version__,
    lifespan=lifespan,
)

# CORS must be configured before the app starts (at module level, not in startup)
_config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(search.router)
app.include_router(conversations.router)
app.include_router(filters.router)
app.include_router(analytics.router)
app.include_router(categories.router)


@app.get("/health", response_model=HealthResponse)
def health(
    store: ConversationStore = Depends(get_store),  # noqa: B008
    manifest: dict = Depends(get_manifest),  # noqa: B008
) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        conversations=store.conversation_count,
        windows=store.window_count,
        embeddings=manifest.get("embeddings", 0),
        version=__version__,
    )


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run("demo.backend.app:app", host=config.host, port=config.port, reload=True)
