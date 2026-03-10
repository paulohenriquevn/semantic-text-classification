"""Conversation API router — GET /conversation/{id}."""

from __future__ import annotations

from demo.backend.dependencies import get_store
from demo.backend.schemas.api_models import ConversationResponse, TurnResponse
from demo.backend.services.conversation_store import ConversationStore
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(tags=["conversations"])


@router.get("/conversation/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_store),  # noqa: B008
) -> ConversationResponse:
    """Get full conversation with turns and metadata."""
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    turns = [
        TurnResponse(
            turn_id=t["turn_id"],
            speaker=t.get("speaker", "unknown"),
            raw_text=t.get("raw_text", ""),
            normalized_text=t.get("normalized_text"),
        )
        for t in conv.get("turns", [])
    ]

    return ConversationResponse(
        conversation_id=conv["conversation_id"],
        domain=conv.get("domain", "unknown"),
        topic=conv.get("topic", "unknown"),
        asr_confidence=conv.get("asr_confidence", 0.0),
        audio_duration_seconds=conv.get("audio_duration_seconds", 0),
        turn_count=conv.get("turn_count", 0),
        window_count=conv.get("window_count", 0),
        text_preview=conv.get("text", "")[:500],
        turns=turns,
    )
