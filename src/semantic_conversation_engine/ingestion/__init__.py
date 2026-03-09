"""Ingestion layer for the Semantic Conversation Intelligence Engine.

Handles data ingestion from multiple sources: call recordings, chat transcripts,
email tickets, CRM metadata, and operational systems. Supports batch and
streaming ingestion.
"""

from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.ingestion.inputs import QualitySignals, SpeakerHint, TranscriptInput

__all__ = [
    "QualitySignals",
    "SourceFormat",
    "SpeakerHint",
    "TranscriptInput",
]
