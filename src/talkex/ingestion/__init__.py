"""Ingestion layer for the TalkEx — Conversation Intelligence Engine.

Handles data ingestion from multiple sources: call recordings, chat transcripts,
email tickets, CRM metadata, and operational systems. Supports batch and
streaming ingestion.
"""

from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import QualitySignals, SpeakerHint, TranscriptInput

__all__ = [
    "QualitySignals",
    "SourceFormat",
    "SpeakerHint",
    "TranscriptInput",
]
