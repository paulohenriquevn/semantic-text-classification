"""Ingestion-specific enums for the Semantic Conversation Intelligence Engine.

These enums define the vocabulary for ingestion boundary objects,
separate from the core domain enums in models.enums.
"""

from enum import StrEnum


class SourceFormat(StrEnum):
    """Format of the source transcript data.

    Determines how the segmenter will parse speaker markers,
    timestamps, and turn boundaries.
    """

    LABELED = "labeled"
    """Turns are explicitly labeled with speaker markers (e.g. 'CUSTOMER:', 'AGENT:')."""

    MULTILINE = "multiline"
    """Each line or paragraph is a separate utterance, speaker inferred from context."""

    PLAIN = "plain"
    """Unstructured text without reliable speaker markers or formatting."""
