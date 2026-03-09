"""Turn parsing from raw transcript text.

Parses raw text into RawTurn intermediate objects based on SourceFormat.
Handles three formats: labeled (SPEAKER: text), multiline (line per turn),
and plain (single block). All functions are pure.

RawTurn is an internal representation — not part of the public API.
It carries the parsed speaker, text content, and character offsets
in the original input text.
"""

import re
from dataclasses import dataclass

from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.models.enums import SpeakerRole
from semantic_conversation_engine.segmentation.config import SegmentationConfig


@dataclass(frozen=True)
class RawTurn:
    """Intermediate turn representation produced by the parser.

    Not part of the public API. Consumed by merging and the segmenter.

    Attributes:
        speaker: Resolved speaker role.
        text: Text content (stripped of speaker labels).
        start_offset: Start position in the source text (inclusive).
        end_offset: End position in the source text (exclusive).
    """

    speaker: SpeakerRole
    text: str
    start_offset: int
    end_offset: int


def parse_transcript(
    text: str,
    source_format: SourceFormat,
    config: SegmentationConfig,
    speaker_map: dict[str, SpeakerRole] | None = None,
) -> list[RawTurn]:
    """Route to the appropriate parser based on source format.

    Args:
        text: The transcript text to parse.
        source_format: How the transcript is structured.
        config: Segmentation configuration (used for speaker_label_pattern).
        speaker_map: Optional mapping from custom labels to SpeakerRole.
            Built from TranscriptInput.speaker_hints by the segmenter.

    Returns:
        List of RawTurn with speaker, text, and character offsets.

    Raises:
        ValueError: If source_format is not a recognized SourceFormat.
    """
    if source_format == SourceFormat.LABELED:
        return _parse_labeled(text, config, speaker_map)
    if source_format == SourceFormat.MULTILINE:
        return _parse_multiline(text)
    if source_format == SourceFormat.PLAIN:
        return _parse_plain(text)
    raise ValueError(f"Unknown source format: {source_format}")


def _parse_labeled(
    text: str,
    config: SegmentationConfig,
    speaker_map: dict[str, SpeakerRole] | None = None,
) -> list[RawTurn]:
    """Parse labeled transcript (SPEAKER: text format).

    Uses speaker_label_pattern from config to detect speaker markers.
    The first capturing group in the pattern must be the speaker label.
    """
    pattern = re.compile(config.speaker_label_pattern, re.MULTILINE)
    turns: list[RawTurn] = []
    matches = list(pattern.finditer(text))

    if not matches:
        # No speaker labels found — treat entire text as single UNKNOWN turn
        stripped = text.strip()
        if stripped:
            start = len(text) - len(text.lstrip())
            end = len(text.rstrip())
            turns.append(
                RawTurn(
                    speaker=SpeakerRole.UNKNOWN,
                    text=stripped,
                    start_offset=start,
                    end_offset=end,
                )
            )
        return turns

    for i, match in enumerate(matches):
        # Turn text runs from after the label to the start of the next label
        turn_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        label = match.group(1)
        speaker = _resolve_speaker(label, speaker_map)

        # Text starts after the matched label pattern (e.g., after "CUSTOMER: ")
        text_start = match.end()
        turn_text = text[text_start:turn_end].strip()

        if turn_text:
            turns.append(
                RawTurn(
                    speaker=speaker,
                    text=turn_text,
                    start_offset=match.start(),
                    end_offset=turn_end,
                )
            )

    return turns


def _parse_multiline(text: str) -> list[RawTurn]:
    """Parse multiline transcript (each non-empty line is a separate turn).

    Speaker is always UNKNOWN since multiline format has no speaker markers.
    """
    turns: list[RawTurn] = []
    offset = 0

    for line in text.split("\n"):
        line_end = offset + len(line)
        stripped = line.strip()

        if stripped:
            content_start = offset + len(line) - len(line.lstrip())
            content_end = line_end - (len(line) - len(line.rstrip()))
            turns.append(
                RawTurn(
                    speaker=SpeakerRole.UNKNOWN,
                    text=stripped,
                    start_offset=content_start,
                    end_offset=content_end,
                )
            )

        offset = line_end + 1  # +1 for the newline character

    return turns


def _parse_plain(text: str) -> list[RawTurn]:
    """Parse plain text as a single turn with UNKNOWN speaker."""
    stripped = text.strip()
    if not stripped:
        return []

    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())

    return [
        RawTurn(
            speaker=SpeakerRole.UNKNOWN,
            text=stripped,
            start_offset=start,
            end_offset=end,
        )
    ]


def _resolve_speaker(
    label: str,
    speaker_map: dict[str, SpeakerRole] | None,
) -> SpeakerRole:
    """Map a speaker label string to a SpeakerRole enum.

    Resolution order:
    1. Check speaker_map (from TranscriptInput.speaker_hints)
    2. Direct match against SpeakerRole enum values
    3. Fall back to UNKNOWN
    """
    normalized_label = label.strip().upper()

    if speaker_map:
        for key, role in speaker_map.items():
            if key.strip().upper() == normalized_label:
                return role

    for role in SpeakerRole:
        if role.value.upper() == normalized_label:
            return role

    return SpeakerRole.UNKNOWN
