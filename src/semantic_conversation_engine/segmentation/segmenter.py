"""TurnSegmenter — orchestrator for the segmentation pipeline.

Implements the Segmenter protocol from pipeline.protocols.
Composes pure functions from normalization, parsing, merging, and
features modules into a fully deterministic pipeline:

    parse → remap speakers → merge → normalize → filter → extract features → build Turns

Turn IDs are deterministic: ``{conversation_id}_turn_{index}`` where
index is the zero-based position in the final output list. Same input
and config always produce byte-identical output — essential for audit
trails, debugging, and regression testing.
"""

from semantic_conversation_engine.ingestion.inputs import SpeakerHint, TranscriptInput
from semantic_conversation_engine.models.enums import SpeakerRole
from semantic_conversation_engine.models.turn import Turn
from semantic_conversation_engine.models.types import TurnId
from semantic_conversation_engine.segmentation.config import SegmentationConfig
from semantic_conversation_engine.segmentation.features import extract_lexical_features
from semantic_conversation_engine.segmentation.merging import (
    merge_consecutive_same_speaker,
)
from semantic_conversation_engine.segmentation.normalization import normalize_text
from semantic_conversation_engine.segmentation.parsing import parse_transcript


class TurnSegmenter:
    """Segments raw transcript text into Turn domain objects.

    Implements the Segmenter protocol. Orchestrates four pure-function
    modules: parsing, merging, normalization, and feature extraction.

    The pipeline is fully deterministic: same input + same config produces
    byte-identical output, including turn IDs. Turn IDs follow the
    pattern ``{conversation_id}_turn_{index}`` (zero-based).
    """

    def segment(
        self,
        transcript: TranscriptInput,
        config: SegmentationConfig,
    ) -> list[Turn]:
        """Segment a transcript into Turn domain objects.

        Args:
            transcript: The raw transcript input (boundary object).
            config: Segmentation configuration controlling parsing,
                normalization, merging, and filtering behavior.

        Returns:
            Ordered list of Turn objects with speaker attribution,
            character offsets, normalized text, and lexical features.
            Empty list if no valid turns are found after filtering.
        """
        speaker_map = _build_speaker_map(transcript.speaker_hints)

        # 1. Parse raw text into intermediate RawTurns
        raw_turns = parse_transcript(
            text=transcript.raw_text,
            source_format=transcript.source_format,
            config=config,
            speaker_map=speaker_map,
        )

        # 2. Merge consecutive same-speaker turns (if enabled)
        if config.merge_consecutive_same_speaker:
            raw_turns = merge_consecutive_same_speaker(raw_turns)

        # 3. Normalize, filter, extract features, and build Turns
        turns: list[Turn] = []
        turn_index = 0
        for rt in raw_turns:
            normalized = normalize_text(rt.text, config)

            # Filter by min_turn_chars on normalized text
            if len(normalized.strip()) < config.min_turn_chars:
                continue

            features = extract_lexical_features(normalized)

            turn = Turn(
                turn_id=TurnId(f"{transcript.conversation_id}_turn_{turn_index}"),
                conversation_id=transcript.conversation_id,
                speaker=rt.speaker,
                raw_text=rt.text,
                start_offset=rt.start_offset,
                end_offset=rt.end_offset,
                normalized_text=normalized,
                metadata=features,
            )
            turns.append(turn)
            turn_index += 1

        return turns


def _build_speaker_map(
    hints: list[SpeakerHint],
) -> dict[str, SpeakerRole] | None:
    """Build a label→role mapping from speaker hints.

    Args:
        hints: Speaker hints from TranscriptInput.

    Returns:
        Mapping dict if hints are provided, None otherwise.
        Unrecognized role strings fall back to UNKNOWN.
    """
    if not hints:
        return None

    mapping: dict[str, SpeakerRole] = {}
    for hint in hints:
        role_upper = hint.role.strip().upper()
        matched_role = SpeakerRole.UNKNOWN
        for sr in SpeakerRole:
            if sr.value.upper() == role_upper:
                matched_role = sr
                break
        mapping[hint.speaker_label] = matched_role

    return mapping
