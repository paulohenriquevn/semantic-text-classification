"""Configuration for the turn segmentation stage.

SegmentationConfig is a frozen, typed configuration object that controls
how raw text is parsed into Turn objects. All parameters have conservative
defaults suitable for production use.

See ADR-002 for the frozen/strict design decision.
"""

from pydantic import BaseModel, ConfigDict, field_validator


class SegmentationConfig(BaseModel):
    """Configuration for the turn segmentation stage.

    Args:
        normalize_unicode: Apply Unicode NFKC normalization to text.
        collapse_whitespace: Replace multiple whitespace characters with single space.
        strip_lines: Strip leading/trailing whitespace from each line.
        merge_consecutive_same_speaker: Merge adjacent turns from the same speaker.
        min_turn_chars: Minimum character count for a turn after normalization.
            Turns shorter than this are discarded (noise filtering).
        max_turn_chars: Maximum character count for a turn. Turns longer than
            this are preserved but flagged in warnings.
        speaker_label_pattern: Regex pattern for detecting speaker labels in
            labeled transcripts. Default matches 'SPEAKER:' style markers.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    strip_lines: bool = True
    merge_consecutive_same_speaker: bool = True
    min_turn_chars: int = 1
    max_turn_chars: int = 10_000
    speaker_label_pattern: str = r"^(CUSTOMER|AGENT|SYSTEM|UNKNOWN)\s*:"

    @field_validator("min_turn_chars")
    @classmethod
    def min_turn_chars_must_be_positive(cls, v: int) -> int:
        """Minimum turn length must be at least 1 character."""
        if v < 1:
            raise ValueError(f"min_turn_chars must be at least 1, got {v}")
        return v

    @field_validator("max_turn_chars")
    @classmethod
    def max_turn_chars_must_be_positive(cls, v: int) -> int:
        """Maximum turn length must be at least 1 character."""
        if v < 1:
            raise ValueError(f"max_turn_chars must be at least 1, got {v}")
        return v
