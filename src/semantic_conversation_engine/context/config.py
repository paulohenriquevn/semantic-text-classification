"""Configuration for the context window builder stage.

ContextWindowConfig is a frozen, typed configuration object that controls
how Turn sequences are converted into ContextWindow objects. All parameters
have conservative defaults based on PRD §12.2 recommendations.

See ADR-002 for the frozen/strict design decision.
"""

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ContextWindowConfig(BaseModel):
    """Configuration for the context window builder.

    Args:
        window_size: Number of turns per window. Default 5 (PRD §12.2).
        stride: Step between consecutive windows. Default 2.
        min_window_size: Minimum turns for a valid window. Tail windows
            smaller than this are discarded unless include_partial_tail is True.
        include_partial_tail: Whether to include the last window even if it
            has fewer turns than window_size.
        render_speaker_labels: Include speaker role labels in window_text
            (e.g. '[CUSTOMER] ...').
        render_turn_delimiter: String used to separate turns in window_text.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    window_size: int = 5
    stride: int = 2
    min_window_size: int = 1
    include_partial_tail: bool = True
    render_speaker_labels: bool = True
    render_turn_delimiter: str = "\n"

    @field_validator("window_size")
    @classmethod
    def window_size_must_be_positive(cls, v: int) -> int:
        """Window size must be at least 1 turn."""
        if v < 1:
            raise ValueError(f"window_size must be at least 1, got {v}")
        return v

    @field_validator("stride")
    @classmethod
    def stride_must_be_positive(cls, v: int) -> int:
        """Stride must be at least 1."""
        if v < 1:
            raise ValueError(f"stride must be at least 1, got {v}")
        return v

    @field_validator("min_window_size")
    @classmethod
    def min_window_size_must_be_positive(cls, v: int) -> int:
        """Minimum window size must be at least 1 turn."""
        if v < 1:
            raise ValueError(f"min_window_size must be at least 1, got {v}")
        return v

    @model_validator(mode="after")
    def min_window_size_must_not_exceed_window_size(self) -> "ContextWindowConfig":
        """min_window_size cannot exceed window_size."""
        if self.min_window_size > self.window_size:
            raise ValueError(
                f"min_window_size ({self.min_window_size}) must not exceed window_size ({self.window_size})"
            )
        return self
