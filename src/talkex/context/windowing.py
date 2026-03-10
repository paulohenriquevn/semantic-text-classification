"""Sliding window generation over turn sequences.

Generates window slices from an ordered list of turns using configurable
window size and stride. Handles partial tail windows and minimum size
constraints. All functions are pure — no shared state, no side effects.

WindowSlice is an internal intermediate representation, not part of the
public API. It is consumed by the SlidingWindowBuilder orchestrator.
"""

from dataclasses import dataclass

from talkex.context.config import ContextWindowConfig
from talkex.models.turn import Turn


@dataclass(frozen=True)
class WindowSlice:
    """Internal specification for a window before rendering.

    Not part of the public API. Consumed by the builder.

    Attributes:
        start_index: Index of first turn in the conversation (0-based).
        end_index: Index of last turn in the conversation (0-based, inclusive).
        turns: The Turn objects in this window (ordered).
    """

    start_index: int
    end_index: int
    turns: tuple[Turn, ...]


def generate_window_slices(
    turns: list[Turn],
    config: ContextWindowConfig,
) -> list[WindowSlice]:
    """Generate sliding window slices from an ordered list of turns.

    Windows are generated at positions 0, stride, 2*stride, etc.
    Each window contains up to window_size consecutive turns.

    Partial windows (fewer than window_size turns) occur at the end
    of the sequence. They are included only if include_partial_tail
    is True and the window has at least min_window_size turns.

    Args:
        turns: Ordered list of turns to window over. Empty list
            produces empty result.
        config: Window configuration controlling size, stride, and
            tail behavior.

    Returns:
        Ordered list of WindowSlice objects. Empty if no valid
        windows can be generated.
    """
    if not turns:
        return []

    slices: list[WindowSlice] = []
    n = len(turns)
    start = 0

    while start < n:
        end = min(start + config.window_size, n)
        window_turns = turns[start:end]
        actual_size = len(window_turns)

        if actual_size < config.window_size:
            # Partial window — apply tail policy
            if not config.include_partial_tail:
                break
            if actual_size < config.min_window_size:
                break

        slices.append(
            WindowSlice(
                start_index=start,
                end_index=start + actual_size - 1,
                turns=tuple(window_turns),
            )
        )

        start += config.stride

    return slices
