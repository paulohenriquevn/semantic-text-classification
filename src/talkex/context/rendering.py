"""Window text rendering and role-aware view extraction.

Renders concatenated text for context windows from their constituent
turns. Supports configurable speaker labels and delimiters. Produces
role-aware views (customer-only, agent-only text) for downstream
classification and rule evaluation.

All functions are pure — no shared state, no side effects.
Uses normalized_text when available, falls back to raw_text.
"""

from talkex.context.config import ContextWindowConfig
from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn


def _turn_text(turn: Turn) -> str:
    """Extract the best available text from a turn.

    Prefers normalized_text (post-normalization), falls back to raw_text.
    """
    return turn.normalized_text if turn.normalized_text is not None else turn.raw_text


def render_window_text(
    turns: tuple[Turn, ...],
    config: ContextWindowConfig,
) -> str:
    """Render the full window_text from an ordered sequence of turns.

    Each turn is rendered as either ``[SPEAKER] text`` (when
    render_speaker_labels is True) or plain ``text``. Turns are
    joined with render_turn_delimiter.

    Args:
        turns: Ordered turns in the window. Must not be empty.
        config: Window configuration controlling labels and delimiter.

    Returns:
        Concatenated window text. Deterministic for the same input.
    """
    parts: list[str] = []
    for turn in turns:
        text = _turn_text(turn)
        if config.render_speaker_labels:
            label = turn.speaker.value.upper()
            parts.append(f"[{label}] {text}")
        else:
            parts.append(text)

    return config.render_turn_delimiter.join(parts)


def extract_role_text(
    turns: tuple[Turn, ...],
    role: SpeakerRole,
    delimiter: str,
) -> str:
    """Extract concatenated text for a single speaker role.

    Filters turns to only those matching the given role, then
    joins their text with the delimiter. Speaker labels are NOT
    included — they would be redundant in a single-role view.

    Args:
        turns: Ordered turns in the window.
        role: The speaker role to extract (e.g. CUSTOMER, AGENT).
        delimiter: String used to separate turns.

    Returns:
        Concatenated text from turns matching the role.
        Empty string if no turns match.
    """
    role_texts = [_turn_text(t) for t in turns if t.speaker == role]
    return delimiter.join(role_texts)
