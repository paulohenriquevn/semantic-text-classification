"""Merge and filter heuristics for parsed turns.

Operates on RawTurn intermediate objects. Merges consecutive same-speaker
turns and filters turns below minimum character threshold. All functions
are pure — no shared state, no side effects.
"""

from semantic_conversation_engine.segmentation.parsing import RawTurn


def merge_consecutive_same_speaker(turns: list[RawTurn]) -> list[RawTurn]:
    """Merge adjacent turns from the same speaker into a single turn.

    Merged turns concatenate text with newline separators and extend
    the offset range to cover all merged segments.

    Args:
        turns: Ordered list of parsed turns.

    Returns:
        New list with consecutive same-speaker turns merged.
        Single-turn input returns a list with that turn unchanged.
        Empty input returns an empty list.
    """
    if not turns:
        return []

    merged: list[RawTurn] = []
    current = turns[0]

    for turn in turns[1:]:
        if turn.speaker == current.speaker:
            current = RawTurn(
                speaker=current.speaker,
                text=current.text + "\n" + turn.text,
                start_offset=current.start_offset,
                end_offset=turn.end_offset,
            )
        else:
            merged.append(current)
            current = turn

    merged.append(current)
    return merged


def filter_short_turns(turns: list[RawTurn], min_chars: int) -> list[RawTurn]:
    """Remove turns whose text is shorter than min_chars.

    Args:
        turns: List of parsed turns.
        min_chars: Minimum character count. Turns with fewer characters
            are discarded.

    Returns:
        Filtered list preserving order. Empty input returns empty list.
    """
    return [t for t in turns if len(t.text) >= min_chars]
