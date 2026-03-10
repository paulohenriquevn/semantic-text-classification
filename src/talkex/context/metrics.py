"""Operational metrics for context windows.

Computes cheap, deterministic metrics from window turns and rendered text.
These metrics enrich ContextWindow.metadata for downstream analysis,
filtering, and debugging.

All functions are pure — no shared state, no side effects.

Per-window metadata namespace convention (stable contract):
    metadata["role_views"]["customer_text"]  — customer-only rendered text
    metadata["role_views"]["agent_text"]     — agent-only rendered text
    metadata["speakers"]["distribution"]     — {role: turn_count} mapping
    metadata["speakers"]["has_customer"]     — bool
    metadata["speakers"]["has_agent"]        — bool
    metadata["speakers"]["customer_turn_count"] — int
    metadata["speakers"]["agent_turn_count"]    — int
    metadata["total_chars"]                  — window_text character count
    metadata["total_words"]                  — window_text word count

Build-level coverage metrics (returned by compute_build_coverage):
    total_windows         — number of windows generated
    total_turns           — total turns in input
    unique_turns_covered  — turns appearing in at least one window
    orphan_turn_count     — turns not covered by any window
    coverage_ratio        — unique_turns_covered / total_turns
    multi_window_turns    — turns appearing in more than one window
"""

from typing import Any

from talkex.models.context_window import ContextWindow
from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn


def compute_window_metrics(
    turns: tuple[Turn, ...],
    window_text: str,
    customer_text: str,
    agent_text: str,
) -> dict[str, Any]:
    """Compute operational metrics for a context window.

    All metrics are O(n) in text/turn count — cheap and deterministic.

    Args:
        turns: Ordered turns in the window.
        window_text: The rendered full window text.
        customer_text: Rendered customer-only text.
        agent_text: Rendered agent-only text.

    Returns:
        Namespaced dictionary. See module docstring for the stable
        key convention.
    """
    speaker_counts: dict[str, int] = {}
    customer_count = 0
    agent_count = 0

    for turn in turns:
        role_key = str(turn.speaker.value)
        speaker_counts[role_key] = speaker_counts.get(role_key, 0) + 1
        if turn.speaker == SpeakerRole.CUSTOMER:
            customer_count += 1
        elif turn.speaker == SpeakerRole.AGENT:
            agent_count += 1

    return {
        "total_chars": len(window_text),
        "total_words": len(window_text.split()),
        "role_views": {
            "customer_text": customer_text,
            "agent_text": agent_text,
        },
        "speakers": {
            "distribution": speaker_counts,
            "has_customer": customer_count > 0,
            "has_agent": agent_count > 0,
            "customer_turn_count": customer_count,
            "agent_turn_count": agent_count,
        },
    }


def compute_build_coverage(
    total_turns: int,
    windows: list[ContextWindow],
) -> dict[str, Any]:
    """Compute build-level coverage metrics across all windows.

    Answers: were all turns covered? How many appear in multiple
    windows? Are there orphan turns?

    Args:
        total_turns: Number of turns in the input list.
        windows: The windows produced by the builder.

    Returns:
        Dictionary with coverage metrics. See module docstring
        for the stable key convention.
    """
    if total_turns == 0:
        return {
            "total_windows": 0,
            "total_turns": 0,
            "unique_turns_covered": 0,
            "orphan_turn_count": 0,
            "coverage_ratio": 0.0,
            "multi_window_turns": 0,
        }

    # Count how many windows each turn index appears in
    turn_window_count: dict[int, int] = {}
    for window in windows:
        for idx in range(window.start_index, window.end_index + 1):
            turn_window_count[idx] = turn_window_count.get(idx, 0) + 1

    unique_covered = len(turn_window_count)
    orphans = total_turns - unique_covered
    multi_window = sum(1 for count in turn_window_count.values() if count > 1)

    return {
        "total_windows": len(windows),
        "total_turns": total_turns,
        "unique_turns_covered": unique_covered,
        "orphan_turn_count": orphans,
        "coverage_ratio": unique_covered / total_turns,
        "multi_window_turns": multi_window,
    }
