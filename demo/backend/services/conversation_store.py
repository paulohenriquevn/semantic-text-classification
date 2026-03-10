"""In-memory store for conversations and windows.

Loaded once at startup from the index build output. Provides O(1)
lookups by conversation_id and window_id for the API layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationStore:
    """In-memory store for conversation and window metadata.

    Loaded from the JSON files produced by build_index.py.
    """

    _conversations: dict[str, dict] = field(default_factory=dict, repr=False)
    _windows: dict[str, dict] = field(default_factory=dict, repr=False)
    _window_to_conv: dict[str, str] = field(default_factory=dict, repr=False)
    _domains: set[str] = field(default_factory=set, repr=False)
    _topics: set[str] = field(default_factory=set, repr=False)

    def load(self, index_dir: str | Path) -> None:
        """Load conversation and window metadata from index directory.

        Args:
            index_dir: Path to the index directory (output of build_index.py).

        Raises:
            FileNotFoundError: If metadata files don't exist.
        """
        index_path = Path(index_dir)

        conversations_path = index_path / "conversations.json"
        with open(conversations_path) as f:
            conversations_list = json.load(f)

        for conv in conversations_list:
            conv_id = conv["conversation_id"]
            self._conversations[conv_id] = conv
            self._domains.add(conv.get("domain", "unknown"))
            self._topics.add(conv.get("topic", "unknown"))

        windows_path = index_path / "windows.json"
        with open(windows_path) as f:
            windows_list = json.load(f)

        for win in windows_list:
            win_id = win["window_id"]
            self._windows[win_id] = win
            self._window_to_conv[win_id] = win["conversation_id"]

        logger.info(
            "ConversationStore loaded: %d conversations, %d windows, %d domains",
            len(self._conversations),
            len(self._windows),
            len(self._domains),
        )

    def get_conversation(self, conversation_id: str) -> dict | None:
        """Lookup conversation by ID."""
        return self._conversations.get(conversation_id)

    def get_window(self, window_id: str) -> dict | None:
        """Lookup window by ID."""
        return self._windows.get(window_id)

    def get_conversation_for_window(self, window_id: str) -> dict | None:
        """Get the conversation that contains a given window."""
        conv_id = self._window_to_conv.get(window_id)
        if conv_id is None:
            return None
        return self._conversations.get(conv_id)

    @property
    def domains(self) -> list[str]:
        return sorted(self._domains)

    @property
    def topics(self) -> list[str]:
        return sorted(self._topics)

    @property
    def conversation_count(self) -> int:
        return len(self._conversations)

    @property
    def window_count(self) -> int:
        return len(self._windows)

    def get_domain_counts(self) -> dict[str, int]:
        """Count conversations per domain."""
        counts: dict[str, int] = {}
        for conv in self._conversations.values():
            domain = conv.get("domain", "unknown")
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    def get_avg_asr_confidence(self) -> float:
        """Average ASR confidence across all conversations."""
        if not self._conversations:
            return 0.0
        total = sum(c.get("asr_confidence", 0.0) for c in self._conversations.values())
        return total / len(self._conversations)

    def iter_windows(self) -> list[tuple[str, dict]]:
        """Iterate over all windows as (window_id, metadata) pairs."""
        return list(self._windows.items())

    def get_avg_turns_per_conversation(self) -> float:
        """Average turns per conversation."""
        if not self._conversations:
            return 0.0
        total = sum(c.get("turn_count", 0) for c in self._conversations.values())
        return total / len(self._conversations)
