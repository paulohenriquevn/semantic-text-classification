"""Cheap lexical feature extraction for turns.

Extracts simple, deterministic features from normalized turn text.
All features are O(n) in text length — no regex, no external models.
These features enrich Turn.metadata for downstream classification
and rule evaluation.
"""

from typing import Any


def extract_lexical_features(text: str) -> dict[str, Any]:
    """Extract cheap lexical features from turn text.

    All features are deterministic and O(n) in text length.

    Args:
        text: The normalized turn text.

    Returns:
        Dictionary of feature name → value. Keys:
        - char_count (int): Number of characters.
        - word_count (int): Number of whitespace-separated tokens.
        - has_question (bool): Whether the text contains '?'.
        - line_count (int): Number of lines (newline-separated).
        - avg_word_length (float): Mean word length, 0.0 if no words.
    """
    words = text.split()
    word_count = len(words)

    return {
        "char_count": len(text),
        "word_count": word_count,
        "has_question": "?" in text,
        "line_count": text.count("\n") + 1,
        "avg_word_length": (sum(len(w) for w in words) / word_count if word_count > 0 else 0.0),
    }
