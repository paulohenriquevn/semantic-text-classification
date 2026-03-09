"""Light text normalization for the segmentation stage.

Applies configurable normalization steps to text. All functions are
pure — no shared state, no side effects.

IMPORTANT: Normalization is applied to Turn.normalized_text only.
Turn.raw_text always preserves the original text exactly as parsed
from the input, ensuring auditability and debuggability.

Steps are applied in a fixed order:
1. Unicode NFKC normalization (canonical decomposition + compatibility composition)
2. Per-line stripping (leading/trailing whitespace per line)
3. Horizontal whitespace collapsing (preserves newlines)
"""

import re
import unicodedata

from semantic_conversation_engine.segmentation.config import SegmentationConfig

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")


def normalize_text(text: str, config: SegmentationConfig) -> str:
    """Apply light normalization to text based on config flags.

    Args:
        text: The text to normalize.
        config: Segmentation configuration controlling which steps apply.

    Returns:
        The normalized text. Returns the input unchanged if all flags are False.
    """
    result = text

    if config.normalize_unicode:
        result = unicodedata.normalize("NFKC", result)

    if config.strip_lines:
        lines = result.split("\n")
        result = "\n".join(line.strip() for line in lines)

    if config.collapse_whitespace:
        result = _HORIZONTAL_WHITESPACE.sub(" ", result)

    return result
