"""Context window construction for multi-turn conversation analysis.

Builds configurable sliding windows of adjacent turns, supporting
variable window sizes, stride, speaker alignment, and recency weighting.
Produces role-aware views (customer-only, agent-only).
"""

from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig

__all__ = [
    "ContextWindowConfig",
    "SlidingWindowBuilder",
]
