"""Context window construction for multi-turn conversation analysis.

Builds configurable sliding windows of adjacent turns, supporting
variable window sizes, stride, speaker alignment, and recency weighting.
Produces role-aware views (customer-only, agent-only).
"""

from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig

__all__ = [
    "ContextWindowConfig",
    "SlidingWindowBuilder",
]
