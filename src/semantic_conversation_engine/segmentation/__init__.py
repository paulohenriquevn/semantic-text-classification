"""Turn segmentation and text normalization.

Produces Turn objects from raw conversation text with speaker attribution,
character offsets, and timestamps. Applies configurable normalization:
Unicode, spacing, optional lowercasing.
"""

from semantic_conversation_engine.segmentation.config import SegmentationConfig
from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

__all__ = [
    "SegmentationConfig",
    "TurnSegmenter",
]
