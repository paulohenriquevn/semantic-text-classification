"""Turn segmentation and text normalization.

Produces Turn objects from raw conversation text with speaker attribution,
character offsets, and timestamps. Applies configurable normalization:
Unicode, spacing, optional lowercasing.
"""

from talkex.segmentation.config import SegmentationConfig
from talkex.segmentation.segmenter import TurnSegmenter

__all__ = [
    "SegmentationConfig",
    "TurnSegmenter",
]
