"""Supervised multi-label, multi-level classification.

Operates at turn, context window, and conversation levels. Combines semantic,
lexical, structural, and contextual features. Every prediction carries label,
score, confidence, threshold, evidence, and model version.
"""

from semantic_conversation_engine.classification.config import (
    ClassificationLevel,
    ClassificationMode,
    ClassifierConfig,
)
from semantic_conversation_engine.classification.features import (
    FeatureSet,
    extract_lexical_features,
    extract_structural_features,
    merge_feature_sets,
)
from semantic_conversation_engine.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)

__all__ = [
    "ClassificationInput",
    "ClassificationLevel",
    "ClassificationMode",
    "ClassificationResult",
    "ClassifierConfig",
    "FeatureSet",
    "LabelScore",
    "extract_lexical_features",
    "extract_structural_features",
    "merge_feature_sets",
]
