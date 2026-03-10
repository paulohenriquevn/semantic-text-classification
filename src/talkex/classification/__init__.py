"""Supervised multi-label, multi-level classification.

Operates at turn, context window, and conversation levels. Combines semantic,
lexical, structural, and contextual features. Every prediction carries label,
score, confidence, threshold, evidence, and model version.
"""

from talkex.classification.config import (
    ClassificationLevel,
    ClassificationMode,
    ClassifierConfig,
)
from talkex.classification.features import (
    FeatureSet,
    extract_lexical_features,
    extract_structural_features,
    merge_feature_sets,
)
from talkex.classification.labels import LabelSpace
from talkex.classification.logistic import (
    LogisticRegressionClassifier,
)
from talkex.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from talkex.classification.orchestrator import (
    ClassificationBatchResult,
    ClassificationOrchestrator,
)
from talkex.classification.serialization import (
    load_similarity_classifier,
    save_similarity_classifier,
)
from talkex.classification.similarity import (
    EmbeddingSimilarityClassifier,
)

__all__ = [
    "ClassificationBatchResult",
    "ClassificationInput",
    "ClassificationLevel",
    "ClassificationMode",
    "ClassificationOrchestrator",
    "ClassificationResult",
    "ClassifierConfig",
    "EmbeddingSimilarityClassifier",
    "FeatureSet",
    "LabelScore",
    "LabelSpace",
    "LogisticRegressionClassifier",
    "extract_lexical_features",
    "extract_structural_features",
    "load_similarity_classifier",
    "merge_feature_sets",
    "save_similarity_classifier",
]
