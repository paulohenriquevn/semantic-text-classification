"""Feature extraction for classification inputs.

Pure functions that extract classification features from text and
metadata. Features are organized into families:

    lexical:     word count, char count, question marks, exclamation marks
    structural:  speaker role, turn position, window size
    embedding:   pre-computed embedding vector (passed through, not extracted here)

Feature extraction is deterministic and stateless.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSet:
    """A named collection of extracted features.

    Args:
        features: Mapping from feature name to float value.
        feature_names: Ordered list of feature names for vector construction.
    """

    features: dict[str, float]
    feature_names: list[str]

    def to_vector(self) -> list[float]:
        """Convert features to an ordered float vector.

        Returns:
            List of feature values in feature_names order.
        """
        return [self.features.get(name, 0.0) for name in self.feature_names]


_LEXICAL_FEATURE_NAMES = [
    "char_count",
    "word_count",
    "avg_word_length",
    "question_count",
    "exclamation_count",
    "uppercase_ratio",
    "digit_ratio",
]


def extract_lexical_features(text: str) -> FeatureSet:
    """Extract lexical features from text.

    Args:
        text: Input text to analyze.

    Returns:
        FeatureSet with lexical features.
    """
    chars = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    question_count = float(text.count("?"))
    exclamation_count = float(text.count("!"))
    alpha_chars = sum(1 for c in text if c.isalpha())
    uppercase_ratio = (
        sum(1 for c in text if c.isupper()) / alpha_chars if alpha_chars > 0 else 0.0
    )
    digit_ratio = sum(1 for c in text if c.isdigit()) / chars if chars > 0 else 0.0

    features = {
        "char_count": float(chars),
        "word_count": float(word_count),
        "avg_word_length": avg_word_len,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
    }
    return FeatureSet(features=features, feature_names=list(_LEXICAL_FEATURE_NAMES))


_STRUCTURAL_FEATURE_NAMES = [
    "is_customer",
    "is_agent",
    "turn_count",
    "speaker_count",
]


def extract_structural_features(
    *,
    is_customer: bool = False,
    is_agent: bool = False,
    turn_count: int = 1,
    speaker_count: int = 1,
) -> FeatureSet:
    """Extract structural features from conversation metadata.

    Args:
        is_customer: Whether the speaker is a customer.
        is_agent: Whether the speaker is an agent.
        turn_count: Number of turns in the context window.
        speaker_count: Number of distinct speakers.

    Returns:
        FeatureSet with structural features.
    """
    features = {
        "is_customer": 1.0 if is_customer else 0.0,
        "is_agent": 1.0 if is_agent else 0.0,
        "turn_count": float(turn_count),
        "speaker_count": float(speaker_count),
    }
    return FeatureSet(
        features=features, feature_names=list(_STRUCTURAL_FEATURE_NAMES)
    )


def merge_feature_sets(*feature_sets: FeatureSet) -> FeatureSet:
    """Merge multiple FeatureSets into one.

    Feature names are concatenated in order. If the same feature name
    appears in multiple sets, the last value wins.

    Args:
        feature_sets: FeatureSets to merge.

    Returns:
        Merged FeatureSet with all features combined.
    """
    merged_features: dict[str, float] = {}
    merged_names: list[str] = []
    seen: set[str] = set()

    for fs in feature_sets:
        for name in fs.feature_names:
            if name not in seen:
                merged_names.append(name)
                seen.add(name)
            merged_features[name] = fs.features.get(name, 0.0)

    return FeatureSet(features=merged_features, feature_names=merged_names)
