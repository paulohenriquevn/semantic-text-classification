"""Online sentiment detection — efficient traditional ML (blueprint M2 D1/D3).

Design decision (evidence-backed): 3-class sentiment (pos/neg/neu) caps at macro-F1 ≈ 0.68 across
lexical, multilingual, chunk-pooled, and OpenAI-embedding methods — a label ceiling on the ambiguous
`neutral` class (F1 0.57), not a model limit. For the monitoring context (alert on dissatisfied
customers) the meaningful, achievable target is NEGATIVE detection: TF-IDF (word + char) + LinearSVC,
macro-F1 ≈ 0.86 (>> 0.70), CPU-instant, no embedding download, no API.

`SentimentDetector` wraps a scikit-learn pipeline behind a small domain interface: `train`,
`predict` (label + margin score), `evaluate` (macro-F1). The vectorizer/classifier are swappable
(Strategy) via the constructor for experimentation without touching callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

NEGATIVE = "negative"
NON_NEGATIVE = "non_negative"

_SPEAKER_MARKER = re.compile(r"\[(customer|agent)\]")


def normalize_text(text: str) -> str:
    """Strip speaker markers so features focus on the utterance content."""
    return _SPEAKER_MARKER.sub(" ", text)


def to_binary_label(sentiment: str) -> str:
    """Map a 3-class sentiment label to the binary negative-detection target."""
    return NEGATIVE if sentiment == NEGATIVE else NON_NEGATIVE


@dataclass(frozen=True)
class SentimentPrediction:
    """A sentiment prediction: the label plus a signed margin (higher = more negative)."""

    label: str
    score: float


def _default_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=50000
                ),
            ),
        ]
    )
    return Pipeline([("features", features), ("clf", LinearSVC(C=1.0, class_weight="balanced"))])


class SentimentDetector:
    """Binary negative-sentiment detector over lexical features (traditional ML)."""

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self._pipeline = pipeline if pipeline is not None else _default_pipeline()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def train(self, texts: list[str], sentiments: list[str]) -> None:
        """Fit on raw texts + 3-class sentiment labels (mapped to binary internally)."""
        if not texts or len(texts) != len(sentiments):
            raise ValueError("texts and sentiments must be non-empty and equal length")
        x = [normalize_text(t) for t in texts]
        y = [to_binary_label(s) for s in sentiments]
        self._pipeline.fit(x, y)
        self._fitted = True

    def predict(self, text: str) -> SentimentPrediction:
        """Predict the binary sentiment label + a margin score for one text."""
        if not self._fitted:
            raise RuntimeError("SentimentDetector must be trained before predict()")
        x = [normalize_text(text)]
        label = str(self._pipeline.predict(x)[0])
        clf: Any = self._pipeline.named_steps["clf"]
        margin = float(self._pipeline.decision_function(x)[0]) if hasattr(clf, "decision_function") else 0.0
        return SentimentPrediction(label=label, score=margin)

    def evaluate(self, texts: list[str], sentiments: list[str]) -> float:
        """Return the macro-F1 of the binary negative-detection task on held-out data."""
        if not self._fitted:
            raise RuntimeError("SentimentDetector must be trained before evaluate()")
        x = [normalize_text(t) for t in texts]
        y_true = [to_binary_label(s) for s in sentiments]
        y_pred = list(self._pipeline.predict(x))
        return float(f1_score(y_true, y_pred, average="macro"))
