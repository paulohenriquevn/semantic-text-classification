"""Unit + DoD-eval tests for SentimentDetector (M2)."""

import json
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from talkex.classification.sentiment import (
    NEGATIVE,
    NON_NEGATIVE,
    SentimentDetector,
    normalize_text,
    to_binary_label,
)


def _tiny_pipeline() -> Pipeline:
    # min_df=1 so a small fixture has a usable vocabulary.
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1)), ("clf", LinearSVC(C=1.0))])


class TestHelpers:
    def test_to_binary_label(self) -> None:
        assert to_binary_label("negative") == NEGATIVE
        assert to_binary_label("neutral") == NON_NEGATIVE
        assert to_binary_label("positive") == NON_NEGATIVE

    def test_normalize_strips_speaker_markers(self) -> None:
        assert "[customer]" not in normalize_text("[customer] quero cancelar")


class TestSentimentDetector:
    def _fixture(self) -> tuple[list[str], list[str]]:
        neg = ["péssimo atendimento horrível", "quero cancelar reclamação problema", "muito ruim insatisfeito"]
        pos = ["excelente atendimento ótimo", "adorei muito bom obrigado", "perfeito recomendo elogio"]
        texts = neg + pos
        sents = ["negative"] * 3 + ["positive"] * 3
        return texts, sents

    def test_predict_before_train_raises(self) -> None:
        with pytest.raises(RuntimeError):
            SentimentDetector(_tiny_pipeline()).predict("x")

    def test_train_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            SentimentDetector(_tiny_pipeline()).train(["a"], ["negative", "positive"])

    def test_train_then_predict_negative(self) -> None:
        det = SentimentDetector(_tiny_pipeline())
        texts, sents = self._fixture()
        det.train(texts, sents)
        assert det.is_fitted
        pred = det.predict("[customer] péssimo horrível quero cancelar")
        assert pred.label == NEGATIVE

    def test_train_then_predict_non_negative(self) -> None:
        det = SentimentDetector(_tiny_pipeline())
        texts, sents = self._fixture()
        det.train(texts, sents)
        pred = det.predict("excelente ótimo adorei recomendo")
        assert pred.label == NON_NEGATIVE


_DATA = Path("experiments/data")


class TestDoD:
    """The M2 acceptance criterion: macro-F1 >= 0.70 beating the majority baseline, on real data."""

    def _load(self, split: str) -> tuple[list[str], list[str]]:
        xs, ys = [], []
        with open(_DATA / f"{split}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                xs.append(r["text"])
                ys.append(r["sentiment"])
        return xs, ys

    def test_macro_f1_meets_dod(self) -> None:
        if not (_DATA / "train.jsonl").exists():
            pytest.skip("labeled corpus not present")
        x_tr, y_tr = self._load("train")
        x_va, y_va = self._load("val")
        x_te, y_te = self._load("test")
        det = SentimentDetector()  # production TF-IDF word+char + LinearSVC
        det.train(x_tr + x_va, y_tr + y_va)
        macro_f1 = det.evaluate(x_te, y_te)
        assert macro_f1 >= 0.70, f"macro-F1 {macro_f1:.3f} below the 0.70 DoD"
        assert macro_f1 > 0.1675  # beats the majority baseline
