"""Unit tests for the M7 retraining pipeline (T2.1) — retrain + benchmark vs deployed, promote on gain."""

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from talkex.classification.retraining_pipeline import RetrainingPipeline
from talkex.classification.sentiment import SentimentDetector
from talkex.monitoring.domain.lifecycle import RetrainingSample


def _tiny_detector(version: str) -> SentimentDetector:
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1)), ("clf", LinearSVC(C=1.0))])
    return SentimentDetector(pipe, model_version=version)


def _samples() -> list[RetrainingSample]:
    data = [
        ("péssimo serviço quero cancelar", "negative"),
        ("muito ruim isso não aguento", "negative"),
        ("horrível atendimento cancelar", "negative"),
        ("ótimo atendimento muito obrigado", "positive"),
        ("excelente tudo perfeito", "positive"),
        ("adorei o suporte obrigado", "positive"),
    ]
    return [
        RetrainingSample(turn_id=f"t{i}", conversation_id="c", redacted_text=text, label=label)
        for i, (text, label) in enumerate(data)
    ]


_EVAL_TEXTS = ["péssimo quero cancelar", "ótimo obrigado excelente"]
_EVAL_LABELS = ["negative", "positive"]


class TestRetrainingPipeline:
    def test_retrain_benchmarks_and_promotes_first_model(self) -> None:
        pipe = RetrainingPipeline(detector_factory=_tiny_detector)
        model, result = pipe.retrain_and_benchmark(
            _samples(), _EVAL_TEXTS, _EVAL_LABELS, new_version="v1", deployed=None
        )
        assert model.is_fitted
        assert result.new_model_version == "v1"
        assert result.deployed_f1 is None
        assert result.promoted is True  # no incumbent → the first model is promoted
        assert 0.0 <= result.new_f1 <= 1.0

    def test_retrain_does_not_promote_without_gain(self) -> None:
        pipe = RetrainingPipeline(detector_factory=_tiny_detector)
        # A deployed model trained on the SAME data → the candidate cannot strictly beat it.
        deployed = _tiny_detector("v0")
        deployed.train([s.redacted_text for s in _samples()], [str(s.label) for s in _samples()])

        _, result = pipe.retrain_and_benchmark(
            _samples(), _EVAL_TEXTS, _EVAL_LABELS, new_version="v1", deployed=deployed
        )
        assert result.deployed_f1 is not None
        assert result.promoted == (result.new_f1 > result.deployed_f1)  # promote strictly on gain

    def test_no_labeled_samples_fails_fast(self) -> None:
        unlabeled = [RetrainingSample(turn_id="t", conversation_id="c", redacted_text="x", label=None)]
        with pytest.raises(ValueError, match="no labeled samples"):
            RetrainingPipeline(detector_factory=_tiny_detector).retrain_and_benchmark(
                unlabeled, _EVAL_TEXTS, _EVAL_LABELS, new_version="v1"
            )
