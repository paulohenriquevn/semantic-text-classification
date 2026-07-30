"""Unit tests for M7 model versioning — every prediction records its model_version (T2.1, D6)."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from talkex.classification.sentiment import SentimentDetector


def _tiny() -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1)), ("clf", LinearSVC(C=1.0))])


class TestSentimentVersioning:
    def test_predict_records_model_version(self) -> None:
        det = SentimentDetector(_tiny(), model_version="sentiment-2026.07.30")
        det.train(
            ["péssimo serviço quero cancelar", "muito ruim isso", "ótimo atendimento obrigado", "excelente tudo"],
            ["negative", "negative", "positive", "positive"],
        )
        pred = det.predict("quero cancelar péssimo")
        assert pred.model_version == "sentiment-2026.07.30"  # evidence axiom — traceable to a version

    def test_default_version(self) -> None:
        assert SentimentDetector(_tiny()).model_version == "v0"
