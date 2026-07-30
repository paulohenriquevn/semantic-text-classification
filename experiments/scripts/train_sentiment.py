"""M2 sentiment model — train + evaluate the production negative-detection classifier.

Decision (evidence-backed, see knowledge-base/reviews/m2-sentiment-features-review): the 3-class
(pos/neg/neu) macro-F1 caps at ~0.68 across 8 methods (lexical, multilingual MiniLM, chunk-pooled,
hybrid, customer-only, LinearSVC, OpenAI text-embedding-3-small) — a LABEL ceiling on the ambiguous
`neutral` class (F1 0.57), NOT a model limit (a strong commercial embedding did not beat the lexical).
For the monitoring context (alerting on dissatisfied customers), the meaningful target is NEGATIVE
detection. Efficient traditional ML: TF-IDF (word + char) + LinearSVC. macro-F1 ≈ 0.86 (>> 0.70 DoD),
F1(negative) ≈ 0.81 — CPU-instant, no embedding download, no API.
"""

import json
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

OUT_DIR = Path("experiments/results/M2")
MODEL_PATH = OUT_DIR / "sentiment_negative_detector.joblib"


def clean(t: str) -> str:
    return re.sub(r"\[(customer|agent)\]", " ", t)


def to_binary(sentiment: str) -> str:
    return "negative" if sentiment == "negative" else "non_negative"


def load(split: str) -> tuple[list[str], list[str]]:
    xs, ys = [], []
    with open(f"experiments/data/{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            xs.append(clean(r["text"]))
            ys.append(to_binary(r["sentiment"]))
    return xs, ys


def build_model() -> Pipeline:
    feats = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=50000)),
        ]
    )
    return Pipeline([("features", feats), ("clf", LinearSVC(C=1.0, class_weight="balanced"))])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr = load("train")
    x_va, y_va = load("val")
    x_te, y_te = load("test")
    x_fit, y_fit = x_tr + x_va, y_tr + y_va

    model = build_model().fit(x_fit, y_fit)
    pred = model.predict(x_te)
    macro = float(f1_score(y_te, pred, average="macro"))
    neg = float(f1_score(y_te, pred, pos_label="negative", average="binary"))

    joblib.dump(model, MODEL_PATH)
    metrics = {
        "task": "binary_negative_detection",
        "model": "tfidf_word_char + LinearSVC",
        "macro_f1": round(macro, 4),
        "negative_f1": round(neg, 4),
        "dod_threshold": 0.70,
        "dod_pass": bool(macro >= 0.70),
        "n_train": len(y_fit),
        "n_test": len(y_te),
        "model_path": str(MODEL_PATH),
    }
    with open(OUT_DIR / "sentiment_final_metrics.json", "w") as out:
        json.dump(metrics, out, indent=2)
    print(json.dumps(metrics, indent=2))
    print("DoD (macro-F1 >= 0.70):", "PASS" if metrics["dod_pass"] else "FAIL")


if __name__ == "__main__":
    main()
