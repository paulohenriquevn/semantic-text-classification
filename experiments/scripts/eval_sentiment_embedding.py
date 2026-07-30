"""M2 de-risk: multilingual embedding + LogisticRegression sentiment macro-F1 vs baseline.

Reuses the project embedding model (paraphrase-multilingual-MiniLM-L12-v2) over the internal
sentiment-labeled corpus (experiments/data/*.jsonl). Blueprint M2 D1-D3.
"""

import json
import re

from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def clean(t: str) -> str:
    return re.sub(r"\[(customer|agent)\]", " ", t)


def load(split: str) -> tuple[list[str], list[str]]:
    xs: list[str] = []
    ys: list[str] = []
    with open(f"experiments/data/{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            xs.append(clean(r["text"]))
            ys.append(r["sentiment"])
    return xs, ys


def main() -> None:
    x_tr, y_tr = load("train")
    x_va, y_va = load("val")
    x_te, y_te = load("test")
    x_fit, y_fit = x_tr + x_va, y_tr + y_va

    print(f"embedding {len(x_fit) + len(x_te)} texts with {MODEL} ...", flush=True)
    model = SentenceTransformer(MODEL)
    e_fit = model.encode(x_fit, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    e_te = model.encode(x_te, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    base = DummyClassifier(strategy="most_frequent").fit(e_fit, y_fit)
    base_f1 = f1_score(y_te, base.predict(e_te), average="macro")

    clf = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(e_fit, y_fit)
    f1 = f1_score(y_te, clf.predict(e_te), average="macro")

    res = {
        "model": MODEL,
        "baseline_macro_f1": round(float(base_f1), 4),
        "embedding_logreg_macro_f1": round(float(f1), 4),
        "dod_pass": bool(f1 >= 0.70 and f1 > base_f1),
        "n_train": len(y_fit),
        "n_test": len(y_te),
    }
    with open("experiments/results/M2/sentiment_metrics.json", "w") as out:
        json.dump(res, out, indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print("DoD (>=0.70 AND > baseline):", "PASS" if res["dod_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
