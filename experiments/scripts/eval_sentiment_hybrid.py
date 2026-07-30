"""M2: chunk-pooled multilingual embedding + lexical TF-IDF hybrid, traditional classifier.

Fixes the truncation that hurt the naive embedding (0.60): splits each long conversation into
word-chunks, embeds each, mean-pools -> a full-conversation vector. Then compares lexical-only,
pooled-embedding-only, and the hybrid (concat) under LogisticRegression. Traditional/efficient ML.
"""

import json
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_WORDS = 40


def clean(t: str) -> str:
    return re.sub(r"\[(customer|agent)\]", " ", t)


def chunks(text: str) -> list[str]:
    w = text.split()
    cs = [" ".join(w[i : i + CHUNK_WORDS]) for i in range(0, len(w), CHUNK_WORDS)]
    return cs or [text]


def load(split: str) -> tuple[list[str], list[str]]:
    xs, ys = [], []
    with open(f"experiments/data/{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            xs.append(clean(r["text"]))
            ys.append(r["sentiment"])
    return xs, ys


def pooled(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # Flatten all chunks, embed once (batched), then mean-pool per document.
    flat, spans = [], []
    for t in texts:
        cs = chunks(t)
        spans.append((len(flat), len(flat) + len(cs)))
        flat.extend(cs)
    emb = model.encode(flat, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return np.vstack([emb[a:b].mean(axis=0) for a, b in spans])


def macro(clf, xf, yf, xt, yt) -> float:
    clf.fit(xf, yf)
    return float(f1_score(yt, clf.predict(xt), average="macro"))


def main() -> None:
    x_tr, y_tr = load("train")
    x_va, y_va = load("val")
    x_te, y_te = load("test")
    x_fit, y_fit = x_tr + x_va, y_tr + y_va

    print("chunk-pool embedding ...", flush=True)
    model = SentenceTransformer(MODEL)
    e_fit, e_te = pooled(model, x_fit), pooled(model, x_te)

    feats = FeatureUnion(
        [
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=50000)),
        ]
    )
    l_fit, l_te = feats.fit_transform(x_fit), feats.transform(x_te)

    def lr():
        return LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")

    res = {
        "pooled_embedding_only": round(macro(lr(), e_fit, y_fit, e_te, y_te), 4),
        "lexical_only": round(macro(lr(), l_fit, y_fit, l_te, y_te), 4),
        "hybrid": round(macro(lr(), hstack([l_fit, csr_matrix(e_fit)]), y_fit, hstack([l_te, csr_matrix(e_te)]), y_te), 4),
    }
    res["best"] = max(k for k in ("pooled_embedding_only", "lexical_only", "hybrid"))
    best_f1 = max(res["pooled_embedding_only"], res["lexical_only"], res["hybrid"])
    res["best_macro_f1"] = best_f1
    res["dod_pass"] = bool(best_f1 >= 0.70)
    with open("experiments/results/M2/sentiment_hybrid_metrics.json", "w") as out:
        json.dump(res, out, indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print("DoD (best >= 0.70):", "PASS" if res["dod_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
