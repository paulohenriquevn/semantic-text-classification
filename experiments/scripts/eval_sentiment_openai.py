"""M2: OpenAI embedding (features) + traditional LogisticRegression sentiment macro-F1.

The local multilingual embedding capped at ~0.60 and the lexical at ~0.67 (< 0.70 DoD). This uses a
stronger embedding (OpenAI text-embedding-3-small, full conversation, no truncation) as FEATURES only;
the classifier stays traditional/efficient (LogisticRegression). Requires OPENAI_API_KEY in the env.
The key is never written here; the embeddings are cached to experiments/results/M2/ to avoid re-billing.
"""

import json
import os
import re

import numpy as np
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

EMBED_MODEL = "text-embedding-3-small"
CACHE = "experiments/results/M2/openai_embeddings.npz"


def clean(t: str) -> str:
    return re.sub(r"\[(customer|agent)\]", " ", t)


def load(split: str) -> tuple[list[str], list[str]]:
    xs, ys = [], []
    with open(f"experiments/data/{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            xs.append(clean(r["text"]))
            ys.append(r["sentiment"])
    return xs, ys


def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), 128):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + 128])
        out.extend(d.embedding for d in resp.data)
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    x_tr, y_tr = load("train")
    x_va, y_va = load("val")
    x_te, y_te = load("test")
    x_fit, y_fit = x_tr + x_va, y_tr + y_va

    if os.path.exists(CACHE):
        z = np.load(CACHE)
        e_fit, e_te = z["fit"], z["te"]
        print("loaded cached OpenAI embeddings", flush=True)
    else:
        client = OpenAI()
        print(f"embedding {len(x_fit) + len(x_te)} texts with {EMBED_MODEL} ...", flush=True)
        e_fit, e_te = embed(client, x_fit), embed(client, x_te)
        np.savez_compressed(CACHE, fit=e_fit, te=e_te)

    clf = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(e_fit, y_fit)
    f1 = float(f1_score(y_te, clf.predict(e_te), average="macro"))

    res = {
        "embedding_model": EMBED_MODEL,
        "classifier": "LogisticRegression",
        "macro_f1": round(f1, 4),
        "baseline_majority_macro_f1": 0.1675,
        "dod_pass": bool(f1 >= 0.70),
        "n_train": len(y_fit),
        "n_test": len(y_te),
    }
    with open("experiments/results/M2/sentiment_openai_metrics.json", "w") as out:
        json.dump(res, out, indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print("DoD (>= 0.70):", "PASS" if res["dod_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
