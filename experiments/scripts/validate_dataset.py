"""Phase 0.5 — Dataset difficulty validation.

Measures the intrinsic difficulty of the classification task BEFORE running
experiments. If the task is trivially easy, all results are inflated and
the experimental contribution is void.

Checks:
  1. Inter-class embedding similarity (high = classes overlap, task is hard)
  2. Intra-class vs inter-class separation (silhouette-like analysis)
  3. Majority class baseline (lower bound for any classifier)
  4. Lexical signal analysis (keyword exclusivity per intent)
  5. Few-shot leakage audit (original→expanded contamination risk)
  6. Ablation: original-only vs expanded difficulty comparison

Usage:
    python experiments/scripts/validate_dataset.py --input demo/data/conversations.jsonl
    python experiments/scripts/validate_dataset.py --input demo/data/conversations.jsonl --with-embeddings
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_records(path: str) -> list[dict]:
    """Load conversation records from JSONL."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Check 1: Majority class baseline
# ---------------------------------------------------------------------------


def check_majority_baseline(records: list[dict]) -> dict:
    """Compute majority class baseline metrics.

    If the majority class F1 is high, the task might be too easy
    or the distribution too skewed.
    """
    intents = [r.get("topic", "outros") for r in records]
    counts = Counter(intents)
    total = len(intents)
    majority_class, majority_count = counts.most_common(1)[0]
    majority_accuracy = majority_count / total

    # Random baseline (uniform guess)
    n_classes = len(counts)
    random_accuracy = 1 / n_classes

    # Majority F1 (only for majority class, others get 0)
    majority_precision = 1.0  # predicts all as majority → all predictions are majority
    majority_recall = majority_count / total  # only correct for majority instances
    # Actually: majority classifier predicts everything as majority_class
    # Precision for majority_class = majority_count / total (all predicted, majority_count correct)
    # Recall for majority_class = 1.0 (all majority instances captured)
    # For other classes: precision=0, recall=0
    # Macro-F1 = average across classes
    majority_f1_macro = (2 * (majority_count / total) * 1.0) / ((majority_count / total) + 1.0) / n_classes

    return {
        "n_classes": n_classes,
        "total_records": total,
        "majority_class": majority_class,
        "majority_count": majority_count,
        "majority_accuracy": round(majority_accuracy, 4),
        "random_accuracy": round(random_accuracy, 4),
        "majority_macro_f1": round(majority_f1_macro, 4),
        "class_distribution": {k: v for k, v in counts.most_common()},
        "imbalance_ratio": round(counts.most_common(1)[0][1] / counts.most_common()[-1][1], 2),
    }


# ---------------------------------------------------------------------------
# Check 2: Lexical signal analysis
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercased tokenization."""
    text = re.sub(r"\[(customer|agent)\]", "", text.lower())
    return [w for w in re.findall(r"\b\w+\b", text) if len(w) > 2]


# Portuguese stopwords (common words with no discriminative power)
PT_STOPWORDS = {
    "que", "com", "para", "por", "uma", "não", "nao", "são", "tem",
    "mais", "como", "seu", "sua", "esse", "essa", "isso", "está",
    "pode", "ser", "ter", "foi", "ele", "ela", "dos", "das", "nos",
    "nas", "num", "uma", "uns", "umas", "meu", "minha", "seu", "sua",
    "este", "esta", "isto", "aqui", "ali", "lhe", "voce", "você",
    "muito", "bem", "sim", "tudo", "todo", "toda", "todos", "todas",
    "também", "tambem", "sobre", "entre", "depois", "antes", "ainda",
    "quando", "onde", "qual", "quais", "quem", "porque", "então",
    "entao", "apenas", "cada", "mesmo", "mesma", "outro", "outra",
    "outros", "outras", "agora", "aquele", "aquela", "seus", "suas",
    "dele", "dela", "deles", "delas", "nosso", "nossa", "nossos",
    "nossas", "vou", "vai", "vamos", "estou", "estamos",
}


def check_lexical_signals(records: list[dict]) -> dict:
    """Analyze keyword exclusivity per intent.

    If each intent has highly exclusive keywords, BM25 will trivially
    separate them → results inflated.
    """
    # Build vocabulary per intent
    intent_vocab: dict[str, Counter] = defaultdict(Counter)
    global_vocab: Counter = Counter()

    for r in records:
        intent = r.get("topic", "outros")
        tokens = _tokenize(r.get("text", ""))
        tokens = [t for t in tokens if t not in PT_STOPWORDS]
        intent_vocab[intent].update(tokens)
        global_vocab.update(tokens)

    # For each intent, find discriminative words
    # Discriminative = appears in this intent much more than in others
    intents = sorted(intent_vocab.keys())
    intent_total: dict[str, int] = {i: sum(intent_vocab[i].values()) for i in intents}

    # Compute TF-IDF-like discriminative score per word per intent
    # Score = (freq_in_intent / total_in_intent) / (freq_global / total_global)
    total_global = sum(global_vocab.values())
    n_intents = len(intents)

    discriminative_words: dict[str, list[tuple[str, float]]] = {}
    exclusivity_scores: dict[str, float] = {}

    for intent in intents:
        word_scores: list[tuple[str, float]] = []
        for word, count in intent_vocab[intent].most_common(100):
            if global_vocab[word] < 5:
                continue  # skip rare words
            tf_intent = count / intent_total[intent]
            tf_global = global_vocab[word] / total_global
            score = tf_intent / tf_global if tf_global > 0 else 0

            # How many intents does this word appear in?
            n_intents_with_word = sum(1 for i in intents if word in intent_vocab[i])
            word_scores.append((word, round(score, 2)))

        discriminative_words[intent] = word_scores[:10]

        # Exclusivity score: average ratio of top-10 words
        # High = words are exclusive to this intent (easy to classify)
        if word_scores:
            exclusivity_scores[intent] = round(
                sum(s for _, s in word_scores[:10]) / min(10, len(word_scores)), 2
            )
        else:
            exclusivity_scores[intent] = 0.0

    # Overall exclusivity
    mean_exclusivity = round(sum(exclusivity_scores.values()) / len(exclusivity_scores), 2)

    # Cross-intent word overlap: what fraction of top-20 words per intent
    # appear in 3+ other intents?
    overlap_count = 0
    total_top_words = 0
    for intent in intents:
        top_words = [w for w, _ in intent_vocab[intent].most_common(20)]
        for word in top_words:
            total_top_words += 1
            n_intents_with = sum(1 for i in intents if word in intent_vocab[i])
            if n_intents_with >= 3:
                overlap_count += 1

    overlap_ratio = round(overlap_count / total_top_words, 4) if total_top_words > 0 else 0

    return {
        "mean_exclusivity_score": mean_exclusivity,
        "overlap_ratio_top20": overlap_ratio,
        "exclusivity_per_intent": exclusivity_scores,
        "top_discriminative_words": discriminative_words,
        "interpretation": _interpret_lexical(mean_exclusivity, overlap_ratio),
    }


def _interpret_lexical(exclusivity: float, overlap: float) -> str:
    """Interpret lexical signal analysis."""
    if exclusivity > 5.0 and overlap < 0.2:
        return "RISK: Very high exclusivity + low overlap → task likely trivially easy for BM25"
    if exclusivity > 3.0 and overlap < 0.3:
        return "WARNING: High exclusivity → BM25 may perform very well, inflating H1 results"
    if exclusivity < 2.0 and overlap > 0.5:
        return "GOOD: Low exclusivity + high overlap → task is genuinely challenging"
    return "MODERATE: Mixed signals — proceed with caution, monitor BM25 baseline"


# ---------------------------------------------------------------------------
# Check 3: Few-shot leakage audit
# ---------------------------------------------------------------------------


def check_leakage(records: list[dict]) -> dict:
    """Audit few-shot contamination risk.

    For expanded conversations, check which original conversations were
    used as few-shot examples. Flag if an original appears in the test set
    while its derived conversations appear in the training set.
    """
    original_ids: set[str] = set()
    expanded_ids: set[str] = set()
    few_shot_map: dict[str, list[str]] = {}  # expanded_id → [original_ids used as few-shot]

    for r in records:
        conv_id = r.get("conversation_id", "")
        metadata = r.get("metadata", {})
        few_shot_ids = metadata.get("few_shot_ids", [])

        if few_shot_ids:
            expanded_ids.add(conv_id)
            few_shot_map[conv_id] = few_shot_ids
        else:
            original_ids.add(conv_id)

    # Build reverse map: original_id → [expanded_ids that used it as few-shot]
    reverse_map: dict[str, list[str]] = defaultdict(list)
    for exp_id, fs_ids in few_shot_map.items():
        for orig_id in fs_ids:
            reverse_map[orig_id].append(exp_id)

    # Count how many times each original was used
    usage_counts = {k: len(v) for k, v in reverse_map.items()}
    max_usage = max(usage_counts.values()) if usage_counts else 0
    mean_usage = sum(usage_counts.values()) / len(usage_counts) if usage_counts else 0

    # Originals most used as few-shot (highest contamination risk)
    most_used = sorted(usage_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "original_count": len(original_ids),
        "expanded_count": len(expanded_ids),
        "originals_used_as_few_shot": len(reverse_map),
        "max_usage_count": max_usage,
        "mean_usage_count": round(mean_usage, 1),
        "most_used_originals": most_used,
        "recommendation": (
            "During split creation, ensure that if an original conversation is in the test set, "
            "all expanded conversations derived from it (via few-shot) are NOT in the training set. "
            f"The {len(most_used)} most-reused originals are highest contamination risk."
        ),
    }


# ---------------------------------------------------------------------------
# Check 4: Embedding-based analysis (optional, requires sentence-transformers)
# ---------------------------------------------------------------------------


def check_embedding_separation(records: list[dict], model_name: str) -> dict:
    """Compute inter-class and intra-class embedding distances.

    High inter-class similarity → classes hard to separate (good).
    Low inter-class similarity → classes trivially separable (inflated results).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {"error": "sentence-transformers not installed. Run: pip install sentence-transformers"}

    logger.info("Loading embedding model %s...", model_name)
    model = SentenceTransformer(model_name)

    # Group texts by intent
    texts_by_intent: dict[str, list[str]] = defaultdict(list)
    for r in records:
        intent = r.get("topic", "outros")
        text = r.get("text", "")[:512]  # truncate for efficiency
        texts_by_intent[intent].append(text)

    # Sample to keep computation manageable
    max_per_class = 100
    sampled: dict[str, list[str]] = {}
    for intent, texts in texts_by_intent.items():
        if len(texts) > max_per_class:
            import random

            rng = random.Random(42)
            sampled[intent] = rng.sample(texts, max_per_class)
        else:
            sampled[intent] = texts

    # Generate embeddings
    logger.info("Generating embeddings for %d intents...", len(sampled))
    embeddings_by_intent: dict[str, np.ndarray] = {}
    for intent, texts in sampled.items():
        embeddings_by_intent[intent] = model.encode(texts, show_progress_bar=False)

    intents = sorted(embeddings_by_intent.keys())

    # Compute centroids
    centroids: dict[str, np.ndarray] = {}
    for intent in intents:
        centroids[intent] = embeddings_by_intent[intent].mean(axis=0)

    # Inter-class similarity (cosine between centroids)
    inter_class_sims: list[float] = []
    inter_class_detail: dict[str, float] = {}
    for i, intent_a in enumerate(intents):
        for intent_b in intents[i + 1 :]:
            sim = float(
                np.dot(centroids[intent_a], centroids[intent_b])
                / (np.linalg.norm(centroids[intent_a]) * np.linalg.norm(centroids[intent_b]))
            )
            inter_class_sims.append(sim)
            inter_class_detail[f"{intent_a}↔{intent_b}"] = round(sim, 4)

    # Intra-class cohesion (mean cosine to centroid)
    intra_class_cohesion: dict[str, float] = {}
    for intent in intents:
        embs = embeddings_by_intent[intent]
        centroid = centroids[intent]
        sims = []
        for emb in embs:
            sim = float(np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid)))
            sims.append(sim)
        intra_class_cohesion[intent] = round(float(np.mean(sims)), 4)

    mean_inter = round(float(np.mean(inter_class_sims)), 4)
    mean_intra = round(float(np.mean(list(intra_class_cohesion.values()))), 4)
    separation_ratio = round(mean_intra / mean_inter, 4) if mean_inter > 0 else float("inf")

    # Most similar class pairs (hardest to separate)
    sorted_pairs = sorted(inter_class_detail.items(), key=lambda x: -x[1])
    hardest_pairs = sorted_pairs[:5]
    easiest_pairs = sorted_pairs[-5:]

    return {
        "model": model_name,
        "mean_inter_class_similarity": mean_inter,
        "mean_intra_class_cohesion": mean_intra,
        "separation_ratio": separation_ratio,
        "hardest_pairs": hardest_pairs,
        "easiest_pairs": easiest_pairs,
        "intra_class_cohesion": intra_class_cohesion,
        "interpretation": _interpret_embeddings(mean_inter, separation_ratio),
    }


def _interpret_embeddings(mean_inter: float, separation_ratio: float) -> str:
    """Interpret embedding separation analysis."""
    if mean_inter > 0.90:
        return "RISK: Classes almost indistinguishable in embedding space → task very hard or embeddings unsuitable"
    if mean_inter > 0.80:
        return "GOOD: High inter-class similarity → task is challenging, results will be meaningful"
    if mean_inter > 0.60:
        return "MODERATE: Moderate separation → task has reasonable difficulty"
    if separation_ratio > 2.0 and mean_inter < 0.50:
        return "WARNING: Very high separation → task may be trivially easy for embedding-based classifiers"
    return "OK: Separation appears reasonable"


# ---------------------------------------------------------------------------
# Check 5: Original vs expanded difficulty comparison
# ---------------------------------------------------------------------------


def check_original_vs_expanded(records: list[dict]) -> dict:
    """Compare characteristics of original vs expanded conversations.

    If expanded conversations are structurally different from originals,
    the expansion may have introduced artifacts.
    """
    original: list[dict] = []
    expanded: list[dict] = []

    for r in records:
        metadata = r.get("metadata", {})
        if metadata.get("few_shot_ids"):
            expanded.append(r)
        else:
            original.append(r)

    if not expanded:
        return {"note": "No expanded conversations found — only original dataset."}

    def stats(group: list[dict]) -> dict:
        word_counts = [r.get("word_count", 0) for r in group]
        intents = Counter(r.get("topic", "outros") for r in group)
        sentiments = Counter(r.get("sentiment", "unknown") for r in group)

        # Estimate turn counts from text
        turn_counts = []
        for r in group:
            text = r.get("text", "")
            turns = len(re.findall(r"\[(customer|agent)\]", text))
            turn_counts.append(turns)

        return {
            "count": len(group),
            "mean_words": round(float(np.mean(word_counts)), 1),
            "std_words": round(float(np.std(word_counts)), 1),
            "mean_turns": round(float(np.mean(turn_counts)), 1),
            "std_turns": round(float(np.std(turn_counts)), 1),
            "min_turns": min(turn_counts),
            "max_turns": max(turn_counts),
            "intent_distribution": {k: v for k, v in intents.most_common()},
            "sentiment_distribution": {k: v for k, v in sentiments.most_common()},
        }

    return {
        "original": stats(original),
        "expanded": stats(expanded),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--input", "input_path", default="demo/data/conversations.jsonl", help="Input JSONL.")
@click.option(
    "--with-embeddings",
    is_flag=True,
    help="Run embedding-based analysis (slower, requires sentence-transformers).",
)
@click.option(
    "--embedding-model",
    default="paraphrase-multilingual-MiniLM-L12-v2",
    help="Embedding model for similarity analysis.",
)
@click.option(
    "--output",
    default="experiments/data/validation_report.json",
    help="Output path for validation report.",
)
def main(
    input_path: str,
    with_embeddings: bool,
    embedding_model: str,
    output: str,
) -> None:
    """Validate dataset difficulty before running experiments."""
    logger.info("=" * 60)
    logger.info("DATASET DIFFICULTY VALIDATION — Phase 0.5")
    logger.info("=" * 60)

    records = load_records(input_path)
    logger.info("Loaded %d conversations from %s", len(records), input_path)

    report: dict = {"input": input_path, "total_records": len(records)}

    # Check 1: Majority baseline
    logger.info("\n--- Check 1: Majority Class Baseline ---")
    majority = check_majority_baseline(records)
    report["majority_baseline"] = majority
    logger.info("  Classes: %d", majority["n_classes"])
    logger.info("  Majority class: %s (%d, %.1f%%)",
                majority["majority_class"], majority["majority_count"],
                majority["majority_accuracy"] * 100)
    logger.info("  Random accuracy: %.1f%%", majority["random_accuracy"] * 100)
    logger.info("  Imbalance ratio: %.1f", majority["imbalance_ratio"])

    # Check 2: Lexical signals
    logger.info("\n--- Check 2: Lexical Signal Analysis ---")
    lexical = check_lexical_signals(records)
    report["lexical_signals"] = lexical
    logger.info("  Mean exclusivity score: %.2f", lexical["mean_exclusivity_score"])
    logger.info("  Cross-intent overlap (top-20): %.1f%%", lexical["overlap_ratio_top20"] * 100)
    logger.info("  Interpretation: %s", lexical["interpretation"])
    for intent, score in sorted(lexical["exclusivity_per_intent"].items(), key=lambda x: -x[1]):
        top_words = [w for w, _ in lexical["top_discriminative_words"].get(intent, [])[:5]]
        logger.info("    %s: %.2f — %s", intent, score, ", ".join(top_words))

    # Check 3: Leakage audit
    logger.info("\n--- Check 3: Few-Shot Leakage Audit ---")
    leakage = check_leakage(records)
    report["leakage_audit"] = leakage
    logger.info("  Original conversations: %d", leakage["original_count"])
    logger.info("  Expanded conversations: %d", leakage["expanded_count"])
    if leakage["expanded_count"] > 0:
        logger.info("  Originals used as few-shot: %d", leakage["originals_used_as_few_shot"])
        logger.info("  Max reuse count: %d", leakage["max_usage_count"])
        logger.info("  Mean reuse count: %.1f", leakage["mean_usage_count"])

    # Check 4: Original vs expanded
    logger.info("\n--- Check 4: Original vs Expanded Comparison ---")
    comparison = check_original_vs_expanded(records)
    report["original_vs_expanded"] = comparison
    if "original" in comparison:
        orig = comparison["original"]
        exp = comparison["expanded"]
        logger.info("  Original: %d convs, %.1f±%.1f words, %.1f±%.1f turns",
                    orig["count"], orig["mean_words"], orig["std_words"],
                    orig["mean_turns"], orig["std_turns"])
        logger.info("  Expanded: %d convs, %.1f±%.1f words, %.1f±%.1f turns",
                    exp["count"], exp["mean_words"], exp["std_words"],
                    exp["mean_turns"], exp["std_turns"])

    # Check 5: Embedding analysis (optional)
    if with_embeddings:
        logger.info("\n--- Check 5: Embedding Separation Analysis ---")
        embedding_analysis = check_embedding_separation(records, embedding_model)
        report["embedding_separation"] = embedding_analysis
        if "error" not in embedding_analysis:
            logger.info("  Model: %s", embedding_analysis["model"])
            logger.info("  Mean inter-class similarity: %.4f", embedding_analysis["mean_inter_class_similarity"])
            logger.info("  Mean intra-class cohesion: %.4f", embedding_analysis["mean_intra_class_cohesion"])
            logger.info("  Separation ratio: %.4f", embedding_analysis["separation_ratio"])
            logger.info("  Interpretation: %s", embedding_analysis["interpretation"])
            logger.info("  Hardest pairs to separate:")
            for pair, sim in embedding_analysis["hardest_pairs"]:
                logger.info("    %s: %.4f", pair, sim)
        else:
            logger.warning("  %s", embedding_analysis["error"])

    # Overall assessment
    logger.info("\n" + "=" * 60)
    logger.info("OVERALL ASSESSMENT")
    logger.info("=" * 60)

    risks = []
    if lexical["mean_exclusivity_score"] > 5.0:
        risks.append("Lexical signals too exclusive — BM25 will trivially separate classes")
    if majority["imbalance_ratio"] > 10:
        risks.append("Extreme class imbalance — majority baseline may be hard to beat")
    if majority["imbalance_ratio"] < 1.5:
        risks.append("Classes too balanced — unrealistic for call center data")
    if with_embeddings and "embedding_separation" in report:
        emb = report["embedding_separation"]
        if "mean_inter_class_similarity" in emb and emb["mean_inter_class_similarity"] < 0.5:
            risks.append("Embedding classes too separable — task may be trivially easy")

    if risks:
        for risk in risks:
            logger.info("  ⚠ %s", risk)
    else:
        logger.info("  No critical risks identified. Proceed with experiments.")

    # Write report
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("\nFull report saved to %s", output_path)


if __name__ == "__main__":
    main()
