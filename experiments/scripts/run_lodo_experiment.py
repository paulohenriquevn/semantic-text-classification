"""Leave-One-Domain-Out (LODO) cross-domain evaluation for TalkEx dissertation.

Evaluates domain generalization of the best-performing classification configuration
from H2 (lexical+structural+embedding LightGBM with context windows).

For each of the 8 domains, the domain is held out as the test set and the
remaining 7 domains are split into train (80%) / val (20%). The pipeline
mirrors H2 exactly: TurnSegmenter -> SlidingWindowBuilder -> lexical+structural+
embedding features -> LightGBM (n_estimators=100, num_leaves=31). Window
predictions are aggregated back to conversation level via average class
probabilities (argmax).

Results measure how well a model trained on 7 domains generalises to
conversations from an unseen domain. Reports Macro-F1, accuracy, per-class F1
per fold, plus mean ± std summary across all 8 folds.

Usage:
    python experiments/scripts/run_lodo_experiment.py
    python experiments/scripts/run_lodo_experiment.py \\
        --splits-dir experiments/data \\
        --output-dir experiments/results/LODO \\
        --seed 42
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np

# ---------------------------------------------------------------------------
# Ensure experiments/scripts is on the path so we can import run_experiment
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# Re-use pipeline functions from run_experiment to avoid duplication (DRY)
from run_experiment import (
    ConversationWindows,
    _aggregate_windows_to_conversations,
    _compute_classification_metrics,
    _extract_window_structural_features,
    _flatten_windows,
    _make_embedding_generator,
    _prepare_windowed_data,
    generate_embeddings_via_talkex,
)

from talkex.classification.features import extract_lexical_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
WINDOW_SIZE = 5
WINDOW_STRIDE = 2
VAL_FRACTION = 0.20  # 20% of non-test domains allocated to val
LGBM_KWARGS: dict[str, Any] = {"n_estimators": 100, "num_leaves": 31, "verbosity": -1, "random_state": 42}

DOMAINS: tuple[str, ...] = (
    "financeiro",
    "restaurante",
    "saude",
    "imobiliario",
    "telecom",
    "ecommerce",
    "tecnologia",
    "educacao",
)


# ---------------------------------------------------------------------------
# Data loading — merges all three splits into one pool for LODO partitioning
# ---------------------------------------------------------------------------


def load_all_splits(splits_dir: Path) -> list[dict]:
    """Load and merge train.jsonl, val.jsonl, and test.jsonl into one pool.

    LODO requires domain-based partitioning, which must be re-done from scratch
    over the entire dataset. Merging all splits first gives each fold access to
    the full domain-stratified data, independent of the original splits.

    Args:
        splits_dir: Directory containing train.jsonl, val.jsonl, test.jsonl.

    Returns:
        Merged list of all records across all three splits.

    Raises:
        FileNotFoundError: If any required split file is missing.
    """
    all_records: list[dict] = []
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}. Run build_splits.py first.")
        split_count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
                    split_count += 1
        logger.info("Loaded %d records from %s/%s.jsonl", split_count, splits_dir, split_name)
    logger.info("Total pool: %d records across all splits", len(all_records))
    return all_records


def _partition_by_domain(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by their 'domain' field.

    Args:
        records: All dataset records.

    Returns:
        Dict mapping domain name to list of records for that domain.
    """
    by_domain: dict[str, list[dict]] = {}
    for r in records:
        domain = r.get("domain", "unknown")
        by_domain.setdefault(domain, []).append(r)
    for domain in sorted(by_domain):
        logger.info("  Domain %-15s: %d records", domain, len(by_domain[domain]))
    return by_domain


def _stratified_train_val_split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split records into train and val, stratified by topic label.

    Stratification preserves the class distribution in each split, which
    is important when the source pool contains all 7 non-held-out domains.

    Args:
        records: Records to split (7 domains pooled together).
        val_fraction: Fraction of records to allocate to validation (0.0-1.0).
        seed: Random seed for reproducibility.

    Returns:
        (train_records, val_records) tuple.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in records:
        label = r.get("topic", "unknown")
        by_label.setdefault(label, []).append(r)

    train: list[dict] = []
    val: list[dict] = []
    for label_records in by_label.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


# ---------------------------------------------------------------------------
# Feature construction (lexical + structural + embedding, mirrors H2)
# ---------------------------------------------------------------------------


def _build_combined_features(
    conv_windows: list[ConversationWindows],
    win_texts: list[str],
    embeddings: np.ndarray,
) -> tuple[list[dict[str, float]], list[str]]:
    """Build lexical + structural + embedding feature dicts per window.

    Mirrors the H2 feature construction from run_experiment.py exactly:
    lexical features + structural metadata features + 384-dim embedding.

    Args:
        conv_windows: Per-conversation windows (for structural extraction).
        win_texts: Flat list of window texts aligned with embeddings.
        embeddings: Embedding matrix of shape (n_windows, dims).

    Returns:
        (features_list, feature_names) — features_list is parallel to win_texts.
    """
    struct_feats = _extract_window_structural_features(conv_windows)
    features_list: list[dict[str, float]] = []

    for i, text in enumerate(win_texts):
        lex = extract_lexical_features(text)
        combined: dict[str, float] = {**(lex.features if hasattr(lex, "features") else dict(lex)), **struct_feats[i]}
        for d in range(embeddings.shape[1]):
            combined[f"emb_{d}"] = float(embeddings[i][d])
        features_list.append(combined)

    feature_names = list(features_list[0].keys()) if features_list else []
    return features_list, feature_names


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Results for a single LODO fold (one held-out domain).

    Attributes:
        held_out_domain: Domain name held out as test set.
        n_train: Training conversations used.
        n_val: Validation conversations used.
        n_test: Test conversations (held-out domain).
        n_train_windows: Training windows used.
        n_test_windows: Test windows evaluated.
        macro_f1: Macro-F1 on the held-out domain (out-of-domain).
        accuracy: Accuracy on the held-out domain.
        per_class_f1: Per-class F1 scores on the held-out domain.
        val_in_domain_f1: Macro-F1 on the validation set (in-domain proxy).
        duration_ms: Wall-clock time for this fold in milliseconds.
    """

    held_out_domain: str
    n_train: int
    n_val: int
    n_test: int
    n_train_windows: int
    n_test_windows: int
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    val_in_domain_f1: float
    duration_ms: float


# ---------------------------------------------------------------------------
# Fold runner
# ---------------------------------------------------------------------------


def _run_fold(
    fold_idx: int,
    held_out_domain: str,
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    seed: int,
    emb_gen: Any,
) -> FoldResult:
    """Execute one LODO fold: train on 7 domains, evaluate on held-out domain.

    Pipeline steps:
        1. Parse conversations into context windows (TalkEx TurnSegmenter +
           SlidingWindowBuilder, window_size=5, stride=2)
        2. Generate window embeddings via paraphrase-multilingual-MiniLM-L12-v2
        3. Build lexical + structural + embedding features per window
        4. Train LightGBM (n_estimators=100, num_leaves=31) on training windows
        5. Evaluate on validation windows, aggregate to conversations (in-domain)
        6. Evaluate on test windows, aggregate to conversations (out-of-domain)
        7. Compute Macro-F1, accuracy, per-class F1

    Args:
        fold_idx: 0-based fold index (for logging and embedding ID namespacing).
        held_out_domain: Domain name held out as test.
        train_records: Training records (80% of 7 source domains).
        val_records: Validation records (20% of 7 source domains).
        test_records: Test records (held-out domain, all records).
        seed: Random seed for LightGBM training.
        emb_gen: Pre-initialized SentenceTransformerGenerator shared across folds.

    Returns:
        FoldResult with all metrics for this fold.

    Raises:
        ValueError: If windowing produces no training or test windows.
    """
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.models import ClassificationInput

    t_fold_start = time.perf_counter()
    logger.info("-" * 60)
    logger.info("Fold %d/%d: held-out = %s", fold_idx + 1, len(DOMAINS), held_out_domain)

    # --- Step 1: Context windows ---
    train_cw = _prepare_windowed_data(train_records)
    val_cw = _prepare_windowed_data(val_records)
    test_cw = _prepare_windowed_data(test_records)

    if not train_cw:
        raise ValueError(
            f"Fold [{held_out_domain}]: no training windows after segmentation "
            f"(checked {len(train_records)} train records)"
        )
    if not test_cw:
        raise ValueError(
            f"Fold [{held_out_domain}]: no test windows after segmentation (checked {len(test_records)} test records)"
        )

    train_win_texts, train_win_labels, _ = _flatten_windows(train_cw)
    val_win_texts, val_win_labels, _ = _flatten_windows(val_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    n_train_windows = len(train_win_texts)
    n_test_windows = len(test_win_texts)

    logger.info(
        "  Windows: train=%d (from %d convs), val=%d (from %d convs), test=%d (from %d convs)",
        n_train_windows,
        len(train_cw),
        len(val_win_texts),
        len(val_cw),
        n_test_windows,
        len(test_cw),
    )

    # Label space spans all labels seen across train + val + test to avoid
    # unknown-label errors when the held-out domain has rare classes
    test_conv_labels = [cw.label for cw in test_cw]
    val_conv_labels = [cw.label for cw in val_cw]
    unique_labels = sorted(set(train_win_labels + val_win_labels + test_conv_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # --- Step 2: Generate embeddings ---
    logger.info("  Generating embeddings for train/val/test windows...")
    train_embs = generate_embeddings_via_talkex(
        train_win_texts,
        [f"fold{fold_idx}_train_{i}" for i in range(n_train_windows)],
        emb_gen,
    )
    val_embs = generate_embeddings_via_talkex(
        val_win_texts,
        [f"fold{fold_idx}_val_{i}" for i in range(len(val_win_texts))],
        emb_gen,
    )
    test_embs = generate_embeddings_via_talkex(
        test_win_texts,
        [f"fold{fold_idx}_test_{i}" for i in range(n_test_windows)],
        emb_gen,
    )
    logger.info("  Embedding dims: %d", train_embs.shape[1])

    # --- Step 3: Build feature dicts (lexical + structural + embedding) ---
    train_features, feature_names = _build_combined_features(train_cw, train_win_texts, train_embs)
    val_features, _ = _build_combined_features(val_cw, val_win_texts, val_embs)
    test_features, _ = _build_combined_features(test_cw, test_win_texts, test_embs)

    # --- Step 4: Build ClassificationInputs ---
    def _make_inputs(
        texts: list[str],
        features: list[dict[str, float]],
        prefix: str,
    ) -> list[ClassificationInput]:
        return [
            ClassificationInput(
                source_id=f"{prefix}_win_{i}",
                source_type="window",
                text=texts[i],
                features=features[i],
            )
            for i in range(len(texts))
        ]

    train_inputs = _make_inputs(train_win_texts, train_features, f"fold{fold_idx}_train")
    val_inputs = _make_inputs(val_win_texts, val_features, f"fold{fold_idx}_val")
    test_inputs = _make_inputs(test_win_texts, test_features, f"fold{fold_idx}_test")

    # --- Step 5: Train LightGBM (mirrors H2 best config) ---
    logger.info("  Training LightGBM on %d windows...", len(train_inputs))
    fold_lgbm_kwargs = {**LGBM_KWARGS, "random_state": seed}
    clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=feature_names,
        model_name="lightgbm",
        lgbm_kwargs=fold_lgbm_kwargs,
    )
    clf.fit(train_inputs, train_win_labels)

    # --- Step 6: Validation evaluation (in-domain proxy) ---
    val_window_preds = clf.classify(val_inputs)
    val_conv_preds = _aggregate_windows_to_conversations(val_window_preds, val_cw, unique_labels)
    val_metrics = _compute_classification_metrics(val_conv_preds, val_conv_labels, unique_labels)
    in_domain_f1 = val_metrics["macro_f1"]
    logger.info("  [VAL in-domain]   macro_f1=%.4f", in_domain_f1)

    # --- Step 7: Test evaluation (out-of-domain) ---
    test_window_preds = clf.classify(test_inputs)
    test_conv_preds = _aggregate_windows_to_conversations(test_window_preds, test_cw, unique_labels)
    test_metrics = _compute_classification_metrics(test_conv_preds, test_conv_labels, unique_labels)

    duration_ms = (time.perf_counter() - t_fold_start) * 1000
    logger.info(
        "  [TEST out-of-domain=%s] macro_f1=%.4f, accuracy=%.4f  (%.0f ms)",
        held_out_domain,
        test_metrics["macro_f1"],
        test_metrics["accuracy"],
        duration_ms,
    )

    return FoldResult(
        held_out_domain=held_out_domain,
        n_train=len(train_records),
        n_val=len(val_records),
        n_test=len(test_records),
        n_train_windows=n_train_windows,
        n_test_windows=n_test_windows,
        macro_f1=test_metrics["macro_f1"],
        accuracy=test_metrics["accuracy"],
        per_class_f1={k.replace("f1_", ""): v for k, v in test_metrics.items() if k.startswith("f1_")},
        val_in_domain_f1=in_domain_f1,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Results serialisation
# ---------------------------------------------------------------------------


def _build_results_dict(
    fold_results: list[FoldResult],
    seed: int,
    total_duration_ms: float,
) -> dict[str, Any]:
    """Assemble the final results dict to be written as results.json.

    Computes mean ± std, min/max across folds and the generalization gap
    (mean in-domain val F1 minus mean out-of-domain test F1).

    Args:
        fold_results: One FoldResult per LODO fold.
        seed: The random seed used for the experiment.
        total_duration_ms: Total wall-clock time for all folds.

    Returns:
        Dict conforming to the LODO results.json schema.
    """
    f1_values = [r.macro_f1 for r in fold_results]
    in_domain_f1_values = [r.val_in_domain_f1 for r in fold_results]

    mean_f1 = float(np.mean(f1_values))
    std_f1 = float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0
    mean_in_domain_f1 = float(np.mean(in_domain_f1_values))
    generalization_gap = mean_in_domain_f1 - mean_f1

    folds_dicts = [
        {
            "held_out_domain": r.held_out_domain,
            "n_train": r.n_train,
            "n_val": r.n_val,
            "n_test": r.n_test,
            "n_train_windows": r.n_train_windows,
            "n_test_windows": r.n_test_windows,
            "macro_f1": round(r.macro_f1, 6),
            "accuracy": round(r.accuracy, 6),
            "per_class_f1": {k: round(v, 6) for k, v in r.per_class_f1.items()},
            "val_in_domain_f1": round(r.val_in_domain_f1, 6),
            "duration_ms": round(r.duration_ms, 1),
        }
        for r in fold_results
    ]

    return {
        "experiment": "LODO",
        "description": "Leave-One-Domain-Out cross-domain generalization evaluation",
        "timestamp": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "classifier": "LightGBM",
            "lgbm_n_estimators": LGBM_KWARGS["n_estimators"],
            "lgbm_num_leaves": LGBM_KWARGS["num_leaves"],
            "val_fraction": VAL_FRACTION,
            "train_val_split_stratified_by_topic": True,
            "seed": seed,
            "feature_set": "lexical+structural+embedding",
        },
        "n_folds": len(fold_results),
        "domains": list(DOMAINS),
        "folds": folds_dicts,
        "summary": {
            "mean_macro_f1": round(mean_f1, 6),
            "std_macro_f1": round(std_f1, 6),
            "min_macro_f1": round(min(f1_values), 6),
            "max_macro_f1": round(max(f1_values), 6),
            "mean_in_domain_val_f1": round(mean_in_domain_f1, 6),
            "generalization_gap": round(generalization_gap, 6),
            "total_duration_ms": round(total_duration_ms, 1),
        },
    }


def _log_summary(results_dict: dict[str, Any]) -> None:
    """Log a human-readable per-fold table plus overall summary stats.

    Args:
        results_dict: The complete results dict from _build_results_dict.
    """
    summary = results_dict["summary"]
    logger.info("=" * 60)
    logger.info("LODO Experiment Complete — Summary")
    logger.info("=" * 60)
    logger.info("%-20s  %9s  %9s", "Domain (held-out)", "F1 (OOD)", "F1 (val)")
    logger.info("%-20s  %9s  %9s", "-" * 20, "-" * 9, "-" * 9)
    for fold in results_dict["folds"]:
        logger.info(
            "%-20s  %9.4f  %9.4f",
            fold["held_out_domain"],
            fold["macro_f1"],
            fold["val_in_domain_f1"],
        )
    logger.info("%-20s  %9s  %9s", "-" * 20, "-" * 9, "-" * 9)
    logger.info(
        "%-20s  %9.4f  %9.4f  (std=%.4f)",
        "MEAN",
        summary["mean_macro_f1"],
        summary["mean_in_domain_val_f1"],
        summary["std_macro_f1"],
    )
    logger.info(
        "Generalization gap (in-domain val F1 - OOD test F1): %.4f",
        summary["generalization_gap"],
    )
    logger.info("Total wall time: %.1f s", summary["total_duration_ms"] / 1000)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--splits-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("experiments/data"),
    show_default=True,
    help="Directory containing train.jsonl, val.jsonl, test.jsonl.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("experiments/results/LODO"),
    show_default=True,
    help="Directory where results.json will be written.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for train/val split and LightGBM training.",
)
def main(splits_dir: Path, output_dir: Path, seed: int) -> None:
    """Run Leave-One-Domain-Out (LODO) evaluation for TalkEx.

    Merges all splits into one pool, partitions by domain, then runs 8 folds —
    each holding out one domain as test and training on the remaining 7
    (80/20 stratified train/val split). Uses LightGBM with lexical+structural+
    embedding features and context windows (window_size=5, stride=2).

    Results are saved to OUTPUT_DIR/results.json.
    """
    logger.info("=" * 60)
    logger.info("TalkEx — Leave-One-Domain-Out (LODO) Experiment")
    logger.info("splits_dir=%s  output_dir=%s  seed=%d", splits_dir, output_dir, seed)
    logger.info("window=%dt/%ds  val_fraction=%.0f%%", WINDOW_SIZE, WINDOW_STRIDE, VAL_FRACTION * 100)
    logger.info("=" * 60)

    t_total_start = time.perf_counter()

    # Load all splits into a single pool (LODO re-partitions by domain)
    all_records = load_all_splits(splits_dir)
    by_domain = _partition_by_domain(all_records)

    missing = set(DOMAINS) - set(by_domain.keys())
    if missing:
        raise ValueError(f"Missing expected domains in dataset: {sorted(missing)}")

    # Load the embedding model once and reuse across all 8 folds to avoid
    # reloading ~100 MB of weights on every iteration
    logger.info("Loading embedding model: %s ...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()
    logger.info("Embedding model loaded.")

    fold_results: list[FoldResult] = []
    for fold_idx, held_out_domain in enumerate(DOMAINS):
        test_records = by_domain[held_out_domain]

        # Pool all other domains as source for stratified train/val split
        source_records: list[dict] = []
        for domain, recs in by_domain.items():
            if domain != held_out_domain:
                source_records.extend(recs)

        train_records, val_records = _stratified_train_val_split(source_records, VAL_FRACTION, seed=seed)
        logger.info(
            "Fold [%s]: source=%d  -> train=%d, val=%d; test=%d",
            held_out_domain,
            len(source_records),
            len(train_records),
            len(val_records),
            len(test_records),
        )

        fold = _run_fold(
            fold_idx=fold_idx,
            held_out_domain=held_out_domain,
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            seed=seed,
            emb_gen=emb_gen,
        )
        fold_results.append(fold)

    total_duration_ms = (time.perf_counter() - t_total_start) * 1000

    results_dict = _build_results_dict(fold_results, seed=seed, total_duration_ms=total_duration_ms)
    _log_summary(results_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
