"""Stratified k-fold cross-validation experiment for the TalkEx dissertation.

Addresses the deterministic multi-seed problem (std=0.000) by using stratified
k-fold CV instead of fixed train/val/test splits. This provides real confidence
intervals by forcing different train/test boundaries.

Uses the same pipeline as run_experiment.py (H2 best config):
    context windows → lexical+structural+embedding features → LightGBM

Also computes calibration metrics (Brier score, ECE) per fold.

Usage:
    python experiments/scripts/run_kfold_experiment.py
    python experiments/scripts/run_kfold_experiment.py --n-folds 10
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np

# Allow sibling imports when run as script
sys.path.insert(0, str(Path(__file__).parent))

from run_experiment import (
    EMBEDDING_MODEL,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    _aggregate_windows_to_conversations_with_probs,
    _build_experiment_rules,
    _compute_calibration_metrics,
    _compute_classification_metrics,
    _extract_window_structural_features,
    _flatten_windows,
    _make_embedding_generator,
    _prepare_windowed_data,
    generate_embeddings_via_talkex,
    load_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_single_fold(
    train_records: list[dict],
    test_records: list[dict],
    fold_idx: int,
    seed: int,
) -> dict[str, Any]:
    """Run one fold of cross-validation using the H2 best pipeline.

    Pipeline: context windows → lexical+structural+embedding+rule features → LightGBM.

    Returns a dict with all metrics for this fold.
    """
    from talkex.classification.features import extract_lexical_features
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.models import ClassificationInput
    from talkex.rules.config import RuleEngineConfig
    from talkex.rules.evaluator import SimpleRuleEvaluator
    from talkex.rules.models import RuleEvaluationInput

    logger.info("  Fold %d: train=%d, test=%d", fold_idx + 1, len(train_records), len(test_records))

    # Step 1: Prepare windowed data
    train_cw = _prepare_windowed_data(train_records)
    test_cw = _prepare_windowed_data(test_records)

    train_win_texts, train_win_labels, _ = _flatten_windows(train_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    test_conv_labels = [cw.label for cw in test_cw]
    unique_labels = sorted(set(train_win_labels + test_conv_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # Step 2: Extract features
    # Lexical
    train_lex = []
    for text in train_win_texts:
        lex = extract_lexical_features(text)
        train_lex.append(lex.features if hasattr(lex, "features") else dict(lex))
    test_lex = []
    for text in test_win_texts:
        lex = extract_lexical_features(text)
        test_lex.append(lex.features if hasattr(lex, "features") else dict(lex))

    # Structural
    train_struct = _extract_window_structural_features(train_cw)
    test_struct = _extract_window_structural_features(test_cw)

    # Embeddings
    emb_gen = _make_embedding_generator()
    train_embs = generate_embeddings_via_talkex(
        train_win_texts, [f"train_{i}" for i in range(len(train_win_texts))], emb_gen
    )
    test_embs = generate_embeddings_via_talkex(
        test_win_texts, [f"test_{i}" for i in range(len(test_win_texts))], emb_gen
    )

    def make_emb_dicts(embeddings: np.ndarray) -> list[dict[str, float]]:
        return [
            {f"emb_{d}": float(embeddings[i][d]) for d in range(embeddings.shape[1])} for i in range(len(embeddings))
        ]

    train_emb_feats = make_emb_dicts(train_embs)
    test_emb_feats = make_emb_dicts(test_embs)

    # Rule features
    rules, _rule_to_label = _build_experiment_rules()
    evaluator = SimpleRuleEvaluator()
    rule_config = RuleEngineConfig()

    def compute_rule_feats(win_texts: list[str]) -> list[dict[str, float]]:
        feats_list = []
        for i, text in enumerate(win_texts):
            eval_input = RuleEvaluationInput(
                source_id=f"win_{i}",
                source_type="window",
                text=text,
            )
            rule_results = evaluator.evaluate(rules, eval_input, rule_config)
            feats = {}
            for rule in rules:
                matched = any(rr.matched for rr in rule_results if rr.rule_id == rule.rule_id)
                feats[f"rule_{rule.rule_id}"] = 1.0 if matched else 0.0
            feats_list.append(feats)
        return feats_list

    train_rule_feats = compute_rule_feats(train_win_texts)
    test_rule_feats = compute_rule_feats(test_win_texts)

    # Merge all features
    def merge_all(
        lex: list[dict], struct: list[dict], emb: list[dict], rule: list[dict]
    ) -> tuple[list[dict[str, float]], list[str]]:
        merged = []
        for i in range(len(lex)):
            features: dict[str, float] = {}
            features.update(lex[i])
            features.update(struct[i])
            features.update(emb[i])
            features.update(rule[i])
            merged.append(features)
        names = list(merged[0].keys()) if merged else []
        return merged, names

    train_feats, feat_names = merge_all(train_lex, train_struct, train_emb_feats, train_rule_feats)
    test_feats, _ = merge_all(test_lex, test_struct, test_emb_feats, test_rule_feats)

    # Step 3: Build ClassificationInputs
    train_inputs = [
        ClassificationInput(
            source_id=f"train_{i}",
            source_type="window",
            text=train_win_texts[i],
            features=train_feats[i],
        )
        for i in range(len(train_win_texts))
    ]
    test_inputs = [
        ClassificationInput(
            source_id=f"test_{i}",
            source_type="window",
            text=test_win_texts[i],
            features=test_feats[i],
        )
        for i in range(len(test_win_texts))
    ]

    # Step 4: Train and evaluate
    t0 = time.perf_counter()
    clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=feat_names,
        lgbm_kwargs={"n_estimators": 100, "num_leaves": 31, "verbosity": -1, "random_state": seed},
    )
    clf.fit(train_inputs, train_win_labels)

    test_window_preds = clf.classify(test_inputs)
    test_conv_preds, test_conv_probs = _aggregate_windows_to_conversations_with_probs(
        test_window_preds, test_cw, unique_labels
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    # Metrics
    cls_metrics = _compute_classification_metrics(test_conv_preds, test_conv_labels, unique_labels)
    cal_metrics = _compute_calibration_metrics(test_conv_probs, test_conv_labels, unique_labels)

    fold_result = {
        "fold": fold_idx + 1,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "n_train_windows": len(train_win_texts),
        "n_test_windows": len(test_win_texts),
        "n_features": len(feat_names),
        "duration_ms": round(duration_ms, 1),
        "macro_f1": cls_metrics["macro_f1"],
        "accuracy": cls_metrics["accuracy"],
        "brier_score": cal_metrics["brier_score"],
        "ece": cal_metrics["ece"],
    }

    # Per-class F1
    for label in unique_labels:
        fold_result[f"f1_{label}"] = cls_metrics.get(f"f1_{label}", 0.0)

    logger.info(
        "  Fold %d: macro_f1=%.4f, accuracy=%.4f, brier=%.4f, ece=%.4f",
        fold_idx + 1,
        cls_metrics["macro_f1"],
        cls_metrics["accuracy"],
        cal_metrics["brier_score"],
        cal_metrics["ece"],
    )
    return fold_result


@click.command()
@click.option("--splits-dir", default="experiments/data", help="Directory with JSONL splits.")
@click.option("--output-dir", default="experiments/results/kfold", help="Output directory.")
@click.option("--n-folds", default=5, help="Number of folds (default: 5).")
@click.option("--seed", default=42, help="Random seed for fold splitting.")
def main(splits_dir: str, output_dir: str, n_folds: int, seed: int) -> None:
    """Run stratified k-fold cross-validation on the full dataset.

    Pools all splits (train+val+test) into one dataset, then performs
    stratified k-fold CV to get real confidence intervals.
    """
    from sklearn.model_selection import StratifiedKFold

    splits_path = Path(splits_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and pool all data
    logger.info("=" * 60)
    logger.info("Stratified %d-Fold Cross-Validation", n_folds)
    logger.info("=" * 60)

    all_records: list[dict] = []
    for split in ["train", "val", "test"]:
        records = load_split(split, splits_path)
        all_records.extend(records)

    labels = [r.get("topic", "unknown") for r in all_records]
    logger.info("Total records: %d, unique labels: %d", len(all_records), len(set(labels)))

    # Stratified k-fold split
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    records_arr = np.array(all_records, dtype=object)
    labels_arr = np.array(labels)

    fold_results: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(records_arr, labels_arr)):
        train_records = [all_records[i] for i in train_idx]
        test_records = [all_records[i] for i in test_idx]

        fold_result = _run_single_fold(train_records, test_records, fold_idx, seed)
        fold_results.append(fold_result)

    # Aggregate across folds
    metric_keys = [
        "macro_f1",
        "accuracy",
        "brier_score",
        "ece",
    ]
    # Add per-class F1 keys
    all_labels = sorted(set(labels))
    for label in all_labels:
        metric_keys.append(f"f1_{label}")

    summary: dict[str, Any] = {
        "experiment": "stratified_kfold_cv",
        "n_folds": n_folds,
        "seed": seed,
        "total_records": len(all_records),
        "pipeline": "lexical+structural+embedding+rules LightGBM",
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "embedding_model": EMBEDDING_MODEL,
        "n_rules": 10,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for key in metric_keys:
        values = [fr[key] for fr in fold_results if key in fr]
        if values:
            summary[f"mean_{key}"] = round(float(np.mean(values)), 4)
            summary[f"std_{key}"] = round(float(np.std(values)), 4)
            summary[f"min_{key}"] = round(float(np.min(values)), 4)
            summary[f"max_{key}"] = round(float(np.max(values)), 4)

    summary["per_fold"] = fold_results

    # Log summary
    logger.info("=" * 60)
    logger.info("k-Fold CV Results (%d folds)", n_folds)
    logger.info("=" * 60)
    logger.info(
        "Macro-F1: %.4f ± %.4f (min=%.4f, max=%.4f)",
        summary.get("mean_macro_f1", 0),
        summary.get("std_macro_f1", 0),
        summary.get("min_macro_f1", 0),
        summary.get("max_macro_f1", 0),
    )
    logger.info(
        "Accuracy: %.4f ± %.4f",
        summary.get("mean_accuracy", 0),
        summary.get("std_accuracy", 0),
    )
    logger.info(
        "Brier: %.4f ± %.4f",
        summary.get("mean_brier_score", 0),
        summary.get("std_brier_score", 0),
    )
    logger.info(
        "ECE: %.4f ± %.4f",
        summary.get("mean_ece", 0),
        summary.get("std_ece", 0),
    )

    # Per-class F1 summary
    for label in all_labels:
        key = f"f1_{label}"
        logger.info(
            "  %s: F1 = %.4f ± %.4f",
            label,
            summary.get(f"mean_{key}", 0),
            summary.get(f"std_{key}", 0),
        )

    # Save results
    results_path = output_path / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
