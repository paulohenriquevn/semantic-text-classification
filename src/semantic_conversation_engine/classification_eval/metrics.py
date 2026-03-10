"""Classification evaluation metrics — pure functions over label sets.

All functions take predicted labels and ground truth labels, returning
a single float score. Higher is always better.

Supported metrics:
    precision:         fraction of predicted labels that are correct
    recall:            fraction of ground truth labels that were predicted
    f1_score:          harmonic mean of precision and recall
    per_label_metrics: per-label precision, recall, F1, and support
    micro_f1:          F1 computed over global TP/FP/FN counts
    macro_f1:          unweighted mean of per-label F1 scores

All functions are deterministic and stateless.
"""

from __future__ import annotations


def precision(predicted: set[str], ground_truth: set[str]) -> float:
    """Fraction of predicted labels that are correct.

    Args:
        predicted: Set of predicted label names.
        ground_truth: Set of ground truth label names.

    Returns:
        Precision in [0.0, 1.0]. Returns 0.0 if predicted is empty.
    """
    if not predicted:
        return 0.0
    return len(predicted & ground_truth) / len(predicted)


def recall(predicted: set[str], ground_truth: set[str]) -> float:
    """Fraction of ground truth labels that were predicted.

    Args:
        predicted: Set of predicted label names.
        ground_truth: Set of ground truth label names.

    Returns:
        Recall in [0.0, 1.0]. Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth:
        return 0.0
    return len(predicted & ground_truth) / len(ground_truth)


def f1_score(predicted: set[str], ground_truth: set[str]) -> float:
    """Harmonic mean of precision and recall.

    Args:
        predicted: Set of predicted label names.
        ground_truth: Set of ground truth label names.

    Returns:
        F1 in [0.0, 1.0]. Returns 0.0 if both sets are empty.
    """
    p = precision(predicted, ground_truth)
    r = recall(predicted, ground_truth)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def per_label_metrics(
    predictions: list[set[str]],
    ground_truths: list[set[str]],
    label_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-label precision, recall, F1, and support.

    For each label, computes binary classification metrics across
    all examples: is this label predicted vs is it in ground truth.

    Args:
        predictions: List of predicted label sets, one per example.
        ground_truths: List of ground truth label sets, one per example.
        label_names: Ordered list of label names to evaluate.

    Returns:
        Dict mapping label name to dict with keys:
        precision, recall, f1, support (count of examples with this label).
    """
    result: dict[str, dict[str, float]] = {}

    for label in label_names:
        tp = 0
        fp = 0
        fn = 0

        for pred, gt in zip(predictions, ground_truths, strict=True):
            in_pred = label in pred
            in_gt = label in gt
            if in_pred and in_gt:
                tp += 1
            elif in_pred and not in_gt:
                fp += 1
            elif not in_pred and in_gt:
                fn += 1

        label_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        label_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if (label_precision + label_recall) > 0
            else 0.0
        )
        support = tp + fn  # Count of ground truth positives for this label

        result[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "support": float(support),
        }

    return result


def micro_f1(
    predictions: list[set[str]],
    ground_truths: list[set[str]],
    label_names: list[str],
) -> float:
    """F1 score computed over global TP/FP/FN counts across all labels.

    Micro-averaging treats each label decision as an independent binary
    classification. This gives equal weight to every prediction,
    favoring performance on frequent labels.

    Args:
        predictions: List of predicted label sets, one per example.
        ground_truths: List of ground truth label sets, one per example.
        label_names: Ordered list of label names to evaluate.

    Returns:
        Micro-averaged F1 in [0.0, 1.0].
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in label_names:
        for pred, gt in zip(predictions, ground_truths, strict=True):
            in_pred = label in pred
            in_gt = label in gt
            if in_pred and in_gt:
                total_tp += 1
            elif in_pred and not in_gt:
                total_fp += 1
            elif not in_pred and in_gt:
                total_fn += 1

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    if micro_precision + micro_recall == 0.0:
        return 0.0
    return 2 * micro_precision * micro_recall / (micro_precision + micro_recall)


def macro_f1(
    predictions: list[set[str]],
    ground_truths: list[set[str]],
    label_names: list[str],
) -> float:
    """Unweighted mean of per-label F1 scores.

    Macro-averaging gives equal weight to every label regardless of
    frequency. This highlights performance on rare labels.

    Args:
        predictions: List of predicted label sets, one per example.
        ground_truths: List of ground truth label sets, one per example.
        label_names: Ordered list of label names to evaluate.

    Returns:
        Macro-averaged F1 in [0.0, 1.0]. Returns 0.0 if label_names is empty.
    """
    if not label_names:
        return 0.0

    label_metrics = per_label_metrics(predictions, ground_truths, label_names)
    total_f1 = sum(m["f1"] for m in label_metrics.values())
    return total_f1 / len(label_names)
