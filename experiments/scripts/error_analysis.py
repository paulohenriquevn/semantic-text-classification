"""Detailed per-class error analysis for the TalkEx lexical+emb LightGBM classifier.

Reproduces the best H2 model (lexical+emb LightGBM, context windows 5t/2s),
then performs a structured error analysis covering:

  1. Confusion matrix (raw counts + recall-normalized)
  2. Top confusion pairs per class (especially for weak classes)
  3. Misclassified conversation samples for weak classes
  4. LightGBM feature importance (gain-based, grouped by family)
  5. Class separability via mean embedding distances between centroids
  6. Breakdown by data origin (synthetic vs original) per class
  7. Average prediction confidence for correct vs incorrect predictions
  8. Most-confused class pairs (symmetric: A→B + B→A)

Results are saved to experiments/results/error_analysis/:
  - results.json        — unified machine-readable output (primary output)
  - confusion_matrix.json
  - error_analysis.json
  - summary.md

Usage:
    python experiments/scripts/error_analysis.py
    python experiments/scripts/error_analysis.py \\
        --splits-dir experiments/data \\
        --output-dir experiments/results/error_analysis
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np

from talkex.classification.features import extract_lexical_features, extract_structural_features
from talkex.context import ContextWindowConfig, SlidingWindowBuilder
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.context_window import ContextWindow
from talkex.models.conversation import Conversation
from talkex.models.enums import Channel
from talkex.models.types import ConversationId
from talkex.segmentation import SegmentationConfig, TurnSegmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror run_experiment.py exactly for reproducibility
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_VERSION = "1.0"
WINDOW_SIZE = 5
WINDOW_STRIDE = 2
SEED = 42

INTENT_CLASSES = [
    "cancelamento",
    "compra",
    "duvida_produto",
    "duvida_servico",
    "elogio",
    "reclamacao",
    "saudacao",
    "suporte_tecnico",
]

# Classes identified as weak in H2 results (F1 < 0.5)
WEAK_CLASSES = ["compra", "saudacao"]

# Max misclassified examples per weak class to include in output
MAX_MISCLASSIFIED_SAMPLES = 5

# Top-N features to report in feature importance section
TOP_N_FEATURES = 20

# Feature family prefixes — order matters for grouping
FEATURE_FAMILIES: dict[str, str] = {
    "lex_": "lexical",
    "struct_": "structural",
    "emb_": "embedding",
    "rule_": "rules",
}

# Pipeline singletons — reusable, stateless
_SEGMENTER = TurnSegmenter()
_WINDOW_BUILDER = SlidingWindowBuilder()
_SEG_CONFIG = SegmentationConfig(
    speaker_label_pattern=r"\[(customer|agent)\]",
    merge_consecutive_same_speaker=False,
    normalize_unicode=False,
    collapse_whitespace=False,
    strip_lines=False,
)
_WINDOW_CONFIG = ContextWindowConfig(window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
_EPOCH = datetime(2024, 1, 1)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationWindows:
    """All windows for one conversation with its ground-truth label.

    Wraps real TalkEx ContextWindow objects with the experiment-specific
    ground-truth label (not part of the TalkEx domain model).
    """

    conversation_id: str
    label: str
    raw_text: str
    windows: list[ContextWindow]
    origin: str = "unknown"  # "synthetic" or "original"


# ---------------------------------------------------------------------------
# Data loading — identical to run_experiment.py helpers
# ---------------------------------------------------------------------------


def load_split(split_name: str, data_dir: Path) -> list[dict]:
    """Load a JSONL split file.

    Args:
        split_name: One of 'train', 'val', 'test'.
        data_dir: Directory containing the JSONL files.

    Returns:
        List of record dicts.

    Raises:
        FileNotFoundError: If the split file does not exist.
    """
    path = data_dir / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}. Run build_splits.py first.")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def _get_record_id(record: dict, fallback: str) -> str:
    """Get conversation ID from a record, supporting both formats."""
    return record.get("conversation_id", record.get("id", fallback))


# ---------------------------------------------------------------------------
# Pipeline — context window builder (mirrors run_experiment.py exactly)
# ---------------------------------------------------------------------------


def _get_record_origin(record: dict) -> str:
    """Determine data origin from source_file field.

    Args:
        record: Conversation record dict.

    Returns:
        "synthetic" if from synthetic expansion, "original" otherwise.
    """
    source_file = record.get("source_file", "")
    if "synthetic" in source_file.lower():
        return "synthetic"
    return "original"


def _prepare_windowed_data(records: list[dict]) -> list[ConversationWindows]:
    """Parse conversations into context windows via the real TalkEx pipeline.

    Uses TurnSegmenter → SlidingWindowBuilder — the same modules as in production.
    Retains raw_text for misclassification sampling and origin for breakdown analysis.

    Args:
        records: JSONL records from a data split.

    Returns:
        One ConversationWindows per successfully-parsed conversation.
    """
    result: list[ConversationWindows] = []
    for r in records:
        conv_id = _get_record_id(r, "unknown")
        label = r.get("topic", "unknown")
        origin = _get_record_origin(r)
        text = r.get("text", "")
        if not text.strip():
            continue

        transcript = TranscriptInput(
            conversation_id=ConversationId(conv_id),
            channel=Channel.CHAT,
            raw_text=text,
            source_format=SourceFormat.LABELED,
        )
        turns = _SEGMENTER.segment(transcript, _SEG_CONFIG)
        if not turns:
            continue

        conversation = Conversation(
            conversation_id=ConversationId(conv_id),
            channel=Channel.CHAT,
            start_time=_EPOCH,
        )
        windows = _WINDOW_BUILDER.build(conversation, turns, _WINDOW_CONFIG)
        if windows:
            result.append(
                ConversationWindows(
                    conversation_id=conv_id,
                    label=label,
                    raw_text=text,
                    windows=windows,
                    origin=origin,
                )
            )
    return result


def _flatten_windows(
    conv_windows: list[ConversationWindows],
) -> tuple[list[str], list[str], list[str]]:
    """Flatten windowed data into parallel lists for training/classification.

    Returns:
        (window_texts, window_labels, window_conv_ids) — all index-aligned.
    """
    texts: list[str] = []
    labels: list[str] = []
    conv_ids: list[str] = []
    for cw in conv_windows:
        for w in cw.windows:
            texts.append(w.window_text)
            labels.append(cw.label)
            conv_ids.append(cw.conversation_id)
    return texts, labels, conv_ids


def _extract_window_structural_features(
    conv_windows: list[ConversationWindows],
) -> list[dict[str, float]]:
    """Extract structural features from real TalkEx ContextWindow metadata."""
    features: list[dict[str, float]] = []
    for cw in conv_windows:
        for w in cw.windows:
            speakers_meta = w.metadata.get("speakers", {})
            feat = extract_structural_features(
                is_customer=speakers_meta.get("has_customer", False),
                is_agent=speakers_meta.get("has_agent", False),
                turn_count=w.window_size,
                speaker_count=len(speakers_meta.get("distribution", {})),
            )
            features.append(feat.features)
    return features


# ---------------------------------------------------------------------------
# Embedding generation — mirrors run_experiment.py exactly
# ---------------------------------------------------------------------------


def _make_embedding_generator():  # -> SentenceTransformerGenerator
    """Create a SentenceTransformerGenerator with the canonical model config.

    Returns:
        Initialized generator ready for embedding generation.
    """
    from talkex.embeddings.config import EmbeddingModelConfig
    from talkex.embeddings.generator import SentenceTransformerGenerator

    model_config = EmbeddingModelConfig(
        model_name=EMBEDDING_MODEL,
        model_version=EMBEDDING_VERSION,
        batch_size=64,
    )
    return SentenceTransformerGenerator(model_config=model_config)


def generate_embeddings(
    texts: list[str],
    record_ids: list[str],
    generator: Any,
) -> np.ndarray:
    """Generate embeddings via the TalkEx SentenceTransformerGenerator.

    Args:
        texts: Texts to embed.
        record_ids: Corresponding IDs (for EmbeddingInput identity).
        generator: Pre-initialized SentenceTransformerGenerator.

    Returns:
        float32 ndarray of shape (len(texts), embedding_dims).
    """
    from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
    from talkex.models.enums import ObjectType

    batch_items = [
        EmbeddingInput(
            embedding_id=f"emb_{rid}",
            object_type=ObjectType.CONVERSATION,
            object_id=rid,
            text=text,
        )
        for rid, text in zip(record_ids, texts, strict=True)
    ]
    batch = EmbeddingBatch(items=batch_items)
    records = generator.generate(batch)
    return np.array([r.vector for r in records], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature construction — mirrors H2 in run_experiment.py
# ---------------------------------------------------------------------------


def build_lexical_plus_embedding_features(
    conv_windows: list[ConversationWindows],
    win_texts: list[str],
    embeddings: np.ndarray,
) -> tuple[list[dict[str, float]], list[str]]:
    """Build lexical + structural + embedding features for each window.

    Reproduces the 'lexical+emb' feature config from H2.

    Args:
        conv_windows: Windowed conversations (for structural feature metadata).
        win_texts: Flat list of window texts (index-aligned with conv_windows).
        embeddings: float32 ndarray of shape (n_windows, emb_dims).

    Returns:
        (features_list, feature_names) where features_list has one dict per window.
    """
    struct_feats = _extract_window_structural_features(conv_windows)
    features_list: list[dict[str, float]] = []

    flat_idx = 0
    for cw in conv_windows:
        for w in cw.windows:
            lex = extract_lexical_features(w.window_text)
            combined: dict[str, float] = {
                **(lex.features if hasattr(lex, "features") else dict(lex)),
                **struct_feats[flat_idx],
            }
            for d in range(embeddings.shape[1]):
                combined[f"emb_{d}"] = float(embeddings[flat_idx][d])
            features_list.append(combined)
            flat_idx += 1

    feature_names = list(features_list[0].keys()) if features_list else []
    return features_list, feature_names


# ---------------------------------------------------------------------------
# Aggregation — mirrors run_experiment.py
# ---------------------------------------------------------------------------


def _aggregate_window_predictions(
    window_results: list,  # list[ClassificationResult]
    labels: list[str],
) -> tuple[str, float]:
    """Aggregate window-level predictions to conversation level.

    Strategy: average class probabilities across windows → argmax.

    Args:
        window_results: ClassificationResult objects for each window.
        labels: All possible labels.

    Returns:
        (predicted_label, confidence) for the conversation.
    """
    if not window_results:
        return labels[0] if labels else "unknown", 0.0

    class_probs: dict[str, float] = {label: 0.0 for label in labels}
    for result in window_results:
        for ls in result.label_scores:
            if ls.label in class_probs:
                class_probs[ls.label] += ls.score

    n_windows = len(window_results)
    for label in class_probs:
        class_probs[label] /= n_windows

    best_label = max(class_probs, key=lambda lbl: class_probs[lbl])
    return best_label, class_probs[best_label]


def aggregate_windows_to_conversations(
    window_preds: list,  # list[ClassificationResult]
    conv_windows: list[ConversationWindows],
    labels: list[str],
) -> list[tuple[str, float]]:
    """Aggregate window predictions to conversation-level (label, confidence) pairs.

    Args:
        window_preds: Flat ClassificationResult list, one per window, index-aligned
            with _flatten_windows() output.
        conv_windows: Original windowed data for grouping.
        labels: All possible labels.

    Returns:
        List of (predicted_label, confidence) tuples, one per conversation.
    """
    results: list[tuple[str, float]] = []
    offset = 0
    for cw in conv_windows:
        n_win = len(cw.windows)
        window_slice = window_preds[offset : offset + n_win]
        label, confidence = _aggregate_window_predictions(window_slice, labels)
        results.append((label, confidence))
        offset += n_win
    return results


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """Compute raw and recall-normalized confusion matrices.

    Args:
        predictions: Predicted labels (one per conversation).
        ground_truth: True labels (one per conversation).
        labels: Ordered list of all class labels.

    Returns:
        Dict with 'labels', 'raw' (counts), and 'normalized' (by true class).
    """
    n = len(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    raw = [[0] * n for _ in range(n)]
    for pred, true in zip(predictions, ground_truth, strict=True):
        true_idx = label_to_idx.get(true)
        pred_idx = label_to_idx.get(pred)
        if true_idx is not None and pred_idx is not None:
            raw[true_idx][pred_idx] += 1

    # Normalize each row by the true class count (recall perspective)
    normalized = []
    for row in raw:
        total = sum(row)
        if total > 0:
            normalized.append([round(v / total, 4) for v in row])
        else:
            normalized.append([0.0] * n)

    return {
        "labels": labels,
        "raw": raw,
        "normalized": normalized,
    }


# ---------------------------------------------------------------------------
# Confusion pair analysis
# ---------------------------------------------------------------------------


def compute_confusion_pairs(
    confusion_matrix: dict[str, Any],
    top_k: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Identify the top-K classes each class is most confused with.

    Args:
        confusion_matrix: Output of compute_confusion_matrix().
        top_k: How many confusion targets to report per class.

    Returns:
        Dict mapping each label to a ranked list of confusion targets with
        raw counts and normalized confusion rates.
    """
    labels = confusion_matrix["labels"]
    raw = confusion_matrix["raw"]
    normalized = confusion_matrix["normalized"]

    pairs: dict[str, list[dict[str, Any]]] = {}
    for i, true_label in enumerate(labels):
        off_diagonal = [
            {
                "predicted_as": labels[j],
                "count": raw[i][j],
                "confusion_rate": normalized[i][j],
            }
            for j in range(len(labels))
            if j != i and raw[i][j] > 0
        ]
        off_diagonal.sort(key=lambda x: -x["count"])
        pairs[true_label] = off_diagonal[:top_k]

    return pairs


# ---------------------------------------------------------------------------
# Misclassified sample extraction
# ---------------------------------------------------------------------------


def extract_misclassified_samples(
    conv_windows: list[ConversationWindows],
    predictions_with_confidence: list[tuple[str, float]],
    target_classes: list[str],
    max_per_class: int = MAX_MISCLASSIFIED_SAMPLES,
) -> dict[str, list[dict[str, Any]]]:
    """Extract representative misclassified examples for target classes.

    Args:
        conv_windows: Windowed conversations (for text and conversation_id).
        predictions_with_confidence: (predicted_label, confidence) per conversation,
            index-aligned with conv_windows.
        target_classes: Only extract samples for these classes.
        max_per_class: Maximum examples per class.

    Returns:
        Dict mapping each target class to a list of misclassification records.
        Each record has: conversation_id, true_label, predicted_label,
        confidence, text_preview (first 200 chars).
    """
    samples: dict[str, list[dict[str, Any]]] = {cls: [] for cls in target_classes}

    for cw, (pred_label, confidence) in zip(conv_windows, predictions_with_confidence, strict=True):
        true_label = cw.label
        if true_label not in target_classes:
            continue
        if pred_label == true_label:
            continue
        if len(samples[true_label]) >= max_per_class:
            continue

        # Build a text preview from the first 200 chars
        text_preview = cw.raw_text[:200].strip()

        samples[true_label].append(
            {
                "conversation_id": cw.conversation_id,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4),
                "text_preview": text_preview,
            }
        )

    return samples


# ---------------------------------------------------------------------------
# Feature importance (LightGBM gain-based)
# ---------------------------------------------------------------------------


def extract_feature_importances(
    classifier: Any,  # LightGBMClassifier
    feature_names: list[str],
    top_n: int = TOP_N_FEATURES,
) -> dict[str, Any]:
    """Extract and group LightGBM feature importances by feature family.

    Gain-based importance reflects the total information gain contributed
    by each feature across all trees — more informative than split count.

    Args:
        classifier: Fitted LightGBMClassifier.
        feature_names: Ordered feature names (matches training order).
        top_n: How many top features to include in the ranked list.

    Returns:
        Dict with 'top_features' (ranked list) and 'by_family' (aggregated
        importance per feature family: lexical, structural, embedding, rules).

    Raises:
        ValueError: If the classifier has not been fitted.
    """
    if not classifier.is_fitted:
        raise ValueError("LightGBMClassifier must be fitted before extracting importances")

    lgbm_model = classifier._model
    raw_importances = lgbm_model.booster_.feature_importance(importance_type="gain")

    # Pair names with importances and normalize
    paired = list(zip(feature_names, raw_importances, strict=True))
    total_gain = sum(imp for _, imp in paired)

    ranked = sorted(paired, key=lambda x: -x[1])

    top_features = [
        {
            "rank": rank + 1,
            "feature_name": name,
            "gain": round(float(imp), 4),
            "gain_pct": round(float(imp) / total_gain * 100, 2) if total_gain > 0 else 0.0,
            "family": _classify_feature_family(name),
        }
        for rank, (name, imp) in enumerate(ranked[:top_n])
    ]

    # Aggregate by family
    family_totals: dict[str, float] = defaultdict(float)
    for name, imp in paired:
        family_totals[_classify_feature_family(name)] += float(imp)

    by_family = {
        family: {
            "total_gain": round(gain, 4),
            "gain_pct": round(gain / total_gain * 100, 2) if total_gain > 0 else 0.0,
        }
        for family, gain in sorted(family_totals.items(), key=lambda x: -x[1])
    }

    return {
        "top_features": top_features,
        "by_family": by_family,
        "total_features": len(feature_names),
        "importance_type": "gain",
    }


def _classify_feature_family(feature_name: str) -> str:
    """Map a feature name to its family based on prefix convention.

    Args:
        feature_name: Feature name such as 'lex_word_count', 'emb_0', 'struct_turn_count'.

    Returns:
        Family label: 'lexical', 'structural', 'embedding', 'rules', or 'unknown'.
    """
    for prefix, family in FEATURE_FAMILIES.items():
        if feature_name.startswith(prefix):
            return family
    return "unknown"


# ---------------------------------------------------------------------------
# Class separability via embedding centroids
# ---------------------------------------------------------------------------


def compute_class_separability(
    conv_windows: list[ConversationWindows],
    win_texts: list[str],
    embeddings: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Compute class separability using mean embedding distances between centroids.

    For each class, the centroid is the mean of all window embeddings whose
    conversation belongs to that class. Mean distance to all other centroids
    indicates separability — low distance means high embedding overlap.

    Args:
        conv_windows: Windowed conversations.
        win_texts: Flat window text list (used for index tracking only).
        embeddings: float32 ndarray of shape (n_windows, emb_dims).
        labels: All class labels to include.

    Returns:
        Dict with 'centroids_computed', 'class_separability' (per class mean
        distance to all others), and 'low_separability_pairs' (closest class pairs).
    """
    # Collect per-class window embeddings
    class_embeddings: dict[str, list[np.ndarray]] = {lbl: [] for lbl in labels}
    flat_idx = 0
    for cw in conv_windows:
        for _ in cw.windows:
            if cw.label in class_embeddings:
                class_embeddings[cw.label].append(embeddings[flat_idx])
            flat_idx += 1

    # Compute centroids
    centroids: dict[str, np.ndarray] = {}
    for lbl, emb_list in class_embeddings.items():
        if emb_list:
            centroids[lbl] = np.mean(emb_list, axis=0)

    if len(centroids) < 2:
        return {
            "centroids_computed": list(centroids.keys()),
            "class_separability": {},
            "low_separability_pairs": [],
        }

    # Pairwise cosine distances between centroids
    centroid_labels = sorted(centroids.keys())
    n = len(centroid_labels)
    pairwise_distances: dict[tuple[str, str], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            lbl_a = centroid_labels[i]
            lbl_b = centroid_labels[j]
            a = centroids[lbl_a]
            b = centroids[lbl_b]
            # Cosine similarity → distance
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
                cos_dist = 1.0 - cos_sim
            else:
                cos_dist = 1.0
            pairwise_distances[(lbl_a, lbl_b)] = round(cos_dist, 4)

    # Mean distance to all other classes (per class)
    class_separability: dict[str, dict[str, Any]] = {}
    for lbl in centroid_labels:
        distances_to_others = [
            {
                "other_class": other,
                "cosine_distance": pairwise_distances.get((min(lbl, other), max(lbl, other)), 0.0),
            }
            for other in centroid_labels
            if other != lbl
        ]
        distances_to_others.sort(key=lambda x: x["cosine_distance"])
        mean_dist = float(np.mean([d["cosine_distance"] for d in distances_to_others]))
        class_separability[lbl] = {
            "mean_distance_to_others": round(mean_dist, 4),
            "n_windows": len(class_embeddings[lbl]),
            "closest_class": distances_to_others[0] if distances_to_others else None,
            "all_distances": distances_to_others,
        }

    # Low separability pairs = closest centroid pairs overall
    sorted_pairs = sorted(pairwise_distances.items(), key=lambda x: x[1])
    low_separability_pairs = [
        {
            "class_a": pair[0],
            "class_b": pair[1],
            "cosine_distance": dist,
        }
        for pair, dist in sorted_pairs[:5]
    ]

    return {
        "centroids_computed": centroid_labels,
        "class_separability": class_separability,
        "low_separability_pairs": low_separability_pairs,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_per_class_metrics(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 per class.

    Args:
        predictions: Predicted labels.
        ground_truth: True labels.
        labels: All class labels.

    Returns:
        Dict mapping each label to {'precision', 'recall', 'f1', 'support'}.
    """
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p != label and g == label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return per_class


# ---------------------------------------------------------------------------
# Origin breakdown per class (synthetic vs original)
# ---------------------------------------------------------------------------


def compute_origin_breakdown(
    conv_windows: list[ConversationWindows],
    predictions_with_confidence: list[tuple[str, float]],
    labels: list[str],
) -> dict[str, Any]:
    """Break down predictions by data origin (synthetic vs original) per class.

    Args:
        conv_windows: Windowed conversations carrying origin metadata.
        predictions_with_confidence: (predicted_label, confidence) per conversation,
            index-aligned with conv_windows.
        labels: All class labels.

    Returns:
        Dict with 'per_class' (accuracy per origin per class) and 'aggregate'
        (overall accuracy per origin across all classes).
    """
    from collections import defaultdict as _defaultdict

    counts: dict[tuple[str, str], dict[str, int]] = _defaultdict(lambda: {"correct": 0, "total": 0})

    for cw, (pred_label, _) in zip(conv_windows, predictions_with_confidence, strict=True):
        true_label = cw.label
        origin = cw.origin
        counts[(true_label, origin)]["total"] += 1
        if pred_label == true_label:
            counts[(true_label, origin)]["correct"] += 1

    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        per_class[label] = {}
        for origin in ("synthetic", "original"):
            entry = counts.get((label, origin), {"correct": 0, "total": 0})
            total = entry["total"]
            correct = entry["correct"]
            per_class[label][origin] = {
                "correct": correct,
                "total": total,
                "accuracy": round(correct / total, 4) if total > 0 else None,
            }

    # Aggregate: overall accuracy per origin (across all classes)
    aggregate: dict[str, Any] = {}
    for origin in ("synthetic", "original"):
        pairs = [
            (pred, cw.label)
            for cw, (pred, _) in zip(conv_windows, predictions_with_confidence, strict=True)
            if cw.origin == origin
        ]
        if pairs:
            n_correct = sum(1 for pred, true in pairs if pred == true)
            total = len(pairs)
            aggregate[origin] = {
                "total": total,
                "correct": n_correct,
                "accuracy": round(n_correct / total, 4),
            }
        else:
            aggregate[origin] = {"total": 0, "correct": 0, "accuracy": None}

    return {"per_class": per_class, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Confidence analysis (correct vs incorrect predictions)
# ---------------------------------------------------------------------------


def compute_confidence_analysis(
    conv_windows: list[ConversationWindows],
    predictions_with_confidence: list[tuple[str, float]],
    labels: list[str],
) -> dict[str, Any]:
    """Compute average prediction confidence for correct vs incorrect predictions.

    The confidence is the aggregated probability assigned to the predicted
    class after window averaging.

    Args:
        conv_windows: Windowed conversations (for true label).
        predictions_with_confidence: (predicted_label, confidence) per conversation,
            index-aligned with conv_windows.
        labels: All class labels.

    Returns:
        Dict with 'overall' and 'per_class' breakdowns of correct vs incorrect
        confidence (mean, median, std, n).
    """
    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []

    per_class_correct: dict[str, list[float]] = {lbl: [] for lbl in labels}
    per_class_incorrect: dict[str, list[float]] = {lbl: [] for lbl in labels}

    for cw, (pred_label, confidence) in zip(conv_windows, predictions_with_confidence, strict=True):
        true_label = cw.label
        if pred_label == true_label:
            correct_confidences.append(confidence)
            if true_label in per_class_correct:
                per_class_correct[true_label].append(confidence)
        else:
            incorrect_confidences.append(confidence)
            if true_label in per_class_incorrect:
                per_class_incorrect[true_label].append(confidence)

    def _stats(values: list[float]) -> dict[str, Any] | None:
        if not values:
            return None
        arr = np.array(values)
        return {
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(arr.std()), 4),
            "n": len(values),
        }

    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        per_class[label] = {
            "correct": _stats(per_class_correct[label]),
            "incorrect": _stats(per_class_incorrect[label]),
        }

    return {
        "overall": {
            "correct": _stats(correct_confidences),
            "incorrect": _stats(incorrect_confidences),
        },
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Most-confused class pairs (symmetric: A→B + B→A)
# ---------------------------------------------------------------------------


def compute_most_confused_pairs(
    confusion_matrix: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Identify class pairs most confused with each other in both directions.

    Sums A→B and B→A errors into a symmetric total confusion score.

    Args:
        confusion_matrix: Output of compute_confusion_matrix().
        top_k: Number of most-confused pairs to return.

    Returns:
        List of {class_a, class_b, a_predicted_as_b, b_predicted_as_a,
        total_confusion} dicts, sorted by total_confusion descending.
    """
    labels = confusion_matrix["labels"]
    raw = confusion_matrix["raw"]
    n = len(labels)

    pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a_as_b = raw[i][j]  # true=i, predicted as j
            b_as_a = raw[j][i]  # true=j, predicted as i
            total = a_as_b + b_as_a
            if total > 0:
                pairs.append(
                    {
                        "class_a": labels[i],
                        "class_b": labels[j],
                        "a_predicted_as_b": a_as_b,
                        "b_predicted_as_a": b_as_a,
                        "total_confusion": total,
                    }
                )

    pairs.sort(key=lambda p: p["total_confusion"], reverse=True)
    return pairs[:top_k]


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def write_summary_report(
    output_dir: Path,
    per_class_metrics: dict[str, dict[str, float]],
    confusion_pairs: dict[str, list[dict[str, Any]]],
    misclassified_samples: dict[str, list[dict[str, Any]]],
    feature_importances: dict[str, Any],
    separability: dict[str, Any],
    macro_f1: float,
    accuracy: float,
    n_test_conversations: int,
) -> None:
    """Write a human-readable Markdown summary report.

    Args:
        output_dir: Directory to write summary.md into.
        per_class_metrics: Per-class precision/recall/F1/support.
        confusion_pairs: Top confusion targets per class.
        misclassified_samples: Misclassified examples for weak classes.
        feature_importances: Feature importance analysis results.
        separability: Class separability results.
        macro_f1: Overall macro-averaged F1.
        accuracy: Overall accuracy.
        n_test_conversations: Number of test conversations evaluated.
    """
    lines: list[str] = []

    lines += [
        "# Error Analysis — TalkEx lexical+emb LightGBM",
        "",
        f"**Model:** lexical+emb LightGBM (n_estimators=100, num_leaves=31, seed={SEED})",
        f"**Pipeline:** context windows {WINDOW_SIZE}t/{WINDOW_STRIDE}s → aggregation: avg_confidence_argmax",
        f"**Dataset:** test split ({n_test_conversations} conversations)",
        "",
        f"**Macro-F1:** {macro_f1:.4f}  |  **Accuracy:** {accuracy:.4f}",
        "",
    ]

    # --- Per-class metrics table ---
    lines += [
        "## Per-Class Performance",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    sorted_classes = sorted(per_class_metrics.items(), key=lambda x: x[1]["f1"])
    for label, m in sorted_classes:
        marker = " **(weak)**" if label in WEAK_CLASSES else ""
        lines.append(f"| {label}{marker} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines.append("")

    # --- Top confusion pairs for weak classes ---
    lines += [
        "## Confusion Analysis — Weak Classes",
        "",
    ]
    for cls in WEAK_CLASSES:
        pairs = confusion_pairs.get(cls, [])
        lines.append(f"### {cls}")
        if pairs:
            lines += [
                "| Confused with | Count | Confusion rate |",
                "|---|---|---|",
            ]
            for pair in pairs:
                lines.append(f"| {pair['predicted_as']} | {pair['count']} | {pair['confusion_rate']:.1%} |")
        else:
            lines.append("_No confusions found._")
        lines.append("")

    # --- Misclassified samples ---
    lines += [
        "## Misclassified Sample Conversations",
        "",
    ]
    for cls in WEAK_CLASSES:
        samples = misclassified_samples.get(cls, [])
        lines.append(f"### {cls} (up to {MAX_MISCLASSIFIED_SAMPLES} examples)")
        if not samples:
            lines.append("_No misclassified examples found._")
        for s in samples:
            lines += [
                f"- **ID:** `{s['conversation_id']}`",
                f"  - Predicted as: `{s['predicted_label']}` (confidence: {s['confidence']:.3f})",
                f"  - Text preview: _{s['text_preview']}_",
                "",
            ]

    # --- Feature importances ---
    lines += [
        "## Feature Importance (Gain-Based, Top 20)",
        "",
        "### By Feature Family",
        "",
        "| Family | Total Gain | Share |",
        "|---|---|---|",
    ]
    for family, info in feature_importances.get("by_family", {}).items():
        lines.append(f"| {family} | {info['total_gain']:.1f} | {info['gain_pct']:.1f}% |")
    lines += [
        "",
        "### Top Features",
        "",
        "| Rank | Feature | Family | Gain | Share |",
        "|---|---|---|---|---|",
    ]
    for feat in feature_importances.get("top_features", []):
        lines.append(
            f"| {feat['rank']} | {feat['feature_name']} | {feat['family']} "
            f"| {feat['gain']:.1f} | {feat['gain_pct']:.1f}% |"
        )
    lines.append("")

    # --- Class separability ---
    lines += [
        "## Class Separability (Embedding Space)",
        "",
        "### Closest Class Pairs (lowest cosine distance = highest overlap)",
        "",
        "| Class A | Class B | Cosine Distance |",
        "|---|---|---|",
    ]
    for pair in separability.get("low_separability_pairs", []):
        lines.append(f"| {pair['class_a']} | {pair['class_b']} | {pair['cosine_distance']:.4f} |")
    lines += [
        "",
        "### Per-Class Mean Distance to Others",
        "",
        "| Class | Mean Distance | Closest Class | Closest Distance | Windows |",
        "|---|---|---|---|---|",
    ]
    sep_data = separability.get("class_separability", {})
    sorted_sep = sorted(sep_data.items(), key=lambda x: x[1]["mean_distance_to_others"])
    for label, info in sorted_sep:
        closest = info.get("closest_class")
        closest_cls = closest["other_class"] if closest else "—"
        closest_dist = f"{closest['cosine_distance']:.4f}" if closest else "—"
        lines.append(
            f"| {label} | {info['mean_distance_to_others']:.4f} | {closest_cls} "
            f"| {closest_dist} | {info['n_windows']} |"
        )
    lines.append("")

    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--splits-dir",
    "--data-dir",
    "splits_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("experiments/data"),
    show_default=True,
    help="Directory containing train.jsonl, val.jsonl, test.jsonl.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("experiments/results/error_analysis"),
    show_default=True,
    help="Directory to write analysis results into.",
)
def main(splits_dir: Path, output_dir: Path) -> None:
    """Run per-class error analysis on the best H2 model (lexical+emb LightGBM).

    Reproduces the best-performing classifier from H2, then analyzes error
    patterns, confusion pairs, misclassified samples, feature importances,
    class separability, origin breakdown (synthetic vs original), and
    confidence analysis (correct vs incorrect predictions).

    Results are saved to OUTPUT_DIR as:
      - results.json        — unified machine-readable output
      - confusion_matrix.json
      - error_analysis.json
      - summary.md
    """
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.models import ClassificationInput

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load data splits
    # ------------------------------------------------------------------
    logger.info("Loading data splits from %s...", splits_dir)
    train_records = load_split("train", splits_dir)
    val_records = load_split("val", splits_dir)
    test_records = load_split("test", splits_dir)

    # ------------------------------------------------------------------
    # Step 2: Build context windows (real TalkEx pipeline)
    # ------------------------------------------------------------------
    logger.info("Building context windows (window_size=%d, stride=%d)...", WINDOW_SIZE, WINDOW_STRIDE)
    train_cw = _prepare_windowed_data(train_records)
    val_cw = _prepare_windowed_data(val_records)
    test_cw = _prepare_windowed_data(test_records)
    logger.info(
        "Windows: train=%d (from %d convs), val=%d (from %d convs), test=%d (from %d convs)",
        sum(len(c.windows) for c in train_cw),
        len(train_cw),
        sum(len(c.windows) for c in val_cw),
        len(val_cw),
        sum(len(c.windows) for c in test_cw),
        len(test_cw),
    )

    train_win_texts, train_win_labels, _ = _flatten_windows(train_cw)
    _, val_win_labels, _ = _flatten_windows(val_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    test_conv_labels = [cw.label for cw in test_cw]
    unique_labels = sorted(set(train_win_labels + val_win_labels + test_conv_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # ------------------------------------------------------------------
    # Step 3: Generate embeddings
    # ------------------------------------------------------------------
    logger.info("Generating window embeddings via talkex (%s)...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()

    train_embeddings = generate_embeddings(
        train_win_texts, [f"train_win_{i}" for i in range(len(train_win_texts))], emb_gen
    )
    test_embeddings = generate_embeddings(
        test_win_texts, [f"test_win_{i}" for i in range(len(test_win_texts))], emb_gen
    )
    logger.info("Window embeddings: %d dims", train_embeddings.shape[1])

    # ------------------------------------------------------------------
    # Step 4: Build features (lexical + structural + embedding)
    # ------------------------------------------------------------------
    logger.info("Extracting lexical + structural + embedding features...")
    train_features, feature_names = build_lexical_plus_embedding_features(train_cw, train_win_texts, train_embeddings)
    test_features, _ = build_lexical_plus_embedding_features(test_cw, test_win_texts, test_embeddings)

    logger.info(
        "Feature vector: %d dimensions (%d train windows, %d test windows)",
        len(feature_names),
        len(train_features),
        len(test_features),
    )

    def make_inputs(win_texts: list[str], features: list[dict[str, float]], prefix: str) -> list[ClassificationInput]:
        return [
            ClassificationInput(
                source_id=f"{prefix}_win_{i}",
                source_type="window",
                text=win_texts[i],
                features=features[i],
            )
            for i in range(len(win_texts))
        ]

    train_inputs = make_inputs(train_win_texts, train_features, "train")
    test_inputs = make_inputs(test_win_texts, test_features, "test")

    # ------------------------------------------------------------------
    # Step 5: Train lexical+emb LightGBM (exact H2 best config)
    # ------------------------------------------------------------------
    logger.info("Training lexical+emb LightGBM (n_estimators=100, num_leaves=31, seed=%d)...", SEED)
    t0 = time.perf_counter()
    clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=feature_names,
        model_name="lightgbm",
        lgbm_kwargs={
            "n_estimators": 100,
            "num_leaves": 31,
            "verbosity": -1,
            "random_state": SEED,
        },
    )
    clf.fit(train_inputs, train_win_labels)
    train_duration_ms = (time.perf_counter() - t0) * 1000
    logger.info("Training done in %.1f ms", train_duration_ms)

    # ------------------------------------------------------------------
    # Step 6: Classify test windows and aggregate to conversations
    # ------------------------------------------------------------------
    logger.info("Classifying %d test windows...", len(test_inputs))
    t0 = time.perf_counter()
    test_window_preds = clf.classify(test_inputs)
    inference_duration_ms = (time.perf_counter() - t0) * 1000

    test_conv_results = aggregate_windows_to_conversations(test_window_preds, test_cw, unique_labels)
    test_conv_preds = [label for label, _ in test_conv_results]

    logger.info("Inference done in %.1f ms for %d conversations", inference_duration_ms, len(test_cw))

    # Overall metrics
    per_class_metrics = compute_per_class_metrics(test_conv_preds, test_conv_labels, unique_labels)
    f1_values = [m["f1"] for m in per_class_metrics.values()]
    macro_f1 = float(np.mean(f1_values))
    correct = sum(1 for p, g in zip(test_conv_preds, test_conv_labels, strict=True) if p == g)
    accuracy = correct / len(test_conv_labels) if test_conv_labels else 0.0
    logger.info("Test Macro-F1=%.4f  Accuracy=%.4f", macro_f1, accuracy)

    for label in INTENT_CLASSES:
        m = per_class_metrics.get(label, {})
        logger.info(
            "  %-20s  F1=%.3f  P=%.3f  R=%.3f  support=%d",
            label,
            m.get("f1", 0),
            m.get("precision", 0),
            m.get("recall", 0),
            m.get("support", 0),
        )

    # ------------------------------------------------------------------
    # Step 7: Confusion matrix
    # ------------------------------------------------------------------
    logger.info("Computing confusion matrix...")
    confusion = compute_confusion_matrix(test_conv_preds, test_conv_labels, unique_labels)
    confusion_pairs = compute_confusion_pairs(confusion, top_k=3)

    # ------------------------------------------------------------------
    # Step 8: Misclassified samples for weak classes
    # ------------------------------------------------------------------
    logger.info("Extracting misclassified samples for weak classes: %s", WEAK_CLASSES)
    misclassified_samples = extract_misclassified_samples(
        test_cw,
        test_conv_results,
        WEAK_CLASSES,
        max_per_class=MAX_MISCLASSIFIED_SAMPLES,
    )
    for cls in WEAK_CLASSES:
        logger.info("  %s: %d misclassified samples collected", cls, len(misclassified_samples[cls]))

    # ------------------------------------------------------------------
    # Step 9: Feature importances (LightGBM gain-based)
    # ------------------------------------------------------------------
    logger.info("Extracting feature importances (gain-based, top %d)...", TOP_N_FEATURES)
    feature_importances = extract_feature_importances(clf, feature_names, top_n=TOP_N_FEATURES)
    for family, info in feature_importances["by_family"].items():
        logger.info("  family=%-12s  gain_pct=%.1f%%", family, info["gain_pct"])

    # ------------------------------------------------------------------
    # Step 10: Class separability (embedding centroids)
    # ------------------------------------------------------------------
    logger.info("Computing class separability from test window embeddings...")
    separability = compute_class_separability(test_cw, test_win_texts, test_embeddings, unique_labels)
    for pair in separability.get("low_separability_pairs", [])[:3]:
        logger.info(
            "  Low separability: %s <-> %s  dist=%.4f",
            pair["class_a"],
            pair["class_b"],
            pair["cosine_distance"],
        )

    # ------------------------------------------------------------------
    # Step 11: Origin breakdown (synthetic vs original) per class
    # ------------------------------------------------------------------
    logger.info("Computing origin breakdown (synthetic vs original)...")
    origin_breakdown = compute_origin_breakdown(test_cw, test_conv_results, unique_labels)
    for origin in ("synthetic", "original"):
        agg = origin_breakdown["aggregate"].get(origin, {})
        logger.info(
            "  %-12s: accuracy=%.4f  n=%d",
            origin,
            agg.get("accuracy") or 0.0,
            agg.get("total", 0),
        )

    # ------------------------------------------------------------------
    # Step 12: Confidence analysis (correct vs incorrect predictions)
    # ------------------------------------------------------------------
    logger.info("Computing confidence analysis (correct vs incorrect)...")
    confidence_analysis = compute_confidence_analysis(test_cw, test_conv_results, unique_labels)
    overall_conf = confidence_analysis["overall"]
    if overall_conf.get("correct"):
        logger.info("  Correct predictions   mean confidence: %.4f", overall_conf["correct"]["mean"])
    if overall_conf.get("incorrect"):
        logger.info("  Incorrect predictions mean confidence: %.4f", overall_conf["incorrect"]["mean"])

    # ------------------------------------------------------------------
    # Step 13: Most-confused class pairs (symmetric)
    # ------------------------------------------------------------------
    logger.info("Computing most-confused class pairs...")
    most_confused_pairs = compute_most_confused_pairs(confusion, top_k=5)
    for pair in most_confused_pairs[:3]:
        logger.info(
            "  %s <-> %s: %d total errors (%d + %d)",
            pair["class_a"],
            pair["class_b"],
            pair["total_confusion"],
            pair["a_predicted_as_b"],
            pair["b_predicted_as_a"],
        )

    # ------------------------------------------------------------------
    # Step 14: Save results
    # ------------------------------------------------------------------
    logger.info("Saving results to %s...", output_dir)

    # confusion_matrix.json
    confusion_path = output_dir / "confusion_matrix.json"
    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(confusion, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", confusion_path)

    # error_analysis.json
    error_analysis_path = output_dir / "error_analysis.json"
    error_analysis_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {
            "name": "lexical+emb LightGBM",
            "n_estimators": 100,
            "num_leaves": 31,
            "seed": SEED,
            "embedding_model": EMBEDDING_MODEL,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
        },
        "dataset": {
            "splits_dir": str(splits_dir),
            "n_train_conversations": len(train_cw),
            "n_test_conversations": len(test_cw),
            "n_train_windows": len(train_inputs),
            "n_test_windows": len(test_inputs),
            "feature_count": len(feature_names),
        },
        "overall_metrics": {
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(accuracy, 4),
            "train_duration_ms": round(train_duration_ms, 1),
            "inference_duration_ms": round(inference_duration_ms, 1),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_pairs": confusion_pairs,
        "most_confused_pairs": most_confused_pairs,
        "misclassified_samples": misclassified_samples,
        "feature_importances": feature_importances,
        "class_separability": separability,
        "origin_breakdown": origin_breakdown,
        "confidence_analysis": confidence_analysis,
    }
    with open(error_analysis_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis_payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", error_analysis_path)

    # results.json — unified output (primary machine-readable target)
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis_payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", results_path)

    # summary.md
    write_summary_report(
        output_dir=output_dir,
        per_class_metrics=per_class_metrics,
        confusion_pairs=confusion_pairs,
        misclassified_samples=misclassified_samples,
        feature_importances=feature_importances,
        separability=separability,
        macro_f1=macro_f1,
        accuracy=accuracy,
        n_test_conversations=len(test_cw),
    )

    logger.info("Error analysis complete.")


if __name__ == "__main__":
    main()
