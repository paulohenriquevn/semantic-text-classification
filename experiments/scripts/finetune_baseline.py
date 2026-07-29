"""Fine-tune MiniLM on intent classification pairs and compare against frozen baseline.

Fine-tunes paraphrase-multilingual-MiniLM-L12-v2 using MultipleNegativesRankingLoss
with (anchor, positive) pairs sampled from the training set, then evaluates three
conditions in the standard TalkEx pipeline (context windows 5t/2s):

  A. Frozen encoder + LightGBM    — current H2 best, lexical+embedding features
  B. Fine-tuned encoder + LightGBM — same pipeline, fine-tuned embeddings
  C. Fine-tuned encoder + linear head — end-to-end torch classification head

All three conditions use identical context window construction and conversation
aggregation so differences are attributable solely to encoder and head choices.

Usage:
    python experiments/scripts/finetune_baseline.py
    python experiments/scripts/finetune_baseline.py --epochs 3 --batch-size 16 --lr 2e-5
    python experiments/scripts/finetune_baseline.py --output-dir experiments/results/finetune_comparison

Results are saved to experiments/results/finetune_comparison/results.json.

Notes:
    - No GPU required — runs on CPU. Expect ~30-90 min for 3 epochs on 1250 training records.
    - Fine-tuned model weights are saved to experiments/models/miniml-finetuned/.
    - Conditions A and B share identical LightGBM hyperparameters (n_estimators=100, num_leaves=31).
    - Condition C trains a torch.nn.Linear head (384 → n_classes) on top of fine-tuned embeddings.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
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
# Constants — mirror run_experiment.py where applicable
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_VERSION = "1.0"
FINETUNED_MODEL_VERSION = "finetuned-1.0"
EMBEDDING_DIMS = 384

SPLITS_DIR = Path("experiments/data")
RESULTS_DIR = Path("experiments/results/finetune_comparison")
MODELS_DIR = Path("experiments/models/miniml-finetuned")

# Linear-head training config (condition C)
LINEAR_HEAD_EPOCHS = 30
LINEAR_HEAD_LR = 1e-3
LINEAR_HEAD_BATCH_SIZE = 32

WINDOW_SIZE = 5
WINDOW_STRIDE = 2

# LightGBM config — identical in both conditions for a fair comparison
LGBM_KWARGS: dict[str, Any] = {
    "n_estimators": 100,
    "num_leaves": 31,
    "verbosity": -1,
    "random_state": 42,
}

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
# Data loading
# ---------------------------------------------------------------------------


def load_split(split_name: str, splits_dir: Path = SPLITS_DIR) -> list[dict]:
    """Load a JSONL split file.

    Args:
        split_name: One of 'train', 'val', 'test'.
        splits_dir: Directory containing the JSONL split files.

    Returns:
        List of record dicts.

    Raises:
        FileNotFoundError: If the split file does not exist.
    """
    path = splits_dir / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}. Run build_splits_v2.py first.")
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Windowed data — mirrors run_experiment.py _prepare_windowed_data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationWindows:
    """Context windows for a single conversation with its ground-truth label.

    Args:
        conversation_id: Source conversation identifier.
        label: Ground-truth intent label.
        windows: Ordered list of TalkEx ContextWindow objects.
    """

    conversation_id: str
    label: str
    windows: list[ContextWindow]


def _prepare_windowed_data(records: list[dict]) -> list[ConversationWindows]:
    """Segment conversations into turns and build context windows.

    Uses the real TalkEx pipeline (TurnSegmenter → SlidingWindowBuilder).
    Conversations without usable turns are silently skipped.

    Args:
        records: Raw records from a JSONL split.

    Returns:
        List of ConversationWindows, one per successfully parsed conversation.
    """
    result: list[ConversationWindows] = []
    for r in records:
        conv_id = r.get("conversation_id", r.get("id", "unknown"))
        label = r.get("topic", "unknown")
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
                    windows=windows,
                )
            )
    return result


def _flatten_windows(
    conv_windows: list[ConversationWindows],
) -> tuple[list[str], list[str], list[str]]:
    """Flatten windowed data into parallel lists.

    Args:
        conv_windows: Windowed conversations.

    Returns:
        Tuple of (window_texts, window_labels, conv_ids) aligned by index.
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


def _extract_structural_features(
    conv_windows: list[ConversationWindows],
) -> list[dict[str, float]]:
    """Extract structural features from ContextWindow metadata.

    Args:
        conv_windows: Windowed conversations.

    Returns:
        Flat list of feature dicts, one per window, aligned with _flatten_windows.
    """
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
# Training pair construction for MNRL
# ---------------------------------------------------------------------------


def build_training_pairs(
    records: list[dict],
    seed: int = 42,
) -> list[dict[str, str]]:
    """Build (anchor, positive) pairs from training conversations.

    For each record, one positive example is sampled from another record
    sharing the same intent label. Intent classes with fewer than 2 examples
    are skipped. Pairs are constructed using the full conversation text so
    that the fine-tuning signal reflects the same representation used at
    inference time.

    Args:
        records: Training records with 'text' and 'topic' fields.
        seed: Random seed for reproducible pair sampling.

    Returns:
        List of dicts with 'anchor' and 'positive' string keys,
        as expected by MultipleNegativesRankingLoss.
    """
    rng = random.Random(seed)

    # Group records by intent label
    label_to_texts: dict[str, list[str]] = {}
    for r in records:
        label = r.get("topic", "unknown")
        text = r.get("text", "").strip()
        if text:
            label_to_texts.setdefault(label, []).append(text)

    # Build one positive pair per record, sampling a different text of same intent
    pairs: list[dict[str, str]] = []
    for r in records:
        label = r.get("topic", "unknown")
        anchor_text = r.get("text", "").strip()
        if not anchor_text:
            continue
        candidates = [t for t in label_to_texts.get(label, []) if t != anchor_text]
        if not candidates:
            # Skip classes with a single example — cannot form a positive pair
            continue
        positive_text = rng.choice(candidates)
        pairs.append({"anchor": anchor_text, "positive": positive_text})

    logger.info(
        "Built %d training pairs from %d records (%d intent classes)",
        len(pairs),
        len(records),
        len(label_to_texts),
    )
    return pairs


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def finetune_minilm(
    train_records: list[dict],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42,
) -> str:
    """Fine-tune MiniLM using MultipleNegativesRankingLoss.

    Trains paraphrase-multilingual-MiniLM-L12-v2 on (anchor, positive) pairs
    derived from same-intent conversations. Uses the sentence-transformers v3
    SentenceTransformerTrainer API. Training runs on CPU.

    The fine-tuned model is saved to output_dir and also returned as a path
    string for loading into SentenceTransformerGenerator.

    Args:
        train_records: Training split records.
        output_dir: Directory to save the fine-tuned model weights.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        lr: Peak learning rate.
        seed: Random seed for pair sampling.

    Returns:
        Absolute path to the saved fine-tuned model directory.

    Raises:
        ImportError: If sentence-transformers or datasets are not installed.
        RuntimeError: If training produces no usable pairs.
    """
    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers>=3.0 and datasets are required for fine-tuning. "
            "Install with: pip install sentence-transformers datasets"
        ) from exc

    pairs = build_training_pairs(train_records, seed=seed)
    if not pairs:
        raise RuntimeError(
            "No training pairs could be constructed from the training records. "
            "Ensure the training split has at least 2 records per intent class."
        )

    logger.info(
        "Fine-tuning %s: epochs=%d, batch_size=%d, lr=%.2e, pairs=%d",
        EMBEDDING_MODEL,
        epochs,
        batch_size,
        lr,
        len(pairs),
    )

    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=False)
    loss = MultipleNegativesRankingLoss(model)

    train_dataset = Dataset.from_list(pairs)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=False,  # CPU only — fp16 is not supported
        bf16=False,
        no_cuda=True,
        dataloader_num_workers=0,  # Avoid multiprocessing overhead on CPU
        save_strategy="no",  # Save only the final model, not intermediate checkpoints
        logging_steps=50,
        seed=seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    logger.info("Starting fine-tuning (CPU — this may take a while)...")
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    logger.info("Fine-tuning complete in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    logger.info("Fine-tuned model saved to %s", output_dir)

    return str(output_dir.resolve())


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def _make_generator_from_path(model_path: str, model_version: str):
    """Create a SentenceTransformerGenerator from a model name or local path.

    Args:
        model_path: HuggingFace model name or local directory path.
        model_version: Version string for provenance tracking.

    Returns:
        Initialized SentenceTransformerGenerator.
    """
    from talkex.embeddings.config import EmbeddingModelConfig
    from talkex.embeddings.generator import SentenceTransformerGenerator

    model_config = EmbeddingModelConfig(
        model_name=model_path,
        model_version=model_version,
        batch_size=32,
    )
    return SentenceTransformerGenerator(model_config=model_config)


def generate_embeddings(
    texts: list[str],
    record_ids: list[str],
    generator: Any,
) -> np.ndarray:
    """Generate embeddings for a list of texts via the TalkEx pipeline.

    Args:
        texts: Input texts to embed.
        record_ids: Identifiers aligned with texts (used for EmbeddingInput identity).
        generator: Initialized SentenceTransformerGenerator.

    Returns:
        Float32 numpy array of shape (len(texts), embedding_dims).
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
# Feature construction (lexical + embedding)
# ---------------------------------------------------------------------------


def _build_lexical_features(
    conv_windows: list[ConversationWindows],
) -> list[dict[str, float]]:
    """Extract lexical + structural features for all windows.

    Args:
        conv_windows: Windowed conversations.

    Returns:
        Flat list of feature dicts, one per window.
    """
    struct_feats = _extract_structural_features(conv_windows)
    all_features: list[dict[str, float]] = []
    for idx, cw in enumerate(conv_windows):
        for w_idx, w in enumerate(cw.windows):
            lex = extract_lexical_features(w.window_text)
            flat_idx = sum(len(c.windows) for c in conv_windows[:idx]) + w_idx
            combined = {**lex.features, **struct_feats[flat_idx]}
            all_features.append(combined)
    return all_features


def _merge_features_with_embeddings(
    lex_features: list[dict[str, float]],
    embeddings: np.ndarray,
) -> tuple[list[dict[str, float]], list[str]]:
    """Combine lexical feature dicts with embedding dimensions.

    Args:
        lex_features: Lexical feature dicts, one per window.
        embeddings: Embedding matrix of shape (n_windows, dims).

    Returns:
        Tuple of (merged feature list, feature name list).
    """
    merged: list[dict[str, float]] = []
    for i, lex in enumerate(lex_features):
        combined = dict(lex)
        for d in range(embeddings.shape[1]):
            combined[f"emb_{d}"] = float(embeddings[i][d])
        merged.append(combined)
    feature_names = list(merged[0].keys()) if merged else []
    return merged, feature_names


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _aggregate_windows_to_conversations(
    window_preds: list[Any],
    conv_windows: list[ConversationWindows],
    labels: list[str],
) -> list[str]:
    """Aggregate window-level predictions to conversation-level labels.

    Uses average class probabilities across windows, then argmax.

    Args:
        window_preds: Flat list of ClassificationResult, one per window,
            in the same order produced by _flatten_windows.
        conv_windows: Original windowed data for conversation grouping.
        labels: All possible labels.

    Returns:
        List of predicted labels, one per conversation.
    """
    conv_preds: list[str] = []
    offset = 0
    for cw in conv_windows:
        n_win = len(cw.windows)
        window_results = window_preds[offset : offset + n_win]

        if not window_results:
            conv_preds.append(labels[0] if labels else "unknown")
            offset += n_win
            continue

        class_probs: dict[str, float] = {label: 0.0 for label in labels}
        for result in window_results:
            for ls in result.label_scores:
                if ls.label in class_probs:
                    class_probs[ls.label] += ls.score

        for label in class_probs:
            class_probs[label] /= n_win

        best_label = max(class_probs, key=lambda lbl: class_probs[lbl])
        conv_preds.append(best_label)
        offset += n_win

    return conv_preds


def _compute_metrics(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """Compute macro-F1 and per-class F1 scores.

    Args:
        predictions: Predicted labels.
        ground_truth: True labels.
        labels: All possible labels (used to define per-class metrics).

    Returns:
        Dict with 'macro_f1', 'accuracy', and per-class 'f1_<label>' entries.
    """
    metrics: dict[str, Any] = {}
    f1_scores: list[float] = []

    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p != label and g == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        metrics[f"f1_{label}"] = round(f1, 4)

    metrics["macro_f1"] = round(float(np.mean(f1_scores)), 4)
    correct = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == g)
    metrics["accuracy"] = round(correct / len(ground_truth), 4) if ground_truth else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Single evaluation pass (frozen or fine-tuned)
# ---------------------------------------------------------------------------


def evaluate_condition(
    model_path: str,
    model_version: str,
    train_cw: list[ConversationWindows],
    val_cw: list[ConversationWindows],
    test_cw: list[ConversationWindows],
    unique_labels: list[str],
    condition_label: str,
) -> dict[str, Any]:
    """Train LightGBM on lexical+embedding features and evaluate on test set.

    Generates embeddings with the specified model, merges them with lexical
    features, trains LightGBM, aggregates window predictions to conversations,
    and returns metrics.

    Args:
        model_path: HuggingFace model name or local path to fine-tuned model.
        model_version: Version string used for embedding provenance.
        train_cw: Training conversations as ConversationWindows.
        val_cw: Validation conversations (used only for logging).
        test_cw: Test conversations for final evaluation.
        unique_labels: Sorted list of all intent labels.
        condition_label: Human-readable label for logging ('frozen' or 'finetuned').

    Returns:
        Dict with macro_f1, per_class_f1, accuracy, and timing.
    """
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.models import ClassificationInput

    logger.info("--- Evaluating condition: %s ---", condition_label)

    # Flatten window texts
    train_win_texts, train_win_labels, _ = _flatten_windows(train_cw)
    val_win_texts, _, _ = _flatten_windows(val_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    # Extract lexical features (independent of embedding model)
    logger.info("Extracting lexical features...")
    train_lex = _build_lexical_features(train_cw)
    val_lex = _build_lexical_features(val_cw)
    test_lex = _build_lexical_features(test_cw)

    # Generate embeddings
    logger.info("Generating embeddings with model: %s...", model_path)
    generator = _make_generator_from_path(model_path, model_version)

    train_ids = [f"{condition_label}_train_win_{i}" for i in range(len(train_win_texts))]
    val_ids = [f"{condition_label}_val_win_{i}" for i in range(len(val_win_texts))]
    test_ids = [f"{condition_label}_test_win_{i}" for i in range(len(test_win_texts))]

    train_emb = generate_embeddings(train_win_texts, train_ids, generator)
    val_emb = generate_embeddings(val_win_texts, val_ids, generator)
    test_emb = generate_embeddings(test_win_texts, test_ids, generator)

    logger.info(
        "Embeddings: train=%d, val=%d, test=%d, dims=%d",
        len(train_win_texts),
        len(val_win_texts),
        len(test_win_texts),
        train_emb.shape[1],
    )

    # Merge features
    train_full, feat_names = _merge_features_with_embeddings(train_lex, train_emb)
    val_full, _ = _merge_features_with_embeddings(val_lex, val_emb)
    test_full, _ = _merge_features_with_embeddings(test_lex, test_emb)

    # Build ClassificationInput objects
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    train_inputs = [
        ClassificationInput(
            source_id=f"{condition_label}_train_{i}",
            source_type="window",
            text=train_win_texts[i],
            features=train_full[i],
        )
        for i in range(len(train_win_texts))
    ]
    val_inputs = [
        ClassificationInput(
            source_id=f"{condition_label}_val_{i}",
            source_type="window",
            text=val_win_texts[i],
            features=val_full[i],
        )
        for i in range(len(val_win_texts))
    ]
    test_inputs = [
        ClassificationInput(
            source_id=f"{condition_label}_test_{i}",
            source_type="window",
            text=test_win_texts[i],
            features=test_full[i],
        )
        for i in range(len(test_win_texts))
    ]

    # Train LightGBM
    logger.info("Training LightGBM on %d windows...", len(train_inputs))
    t0 = time.perf_counter()
    clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=feat_names,
        model_name=f"lightgbm-{condition_label}",
        lgbm_kwargs=LGBM_KWARGS,
    )
    clf.fit(train_inputs, train_win_labels)

    # Validation aggregation (for logging only)
    val_window_preds = clf.classify(val_inputs)
    val_conv_labels = [cw.label for cw in val_cw]
    val_conv_preds = _aggregate_windows_to_conversations(val_window_preds, val_cw, unique_labels)
    val_metrics = _compute_metrics(val_conv_preds, val_conv_labels, unique_labels)
    logger.info(
        "[VAL] %s: macro_f1=%.4f, accuracy=%.4f",
        condition_label,
        val_metrics["macro_f1"],
        val_metrics["accuracy"],
    )

    # Test aggregation
    test_window_preds = clf.classify(test_inputs)
    test_conv_labels = [cw.label for cw in test_cw]
    test_conv_preds = _aggregate_windows_to_conversations(test_window_preds, test_cw, unique_labels)
    elapsed = time.perf_counter() - t0

    test_metrics = _compute_metrics(test_conv_preds, test_conv_labels, unique_labels)
    logger.info(
        "[TEST] %s: macro_f1=%.4f, accuracy=%.4f (train+eval: %.1fs)",
        condition_label,
        test_metrics["macro_f1"],
        test_metrics["accuracy"],
        elapsed,
    )

    per_class_f1 = {label: test_metrics[f"f1_{label}"] for label in unique_labels if f"f1_{label}" in test_metrics}

    return {
        "macro_f1": test_metrics["macro_f1"],
        "accuracy": test_metrics["accuracy"],
        "per_class_f1": per_class_f1,
        "val_macro_f1": val_metrics["macro_f1"],
        "train_eval_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Condition C: fine-tuned encoder + linear head (end-to-end)
# ---------------------------------------------------------------------------


def evaluate_linear_head(
    model_path: str,
    train_cw: list[ConversationWindows],
    val_cw: list[ConversationWindows],
    test_cw: list[ConversationWindows],
    unique_labels: list[str],
    head_epochs: int = LINEAR_HEAD_EPOCHS,
    head_lr: float = LINEAR_HEAD_LR,
    head_batch_size: int = LINEAR_HEAD_BATCH_SIZE,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a linear classification head on frozen fine-tuned embeddings.

    Embeds all context windows using the fine-tuned model (encode-then-freeze),
    then trains a torch.nn.Linear layer via cross-entropy on the window-level
    labels. Predictions are aggregated to conversation level via argmax on
    averaged class probabilities.

    This condition isolates whether a linear probe on top of fine-tuned
    representations is competitive with LightGBM on raw features.

    Args:
        model_path: Local path to the fine-tuned SentenceTransformer model.
        train_cw: Training conversations as ConversationWindows.
        val_cw: Validation conversations.
        test_cw: Test conversations for final evaluation.
        unique_labels: Sorted list of all intent labels.
        head_epochs: Number of epochs to train the linear head.
        head_lr: Learning rate for the linear head optimizer.
        head_batch_size: Batch size for head training.
        seed: Random seed for reproducibility.

    Returns:
        Dict with macro_f1, accuracy, per_class_f1, val_macro_f1, and timing.

    Raises:
        ImportError: If torch is not installed.
        RuntimeError: If the training data produces no usable windows.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("torch is required for linear head evaluation. Install with: pip install torch") from exc

    logger.info("--- Evaluating condition C: fine-tuned encoder + linear head ---")

    torch.manual_seed(seed)

    n_classes = len(unique_labels)
    label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}

    # Flatten window texts and labels
    train_win_texts, train_win_labels, _ = _flatten_windows(train_cw)
    val_win_texts, _, _ = _flatten_windows(val_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    if not train_win_texts:
        raise RuntimeError(
            "No training windows available for linear head training. "
            "Ensure the training split has non-empty conversations."
        )

    # Generate embeddings using the fine-tuned model (frozen for head training)
    logger.info("Generating embeddings with fine-tuned model for linear head...")
    generator = _make_generator_from_path(model_path, FINETUNED_MODEL_VERSION)

    train_emb = generate_embeddings(
        train_win_texts,
        [f"lh_train_{i}" for i in range(len(train_win_texts))],
        generator,
    )
    val_emb = generate_embeddings(
        val_win_texts,
        [f"lh_val_{i}" for i in range(len(val_win_texts))],
        generator,
    )
    test_emb = generate_embeddings(
        test_win_texts,
        [f"lh_test_{i}" for i in range(len(test_win_texts))],
        generator,
    )
    emb_dim = train_emb.shape[1]
    logger.info(
        "Embeddings ready: train=%d, val=%d, test=%d, dims=%d",
        len(train_win_texts),
        len(val_win_texts),
        len(test_win_texts),
        emb_dim,
    )

    # Build tensors
    X_train = torch.from_numpy(train_emb)
    y_train = torch.tensor(
        [label_to_idx[lbl] for lbl in train_win_labels],
        dtype=torch.long,
    )
    X_val = torch.from_numpy(val_emb)
    X_test = torch.from_numpy(test_emb)

    # Define and train the linear head
    head = nn.Linear(emb_dim, n_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=head_lr)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(
        "Training linear head: epochs=%d, lr=%.1e, batch_size=%d, classes=%d",
        head_epochs,
        head_lr,
        head_batch_size,
        n_classes,
    )

    t0 = time.perf_counter()
    head.train()
    best_val_f1 = -1.0
    best_head_state: dict[str, Any] = {}

    for epoch in range(head_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = head(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate at the end of each epoch for best model selection
        head.eval()
        with torch.no_grad():
            val_logits = head(X_val)
            val_probs = torch.softmax(val_logits, dim=1).numpy()

        val_preds = _aggregate_probs_to_conversations(val_probs, val_cw, unique_labels)
        val_conv_labels = [cw.label for cw in val_cw]
        val_metrics = _compute_metrics(val_preds, val_conv_labels, unique_labels)
        val_f1 = val_metrics["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "  epoch %d/%d: loss=%.4f, val_macro_f1=%.4f",
                epoch + 1,
                head_epochs,
                epoch_loss / max(len(train_loader), 1),
                val_f1,
            )
        head.train()

    # Restore best checkpoint (selected on val)
    if best_head_state:
        head.load_state_dict(best_head_state)

    # Test evaluation
    head.eval()
    with torch.no_grad():
        test_logits = head(X_test)
        test_probs = torch.softmax(test_logits, dim=1).numpy()

    test_conv_labels = [cw.label for cw in test_cw]
    test_preds = _aggregate_probs_to_conversations(test_probs, test_cw, unique_labels)
    elapsed = time.perf_counter() - t0

    test_metrics = _compute_metrics(test_preds, test_conv_labels, unique_labels)
    logger.info(
        "[TEST] linear-head: macro_f1=%.4f, accuracy=%.4f (total: %.1fs)",
        test_metrics["macro_f1"],
        test_metrics["accuracy"],
        elapsed,
    )

    per_class_f1 = {label: test_metrics[f"f1_{label}"] for label in unique_labels if f"f1_{label}" in test_metrics}

    return {
        "macro_f1": test_metrics["macro_f1"],
        "accuracy": test_metrics["accuracy"],
        "per_class_f1": per_class_f1,
        "val_macro_f1": best_val_f1,
        "train_eval_seconds": round(elapsed, 1),
    }


def _aggregate_probs_to_conversations(
    window_probs: np.ndarray,
    conv_windows: list[ConversationWindows],
    unique_labels: list[str],
) -> list[str]:
    """Aggregate per-window softmax probabilities to conversation predictions.

    Averages class probabilities across all windows of a conversation,
    then takes argmax.

    Args:
        window_probs: Array of shape (n_windows, n_classes) with softmax probs.
        conv_windows: Windowed conversations providing grouping structure.
        unique_labels: Ordered list of class labels (index aligns with prob columns).

    Returns:
        List of predicted labels, one per conversation.
    """
    conv_preds: list[str] = []
    offset = 0
    for cw in conv_windows:
        n_win = len(cw.windows)
        if n_win == 0:
            conv_preds.append(unique_labels[0] if unique_labels else "unknown")
            continue
        window_slice = window_probs[offset : offset + n_win]  # (n_win, n_classes)
        avg_probs = window_slice.mean(axis=0)  # (n_classes,)
        best_idx = int(np.argmax(avg_probs))
        conv_preds.append(unique_labels[best_idx])
        offset += n_win
    return conv_preds


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def save_results(
    frozen_results: dict[str, Any],
    finetuned_results: dict[str, Any],
    linear_head_results: dict[str, Any],
    finetune_config: dict[str, Any],
    splits_dir: Path,
    output_dir: Path,
) -> None:
    """Save three-way comparison results to JSON.

    Writes a structured JSON file with results for all three conditions
    (frozen + LightGBM, fine-tuned + LightGBM, fine-tuned + linear head)
    along with a human-readable conclusion about the best-performing setup.

    Args:
        frozen_results: Evaluation metrics for condition A (frozen + LightGBM).
        finetuned_results: Evaluation metrics for condition B (fine-tuned + LightGBM).
        linear_head_results: Evaluation metrics for condition C (fine-tuned + linear head).
        finetune_config: Fine-tuning hyperparameters used for conditions B and C.
        splits_dir: Path to the directory containing the JSONL split files.
        output_dir: Directory where results.json will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deltas relative to frozen baseline (condition A)
    delta_b = round(finetuned_results["macro_f1"] - frozen_results["macro_f1"], 4)
    delta_c = round(linear_head_results["macro_f1"] - frozen_results["macro_f1"], 4)
    all_f1 = {
        "frozen_lgbm": frozen_results["macro_f1"],
        "finetuned_lgbm": finetuned_results["macro_f1"],
        "finetuned_linear": linear_head_results["macro_f1"],
    }
    best_condition = max(all_f1, key=lambda k: all_f1[k])
    best_f1 = all_f1[best_condition]

    def _delta_summary(delta: float, a_f1: float, b_f1: float, condition_name: str) -> str:
        if delta > 0.005:
            return f"{condition_name} improved Macro-F1 by {delta:+.4f} ({a_f1:.4f} → {b_f1:.4f})."
        if delta < -0.005:
            return f"{condition_name} degraded Macro-F1 by {delta:+.4f} ({a_f1:.4f} → {b_f1:.4f})."
        return f"{condition_name} produced negligible change: delta={delta:+.4f} ({a_f1:.4f} → {b_f1:.4f})."

    conclusion = (
        f"Best condition: {best_condition} (Macro-F1={best_f1:.4f}). "
        + _delta_summary(delta_b, frozen_results["macro_f1"], finetuned_results["macro_f1"], "Fine-tuned + LightGBM")
        + " "
        + _delta_summary(
            delta_c, frozen_results["macro_f1"], linear_head_results["macro_f1"], "Fine-tuned + linear head"
        )
    )

    payload = {
        "experiment": "finetune_comparison",
        "timestamp": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "train": str(splits_dir / "train.jsonl"),
            "val": str(splits_dir / "val.jsonl"),
            "test": str(splits_dir / "test.jsonl"),
        },
        "pipeline": {
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "aggregation": "avg_probs_argmax",
        },
        "condition_A_frozen_lgbm": {
            "description": "Frozen encoder + LightGBM (current H2 best)",
            "model": EMBEDDING_MODEL,
            "model_version": EMBEDDING_VERSION,
            "classifier": "LightGBM",
            "lgbm_kwargs": LGBM_KWARGS,
            "features": "lexical+structural+embeddings",
            "macro_f1": frozen_results["macro_f1"],
            "accuracy": frozen_results["accuracy"],
            "val_macro_f1": frozen_results["val_macro_f1"],
            "per_class_f1": frozen_results["per_class_f1"],
            "train_eval_seconds": frozen_results["train_eval_seconds"],
        },
        "condition_B_finetuned_lgbm": {
            "description": "Fine-tuned encoder + LightGBM (same pipeline, adapted embeddings)",
            "model": EMBEDDING_MODEL,
            "model_path": finetune_config["model_path"],
            "model_version": FINETUNED_MODEL_VERSION,
            "classifier": "LightGBM",
            "lgbm_kwargs": LGBM_KWARGS,
            "features": "lexical+structural+embeddings",
            "finetune_training": {
                "loss": "MultipleNegativesRankingLoss",
                "epochs": finetune_config["epochs"],
                "batch_size": finetune_config["batch_size"],
                "lr": finetune_config["lr"],
                "warmup_ratio": 0.1,
                "n_pairs": finetune_config["n_pairs"],
                "finetune_seconds": finetune_config["finetune_seconds"],
            },
            "macro_f1": finetuned_results["macro_f1"],
            "accuracy": finetuned_results["accuracy"],
            "val_macro_f1": finetuned_results["val_macro_f1"],
            "per_class_f1": finetuned_results["per_class_f1"],
            "train_eval_seconds": finetuned_results["train_eval_seconds"],
            "delta_vs_frozen": delta_b,
        },
        "condition_C_finetuned_linear": {
            "description": "Fine-tuned encoder + linear head (end-to-end classification)",
            "model": EMBEDDING_MODEL,
            "model_path": finetune_config["model_path"],
            "model_version": FINETUNED_MODEL_VERSION,
            "classifier": "torch.nn.Linear",
            "head_epochs": LINEAR_HEAD_EPOCHS,
            "head_lr": LINEAR_HEAD_LR,
            "head_batch_size": LINEAR_HEAD_BATCH_SIZE,
            "finetune_training": {
                "loss": "MultipleNegativesRankingLoss",
                "epochs": finetune_config["epochs"],
                "batch_size": finetune_config["batch_size"],
                "lr": finetune_config["lr"],
                "warmup_ratio": 0.1,
                "n_pairs": finetune_config["n_pairs"],
                "finetune_seconds": finetune_config["finetune_seconds"],
            },
            "macro_f1": linear_head_results["macro_f1"],
            "accuracy": linear_head_results["accuracy"],
            "val_macro_f1": linear_head_results["val_macro_f1"],
            "per_class_f1": linear_head_results["per_class_f1"],
            "train_eval_seconds": linear_head_results["train_eval_seconds"],
            "delta_vs_frozen": delta_c,
        },
        "best_condition": best_condition,
        "best_macro_f1": best_f1,
        "conclusion": conclusion,
    }

    output_path = output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", output_path)
    logger.info("Condition A (frozen + LightGBM):  Macro-F1=%.4f", frozen_results["macro_f1"])
    logger.info(
        "Condition B (finetuned + LightGBM): Macro-F1=%.4f (delta=%+.4f)", finetuned_results["macro_f1"], delta_b
    )
    logger.info(
        "Condition C (finetuned + linear):  Macro-F1=%.4f (delta=%+.4f)", linear_head_results["macro_f1"], delta_c
    )
    logger.info("Best: %s (%.4f)", best_condition, best_f1)
    logger.info("Conclusion: %s", conclusion)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--splits-dir",
    default=str(SPLITS_DIR),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing train.jsonl, val.jsonl, test.jsonl.",
)
@click.option(
    "--epochs",
    default=3,
    show_default=True,
    type=int,
    help="Number of fine-tuning epochs (encoder fine-tuning via MNRL).",
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    type=int,
    help="Per-device training batch size for encoder fine-tuning (CPU only).",
)
@click.option(
    "--lr",
    default=2e-5,
    show_default=True,
    type=float,
    help="Peak learning rate for encoder fine-tuning.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for pair sampling, LightGBM, and linear head.",
)
@click.option(
    "--output-dir",
    default=str(RESULTS_DIR),
    show_default=True,
    type=click.Path(),
    help="Directory to write results.json.",
)
@click.option(
    "--model-dir",
    default=str(MODELS_DIR),
    show_default=True,
    type=click.Path(),
    help="Directory to save fine-tuned model weights.",
)
@click.option(
    "--skip-finetune",
    is_flag=True,
    default=False,
    help="Skip encoder fine-tuning and load an existing model from --model-dir. "
    "Useful to re-run evaluations on a previously trained model.",
)
def main(
    splits_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    output_dir: str,
    model_dir: str,
    skip_finetune: bool,
) -> None:
    """Fine-tune MiniLM and compare frozen vs fine-tuned across three conditions.

    Runs three conditions in sequence:
    \b
      A. Frozen encoder + LightGBM      — current H2 best (lexical+emb features)
      B. Fine-tuned encoder + LightGBM  — same pipeline, adapted embeddings
      C. Fine-tuned encoder + linear head — end-to-end torch classification

    All conditions use context windows (5 turns, stride 2) and conversation-level
    aggregation via average class probabilities → argmax. Results are saved to
    --output-dir/results.json.
    """
    output_path = Path(output_dir)
    model_path = Path(model_dir)

    logger.info("=" * 60)
    logger.info("MiniLM Fine-tune vs Frozen — Three-Condition Comparison")
    logger.info("=" * 60)
    logger.info("splits_dir=%s", splits_dir)
    logger.info(
        "Config: epochs=%d, batch_size=%d, lr=%.2e, seed=%d",
        epochs,
        batch_size,
        lr,
        seed,
    )

    # --- Load splits ---
    train_records = load_split("train", splits_dir)
    val_records = load_split("val", splits_dir)
    test_records = load_split("test", splits_dir)

    # --- Build windowed data (shared across all three conditions) ---
    logger.info("Building context windows from splits...")
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

    # Derive label space from all splits
    all_labels = [cw.label for cw in train_cw] + [cw.label for cw in val_cw] + [cw.label for cw in test_cw]
    unique_labels = sorted(set(all_labels))
    logger.info("Label space (%d classes): %s", len(unique_labels), unique_labels)

    # --- Condition A: Frozen encoder + LightGBM ---
    logger.info("")
    logger.info("=== Condition A: Frozen encoder + LightGBM ===")
    frozen_results = evaluate_condition(
        model_path=EMBEDDING_MODEL,
        model_version=EMBEDDING_VERSION,
        train_cw=train_cw,
        val_cw=val_cw,
        test_cw=test_cw,
        unique_labels=unique_labels,
        condition_label="frozen",
    )

    # --- Fine-tune encoder (shared by conditions B and C) ---
    logger.info("")
    logger.info("=== Encoder fine-tuning (MultipleNegativesRankingLoss) ===")

    finetune_config: dict[str, Any]

    if skip_finetune:
        if not model_path.exists():
            raise FileNotFoundError(
                f"--skip-finetune specified but model directory not found: {model_path}. "
                "Run without --skip-finetune first to train the model."
            )
        logger.info("Skipping fine-tuning, loading existing model from %s", model_path)
        finetune_config = {
            "model_path": str(model_path.resolve()),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_pairs": -1,  # Unknown when skipping
            "finetune_seconds": 0.0,
        }
    else:
        pairs_preview = build_training_pairs(train_records, seed=seed)
        t_ft_start = time.perf_counter()
        saved_model_path = finetune_minilm(
            train_records=train_records,
            output_dir=model_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )
        finetune_seconds = time.perf_counter() - t_ft_start
        finetune_config = {
            "model_path": saved_model_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_pairs": len(pairs_preview),
            "finetune_seconds": round(finetune_seconds, 1),
        }

    # --- Condition B: Fine-tuned encoder + LightGBM ---
    logger.info("")
    logger.info("=== Condition B: Fine-tuned encoder + LightGBM ===")
    finetuned_results = evaluate_condition(
        model_path=finetune_config["model_path"],
        model_version=FINETUNED_MODEL_VERSION,
        train_cw=train_cw,
        val_cw=val_cw,
        test_cw=test_cw,
        unique_labels=unique_labels,
        condition_label="finetuned",
    )

    # --- Condition C: Fine-tuned encoder + linear head ---
    logger.info("")
    logger.info("=== Condition C: Fine-tuned encoder + linear head ===")
    linear_head_results = evaluate_linear_head(
        model_path=finetune_config["model_path"],
        train_cw=train_cw,
        val_cw=val_cw,
        test_cw=test_cw,
        unique_labels=unique_labels,
        seed=seed,
    )

    # --- Save results ---
    logger.info("")
    logger.info("=== Saving results ===")
    save_results(
        frozen_results=frozen_results,
        finetuned_results=finetuned_results,
        linear_head_results=linear_head_results,
        finetune_config=finetune_config,
        splits_dir=splits_dir,
        output_dir=output_path,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
