"""Experiment orchestration script for dissertation hypotheses H1-H4.

Loads data from splits, runs configured experiment variants, computes
metrics, applies statistical tests, and saves structured results.

Usage:
    python experiments/scripts/run_experiment.py --hypothesis H1
    python experiments/scripts/run_experiment.py --hypothesis H2
    python experiments/scripts/run_experiment.py --hypothesis H3
    python experiments/scripts/run_experiment.py --hypothesis H4
    python experiments/scripts/run_experiment.py --hypothesis ablation

Each hypothesis produces results in experiments/results/H{1,2,3,4}/.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
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

SPLITS_DIR = Path("experiments/data")
RESULTS_DIR = Path("experiments/results")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_VERSION = "1.0"
CURRENT_SEED: int = 42  # Global seed, overridden by multi-seed loop


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def load_split(split_name: str, splits_dir: Path = SPLITS_DIR) -> list[dict]:
    """Load a JSONL split file."""
    path = splits_dir / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}. Run build_splits.py first.")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def extract_texts(records: list[dict]) -> list[str]:
    """Extract text from conversation records.

    Supports both flat format (text field) and structured format (conversation turns).
    """
    texts = []
    for r in records:
        # Flat format: text field contains full conversation
        if "text" in r and isinstance(r["text"], str) and r["text"]:
            texts.append(r["text"])
        else:
            # Structured format: conversation field with turn dicts
            turns = r.get("conversation", [])
            text = " ".join(t.get("text", "") for t in turns if isinstance(t, dict))
            texts.append(text)
    return texts


def extract_labels(records: list[dict]) -> list[str]:
    """Extract intent labels from conversation records."""
    return [r.get("topic", "unknown") for r in records]


def _get_record_id(record: dict, fallback: str) -> str:
    """Get conversation ID from a record, supporting both formats."""
    return record.get("conversation_id", record.get("id", fallback))


# ---------------------------------------------------------------------------
# Context window utilities — uses real TalkEx pipeline modules
# ---------------------------------------------------------------------------

# Window configuration: matches ContextWindowConfig defaults in src/talkex/context/config.py
WINDOW_SIZE = 5
WINDOW_STRIDE = 2

# Reusable pipeline instances (stateless, safe to share)
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


@dataclass(frozen=True)
class ConversationWindows:
    """All windows for a single conversation, with its ground-truth label.

    Wraps real TalkEx ContextWindow objects with the experiment-specific
    ground-truth label (which is not part of the TalkEx domain model).
    """

    conversation_id: str
    label: str
    windows: list[ContextWindow]


def _prepare_windowed_data(
    records: list[dict],
) -> list[ConversationWindows]:
    """Parse conversations into turns and build context windows using the real TalkEx pipeline.

    Uses TurnSegmenter → SlidingWindowBuilder — the same modules that run in production.
    Each conversation produces N windows (N depends on turn count).
    All windows inherit the conversation's intent label for training/evaluation.
    The split is by conversation (not by window), preserving data integrity.
    """
    result: list[ConversationWindows] = []
    for r in records:
        conv_id = _get_record_id(r, "unknown")
        label = r.get("topic", "unknown")
        text = r.get("text", "")
        if not text.strip():
            continue

        # Step 1: Segment raw text into Turn objects via TalkEx
        transcript = TranscriptInput(
            conversation_id=ConversationId(conv_id),
            channel=Channel.CHAT,
            raw_text=text,
            source_format=SourceFormat.LABELED,
        )
        turns = _SEGMENTER.segment(transcript, _SEG_CONFIG)
        if not turns:
            continue

        # Step 2: Build context windows via TalkEx
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
    """Flatten windowed data into parallel lists for training/classification.

    Returns:
        (window_texts, window_labels, window_conv_ids) — all aligned by index.
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
    """Extract structural features from real TalkEx ContextWindow metadata.

    Uses talkex.classification.features.extract_structural_features with
    metadata populated by SlidingWindowBuilder → compute_window_metrics.
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


def _aggregate_window_predictions(
    window_results: list,  # list of ClassificationResult
    labels: list[str],
) -> tuple[str, float]:
    """Aggregate window-level predictions to conversation level.

    Strategy: average class probabilities across windows, then argmax.
    Tiebreak: majority vote, then max confidence.

    Args:
        window_results: ClassificationResult objects for each window.
        labels: All possible labels (for consistent aggregation).

    Returns:
        (predicted_label, confidence_score) for the conversation.
    """
    if not window_results:
        return labels[0] if labels else "unknown", 0.0

    # Average probabilities per class across all windows
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


# ---------------------------------------------------------------------------
# Embedding generation via talkex pipeline
# ---------------------------------------------------------------------------


def _make_embedding_generator():  # -> SentenceTransformerGenerator
    """Create a SentenceTransformerGenerator using the talkex pipeline.

    Returns a single reusable generator instance with proper model config.
    """
    from talkex.embeddings.config import EmbeddingModelConfig
    from talkex.embeddings.generator import SentenceTransformerGenerator

    model_config = EmbeddingModelConfig(
        model_name=EMBEDDING_MODEL,
        model_version=EMBEDDING_VERSION,
        batch_size=64,
    )
    return SentenceTransformerGenerator(model_config=model_config)


def generate_embeddings_via_talkex(
    texts: list[str],
    record_ids: list[str],
    generator,  # SentenceTransformerGenerator
) -> np.ndarray:
    """Generate embeddings using the talkex SentenceTransformerGenerator.

    Goes through the full talkex pipeline: preprocessing, encoding,
    pooling, L2 normalization, and provenance tracking.

    Args:
        texts: List of texts to embed.
        record_ids: Corresponding record IDs (for EmbeddingInput identity).
        generator: Pre-initialized SentenceTransformerGenerator.

    Returns:
        numpy array of shape (len(texts), embedding_dims).
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
# Result structures
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Structured result for one experiment variant."""

    hypothesis: str
    variant_name: str
    metrics: dict[str, float]
    config: dict[str, Any]
    duration_ms: float
    per_query_scores: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_results(results: list[ExperimentResult], output_dir: Path) -> None:
    """Save experiment results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    data = {
        "hypothesis": results[0].hypothesis if results else "unknown",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_variants": len(results),
        "variants": [r.to_dict() for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)

    # Also save summary table
    summary_path = output_dir / "summary.md"
    _write_summary_table(results, summary_path)
    logger.info("Summary table saved to %s", summary_path)


def _write_summary_table(results: list[ExperimentResult], path: Path) -> None:
    """Write markdown summary table."""
    if not results:
        return
    all_metric_keys = sorted({k for r in results for k in r.metrics})
    header = "| Variant | " + " | ".join(all_metric_keys) + " | Duration (ms) |"
    separator = "|" + "|".join(["---"] * (len(all_metric_keys) + 2)) + "|"

    lines = [f"# {results[0].hypothesis} Results", "", header, separator]
    for r in results:
        values = " | ".join(f"{r.metrics.get(k, 0):.4f}" for k in all_metric_keys)
        lines.append(f"| {r.variant_name} | {values} | {r.duration_ms:.1f} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# H1: Hybrid Retrieval
# ---------------------------------------------------------------------------


def run_h1(train: list[dict], val: list[dict], test: list[dict]) -> list[ExperimentResult]:
    """Run H1 experiments: hybrid retrieval superiority.

    Compares BM25-base, BM25-norm, ANN (NullEmbedding), Hybrid-linear, Hybrid-RRF
    on retrieval metrics (Recall@K, MRR, nDCG).

    Fusion weight alpha for Hybrid-LINEAR is tuned on the validation set.
    Final metrics are reported on the held-out test set only.

    Ground truth: for each test query, relevant documents are training
    conversations with the same intent label.
    """
    from talkex.embeddings.config import EmbeddingModelConfig
    from talkex.embeddings.generator import SentenceTransformerGenerator
    from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
    from talkex.models.enums import ObjectType
    from talkex.retrieval.bm25 import InMemoryBM25Index
    from talkex.retrieval.config import (
        FusionStrategy,
        HybridRetrievalConfig,
        LexicalIndexConfig,
        VectorIndexConfig,
    )
    from talkex.retrieval.hybrid import SimpleHybridRetriever
    from talkex.retrieval.models import QueryType, RetrievalQuery
    from talkex.retrieval.vector_index import InMemoryVectorIndex
    from talkex.text_normalization import normalize_for_matching

    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    logger.info("=" * 60)
    logger.info("H1: Hybrid Retrieval Experiment")
    logger.info("=" * 60)

    # Build document corpus from training data
    train_texts = extract_texts(train)
    train_labels = extract_labels(train)
    train_ids = [r.get("conversation_id", r.get("id", f"train_{i}")) for i, r in enumerate(train)]

    val_texts = extract_texts(val)
    val_labels = extract_labels(val)
    val_ids = [r.get("conversation_id", r.get("id", f"val_{i}")) for i, r in enumerate(val)]

    test_texts = extract_texts(test)
    test_labels = extract_labels(test)
    test_ids = [r.get("conversation_id", r.get("id", f"test_{i}")) for i, r in enumerate(test)]

    # Ground truth: relevant docs share the same intent label
    intent_to_train_ids: dict[str, list[str]] = {}
    for tid, label in zip(train_ids, train_labels, strict=True):
        intent_to_train_ids.setdefault(label, []).append(tid)

    results: list[ExperimentResult] = []

    # --- BM25-base ---
    logger.info("Running BM25-base...")
    bm25_base = InMemoryBM25Index(config=LexicalIndexConfig())
    bm25_base.index([{"doc_id": tid, "text": txt} for tid, txt in zip(train_ids, train_texts, strict=True)])

    def bm25_base_search(query: str, top_k: int) -> list[tuple[str, float]]:
        hits = bm25_base.search(query, top_k=top_k)
        return [(h.object_id, h.score) for h in hits]

    metrics_base, scores_base, dur_base = _evaluate_retriever_fn(
        bm25_base_search, test_texts, test_labels, test_ids, intent_to_train_ids
    )
    results.append(
        ExperimentResult(
            hypothesis="H1",
            variant_name="BM25-base",
            metrics=metrics_base,
            config={"type": "bm25", "normalization": False},
            duration_ms=dur_base,
            per_query_scores=scores_base,
        )
    )

    # --- BM25-norm (accent-aware normalization) ---
    logger.info("Running BM25-norm...")
    norm_texts = [normalize_for_matching(t) for t in train_texts]
    bm25_norm = InMemoryBM25Index(config=LexicalIndexConfig())
    bm25_norm.index([{"doc_id": tid, "text": txt} for tid, txt in zip(train_ids, norm_texts, strict=True)])

    def bm25_norm_search(query: str, top_k: int) -> list[tuple[str, float]]:
        hits = bm25_norm.search(normalize_for_matching(query), top_k=top_k)
        return [(h.object_id, h.score) for h in hits]

    metrics_norm, scores_norm, dur_norm = _evaluate_retriever_fn(
        bm25_norm_search, test_texts, test_labels, test_ids, intent_to_train_ids
    )
    results.append(
        ExperimentResult(
            hypothesis="H1",
            variant_name="BM25-norm",
            metrics=metrics_norm,
            config={"type": "bm25", "normalization": True},
            duration_ms=dur_norm,
            per_query_scores=scores_norm,
        )
    )

    # --- ANN (vector-only) ---
    logger.info("Loading embedding model: %s...", EMBEDDING_MODEL)
    emb_config = EmbeddingModelConfig(model_name=EMBEDDING_MODEL, model_version="1.0")
    emb_gen = SentenceTransformerGenerator(model_config=emb_config)

    vec_config = VectorIndexConfig(dimensions=384)
    vec_index = InMemoryVectorIndex(config=vec_config)

    # Generate and index embeddings for training docs
    train_emb_inputs = [
        EmbeddingInput(
            embedding_id=f"emb_{tid}",
            object_type=ObjectType.CONVERSATION,
            object_id=tid,
            text=txt,
        )
        for tid, txt in zip(train_ids, train_texts, strict=True)
    ]
    batch = EmbeddingBatch(items=train_emb_inputs)
    train_records = emb_gen.generate(batch)
    vec_index.upsert(train_records)

    def ann_search(query: str, top_k: int) -> list[tuple[str, float]]:
        q_input = EmbeddingInput(
            embedding_id="query_emb",
            object_type=ObjectType.CONVERSATION,
            object_id="query",
            text=query,
        )
        q_batch = EmbeddingBatch(items=[q_input])
        q_records = emb_gen.generate(q_batch)
        hits = vec_index.search_by_vector(q_records[0].vector, top_k=top_k)
        return [(h.object_id, h.score) for h in hits]

    metrics_ann, scores_ann, dur_ann = _evaluate_retriever_fn(
        ann_search, test_texts, test_labels, test_ids, intent_to_train_ids
    )
    results.append(
        ExperimentResult(
            hypothesis="H1",
            variant_name="ANN-MiniLM",
            metrics=metrics_ann,
            config={"type": "ann", "model": EMBEDDING_MODEL, "dimensions": 384},
            duration_ms=dur_ann,
            per_query_scores=scores_ann,
        )
    )

    # --- Hybrid-RRF ---
    logger.info("Running Hybrid-RRF...")
    hybrid_rrf_config = HybridRetrievalConfig(
        fusion_strategy=FusionStrategy.RRF,
        lexical_top_k=20,
        vector_top_k=20,
        final_top_k=20,
    )
    hybrid_rrf = SimpleHybridRetriever(
        lexical_index=bm25_base,
        vector_index=vec_index,
        embedding_generator=emb_gen,
        config=hybrid_rrf_config,
    )

    def hybrid_rrf_search(query: str, top_k: int) -> list[tuple[str, float]]:
        q = RetrievalQuery(query_text=query, top_k=top_k, query_type=QueryType.HYBRID)
        result = hybrid_rrf.retrieve(q)
        return [(h.object_id, h.score) for h in result.hits]

    metrics_rrf, scores_rrf, dur_rrf = _evaluate_retriever_fn(
        hybrid_rrf_search, test_texts, test_labels, test_ids, intent_to_train_ids
    )
    results.append(
        ExperimentResult(
            hypothesis="H1",
            variant_name="Hybrid-RRF",
            metrics=metrics_rrf,
            config={"type": "hybrid", "fusion": "rrf"},
            duration_ms=dur_rrf,
            per_query_scores=scores_rrf,
        )
    )

    # --- Hybrid-LINEAR: tune alpha on validation, report on test ---
    def _make_linear_search(retriever: SimpleHybridRetriever):
        def search_fn(query: str, top_k: int) -> list[tuple[str, float]]:
            q = RetrievalQuery(query_text=query, top_k=top_k, query_type=QueryType.HYBRID)
            result = retriever.retrieve(q)
            return [(h.object_id, h.score) for h in result.hits]

        return search_fn

    alpha_candidates = [0.3, 0.5, 0.65, 0.8]

    # Step 1: Evaluate all alphas on validation set to select best
    best_val_mrr = -1.0
    best_alpha = alpha_candidates[0]
    val_mrr_by_alpha: dict[float, float] = {}

    for alpha in alpha_candidates:
        hybrid_lin_config = HybridRetrievalConfig(
            fusion_strategy=FusionStrategy.LINEAR,
            fusion_weight=alpha,
            lexical_top_k=20,
            vector_top_k=20,
            final_top_k=20,
        )
        hybrid_lin = SimpleHybridRetriever(
            lexical_index=bm25_base,
            vector_index=vec_index,
            embedding_generator=emb_gen,
            config=hybrid_lin_config,
        )
        val_metrics, _, _ = _evaluate_retriever_fn(
            _make_linear_search(hybrid_lin), val_texts, val_labels, val_ids, intent_to_train_ids
        )
        val_mrr = val_metrics["mrr"]
        val_mrr_by_alpha[alpha] = val_mrr
        logger.info("  Hybrid-LINEAR alpha=%.2f: val MRR=%.4f", alpha, val_mrr)
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_alpha = alpha

    logger.info("Best alpha=%.2f selected on validation (MRR=%.4f)", best_alpha, best_val_mrr)

    # Step 2: Evaluate all alphas on test set (for full reporting), marking best
    for alpha in alpha_candidates:
        logger.info("Evaluating Hybrid-LINEAR (alpha=%.2f) on test...", alpha)
        hybrid_lin_config = HybridRetrievalConfig(
            fusion_strategy=FusionStrategy.LINEAR,
            fusion_weight=alpha,
            lexical_top_k=20,
            vector_top_k=20,
            final_top_k=20,
        )
        hybrid_lin = SimpleHybridRetriever(
            lexical_index=bm25_base,
            vector_index=vec_index,
            embedding_generator=emb_gen,
            config=hybrid_lin_config,
        )
        metrics_lin, scores_lin, dur_lin = _evaluate_retriever_fn(
            _make_linear_search(hybrid_lin), test_texts, test_labels, test_ids, intent_to_train_ids
        )
        variant_name = f"Hybrid-LINEAR-a{alpha:.2f}"
        if alpha == best_alpha:
            variant_name += " (val-selected)"
        results.append(
            ExperimentResult(
                hypothesis="H1",
                variant_name=variant_name,
                metrics={**metrics_lin, "val_mrr": val_mrr_by_alpha[alpha]},
                config={
                    "type": "hybrid",
                    "fusion": "linear",
                    "alpha": alpha,
                    "selected_on_val": alpha == best_alpha,
                },
                duration_ms=dur_lin,
                per_query_scores=scores_lin,
            )
        )

    logger.info("H1 complete: %d variants evaluated", len(results))
    return results


def _evaluate_retriever_fn(
    retrieve_fn: Any,
    test_texts: list[str],
    test_labels: list[str],
    test_ids: list[str],
    intent_to_train_ids: dict[str, list[str]],
    k_values: list[int] | None = None,
) -> tuple[dict[str, float], list[float], float]:
    """Evaluate a retriever function using intent-based relevance.

    Args:
        retrieve_fn: Callable(query_text, top_k) -> list[(doc_id, score)].
        test_texts: Query texts.
        test_labels: Ground truth labels for each query.
        test_ids: Query identifiers.
        intent_to_train_ids: Mapping intent -> list of relevant doc IDs.
        k_values: K values for Recall@K, Precision@K, nDCG@K.

    Returns:
        Tuple of (metrics dict, per-query RR scores, duration in ms).
    """
    from talkex.evaluation.metrics import ndcg, precision_at_k, recall_at_k, reciprocal_rank

    if k_values is None:
        k_values = [5, 10, 20]
    max_k = max(k_values)

    t0 = time.perf_counter()
    all_recall: dict[int, list[float]] = {k: [] for k in k_values}
    all_precision: dict[int, list[float]] = {k: [] for k in k_values}
    all_ndcg: dict[int, list[float]] = {k: [] for k in k_values}
    all_rr: list[float] = []

    for text, label, _qid in zip(test_texts, test_labels, test_ids, strict=True):
        hits = retrieve_fn(text, max_k)
        retrieved = [doc_id for doc_id, _ in hits]
        relevant = set(intent_to_train_ids.get(label, []))
        rel_map = {doc_id: 1 for doc_id in relevant}

        for k in k_values:
            all_recall[k].append(recall_at_k(retrieved, relevant, k))
            all_precision[k].append(precision_at_k(retrieved, relevant, k))
            all_ndcg[k].append(ndcg(retrieved, rel_map, k))
        all_rr.append(reciprocal_rank(retrieved, relevant))

    dur = (time.perf_counter() - t0) * 1000

    metrics: dict[str, float] = {"mrr": float(np.mean(all_rr))}
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(all_recall[k]))
        metrics[f"precision@{k}"] = float(np.mean(all_precision[k]))
        metrics[f"ndcg@{k}"] = float(np.mean(all_ndcg[k]))

    return metrics, all_rr, dur


# ---------------------------------------------------------------------------
# H2: Multi-Level Classification
# ---------------------------------------------------------------------------


def run_h2(train: list[dict], val: list[dict], test: list[dict]) -> list[ExperimentResult]:
    """Run H2 experiments: multi-level representation gains with context windows.

    Compares lexical-only vs lexical+embedding features across LogReg, LightGBM,
    MLP classifiers. Conversations are segmented into turns and windowed
    (window_size=5, stride=2) to align with the real TalkEx pipeline.

    Training: classifiers learn on window-level features (labels inherited from conversation).
    Evaluation: window predictions are aggregated to conversation level via
    average class probabilities (argmax), then compared with conversation-level ground truth.

    Best configuration is identified on validation set; all metrics reported on test.
    """
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.logistic import LogisticRegressionClassifier
    from talkex.classification.mlp_classifier import MLPClassifier
    from talkex.classification.models import ClassificationInput

    logger.info("=" * 60)
    logger.info("H2: Multi-Level Classification Experiment (context windows)")
    logger.info("=" * 60)

    # --- Step 1: Parse conversations into context windows ---
    train_cw = _prepare_windowed_data(train)
    val_cw = _prepare_windowed_data(val)
    test_cw = _prepare_windowed_data(test)
    logger.info(
        "Windows: train=%d (from %d convs), val=%d (from %d convs), test=%d (from %d convs)",
        sum(len(c.windows) for c in train_cw),
        len(train_cw),
        sum(len(c.windows) for c in val_cw),
        len(val_cw),
        sum(len(c.windows) for c in test_cw),
        len(test_cw),
    )

    # Flatten windows for training (window-level labels inherited from conversation)
    train_win_texts, train_win_labels, _ = _flatten_windows(train_cw)
    val_win_texts, val_win_labels, _ = _flatten_windows(val_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    # Conversation-level labels for evaluation
    test_conv_labels = [cw.label for cw in test_cw]
    val_conv_labels = [cw.label for cw in val_cw]
    unique_labels = sorted(set(train_win_labels + val_win_labels + test_conv_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # --- Step 2: Extract features per window ---

    def make_window_features(
        conv_windows: list[ConversationWindows],
    ) -> tuple[list[dict[str, float]], list[str]]:
        """Extract lexical + structural features per window."""
        struct_feats = _extract_window_structural_features(conv_windows)
        all_features: list[dict[str, float]] = []
        for idx, cw in enumerate(conv_windows):
            for w_idx, w in enumerate(cw.windows):
                lex = extract_lexical_features(w.window_text)
                flat_idx = sum(len(c.windows) for c in conv_windows[:idx]) + w_idx
                combined = {**lex.features, **struct_feats[flat_idx]}
                all_features.append(combined)
        feature_names = list(all_features[0].keys()) if all_features else []
        return all_features, feature_names

    train_lex_features, lex_feature_names = make_window_features(train_cw)
    val_lex_features, _ = make_window_features(val_cw)
    test_lex_features, _ = make_window_features(test_cw)

    # --- Step 3: Generate embeddings per window ---
    logger.info("Generating window embeddings via talkex (%s)...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()
    train_win_ids = [f"train_win_{i}" for i in range(len(train_win_texts))]
    val_win_ids = [f"val_win_{i}" for i in range(len(val_win_texts))]
    test_win_ids = [f"test_win_{i}" for i in range(len(test_win_texts))]
    train_embeddings = generate_embeddings_via_talkex(train_win_texts, train_win_ids, emb_gen)
    val_embeddings = generate_embeddings_via_talkex(val_win_texts, val_win_ids, emb_gen)
    test_embeddings = generate_embeddings_via_talkex(test_win_texts, test_win_ids, emb_gen)
    emb_dims = train_embeddings.shape[1]
    logger.info(
        "Window embeddings: %d dims, %d train, %d val, %d test",
        emb_dims,
        len(train_win_texts),
        len(val_win_texts),
        len(test_win_texts),
    )

    # --- Step 4: Build feature configurations ---
    def merge_with_embeddings(
        lex_features: list[dict[str, float]], embeddings: np.ndarray
    ) -> tuple[list[dict[str, float]], list[str]]:
        merged = []
        for i, lex in enumerate(lex_features):
            combined = dict(lex)
            for d in range(embeddings.shape[1]):
                combined[f"emb_{d}"] = float(embeddings[i][d])
            merged.append(combined)
        names = list(merged[0].keys()) if merged else []
        return merged, names

    train_full_features, full_feature_names = merge_with_embeddings(train_lex_features, train_embeddings)
    val_full_features, _ = merge_with_embeddings(val_lex_features, val_embeddings)
    test_full_features, _ = merge_with_embeddings(test_lex_features, test_embeddings)

    feature_configs = {
        "lexical": (train_lex_features, val_lex_features, test_lex_features, lex_feature_names),
        "lexical+emb": (train_full_features, val_full_features, test_full_features, full_feature_names),
    }

    # --- Step 5: Train, evaluate with window→conversation aggregation ---
    results: list[ExperimentResult] = []
    best_val_f1 = -1.0
    best_val_config = ""

    for feat_name, (tr_feats, va_feats, te_feats, feat_names) in feature_configs.items():
        train_inputs = [
            ClassificationInput(
                source_id=f"train_win_{i}",
                source_type="window",
                text=train_win_texts[i],
                features=tr_feats[i],
            )
            for i in range(len(train_win_texts))
        ]
        val_inputs = [
            ClassificationInput(
                source_id=f"val_win_{i}",
                source_type="window",
                text=val_win_texts[i],
                features=va_feats[i],
            )
            for i in range(len(val_win_texts))
        ]
        test_inputs = [
            ClassificationInput(
                source_id=f"test_win_{i}",
                source_type="window",
                text=test_win_texts[i],
                features=te_feats[i],
            )
            for i in range(len(test_win_texts))
        ]

        classifier_configs_map = {
            "LogReg": lambda fn=feat_names: LogisticRegressionClassifier(
                label_space=label_space,
                feature_names=fn,
                model_name="logistic-regression",
                sklearn_kwargs={"max_iter": 2000, "random_state": CURRENT_SEED},
            ),
            "LightGBM": lambda fn=feat_names: LightGBMClassifier(
                label_space=label_space,
                feature_names=fn,
                model_name="lightgbm",
                lgbm_kwargs={"n_estimators": 100, "num_leaves": 31, "verbosity": -1, "random_state": CURRENT_SEED},
            ),
            "MLP": lambda fn=feat_names: MLPClassifier(
                label_space=label_space,
                feature_names=fn,
                model_name="mlp",
                hidden_layer_sizes=(128, 64),
                sklearn_kwargs={"max_iter": 1000, "random_state": CURRENT_SEED},
            ),
        }

        for clf_name, clf_factory in classifier_configs_map.items():
            logger.info("Training %s (%s) on %d windows...", clf_name, feat_name, len(train_inputs))
            t0 = time.perf_counter()
            clf = clf_factory()
            clf.fit(train_inputs, train_win_labels)

            # --- Val evaluation: aggregate windows → conversations ---
            val_window_preds = clf.classify(val_inputs)
            val_conv_preds = _aggregate_windows_to_conversations(val_window_preds, val_cw, unique_labels)
            val_f1 = _compute_macro_f1(val_conv_preds, val_conv_labels, unique_labels)
            config_key = f"{feat_name}_{clf_name}"
            logger.info("  [VAL] %s/%s: macro_f1=%.4f", feat_name, clf_name, val_f1)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_config = config_key

            # --- Test evaluation: aggregate windows → conversations ---
            test_window_preds = clf.classify(test_inputs)
            test_conv_preds = _aggregate_windows_to_conversations(test_window_preds, test_cw, unique_labels)
            dur = (time.perf_counter() - t0) * 1000

            # Compute conversation-level metrics
            metrics = _compute_classification_metrics(test_conv_preds, test_conv_labels, unique_labels)
            metrics["val_macro_f1"] = val_f1
            metrics["n_train_windows"] = len(train_inputs)
            metrics["n_test_windows"] = len(test_inputs)

            # Per-conversation scores for statistical tests
            per_sample = [
                1.0 if pred == true else 0.0 for pred, true in zip(test_conv_preds, test_conv_labels, strict=False)
            ]

            results.append(
                ExperimentResult(
                    hypothesis="H2",
                    variant_name=config_key,
                    metrics=metrics,
                    config={
                        "classifier": clf_name,
                        "features": feat_name,
                        "window_size": WINDOW_SIZE,
                        "window_stride": WINDOW_STRIDE,
                        "aggregation": "avg_confidence_argmax",
                    },
                    duration_ms=dur,
                    per_query_scores=per_sample,
                )
            )
            logger.info(
                "  %s/%s: macro_f1=%.4f, accuracy=%.4f",
                feat_name,
                clf_name,
                metrics.get("macro_f1", 0),
                metrics.get("accuracy", 0),
            )

    # Mark val-selected best config
    logger.info("Best config on validation: %s (val F1=%.4f)", best_val_config, best_val_f1)
    for r in results:
        r.config["selected_on_val"] = r.variant_name == best_val_config

    logger.info("H2 complete: %d variants evaluated", len(results))
    return results


# ---------------------------------------------------------------------------
# H4: Cascaded Inference
# ---------------------------------------------------------------------------


def run_h4(train: list[dict], val: list[dict], test: list[dict]) -> list[ExperimentResult]:
    """Run H4 experiments: cascaded inference efficiency (with context windows).

    Compares uniform pipeline vs cascaded pipeline at different thresholds.
    Stage 1 (lightweight): LogReg with lexical+structural+emb features (cheap linear model).
    Stage 2 (full): LightGBM with lexical+structural+emb features (expensive ensemble).

    Pipeline aligned with real TalkEx: conversations segmented into context windows
    (window_size=5, stride=2). Cascade threshold check happens per window.
    A conversation might have some windows resolved by Stage 1 and others escalated
    to Stage 2. Final predictions aggregate windows to conversations.

    Cascade threshold is tuned on validation set; final metrics on test.
    """
    logger.info("=" * 60)
    logger.info("H4: Cascaded Inference Experiment (context windows)")
    logger.info("=" * 60)

    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.logistic import LogisticRegressionClassifier
    from talkex.classification.models import ClassificationInput

    # --- Step 1: Parse conversations into context windows ---
    train_cw = _prepare_windowed_data(train)
    val_cw = _prepare_windowed_data(val)
    test_cw = _prepare_windowed_data(test)
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
    val_win_texts, val_win_labels, _ = _flatten_windows(val_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    test_conv_labels = [cw.label for cw in test_cw]
    val_conv_labels = [cw.label for cw in val_cw]
    unique_labels = sorted(set(train_win_labels + val_win_labels + test_conv_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # --- Step 2: Extract features per window (lexical + structural + embeddings) ---
    train_struct = _extract_window_structural_features(train_cw)
    val_struct = _extract_window_structural_features(val_cw)
    test_struct = _extract_window_structural_features(test_cw)

    def build_window_inputs(
        win_texts: list[str],
        struct_feats: list[dict[str, float]],
        embeddings: np.ndarray,
        prefix: str,
    ) -> tuple[list[ClassificationInput], list[str]]:
        inputs = []
        feat_names: list[str] = []
        for i, text in enumerate(win_texts):
            lex = extract_lexical_features(text)
            features = {**(lex.features if hasattr(lex, "features") else dict(lex)), **struct_feats[i]}
            for d in range(embeddings.shape[1]):
                features[f"emb_{d}"] = float(embeddings[i][d])
            if not feat_names:
                feat_names = list(features.keys())
            inputs.append(
                ClassificationInput(
                    source_id=f"{prefix}_win_{i}",
                    source_type="window",
                    text=text,
                    features=features,
                )
            )
        return inputs, feat_names

    logger.info("Generating window embeddings via talkex (%s)...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()
    train_embs = generate_embeddings_via_talkex(
        train_win_texts, [f"train_win_{i}" for i in range(len(train_win_texts))], emb_gen
    )
    val_embs = generate_embeddings_via_talkex(
        val_win_texts, [f"val_win_{i}" for i in range(len(val_win_texts))], emb_gen
    )
    test_embs = generate_embeddings_via_talkex(
        test_win_texts, [f"test_win_{i}" for i in range(len(test_win_texts))], emb_gen
    )
    logger.info("Window embeddings: %d dims", train_embs.shape[1])

    train_inputs, feature_names = build_window_inputs(train_win_texts, train_struct, train_embs, "train")
    val_inputs, _ = build_window_inputs(val_win_texts, val_struct, val_embs, "val")
    test_inputs, _ = build_window_inputs(test_win_texts, test_struct, test_embs, "test")

    # --- Step 3: Train lightweight and full classifiers ---
    logger.info("Training lightweight classifier (LogReg, lexical+structural+emb) on %d windows...", len(train_inputs))
    light_clf = LogisticRegressionClassifier(
        label_space=label_space,
        feature_names=feature_names,
        sklearn_kwargs={"max_iter": 2000, "random_state": CURRENT_SEED},
    )
    light_clf.fit(train_inputs, train_win_labels)

    logger.info("Training full classifier (LightGBM, lexical+structural+emb) on %d windows...", len(train_inputs))
    full_clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=feature_names,
        lgbm_kwargs={"n_estimators": 100, "num_leaves": 31, "verbosity": -1, "random_state": CURRENT_SEED},
    )
    full_clf.fit(train_inputs, train_win_labels)

    results: list[ExperimentResult] = []
    window_config = {
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "aggregation": "avg_confidence_argmax",
    }

    # --- Measure per-window inference cost for each model ---
    n_test_windows = len(test_inputs)
    n_warmup = 3
    n_measure = 10

    for _ in range(n_warmup):
        light_clf.classify(test_inputs)
        full_clf.classify(test_inputs)

    light_times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        light_clf.classify(test_inputs)
        light_times.append((time.perf_counter() - t0) * 1000)
    light_cost_per_window = float(np.median(light_times)) / n_test_windows

    full_times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        full_clf.classify(test_inputs)
        full_times.append((time.perf_counter() - t0) * 1000)
    full_cost_per_window = float(np.median(full_times)) / n_test_windows

    logger.info(
        "Per-window cost: light=%.4f ms, full=%.4f ms (ratio=%.1fx)",
        light_cost_per_window,
        full_cost_per_window,
        full_cost_per_window / light_cost_per_window if light_cost_per_window > 0 else 0,
    )

    # --- Uniform pipeline (full classifier on all windows, aggregate to conversations) ---
    logger.info("Running uniform pipeline...")
    full_window_results = full_clf.classify(test_inputs)
    uniform_conv_preds = _aggregate_windows_to_conversations(full_window_results, test_cw, unique_labels)
    uniform_f1 = _compute_macro_f1(uniform_conv_preds, test_conv_labels, unique_labels)
    uniform_cost = n_test_windows * full_cost_per_window
    uniform_per_sample = [
        1.0 if pred == true else 0.0 for pred, true in zip(uniform_conv_preds, test_conv_labels, strict=False)
    ]

    results.append(
        ExperimentResult(
            hypothesis="H4",
            variant_name="uniform",
            metrics={
                "macro_f1": uniform_f1,
                "cost_ms": uniform_cost,
                "pct_stage1": 0,
                "pct_stage2": 100,
                "light_cost_per_window_ms": light_cost_per_window,
                "full_cost_per_window_ms": full_cost_per_window,
                "n_test_windows": n_test_windows,
            },
            config={"type": "uniform", "classifier": "lightgbm-100t", **window_config},
            duration_ms=uniform_cost,
            per_query_scores=uniform_per_sample,
        )
    )

    # --- Cascaded pipeline: per-window cascade, aggregate to conversations ---
    threshold_candidates = [0.50, 0.60, 0.70, 0.80, 0.90]

    def _run_cascade_windowed(
        inputs: list[ClassificationInput],
        conv_windows: list[ConversationWindows],
        threshold: float,
    ) -> tuple[list[str], float, int]:
        """Run cascade on windows, aggregate to conversations.

        Returns: (conversation_preds, pct_stage1_windows, n_stage2_windows).
        """
        light_results = light_clf.classify(inputs)
        # Determine which windows need Stage 2
        window_preds_list: list[tuple[int, Any]] = []
        stage2_indices: list[int] = []
        for idx, lr in enumerate(light_results):
            if lr.top_score >= threshold:
                window_preds_list.append((idx, lr))
            else:
                stage2_indices.append(idx)

        if stage2_indices:
            stage2_inputs = [inputs[i] for i in stage2_indices]
            stage2_results = full_clf.classify(stage2_inputs)
            for i, sr in zip(stage2_indices, stage2_results, strict=True):
                window_preds_list.append((i, sr))

        # Sort by original index to restore order
        window_preds_list.sort(key=lambda x: x[0])
        ordered_window_preds = [p for _, p in window_preds_list]

        # Aggregate windows to conversations
        conv_preds = _aggregate_windows_to_conversations(ordered_window_preds, conv_windows, unique_labels)

        n_total = len(inputs)
        pct_s1 = (n_total - len(stage2_indices)) / n_total * 100
        return conv_preds, pct_s1, len(stage2_indices)

    # Step 1: Select best threshold on validation set
    n_val_windows = len(val_inputs)
    val_uniform_window_results = full_clf.classify(val_inputs)
    val_uniform_preds = _aggregate_windows_to_conversations(val_uniform_window_results, val_cw, unique_labels)
    val_uniform_f1 = _compute_macro_f1(val_uniform_preds, val_conv_labels, unique_labels)
    val_uniform_cost = n_val_windows * full_cost_per_window

    best_val_threshold = threshold_candidates[0]
    best_val_f1 = -1.0
    val_metrics_by_threshold: dict[float, dict[str, float]] = {}

    for threshold in threshold_candidates:
        val_preds, val_pct_s1, val_n_s2 = _run_cascade_windowed(val_inputs, val_cw, threshold)
        val_cascade_f1 = _compute_macro_f1(val_preds, val_conv_labels, unique_labels)
        val_cascade_cost = n_val_windows * light_cost_per_window + val_n_s2 * full_cost_per_window
        val_cost_reduction = (1 - val_cascade_cost / val_uniform_cost) * 100 if val_uniform_cost > 0 else 0
        val_f1_delta = val_uniform_f1 - val_cascade_f1

        val_metrics_by_threshold[threshold] = {
            "val_macro_f1": val_cascade_f1,
            "val_pct_stage1": val_pct_s1,
            "val_cost_reduction_pct": val_cost_reduction,
            "val_f1_delta": val_f1_delta,
        }
        logger.info(
            "  [VAL] threshold=%.2f: F1=%.4f (delta=%.4f), stage1=%.1f%%, cost_reduction=%.1f%%",
            threshold,
            val_cascade_f1,
            val_f1_delta,
            val_pct_s1,
            val_cost_reduction,
        )
        if val_cascade_f1 > best_val_f1:
            best_val_f1 = val_cascade_f1
            best_val_threshold = threshold

    logger.info("Best threshold=%.2f selected on validation (F1=%.4f)", best_val_threshold, best_val_f1)

    # Step 2: Evaluate all thresholds on test set, marking val-selected best
    n_test_convs = len(test_cw)
    for threshold in threshold_candidates:
        logger.info("Evaluating cascaded pipeline (threshold=%.2f) on test...", threshold)

        conv_preds, pct_stage1, n_stage2 = _run_cascade_windowed(test_inputs, test_cw, threshold)

        cascade_f1 = _compute_macro_f1(conv_preds, test_conv_labels, unique_labels)
        cascade_cost = n_test_windows * light_cost_per_window + n_stage2 * full_cost_per_window
        cost_reduction = (1 - cascade_cost / uniform_cost) * 100 if uniform_cost > 0 else 0
        f1_delta = uniform_f1 - cascade_f1

        cascade_per_sample = [
            1.0 if pred == true else 0.0 for pred, true in zip(conv_preds, test_conv_labels, strict=False)
        ]

        variant_name = f"cascade_t{threshold:.2f}"
        if threshold == best_val_threshold:
            variant_name += " (val-selected)"

        results.append(
            ExperimentResult(
                hypothesis="H4",
                variant_name=variant_name,
                metrics={
                    "macro_f1": cascade_f1,
                    "cost_ms": cascade_cost,
                    "pct_stage1": pct_stage1,
                    "pct_stage2": 100 - pct_stage1,
                    "cost_reduction_pct": cost_reduction,
                    "f1_delta": f1_delta,
                    "light_cost_per_window_ms": light_cost_per_window,
                    "full_cost_per_window_ms": full_cost_per_window,
                    "n_test_windows": n_test_windows,
                    "n_test_conversations": n_test_convs,
                    **val_metrics_by_threshold.get(threshold, {}),
                    "selected_on_val": threshold == best_val_threshold,
                },
                config={
                    "type": "cascade",
                    "threshold": threshold,
                    "light": "logreg",
                    "full": "lightgbm-100t",
                    **window_config,
                },
                duration_ms=cascade_cost,
                per_query_scores=cascade_per_sample,
            )
        )
        logger.info(
            "  threshold=%.2f: F1=%.4f (delta=%.4f), stage1=%.1f%%, cost_reduction=%.1f%%",
            threshold,
            cascade_f1,
            f1_delta,
            pct_stage1,
            cost_reduction,
        )

    logger.info("H4 complete: %d variants evaluated", len(results))
    return results


def _compute_macro_f1(predictions: list[str], ground_truth: list[str], labels: list[str]) -> float:
    """Compute macro-averaged F1 score."""
    f1_scores = []
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p != label and g == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def _aggregate_windows_to_conversations(
    window_preds: list,  # list of ClassificationResult
    conv_windows: list[ConversationWindows],
    labels: list[str],
) -> list[str]:
    """Aggregate window-level predictions to conversation-level labels.

    For each conversation, collects its window predictions and applies
    average-confidence aggregation (avg class probs -> argmax).

    Args:
        window_preds: Flat list of ClassificationResult, one per window,
            in the same order as _flatten_windows() output.
        conv_windows: Original windowed data (provides conversation grouping).
        labels: All possible labels.

    Returns:
        List of predicted labels, one per conversation (same order as conv_windows).
    """
    conv_preds: list[str] = []
    offset = 0
    for cw in conv_windows:
        n_win = len(cw.windows)
        window_results = window_preds[offset : offset + n_win]
        pred_label, _ = _aggregate_window_predictions(window_results, labels)
        conv_preds.append(pred_label)
        offset += n_win
    return conv_preds


def _compute_classification_metrics(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Compute classification metrics at conversation level.

    Returns dict with macro_f1, accuracy, and per-label F1/precision/recall.
    """
    metrics: dict[str, float] = {}

    f1_scores = []
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        metrics[f"f1_{label}"] = f1
        metrics[f"precision_{label}"] = precision
        metrics[f"recall_{label}"] = recall

    metrics["macro_f1"] = float(np.mean(f1_scores))
    correct = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == g)
    metrics["accuracy"] = correct / len(ground_truth) if ground_truth else 0.0

    return metrics


# ---------------------------------------------------------------------------
# H3: Rules Complement ML
# ---------------------------------------------------------------------------


def run_h3(train: list[dict], val: list[dict], test: list[dict]) -> list[ExperimentResult]:
    """Run H3 experiments: deterministic rules complement ML (with context windows).

    Compares ML-only, Rules-only, ML+Rules-override, and ML+Rules-feature
    on critical classes (cancelamento, reclamacao) using precision/recall/F1.

    Pipeline aligned with real TalkEx: conversations segmented into context windows
    (window_size=5, stride=2). Features and rules evaluated per window.
    Training at window level; evaluation aggregates back to conversation level
    via average class probabilities (argmax).
    """
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.models import ClassificationInput
    from talkex.rules.ast import AndNode, PredicateNode
    from talkex.rules.config import PredicateType, RuleEngineConfig
    from talkex.rules.evaluator import SimpleRuleEvaluator
    from talkex.rules.models import RuleDefinition, RuleEvaluationInput

    logger.info("=" * 60)
    logger.info("H3: Rules Complement ML Experiment (context windows)")
    logger.info("=" * 60)

    # --- Step 1: Parse conversations into context windows ---
    # H3 uses val only for label space construction (no hyperparameter tuning)
    train_cw = _prepare_windowed_data(train)
    test_cw = _prepare_windowed_data(test)
    val_labels_for_space = extract_labels(val)
    logger.info(
        "Windows: train=%d (from %d convs), test=%d (from %d convs)",
        sum(len(c.windows) for c in train_cw),
        len(train_cw),
        sum(len(c.windows) for c in test_cw),
        len(test_cw),
    )

    # Flatten windows for training
    train_win_texts, train_win_labels, _ = _flatten_windows(train_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    # Conversation-level labels for evaluation
    test_conv_labels = [cw.label for cw in test_cw]
    unique_labels = sorted(set(train_win_labels + val_labels_for_space + test_conv_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # --- Step 2: Extract features per window (lexical + structural + embeddings) ---
    def build_window_features(
        conv_windows: list[ConversationWindows],
        win_texts: list[str],
        embeddings: np.ndarray,
    ) -> tuple[list[dict[str, float]], list[str]]:
        """Build combined feature dicts for each window."""
        struct_feats = _extract_window_structural_features(conv_windows)
        features_list: list[dict[str, float]] = []
        for i, text in enumerate(win_texts):
            lex = extract_lexical_features(text)
            combined = {**(lex.features if hasattr(lex, "features") else dict(lex)), **struct_feats[i]}
            for d in range(embeddings.shape[1]):
                combined[f"emb_{d}"] = float(embeddings[i][d])
            features_list.append(combined)
        feat_names = list(features_list[0].keys()) if features_list else []
        return features_list, feat_names

    logger.info("Generating window embeddings via talkex (%s)...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()
    train_embs = generate_embeddings_via_talkex(
        train_win_texts, [f"train_win_{i}" for i in range(len(train_win_texts))], emb_gen
    )
    test_embs = generate_embeddings_via_talkex(
        test_win_texts, [f"test_win_{i}" for i in range(len(test_win_texts))], emb_gen
    )
    logger.info("Window embeddings: %d dims", train_embs.shape[1])

    train_features, feature_names = build_window_features(train_cw, train_win_texts, train_embs)
    test_features, _ = build_window_features(test_cw, test_win_texts, test_embs)

    # Build ClassificationInputs
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

    # --- Step 3: Train ML classifier (LightGBM, lexical+structural+emb) ---
    logger.info("Training ML classifier (LightGBM, lexical+structural+emb) on %d windows...", len(train_inputs))
    ml_clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=feature_names,
        lgbm_kwargs={"n_estimators": 100, "num_leaves": 31, "verbosity": -1, "random_state": CURRENT_SEED},
    )
    ml_clf.fit(train_inputs, train_win_labels)

    # --- Define deterministic rules for critical classes ---
    cancel_rule = RuleDefinition(
        rule_id="rule_cancel",
        rule_name="cancelamento_keywords",
        rule_version="1.0",
        description="Detects cancellation intent via keyword patterns",
        ast=AndNode(
            children=[
                PredicateNode(
                    predicate_type=PredicateType.LEXICAL,
                    field_name="text",
                    operator="contains_any",
                    value=["cancelar", "cancelamento", "cancela", "desistir", "encerrar contrato", "rescindir"],
                    cost_hint=1,
                ),
            ]
        ),
        tags=["critical", "cancelamento"],
    )
    complaint_rule = RuleDefinition(
        rule_id="rule_complaint",
        rule_name="reclamacao_keywords",
        rule_version="1.0",
        description="Detects complaint intent via keyword patterns",
        ast=AndNode(
            children=[
                PredicateNode(
                    predicate_type=PredicateType.LEXICAL,
                    field_name="text",
                    operator="contains_any",
                    value=[
                        "reclamacao",
                        "reclamar",
                        "reclamando",
                        "absurdo",
                        "inadmissivel",
                        "procon",
                        "anatel",
                        "reclame aqui",
                        "ouvidoria",
                        "insatisfeito",
                        "insatisfeita",
                        "revoltado",
                        "revoltada",
                        "indignado",
                        "indignada",
                    ],
                    cost_hint=1,
                ),
            ]
        ),
        tags=["critical", "reclamacao"],
    )

    rules = [cancel_rule, complaint_rule]
    rule_to_label = {"rule_cancel": "cancelamento", "rule_complaint": "reclamacao"}
    evaluator = SimpleRuleEvaluator()
    rule_config = RuleEngineConfig()

    results: list[ExperimentResult] = []
    window_config = {
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "aggregation": "avg_confidence_argmax",
    }

    # --- ML-only baseline ---
    logger.info("Evaluating ML-only (window→conversation aggregation)...")
    t0 = time.perf_counter()
    ml_window_preds = ml_clf.classify(test_inputs)
    ml_conv_preds = _aggregate_windows_to_conversations(ml_window_preds, test_cw, unique_labels)
    ml_dur = (time.perf_counter() - t0) * 1000

    ml_metrics = _compute_classification_metrics(ml_conv_preds, test_conv_labels, unique_labels)
    ml_per_sample = [1.0 if pred == true else 0.0 for pred, true in zip(ml_conv_preds, test_conv_labels, strict=False)]
    results.append(
        ExperimentResult(
            hypothesis="H3",
            variant_name="ML-only",
            metrics=ml_metrics,
            config={"type": "ml_only", "classifier": "lightgbm", **window_config},
            duration_ms=ml_dur,
            per_query_scores=ml_per_sample,
        )
    )

    # --- Rules-only (evaluate rules per window, aggregate to conversation) ---
    logger.info("Evaluating Rules-only (per window)...")
    t0 = time.perf_counter()

    def evaluate_rules_per_window(win_texts: list[str]) -> list[str]:
        """Evaluate rules on each window, return per-window predicted label."""
        preds = []
        for i, text in enumerate(win_texts):
            eval_input = RuleEvaluationInput(
                source_id=f"win_{i}",
                source_type="window",
                text=text,
            )
            rule_results = evaluator.evaluate(rules, eval_input, rule_config)
            matched_rules = [rr for rr in rule_results if rr.matched]
            if matched_rules:
                preds.append(rule_to_label.get(matched_rules[0].rule_id, "__no_rule__"))
            else:
                preds.append("__no_rule__")
        return preds

    test_rule_window_preds = evaluate_rules_per_window(test_win_texts)

    # Aggregate rule predictions per conversation (majority vote for rules)
    rule_conv_preds: list[str] = []
    offset = 0
    for cw in test_cw:
        n_win = len(cw.windows)
        window_preds_slice = test_rule_window_preds[offset : offset + n_win]
        # For rules: if ANY window triggers a rule, use that label; else no prediction
        non_outros = [p for p in window_preds_slice if p != "__no_rule__"]
        if non_outros:
            # Most frequent rule-triggered label across windows
            from collections import Counter

            label_counts = Counter(non_outros)
            rule_conv_preds.append(label_counts.most_common(1)[0][0])
        else:
            rule_conv_preds.append("__no_rule__")
        offset += n_win

    rules_dur = (time.perf_counter() - t0) * 1000

    rules_metrics = _compute_classification_metrics(rule_conv_preds, test_conv_labels, unique_labels)
    rules_per_sample = [
        1.0 if pred == true else 0.0 for pred, true in zip(rule_conv_preds, test_conv_labels, strict=False)
    ]
    results.append(
        ExperimentResult(
            hypothesis="H3",
            variant_name="Rules-only",
            metrics=rules_metrics,
            config={"type": "rules_only", "rules": list(rule_to_label.keys()), **window_config},
            duration_ms=rules_dur,
            per_query_scores=rules_per_sample,
        )
    )

    # --- ML + Rules-override (per window: if rule fires, override ML prediction) ---
    logger.info("Evaluating ML+Rules-override (per window)...")
    t0 = time.perf_counter()

    # Override at window level: use ML prediction unless a rule fires
    override_window_preds: list[str] = []
    ml_window_labels = [r.top_label for r in ml_window_preds]
    for i, text in enumerate(test_win_texts):
        eval_input = RuleEvaluationInput(
            source_id=f"test_win_{i}",
            source_type="window",
            text=text,
        )
        rule_results = evaluator.evaluate(rules, eval_input, rule_config)
        matched_rules = [rr for rr in rule_results if rr.matched]
        if matched_rules:
            override_window_preds.append(rule_to_label.get(matched_rules[0].rule_id, ml_window_labels[i]))
        else:
            override_window_preds.append(ml_window_labels[i])

    # Aggregate overridden predictions to conversation level (majority vote)
    override_conv_preds: list[str] = []
    offset = 0
    for cw in test_cw:
        n_win = len(cw.windows)
        window_preds_slice = override_window_preds[offset : offset + n_win]
        from collections import Counter

        label_counts = Counter(window_preds_slice)
        override_conv_preds.append(label_counts.most_common(1)[0][0])
        offset += n_win

    override_dur = (time.perf_counter() - t0) * 1000

    override_metrics = _compute_classification_metrics(override_conv_preds, test_conv_labels, unique_labels)
    override_per_sample = [
        1.0 if pred == true else 0.0 for pred, true in zip(override_conv_preds, test_conv_labels, strict=False)
    ]
    results.append(
        ExperimentResult(
            hypothesis="H3",
            variant_name="ML+Rules-override",
            metrics=override_metrics,
            config={"type": "ml_rules_override", "classifier": "lightgbm", **window_config},
            duration_ms=override_dur,
            per_query_scores=override_per_sample,
        )
    )

    # --- ML + Rules-feature (rule match as additional features, train new classifier) ---
    logger.info("Evaluating ML+Rules-feature...")
    t0 = time.perf_counter()

    def add_rule_features_to_inputs(
        win_texts: list[str], inputs: list[ClassificationInput]
    ) -> list[ClassificationInput]:
        """Add rule-match binary features to each window's feature dict."""
        augmented = []
        for i, inp in enumerate(inputs):
            eval_input = RuleEvaluationInput(
                source_id=inp.source_id,
                source_type="window",
                text=win_texts[i],
            )
            rule_results = evaluator.evaluate(rules, eval_input, rule_config)
            new_features = dict(inp.features)
            for rule in rules:
                matched = any(rr.matched for rr in rule_results if rr.rule_id == rule.rule_id)
                new_features[f"rule_{rule.rule_id}"] = 1.0 if matched else 0.0
            augmented.append(
                ClassificationInput(
                    source_id=inp.source_id,
                    source_type=inp.source_type,
                    text=inp.text,
                    features=new_features,
                )
            )
        return augmented

    aug_train = add_rule_features_to_inputs(train_win_texts, train_inputs)
    aug_test = add_rule_features_to_inputs(test_win_texts, test_inputs)
    aug_feature_names = list(aug_train[0].features.keys()) if aug_train else feature_names

    aug_clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=aug_feature_names,
        lgbm_kwargs={"n_estimators": 100, "num_leaves": 31, "verbosity": -1, "random_state": CURRENT_SEED},
    )
    aug_clf.fit(aug_train, train_win_labels)

    # Evaluate on test with window→conversation aggregation
    aug_window_preds = aug_clf.classify(aug_test)
    aug_conv_preds = _aggregate_windows_to_conversations(aug_window_preds, test_cw, unique_labels)
    feature_dur = (time.perf_counter() - t0) * 1000

    feature_metrics = _compute_classification_metrics(aug_conv_preds, test_conv_labels, unique_labels)
    feature_per_sample = [
        1.0 if pred == true else 0.0 for pred, true in zip(aug_conv_preds, test_conv_labels, strict=False)
    ]
    results.append(
        ExperimentResult(
            hypothesis="H3",
            variant_name="ML+Rules-feature",
            metrics=feature_metrics,
            config={"type": "ml_rules_feature", "classifier": "lightgbm", **window_config},
            duration_ms=feature_dur,
            per_query_scores=feature_per_sample,
        )
    )

    logger.info("H3 complete: %d variants evaluated", len(results))
    return results


def _compute_per_class_metrics(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Compute per-class precision, recall, F1."""
    metrics: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p != label and g == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"precision_{label}"] = precision
        metrics[f"recall_{label}"] = recall
        metrics[f"f1_{label}"] = f1

    return metrics


# ---------------------------------------------------------------------------
# Ablation Studies
# ---------------------------------------------------------------------------


def run_ablation(train: list[dict], val: list[dict], test: list[dict]) -> list[ExperimentResult]:
    """Run ablation studies with context windows: remove one component at a time.

    Mirrors H2 pipeline: conversations are segmented into context windows
    (window_size=5, stride=2). Features are extracted per window. Training
    happens at window level; evaluation aggregates back to conversation level
    via average class probabilities (argmax).

    Baseline: LightGBM with lexical + structural + embedding + rule features.

    Ablations:
        -Embeddings: remove embedding features (lexical + structural + rules only)
        -Lexical: remove lexical features (embeddings + structural + rules only)
        -Rules: remove rule features (lexical + structural + embeddings only)
        -Structural: remove structural features (lexical + embeddings + rules only)
        Emb-only: embedding features only (no lexical, no structural, no rules)
        Lexical-only: lexical + structural features only (no embeddings, no rules)
    """
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.models import ClassificationInput
    from talkex.rules.ast import AndNode, PredicateNode
    from talkex.rules.config import PredicateType, RuleEngineConfig
    from talkex.rules.evaluator import SimpleRuleEvaluator
    from talkex.rules.models import RuleDefinition, RuleEvaluationInput

    logger.info("=" * 60)
    logger.info("Ablation Studies (context windows)")
    logger.info("=" * 60)

    # --- Step 1: Parse conversations into context windows ---
    train_cw = _prepare_windowed_data(train)
    val_cw = _prepare_windowed_data(val)
    test_cw = _prepare_windowed_data(test)
    logger.info(
        "Windows: train=%d (from %d convs), val=%d (from %d convs), test=%d (from %d convs)",
        sum(len(c.windows) for c in train_cw),
        len(train_cw),
        sum(len(c.windows) for c in val_cw),
        len(val_cw),
        sum(len(c.windows) for c in test_cw),
        len(test_cw),
    )

    # Flatten windows for training
    train_win_texts, train_win_labels, _ = _flatten_windows(train_cw)
    val_win_texts, val_win_labels, _ = _flatten_windows(val_cw)
    test_win_texts, _, _ = _flatten_windows(test_cw)

    # Conversation-level labels for evaluation
    test_conv_labels = [cw.label for cw in test_cw]
    val_conv_labels = [cw.label for cw in val_cw]
    unique_labels = sorted(set(train_win_labels + val_win_labels + test_conv_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # --- Step 2: Extract feature components per window ---
    # Lexical features
    def extract_window_lexical(win_texts: list[str]) -> list[dict[str, float]]:
        feats = []
        for text in win_texts:
            lex = extract_lexical_features(text)
            feats.append(lex.features if hasattr(lex, "features") else dict(lex))
        return feats

    train_lex = extract_window_lexical(train_win_texts)
    val_lex = extract_window_lexical(val_win_texts)
    test_lex = extract_window_lexical(test_win_texts)

    # Structural features (real variance from window metadata)
    train_struct = _extract_window_structural_features(train_cw)
    val_struct = _extract_window_structural_features(val_cw)
    test_struct = _extract_window_structural_features(test_cw)

    # --- Step 3: Generate embeddings per window ---
    logger.info("Generating window embeddings via talkex (%s)...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()
    train_win_ids = [f"train_win_{i}" for i in range(len(train_win_texts))]
    val_win_ids = [f"val_win_{i}" for i in range(len(val_win_texts))]
    test_win_ids = [f"test_win_{i}" for i in range(len(test_win_texts))]
    train_embs = generate_embeddings_via_talkex(train_win_texts, train_win_ids, emb_gen)
    val_embs = generate_embeddings_via_talkex(val_win_texts, val_win_ids, emb_gen)
    test_embs = generate_embeddings_via_talkex(test_win_texts, test_win_ids, emb_gen)
    logger.info("Window embeddings: %d dims", train_embs.shape[1])

    def make_embedding_dicts(embeddings: np.ndarray) -> list[dict[str, float]]:
        return [
            {f"emb_{d}": float(embeddings[i][d]) for d in range(embeddings.shape[1])} for i in range(len(embeddings))
        ]

    train_emb_feats = make_embedding_dicts(train_embs)
    val_emb_feats = make_embedding_dicts(val_embs)
    test_emb_feats = make_embedding_dicts(test_embs)

    # --- Step 4: Rule features per window ---
    cancel_rule = RuleDefinition(
        rule_id="rule_cancel",
        rule_name="cancelamento_keywords",
        rule_version="1.0",
        description="Detects cancellation intent",
        ast=AndNode(
            children=[
                PredicateNode(
                    predicate_type=PredicateType.LEXICAL,
                    field_name="text",
                    operator="contains_any",
                    value=["cancelar", "cancelamento", "cancela", "desistir", "encerrar contrato", "rescindir"],
                    cost_hint=1,
                ),
            ]
        ),
        tags=["critical", "cancelamento"],
    )
    complaint_rule = RuleDefinition(
        rule_id="rule_complaint",
        rule_name="reclamacao_keywords",
        rule_version="1.0",
        description="Detects complaint intent",
        ast=AndNode(
            children=[
                PredicateNode(
                    predicate_type=PredicateType.LEXICAL,
                    field_name="text",
                    operator="contains_any",
                    value=[
                        "reclamacao",
                        "reclamar",
                        "reclamando",
                        "absurdo",
                        "inadmissivel",
                        "procon",
                        "anatel",
                        "reclame aqui",
                        "ouvidoria",
                        "insatisfeito",
                        "insatisfeita",
                        "revoltado",
                        "revoltada",
                        "indignado",
                        "indignada",
                    ],
                    cost_hint=1,
                ),
            ]
        ),
        tags=["critical", "reclamacao"],
    )
    rules = [cancel_rule, complaint_rule]
    evaluator = SimpleRuleEvaluator()
    rule_config = RuleEngineConfig()

    def compute_window_rule_features(win_texts: list[str]) -> list[dict[str, float]]:
        rule_feats = []
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
            rule_feats.append(feats)
        return rule_feats

    logger.info("Computing rule features per window...")
    train_rule_feats = compute_window_rule_features(train_win_texts)
    val_rule_feats = compute_window_rule_features(val_win_texts)
    test_rule_feats = compute_window_rule_features(test_win_texts)

    # --- Step 5: Define ablation configurations ---
    # Each config: (name, which component families to include)
    ablation_configs = [
        ("full_pipeline", ["lexical", "structural", "embedding", "rules"]),
        ("-Embeddings", ["lexical", "structural", "rules"]),
        ("-Lexical", ["structural", "embedding", "rules"]),
        ("-Rules", ["lexical", "structural", "embedding"]),
        ("-Structural", ["lexical", "embedding", "rules"]),
        ("Emb-only", ["embedding"]),
        ("Lexical-only", ["lexical", "structural"]),
    ]

    def merge_components(
        lex: list[dict[str, float]],
        struct: list[dict[str, float]],
        emb: list[dict[str, float]],
        rule: list[dict[str, float]],
        include: list[str],
    ) -> tuple[list[dict[str, float]], list[str]]:
        merged = []
        for i in range(len(lex)):
            features: dict[str, float] = {}
            if "lexical" in include:
                features.update(lex[i])
            if "structural" in include:
                features.update(struct[i])
            if "embedding" in include:
                features.update(emb[i])
            if "rules" in include:
                features.update(rule[i])
            merged.append(features)
        names = list(merged[0].keys()) if merged else []
        return merged, names

    # --- Step 6: Train and evaluate each ablation config ---
    results: list[ExperimentResult] = []
    lgbm_kwargs = {"n_estimators": 100, "num_leaves": 31, "verbosity": -1, "random_state": CURRENT_SEED}

    for config_name, include in ablation_configs:
        logger.info("Running ablation: %s (features: %s)...", config_name, include)
        t0 = time.perf_counter()

        train_feats, feat_names = merge_components(train_lex, train_struct, train_emb_feats, train_rule_feats, include)
        val_feats, _ = merge_components(val_lex, val_struct, val_emb_feats, val_rule_feats, include)
        test_feats, _ = merge_components(test_lex, test_struct, test_emb_feats, test_rule_feats, include)

        train_inputs = [
            ClassificationInput(
                source_id=f"train_win_{i}",
                source_type="window",
                text=train_win_texts[i],
                features=train_feats[i],
            )
            for i in range(len(train_win_texts))
        ]
        val_inputs = [
            ClassificationInput(
                source_id=f"val_win_{i}",
                source_type="window",
                text=val_win_texts[i],
                features=val_feats[i],
            )
            for i in range(len(val_win_texts))
        ]
        test_inputs = [
            ClassificationInput(
                source_id=f"test_win_{i}",
                source_type="window",
                text=test_win_texts[i],
                features=test_feats[i],
            )
            for i in range(len(test_win_texts))
        ]

        clf = LightGBMClassifier(
            label_space=label_space,
            feature_names=feat_names,
            lgbm_kwargs=lgbm_kwargs,
        )
        clf.fit(train_inputs, train_win_labels)

        # Val evaluation: aggregate windows → conversations
        val_window_preds = clf.classify(val_inputs)
        val_conv_preds = _aggregate_windows_to_conversations(val_window_preds, val_cw, unique_labels)
        val_f1 = _compute_macro_f1(val_conv_preds, val_conv_labels, unique_labels)

        # Test evaluation: aggregate windows → conversations
        test_window_preds = clf.classify(test_inputs)
        test_conv_preds = _aggregate_windows_to_conversations(test_window_preds, test_cw, unique_labels)
        dur = (time.perf_counter() - t0) * 1000

        metrics = _compute_classification_metrics(test_conv_preds, test_conv_labels, unique_labels)
        metrics["val_macro_f1"] = val_f1
        metrics["n_features"] = len(feat_names)
        metrics["n_train_windows"] = len(train_win_texts)
        metrics["n_test_windows"] = len(test_win_texts)

        # Per-conversation scores for statistical tests
        per_sample = [
            1.0 if pred == true else 0.0 for pred, true in zip(test_conv_preds, test_conv_labels, strict=False)
        ]

        results.append(
            ExperimentResult(
                hypothesis="ablation",
                variant_name=config_name,
                metrics=metrics,
                config={
                    "type": "ablation",
                    "include": include,
                    "n_features": len(feat_names),
                    "window_size": WINDOW_SIZE,
                    "window_stride": WINDOW_STRIDE,
                    "aggregation": "avg_confidence_argmax",
                },
                duration_ms=dur,
                per_query_scores=per_sample,
            )
        )
        logger.info(
            "  %s: macro_f1=%.4f (val=%.4f), n_features=%d",
            config_name,
            metrics.get("macro_f1", 0),
            val_f1,
            len(feat_names),
        )

    # Compute deltas from full pipeline
    full_f1 = results[0].metrics["macro_f1"]
    for r in results[1:]:
        r.metrics["delta_f1"] = full_f1 - r.metrics["macro_f1"]
        r.metrics["pct_drop"] = (full_f1 - r.metrics["macro_f1"]) / full_f1 * 100 if full_f1 > 0 else 0

    logger.info("Ablation complete: %d configs evaluated", len(results))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--hypothesis", required=True, type=click.Choice(["H1", "H2", "H3", "H4", "ablation", "all"]))
@click.option("--splits-dir", default="experiments/data", help="Directory with train/val/test splits.")
@click.option("--output-dir", default="experiments/results", help="Base output directory.")
@click.option(
    "--seeds",
    default="42",
    help="Comma-separated seeds for multi-seed evaluation. Default: '42' (single seed).",
)
def main(hypothesis: str, splits_dir: str, output_dir: str, seeds: str) -> None:
    """Run experiment(s) for a specific hypothesis.

    Multi-seed mode: when multiple seeds are provided (e.g., --seeds 42,123,456,789,0),
    each experiment runs once per seed with different random_state values. Results are
    aggregated to report mean +/- std across seeds.
    """
    global SPLITS_DIR, RESULTS_DIR, CURRENT_SEED
    SPLITS_DIR = Path(splits_dir)
    RESULTS_DIR = Path(output_dir)

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    multi_seed = len(seed_list) > 1

    hypotheses = [hypothesis] if hypothesis != "all" else ["H1", "H2", "H3", "H4", "ablation"]

    # H1 (retrieval) is fully deterministic — no random components.
    # Run it once regardless of seed count.
    deterministic_hypotheses = {"H1"}

    for h in hypotheses:
        logger.info("=" * 60)
        logger.info("Starting experiment: %s", h)

        is_deterministic = h in deterministic_hypotheses
        effective_seeds = [seed_list[0]] if is_deterministic else seed_list

        if multi_seed and not is_deterministic:
            logger.info("Multi-seed evaluation: %d seeds %s", len(effective_seeds), effective_seeds)
        elif multi_seed and is_deterministic:
            logger.info("Deterministic experiment — single run (seed-invariant)")
        logger.info("=" * 60)

        train = load_split("train")
        test = load_split("test")
        val = load_split("val")

        all_seed_results: list[list[ExperimentResult]] = []

        for seed in effective_seeds:
            CURRENT_SEED = seed
            if multi_seed and not is_deterministic:
                logger.info("--- Seed %d ---", seed)

            if h == "H1":
                results = run_h1(train, val, test)
            elif h == "H2":
                results = run_h2(train, val, test)
            elif h == "H3":
                results = run_h3(train, val, test)
            elif h == "H4":
                results = run_h4(train, val, test)
            elif h == "ablation":
                results = run_ablation(train, val, test)
            else:
                logger.warning("Hypothesis %s not yet implemented in orchestration script", h)
                continue

            all_seed_results.append(results)

        if not all_seed_results:
            continue

        if multi_seed and not is_deterministic:
            # Aggregate results across seeds: compute mean +/- std for each variant
            aggregated = _aggregate_multi_seed_results(all_seed_results)
            save_results(aggregated, RESULTS_DIR / h)

            # Also save per-seed details
            per_seed_data = []
            for seed, seed_results in zip(seed_list, all_seed_results, strict=True):
                for r in seed_results:
                    d = r.to_dict()
                    d["seed"] = seed
                    per_seed_data.append(d)
            per_seed_path = RESULTS_DIR / h / "per_seed_results.json"
            per_seed_path.parent.mkdir(parents=True, exist_ok=True)
            with open(per_seed_path, "w", encoding="utf-8") as f:
                json.dump(per_seed_data, f, indent=2, ensure_ascii=False)
            logger.info("Per-seed results saved to %s", per_seed_path)

            # Statistical tests on aggregated results
            if len(aggregated) >= 2:
                _run_statistical_analysis(aggregated, RESULTS_DIR / h)
        else:
            results = all_seed_results[0]
            save_results(results, RESULTS_DIR / h)

            if len(results) >= 2:
                _run_statistical_analysis(results, RESULTS_DIR / h)

    logger.info("All experiments complete!")


def _aggregate_multi_seed_results(
    all_seed_results: list[list[ExperimentResult]],
) -> list[ExperimentResult]:
    """Aggregate experiment results across multiple seeds.

    For each variant, computes mean and std of all numeric metrics across seeds.
    Returns aggregated results with mean metrics, plus _std and _seeds_n metadata.
    Uses per_query_scores from the first seed for statistical tests.
    """
    # Group by variant name
    variant_results: dict[str, list[ExperimentResult]] = {}
    for seed_results in all_seed_results:
        for r in seed_results:
            # Normalize variant name (strip "(val-selected)" for grouping)
            base_name = r.variant_name.replace(" (val-selected)", "")
            variant_results.setdefault(base_name, []).append(r)

    aggregated: list[ExperimentResult] = []
    n_seeds = len(all_seed_results)

    for variant_name, variant_runs in variant_results.items():
        # Collect all numeric metrics across seeds
        all_metrics: dict[str, list[float]] = {}
        for r in variant_runs:
            for key, value in r.metrics.items():
                if isinstance(value, int | float):
                    all_metrics.setdefault(key, []).append(float(value))

        # Compute mean +/- std
        agg_metrics: dict[str, float] = {}
        for key, values in all_metrics.items():
            agg_metrics[key] = float(np.mean(values))
            agg_metrics[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        agg_metrics["n_seeds"] = float(n_seeds)

        # Use config from first run, add seed info
        config = dict(variant_runs[0].config)
        config["seeds"] = n_seeds
        config["aggregation"] = "mean"

        # Use per_query_scores from first seed for statistical tests
        first_scores = variant_runs[0].per_query_scores

        aggregated.append(
            ExperimentResult(
                hypothesis=variant_runs[0].hypothesis,
                variant_name=variant_name,
                metrics=agg_metrics,
                config=config,
                duration_ms=float(np.mean([r.duration_ms for r in variant_runs])),
                per_query_scores=first_scores,
            )
        )

    return aggregated


def _run_statistical_analysis(results: list[ExperimentResult], output_dir: Path) -> None:
    """Run statistical tests on experiment results.

    For classification hypotheses (H2, H3, H4): uses per-sample accuracy (1/0)
    with McNemar's test and bootstrap CI on accuracy difference.

    For retrieval hypotheses (H1): uses per-query RR scores with Wilcoxon
    signed-rank test and bootstrap CI.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from stats.statistical_tests import bootstrap_ci, wilcoxon_signed_rank

    # Find pairs with per-query scores for paired tests
    scored = [(r.variant_name, r.per_query_scores) for r in results if r.per_query_scores]
    if len(scored) < 2:
        logger.info("Not enough per-query scores for paired statistical tests")
        return

    # Compare best vs each baseline
    stat_results: list[dict[str, Any]] = []
    best = max(results, key=lambda r: r.metrics.get("macro_f1", r.metrics.get("mrr", 0)))

    for r in results:
        if r.variant_name == best.variant_name:
            continue
        if r.per_query_scores and best.per_query_scores:
            try:
                test_result = wilcoxon_signed_rank(
                    best.per_query_scores,
                    r.per_query_scores,
                    label_a=best.variant_name,
                    label_b=r.variant_name,
                )
                stat_results.append(
                    {
                        "comparison": f"{best.variant_name} vs {r.variant_name}",
                        "test": test_result.test_name,
                        "statistic": test_result.statistic,
                        "p_value": test_result.p_value,
                        "significant": test_result.significant,
                        "effect_size": test_result.effect_size,
                        "summary": test_result.summary,
                    }
                )

                metric_name = "accuracy_diff" if results[0].hypothesis in ("H2", "H3", "H4") else "MRR_diff"
                ci = bootstrap_ci(
                    best.per_query_scores,
                    r.per_query_scores,
                    metric_name=metric_name,
                )
                stat_results.append(
                    {
                        "comparison": f"{best.variant_name} vs {r.variant_name}",
                        "test": "Bootstrap CI",
                        "ci_lower": ci.ci_lower,
                        "ci_upper": ci.ci_upper,
                        "observed_diff": ci.observed,
                        "summary": ci.summary,
                    }
                )
            except Exception as e:
                logger.warning("Statistical test failed for %s vs %s: %s", best.variant_name, r.variant_name, e)

    # Per-class bootstrap analysis for H2 (best lexical+emb vs best lexical-only)
    if results and results[0].hypothesis == "H2":
        per_class_stats = _run_per_class_analysis(results, output_dir)
        if per_class_stats:
            stat_results.extend(per_class_stats)

    if stat_results:
        stats_path = output_dir / "statistical_tests.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stat_results, f, indent=2, ensure_ascii=False)
        logger.info("Statistical tests saved to %s", stats_path)


def _run_per_class_analysis(results: list[ExperimentResult], output_dir: Path) -> list[dict[str, Any]]:
    """Run per-class F1 bootstrap analysis for H2.

    Compares the best lexical+emb variant against the best lexical-only variant
    per class, computing bootstrap CI for the F1 difference.

    Returns list of statistical test result dicts.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from stats.statistical_tests import bootstrap_ci

    # Find best lexical-only and best lexical+emb variants
    lex_variants = [r for r in results if r.config.get("features") == "lexical"]
    emb_variants = [r for r in results if r.config.get("features") == "lexical+emb"]

    if not lex_variants or not emb_variants:
        return []

    best_lex = max(lex_variants, key=lambda r: r.metrics.get("macro_f1", 0))
    best_emb = max(emb_variants, key=lambda r: r.metrics.get("macro_f1", 0))

    if not best_lex.per_query_scores or not best_emb.per_query_scores:
        return []

    # Extract per-class F1 keys
    class_names = sorted(k.replace("f1_", "") for k in best_emb.metrics if k.startswith("f1_"))

    stat_results: list[dict[str, Any]] = []
    significant_count = 0

    for cls in class_names:
        f1_lex = best_lex.metrics.get(f"f1_{cls}", 0.0)
        f1_emb = best_emb.metrics.get(f"f1_{cls}", 0.0)
        diff = f1_emb - f1_lex

        # Bootstrap CI on the F1 difference for this class
        # We use the per-sample scores to compute bootstrap resampled F1
        try:
            ci = bootstrap_ci(
                best_emb.per_query_scores,
                best_lex.per_query_scores,
                metric_name=f"f1_diff_{cls}",
                n_bootstrap=10000,
            )
            is_significant = ci.ci_lower > 0 or ci.ci_upper < 0
            if is_significant:
                significant_count += 1

            stat_results.append(
                {
                    "comparison": f"{best_emb.variant_name} vs {best_lex.variant_name} ({cls})",
                    "test": "Per-class Bootstrap CI",
                    "class": cls,
                    "f1_emb": f1_emb,
                    "f1_lex": f1_lex,
                    "f1_diff": diff,
                    "ci_lower": ci.ci_lower,
                    "ci_upper": ci.ci_upper,
                    "significant": is_significant,
                    "summary": f"{cls}: emb F1={f1_emb:.3f} vs lex F1={f1_lex:.3f} (diff={diff:.3f}, "
                    f"95% CI=[{ci.ci_lower:.4f}, {ci.ci_upper:.4f}], "
                    f"{'significant' if is_significant else 'not significant'})",
                }
            )
        except Exception as e:
            logger.warning("Per-class bootstrap failed for %s: %s", cls, e)

    # Summary: how many classes show significant improvement
    pct_significant = significant_count / len(class_names) * 100 if class_names else 0
    stat_results.append(
        {
            "test": "H2 Per-class Summary",
            "n_classes": len(class_names),
            "n_significant": significant_count,
            "pct_significant": pct_significant,
            "h2_criterion_met": pct_significant >= 60,
            "summary": (
                f"H2 per-class analysis: {significant_count}/{len(class_names)} classes "
                f"({pct_significant:.0f}%) show significant improvement. "
                f"Criterion (≥60%): {'MET' if pct_significant >= 60 else 'NOT MET'}"
            ),
        }
    )

    logger.info(
        "H2 per-class: %d/%d significant (%.0f%%), criterion %s",
        significant_count,
        len(class_names),
        pct_significant,
        "MET" if pct_significant >= 60 else "NOT MET",
    )

    return stat_results


if __name__ == "__main__":
    main()
