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
from pathlib import Path
from typing import Any

import click
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLITS_DIR = Path("demo/data/splits")
RESULTS_DIR = Path("experiments/results")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_VERSION = "1.0"


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
    return [r.get("topic", "outros") for r in records]


def _get_record_id(record: dict, fallback: str) -> str:
    """Get conversation ID from a record, supporting both formats."""
    return record.get("conversation_id", record.get("id", fallback))


def _estimate_turn_count(record: dict) -> int:
    """Estimate turn count from a record.

    Supports structured format (conversation list) and flat format
    (count [customer]/[agent] markers in text).
    """
    turns = record.get("conversation", [])
    if turns:
        return len(turns)
    import re
    text = record.get("text", "")
    markers = len(re.findall(r"\[(customer|agent)\]", text, re.IGNORECASE))
    return max(markers, 1)


# ---------------------------------------------------------------------------
# Embedding generation via talkex pipeline
# ---------------------------------------------------------------------------


def _make_embedding_generator() -> "SentenceTransformerGenerator":
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
    generator: "SentenceTransformerGenerator",
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


def run_h1(train: list[dict], test: list[dict]) -> list[ExperimentResult]:
    """Run H1 experiments: hybrid retrieval superiority.

    Compares BM25-base, BM25-norm, ANN (NullEmbedding), Hybrid-linear, Hybrid-RRF
    on retrieval metrics (Recall@K, MRR, nDCG).

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

    # --- Hybrid-LINEAR at different fusion weights ---
    for alpha in [0.3, 0.5, 0.65, 0.8]:
        logger.info("Running Hybrid-LINEAR (alpha=%.2f)...", alpha)
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

        def _make_linear_search(retriever: SimpleHybridRetriever):
            def search_fn(query: str, top_k: int) -> list[tuple[str, float]]:
                q = RetrievalQuery(query_text=query, top_k=top_k, query_type=QueryType.HYBRID)
                result = retriever.retrieve(q)
                return [(h.object_id, h.score) for h in result.hits]

            return search_fn

        metrics_lin, scores_lin, dur_lin = _evaluate_retriever_fn(
            _make_linear_search(hybrid_lin), test_texts, test_labels, test_ids, intent_to_train_ids
        )
        results.append(
            ExperimentResult(
                hypothesis="H1",
                variant_name=f"Hybrid-LINEAR-a{alpha:.2f}",
                metrics=metrics_lin,
                config={"type": "hybrid", "fusion": "linear", "alpha": alpha},
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
    """Run H2 experiments: multi-level representation gains.

    Compares lexical-only vs lexical+embedding features
    across LogReg, LightGBM, MLP classifiers.
    Uses talkex SentenceTransformerGenerator for embedding generation.
    """
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.logistic import LogisticRegressionClassifier
    from talkex.classification.mlp_classifier import MLPClassifier
    from talkex.classification.models import ClassificationInput
    from talkex.classification_eval.dataset import ClassificationDataset, ClassificationExample, GroundTruthLabel
    from talkex.classification_eval.runner import ClassificationBenchmarkRunner

    logger.info("=" * 60)
    logger.info("H2: Multi-Level Classification Experiment")
    logger.info("=" * 60)

    train_labels = extract_labels(train)
    test_labels = extract_labels(test)
    train_texts = extract_texts(train)
    test_texts_h2 = extract_texts(test)
    train_ids = [_get_record_id(r, f"train_{i}") for i, r in enumerate(train)]
    test_ids = [_get_record_id(r, f"test_{i}") for i, r in enumerate(test)]
    unique_labels = sorted(set(train_labels + test_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # Feature extraction: lexical + structural
    from talkex.classification.features import extract_lexical_features, extract_structural_features, merge_feature_sets

    def make_lexical_features(records: list[dict]) -> tuple[list[dict[str, float]], list[str]]:
        texts = extract_texts(records)
        all_features = []
        for i, text in enumerate(texts):
            r = records[i]
            lex = extract_lexical_features(text)
            struct = extract_structural_features(
                is_customer=True,
                is_agent=True,
                turn_count=_estimate_turn_count(r),
                speaker_count=2,
            )
            merged = merge_feature_sets(lex, struct)
            all_features.append(merged.features)
        feature_names = list(all_features[0].keys()) if all_features else []
        return all_features, feature_names

    train_lex_features, lex_feature_names = make_lexical_features(train)
    test_lex_features, _ = make_lexical_features(test)

    # Generate embeddings via talkex pipeline
    logger.info("Generating embeddings via talkex SentenceTransformerGenerator (%s)...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()
    train_embeddings = generate_embeddings_via_talkex(train_texts, train_ids, emb_gen)
    test_embeddings = generate_embeddings_via_talkex(test_texts_h2, test_ids, emb_gen)
    emb_dims = train_embeddings.shape[1]
    logger.info("Embeddings generated: %d dims (via talkex pipeline)", emb_dims)

    # Build feature sets: lexical-only and lexical+embedding
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
    test_full_features, _ = merge_with_embeddings(test_lex_features, test_embeddings)

    # Build feature configurations to test
    feature_configs = {
        "lexical": (train_lex_features, test_lex_features, lex_feature_names),
        "lexical+emb": (train_full_features, test_full_features, full_feature_names),
    }

    results: list[ExperimentResult] = []

    for feat_name, (tr_feats, te_feats, feat_names) in feature_configs.items():
        # Build inputs for this feature config
        train_inputs = [
            ClassificationInput(
                source_id=_get_record_id(train[i], f"train_{i}"),
                source_type="conversation",
                text=train_texts[i],
                features=tr_feats[i],
            )
            for i in range(len(train))
        ]

        test_examples = [
            ClassificationExample(
                example_id=_get_record_id(test[i], f"test_{i}"),
                source_type="conversation",
                text=test_texts_h2[i],
                features=te_feats[i],
                ground_truth=[GroundTruthLabel(label=test_labels[i])],
            )
            for i in range(len(test))
        ]

        eval_dataset = ClassificationDataset(
            name=f"talkex-h2-{feat_name}",
            version="1.0",
            examples=test_examples,
            label_names=unique_labels,
        )
        runner = ClassificationBenchmarkRunner(dataset=eval_dataset)

        # Define classifiers
        classifier_configs = {
            "LogReg": lambda fn=feat_names: LogisticRegressionClassifier(
                label_space=label_space,
                feature_names=fn,
                model_name="logistic-regression",
                sklearn_kwargs={"max_iter": 2000},
            ),
            "LightGBM": lambda fn=feat_names: LightGBMClassifier(
                label_space=label_space,
                feature_names=fn,
                model_name="lightgbm",
                lgbm_kwargs={"n_estimators": 100, "num_leaves": 31, "verbosity": -1},
            ),
            "MLP": lambda fn=feat_names: MLPClassifier(
                label_space=label_space,
                feature_names=fn,
                model_name="mlp",
                hidden_layer_sizes=(128, 64),
                sklearn_kwargs={"max_iter": 1000, "random_state": 42},
            ),
        }

        for clf_name, clf_factory in classifier_configs.items():
            logger.info("Training %s (%s)...", clf_name, feat_name)
            t0 = time.perf_counter()
            clf = clf_factory()
            clf.fit(train_inputs, train_labels)

            method_result = runner.evaluate(clf, clf_name)
            dur = (time.perf_counter() - t0) * 1000

            metrics = method_result.aggregated.copy()
            # Add per-label F1
            for label_name, label_metrics in method_result.per_label.items():
                metrics[f"f1_{label_name}"] = label_metrics.get("f1", 0.0)

            results.append(
                ExperimentResult(
                    hypothesis="H2",
                    variant_name=f"{feat_name}_{clf_name}",
                    metrics=metrics,
                    config={"classifier": clf_name, "features": feat_name},
                    duration_ms=dur,
                )
            )
            logger.info(
                "  %s/%s: macro_f1=%.4f, micro_f1=%.4f",
                feat_name, clf_name, metrics.get("macro_f1", 0), metrics.get("micro_f1", 0),
            )

    logger.info("H2 complete: %d variants evaluated", len(results))
    return results


# ---------------------------------------------------------------------------
# H4: Cascaded Inference
# ---------------------------------------------------------------------------


def run_h4(train: list[dict], test: list[dict]) -> list[ExperimentResult]:
    """Run H4 experiments: cascaded inference efficiency.

    Compares uniform pipeline vs cascaded pipeline at different thresholds.
    Stage 1 (lightweight): LogReg with lexical+emb features (cheap linear model).
    Stage 2 (full): LightGBM with lexical+emb features (expensive ensemble).
    Uses talkex SentenceTransformerGenerator for embedding generation.
    """
    logger.info("=" * 60)
    logger.info("H4: Cascaded Inference Experiment")
    logger.info("=" * 60)

    from talkex.classification.features import extract_lexical_features, extract_structural_features, merge_feature_sets
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.logistic import LogisticRegressionClassifier
    from talkex.classification.models import ClassificationInput

    train_labels = extract_labels(train)
    test_labels = extract_labels(test)
    train_texts = extract_texts(train)
    test_texts_h4 = extract_texts(test)
    train_ids = [_get_record_id(r, f"train_{i}") for i, r in enumerate(train)]
    test_ids = [_get_record_id(r, f"test_{i}") for i, r in enumerate(test)]
    unique_labels = sorted(set(train_labels + test_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # Build lexical features (for lightweight classifier)
    def build_lex_inputs(records: list[dict], texts: list[str]) -> tuple[list[ClassificationInput], list[str]]:
        inputs = []
        feature_names_ref: list[str] = []
        for i, text in enumerate(texts):
            r = records[i]
            lex = extract_lexical_features(text)
            struct = extract_structural_features(
                is_customer=True,
                is_agent=True,
                turn_count=_estimate_turn_count(r),
                speaker_count=2,
            )
            merged = merge_feature_sets(lex, struct)
            if not feature_names_ref:
                feature_names_ref.extend(merged.features.keys())
            inputs.append(
                ClassificationInput(
                    source_id=_get_record_id(r, f"rec_{i}"),
                    source_type="conversation",
                    text=text,
                    features=merged.features,
                )
            )
        return inputs, feature_names_ref

    train_lex_inputs, lex_feature_names = build_lex_inputs(train, train_texts)
    test_lex_inputs, _ = build_lex_inputs(test, test_texts_h4)

    # Generate embeddings via talkex pipeline
    logger.info("Generating embeddings via talkex SentenceTransformerGenerator (%s)...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()
    train_embs = generate_embeddings_via_talkex(train_texts, train_ids, emb_gen)
    test_embs = generate_embeddings_via_talkex(test_texts_h4, test_ids, emb_gen)

    def build_full_inputs(
        lex_inputs: list[ClassificationInput], embeddings: np.ndarray
    ) -> tuple[list[ClassificationInput], list[str]]:
        full_inputs = []
        full_names: list[str] = []
        for i, inp in enumerate(lex_inputs):
            features = dict(inp.features)
            for d in range(embeddings.shape[1]):
                features[f"emb_{d}"] = float(embeddings[i][d])
            if not full_names:
                full_names = list(features.keys())
            full_inputs.append(
                ClassificationInput(
                    source_id=inp.source_id,
                    source_type=inp.source_type,
                    text=inp.text,
                    features=features,
                )
            )
        return full_inputs, full_names

    train_full_inputs, full_feature_names = build_full_inputs(train_lex_inputs, train_embs)
    test_full_inputs, _ = build_full_inputs(test_lex_inputs, test_embs)

    # Train lightweight (LogReg with embeddings, fast inference) and full (LightGBM with embeddings, expensive)
    # Both use the same features; the cost difference is model complexity (linear vs ensemble)
    logger.info("Training lightweight classifier (LogReg, lexical+emb)...")
    light_clf = LogisticRegressionClassifier(
        label_space=label_space,
        feature_names=full_feature_names,
        sklearn_kwargs={"max_iter": 2000},
    )
    light_clf.fit(train_full_inputs, train_labels)

    logger.info("Training full classifier (LightGBM, lexical+emb)...")
    full_clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=full_feature_names,
        lgbm_kwargs={"n_estimators": 200, "num_leaves": 63, "verbosity": -1},
    )
    full_clf.fit(train_full_inputs, train_labels)

    results: list[ExperimentResult] = []

    # --- Measure per-sample inference cost for each model ---
    # Run each model multiple times to get stable per-sample cost estimates
    n_test = len(test_full_inputs)
    n_warmup = 3
    n_measure = 10

    for _ in range(n_warmup):
        light_clf.classify(test_full_inputs)
        full_clf.classify(test_full_inputs)

    light_times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        light_clf.classify(test_full_inputs)
        light_times.append((time.perf_counter() - t0) * 1000)
    light_cost_per_sample = float(np.median(light_times)) / n_test

    full_times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        full_clf.classify(test_full_inputs)
        full_times.append((time.perf_counter() - t0) * 1000)
    full_cost_per_sample = float(np.median(full_times)) / n_test

    logger.info(
        "Per-sample cost: light=%.4f ms, full=%.4f ms (ratio=%.1fx)",
        light_cost_per_sample,
        full_cost_per_sample,
        full_cost_per_sample / light_cost_per_sample if light_cost_per_sample > 0 else 0,
    )

    # --- Uniform pipeline (full classifier on all) ---
    logger.info("Running uniform pipeline...")
    full_results = full_clf.classify(test_full_inputs)
    uniform_preds = [r.top_label for r in full_results]
    uniform_f1 = _compute_macro_f1(uniform_preds, test_labels, unique_labels)
    uniform_cost = n_test * full_cost_per_sample

    results.append(
        ExperimentResult(
            hypothesis="H4",
            variant_name="uniform",
            metrics={
                "macro_f1": uniform_f1,
                "cost_ms": uniform_cost,
                "pct_stage1": 0,
                "pct_stage2": 100,
                "light_cost_per_sample_ms": light_cost_per_sample,
                "full_cost_per_sample_ms": full_cost_per_sample,
            },
            config={"type": "uniform", "classifier": "lightgbm-200t"},
            duration_ms=uniform_cost,
        )
    )

    # --- Cascaded pipeline ---
    # Thresholds tuned for 9-class problem: LogReg with embeddings can reach ~0.4-0.7 on easy cases
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        logger.info("Running cascaded pipeline (threshold=%.2f)...", threshold)

        # Stage 1: lightweight classifier (LogReg, fast linear inference)
        light_results = light_clf.classify(test_full_inputs)
        cascaded_preds = []
        stage2_indices = []

        for idx, lr in enumerate(light_results):
            if lr.top_score >= threshold:
                cascaded_preds.append((idx, lr.top_label))
            else:
                stage2_indices.append(idx)

        # Stage 2: full classifier on unresolved (LightGBM, expensive ensemble)
        if stage2_indices:
            stage2_inputs = [test_full_inputs[i] for i in stage2_indices]
            stage2_results = full_clf.classify(stage2_inputs)
            for i, sr in zip(stage2_indices, stage2_results, strict=True):
                cascaded_preds.append((i, sr.top_label))

        cascaded_preds.sort(key=lambda x: x[0])
        final_preds = [p for _, p in cascaded_preds]

        cascade_f1 = _compute_macro_f1(final_preds, test_labels, unique_labels)
        pct_stage1 = (n_test - len(stage2_indices)) / n_test * 100
        # Theoretical cost: all samples go through light, only stage2 goes through full
        cascade_cost = n_test * light_cost_per_sample + len(stage2_indices) * full_cost_per_sample
        cost_reduction = (1 - cascade_cost / uniform_cost) * 100 if uniform_cost > 0 else 0
        f1_delta = uniform_f1 - cascade_f1

        results.append(
            ExperimentResult(
                hypothesis="H4",
                variant_name=f"cascade_t{threshold:.2f}",
                metrics={
                    "macro_f1": cascade_f1,
                    "cost_ms": cascade_cost,
                    "pct_stage1": pct_stage1,
                    "pct_stage2": 100 - pct_stage1,
                    "cost_reduction_pct": cost_reduction,
                    "f1_delta": f1_delta,
                },
                config={"type": "cascade", "threshold": threshold, "light": "logreg", "full": "lightgbm-200t"},
                duration_ms=cascade_cost,
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


# ---------------------------------------------------------------------------
# H3: Rules Complement ML
# ---------------------------------------------------------------------------


def run_h3(train: list[dict], test: list[dict]) -> list[ExperimentResult]:
    """Run H3 experiments: deterministic rules complement ML.

    Compares ML-only, Rules-only, and ML+Rules-override on critical classes
    (cancelamento, reclamacao) using precision/recall/F1.
    Uses lexical + embedding features for the ML classifier (same as H2 best config).
    Uses talkex SentenceTransformerGenerator for embedding generation.
    """
    from talkex.classification.features import extract_lexical_features, extract_structural_features, merge_feature_sets
    from talkex.classification.labels import LabelSpace
    from talkex.classification.lightgbm_classifier import LightGBMClassifier
    from talkex.classification.models import ClassificationInput
    from talkex.rules.ast import AndNode, PredicateNode
    from talkex.rules.config import PredicateType, RuleEngineConfig
    from talkex.rules.evaluator import SimpleRuleEvaluator
    from talkex.rules.models import RuleDefinition, RuleEvaluationInput

    logger.info("=" * 60)
    logger.info("H3: Rules Complement ML Experiment")
    logger.info("=" * 60)

    train_labels = extract_labels(train)
    test_labels = extract_labels(test)
    train_texts = extract_texts(train)
    test_texts = extract_texts(test)
    train_ids = [_get_record_id(r, f"train_{i}") for i, r in enumerate(train)]
    test_ids = [_get_record_id(r, f"test_{i}") for i, r in enumerate(test)]
    unique_labels = sorted(set(train_labels + test_labels))
    label_space = LabelSpace(labels=unique_labels, default_threshold=0.3)

    # --- Generate embeddings via talkex pipeline ---
    logger.info("Generating embeddings via talkex SentenceTransformerGenerator (%s)...", EMBEDDING_MODEL)
    emb_gen = _make_embedding_generator()
    train_embs = generate_embeddings_via_talkex(train_texts, train_ids, emb_gen)
    test_embs = generate_embeddings_via_talkex(test_texts, test_ids, emb_gen)
    logger.info("Embeddings generated: %d dims (via talkex pipeline)", train_embs.shape[1])

    # --- Build features (lexical + structural + embeddings) and train ML classifier ---
    def build_inputs(records: list[dict], texts: list[str], embeddings: np.ndarray) -> tuple[list[ClassificationInput], list[str]]:
        inputs = []
        feature_names_ref: list[str] = []
        for i, text in enumerate(texts):
            r = records[i]
            lex = extract_lexical_features(text)
            struct = extract_structural_features(
                is_customer=True,
                is_agent=True,
                turn_count=_estimate_turn_count(r),
                speaker_count=2,
            )
            merged = merge_feature_sets(lex, struct)
            features = dict(merged.features)
            # Add embedding dimensions
            for d in range(embeddings.shape[1]):
                features[f"emb_{d}"] = float(embeddings[i][d])
            if not feature_names_ref:
                feature_names_ref.extend(features.keys())
            inputs.append(
                ClassificationInput(
                    source_id=_get_record_id(r, f"rec_{i}"),
                    source_type="conversation",
                    text=text,
                    features=features,
                )
            )
        return inputs, feature_names_ref

    train_inputs, feature_names = build_inputs(train, train_texts, train_embs)
    test_inputs, _ = build_inputs(test, test_texts, test_embs)

    logger.info("Training ML classifier (LightGBM, lexical+emb)...")
    ml_clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=feature_names,
        lgbm_kwargs={"n_estimators": 100, "num_leaves": 31, "verbosity": -1},
    )
    ml_clf.fit(train_inputs, train_labels)

    # --- Define deterministic rules for critical classes ---
    # Rule: cancelamento — keywords indicating cancellation intent
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

    # Rule: reclamacao — keywords indicating complaints
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
                    value=["reclamacao", "reclamar", "reclamando", "absurdo", "inadmissivel", "procon", "anatel", "reclame aqui", "ouvidoria", "insatisfeito", "insatisfeita", "revoltado", "revoltada", "indignado", "indignada"],
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

    # --- ML-only baseline ---
    logger.info("Evaluating ML-only...")
    t0 = time.perf_counter()
    ml_results = ml_clf.classify(test_inputs)
    ml_preds = [r.top_label for r in ml_results]
    ml_dur = (time.perf_counter() - t0) * 1000

    ml_metrics = _compute_per_class_metrics(ml_preds, test_labels, unique_labels)
    ml_metrics["macro_f1"] = _compute_macro_f1(ml_preds, test_labels, unique_labels)
    results.append(
        ExperimentResult(
            hypothesis="H3",
            variant_name="ML-only",
            metrics=ml_metrics,
            config={"type": "ml_only", "classifier": "lightgbm"},
            duration_ms=ml_dur,
        )
    )

    # --- Rules-only ---
    logger.info("Evaluating Rules-only...")
    t0 = time.perf_counter()
    rule_preds = []
    for i, text in enumerate(test_texts):
        eval_input = RuleEvaluationInput(
            source_id=_get_record_id(test[i], f"test_{i}"),
            source_type="conversation",
            text=text,
        )
        rule_results = evaluator.evaluate(rules, eval_input, rule_config)
        matched_rules = [rr for rr in rule_results if rr.matched]

        if matched_rules:
            # Take the first matching rule's label
            rule_preds.append(rule_to_label.get(matched_rules[0].rule_id, "outros"))
        else:
            rule_preds.append("outros")  # Default when no rule matches
    rules_dur = (time.perf_counter() - t0) * 1000

    rules_metrics = _compute_per_class_metrics(rule_preds, test_labels, unique_labels)
    rules_metrics["macro_f1"] = _compute_macro_f1(rule_preds, test_labels, unique_labels)
    results.append(
        ExperimentResult(
            hypothesis="H3",
            variant_name="Rules-only",
            metrics=rules_metrics,
            config={"type": "rules_only", "rules": list(rule_to_label.keys())},
            duration_ms=rules_dur,
        )
    )

    # --- ML + Rules-override (rules override ML on critical classes when matched) ---
    logger.info("Evaluating ML+Rules-override...")
    t0 = time.perf_counter()
    override_preds = []
    for i, text in enumerate(test_texts):
        eval_input = RuleEvaluationInput(
            source_id=_get_record_id(test[i], f"test_{i}"),
            source_type="conversation",
            text=text,
        )
        rule_results = evaluator.evaluate(rules, eval_input, rule_config)
        matched_rules = [rr for rr in rule_results if rr.matched]

        if matched_rules:
            # Rule fires → use rule's label (deterministic override)
            override_preds.append(rule_to_label.get(matched_rules[0].rule_id, ml_preds[i]))
        else:
            # No rule fires → use ML prediction
            override_preds.append(ml_preds[i])
    override_dur = (time.perf_counter() - t0) * 1000

    override_metrics = _compute_per_class_metrics(override_preds, test_labels, unique_labels)
    override_metrics["macro_f1"] = _compute_macro_f1(override_preds, test_labels, unique_labels)
    results.append(
        ExperimentResult(
            hypothesis="H3",
            variant_name="ML+Rules-override",
            metrics=override_metrics,
            config={"type": "ml_rules_override", "classifier": "lightgbm"},
            duration_ms=override_dur,
        )
    )

    # --- ML + Rules-feature (rule match as additional feature) ---
    logger.info("Evaluating ML+Rules-feature...")
    t0 = time.perf_counter()

    # Add rule-match features to training data
    def add_rule_features(records: list[dict], inputs: list[ClassificationInput]) -> list[ClassificationInput]:
        augmented = []
        for _i, inp in enumerate(inputs):
            text = inp.text
            eval_input = RuleEvaluationInput(
                source_id=inp.source_id,
                source_type="conversation",
                text=text,
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

    aug_train = add_rule_features(train, train_inputs)
    aug_test = add_rule_features(test, test_inputs)
    aug_feature_names = list(aug_train[0].features.keys()) if aug_train else feature_names

    aug_clf = LightGBMClassifier(
        label_space=label_space,
        feature_names=aug_feature_names,
        lgbm_kwargs={"n_estimators": 100, "num_leaves": 31, "verbosity": -1},
    )
    aug_clf.fit(aug_train, train_labels)
    aug_results_ml = aug_clf.classify(aug_test)
    aug_preds = [r.top_label for r in aug_results_ml]
    feature_dur = (time.perf_counter() - t0) * 1000

    feature_metrics = _compute_per_class_metrics(aug_preds, test_labels, unique_labels)
    feature_metrics["macro_f1"] = _compute_macro_f1(aug_preds, test_labels, unique_labels)
    results.append(
        ExperimentResult(
            hypothesis="H3",
            variant_name="ML+Rules-feature",
            metrics=feature_metrics,
            config={"type": "ml_rules_feature", "classifier": "lightgbm"},
            duration_ms=feature_dur,
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
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--hypothesis", required=True, type=click.Choice(["H1", "H2", "H3", "H4", "ablation", "all"]))
@click.option("--splits-dir", default="demo/data/splits", help="Directory with train/val/test splits.")
@click.option("--output-dir", default="experiments/results", help="Base output directory.")
def main(hypothesis: str, splits_dir: str, output_dir: str) -> None:
    """Run experiment(s) for a specific hypothesis."""
    global SPLITS_DIR, RESULTS_DIR
    SPLITS_DIR = Path(splits_dir)
    RESULTS_DIR = Path(output_dir)

    hypotheses = [hypothesis] if hypothesis != "all" else ["H1", "H2", "H3", "H4"]

    for h in hypotheses:
        logger.info("=" * 60)
        logger.info("Starting experiment: %s", h)
        logger.info("=" * 60)

        train = load_split("train")
        test = load_split("test")

        if h == "H1":
            results = run_h1(train, test)
        elif h == "H2":
            val = load_split("val")
            results = run_h2(train, val, test)
        elif h == "H3":
            results = run_h3(train, test)
        elif h == "H4":
            results = run_h4(train, test)
        else:
            logger.warning("Hypothesis %s not yet implemented in orchestration script", h)
            continue

        save_results(results, RESULTS_DIR / h)

        # Apply statistical tests if multiple variants
        if len(results) >= 2:
            _run_statistical_analysis(results, RESULTS_DIR / h)

    logger.info("All experiments complete!")


def _run_statistical_analysis(results: list[ExperimentResult], output_dir: Path) -> None:
    """Run statistical tests on experiment results."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from stats.statistical_tests import bootstrap_ci, wilcoxon_signed_rank

    # Find pairs with per-query scores for paired tests
    scored = [(r.variant_name, r.per_query_scores) for r in results if r.per_query_scores]
    if len(scored) < 2:
        logger.info("Not enough per-query scores for paired statistical tests")
        return

    # Compare best vs each baseline
    stat_results = []
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

                ci = bootstrap_ci(
                    best.per_query_scores,
                    r.per_query_scores,
                    metric_name="MRR_diff",
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
                logger.warning("Statistical test failed: %s", e)

    if stat_results:
        stats_path = output_dir / "statistical_tests.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stat_results, f, indent=2, ensure_ascii=False)
        logger.info("Statistical tests saved to %s", stats_path)


if __name__ == "__main__":
    main()
