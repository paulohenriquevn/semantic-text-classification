"""Build BM25 and vector indexes from the unified corpus.

Loads conversations from the corpus, generates embeddings at conversation
level (concatenated turn texts), and builds both a BM25 lexical index and
an in-memory vector index. Persists both indexes to disk for use by
experiment scripts.

Usage:
    python experiments/scripts/build_index.py
    python experiments/scripts/build_index.py --corpus demo/data/conversations.jsonl
    python experiments/scripts/build_index.py --use-null-embeddings  # fast, for testing
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CORPUS = "demo/data/conversations.jsonl"
DEFAULT_INDEX_DIR = "experiments/indexes"


def load_corpus(path: Path) -> list[dict]:
    """Load conversations from JSONL corpus."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d conversations from %s", len(records), path)
    return records


def extract_conversation_text(record: dict) -> str:
    """Extract text from a conversation record.

    Supports both flat format (text field) and structured format (conversation turns).
    """
    if "text" in record and isinstance(record["text"], str) and record["text"]:
        return record["text"]
    turns = record.get("conversation", [])
    return " ".join(t.get("text", "") for t in turns if isinstance(t, dict))


@click.command()
@click.option("--corpus", default=DEFAULT_CORPUS, help="Path to unified corpus JSONL.")
@click.option("--index-dir", default=DEFAULT_INDEX_DIR, help="Output directory for indexes.")
@click.option("--model-name", default="null-384", help="Embedding model name (or 'null-384' for testing).")
@click.option("--dimensions", default=384, type=int, help="Embedding dimensions.")
@click.option(
    "--use-null-embeddings",
    is_flag=True,
    default=False,
    help="Use NullEmbeddingGenerator (fast, deterministic, no GPU needed).",
)
@click.option("--batch-size", default=32, type=int, help="Embedding batch size.")
def main(
    corpus: str,
    index_dir: str,
    model_name: str,
    dimensions: int,
    use_null_embeddings: bool,
    batch_size: int,
) -> None:
    """Build BM25 and vector indexes from the unified corpus."""
    from talkex.embeddings.config import EmbeddingModelConfig
    from talkex.embeddings.generator import NullEmbeddingGenerator, SentenceTransformerGenerator
    from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
    from talkex.models.enums import ObjectType
    from talkex.retrieval.bm25 import InMemoryBM25Index
    from talkex.retrieval.config import LexicalIndexConfig, VectorIndexConfig
    from talkex.retrieval.vector_index import InMemoryVectorIndex

    corpus_path = Path(corpus)
    output_path = Path(index_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise click.ClickException(f"Corpus not found: {corpus_path}. Run build_splits.py first.")

    records = load_corpus(corpus_path)
    if not records:
        raise click.ClickException("Corpus is empty.")

    # Extract texts and IDs
    doc_ids = [r.get("conversation_id", r.get("id", f"conv_{i}")) for i, r in enumerate(records)]
    texts = [extract_conversation_text(r) for r in records]
    labels = [r.get("topic", "outros") for r in records]

    # --- Build BM25 index ---
    logger.info("Building BM25 index...")
    t0 = time.perf_counter()
    bm25_config = LexicalIndexConfig()
    bm25_index = InMemoryBM25Index(config=bm25_config)
    bm25_docs = [{"doc_id": did, "text": txt} for did, txt in zip(doc_ids, texts, strict=True)]
    bm25_index.index(bm25_docs)
    bm25_dur = (time.perf_counter() - t0) * 1000
    logger.info("BM25 index built: %d docs in %.1fms", len(doc_ids), bm25_dur)

    # --- Build embedding generator ---
    emb_config = EmbeddingModelConfig(
        model_name=model_name,
        model_version="1.0",
        batch_size=batch_size,
    )

    if use_null_embeddings or model_name == "null-384":
        logger.info("Using NullEmbeddingGenerator (deterministic, no model loading)")
        emb_gen = NullEmbeddingGenerator(model_config=emb_config, dimensions=dimensions)
    else:
        logger.info("Loading SentenceTransformer model: %s", model_name)
        emb_gen = SentenceTransformerGenerator(model_config=emb_config)

    # --- Generate embeddings ---
    logger.info("Generating embeddings for %d conversations...", len(doc_ids))
    t0 = time.perf_counter()

    all_records = []
    for batch_start in range(0, len(doc_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(doc_ids))
        batch_inputs = [
            EmbeddingInput(
                embedding_id=f"emb_{doc_ids[i]}",
                object_type=ObjectType.CONVERSATION,
                object_id=doc_ids[i],
                text=texts[i],
            )
            for i in range(batch_start, batch_end)
        ]
        batch = EmbeddingBatch(items=batch_inputs)
        batch_records = emb_gen.generate(batch)
        all_records.extend(batch_records)

        if (batch_start // batch_size) % 10 == 0:
            logger.info(
                "  Progress: %d/%d embeddings generated", len(all_records), len(doc_ids)
            )

    emb_dur = (time.perf_counter() - t0) * 1000
    logger.info("Embeddings generated: %d records in %.1fms", len(all_records), emb_dur)

    # --- Build vector index ---
    logger.info("Building vector index...")
    t0 = time.perf_counter()
    actual_dims = all_records[0].dimensions if all_records else dimensions
    vec_config = VectorIndexConfig(dimensions=actual_dims)
    vec_index = InMemoryVectorIndex(config=vec_config)
    vec_index.upsert(all_records)
    vec_dur = (time.perf_counter() - t0) * 1000
    logger.info("Vector index built: %d vectors in %.1fms", vec_index.vector_count, vec_dur)

    # --- Persist indexes ---
    vec_index_path = output_path / "vector_index"
    vec_index.save(vec_index_path)
    logger.info("Vector index saved to %s", vec_index_path)

    # Save manifest
    manifest = {
        "corpus_path": str(corpus_path),
        "n_documents": len(doc_ids),
        "embedding_model": model_name,
        "embedding_dimensions": actual_dims,
        "use_null_embeddings": use_null_embeddings or model_name == "null-384",
        "bm25_config": {"k1": bm25_config.k1, "b": bm25_config.b},
        "vector_index_path": str(vec_index_path),
        "label_distribution": {},
        "build_time_ms": {
            "bm25": round(bm25_dur, 1),
            "embeddings": round(emb_dur, 1),
            "vector_index": round(vec_dur, 1),
        },
    }
    # Count label distribution
    from collections import Counter

    manifest["label_distribution"] = dict(Counter(labels))

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest saved to %s", manifest_path)

    logger.info("=" * 60)
    logger.info("Index build complete!")
    logger.info("  BM25: %d docs (%.1fms)", len(doc_ids), bm25_dur)
    logger.info("  Vector: %d vectors, %d dims (%.1fms)", vec_index.vector_count, actual_dims, vec_dur)
    logger.info("  Output: %s", output_path)


if __name__ == "__main__":
    main()
