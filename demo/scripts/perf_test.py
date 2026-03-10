"""Performance test — measures search latency p50/p95/p99 over the demo index.

Loads pre-built indexes (output of build_index.py) and executes a battery
of realistic search queries to measure latency percentiles.  Results are
printed as a table and optionally saved to JSON.

Usage:
    python demo/scripts/perf_test.py
    python demo/scripts/perf_test.py --index-dir demo/data/index --queries 200 --output demo/data/perf_results.json
"""

from __future__ import annotations

import json
import logging
import pickle
import statistics
import time
from pathlib import Path

import click
import numpy as np

from talkex.embeddings.config import EmbeddingModelConfig
from talkex.embeddings.generator import NullEmbeddingGenerator
from talkex.models.enums import PoolingStrategy
from talkex.retrieval.config import DistanceMetric, HybridRetrievalConfig, VectorIndexConfig
from talkex.retrieval.hybrid import SimpleHybridRetriever
from talkex.retrieval.models import QueryType, RetrievalQuery
from talkex.retrieval.qdrant import QdrantVectorIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Realistic search queries spanning different domains and intents
SAMPLE_QUERIES = [
    "customer complaint about billing",
    "agent offered a refund",
    "call center wait time",
    "issue with internet connection",
    "subscription cancellation request",
    "technical support for router",
    "payment failed credit card",
    "shipping delay tracking number",
    "warranty claim for defective product",
    "password reset account locked",
    "how to upgrade my plan",
    "return policy for electronics",
    "agent escalated to supervisor",
    "customer asked for discount",
    "network outage in area",
    "installation appointment scheduling",
    "service interruption notification",
    "loyalty program benefits",
    "international calling rates",
    "data usage exceeded limit",
    "voicemail setup instructions",
    "transfer to another department",
    "contract renewal terms",
    "emergency service request",
    "feedback about customer experience",
    "billing dispute resolution",
    "account verification process",
    "promotional offer eligibility",
    "troubleshooting steps provided",
    "customer satisfaction survey",
]


def _percentile(data: list[float], p: float) -> float:
    """Calculate the p-th percentile of a list of values."""
    return float(np.percentile(data, p))


def _load_retriever(index_dir: str) -> SimpleHybridRetriever:
    """Load pre-built indexes and wire a retriever."""
    index_path = Path(index_dir)

    manifest_path = index_path / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # BM25
    bm25_path = index_path / "bm25_index.pkl"
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)

    # Qdrant
    dims = manifest.get("dimensions", 384)
    vec_config = VectorIndexConfig(dimensions=dims, metric=DistanceMetric.COSINE)
    qdrant_index = QdrantVectorIndex(
        config=vec_config,
        collection_name="talkex_demo",
        path=str(index_path / "qdrant_storage"),
    )

    # Embedding generator (same as used during indexing)
    emb_config = EmbeddingModelConfig(
        model_name="null",
        model_version="1.0",
        pooling_strategy=PoolingStrategy.MEAN,
    )
    emb_gen = NullEmbeddingGenerator(model_config=emb_config, dimensions=dims)

    return SimpleHybridRetriever(
        lexical_index=bm25_index,
        vector_index=qdrant_index,
        embedding_generator=emb_gen,
        config=HybridRetrievalConfig(),
    )


def _run_benchmark(
    retriever: SimpleHybridRetriever,
    queries: list[str],
    top_k: int,
    query_type: QueryType,
) -> dict:
    """Run a set of queries and collect latency measurements.

    Returns dict with latency stats and hit counts.
    """
    latencies: list[float] = []
    hit_counts: list[int] = []

    # Warm-up: 3 queries to prime caches and JIT paths
    for q in queries[:3]:
        retriever.retrieve(RetrievalQuery(query_text=q, top_k=top_k, query_type=query_type))

    for q in queries:
        start = time.monotonic()
        result = retriever.retrieve(RetrievalQuery(query_text=q, top_k=top_k, query_type=query_type))
        elapsed_ms = (time.monotonic() - start) * 1000
        latencies.append(elapsed_ms)
        hit_counts.append(len(result.hits))

    return {
        "query_type": query_type.value,
        "query_count": len(queries),
        "top_k": top_k,
        "latency_p50_ms": round(_percentile(latencies, 50), 2),
        "latency_p95_ms": round(_percentile(latencies, 95), 2),
        "latency_p99_ms": round(_percentile(latencies, 99), 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
        "avg_hits": round(statistics.mean(hit_counts), 1),
    }


@click.command()
@click.option("--index-dir", default="demo/data/index", help="Index directory (output of build_index.py).")
@click.option("--queries", "num_queries", default=100, type=int, help="Number of query iterations.")
@click.option("--top-k", default=20, type=int, help="Top-K results per query.")
@click.option("--output", "output_path", default=None, help="Optional JSON file for results.")
def main(index_dir: str, num_queries: int, top_k: int, output_path: str | None) -> None:
    """Run performance benchmarks on the demo search indexes."""
    logger.info("Loading indexes from %s...", index_dir)
    retriever = _load_retriever(index_dir)

    # Build query list by cycling through samples
    queries = [SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)] for i in range(num_queries)]

    logger.info("Running %d queries (top_k=%d)...", num_queries, top_k)

    # Benchmark each query type
    results = {}
    for qt in [QueryType.LEXICAL, QueryType.SEMANTIC, QueryType.HYBRID]:
        logger.info("  Benchmarking %s...", qt.value)
        stats = _run_benchmark(retriever, queries, top_k, qt)
        results[qt.value] = stats

    # Print results table
    print()
    header = (
        f"{'Mode':<12} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}"
        f" {'mean ms':>8} {'min ms':>8} {'max ms':>8} {'avg hits':>9}"
    )
    print(header)
    print("─" * 80)
    for mode, stats in results.items():
        print(
            f"{mode:<12} "
            f"{stats['latency_p50_ms']:>8.2f} "
            f"{stats['latency_p95_ms']:>8.2f} "
            f"{stats['latency_p99_ms']:>8.2f} "
            f"{stats['latency_mean_ms']:>8.2f} "
            f"{stats['latency_min_ms']:>8.2f} "
            f"{stats['latency_max_ms']:>8.2f} "
            f"{stats['avg_hits']:>9.1f}"
        )
    print()

    # PRD target: < 200ms p95 for hybrid search
    hybrid_p95 = results["hybrid"]["latency_p95_ms"]
    target_ms = 200.0
    status = "PASS" if hybrid_p95 < target_ms else "FAIL"
    logger.info("PRD target (hybrid p95 < %.0f ms): %s (actual: %.2f ms)", target_ms, status, hybrid_p95)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
