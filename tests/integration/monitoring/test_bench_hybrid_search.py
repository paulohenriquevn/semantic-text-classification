"""Integration test for the M5 benchmark harness (T4.1) — p95 + metrics shape, hermetic.

Uses the deterministic embedder so the run is download-free and fast. Asserts the p95 < 200 ms DoD
on the SQL+fusion read path under concurrent ingest, and that the metrics carry the hybrid-vs-BM25
comparison. Relevance QUALITY (hybrid ≥ BM25) is only meaningful with real embeddings — that run is
driven manually via `python experiments/scripts/bench_hybrid_search.py` and recorded as evidence.
"""

import psycopg
import pytest
from experiments.scripts.bench_hybrid_search import run_benchmark

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.embedder import DeterministicEmbedder
from talkex.monitoring.infrastructure.pool import MonitoringPool

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn


class TestBenchmark:
    async def test_benchmark_emits_p95_and_baseline(self, conn: psycopg.AsyncConnection) -> None:
        pool = MonitoringPool(DSN, min_size=2, max_size=8)
        await pool.open()
        try:
            result = await run_benchmark(conn, pool, DeterministicEmbedder(), latency_samples=40)
        finally:
            await pool.close()

        metrics = result.to_dict()
        assert "p95_ms" in metrics and metrics["p95_ms"] > 0
        assert "hybrid_mrr" in metrics and "bm25_mrr" in metrics  # the mandatory BM25 comparison
        assert metrics["corpus_size"] == 15
        assert metrics["samples"] == 40

    async def test_search_p95_under_concurrent_ingest(self, conn: psycopg.AsyncConnection) -> None:
        pool = MonitoringPool(DSN, min_size=2, max_size=8)
        await pool.open()
        try:
            result = await run_benchmark(conn, pool, DeterministicEmbedder(), latency_samples=60)
        finally:
            await pool.close()

        assert result.p95_ms < 200.0, f"p95 {result.p95_ms:.1f}ms exceeds the 200ms DoD"
