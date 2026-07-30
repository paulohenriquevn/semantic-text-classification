"""M5 benchmark — p95 latency under concurrent ingest + hybrid-vs-BM25 relevance (T4.1, blueprint D4).

Two DoD-critical measurements, both against a real TimescaleDB:

1. **p95 latency** of the hybrid search while turns are ingested concurrently — must be < 200 ms.
2. **Relevance**: mean reciprocal rank (MRR@k) of the hybrid ranking vs a BM25-only baseline over a
   small labeled probe (exact + paraphrase queries). The KB axiom is: always benchmark hybrid against
   BM25 (`docs/KB.md § BM25 baseline`). Honest by design — if hybrid does NOT beat BM25 (e.g. with the
   deterministic embedder, which carries no semantic signal), the number says so; ship BM25-only and
   record the negative (plan R2) rather than fake a win.

`run_benchmark` takes an injected embedder (DIP) so the automated test drives it deterministically;
`__main__` runs it with the real multilingual MiniLM for genuine relevance evidence.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import psycopg

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.application.search_service import SearchService
from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.domain.ports import TurnEmbedder
from talkex.monitoring.domain.search import SearchQuery
from talkex.monitoring.infrastructure.pool import MonitoringPool
from talkex.monitoring.infrastructure.read_repo import TimescaleReadRepository
from talkex.monitoring.infrastructure.timescale_repo import TimescaleTurnRepository

# A small labeled corpus. Each entry: (turn_id, text). Conversations grouped by prefix.
CORPUS: list[tuple[str, str]] = [
    ("t01", "quero cancelar o plano imediatamente"),
    ("t02", "desejo encerrar minha assinatura hoje"),  # paraphrase of cancel intent (low lexical overlap)
    ("t03", "como faço para pagar a fatura em atraso"),
    ("t04", "o boleto venceu e preciso de uma segunda via"),
    ("t05", "o atendente foi muito educado e resolveu tudo"),
    ("t06", "estou muito satisfeito com o suporte de voces"),
    ("t07", "quero falar com um supervisor agora"),
    ("t08", "me transfere para o gerente responsavel"),  # paraphrase of escalation
    ("t09", "minha internet esta muito lenta ha dias"),
    ("t10", "a conexao cai toda hora e nao aguento mais"),
    ("t11", "gostaria de contratar um plano novo"),
    ("t12", "quais promocoes voces tem disponiveis"),
    ("t13", "preciso alterar meu endereco de cobranca"),
    ("t14", "meu cartao foi recusado no pagamento"),
    ("t15", "obrigado pelo excelente atendimento de hoje"),
]

# Labeled probe: query -> relevant turn_ids. Includes paraphrase queries where a semantic match
# (t02, t08) shares little vocabulary with the query, so the ANN half should help the hybrid.
PROBE: list[tuple[str, set[str]]] = [
    ("quero cancelar o plano", {"t01", "t02"}),
    ("encerrar assinatura", {"t02", "t01"}),
    ("pagar fatura atrasada", {"t03", "t04"}),
    ("falar com supervisor", {"t07", "t08"}),
    ("transferir para o gerente", {"t08", "t07"}),
    ("internet lenta caindo", {"t09", "t10"}),
]


@dataclass
class BenchResult:
    p95_ms: float
    mean_ms: float
    samples: int
    hybrid_mrr: float
    bm25_mrr: float
    corpus_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "p95_ms": round(self.p95_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "samples": self.samples,
            "hybrid_mrr": round(self.hybrid_mrr, 4),
            "bm25_mrr": round(self.bm25_mrr, 4),
            "hybrid_beats_bm25": self.hybrid_mrr >= self.bm25_mrr,
            "corpus_size": self.corpus_size,
            "p95_under_200ms": self.p95_ms < 200.0,
        }


def _turn(turn_id: str, text: str, conv: str = "bench") -> Turn:
    return Turn(
        turn_id=TurnId(turn_id),
        conversation_id=ConversationId(conv),
        speaker=SpeakerRole.CUSTOMER,
        raw_text=text,
        start_offset=0,
        end_offset=len(text),
    )


def _reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, tid in enumerate(ranked_ids, start=1):
        if tid in relevant:
            return 1.0 / i
    return 0.0


async def run_benchmark(
    conn: psycopg.AsyncConnection, pool: MonitoringPool, embedder: TurnEmbedder, *, latency_samples: int = 100
) -> BenchResult:
    """Seed the corpus, measure p95 under concurrent ingest, and MRR of hybrid vs BM25-only."""
    seed_repo = TimescaleTurnRepository(conn, embedder=embedder)
    for turn_id, text in CORPUS:
        await seed_repo.save(_turn(turn_id, text))

    read_repo = TimescaleReadRepository(pool)
    service = SearchService(read_repo, embedder)

    # --- Relevance: hybrid (fused) vs BM25-only (lexical candidates alone) ---
    hybrid_rr: list[float] = []
    bm25_rr: list[float] = []
    for query, relevant in PROBE:
        hybrid_hits = await service.search(SearchQuery(query_text=query, top_k=10))
        hybrid_rr.append(_reciprocal_rank([h.turn_id for h in hybrid_hits], relevant))
        bm25_hits = await read_repo.lexical_candidates(query, top_k=10, window_days=30)
        bm25_rr.append(_reciprocal_rank([h.object_id for h in bm25_hits], relevant))

    # --- Latency: p95 of hybrid search while ingesting concurrently ---
    stop = asyncio.Event()

    async def _ingest_load() -> None:
        i = 0
        while not stop.is_set():
            async with pool.connection() as c:
                await TimescaleTurnRepository(c, embedder=embedder).save(
                    _turn(f"load_{i}", f"trafego de fundo numero {i}", conv="load")
                )
            i += 1
            await asyncio.sleep(0.005)

    load_task = asyncio.ensure_future(_ingest_load())
    latencies: list[float] = []
    try:
        for n in range(latency_samples):
            query = PROBE[n % len(PROBE)][0]
            start = time.perf_counter()
            await service.search(SearchQuery(query_text=query, top_k=10))
            latencies.append((time.perf_counter() - start) * 1000.0)
    finally:
        stop.set()
        await load_task

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    return BenchResult(
        p95_ms=p95,
        mean_ms=statistics.mean(latencies),
        samples=len(latencies),
        hybrid_mrr=statistics.mean(hybrid_rr),
        bm25_mrr=statistics.mean(bm25_rr),
        corpus_size=len(CORPUS),
    )


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="M5 hybrid-search benchmark (T4.1)")
    parser.add_argument("--deterministic", action="store_true", help="use the download-free embedder")
    parser.add_argument("--out", default="experiments/results/m5_hybrid_bench.json")
    args = parser.parse_args()

    if args.deterministic:
        from talkex.monitoring.infrastructure.embedder import DeterministicEmbedder

        embedder: TurnEmbedder = DeterministicEmbedder()
    else:
        from talkex.monitoring.infrastructure.embedder import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()

    cfg = MonitoringConfig()
    conn = await psycopg.AsyncConnection.connect(cfg.dsn)
    await conn.execute("TRUNCATE turns, alerts, labels")
    await conn.commit()
    pool = MonitoringPool(cfg.dsn, min_size=2, max_size=8)
    await pool.open()
    try:
        result = await run_benchmark(conn, pool, embedder)
    finally:
        await pool.close()
        await conn.close()

    import pathlib

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
