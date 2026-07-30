"""Backfill the `turns.embedding` column (M5 Phase 0, T0.1).

The `embedding vector(384)` column shipped empty in M1. This one-shot, idempotent backfill fills
rows where `embedding IS NULL` in batches — so the pgvector ANN half of the M5 hybrid search has
real data. Re-running is safe: only NULL rows are selected, so a second pass updates zero rows.

Usage (production, real embeddings):
    python experiments/scripts/backfill_embeddings.py

The `backfill` coroutine takes an injected `TurnEmbedder` (DIP) so tests drive it with a
deterministic embedder against a real TimescaleDB.
"""

from __future__ import annotations

import argparse
import asyncio

import psycopg

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.domain.ports import TurnEmbedder
from talkex.monitoring.infrastructure.embedder import (
    DeterministicEmbedder,
    SentenceTransformerEmbedder,
    to_vector_literal,
)


async def backfill(conn: psycopg.AsyncConnection, embedder: TurnEmbedder, batch_size: int = 256) -> int:
    """Fill NULL embeddings in batches. Returns the number of rows updated (idempotent).

    Fail-fast: any DB or encoder error propagates (no silent partial backfill).
    """
    total = 0
    while True:
        cur = await conn.execute(
            "SELECT turn_id, raw_text FROM turns WHERE embedding IS NULL LIMIT %s",
            (batch_size,),
        )
        rows = await cur.fetchall()
        if not rows:
            break
        for turn_id, raw_text in rows:
            literal = to_vector_literal(embedder.embed(raw_text))
            await conn.execute(
                "UPDATE turns SET embedding = %s::vector WHERE turn_id = %s",
                (literal, turn_id),
            )
        await conn.commit()
        total += len(rows)
    return total


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Backfill turns.embedding (M5 Phase 0)")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use the download-free deterministic embedder (smoke test) instead of the real model",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    embedder: TurnEmbedder = DeterministicEmbedder() if args.deterministic else SentenceTransformerEmbedder()
    conn = await psycopg.AsyncConnection.connect(MonitoringConfig().dsn)
    try:
        updated = await backfill(conn, embedder, batch_size=args.batch_size)
        print(f"backfilled {updated} embeddings")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(_main())
