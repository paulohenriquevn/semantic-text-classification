"""Integration tests for M5 Phase 0 — embedding population (T0.1).

The `embedding vector(384)` column shipped empty in M1 (nothing wrote it). These tests prove the
ingest path now writes a non-null 384-dim embedding, and that the backfill fills pre-existing NULL
rows idempotently — so the M5 ANN half is genuinely live (not BM25-in-disguise).
"""

import psycopg
import pytest

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.infrastructure.embedder import DeterministicEmbedder
from talkex.monitoring.infrastructure.timescale_repo import TimescaleTurnRepository

pytestmark = pytest.mark.integration

EMBED_DIM = 384


def _make_turn(turn_id: str, text: str) -> Turn:
    return Turn(
        turn_id=TurnId(turn_id),
        conversation_id=ConversationId("conv_emb"),
        speaker=SpeakerRole.CUSTOMER,
        raw_text=text,
        start_offset=0,
        end_offset=len(text),
    )


class TestIngestWritesEmbedding:
    async def test_ingest_writes_embedding(self, conn: psycopg.AsyncConnection) -> None:
        repo = TimescaleTurnRepository(conn, embedder=DeterministicEmbedder())
        await repo.save(_make_turn("turn_emb_1", "quero cancelar o plano"))

        cur = await conn.execute(
            "SELECT embedding, vector_dims(embedding) FROM turns WHERE turn_id = %s", ("turn_emb_1",)
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[0] is not None  # embedding populated at ingest
        assert row[1] == EMBED_DIM  # 384-dim, matching the vector(384) column


class TestBackfill:
    async def test_backfill_fills_null_embeddings(self, conn: psycopg.AsyncConnection) -> None:
        # Seed two turns WITHOUT an embedder (embedding stays NULL — the shipped-M1 state).
        plain = TimescaleTurnRepository(conn)
        await plain.save(_make_turn("turn_bf_1", "primeiro turno sem embedding"))
        await plain.save(_make_turn("turn_bf_2", "segundo turno sem embedding"))

        from experiments.scripts.backfill_embeddings import backfill

        updated = await backfill(conn, DeterministicEmbedder(), batch_size=8)
        assert updated == 2

        cur = await conn.execute("SELECT count(*) FROM turns WHERE embedding IS NULL")
        row = await cur.fetchone()
        assert row is not None and row[0] == 0

    async def test_backfill_is_idempotent(self, conn: psycopg.AsyncConnection) -> None:
        plain = TimescaleTurnRepository(conn)
        await plain.save(_make_turn("turn_bf_3", "turno para idempotencia"))

        from experiments.scripts.backfill_embeddings import backfill

        first = await backfill(conn, DeterministicEmbedder(), batch_size=8)
        second = await backfill(conn, DeterministicEmbedder(), batch_size=8)
        assert first == 1
        assert second == 0  # re-run skips already-populated rows
