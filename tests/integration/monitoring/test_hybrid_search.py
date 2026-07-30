"""Integration tests for M5 hybrid search against a real TimescaleDB (T1.1).

Seeds turns (lexical `search_vector` generated + `embedding` populated via the deterministic
embedder, so vectors are reproducible), then drives the `SearchService` over the real
`TimescaleReadRepository` and asserts fused ranking, 30-day window scoping, and behaviour under
concurrent ingest.
"""

import asyncio

import psycopg
import pytest
from psycopg.types.json import Jsonb

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.application.search_service import SearchService
from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.domain.search import SearchQuery
from talkex.monitoring.infrastructure.embedder import DeterministicEmbedder, to_vector_literal
from talkex.monitoring.infrastructure.pool import MonitoringPool
from talkex.monitoring.infrastructure.read_repo import TimescaleReadRepository
from talkex.monitoring.infrastructure.timescale_repo import TimescaleTurnRepository

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn
EMBEDDER = DeterministicEmbedder()


def _turn(turn_id: str, text: str) -> Turn:
    return Turn(
        turn_id=TurnId(turn_id),
        conversation_id=ConversationId("conv_hs"),
        speaker=SpeakerRole.CUSTOMER,
        raw_text=text,
        start_offset=0,
        end_offset=len(text),
    )


async def _seed(conn: psycopg.AsyncConnection, turns: list[Turn]) -> None:
    repo = TimescaleTurnRepository(conn, embedder=EMBEDDER)
    for t in turns:
        await repo.save(t)


class TestHybridSearch:
    async def test_hybrid_search_returns_fused_windows(self, conn: psycopg.AsyncConnection) -> None:
        await _seed(
            conn,
            [
                _turn("hs_1", "quero cancelar o plano agora"),
                _turn("hs_2", "gostaria de falar sobre a fatura"),
                _turn("hs_3", "obrigado pelo atendimento"),
            ],
        )
        pool = MonitoringPool(DSN, min_size=1, max_size=4)
        await pool.open()
        try:
            service = SearchService(TimescaleReadRepository(pool), EMBEDDER)
            hits = await service.search(SearchQuery(query_text="quero cancelar o plano agora", top_k=5))
        finally:
            await pool.close()

        assert hits, "hybrid search returned no windows"
        assert hits[0].turn_id == "hs_1"  # lexical + semantic match → top of the fused list
        assert hits[0].raw_text == "quero cancelar o plano agora"  # evidence carried
        assert hits[0].conversation_id == "conv_hs"

    async def test_window_scoping_excludes_old_turns(self, conn: psycopg.AsyncConnection) -> None:
        # Timescale partitions by created_at (the time dimension) and forbids UPDATEs across chunks,
        # so the old row must be INSERTed with an explicit 40-days-ago timestamp (routes to its chunk).
        literal = to_vector_literal(EMBEDDER.embed("quero cancelar o plano agora"))
        await conn.execute(
            "INSERT INTO turns (turn_id, conversation_id, speaker, raw_text, normalized_text, "
            "start_offset, end_offset, metadata, embedding, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, now() - interval '40 days')",
            (
                "hs_old",
                "conv_hs",
                str(SpeakerRole.CUSTOMER),
                "quero cancelar o plano agora",
                None,
                0,
                28,
                Jsonb({}),
                literal,
            ),
        )
        await conn.commit()

        pool = MonitoringPool(DSN, min_size=1, max_size=4)
        await pool.open()
        try:
            service = SearchService(TimescaleReadRepository(pool), EMBEDDER)
            hits = await service.search(SearchQuery(query_text="quero cancelar o plano agora", top_k=5, window_days=30))
        finally:
            await pool.close()

        assert all(h.turn_id != "hs_old" for h in hits)  # older than 30 days → excluded

    async def test_hybrid_search_under_concurrent_ingest(self, conn: psycopg.AsyncConnection) -> None:
        await _seed(conn, [_turn("hs_seed", "quero cancelar o plano agora")])
        pool = MonitoringPool(DSN, min_size=2, max_size=6)
        await pool.open()
        try:
            service = SearchService(TimescaleReadRepository(pool), EMBEDDER)

            async def _ingest(i: int) -> None:
                async with pool.connection() as c:
                    await TimescaleTurnRepository(c, embedder=EMBEDDER).save(
                        _turn(f"hs_conc_{i}", f"mensagem concorrente numero {i}")
                    )

            async def _query() -> int:
                res = await service.search(SearchQuery(query_text="quero cancelar o plano agora", top_k=5))
                return len(res)

            results = await asyncio.gather(*[_ingest(i) for i in range(8)], *[_query() for _ in range(8)])
        finally:
            await pool.close()

        query_lengths = [r for r in results if isinstance(r, int)]
        assert all(n >= 1 for n in query_lengths)  # every concurrent query saw the seeded match, no errors
