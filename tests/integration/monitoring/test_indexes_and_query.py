"""Integration tests for keyset pagination + index usage (T2.1)."""

import psycopg
import pytest

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.pool import MonitoringPool
from talkex.monitoring.infrastructure.read_repo import TimescaleReadRepository

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn


async def _seed(conn: psycopg.AsyncConnection, n: int) -> None:
    for i in range(n):
        await conn.execute(
            "INSERT INTO turns (turn_id, conversation_id, speaker, raw_text, start_offset, end_offset, created_at) "
            "VALUES (%s, 'conv_q', 'customer', %s, 0, 3, now() - (%s || ' seconds')::interval)",
            (f"turn_q_{i}", f"msg {i}", n - i),  # older -> newer
        )
    await conn.commit()


class TestKeysetPagination:
    async def test_recent_turns_keyset_paginates_without_overlap(self, conn: psycopg.AsyncConnection) -> None:
        await _seed(conn, 5)
        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            repo = TimescaleReadRepository(pool)
            page1 = await repo.recent_turns("conv_q", before=None, limit=2)
            assert len(page1) == 2
            page2 = await repo.recent_turns("conv_q", before=page1[-1].created_at, limit=2)
            assert len(page2) == 2
            page3 = await repo.recent_turns("conv_q", before=page2[-1].created_at, limit=2)
            assert len(page3) == 1  # 5 total
            ids = [t.turn_id for t in page1 + page2 + page3]
            assert len(ids) == len(set(ids)) == 5  # no overlap, no loss
        finally:
            await pool.close()


class TestIndexUsage:
    async def test_recent_turns_query_uses_composite_index(self, conn: psycopg.AsyncConnection) -> None:
        await _seed(conn, 5)
        # Force the planner off seq-scan to prove the query is index-alignable (small table
        # would otherwise seq-scan). The index must be usable for this WHERE+ORDER.
        await conn.execute("SET LOCAL enable_seqscan = off")
        cur = await conn.execute(
            "EXPLAIN SELECT turn_id FROM turns WHERE conversation_id = 'conv_q' ORDER BY created_at DESC LIMIT 5"
        )
        plan = "\n".join(r[0] for r in await cur.fetchall())
        assert "Index Scan" in plan or "Index Only Scan" in plan
        assert "conv" in plan.lower()  # the (conversation_id, created_at) index
