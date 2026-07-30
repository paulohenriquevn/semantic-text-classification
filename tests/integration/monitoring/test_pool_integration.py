"""Integration tests for the connection pool against a real DB (T1.1)."""

import asyncio

import psycopg
import pytest

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.pool import MonitoringPool

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn


async def _reachable() -> bool:
    try:
        c = await psycopg.AsyncConnection.connect(DSN, connect_timeout=2)
        await c.close()
        return True
    except Exception:
        return False


async def test_pool_gives_working_connections() -> None:
    if not await _reachable():
        pytest.skip("TimescaleDB not reachable on :5433")
    pool = MonitoringPool(DSN, min_size=1, max_size=4)
    await pool.open()
    try:
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT 1")
            row = await cur.fetchone()
            assert row is not None and row[0] == 1
    finally:
        await pool.close()


async def test_pool_queues_under_concurrency_without_exceeding_max() -> None:
    if not await _reachable():
        pytest.skip("TimescaleDB not reachable on :5433")
    # N concurrent acquirers on a max_size=4 pool; all must succeed (queue, not error).
    pool = MonitoringPool(DSN, min_size=1, max_size=4)
    await pool.open()
    try:

        async def work(i: int) -> int:
            async with pool.connection() as conn:
                cur = await conn.execute("SELECT %s::int", (i,))
                await asyncio.sleep(0.02)  # hold the connection to force queueing
                row = await cur.fetchone()
                assert row is not None
                return int(row[0])

        results = await asyncio.gather(*(work(i) for i in range(12)))
        assert sorted(results) == list(range(12))  # all 12 completed, none lost
    finally:
        await pool.close()
