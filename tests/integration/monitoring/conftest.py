"""Fixtures for monitoring integration tests — a real TimescaleDB connection.

Requires the dev container: `docker compose -f deploy/monitoring/docker-compose.yml up -d`.
If the DB is unreachable, the whole integration module is skipped (unit tier is unaffected).
"""

from collections.abc import AsyncIterator

import psycopg
import pytest

from talkex.monitoring.config import MonitoringConfig

DSN = MonitoringConfig().dsn


async def _can_connect() -> bool:
    try:
        conn = await psycopg.AsyncConnection.connect(DSN, connect_timeout=2)
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def conn() -> AsyncIterator[psycopg.AsyncConnection]:
    if not await _can_connect():
        pytest.skip("TimescaleDB not reachable on :5433 — start deploy/monitoring/docker-compose.yml")
    connection = await psycopg.AsyncConnection.connect(DSN)
    await connection.execute("TRUNCATE turns, alerts, labels")
    await connection.commit()
    try:
        yield connection
    finally:
        await connection.close()
