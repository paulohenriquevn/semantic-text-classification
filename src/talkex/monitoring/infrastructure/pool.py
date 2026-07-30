"""Async connection pool sized to concurrency (blueprint D4 / ADR-005 R3).

Mirrors chatwoot's pool==concurrency design (config/database.yml:2): a bounded pool absorbs the
concurrent ingest writes + supervisor reads without connection churn under load, ahead of a
read-replica split (deferred). Thin wrapper over psycopg_pool.AsyncConnectionPool.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from psycopg_pool import AsyncConnectionPool


class MonitoringPool:
    """A bounded async connection pool for the monitoring storage layer."""

    def __init__(self, dsn: str, *, min_size: int = 2, max_size: int = 10) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self._max_size = max_size
        # open=False: do not connect at construction; call open() explicitly (lifespan).
        self._pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False)

    @property
    def max_size(self) -> int:
        return self._max_size

    async def open(self) -> None:
        await self._pool.open()

    async def close(self) -> None:
        await self._pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection]:
        """Acquire a pooled connection; queues (does not error) when the pool is saturated."""
        async with self._pool.connection() as conn:
            yield conn
