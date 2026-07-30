"""Pool-backed DB readiness probe (M8 D1).

A thin `_DbProbe` implementation: `ping()` runs `SELECT 1` over the pool and reports reachability. A
failure returns False (not-ready) rather than propagating — the readiness endpoint must answer 503, not
crash (fail-soft at the probe boundary).
"""

from __future__ import annotations

from talkex.monitoring.infrastructure.pool import MonitoringPool


class PoolDbProbe:
    """Reports DB reachability via a pooled `SELECT 1`."""

    def __init__(self, pool: MonitoringPool) -> None:
        self._pool = pool

    async def ping(self) -> bool:
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute("SELECT 1")
                return (await cur.fetchone()) is not None
        except Exception:
            return False  # unreachable → not ready (the probe reports, it does not raise)
