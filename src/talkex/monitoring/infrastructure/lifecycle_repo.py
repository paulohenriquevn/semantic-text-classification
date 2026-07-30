"""Timescale lifecycle repository — reads the export window + purges old chunks (M7 D4).

Implements `LifecycleReadPort`. The read left-joins turns with their QA labels (M5); the purge drops
raw `turns`/`alerts` chunks older than a cutoff — the caller (the exporter) MUST have verified a
successful export first (export-before-purge, blueprint D4).
"""

from __future__ import annotations

from datetime import datetime

from talkex.monitoring.infrastructure.pool import MonitoringPool


class TimescaleLifecycleRepository:
    """Reads the to-be-exported window and purges raw chunks."""

    def __init__(self, pool: MonitoringPool) -> None:
        self._pool = pool

    async def read_export_rows(self, from_time: datetime, to_time: datetime) -> list[tuple[str, str, str, str | None]]:
        sql = (
            "SELECT t.turn_id, t.conversation_id, t.raw_text, l.label "
            "FROM turns t LEFT JOIN labels l ON l.turn_id = t.turn_id "
            "WHERE t.created_at >= %s AND t.created_at < %s "
            "ORDER BY t.created_at"
        )
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, (from_time, to_time))
            rows = await cur.fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    async def purge_before(self, older_than: datetime) -> None:
        async with self._pool.connection() as conn:
            await conn.execute("SELECT drop_chunks('turns', older_than => %s)", (older_than,))
            await conn.execute("SELECT drop_chunks('alerts', older_than => %s)", (older_than,))
