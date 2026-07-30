"""Index-aligned keyset-paginated read repository (blueprint D1/Corner 4).

Reads recent turns for a conversation ordered by `created_at DESC`, aligned to the
`(conversation_id, created_at DESC)` composite index (chatwoot query-aligned precedent
conversation_finder.rb:108). Keyset pagination (no OFFSET, no N+1) via a `before` cursor.
"""

from __future__ import annotations

from datetime import datetime

from talkex.monitoring.domain.models import RecentTurn
from talkex.monitoring.infrastructure.pool import MonitoringPool


class TimescaleReadRepository:
    """Implements TurnReadPort over a pooled TimescaleDB connection."""

    def __init__(self, pool: MonitoringPool) -> None:
        self._pool = pool

    async def recent_turns(
        self, conversation_id: str, before: datetime | None = None, limit: int = 50
    ) -> list[RecentTurn]:
        # Index-aligned WHERE + ORDER matching (conversation_id, created_at DESC); keyset via `before`.
        sql = (
            "SELECT turn_id, raw_text, created_at FROM turns "
            "WHERE conversation_id = %s "
            + ("AND created_at < %s " if before is not None else "")
            + "ORDER BY created_at DESC LIMIT %s"
        )
        params: tuple[object, ...] = (
            (conversation_id, before, limit) if before is not None else (conversation_id, limit)
        )
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [RecentTurn(turn_id=r[0], raw_text=r[1], created_at=r[2]) for r in rows]
