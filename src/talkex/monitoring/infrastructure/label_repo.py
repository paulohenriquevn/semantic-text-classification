"""Timescale label repository — persists QA labels for retraining (M5 Phase 2, blueprint D2).

Implements `LabelRepository` over a pooled connection. The `labels` table is a plain table (not a
hypertable), so labels survive the 30-day raw-data purge — retraining is long-term memory.
"""

from __future__ import annotations

from talkex.monitoring.domain.search import Label
from talkex.monitoring.infrastructure.pool import MonitoringPool


class TimescaleLabelRepository:
    """Persists and retrieves QA labels."""

    def __init__(self, pool: MonitoringPool) -> None:
        self._pool = pool

    async def save(self, label: Label) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO labels (label_id, turn_id, conversation_id, label, labeled_by) "
                "VALUES (%s, %s, %s, %s, %s)",
                (label.label_id, label.turn_id, label.conversation_id, label.label, label.labeled_by),
            )

    async def get(self, label_id: str) -> Label | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT label_id, turn_id, conversation_id, label, labeled_by FROM labels WHERE label_id = %s",
                (label_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return Label(label_id=row[0], turn_id=row[1], conversation_id=row[2], label=row[3], labeled_by=row[4])
