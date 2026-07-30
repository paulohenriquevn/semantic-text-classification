"""TimescaleDB repositories — per-Turn / per-Alert hypertable inserts (blueprint D5).

Implement the domain ports (TurnRepository / AlertRepository) via async psycopg against
the `turns` / `alerts` hypertables. Evidence is stored as JSONB. Each save commits so a
committed row is what a subsequent LISTEN/NOTIFY push (D3) reads back.
"""

from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from talkex.models.turn import Turn
from talkex.models.types import ConversationId
from talkex.monitoring.domain.models import Alert, AlertId


class TimescaleTurnRepository:
    """Persists a Turn as it arrives."""

    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self._conn = conn

    async def save(self, turn: Turn) -> None:
        await self._conn.execute(
            "INSERT INTO turns "
            "(turn_id, conversation_id, speaker, raw_text, normalized_text, start_offset, end_offset, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                turn.turn_id,
                turn.conversation_id,
                str(turn.speaker),
                turn.raw_text,
                turn.normalized_text,
                turn.start_offset,
                turn.end_offset,
                Jsonb(turn.metadata),
            ),
        )
        await self._conn.commit()


class TimescaleAlertRepository:
    """Persists and retrieves alerts."""

    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self._conn = conn

    async def save(self, alert: Alert) -> None:
        await self._conn.execute(
            "INSERT INTO alerts (alert_id, conversation_id, window_id, rule_name, evidence) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                alert.alert_id,
                alert.conversation_id,
                alert.window_id,
                alert.rule_name,
                Jsonb(alert.evidence),
            ),
        )
        await self._conn.commit()

    async def get(self, alert_id: AlertId) -> Alert | None:
        cur = await self._conn.execute(
            "SELECT alert_id, conversation_id, window_id, rule_name, evidence FROM alerts WHERE alert_id = %s",
            (alert_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return Alert(
            alert_id=AlertId(row[0]),
            conversation_id=ConversationId(row[1]),
            window_id=row[2],
            rule_name=row[3],
            evidence=row[4],
        )
