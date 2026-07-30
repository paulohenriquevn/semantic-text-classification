"""Postgres LISTEN/NOTIFY alert broadcaster (blueprint D3).

Implements the AlertBroadcaster port. `notify` sends the alert id (payload is capped at
8000 bytes, so only the id travels — the SSE handler reads full evidence from the repo by
id, drawback R2). `listen` yields ids for the supervisor SSE stream. The channel name comes
from trusted config and is quoted as an identifier for LISTEN (which cannot be parameterized);
NOTIFY uses the parameterizable `pg_notify` function.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import psycopg
from psycopg import sql

from talkex.monitoring.domain.models import AlertId


class NotifyAlertBroadcaster:
    """Sends and receives alert ids over a Postgres LISTEN/NOTIFY channel."""

    def __init__(self, conn: psycopg.AsyncConnection, channel: str) -> None:
        self._conn = conn
        self._channel = channel

    async def notify(self, alert_id: AlertId) -> None:
        await self._conn.execute("SELECT pg_notify(%s, %s)", (self._channel, str(alert_id)))
        await self._conn.commit()

    async def listen(self) -> AsyncIterator[str]:
        """Start LISTENing and yield each notified alert id payload."""
        await self._conn.execute(sql.SQL("LISTEN {}").format(sql.Identifier(self._channel)))
        await self._conn.commit()
        async for notify in self._conn.notifies():
            yield notify.payload
