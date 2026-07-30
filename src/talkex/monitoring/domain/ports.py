"""DIP ports for the monitoring domain (blueprint D4).

The domain declares these Protocols; infrastructure implements them (Timescale repo,
LISTEN/NOTIFY broadcaster) and the interface layer injects the concretes. The domain
never imports psycopg or FastAPI.
"""

from __future__ import annotations

from typing import Protocol

from talkex.models.turn import Turn
from talkex.monitoring.domain.models import Alert, AlertId


class TurnRepository(Protocol):
    """Persists a Turn as it arrives (per-Turn streaming insert, blueprint D5)."""

    async def save(self, turn: Turn) -> None: ...


class AlertRepository(Protocol):
    """Persists and retrieves alerts."""

    async def save(self, alert: Alert) -> None: ...

    async def get(self, alert_id: AlertId) -> Alert | None: ...


class AlertBroadcaster(Protocol):
    """Pushes an alert id to subscribers after the transaction commits (blueprint D3)."""

    async def notify(self, alert_id: AlertId) -> None: ...
