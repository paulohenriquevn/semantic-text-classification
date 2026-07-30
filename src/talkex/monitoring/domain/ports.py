"""DIP ports for the monitoring domain (blueprint D4).

The domain declares these Protocols; infrastructure implements them (Timescale repo,
LISTEN/NOTIFY broadcaster) and the interface layer injects the concretes. The domain
never imports psycopg or FastAPI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from talkex.models.turn import Turn
from talkex.monitoring.domain.models import Alert, AlertId, RecentTurn


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


class TurnReadPort(Protocol):
    """Index-aligned keyset-paginated reads for the supervisor/QA view (blueprint Corner 4)."""

    async def recent_turns(self, conversation_id: str, before: datetime | None, limit: int) -> list[RecentTurn]: ...


class TurnEmbedder(Protocol):
    """Produces a dense embedding for a turn's text (M5 Phase 0 — feeds the pgvector ANN half).

    Narrow by design (ISP): the ingest/search paths need only text -> vector, not the full
    batch generator API in `talkex.embeddings`. Infrastructure supplies a deterministic
    (no-download) adapter for tests and a sentence-transformer adapter for production.
    """

    def embed(self, text: str) -> list[float]: ...
