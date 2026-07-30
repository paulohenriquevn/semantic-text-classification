"""DIP ports for the monitoring domain (blueprint D4).

The domain declares these Protocols; infrastructure implements them (Timescale repo,
LISTEN/NOTIFY broadcaster) and the interface layer injects the concretes. The domain
never imports psycopg or FastAPI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from talkex.models.turn import Turn
from talkex.monitoring.domain.dashboard import KpiBucket, KpiQuery
from talkex.monitoring.domain.models import Alert, AlertId, RecentTurn
from talkex.monitoring.domain.search import Criterion, Label
from talkex.retrieval.models import RetrievalHit


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


class TurnSearchPort(Protocol):
    """DB-side candidate retrieval for M5 hybrid search (blueprint D1).

    Two narrow methods — one per signal — so the application `SearchService` owns the fusion
    (RRF) and stays unit-testable with a fake port. Both return `RetrievalHit` (from
    `talkex.retrieval`), scoped to the retention window and optionally criterion-filtered.
    """

    async def lexical_candidates(
        self, query_text: str, top_k: int, window_days: int, criteria: tuple[Criterion, ...] = ()
    ) -> list[RetrievalHit]: ...

    async def semantic_candidates(
        self, query_vector: list[float], top_k: int, window_days: int, criteria: tuple[Criterion, ...] = ()
    ) -> list[RetrievalHit]: ...


class LabelRepository(Protocol):
    """Persists QA labels destined for retraining (blueprint D2, survives the 30-day purge)."""

    async def save(self, label: Label) -> None: ...

    async def get(self, label_id: str) -> Label | None: ...


class KpiReadPort(Protocol):
    """Reads pre-bucketed KPI rollups from the `alerts_kpi_5min` continuous aggregate (M6, D5)."""

    async def kpi_rollups(self, query: KpiQuery) -> list[KpiBucket]: ...
