"""Health/readiness application service (M8 D1).

Probes the monitor's own state — session LISTENING, the bounded channel not saturated, and the DB
reachable — and reports readiness. Liveness (the process is up) is trivially true at the route; readiness
is what this service computes. The probe reads shipped domain state (DIP); it owns no infrastructure.
"""

from __future__ import annotations

from typing import Protocol

from talkex.monitoring.domain.channel import TurnChannel
from talkex.monitoring.domain.health import HealthReport
from talkex.monitoring.domain.models import SessionState

# Queue at/above this fraction of capacity = backpressure → not ready (blueprint D1 / risk R4).
_SATURATION = 0.9


class _StateSource(Protocol):
    @property
    def state(self) -> SessionState: ...


class _DbProbe(Protocol):
    async def ping(self) -> bool: ...


class HealthService:
    """Computes a readiness `HealthReport` from the session, channel, and a DB probe."""

    def __init__(self, session: _StateSource, channel: TurnChannel, db: _DbProbe) -> None:
        self._session = session
        self._channel = channel
        self._db = db

    async def readiness(self) -> HealthReport:
        depth = self._channel.qsize()
        maxsize = self._channel.maxsize
        saturated = depth >= int(maxsize * _SATURATION)
        listening = self._session.state == SessionState.LISTENING
        db_ok = await self._db.ping()
        return HealthReport(
            ready=listening and db_ok and not saturated,
            session_state=str(self._session.state),
            queue_depth=depth,
            queue_maxsize=maxsize,
            db_reachable=db_ok,
        )
