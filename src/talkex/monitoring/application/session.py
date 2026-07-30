"""Live monitoring session — explicit state machine with two-phase drain (blueprint D2).

Mirrors livekit agent_session.py: initializing → listening → closing, with drain-then-close
so an in-flight window/rule evaluation finishes before teardown. Owns a bounded TurnChannel
(backpressure, D1) and a single consumer task that feeds the orchestrator.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from talkex.models.turn import Turn
from talkex.monitoring.domain.channel import TurnChannel
from talkex.monitoring.domain.models import SessionState


class _TurnHandler(Protocol):
    async def handle(self, turn: Turn) -> None: ...


class MonitoringSession:
    """Drives a bounded channel through a consumer into a turn handler (orchestrator)."""

    def __init__(self, channel: TurnChannel, handler: _TurnHandler) -> None:
        self._channel = channel
        self._handler = handler
        self._state = SessionState.INITIALIZING
        self._state_history: list[SessionState] = [SessionState.INITIALIZING]
        self._consumer: asyncio.Task[None] | None = None

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def state_history(self) -> list[SessionState]:
        return list(self._state_history)

    def _transition(self, state: SessionState) -> None:
        if state != self._state:
            self._state = state
            self._state_history.append(state)

    async def start(self) -> None:
        """Begin consuming: transition to LISTENING and spawn the consumer task."""
        self._transition(SessionState.LISTENING)
        self._consumer = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        async for turn in self._channel:
            await self._handler.handle(turn)

    async def aclose(self) -> None:
        """Two-phase drain-then-close: close the channel, let the consumer finish, then CLOSING."""
        self._channel.close()  # iteration drains remaining items, then stops
        if self._consumer is not None:
            await self._consumer  # drain: in-flight + buffered turns are fully processed
        self._transition(SessionState.CLOSING)
