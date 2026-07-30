"""Bounded async channel — the ingest backpressure primitive (blueprint D1).

Modeled on livekit-agents' bounded `Chan`
(knowledge-base/references/livekit-agents/.../utils/aio/channel.py:49,:71): `put` awaits
while the buffer is full, so a fast producer (transcript source) cannot unbounded-buffer
ahead of the slower window/rule consumer. Closing terminates async iteration deterministically.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from talkex.models.turn import Turn


class ChannelClosed(Exception):
    """Raised when putting onto a closed channel."""


class TurnChannel:
    """A bounded FIFO channel of Turns with awaiting backpressure on `put`."""

    # Poll interval used only to notice close while a getter waits on an empty queue.
    _CLOSE_POLL_SECONDS = 0.02

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0 (a bounded channel is the backpressure lever)")
        self._queue: asyncio.Queue[Turn] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def maxsize(self) -> int:
        """The bounded capacity — the readiness probe compares qsize against this (M8)."""
        return self._queue.maxsize

    async def put(self, turn: Turn) -> None:
        """Enqueue a turn, awaiting when the channel is full (backpressure)."""
        if self._closed:
            raise ChannelClosed("cannot put on a closed channel")
        await self._queue.put(turn)

    async def get(self) -> Turn:
        """Dequeue the next turn, awaiting when empty."""
        return await self._queue.get()

    def close(self) -> None:
        """Mark the channel closed. Pending items are still drainable via iteration."""
        self._closed = True

    async def __aiter__(self) -> AsyncIterator[Turn]:
        """Yield turns until the channel is closed AND drained."""
        while True:
            if self._closed and self._queue.empty():
                return
            try:
                turn = await asyncio.wait_for(self._queue.get(), timeout=self._CLOSE_POLL_SECONDS)
            except TimeoutError:
                continue
            yield turn
