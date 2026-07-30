"""Unit tests for the bounded TurnChannel — the backpressure primitive (T1.1)."""

import asyncio

import pytest

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.domain.channel import ChannelClosed, TurnChannel


def _turn(i: int) -> Turn:
    return Turn(
        turn_id=TurnId(f"turn_{i}"),
        conversation_id=ConversationId("conv_1"),
        speaker=SpeakerRole.CUSTOMER,
        raw_text=f"msg {i}",
        start_offset=0,
        end_offset=5,
    )


class TestBackpressure:
    def test_rejects_non_positive_maxsize(self) -> None:
        with pytest.raises(ValueError):
            TurnChannel(maxsize=0)

    @pytest.mark.asyncio
    async def test_put_blocks_when_full(self) -> None:
        ch = TurnChannel(maxsize=1)
        await ch.put(_turn(1))  # fills the single slot
        # A second put must NOT complete until a get frees a slot — this IS backpressure.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ch.put(_turn(2)), timeout=0.1)

    @pytest.mark.asyncio
    async def test_put_unblocks_after_get(self) -> None:
        ch = TurnChannel(maxsize=1)
        await ch.put(_turn(1))
        put_task = asyncio.create_task(ch.put(_turn(2)))
        got = await ch.get()  # frees the slot
        await asyncio.wait_for(put_task, timeout=0.5)
        assert got.turn_id == "turn_1"
        assert ch.qsize() == 1


class TestClose:
    @pytest.mark.asyncio
    async def test_put_after_close_raises(self) -> None:
        ch = TurnChannel(maxsize=2)
        ch.close()
        with pytest.raises(ChannelClosed):
            await ch.put(_turn(1))

    @pytest.mark.asyncio
    async def test_iteration_drains_then_stops_after_close(self) -> None:
        ch = TurnChannel(maxsize=4)
        for i in range(3):
            await ch.put(_turn(i))
        ch.close()
        seen = [t.turn_id async for t in ch]
        assert seen == ["turn_0", "turn_1", "turn_2"]


class TestConcurrencyInvariant:
    @pytest.mark.asyncio
    async def test_no_loss_under_many_producers(self) -> None:
        # N producers put M turns each into a small bounded channel; one consumer drains.
        # Invariant: every produced turn is consumed exactly once (no Lost Update).
        ch = TurnChannel(maxsize=4)
        n_producers, m_each = 5, 20
        total = n_producers * m_each

        async def produce(pid: int) -> None:
            for j in range(m_each):
                await ch.put(_turn(pid * 1000 + j))

        consumed: list[str] = []

        async def consume() -> None:
            async for t in ch:
                consumed.append(t.turn_id)

        consumer = asyncio.create_task(consume())
        await asyncio.gather(*(produce(p) for p in range(n_producers)))
        ch.close()
        await asyncio.wait_for(consumer, timeout=2.0)

        assert len(consumed) == total
        assert len(set(consumed)) == total  # no duplicates, no loss
