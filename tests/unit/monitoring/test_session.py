"""Unit tests for MonitoringSession state machine + drain (T2.1)."""

import asyncio

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.application.session import MonitoringSession
from talkex.monitoring.domain.channel import TurnChannel
from talkex.monitoring.domain.models import SessionState


def _turn(i: int) -> Turn:
    return Turn(
        turn_id=TurnId(f"turn_{i}"),
        conversation_id=ConversationId("conv_1"),
        speaker=SpeakerRole.CUSTOMER,
        raw_text=f"msg {i}",
        start_offset=0,
        end_offset=5,
    )


class _RecordingHandler:
    def __init__(self) -> None:
        self.handled: list[str] = []

    async def handle(self, turn: Turn) -> None:
        self.handled.append(turn.turn_id)


class TestStateMachine:
    async def test_state_sequence_init_listening_closing(self) -> None:
        session = MonitoringSession(TurnChannel(maxsize=4), _RecordingHandler())
        assert session.state_history == [SessionState.INITIALIZING]
        await session.start()
        assert session.state == SessionState.LISTENING
        await session.aclose()
        assert session.state_history == [
            SessionState.INITIALIZING,
            SessionState.LISTENING,
            SessionState.CLOSING,
        ]


class TestDrain:
    async def test_inflight_turns_processed_before_close(self) -> None:
        channel = TurnChannel(maxsize=8)
        handler = _RecordingHandler()
        session = MonitoringSession(channel, handler)
        for i in range(3):
            await channel.put(_turn(i))
        await session.start()
        await session.aclose()  # must drain the 3 buffered turns before CLOSING
        assert handler.handled == ["turn_0", "turn_1", "turn_2"]
        assert session._consumer is not None and session._consumer.done()

    async def test_close_is_idempotent_under_race(self) -> None:
        channel = TurnChannel(maxsize=4)
        session = MonitoringSession(channel, _RecordingHandler())
        await session.start()
        await asyncio.gather(session.aclose(), session.aclose())
        assert session.state == SessionState.CLOSING
