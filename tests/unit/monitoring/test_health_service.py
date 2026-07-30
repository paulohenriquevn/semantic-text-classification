"""Unit tests for the M8 readiness probe (T0.1)."""

from talkex.monitoring.application.health_service import HealthService
from talkex.monitoring.domain.channel import TurnChannel
from talkex.monitoring.domain.models import SessionState


class _Session:
    def __init__(self, state: SessionState) -> None:
        self._state = state

    @property
    def state(self) -> SessionState:
        return self._state


class _DbUp:
    async def ping(self) -> bool:
        return True


class _DbDown:
    async def ping(self) -> bool:
        return False


class TestReadinessProbe:
    async def test_ready_when_listening_db_up_queue_ok(self) -> None:
        svc = HealthService(_Session(SessionState.LISTENING), TurnChannel(maxsize=10), _DbUp())
        report = await svc.readiness()
        assert report.ready is True
        assert report.session_state == "listening"
        assert report.db_reachable is True

    async def test_not_ready_when_db_down(self) -> None:
        svc = HealthService(_Session(SessionState.LISTENING), TurnChannel(maxsize=10), _DbDown())
        report = await svc.readiness()
        assert report.ready is False
        assert report.db_reachable is False

    async def test_not_ready_when_not_listening(self) -> None:
        svc = HealthService(_Session(SessionState.INITIALIZING), TurnChannel(maxsize=10), _DbUp())
        report = await svc.readiness()
        assert report.ready is False

    async def test_ready_false_when_queue_saturated(self) -> None:
        channel = TurnChannel(maxsize=10)
        for i in range(9):  # 9/10 = 90% ≥ the 0.9 saturation threshold
            await channel.put(_fake_turn(i))
        svc = HealthService(_Session(SessionState.LISTENING), channel, _DbUp())
        report = await svc.readiness()
        assert report.ready is False  # backpressure surfaces as not-ready
        assert report.queue_depth == 9


def _fake_turn(i: int):
    from talkex.models.enums import SpeakerRole
    from talkex.models.turn import Turn
    from talkex.models.types import ConversationId, TurnId

    return Turn(
        turn_id=TurnId(f"t{i}"),
        conversation_id=ConversationId("c"),
        speaker=SpeakerRole.CUSTOMER,
        raw_text="x",
        start_offset=0,
        end_offset=1,
    )
