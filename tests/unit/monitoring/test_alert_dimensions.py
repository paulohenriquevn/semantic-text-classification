"""Unit tests for M6 alert KPI dimensions — queue + sentiment populated at emit (T0.1)."""

from talkex.models.enums import Channel, SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.application.orchestrator import TurnOrchestrator
from talkex.monitoring.domain.critical_rules import build_critical_rules


class _CapturingAlertRepo:
    def __init__(self) -> None:
        self.saved: list = []

    async def save(self, alert) -> None:
        self.saved.append(alert)


class _NullBroadcaster:
    async def notify(self, alert_id) -> None:
        return None


class _NullTurnRepo:
    async def save(self, turn) -> None:
        return None


class _FakeSentiment:
    is_fitted = True

    def predict(self, text: str):
        from types import SimpleNamespace

        return SimpleNamespace(label="negative", score=1.2)


def _turn(text: str, queue: str) -> Turn:
    return Turn(
        turn_id=TurnId("t_dim"),
        conversation_id=ConversationId("conv_dim"),
        speaker=SpeakerRole.CUSTOMER,
        raw_text=text,
        start_offset=0,
        end_offset=len(text),
        metadata={"queue": queue},
    )


class TestAlertDimensions:
    async def test_emit_populates_queue_and_sentiment(self) -> None:
        repo = _CapturingAlertRepo()
        orch = TurnOrchestrator(
            turn_repo=_NullTurnRepo(),
            alert_repo=repo,
            broadcaster=_NullBroadcaster(),
            rules=build_critical_rules(),
            channel=Channel.VOICE,
            sentiment_detector=_FakeSentiment(),
        )
        await orch.handle(_turn("quero cancelar o plano agora", queue="retention"))

        assert repo.saved, "no alert emitted for a cancellation utterance"
        alert = repo.saved[0]
        assert alert.queue == "retention"  # queue promoted from turn.metadata
        assert alert.sentiment == "negative"  # sentiment label promoted from cascade evidence

    async def test_queue_defaults_when_absent(self) -> None:
        repo = _CapturingAlertRepo()
        orch = TurnOrchestrator(
            turn_repo=_NullTurnRepo(),
            alert_repo=repo,
            broadcaster=_NullBroadcaster(),
            rules=build_critical_rules(),
            channel=Channel.VOICE,
        )
        turn = Turn(
            turn_id=TurnId("t_dim2"),
            conversation_id=ConversationId("conv_dim2"),
            speaker=SpeakerRole.CUSTOMER,
            raw_text="quero cancelar o plano agora",
            start_offset=0,
            end_offset=27,
        )
        await orch.handle(turn)

        assert repo.saved
        assert repo.saved[0].queue == "default"  # honest default when ingest did not capture a queue
        assert repo.saved[0].sentiment is None  # no detector wired → NULL dimension
