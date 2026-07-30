"""Unit tests for TurnOrchestrator over fakes (T2.2)."""

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.application.orchestrator import TurnOrchestrator
from talkex.monitoring.domain.models import Alert, AlertId
from talkex.rules.compiler import SimpleRuleCompiler


def _turn(i: int, text: str) -> Turn:
    return Turn(
        turn_id=TurnId(f"turn_{i}"),
        conversation_id=ConversationId("conv_1"),
        speaker=SpeakerRole.CUSTOMER,
        raw_text=text,
        normalized_text=text,
        start_offset=0,
        end_offset=len(text),
    )


class _FakeTurnRepo:
    def __init__(self, log: list[str]) -> None:
        self.saved: list[Turn] = []
        self._log = log

    async def save(self, turn: Turn) -> None:
        self.saved.append(turn)
        self._log.append("turn")


class _FakeAlertRepo:
    def __init__(self, log: list[str]) -> None:
        self.saved: list[Alert] = []
        self._log = log

    async def save(self, alert: Alert) -> None:
        self.saved.append(alert)
        self._log.append("alert")

    async def get(self, alert_id: AlertId) -> Alert | None:
        return next((a for a in self.saved if a.alert_id == alert_id), None)


class _FakeBroadcaster:
    def __init__(self, log: list[str]) -> None:
        self.notified: list[AlertId] = []
        self._log = log

    async def notify(self, alert_id: AlertId) -> None:
        self.notified.append(alert_id)
        self._log.append("notify")


def _make(log: list[str], dsl: str = 'contains_any("cancelar", "cancelamento")') -> tuple:
    rule = SimpleRuleCompiler().compile(dsl_text=dsl, rule_id="rule_m0", rule_name="cancellation_risk")
    tr, ar, br = _FakeTurnRepo(log), _FakeAlertRepo(log), _FakeBroadcaster(log)
    orch = TurnOrchestrator(turn_repo=tr, alert_repo=ar, broadcaster=br, rule=rule)
    return orch, tr, ar, br


class TestAlerting:
    async def test_matching_turn_raises_alert_with_evidence(self) -> None:
        log: list[str] = []
        orch, _tr, ar, br = _make(log)
        await orch.handle(_turn(0, "quero cancelar minha conta agora"))
        assert len(ar.saved) == 1
        assert ar.saved[0].rule_name == "cancellation_risk"
        assert ar.saved[0].evidence  # non-empty evidence from the DSL match
        assert br.notified == [ar.saved[0].alert_id]

    async def test_non_matching_turn_raises_no_alert(self) -> None:
        log: list[str] = []
        orch, tr, ar, br = _make(log)
        await orch.handle(_turn(0, "obrigado pelo excelente atendimento"))
        assert len(tr.saved) == 1
        assert ar.saved == []
        assert br.notified == []


class TestOrdering:
    async def test_turn_persisted_before_alert_and_notify(self) -> None:
        log: list[str] = []
        orch, _tr, _ar, _br = _make(log)
        await orch.handle(_turn(0, "cancelamento por favor"))
        # commit-then-notify (D3): turn saved, then alert saved, then notify
        assert log == ["turn", "alert", "notify"]
