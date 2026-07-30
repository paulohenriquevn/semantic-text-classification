"""M3 tests: critical-rule catalogue, multi-rule alerting, precision, latency."""

import json
import re
import time
from pathlib import Path

import pytest

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.application.orchestrator import TurnOrchestrator
from talkex.monitoring.domain.critical_rules import build_critical_rules
from talkex.monitoring.domain.models import Alert, AlertId
from talkex.rules.config import RuleEngineConfig
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.rules.models import RuleEvaluationInput

_DATA = Path("experiments/data/test.jsonl")


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


class _Repo:
    def __init__(self) -> None:
        self.saved: list = []

    async def save(self, obj: object) -> None:
        self.saved.append(obj)

    async def get(self, alert_id: AlertId) -> Alert | None:
        return None


class _Bc:
    async def notify(self, alert_id: AlertId) -> None:
        return None


class TestCatalogue:
    def test_catalogue_has_critical_rules(self) -> None:
        names = {r.rule_name for r in build_critical_rules()}
        assert {"cancellation", "escalation"} <= names


class TestMultiRuleAlerting:
    async def test_two_rules_two_alerts(self) -> None:
        tr, ar, br = _Repo(), _Repo(), _Bc()
        orch = TurnOrchestrator(turn_repo=tr, alert_repo=ar, broadcaster=br, rules=build_critical_rules())
        # A single customer turn expressing BOTH cancellation intent AND escalation.
        await orch.handle(_turn(0, "quero cancelar minha conta e falar com o gerente agora"))
        names = sorted(a.rule_name for a in ar.saved)
        assert names == ["cancellation", "escalation"]


class TestPrecision:
    """The M3 DoD: cancellation-alert precision ≥ 0.80 against the topic labels (customer-scoped)."""

    def _customer(self, t: str) -> str:
        parts = re.split(r"\[(customer|agent)\]", t)
        out, i = [], 1
        while i < len(parts):
            if parts[i] == "customer":
                out.append(parts[i + 1] if i + 1 < len(parts) else "")
            i += 2
        return " ".join(out)

    def test_alert_precision_meets_bar(self) -> None:
        if not _DATA.exists():
            pytest.skip("labeled corpus not present")
        rule = next(r for r in build_critical_rules() if r.rule_name == "cancellation")
        ev = SimpleRuleEvaluator()
        fire = tp = 0
        for line in _DATA.read_text().splitlines():
            r = json.loads(line)
            res = ev.evaluate(
                [rule],
                RuleEvaluationInput(source_id="w", source_type="context_window", text=self._customer(r["text"])),
                RuleEngineConfig(),
            )[0]
            if res.matched:
                fire += 1
                if "cancel" in r.get("topic", "").lower():
                    tp += 1
        precision = tp / fire if fire else 0.0
        assert precision >= 0.80, f"cancellation-alert precision {precision:.3f} below the 0.80 DoD"


class TestLatency:
    async def test_turn_to_alert_p95_under_2s(self) -> None:
        tr, ar, br = _Repo(), _Repo(), _Bc()
        orch = TurnOrchestrator(turn_repo=tr, alert_repo=ar, broadcaster=br, rules=build_critical_rules())
        elapsed: list[float] = []
        for i in range(30):
            t0 = time.perf_counter()
            await orch.handle(_turn(i, "quero cancelar minha conta"))
            elapsed.append(time.perf_counter() - t0)
        elapsed.sort()
        p95 = elapsed[int(0.95 * len(elapsed)) - 1]
        assert p95 < 2.0, f"turn→alert p95 {p95:.3f}s exceeds the 2s budget"
