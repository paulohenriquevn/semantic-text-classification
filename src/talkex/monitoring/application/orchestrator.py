"""Turn orchestrator — the online 'decide' step (blueprint D1/D4).

For each ingested Turn: persist it (per-Turn insert, D5), rebuild the conversation's
context windows (reusing SlidingWindowBuilder), evaluate ONE DSL rule on the latest
window (reusing the rules engine — the decide step is the DSL, NOT an LLM, rejecting
the ai-call-center online-LLM anti-pattern), and on a match emit an evidence-backed
Alert, persist it, then notify AFTER persistence (commit-then-notify, D3).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from talkex.classification.sentiment import SentimentDetector
from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig
from talkex.models.conversation import Conversation
from talkex.models.enums import Channel
from talkex.models.rule_execution import EvidenceItem
from talkex.models.turn import Turn
from talkex.monitoring.domain.models import Alert, AlertId
from talkex.monitoring.domain.ports import AlertBroadcaster, AlertRepository, TurnRepository
from talkex.rules.config import RuleEngineConfig
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.rules.models import RuleDefinition, RuleEvaluationInput

# A window forms as soon as the first turn lands (M0 needs immediacy, not a full 5-turn window).
_M0_WINDOW_CONFIG = ContextWindowConfig(
    window_size=3,
    stride=1,
    min_window_size=1,
    include_partial_tail=True,
)
_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)


class TurnOrchestrator:
    """Reuses segmentation output + windowing + the DSL engine to raise alerts."""

    def __init__(
        self,
        *,
        turn_repo: TurnRepository,
        alert_repo: AlertRepository,
        broadcaster: AlertBroadcaster,
        rule: RuleDefinition,
        channel: Channel = Channel.VOICE,
        sentiment_detector: SentimentDetector | None = None,
    ) -> None:
        self._turn_repo = turn_repo
        self._alert_repo = alert_repo
        self._broadcaster = broadcaster
        self._rule = rule
        self._channel = channel
        self._sentiment = sentiment_detector
        self._builder = SlidingWindowBuilder()
        self._evaluator = SimpleRuleEvaluator()
        self._buffers: dict[str, list[Turn]] = {}

    def _sentiment_evidence(self, window_text: str) -> list[EvidenceItem]:
        """Compute sentiment on the window as cascade evidence (M2), when a detector is present."""
        if self._sentiment is None or not self._sentiment.is_fitted:
            return []
        pred = self._sentiment.predict(window_text)
        return [EvidenceItem(predicate_type="sentiment", matched_text=pred.label, score=pred.score)]

    async def handle(self, turn: Turn) -> None:
        """Persist the turn, window the conversation, evaluate the rule, alert on match."""
        await self._turn_repo.save(turn)

        buffer = self._buffers.setdefault(turn.conversation_id, [])
        buffer.append(turn)

        conversation = Conversation(
            conversation_id=turn.conversation_id,
            channel=self._channel,
            start_time=_EPOCH,
        )
        windows = self._builder.build(conversation, buffer, _M0_WINDOW_CONFIG)
        if not windows:
            return

        window = windows[-1]
        result = self._evaluator.evaluate(
            [self._rule],
            RuleEvaluationInput(
                source_id=window.window_id,
                source_type="context_window",
                text=window.window_text,
            ),
            RuleEngineConfig(),
        )[0]

        if not result.matched:
            return

        evidence = [pr.to_evidence_item() for pr in result.predicate_results]
        evidence.extend(self._sentiment_evidence(window.window_text))  # M2 cascade feature
        alert = Alert(
            alert_id=AlertId(f"alert_{uuid.uuid4().hex[:12]}"),
            conversation_id=turn.conversation_id,
            window_id=window.window_id,
            rule_name=result.rule_name,
            evidence=evidence,
        )
        await self._alert_repo.save(alert)  # commit first ...
        await self._broadcaster.notify(alert.alert_id)  # ... then notify (D3)
