"""Turn orchestrator — the online 'decide' step (blueprint M0 D1/D4, M3 catalogue).

For each ingested Turn: persist it (per-Turn insert, D5), rebuild the conversation's
context windows (reusing SlidingWindowBuilder), evaluate the CRITICAL-RULE CATALOGUE on the
customer-scoped window text (M3 — the decide step is the DSL, NOT an LLM), attach optional M2
sentiment evidence, and emit one evidence-backed Alert PER matched rule — persist then notify
AFTER persistence (commit-then-notify, D3).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from talkex.classification.sentiment import SentimentDetector
from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig
from talkex.models.context_window import ContextWindow
from talkex.models.conversation import Conversation
from talkex.models.enums import Channel, SpeakerRole
from talkex.models.rule_execution import EvidenceItem
from talkex.models.turn import Turn
from talkex.monitoring.domain.models import Alert, AlertId
from talkex.monitoring.domain.ports import AlertBroadcaster, AlertRepository, TurnRepository
from talkex.rules.config import RuleEngineConfig
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.rules.models import RuleDefinition, RuleEvaluationInput, RuleResult

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
        rules: list[RuleDefinition],
        channel: Channel = Channel.VOICE,
        sentiment_detector: SentimentDetector | None = None,
    ) -> None:
        if not rules:
            raise ValueError("at least one rule is required")
        self._turn_repo = turn_repo
        self._alert_repo = alert_repo
        self._broadcaster = broadcaster
        self._rules = rules
        self._channel = channel
        self._sentiment = sentiment_detector
        self._builder = SlidingWindowBuilder()
        self._evaluator = SimpleRuleEvaluator()
        self._buffers: dict[str, list[Turn]] = {}

    @staticmethod
    def _customer_text(window: ContextWindow, buffer: list[Turn]) -> str:
        """Join the customer utterances within the window (critical intent is the customer's, M3 D2)."""
        ids = set(window.turn_ids)
        return " ".join(t.raw_text for t in buffer if t.turn_id in ids and t.speaker == SpeakerRole.CUSTOMER)

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
        # Critical intent is the CUSTOMER's — scope rule evaluation to customer utterances so the
        # agent explaining a process ("para cancelamento você liga...") does not raise a false alert
        # (this scoping lifts cancellation-alert precision from 0.51 to 0.89 — blueprint M3 D2).
        customer_text = self._customer_text(window, buffer) or window.window_text
        results = self._evaluator.evaluate(
            self._rules,
            RuleEvaluationInput(
                source_id=window.window_id,
                source_type="context_window",
                text=customer_text,
            ),
            RuleEngineConfig(),
        )
        sentiment = self._sentiment_evidence(window.window_text)  # M2 cascade feature, shared
        for result in results:  # M3: one evidence-backed alert per matched critical rule
            if result.matched:
                await self._emit_alert(turn, window, result, sentiment)

    async def _emit_alert(
        self, turn: Turn, window: ContextWindow, result: RuleResult, sentiment: list[EvidenceItem]
    ) -> None:
        """Build the evidence-backed alert, persist it, then notify (commit-then-notify, D3)."""
        evidence = [pr.to_evidence_item() for pr in result.predicate_results]
        evidence.extend(sentiment)
        # M6: promote queue (from the turn's metadata) + the sentiment label into first-class KPI
        # dimensions so the alerts_kpi_5min continuous aggregate can group by them.
        queue = str(turn.metadata.get("queue", "default"))
        sentiment_label = sentiment[0].get("matched_text") if sentiment else None
        alert = Alert(
            alert_id=AlertId(f"alert_{uuid.uuid4().hex[:12]}"),
            conversation_id=turn.conversation_id,
            window_id=window.window_id,
            rule_name=result.rule_name,
            evidence=evidence,
            queue=queue,
            sentiment=sentiment_label,
        )
        await self._alert_repo.save(alert)  # commit first ...
        await self._broadcaster.notify(alert.alert_id)  # ... then notify (D3)
