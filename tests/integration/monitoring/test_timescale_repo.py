"""Integration tests for the Timescale repositories against a real hypertable (T3.1)."""

import psycopg
import pytest

from talkex.models.enums import SpeakerRole
from talkex.models.rule_execution import EvidenceItem
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.domain.models import Alert, AlertId
from talkex.monitoring.infrastructure.timescale_repo import (
    TimescaleAlertRepository,
    TimescaleTurnRepository,
)

pytestmark = pytest.mark.integration


class TestTurnRepository:
    async def test_save_turn_lands_in_hypertable_with_created_at(self, conn: psycopg.AsyncConnection) -> None:
        repo = TimescaleTurnRepository(conn)
        turn = Turn(
            turn_id=TurnId("turn_it_1"),
            conversation_id=ConversationId("conv_it"),
            speaker=SpeakerRole.CUSTOMER,
            raw_text="quero cancelar",
            start_offset=0,
            end_offset=14,
        )
        await repo.save(turn)

        cur = await conn.execute("SELECT turn_id, raw_text, created_at FROM turns WHERE turn_id = %s", ("turn_it_1",))
        row = await cur.fetchone()
        assert row is not None
        assert row[1] == "quero cancelar"
        assert row[2] is not None  # created_at populated by the hypertable default


class TestAlertRepository:
    async def test_alert_evidence_roundtrips_jsonb(self, conn: psycopg.AsyncConnection) -> None:
        repo = TimescaleAlertRepository(conn)
        alert = Alert(
            alert_id=AlertId("alert_it_1"),
            conversation_id=ConversationId("conv_it"),
            window_id="win_it_1",
            rule_name="cancellation_risk",
            evidence=[EvidenceItem(predicate_type="contains_any", matched_text="cancelar", score=1.0)],
        )
        await repo.save(alert)

        got = await repo.get(AlertId("alert_it_1"))
        assert got is not None
        assert got.rule_name == "cancellation_risk"
        assert got.evidence[0]["matched_text"] == "cancelar"

    async def test_get_missing_alert_returns_none(self, conn: psycopg.AsyncConnection) -> None:
        repo = TimescaleAlertRepository(conn)
        assert await repo.get(AlertId("nope")) is None


class TestFailureScenarios:
    """Plan `## Failure scenarios`: the DB layer must fail-fast, not swallow (Rule 8)."""

    async def test_save_on_broken_connection_raises_not_silent(self) -> None:
        # Simulate a connection reset: connect, close, then save -> must raise, not no-op.
        from talkex.monitoring.config import MonitoringConfig

        broken = await psycopg.AsyncConnection.connect(MonitoringConfig().dsn)
        await broken.close()
        repo = TimescaleTurnRepository(broken)
        turn = Turn(
            turn_id=TurnId("turn_fail_1"),
            conversation_id=ConversationId("conv_fail"),
            speaker=SpeakerRole.CUSTOMER,
            raw_text="x",
            start_offset=0,
            end_offset=1,
        )
        with pytest.raises(psycopg.OperationalError):
            await repo.save(turn)
