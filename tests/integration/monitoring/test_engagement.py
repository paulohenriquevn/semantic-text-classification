"""Integration test for the M8 alert-engagement proxy (T1.1)."""

from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg.types.json import Jsonb

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.engagement_repo import TimescaleEngagementRepository
from talkex.monitoring.infrastructure.pool import MonitoringPool

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn


class TestEngagement:
    async def test_acted_on_rate(self, conn: psycopg.AsyncConnection) -> None:
        # 4 alerts, 2 of them acted on (labelled) → acted_on_rate == 0.5.
        for i in range(4):
            await conn.execute(
                "INSERT INTO alerts (alert_id, conversation_id, window_id, rule_name, evidence, queue, sentiment) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (f"eng_a{i}", "conv_e", "win_e", "cancellation", Jsonb([]), "default", "negative"),
            )
        for i in range(2):
            await conn.execute(
                "INSERT INTO labels (label_id, turn_id, conversation_id, label) VALUES (%s, %s, %s, %s)",
                (f"eng_l{i}", f"t{i}", "conv_e", "reviewed"),
            )
        await conn.commit()

        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            repo = TimescaleEngagementRepository(pool)
            metric = await repo.engagement(
                datetime.now(UTC) - timedelta(hours=1), datetime.now(UTC) + timedelta(hours=1)
            )
        finally:
            await pool.close()

        assert metric.alerts == 4
        assert metric.labels == 2
        assert metric.acted_on_rate == 0.5

    async def test_zero_alerts_yields_zero_rate(self, conn: psycopg.AsyncConnection) -> None:
        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            metric = await TimescaleEngagementRepository(pool).engagement(
                datetime.now(UTC) - timedelta(hours=1), datetime.now(UTC) + timedelta(hours=1)
            )
        finally:
            await pool.close()
        assert metric.acted_on_rate == 0.0  # no alerts → no division by zero
