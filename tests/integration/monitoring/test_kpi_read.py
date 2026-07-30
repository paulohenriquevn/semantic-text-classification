"""Integration test for M6 KPI read over the continuous aggregate (T1.1)."""

from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg.types.json import Jsonb

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.domain.dashboard import KpiQuery
from talkex.monitoring.infrastructure.kpi_repo import TimescaleKpiRepository
from talkex.monitoring.infrastructure.pool import MonitoringPool

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn


async def _seed(conn: psycopg.AsyncConnection) -> None:
    rows = [
        ("k_1", "cancellation", "retention"),
        ("k_2", "cancellation", "retention"),
        ("k_3", "escalation", "billing"),
        ("k_4", "cancellation", "billing"),
    ]
    for alert_id, rule, queue in rows:
        await conn.execute(
            "INSERT INTO alerts (alert_id, conversation_id, window_id, rule_name, evidence, queue, sentiment) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (alert_id, "conv_r", "win_r", rule, Jsonb([]), queue, "negative"),
        )
    await conn.commit()
    ac = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    try:
        await ac.execute("CALL refresh_continuous_aggregate('alerts_kpi_5min', NULL, NULL)")
    finally:
        await ac.close()
    await conn.commit()


class TestKpiRead:
    async def test_kpi_rollups_grouped_by_queue(self, conn: psycopg.AsyncConnection) -> None:
        await _seed(conn)
        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        window = KpiQuery(
            from_time=datetime.now(UTC) - timedelta(hours=1),
            to_time=datetime.now(UTC) + timedelta(hours=1),
        )
        try:
            repo = TimescaleKpiRepository(pool)
            allb = await repo.kpi_rollups(window)
            retention_only = await repo.kpi_rollups(window.model_copy(update={"queue": "retention"}))
        finally:
            await pool.close()

        # Aggregate the (possibly multi-bucket) rows per (rule, queue).
        by_dim: dict[tuple[str, str], int] = {}
        for b in allb:
            by_dim[(b.rule_name, b.queue)] = by_dim.get((b.rule_name, b.queue), 0) + b.alert_count
        assert by_dim[("cancellation", "retention")] == 2
        assert by_dim[("escalation", "billing")] == 1
        assert by_dim[("cancellation", "billing")] == 1

        assert all(b.queue == "retention" for b in retention_only)  # queue filter narrows
        assert sum(b.alert_count for b in retention_only) == 2
