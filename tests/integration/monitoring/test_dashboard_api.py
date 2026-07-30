"""Integration test for M6 Phase 2 — GET /dashboard/kpis endpoint (T2.1)."""

from datetime import UTC, datetime, timedelta

import httpx
import psycopg
import pytest
from asgi_lifespan import LifespanManager
from psycopg.types.json import Jsonb

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.embedder import DeterministicEmbedder
from talkex.monitoring.interface.app import create_app

pytestmark = pytest.mark.integration

CFG = MonitoringConfig()


async def _reachable() -> bool:
    try:
        c = await psycopg.AsyncConnection.connect(CFG.dsn, connect_timeout=2)
        await c.close()
        return True
    except Exception:
        return False


async def _seed_and_refresh() -> None:
    c = await psycopg.AsyncConnection.connect(CFG.dsn)
    await c.execute("TRUNCATE turns, alerts, labels")
    for alert_id, rule, queue in [
        ("d_1", "cancellation", "retention"),
        ("d_2", "cancellation", "retention"),
        ("d_3", "escalation", "billing"),
    ]:
        await c.execute(
            "INSERT INTO alerts (alert_id, conversation_id, window_id, rule_name, evidence, queue, sentiment) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (alert_id, "conv_d", "win_d", rule, Jsonb([]), queue, "negative"),
        )
    await c.commit()
    await c.close()
    ac = await psycopg.AsyncConnection.connect(CFG.dsn, autocommit=True)
    await ac.execute("CALL refresh_continuous_aggregate('alerts_kpi_5min', NULL, NULL)")
    await ac.close()


class TestDashboardApi:
    async def test_dashboard_endpoint_returns_kpis(self) -> None:
        if not await _reachable():
            pytest.skip("TimescaleDB not reachable on :5433")
        await _seed_and_refresh()
        app = create_app(CFG, embedder=DeterministicEmbedder())
        params = {
            "from_time": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            "to_time": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        }
        async with (
            LifespanManager(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
        ):
            resp = await client.get("/dashboard/kpis", params=params)
            assert resp.status_code == 200
            buckets = resp.json()
            assert buckets, "dashboard returned no KPI buckets"

            # A queue filter narrows to the retention queue.
            resp2 = await client.get("/dashboard/kpis", params={**params, "queue": "retention"})
            assert resp2.status_code == 200
            retention = resp2.json()
            assert all(b["queue"] == "retention" for b in retention)
            assert sum(b["alert_count"] for b in retention) == 2  # two cancellation alerts in retention
