"""Integration test for M8 /health + /ready endpoints (T0.1)."""

import httpx
import psycopg
import pytest
from asgi_lifespan import LifespanManager

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


class TestHealthApi:
    async def test_health_and_ready_endpoints(self) -> None:
        if not await _reachable():
            pytest.skip("TimescaleDB not reachable on :5433")
        app = create_app(CFG, embedder=DeterministicEmbedder())
        async with (
            LifespanManager(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
        ):
            live = await client.get("/health")
            assert live.status_code == 200
            assert live.json()["status"] == "alive"

            ready = await client.get("/ready")
            assert ready.status_code == 200  # session LISTENING + DB reachable + queue empty
            body = ready.json()
            assert body["ready"] is True
            assert body["session_state"] == "listening"
            assert body["db_reachable"] is True
