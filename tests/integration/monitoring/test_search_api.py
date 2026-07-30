"""Integration tests for M5 Phase 3 — QA search + label HTTP endpoints (T3.1)."""

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


async def _clean() -> None:
    c = await psycopg.AsyncConnection.connect(CFG.dsn)
    await c.execute("TRUNCATE turns, alerts, labels")
    await c.commit()
    await c.close()


class TestSearchApi:
    async def test_search_endpoint_returns_windows_with_evidence(self) -> None:
        if not await _reachable():
            pytest.skip("TimescaleDB not reachable on :5433")
        await _clean()
        app = create_app(CFG, embedder=DeterministicEmbedder())
        async with (
            LifespanManager(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
        ):
            await client.post(
                "/ingest", json={"conversation_id": "conv_api", "raw_text": "[customer] quero cancelar o plano agora"}
            )
            # let the async consumer persist the turn(s)
            import asyncio

            await asyncio.sleep(0.5)

            resp = await client.post("/search", json={"query_text": "quero cancelar o plano agora", "top_k": 5})
            assert resp.status_code == 200
            hits = resp.json()
            assert hits, "search returned no windows"
            top = hits[0]
            assert "cancelar" in top["raw_text"]  # evidence window carried
            assert top["conversation_id"] == "conv_api"
            assert top["score"] > 0

    async def test_label_endpoint_persists(self) -> None:
        if not await _reachable():
            pytest.skip("TimescaleDB not reachable on :5433")
        await _clean()
        app = create_app(CFG, embedder=DeterministicEmbedder())
        async with (
            LifespanManager(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
        ):
            resp = await client.post(
                "/label",
                json={
                    "turn_id": "turn_api_1",
                    "conversation_id": "conv_api",
                    "label": "script_deviation",
                    "labeled_by": "qa_bob",
                },
            )
            assert resp.status_code == 201
            label_id = resp.json()["label_id"]
            assert label_id.startswith("lbl_")

        # Confirm the row persisted.
        c = await psycopg.AsyncConnection.connect(CFG.dsn)
        cur = await c.execute("SELECT label, labeled_by FROM labels WHERE label_id = %s", (label_id,))
        row = await cur.fetchone()
        await c.close()
        assert row is not None
        assert row[0] == "script_deviation"
        assert row[1] == "qa_bob"
