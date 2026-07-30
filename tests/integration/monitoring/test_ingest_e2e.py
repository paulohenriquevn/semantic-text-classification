"""End-to-end DoD test for the M0 walking skeleton (Final Integration Validation).

Proves the four M0 DoDs against a real TimescaleDB and the wired FastAPI app:
  (1) a posted transcript segment lands as a Turn in the hypertable with a timestamp;
  (2) segmentation + context window produce a window from the live stream;
  (3) one DSL rule fires on a window and produces an alert with evidence;
  (4) the alert reaches a supervisor LISTEN subscriber (the SSE stream's data source).
"""

import asyncio

import httpx
import psycopg
import pytest
from asgi_lifespan import LifespanManager
from psycopg import sql

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.embedder import DeterministicEmbedder
from talkex.monitoring.infrastructure.timescale_repo import TimescaleAlertRepository
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


async def test_ingest_to_supervisor_alert_e2e() -> None:
    if not await _reachable():
        pytest.skip("TimescaleDB not reachable on :5433")

    # Clean slate.
    async with await psycopg.AsyncConnection.connect(CFG.dsn) as c:
        await c.execute("TRUNCATE turns, alerts")
        await c.commit()

    # A supervisor LISTEN subscriber — exactly the data source the SSE endpoint uses (DoD-4).
    listen_conn = await psycopg.AsyncConnection.connect(CFG.dsn, autocommit=True)
    await listen_conn.execute(sql.SQL("LISTEN {}").format(sql.Identifier(CFG.notify_channel)))

    app = create_app(CFG, embedder=DeterministicEmbedder())  # download-free embedder for the e2e
    try:
        async with (
            LifespanManager(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
        ):
            notif = asyncio.ensure_future(anext(listen_conn.notifies()))
            await asyncio.sleep(0.2)  # ensure LISTEN is registered before the NOTIFY

            resp = await client.post(
                "/ingest",
                json={"conversation_id": "conv_e2e", "raw_text": "[customer] quero cancelar minha conta"},
            )
            assert resp.status_code == 202
            assert resp.json()["enqueued"] >= 1  # DoD-2: segmentation produced ≥1 turn

            # DoD-4: the alert id reaches the supervisor subscriber within seconds.
            alert_notify = await asyncio.wait_for(notif, timeout=5.0)
            alert_id = alert_notify.payload

            # DoD-3: the alert is persisted with evidence.
            async with await psycopg.AsyncConnection.connect(CFG.dsn) as read:
                alert = await TimescaleAlertRepository(read).get(alert_id)  # type: ignore[arg-type]
            assert alert is not None
            assert alert.rule_name == "cancellation"
            assert any(e.get("matched_text") for e in alert.evidence)

            # DoD-1: the turn landed in the hypertable with a timestamp.
            async with await psycopg.AsyncConnection.connect(CFG.dsn) as read2:
                cur = await read2.execute(
                    "SELECT count(*), min(created_at) FROM turns WHERE conversation_id = %s",
                    ("conv_e2e",),
                )
                count, first_ts = await cur.fetchone()  # type: ignore[misc]
            assert count >= 1
            assert first_ts is not None

            # The supervisor page is served.
            page = await client.get("/supervisor")
            assert page.status_code == 200
            assert "EventSource" in page.text
    finally:
        await listen_conn.close()
