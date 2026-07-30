"""Integration test for the LISTEN/NOTIFY broadcaster (T3.2)."""

import asyncio

import psycopg
import pytest

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.domain.models import AlertId
from talkex.monitoring.infrastructure.notify_broadcaster import NotifyAlertBroadcaster

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn
CHANNEL = "talkex_alerts_test"


async def test_notify_delivers_alert_id_to_listener() -> None:
    listen_conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    notify_conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    try:
        listener = NotifyAlertBroadcaster(listen_conn, CHANNEL)
        # Prime the listener: the first __anext__ runs LISTEN, then blocks on the first notify.
        agen = listener.listen()
        first = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.3)  # let LISTEN register before we NOTIFY (happens-before)

        await NotifyAlertBroadcaster(notify_conn, CHANNEL).notify(AlertId("alert_notify_1"))

        payload = await asyncio.wait_for(first, timeout=3.0)
        assert payload == "alert_notify_1"
        await agen.aclose()
    finally:
        await listen_conn.close()
        await notify_conn.close()
