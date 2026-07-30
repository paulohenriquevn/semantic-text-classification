"""Composition root — FastAPI app wiring the M0 walking skeleton (interface layer).

Wires the thin edges (blueprint D1/D3/D4): POST /ingest segments a posted transcript into
Turns (reusing TurnSegmenter, DoD-2) and enqueues them on the session's bounded channel
(never doing rule/DB work inline, D1); GET /supervisor/stream is an SSE endpoint that LISTENs
for alert ids and forwards each with its full evidence read from the alert repo (D3, R2);
GET /supervisor serves a minimal page. Concretes are injected here, not in the domain.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel
from talkex.models.types import ConversationId
from talkex.monitoring.application.orchestrator import TurnOrchestrator
from talkex.monitoring.application.session import MonitoringSession
from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.domain.channel import TurnChannel
from talkex.monitoring.domain.models import AlertId
from talkex.monitoring.infrastructure.notify_broadcaster import NotifyAlertBroadcaster
from talkex.monitoring.infrastructure.timescale_repo import (
    TimescaleAlertRepository,
    TimescaleTurnRepository,
)
from talkex.rules.compiler import SimpleRuleCompiler
from talkex.segmentation.config import SegmentationConfig
from talkex.segmentation.segmenter import TurnSegmenter

# M0 critical rule: cancellation-risk (blueprint Q3 — DSL is the online decide step, not an LLM).
_M0_RULE_DSL = 'contains_any("cancelar", "cancelamento", "cancela")'
_SEG_CONFIG = SegmentationConfig()


class IngestRequest(BaseModel):
    """A posted transcript chunk (labeled speakers, e.g. '[customer] quero cancelar')."""

    conversation_id: str
    raw_text: str


def create_app(config: MonitoringConfig | None = None) -> FastAPI:
    """Build the wired FastAPI app (composition root)."""
    cfg = config or MonitoringConfig()
    segmenter = TurnSegmenter()
    rule = SimpleRuleCompiler().compile(dsl_text=_M0_RULE_DSL, rule_id="rule_m0_cancel", rule_name="cancellation_risk")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        write_conn = await psycopg.AsyncConnection.connect(cfg.dsn)
        notify_conn = await psycopg.AsyncConnection.connect(cfg.dsn, autocommit=True)
        channel = TurnChannel(maxsize=cfg.queue_maxsize)
        orchestrator = TurnOrchestrator(
            turn_repo=TimescaleTurnRepository(write_conn),
            alert_repo=TimescaleAlertRepository(write_conn),
            broadcaster=NotifyAlertBroadcaster(notify_conn, cfg.notify_channel),
            rule=rule,
        )
        session = MonitoringSession(channel, orchestrator)
        await session.start()
        app.state.channel = channel
        app.state.session = session
        try:
            yield
        finally:
            await session.aclose()
            await write_conn.close()
            await notify_conn.close()

    app = FastAPI(title="TalkEx Monitoring (M0)", lifespan=lifespan)

    @app.post("/ingest", status_code=202)
    async def ingest(req: IngestRequest) -> dict[str, int]:
        # Reuse TurnSegmenter (DoD-2), then only enqueue — no inline rule/DB work (D1).
        transcript = TranscriptInput(
            conversation_id=ConversationId(req.conversation_id),
            raw_text=req.raw_text,
            source_format=SourceFormat.LABELED,
            channel=Channel.VOICE,
        )
        turns = segmenter.segment(transcript, _SEG_CONFIG)
        for turn in turns:
            await app.state.channel.put(turn)
        return {"enqueued": len(turns)}

    @app.get("/supervisor/stream")
    async def supervisor_stream() -> EventSourceResponse:
        async def events() -> AsyncIterator[dict[str, Any]]:
            listen_conn = await psycopg.AsyncConnection.connect(cfg.dsn, autocommit=True)
            read_conn = await psycopg.AsyncConnection.connect(cfg.dsn, autocommit=True)
            alert_repo = TimescaleAlertRepository(read_conn)
            broadcaster = NotifyAlertBroadcaster(listen_conn, cfg.notify_channel)
            try:
                async for alert_id in broadcaster.listen():
                    alert = await alert_repo.get(AlertId(alert_id))
                    if alert is not None:
                        yield {"event": "alert", "data": alert.model_dump_json()}
            finally:
                await listen_conn.close()
                await read_conn.close()

        return EventSourceResponse(events())

    @app.get("/supervisor", response_class=HTMLResponse)
    async def supervisor_page() -> str:
        return _SUPERVISOR_HTML

    return app


_SUPERVISOR_HTML = """<!doctype html>
<html lang="pt-br"><head><meta charset="utf-8"><title>TalkEx — Supervisor (M0)</title></head>
<body>
<h1>Monitoramento ao vivo</h1>
<ul id="alerts"></ul>
<script>
const es = new EventSource("/supervisor/stream");
es.addEventListener("alert", (e) => {
  const a = JSON.parse(e.data);
  const li = document.createElement("li");
  li.textContent = `[${a.rule_name}] conversa ${a.conversation_id} — evidência: ` +
    JSON.stringify(a.evidence.map(x => x.matched_text || x.predicate_type));
  document.getElementById("alerts").prepend(li);
});
</script>
</body></html>
"""
