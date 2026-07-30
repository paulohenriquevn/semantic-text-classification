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
from datetime import datetime
from typing import Any
from uuid import uuid4

import psycopg
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel
from talkex.models.types import ConversationId
from talkex.monitoring.application.health_service import HealthService
from talkex.monitoring.application.orchestrator import TurnOrchestrator
from talkex.monitoring.application.search_service import SearchService
from talkex.monitoring.application.session import MonitoringSession
from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.domain.channel import TurnChannel
from talkex.monitoring.domain.critical_rules import build_critical_rules
from talkex.monitoring.domain.dashboard import KpiQuery
from talkex.monitoring.domain.models import AlertId
from talkex.monitoring.domain.ports import TurnEmbedder
from talkex.monitoring.domain.search import Criterion, Label, SearchQuery
from talkex.monitoring.infrastructure.db_probe import PoolDbProbe
from talkex.monitoring.infrastructure.embedder import SentenceTransformerEmbedder
from talkex.monitoring.infrastructure.kpi_repo import TimescaleKpiRepository
from talkex.monitoring.infrastructure.label_repo import TimescaleLabelRepository
from talkex.monitoring.infrastructure.notify_broadcaster import NotifyAlertBroadcaster
from talkex.monitoring.infrastructure.pool import MonitoringPool
from talkex.monitoring.infrastructure.read_repo import TimescaleReadRepository
from talkex.monitoring.infrastructure.timescale_repo import (
    TimescaleAlertRepository,
    TimescaleTurnRepository,
)
from talkex.segmentation.config import SegmentationConfig
from talkex.segmentation.segmenter import TurnSegmenter

# M0 critical rule: cancellation-risk (blueprint Q3 — DSL is the online decide step, not an LLM).
_SEG_CONFIG = SegmentationConfig()


class IngestRequest(BaseModel):
    """A posted transcript chunk (labeled speakers, e.g. '[customer] quero cancelar')."""

    conversation_id: str
    raw_text: str
    queue: str = "default"  # M6 KPI dimension — the service queue/domain this conversation belongs to


class CriterionRequest(BaseModel):
    """A single QA filter predicate in a search request."""

    field: str
    value: str


class SearchRequest(BaseModel):
    """A QA hybrid-search request (criterion + query over the retention window)."""

    query_text: str
    top_k: int = 10
    window_days: int = 30
    criteria: list[CriterionRequest] = []


class LabelRequest(BaseModel):
    """A QA audit/label action destined for retraining."""

    turn_id: str
    conversation_id: str
    label: str
    labeled_by: str = "qa"


def create_app(config: MonitoringConfig | None = None, embedder: TurnEmbedder | None = None) -> FastAPI:
    """Build the wired FastAPI app (composition root).

    `embedder` is injectable (DIP) so tests can pass a download-free `DeterministicEmbedder`;
    production defaults to a real multilingual MiniLM.
    """
    cfg = config or MonitoringConfig()
    turn_embedder = embedder or SentenceTransformerEmbedder()
    segmenter = TurnSegmenter()
    rules = build_critical_rules()  # M3 critical-rule catalogue

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        write_conn = await psycopg.AsyncConnection.connect(cfg.dsn)
        notify_conn = await psycopg.AsyncConnection.connect(cfg.dsn, autocommit=True)
        channel = TurnChannel(maxsize=cfg.queue_maxsize)
        # M5 Phase 0: write the pgvector embedding at ingest so the ANN half is live. The ~tens-of-ms
        # encode cost sits on the pre-alert write path but well within the M3 turn→alert p95 < 2s budget.
        orchestrator = TurnOrchestrator(
            turn_repo=TimescaleTurnRepository(write_conn, embedder=turn_embedder),
            alert_repo=TimescaleAlertRepository(write_conn),
            broadcaster=NotifyAlertBroadcaster(notify_conn, cfg.notify_channel),
            rules=rules,
        )
        session = MonitoringSession(channel, orchestrator)
        await session.start()
        # M5 Phase 3: a pooled read path for QA hybrid search + label persistence (ingest/query split).
        read_pool = MonitoringPool(cfg.dsn)
        await read_pool.open()
        app.state.channel = channel
        app.state.session = session
        app.state.search_service = SearchService(TimescaleReadRepository(read_pool), turn_embedder)
        app.state.label_repo = TimescaleLabelRepository(read_pool)
        app.state.kpi_repo = TimescaleKpiRepository(read_pool)  # M6 manager-dashboard read
        app.state.health = HealthService(session, channel, PoolDbProbe(read_pool))  # M8 readiness probe
        try:
            yield
        finally:
            await session.aclose()
            await read_pool.close()
            await write_conn.close()
            await notify_conn.close()

    app = FastAPI(title="TalkEx Monitoring (M0)", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        # M8 liveness — the process is up and serving (readiness is a separate, deeper probe).
        return {"status": "alive"}

    @app.get("/ready")
    async def ready(response: Response) -> dict[str, object]:
        # M8 readiness — session LISTENING + queue not saturated + DB reachable, else 503.
        report = await app.state.health.readiness()
        if not report.ready:
            response.status_code = 503
        return report.model_dump()

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
            # M6: carry the queue on each turn's metadata so the alert (and its KPI rollup) is scoped.
            queued = turn.model_copy(update={"metadata": {**turn.metadata, "queue": req.queue}})
            await app.state.channel.put(queued)
        return {"enqueued": len(turns)}

    @app.post("/search")
    async def search(req: SearchRequest) -> list[dict[str, Any]]:
        # M5 QA hybrid search — fused evidence-carrying windows over the retention window.
        query = SearchQuery(
            query_text=req.query_text,
            top_k=req.top_k,
            window_days=req.window_days,
            criteria=tuple(Criterion(field=c.field, value=c.value) for c in req.criteria),
        )
        hits = await app.state.search_service.search(query)
        return [h.model_dump(mode="json") for h in hits]

    @app.get("/dashboard/kpis")
    async def dashboard_kpis(
        from_time: datetime, to_time: datetime, queue: str | None = None, rule_name: str | None = None
    ) -> list[dict[str, Any]]:
        # M6 manager dashboard — pre-bucketed KPI rollups straight from the continuous aggregate.
        query = KpiQuery(from_time=from_time, to_time=to_time, queue=queue, rule_name=rule_name)
        buckets = await app.state.kpi_repo.kpi_rollups(query)
        return [b.model_dump(mode="json") for b in buckets]

    @app.post("/label", status_code=201)
    async def label(req: LabelRequest) -> dict[str, str]:
        # M5 QA audit action — persist a label destined for retraining.
        label = Label(
            label_id=f"lbl_{uuid4().hex}",
            turn_id=req.turn_id,
            conversation_id=req.conversation_id,
            label=req.label,
            labeled_by=req.labeled_by,
        )
        await app.state.label_repo.save(label)
        return {"label_id": label.label_id}

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
