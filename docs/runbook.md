# Operational Runbook — Real-Time Attendance Monitoring

Operational guide for the TalkEx monitoring service (M0–M8). Covers startup, health probes, the V1
acceptance gate, and the failure-recovery posture. Companion to `docs/adr/ADR-005`.

## 1. Startup

```bash
# 1. Storage (TimescaleDB + pgvector) — port 5433
docker compose -f deploy/monitoring/docker-compose.yml up -d

# 2. Apply migrations in order (0001 → 0004) if the DB is fresh
#    (docker-entrypoint-initdb.d runs them on first init; apply manually on an existing volume)

# 3. Serve the monitor
uvicorn talkex.monitoring.interface.app:create_app --factory --host 0.0.0.0 --port 8000
```

## 2. Health probes (M8)

| Endpoint | Meaning | Healthy | Unhealthy |
|---|---|---|---|
| `GET /health` | Liveness — the process is up | `200 {"status":"alive"}` | connection refused (process down) |
| `GET /ready` | Readiness — can it do work? | `200 {"ready":true,...}` | `503 {"ready":false,...}` |

`/ready` is **not-ready (503)** when any of:

- the session is not `LISTENING` (still initializing or closing);
- the bounded ingest queue is ≥ 90% of capacity (**backpressure** — the producer is outrunning the consumer);
- the DB is unreachable (`SELECT 1` fails).

**Operator action on 503:** read the body. `db_reachable:false` → check TimescaleDB. `queue_depth` near
`queue_maxsize` → the consumer is behind; check for a slow rule/DB, scale the consumer, or shed load upstream.

## 3. V1 acceptance gate (M8)

Before claiming V1, run the acceptance harness — it RE-RUNS every ship criterion (no hard-coded PASS):

```bash
# Produce a fresh retrieval benchmark first (feeds the retrieval-p95 criterion)
python experiments/scripts/bench_hybrid_search.py

# Then run the gate
python experiments/scripts/v1_acceptance.py --out experiments/results/v1_readiness.json
```

Criteria: retrieval p95 < 200 ms, alert p95 < 2 s, critical-alert precision ≥ 0.8, sentiment macro-F1 ≥
0.70, purge working. The verdict is **PASS only when every criterion passes**; a missing artifact or a
failing check is a **FAIL**, never a silent PASS.

> **Honesty caveat (in the report):** the criteria are validated on **synthetic load**. Real 8 kHz
> call-center drift is **NOT** yet validated — re-run this harness against real pilot data before an
> unqualified V1 claim.

## 4. Engagement (north-star proxy)

```bash
GET /dashboard/engagement?from_time=...&to_time=...
# → {"alerts": A, "labels": L, "acted_on_rate": L/A}
```

`acted_on_rate` = supervisor labels ÷ alerts in the window (a QA label = acting on an alert). It is an
**instrumented proxy**; its production value is meaningful only under a real pilot with supervisors.

## 5. Failure-recovery posture

The recovery mechanisms are shipped and tested (M0/M1/M4); this section documents how they behave:

| Failure | Mechanism | Behavior |
|---|---|---|
| Ingest burst (producer faster than consumer) | **Bounded channel backpressure** (`TurnChannel`, M0) | `put()` blocks at capacity; `/ready` flips to 503 — no unbounded memory growth |
| Ingest×query contention | **Sized connection pool** (`MonitoringPool`, M1) | queries queue for a connection (no churn, no error) under load |
| Supervisor SSE drop | **`EventSource` auto-reconnect** (M4) | the browser reconnects automatically; the stream resumes |
| DB transient unavailability | **Readiness probe** (M8) | `/ready` reports 503; traffic can be gated until the DB returns |
| Raw-data purge (30 days) | **Export-before-purge** (M7) + **CA outlives raw** (M6) | retraining sample exported before the drop; KPI rollups persist |

## 6. Data lifecycle (M7)

```bash
# Export an anonymized retraining sample BEFORE the purge, then retrain + benchmark
python experiments/scripts/run_retraining.py --days 30 --now <ISO-timestamp>
```

Raw turns/alerts purge at 30 days; the anonymized Parquet cold sample (PII-redacted) is exported first,
and a `drop_chunks` never runs behind an un-verified export.
