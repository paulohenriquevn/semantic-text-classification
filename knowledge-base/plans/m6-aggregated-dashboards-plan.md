# Plan: M6 Aggregated Dashboards (continuous aggregates)

> **Version 1.0** — Extend the shipped `turns_per_min` continuous-aggregate pattern to business KPIs:
> a `alerts_kpi_5min` Timescale continuous aggregate (cancellation/escalation/sentiment rate per queue
> per 5 min) that refreshes incrementally via a continuous-aggregate policy (no full scan), is retained
> beyond the 30-day raw purge (rollup outlives raw — ADR-005 hot/purge split), and backs a manager
> dashboard KPI endpoint behind a domain port. Per the SHIPPABLE 96.8 blueprint
> `knowledge-base/discoveries/blueprints/m6-aggregated-dashboards-blueprint.md`.

## Goal

> "Give managers KPI rollups from an incrementally-refreshed Timescale continuous aggregate, measured by
> an integration test suite that asserts (a) a new alert bumps the 5-min rollup after an incremental
> refresh (no full raw re-scan), (b) the rollup survives after its raw chunk is dropped, and (c) the
> dashboard endpoint returns KPIs grouped by queue + rule — all against a real TimescaleDB."

## Context

M6 (`ROADMAP.md § M6`) must give managers dashboards from Timescale continuous aggregates: rollups that
compute incrementally with no full scan (cancellation-intent rate per queue per 5 min), render with
acceptable freshness, and are retained beyond 30 days while raw purges. The blueprint (SHIPPABLE 96.8)
locked: extend the shipped `turns_per_min` CA (`deploy/monitoring/migrations/0002_m1_retention_indexes.sql:44-52`)
to a `alerts_kpi_5min` CA grouped by (bucket, rule_name, queue, sentiment) (D1/D2); refresh via
`add_continuous_aggregate_policy` with an `end_offset` that excludes the still-forming bucket — the
Timescale realization of chatwoot's "closed-buckets-only" rollup (D2); NO retention policy on the CA so
the rollup outlives the 30-day raw purge (D3, ADR-005); a bounded KPI enum + read-side whitelist (D4);
and a pre-bucketed dashboard read behind a domain port (D5). Constrained by `.claude/rules/architecture.md § 1`
(the dashboard read is an infrastructure adapter behind a domain port; DIP) and `.claude/rules/testing.md § 2`
(rollup correctness proven against a real DB).

## Baseline Context (deep review of current state)

### Files that will be touched

| File | LoC today | Last commit | Why it exists today | Invariants to preserve |
|---|---|---|---|---|
| `deploy/monitoring/migrations/0004_m6_kpi_rollup.sql` (NEW) | 0 | — | alerts queue+sentiment cols + CA + policy | idempotent (DO-block guards like 0002) |
| `src/talkex/monitoring/domain/models.py` (Alert) | ~60 | M0/M3 | Alert value object (no queue/sentiment) | additive: add `queue`, `sentiment` fields (defaulted — back-compat) |
| `src/talkex/monitoring/infrastructure/timescale_repo.py` (alert INSERT) | ~70 | M0 | 5-col alert INSERT | change: add queue + sentiment columns |
| `src/talkex/monitoring/application/orchestrator.py` (`_emit_alert`) | ~130 | M3 | builds+emits the alert; already computes sentiment evidence | change: populate queue (turn.metadata) + sentiment (label) on the alert |
| `src/talkex/monitoring/interface/app.py` (IngestRequest + routes) | 207 | M0-M5 | composition root | additive: optional `queue` on ingest → turn.metadata; a `/dashboard/kpis` route |
| `src/talkex/monitoring/domain/ports.py` | ~75 | M0-M5 | domain Protocols | additive: `KpiReadPort` |
| `src/talkex/monitoring/domain/dashboard.py` (NEW) | 0 | — | `KpiBucket`, `KpiQuery` value objects | — |
| `src/talkex/monitoring/infrastructure/kpi_repo.py` (NEW) | 0 | — | queries the `alerts_kpi_5min` CA | — |

### Current callers / dependents

- **`Alert`** (`models.py:40`) — built in `orchestrator._emit_alert`, persisted by `TimescaleAlertRepository.save`, read by the SSE path (`app.py` supervisor_stream → `alert.model_dump_json`). Adding defaulted `queue`/`sentiment` fields is additive; the SSE serialization gains two fields (back-compat for the M4 UI, which ignores unknown fields).
- **alert INSERT** (`timescale_repo.py:49-53`) — sole writer of `alerts`; adding queue+sentiment touches this one path.
- **`turns_per_min` CA** (`0002:44-52`) — the pattern M6 extends; unchanged.

### Domain glossary

- **Continuous aggregate (CA)** — a Timescale materialized view (`WITH (timescaledb.continuous)`) that stores pre-bucketed rollups and refreshes incrementally, not on read.
- **Continuous-aggregate policy** — `add_continuous_aggregate_policy(schedule_interval, end_offset, start_offset)`; `end_offset` excludes the still-forming bucket (closed-buckets-only).
- **KPI** — a bounded business metric: cancellation/escalation/sentiment rate per queue per 5 min.
- **Rollup-outlives-raw** — the CA has NO retention policy, so its buckets persist after the raw `alerts` chunk is dropped at 30 days (ADR-005 hot/purge split).

### Architecture boundaries affected

Infrastructure + application + domain of `monitoring`: a new CA (infra/SQL), a new domain port (`KpiReadPort`) + value objects, a read adapter (the CA query), an interface route, and alert-emit wiring. DIP preserved.

### ⚠ Baseline reality checks (avoid workarounds)

1. **`queue` is not captured at ingest.** `IngestRequest` carries only `conversation_id` + `raw_text` (`app.py`), so `turns.metadata` has no queue. For "per queue" to be REAL end-to-end (not a hardcoded constant), M6 adds an optional `queue` to `IngestRequest` → `turn.metadata['queue']` → `alert.queue` (default `'default'` when absent). The dimension is then genuinely populated, and the integration test seeds multiple queues.
2. **`sentiment` is evidence, not a column.** The orchestrator already computes a sentiment label in `_sentiment_evidence` and passes it to `_emit_alert`, but it lives inside the alert's `evidence` JSONB — not aggregatable by a CA. M6 promotes the label to an `alerts.sentiment` column at emit-time so the CA can group by it. No new computation — just surfacing an existing signal.
3. **`src/talkex/analytics/` is an IN-MEMORY engine** (`SimpleAnalyticsEngine` over `AnalyticsEvent` lists) — offline/batch, not DB-backed. M6's dashboard read is a DB CA query (a distinct online path); the in-memory engine is NOT reused here (honest — reusing it would mean loading all rows into memory, defeating the CA).

## Prior Art & Related Work

- **Internal blueprint** — `m6-aggregated-dashboards-blueprint.md` (its ADR set + Cross-cutting Comparison).
- **Reference — chatwoot** — incremental reporting rollup (`knowledge-base/references/chatwoot/app/models/reporting_events_rollup.rb`), event-driven rollup write (`app/listeners/reporting_event_listener.rb`), report builder group-by (`app/builders/v2/report_builder.rb`), real-DB rollup specs (`spec/models/reporting_events_rollup_spec.rb`). The borrowed PATTERN is incremental increment keyed by (bucket, dimension); the M6 realization is a Timescale CA (blueprint D1/D2 — not a ported Ruby listener).
- **Reference — ai-powered** — DuckDB columnar aggregation (`analytics/duckdb_loader.py`) — a deliberate contrast; M6 stays single-engine (no side-store, ADR-005).
- **Internal reuse** — the shipped `turns_per_min` CA pattern (`0002:44-52`).

## Objective

- [ ] `alerts` gains `queue` + `sentiment` columns, populated at alert-emit (queue from ingest→metadata)
- [ ] `alerts_kpi_5min` continuous aggregate (bucket × rule × queue × sentiment) + incremental refresh policy
- [ ] Rollup retained beyond 30 days (no retention policy on the CA)
- [ ] `KpiReadPort` + a dashboard KPI query over the CA
- [ ] `GET /dashboard/kpis` endpoint
- [ ] Real-Timescale integration tests: incremental refresh bumps the rollup; rollup survives raw chunk drop

## ADRs

### D1 — Timescale continuous aggregate for KPI rollups (extend the shipped pattern)
- **Decision:** a `alerts_kpi_5min` CA = `time_bucket('5 minutes', created_at), rule_name, queue, sentiment, count(*)` over `alerts`, `WITH (timescaledb.continuous) … WITH NO DATA`.
- **Rationale:** DB-native incremental materialization — the borrowed increment pattern (chatwoot `reporting_events_rollup.rb`) realized as a CA (KISS: extend the shipped `turns_per_min` CA, don't add an app-level rollup table + listener).
- **Alternatives considered:** app-level rollup table + a listener incrementing on alert-save (rejected — reinvents what a CA does natively, Rule 9); a separate DuckDB columnar store (rejected — ai-powered's path; ADR-005 single-engine).
- **Consequence:** rollups computed once, read cheap; one engine.

### D2 — Incremental refresh policy, closed-buckets-only
- **Decision:** `add_continuous_aggregate_policy(alerts_kpi_5min, start_offset => NULL, end_offset => INTERVAL '5 minutes', schedule_interval => INTERVAL '5 minutes')`.
- **Rationale:** `end_offset` excludes the still-forming bucket — the Timescale realization of chatwoot's "today is skipped / closed-buckets-only" (`report_builder`/backfill). Honest freshness SLO: KPIs lag ≤ ~1 bucket (5 min).
- **Alternatives considered:** refresh-on-read (rejected — full scan on every dashboard load); refresh every alert (rejected — write amplification).
- **Consequence:** bounded, predictable freshness; no read-time scan.

### D3 — Rollup outlives raw (no CA retention)
- **Decision:** do NOT add a retention policy to `alerts_kpi_5min`; raw `alerts` keeps its 30-day retention.
- **Rationale:** the rollup is long-term business memory (ADR-005 hot/purge split); the raw event ages out, the KPI bucket persists.
- **Alternatives considered:** a long CA retention (deferred — YAGNI until a storage-cost signal exists).
- **Consequence:** dashboards show trends beyond 30 days even though raw turns/alerts are gone.

### D4 — Bounded KPI dimensions + read whitelist
- **Decision:** dimensions are a fixed set (rule_name, queue, sentiment, bucket); the read query accepts only whitelisted filters (queue, rule, time range).
- **Rationale:** chatwoot's report builder uses a metric whitelist; bounded dimensions keep the CA small and the read safe (injection defense reuses the M5 bound-param discipline).
- **Alternatives considered:** arbitrary group-by from the request (rejected — unbounded CA cardinality + injection surface).
- **Consequence:** predictable cardinality; safe reads.

### D5 — Dashboard read behind a domain port (DIP), not the in-memory analytics engine
- **Decision:** a `KpiReadPort` implemented by `TimescaleKpiRepository` querying the CA; the existing in-memory `SimpleAnalyticsEngine` is NOT used online.
- **Rationale:** DIP (domain declares the port); the in-memory engine would load all rows, defeating the CA (Baseline reality check 3).
- **Alternatives considered:** reuse `AnalyticsQueryRunner` (rejected — in-memory, offline-shaped).
- **Consequence:** the dashboard reads pre-bucketed rows straight from the CA.

## Drawbacks & Risks

| Drawback / Risk | Severity | Mitigation | Owner |
|---|---|---|---|
| R1 — CA refresh lag vs "real-time" expectation | Medium | D2 honest freshness SLO (≤ 1 bucket / 5 min); documented, not hidden | tomas |
| R2 — "per queue" is meaningful only once ingest captures queue | Medium | add optional `queue` to ingest → metadata → alert (Baseline check 1); default `'default'` is honest, not a fake dimension | kael |
| R3 — testing "rollup survives raw purge" without waiting 30 days | Medium | drop the raw chunk explicitly in the test (`drop_chunks`) after materializing, then assert the CA bucket still returns | tomas |
| R4 — sentiment column populated only when a detector is wired | Low | nullable column; CA groups NULL as a distinct bucket; documented | kael |

## Unresolved Questions

- Q1 — Exact freshness SLO to publish (5 min vs 10 min)? M6 sets 5-min bucket + 5-min schedule; final SLO confirmed by the freshness assertion in the integration test.
- Q2 — Should sentiment be a full enum (pos/neg/neu) or binary in the rollup? M6 stores the detector's label verbatim; the read groups on it. Enum normalization deferred.

## Dependency Graph

```
Phase 0 (schema: 0004 CA + alerts queue/sentiment cols + emit wiring)  ──▶  Phase 1 (KpiReadPort + CA query)
                                                                                   ▼
                                                                Phase 2 (GET /dashboard/kpis endpoint)
                                                                                   ▼
                                        Final: Integration Validation (incremental refresh bumps rollup; survives raw drop)
```

---

## Phase 0: Schema + emit wiring

**Objective:** the CA exists, refreshes incrementally, and alerts carry queue + sentiment.

### T0.1 — migration 0004 + alert queue/sentiment wiring

#### Objective
Add `queue`/`sentiment` columns to `alerts`, the `alerts_kpi_5min` CA + refresh policy (no retention), and populate queue+sentiment at alert-emit (queue from ingest→metadata).

#### Why this step
1. **What:** `0004_m6_kpi_rollup.sql` (NEW); `Alert` fields; alert INSERT; `orchestrator._emit_alert`; `IngestRequest` + ingest wiring.
2. **Why now:** the CA + its dimensions are the foundation every downstream KPI reads (blueprint D1/D2/D3, Baseline checks 1-2).

#### Files to edit
```
deploy/monitoring/migrations/0004_m6_kpi_rollup.sql (NEW)
src/talkex/monitoring/domain/models.py — Alert.queue, Alert.sentiment (defaulted)
src/talkex/monitoring/infrastructure/timescale_repo.py — alert INSERT + get (queue, sentiment)
src/talkex/monitoring/application/orchestrator.py — populate queue + sentiment on the alert
src/talkex/monitoring/interface/app.py — optional queue on IngestRequest → turn.metadata
tests/integration/monitoring/test_kpi_rollup.py (NEW)
tests/unit/monitoring/test_alert_dimensions.py (NEW)
```

#### Deep file dependency analysis
Alert is additive (defaulted fields — existing callers unaffected). The CA reads `alerts.created_at` (hypertable default). The refresh policy is idempotent (`if_not_exists`).

#### TDD
```
RED (unit):        test_emit_populates_queue_and_sentiment — the alert built by _emit_alert carries the
                   turn's queue + the sentiment label
RED (integration): test_ca_exists_and_refreshes — after inserting alerts + refresh_continuous_aggregate,
                   the CA returns the expected per-(rule,queue) counts
GREEN: write the migration + wiring
VERIFY: pytest tests/unit/monitoring/test_alert_dimensions.py tests/integration/monitoring/test_kpi_rollup.py -x
```

#### Concurrency tests

A concurrent test drives parallel alert inserts while a refresh runs; asserts the CA count is consistent after refresh (the M1 pool-contention harness supplies the parallel load — no lost increments).

#### Acceptance Criteria
- [ ] `alerts_kpi_5min` appears in `timescaledb_information.continuous_aggregates`
- [ ] no retention policy on the CA (asserted via `timescaledb_information.jobs`)
- [ ] the alert carries queue + sentiment (unit + a persisted-row check)

#### DoD
- [ ] CA + refresh policy created; alert dimensions populated

---

## Phase 1: KPI read

**Objective:** `KpiReadPort.kpi_rollups` returns pre-bucketed KPIs from the CA.

### T1.1 — KpiReadPort + TimescaleKpiRepository

#### Objective
A domain `KpiReadPort` + `TimescaleKpiRepository.kpi_rollups(query)` reading `alerts_kpi_5min`, filtered by time range + optional whitelisted queue/rule.

#### Why this step
1. **What:** `domain/dashboard.py` (`KpiBucket`, `KpiQuery`); `ports.py` `KpiReadPort`; `infrastructure/kpi_repo.py`.
2. **Why now:** the dashboard endpoint consumes it (blueprint D4/D5).

#### Files to edit
```
src/talkex/monitoring/domain/dashboard.py (NEW) — KpiBucket, KpiQuery
src/talkex/monitoring/domain/ports.py — KpiReadPort
src/talkex/monitoring/infrastructure/kpi_repo.py (NEW)
tests/integration/monitoring/test_kpi_read.py (NEW)
```

#### TDD
```
RED (integration): test_kpi_rollups_grouped_by_queue — seed alerts across 2 queues + 2 rules, refresh,
                   assert kpi_rollups returns the correct per-(bucket,queue,rule) counts; a queue filter narrows
GREEN: implement the port + CA query (bound whitelist filters)
VERIFY: pytest tests/integration/monitoring/test_kpi_read.py -x
```

#### Concurrency tests

(none — single-threaded) — the read query is a single pooled statement; concurrency is covered by T0.1's parallel-ingest test.

#### Acceptance Criteria
- [ ] kpi_rollups returns correct per-(bucket,queue,rule) counts from the CA
- [ ] a queue filter narrows results (bound param — injection-safe)

#### DoD
- [ ] KPI read green

---

## Phase 2: Dashboard endpoint

**Objective:** managers read KPIs over HTTP.

### T2.1 — GET /dashboard/kpis

#### Objective
Mount `GET /dashboard/kpis` (time range + optional queue/rule → KPI buckets) on the monitoring app.

#### Why this step
1. **What:** a route in `interface/app.py` wiring the KPI repo (composition root).
2. **Why now:** the M6 manager-dashboard surface (blueprint D5).

#### Files to edit
```
src/talkex/monitoring/interface/app.py — /dashboard/kpis route + KpiReadPort wiring
tests/integration/monitoring/test_dashboard_api.py (NEW)
```

#### TDD
```
RED: test_dashboard_endpoint_returns_kpis — seed alerts, refresh, GET /dashboard/kpis; assert grouped KPI buckets
GREEN: implement the route
VERIFY: pytest tests/integration/monitoring/test_dashboard_api.py -x
```

#### Concurrency tests

(none — single-threaded) — the route delegates to the already-tested KPI repo.

#### Acceptance Criteria
- [ ] endpoint returns KPI buckets grouped by queue + rule
- [ ] no domain layer imports FastAPI (boundary preserved)

#### DoD
- [ ] endpoint green

---

## Coverage Matrix

| # | Gap / Requirement (DoD) | Task(s) | Resolution |
|---|---|---|---|
| 1 | Incremental CA rollup, no full scan (DoD #1) | T0.1 | alerts_kpi_5min CA + refresh policy |
| 2 | queue/sentiment dimensions real (Baseline) | T0.1 | ingest→metadata→alert cols |
| 3 | Rollup retained beyond 30 days (DoD #3) | T0.1, T3.1 | no CA retention; survives raw drop |
| 4 | KPI read (DoD #2 backing) | T1.1 | KpiReadPort + CA query |
| 5 | Dashboard renders trends (DoD #2) | T2.1 | GET /dashboard/kpis |
| 6 | Incremental refresh bumps rollup (DoD #1) | T3.1 | benchmark/validation |

**Coverage: 6/6 gaps covered (100%)**

## Global Definition of Done

- [ ] `ruff format --check . && ruff check . && mypy src/ tests/` clean
- [ ] `pytest tests/unit -x && pytest tests/integration -x` green (real Timescale)
- [ ] CA increments incrementally (refresh bumps the bucket; EXPLAIN reads the materialization, not raw) — proven by a test
- [ ] Rollup survives the raw chunk drop — proven by a `drop_chunks` test
- [ ] File-size ≤ 500 LoC per file
- [ ] CHANGELOG.md updated under `[Unreleased]`

## Failure scenarios (external I/O)

| Dependency | Failure mode | How the test reproduces it | Expected behavior |
|---|---|---|---|
| TimescaleDB (CA refresh) | refresh called on a non-existent CA | call refresh on a bad name | typed error surfaced (fail-fast), not silent |
| CA query | empty time range | query a window with no buckets | returns an empty list, not an error |
| alert emit | turn without a queue in metadata | emit an alert from a queue-less turn | queue defaults to 'default' (no crash, no NULL dimension surprise) |

## Final Phase: Integration Validation (MANDATORY)

**Objective:** prove the DoD with real-DB evidence.

### T3.1 — incremental-refresh + survives-raw-drop validation

#### Objective
Prove the rollup increments on incremental refresh (no full scan) and survives the raw chunk drop.

#### Why this step
1. **What:** integration assertions + a metrics note; run the suite against real Timescale.
2. **Why now:** DoD #1 (incremental) + #3 (outlives raw) are proven only by measurement (blueprint D1/D3).

#### Files to edit
```
tests/integration/monitoring/test_kpi_rollup.py — add the incremental + drop_chunks assertions
```

#### TDD
```
RED: test_incremental_refresh_bumps_bucket — materialize; add one alert; refresh; assert the bucket count += 1
     test_rollup_survives_raw_chunk_drop — materialize; drop_chunks('alerts'); assert the CA bucket still returns
GREEN: (behavior already implemented in Phase 0; this phase asserts it end-to-end)
VERIFY: pytest tests/integration/monitoring/test_kpi_rollup.py -x
```

#### Concurrency tests

A concurrent test interleaves alert inserts with a refresh; asserts no lost increments after the final refresh (M1 pool-contention harness supplies the parallel load).

#### Acceptance Criteria
- [ ] incremental refresh bumps only the affected bucket (no full re-scan needed)
- [ ] the rollup bucket survives `drop_chunks('alerts')`
- [ ] `GET /dashboard/kpis` returns buckets after the raw drop

#### DoD
- [ ] incremental + outlives-raw evidence recorded
