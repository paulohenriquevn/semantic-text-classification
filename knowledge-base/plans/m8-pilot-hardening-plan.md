# Plan: M8 Pilot Hardening & V1 Ship

> **Version 1.0** — Harden the monitor for operation and prove every V1 ship criterion: a `/health`
> (liveness) + `/ready` (readiness: session LISTENING + queue not saturated + DB reachable) endpoint, an
> alert-engagement metric (`acted_on_rate` = labels/window ÷ alerts/window, reusing M5 labels), a V1-acceptance
> harness that RE-RUNS each criterion (no hard-coded PASS) and emits a readiness report, and an operational
> runbook. Per the SHIPPABLE 97.2 blueprint `knowledge-base/discoveries/blueprints/m8-pilot-hardening-blueprint.md`.
> Honest scope: no real 8 kHz pilot data exists here, so V1 criteria are validated on synthetic load and the
> real-drift risk is documented (Rule 3).

## Goal

> "Harden the monitor and gate the V1 claim, measured by an acceptance harness that RE-RUNS every V1 ship
> criterion (alert p95 < 2 s, retrieval p95 < 200 ms, sentiment macro-F1 ≥ 0.70, critical-alert precision ≥
> 0.8, purge working) and emits a readiness report whose verdict is PASS only when every criterion passes on
> live evidence — plus a `/ready` probe and an `acted_on_rate` engagement metric, all test-proven."

## Context

M8 (`ROADMAP.md § M8`) is the capstone: prove every V1 ship criterion, track supervisor alert-engagement
(north-star proxy), and add health self-monitoring + an operational runbook + failure recovery. It depends on
M0-M7, whose evidence already exists (alert p95 < 2 s + precision 0.89 from M3; retrieval p95 = 160.95 ms from
M5, `experiments/results/m5_hybrid_bench.json`; sentiment macro-F1 0.856 from M2; purge/rollup-survives-drop
from M6). The blueprint (SHIPPABLE 97.2) locked: a readiness probe over the existing session/channel/pool (D1);
frozen value-object self-metrics (D2); a V1-acceptance report that RE-RUNS live checks / reads fresh metrics
artifacts — explicit "no hard-coded PASS" (D3); an `acted_on_rate` engagement metric reusing `POST /label`
(D4); a failure-recovery posture over the shipped backpressure + pool queueing (D5); and an honesty caveat —
synthetic-load labeled, 8 kHz drift unvalidated (D6). Constrained by `.claude/rules/architecture.md § 1` (the
health endpoint is an interface adapter) and `.claude/rules/testing.md` (the harness re-runs real checks, not a
hard-coded PASS — the same acceptance-theatre failure the plan-confidence golden rule guards against).

## Baseline Context (deep review of current state)

### Files that will be touched

| File | LoC today | Last commit | Why it exists today | Invariants to preserve |
|---|---|---|---|---|
| `src/talkex/monitoring/interface/app.py` | 223 | M0-M7 | composition root (ingest/SSE/search/label/dashboard) | additive: `/health`, `/ready`, `/dashboard/engagement` routes |
| `src/talkex/monitoring/domain/channel.py` | ~45 | M0 | `TurnChannel.qsize()`/`.closed`/`.maxsize` | reused read-only (queue-depth probe) |
| `src/talkex/monitoring/application/session.py` | ~60 | M0 | `MonitoringSession.state` (SessionState) | reused read-only (session-running probe) |
| `src/talkex/monitoring/infrastructure/pool.py` | ~40 | M1 | `MonitoringPool.connection()` | reused read-only (DB-reachable probe) |
| `src/talkex/monitoring/domain/health.py` (NEW) | 0 | — | `HealthReport`, `EngagementMetric` value objects | — |
| `src/talkex/monitoring/application/health_service.py` (NEW) | 0 | — | readiness probe + engagement query | — |
| `src/talkex/monitoring/infrastructure/engagement_repo.py` (NEW) | 0 | — | labels-per-window / alerts-per-window | — |
| `experiments/scripts/v1_acceptance.py` (NEW) | 0 | — | runs each V1 criterion → readiness report | — |
| `docs/runbook.md` (NEW) | 0 | — | operational runbook | — |

### Current callers / dependents

- **`MonitoringSession.state`** (`session.py:33`) — read by the readiness probe; SessionState.LISTENING = healthy.
- **`TurnChannel.qsize()`** (`channel.py:37`) — read by the probe; near-`maxsize` = backpressure/degraded.
- **`labels` table** (M5) + **`alerts`** (M0/M6) — the engagement metric's two counts (a label = an act-on-alert).
- **`app.state.session`/`channel`** — set in the lifespan; the probe reads them.

### Domain glossary

- **Liveness** — the process is up (always 200 unless crashed).
- **Readiness** — the monitor can do work: session LISTENING, queue not saturated, DB reachable (200 or 503).
- **acted_on_rate** — engagement proxy: labels created in a window ÷ alerts raised in that window.
- **V1-acceptance harness** — re-runs each ship criterion and emits a PASS/FAIL-per-criterion readiness report.

### Architecture boundaries affected

Interface + application + domain of `monitoring`: two new value objects, a health application service (probe +
engagement), an engagement read adapter, and three routes. An offline acceptance script + a runbook doc. DIP
preserved (the probe reads domain state; the engagement adapter is infrastructure).

### ⚠ Baseline reality checks

1. **No real pilot data.** The V1 criteria are validated on synthetic load (as M2-M6 did). The acceptance
   harness proves the criteria hold on that evidence; real-world 8 kHz drift (ROADMAP risk 1) is unvalidated
   and MUST be labeled in the report (Rule 3) — no unqualified "production-ready" claim.
2. **Engagement rate is a proxy.** The plumbing (label→count→rate) is real and testable; the RATE on synthetic
   data is not a production signal. The metric is instrumented-and-tested; its production value awaits a pilot.
3. **The harness MUST re-run, not hard-code.** Each criterion is sourced from a live check or a freshly-produced
   metrics artifact — a hard-coded PASS is acceptance theatre (blueprint D3 / EC-1).

## Prior Art & Related Work

- **Internal blueprint** — `m8-pilot-hardening-blueprint.md` (its ADR set + Cross-cutting Comparison).
- **Reference — chatwoot** — a liveness health controller (`knowledge-base/references/chatwoot/app/controllers/health_controller.rb`) + its spec (`spec/controllers/health_controller_spec.rb`); an engagement/presence tracker (`lib/online_status_tracker.rb`).
- **Reference — livekit** — a self-metrics collect+summarize pattern (`knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/metrics/usage_collector.py`, `metrics/base.py`).
- **Internal reuse** — the shipped session/channel/pool (probes), the M5 labels + M6 alerts (engagement), and the M2/M3/M5/M6 benchmarks (V1 criteria evidence).

## Objective

- [ ] `/health` (liveness) + `/ready` (readiness: session + queue + DB) endpoints
- [ ] `acted_on_rate` engagement metric + `GET /dashboard/engagement`
- [ ] A V1-acceptance harness that RE-RUNS each criterion → a readiness report (no hard-coded PASS)
- [ ] An operational runbook (`docs/runbook.md`)

## ADRs

### D1 — Liveness/readiness split over the shipped session/channel/pool
- **Decision:** `/health` = liveness (200 if up); `/ready` = readiness probing session LISTENING + queue < saturation + DB reachable, returning 200 or 503.
- **Rationale:** chatwoot's health is liveness-only (`health_controller.rb`); a real orchestrator needs readiness. Reuses shipped state (`architecture.md § 1` — interface adapter); no new infra.
- **Alternatives considered:** a single health endpoint (rejected — cannot distinguish "up" from "can work"); an external probe (rejected — the monitor should self-report).
- **Consequence:** an operator (or k8s) can gate traffic on readiness.

### D2 — Frozen value-object self-metrics
- **Decision:** `HealthReport`/`EngagementMetric` as frozen pydantic value objects.
- **Rationale:** livekit's metrics are frozen records with a discriminator (`base.py`); immutable snapshots are safe to serialize.
- **Alternatives considered:** ad-hoc dicts (rejected — no schema).
- **Consequence:** typed, serializable health/engagement.

### D3 — V1-acceptance harness RE-RUNS, never hard-codes
- **Decision:** the harness runs each criterion live (re-run the M2/M3/M6 checks; read a freshly-produced `m5_hybrid_bench.json`) and computes a per-criterion PASS/FAIL + an overall verdict.
- **Rationale:** a hard-coded PASS is acceptance theatre (`testing.md`; plan-confidence golden rule). Blueprint D3 / EC-1.
- **Alternatives considered:** embed historical numbers as constants (rejected — proves nothing on the current tree).
- **Consequence:** the readiness verdict reflects the ACTUAL current state.

### D4 — Engagement = acted_on_rate (labels ÷ alerts), reusing M5 labels
- **Decision:** `acted_on_rate` = labels created in a window ÷ alerts raised in that window (a label = a supervisor acting on an alert).
- **Rationale:** reuses the shipped `POST /label` + labels table (DRY); the north-star proxy without new instrumentation.
- **Alternatives considered:** a new ack endpoint (rejected — a label already IS an act-on-alert); client-side tracking (rejected — not durable).
- **Consequence:** engagement is queryable from existing durable data; framed as a proxy (production value pending a pilot).

### D5 — Failure recovery over shipped backpressure + pool queueing
- **Decision:** document the recovery posture — bounded-channel backpressure (`channel.py`), pool queueing under load (`pool.py`), SSE auto-reconnect (M4 EventSource) — in the runbook; the readiness probe surfaces degradation.
- **Rationale:** the mechanisms already exist and are tested (M0/M1/M4); M8 makes them operable, not newly built (YAGNI).
- **Alternatives considered:** a new circuit breaker (rejected — no evidence it's needed at pilot scale).
- **Consequence:** a documented, probe-observable recovery story.

### D6 — Honest synthetic-load caveat
- **Decision:** the readiness report carries an explicit "validated on synthetic load; real 8 kHz drift pending a live pilot" caveat.
- **Rationale:** Rule 3 honesty; no real pilot data exists here.
- **Alternatives considered:** an unqualified production-ready claim (rejected — dishonest).
- **Consequence:** the V1 claim is scoped truthfully.

## Drawbacks & Risks

| Drawback / Risk | Severity | Mitigation | Owner |
|---|---|---|---|
| R1 — real 8 kHz drift degrades the synthetic-load numbers | High | D6 caveat in the report; the harness re-runnable on real data when it arrives | tomas |
| R2 — the harness hard-coding a PASS (acceptance theatre) | High | D3 re-run/fresh-artifact sourcing; a test asserts a FAIL when a criterion is unmet | tomas |
| R3 — engagement rate misread as a production signal | Medium | D4/D6 framing: instrumented proxy, production value pending a pilot | kael |
| R4 — readiness probe false-green under a saturated queue | Medium | probe checks qsize against maxsize; a concurrent test drives the queue near saturation | kael |

## Unresolved Questions

- Q1 — Exact readiness saturation threshold (queue depth %)? M8 uses ≥ 90% of maxsize = degraded; tuned post-pilot.
- Q2 — The V1 alert-engagement target N/day? Set post-pilot with real supervisors; M8 ships the metric, not the target.

## Dependency Graph

```
Phase 0 (health/readiness endpoints + value objects)  ──▶  Phase 1 (engagement metric + endpoint)
                                                                  ▼
                                             Phase 2 (V1-acceptance harness + runbook)
                                                                  ▼
                        Final: Integration Validation (run the harness → readiness report; a criterion FAIL fails the verdict)
```

---

## Phase 0: Health & readiness

**Objective:** the monitor self-reports liveness + readiness.

### T0.1 — /health + /ready endpoints

#### Objective
A liveness `/health` and a readiness `/ready` probing session LISTENING + queue not saturated + DB reachable.

#### Why this step
1. **What:** `domain/health.py` (`HealthReport`); `application/health_service.py` (probe); `interface/app.py` routes.
2. **Why now:** operational self-monitoring (blueprint D1; risk R4).

#### Files to edit
```
src/talkex/monitoring/domain/health.py (NEW) — HealthReport
src/talkex/monitoring/application/health_service.py (NEW) — readiness probe
src/talkex/monitoring/interface/app.py — /health + /ready routes
tests/integration/monitoring/test_health_api.py (NEW)
tests/unit/monitoring/test_health_service.py (NEW)
```

#### TDD
```
RED (unit):        test_ready_false_when_queue_saturated — a probe over a near-full channel reports not-ready
RED (integration): test_health_and_ready_endpoints — GET /health = 200; GET /ready = 200 with a running session
GREEN: implement the probe + routes
VERIFY: pytest tests/unit/monitoring/test_health_service.py tests/integration/monitoring/test_health_api.py -x
```

#### Concurrency tests

A concurrent test fills the bounded channel toward saturation while probing `/ready`; asserts the readiness flips to not-ready (503) — the backpressure signal surfaces (the M0 channel is the parallel-load lever).

#### Acceptance Criteria
- [ ] `/health` returns 200 (liveness)
- [ ] `/ready` returns 200 when the session is LISTENING + DB reachable; 503 when the queue is saturated
- [ ] no domain layer imports FastAPI (boundary preserved)

#### DoD
- [ ] health/readiness endpoints green

---

## Phase 1: Engagement metric

**Objective:** `acted_on_rate` is queryable.

### T1.1 — engagement metric + endpoint

#### Objective
An `EngagementMetric` (`acted_on_rate` = labels/window ÷ alerts/window) + `GET /dashboard/engagement`.

#### Why this step
1. **What:** `domain/health.py` `EngagementMetric`; `infrastructure/engagement_repo.py`; a route.
2. **Why now:** the north-star proxy (blueprint D4; DoD #2).

#### Files to edit
```
src/talkex/monitoring/domain/health.py — EngagementMetric
src/talkex/monitoring/infrastructure/engagement_repo.py (NEW)
src/talkex/monitoring/interface/app.py — /dashboard/engagement route
tests/integration/monitoring/test_engagement.py (NEW)
```

#### TDD
```
RED (integration): test_acted_on_rate — seed A alerts + L labels in a window; assert acted_on_rate == L/A
GREEN: implement the repo query + route
VERIFY: pytest tests/integration/monitoring/test_engagement.py -x
```

#### Concurrency tests

(none — single-threaded) — the engagement query is a single pooled read.

#### Acceptance Criteria
- [ ] `acted_on_rate` == labels/window ÷ alerts/window (asserted on seeded counts)
- [ ] the metric is framed as a proxy (docstring notes production value pending a pilot)

#### DoD
- [ ] engagement metric green

---

## Phase 2: V1-acceptance harness + runbook

**Objective:** a re-running readiness gate + an operational runbook.

### T2.1 — v1_acceptance harness + runbook

#### Objective
A harness that RE-RUNS each V1 criterion and emits a readiness report; an operational runbook.

#### Why this step
1. **What:** `experiments/scripts/v1_acceptance.py` (NEW); `docs/runbook.md` (NEW).
2. **Why now:** DoD #1 (all criteria pass) + #3 (runbook); blueprint D3/D5/D6.

#### Files to edit
```
experiments/scripts/v1_acceptance.py (NEW) — run criteria → readiness report JSON
docs/runbook.md (NEW) — operational runbook + recovery posture
tests/unit/monitoring/test_v1_acceptance.py (NEW)
```

#### TDD
```
RED: test_report_fails_when_a_criterion_fails — feed the report builder a failing criterion; the overall verdict is FAIL
     test_report_passes_when_all_meet — all criteria meet → PASS + the synthetic-load caveat present
GREEN: implement the report builder (criteria → per-criterion PASS/FAIL → verdict + caveat)
VERIFY: pytest tests/unit/monitoring/test_v1_acceptance.py -x
```

#### Concurrency tests

(none — single-threaded) — the harness is an offline batch aggregator.

#### Acceptance Criteria
- [ ] a single failing criterion makes the overall verdict FAIL (no hard-coded PASS — risk R2)
- [ ] the report carries the synthetic-load / real-drift caveat (D6)
- [ ] `docs/runbook.md` documents startup, health probes, and the recovery posture

#### DoD
- [ ] acceptance harness + runbook green

---

## Coverage Matrix

| # | Gap / Requirement (DoD) | Task(s) | Resolution |
|---|---|---|---|
| 1 | Health self-monitoring (DoD #3) | T0.1 | /health + /ready |
| 2 | Readiness surfaces degradation (risk R4) | T0.1 | /ready 503 on saturation |
| 3 | Alert-engagement tracked (DoD #2) | T1.1 | acted_on_rate + endpoint |
| 4 | All V1 criteria pass, re-run (DoD #1) | T2.1, T3.1 | v1_acceptance harness |
| 5 | Operational runbook + recovery (DoD #3) | T2.1 | docs/runbook.md |
| 6 | Honest synthetic-load caveat (D6) | T2.1, T3.1 | caveat in the report |

**Coverage: 6/6 gaps covered (100%)**

## Global Definition of Done

- [ ] `ruff format --check . && ruff check . && mypy src/ tests/` clean
- [ ] `pytest tests/unit -x && pytest tests/integration -x` green
- [ ] The V1-acceptance harness runs and emits a readiness report — a criterion FAIL fails the verdict (proven)
- [ ] `/ready` flips to 503 under queue saturation — proven by a concurrent test
- [ ] File-size ≤ 500 LoC per file
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] The readiness report carries the synthetic-load caveat (no unqualified production claim)

## Failure scenarios (external I/O)

| Dependency | Failure mode | How the test reproduces it | Expected behavior |
|---|---|---|---|
| TimescaleDB (readiness probe) | DB unreachable | probe with a closed pool | `/ready` returns 503 (not-ready), not a crash |
| TurnChannel (readiness) | queue saturated | fill the channel to maxsize | `/ready` returns 503 (degraded) |
| V1-acceptance (criterion source) | a metrics artifact missing | run the harness with a missing artifact | that criterion is FAIL (missing evidence ≠ silent PASS) |

## Final Phase: Integration Validation (MANDATORY)

**Objective:** run the acceptance harness and prove the readiness gate.

### T3.1 — run the V1-acceptance harness

#### Objective
Run the harness against the real tree + fresh benchmarks; emit the readiness report; assert a criterion FAIL fails the verdict.

#### Why this step
1. **What:** run `v1_acceptance.py`; persist the readiness report JSON.
2. **Why now:** DoD #1 is proven only by running the gate (blueprint D3).

#### Files to edit
```
tests/unit/monitoring/test_v1_acceptance.py — add the missing-artifact FAIL assertion
experiments/results/v1_readiness.json — the produced report (evidence)
```

#### TDD
```
RED: test_missing_artifact_is_fail — a criterion whose evidence artifact is absent yields FAIL, not PASS
GREEN: (behavior implemented in Phase 2)
VERIFY: python experiments/scripts/v1_acceptance.py && pytest tests/unit/monitoring/test_v1_acceptance.py -x
```

#### Concurrency tests

(none — single-threaded) — the harness is an offline aggregator.

#### Acceptance Criteria
- [ ] the harness emits a readiness report with per-criterion PASS/FAIL
- [ ] a missing/failing criterion yields an overall FAIL
- [ ] the report carries the synthetic-load caveat

#### DoD
- [ ] readiness report produced; the gate is honest (FAIL on unmet criteria)
