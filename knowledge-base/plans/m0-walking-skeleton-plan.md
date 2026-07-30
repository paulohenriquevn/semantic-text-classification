# Plan: M0 Walking Skeleton — Real-Time Attendance Monitoring

> **Version 1.0** — Build the thinnest end-to-end slice of the monitoring platform: a FastAPI streaming
> ingest endpoint feeds a bounded queue (backpressure) → a session state machine drains it → the existing
> `talkex` segmentation + context builder + one DSL rule produce an evidence-backed alert → a per-Turn
> insert lands in a real TimescaleDB hypertable → a commit-time LISTEN/NOTIFY push drives a minimal
> supervisor SSE page. Proves the ADR-005 architecture end-to-end with two-tier tests.

## Goal

> "Enable a call-center supervisor to see a live evidence-backed alert for an ongoing conversation so that
> a critical turn is surfaced within seconds of ingestion, measured by the integration test
> `test_ingest_to_supervisor_alert_e2e` passing (segment posted → Turn in hypertable → DSL rule fires →
> alert delivered on the SSE stream)."

## Context

M0 is the walking skeleton of `ROADMAP.md § M0`. The architecture is locked by the discovery blueprint
`knowledge-base/discoveries/blueprints/m0-walking-skeleton-realtime-monitoring-blueprint.md` (discover-confidence
SHIPPABLE 99.4) and the storage decision `docs/adr/ADR-005-online-storage-realtime-monitoring.md`. The domain
primitives already exist in `talkex` (segmentation, context, rules, models); what is undesigned is the
streaming ingestion boundary, the live supervisor push, and the integration-test harness against a real
Timescale. This plan builds exactly that thin slice — nothing from M1+ (no pgvector/BM25, no sentiment ML,
no retention/continuous-aggregates).

## Baseline Context (deep review of current state)

### Files that will be touched

| File | LoC today | Last commit (sha + date) | Why it exists today | Invariants to preserve |
|---|---|---|---|---|
| `src/talkex/segmentation/segmenter.py` | 123 | `b0ac1e1` (2026-03-10) | `TurnSegmenter.segment(transcript, config) -> list[Turn]` (`:39`) | Reused read-only; do not modify its signature |
| `src/talkex/context/builder.py` | 87 | `b0ac1e1` (2026-03-10) | `SlidingWindowBuilder.build(conversation, turns, config) -> list[ContextWindow]` (`:38`) | Reused read-only |
| `src/talkex/models/turn.py` | 100 | `b0ac1e1` (2026-03-10) | `Turn` frozen model (`:36`): turn_id, conversation_id, speaker, raw_text, offsets, metadata | Reused read-only; frozen contract |
| `src/talkex/models/context_window.py` | 124 | `b0ac1e1` (2026-03-10) | `ContextWindow` (ADR-004 structural fields) | Reused read-only |
| `src/talkex/models/rule_execution.py` | 139 | `b0ac1e1` (2026-03-10) | `EvidenceItem` (TypedDict) + `RuleExecution` | Reused for alert evidence shape |
| `src/talkex/rules/parser.py` | 1404 | `dc1970b` (2026-03-10) | `parse_dsl` / `parse_rule_block` | Reused read-only |
| `src/talkex/rules/evaluator.py` | 851 | `dc1970b` (2026-03-10) | DSL AST evaluation producing `PredicateResult` | Reused read-only |
| `src/talkex/monitoring/__init__.py` (NEW) | 0 | — | new package root | — |
| `src/talkex/monitoring/config.py` (NEW) | 0 | — | monitoring config (queue size, DSN, topic) | — |
| `src/talkex/monitoring/domain/models.py` (NEW) | 0 | — | `Alert`, `SessionState`, domain events | — |
| `src/talkex/monitoring/domain/ports.py` (NEW) | 0 | — | `TurnRepository`/`AlertRepository`/`AlertBroadcaster` Protocols | — |
| `src/talkex/monitoring/domain/channel.py` (NEW) | 0 | — | `TurnChannel` bounded queue (Chan semantics) | — |
| `src/talkex/monitoring/application/session.py` (NEW) | 0 | — | session state machine + drain | — |
| `src/talkex/monitoring/application/orchestrator.py` (NEW) | 0 | — | consumer: segment→window→rule→alert | — |
| `src/talkex/monitoring/infrastructure/timescale_repo.py` (NEW) | 0 | — | psycopg per-Turn/Alert hypertable insert | — |
| `src/talkex/monitoring/infrastructure/notify_broadcaster.py` (NEW) | 0 | — | Postgres LISTEN/NOTIFY adapter | — |
| `src/talkex/monitoring/interface/ingest_api.py` (NEW) | 0 | — | FastAPI streaming ingest route | — |
| `src/talkex/monitoring/interface/supervisor_sse.py` (NEW) | 0 | — | SSE endpoint + minimal supervisor HTML | — |
| `deploy/monitoring/docker-compose.yml` (NEW) | 0 | — | timescale/timescaledb-ha:pg16 on :5433 | — |
| `deploy/monitoring/migrations/0001_m0_hypertable.sql` (NEW) | 0 | — | turns + alerts hypertables | — |
| `tests/unit/monitoring/test_channel.py` (NEW) | 0 | — | RED tests for bounded channel | — |
| `tests/unit/monitoring/test_session.py` (NEW) | 0 | — | RED tests for session state machine | — |
| `tests/unit/monitoring/test_orchestrator.py` (NEW) | 0 | — | RED tests for orchestrator over fakes | — |
| `tests/integration/monitoring/test_timescale_repo.py` (NEW) | 0 | — | real-Timescale repo tests | — |
| `tests/integration/monitoring/test_notify_broadcaster.py` (NEW) | 0 | — | real LISTEN/NOTIFY test | — |
| `tests/integration/monitoring/test_ingest_e2e.py` (NEW) | 0 | — | full-chain DoD test | — |
| `pyproject.toml` | (existing) | — | add `[monitoring]` extra: fastapi, uvicorn, psycopg[binary], sse-starlette | Keep existing extras intact |

### Current callers / dependents

- **Symbol:** `TurnSegmenter.segment` (`src/talkex/segmentation/segmenter.py:39`) — Callers (production): `src/talkex/pipeline/pipeline.py`; Callers (tests): `tests/unit/test_segmentation*.py`. External public API: no. M0 adds a NEW caller (`orchestrator.py`), does not modify the symbol.
- **Symbol:** `SlidingWindowBuilder.build` (`src/talkex/context/builder.py:38`) — Callers (production): `src/talkex/pipeline/pipeline.py`; Callers (tests): `tests/unit/test_context*.py`. M0 adds a NEW caller only.
- **Symbol:** `parse_dsl` / evaluator — Callers: `src/talkex/rules/*`, `tests/unit/test_rule*`. M0 reuses read-only.
- New monitoring package has no existing callers (first-of-its-kind).

### Domain glossary

- **Turn** — a single speaker utterance; finest unit for the pipeline (`models/turn.py`).
- **ContextWindow** — sliding window of N adjacent turns; the unit a rule evaluates (`models/context_window.py`).
- **DSL rule** — a `RULE...WHEN...THEN` expression compiled to an AST and evaluated with evidence (`src/talkex/rules/`).
- **EvidenceItem** — TypedDict carrying predicate_type/score/threshold/matched_text for a rule match (`models/rule_execution.py`).
- **Alert** (NEW, M0) — a domain event emitted when a critical rule fires on a window, carrying its evidence.
- **Hypertable** — a TimescaleDB time-partitioned table (ADR-005).

### Architecture boundaries affected

Per `.claude/rules/architecture.md § 1–2`: introduces a new feature package `talkex.monitoring` with
package-by-layer sub-modules (interface → application → domain ← infrastructure). New DIP ports live in
`domain/ports.py`; Timescale and LISTEN/NOTIFY are infrastructure adapters injected at the composition root
(`interface`). The domain must not import FastAPI, psycopg, or asyncio transport concretes.

## Prior Art & Related Work

- **Internal blueprint** — `knowledge-base/discoveries/blueprints/m0-walking-skeleton-realtime-monitoring-blueprint.md`
  §"Coverage Corner 4 — Techniques" (bounded channel, commit-time broadcast) and §"ADRs" D1–D6.
- **Reference — livekit-agents** — bounded channel `knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/utils/aio/channel.py:49,:71`; session state machine `.../voice/agent_session.py:555,:1013`; Toxiproxy fault tests `.../tests/toxic_proxy.py:89`.
- **Reference — chatwoot** — commit-time domain-event → async broadcast `knowledge-base/references/chatwoot/app/listeners/action_cable_listener.rb:222`, `app/models/conversation.rb:134`; channel contract spec `spec/channels/room_channel_spec.rb:12`.
- **Reference — ai-powered-call-center-intelligence** — persistence shape `analytics/duckdb_loader.py:17`; the online-LLM anti-pattern rejected `backend/main.py:49`.
- **ADR** — `docs/adr/ADR-005-online-storage-realtime-monitoring.md` (Postgres/TimescaleDB single spine).

## Objective

- [ ] Bounded `TurnChannel` with awaiting `put` (backpressure) + hermetic unit tests
- [ ] Session state machine `initializing→listening→closing` with two-phase drain + unit tests
- [ ] Orchestrator wiring reused `segment`→`build`→one DSL rule→`Alert` over fakes + unit tests
- [ ] Timescale repository: per-Turn + per-Alert hypertable insert, integration-tested against real Timescale
- [ ] FastAPI streaming ingest endpoint enqueueing Turns (never inline work)
- [ ] LISTEN/NOTIFY broadcaster + minimal supervisor SSE page, integration-tested
- [ ] Full-chain DoD integration test green (ingest → hypertable → rule → alert on SSE)

## ADRs

### D1 — Producer/Consumer ingest with a bounded channel
- **Decision:** the ingest route only enqueues a `Turn` onto a bounded `TurnChannel(maxsize>0)` whose `put` awaits when full; a separate consumer task drains it. No segmentation/rule/DB work inline in the request handler.
- **Rationale:** copies the livekit `Chan` backpressure primitive (`channel.py:71`); rejects the ai-call-center inline-work anti-pattern (`main.py:49`). Satisfies `architecture.md § 1` (interface stays thin).
- **Alternatives considered:** unbounded `asyncio.Queue` (rejected — a fast producer OOMs ahead of the slower consumer); do work inline in the handler (rejected — couples ingest latency to rule/DB latency, the exact anti-pattern).
- **Consequences:** backpressure is explicit; ingest latency is decoupled from processing; adds a consumer-lifecycle concern (handled by D2).

### D2 — Explicit session State pattern with two-phase drain
- **Decision:** model the live session as `initializing → listening → closing` with a single closing guard and `drain()`-before-`aclose()`.
- **Rationale:** mirrors `agent_session.py:555,:1013,:1049`; guarantees an in-flight window/alert finishes before teardown.
- **Alternatives considered:** implicit lifecycle via task cancellation only (rejected — drops in-flight turns, non-deterministic shutdown).
- **Consequences:** deterministic shutdown; one extra state field + guard.

### D3 — Commit-time live push via Postgres LISTEN/NOTIFY
- **Decision:** the alert broadcast fires only after the DB transaction commits, via Postgres `NOTIFY`; the SSE endpoint `LISTEN`s and forwards to the supervisor.
- **Rationale:** chatwoot broadcasts on `after_create_commit` (`conversation.rb:134`); LISTEN/NOTIFY keeps M0 dependency-light and aligns with the ADR-005 single-spine (no Redis in M0).
- **Alternatives considered:** Redis pub/sub (rejected for M0 — extra infra dependency; kept as scale escape hatch); broadcast before commit (rejected — supervisor could see an alert that rolls back).
- **Consequences:** zero extra infra; LISTEN/NOTIFY payload is size-limited (8000 bytes) — payload carries an alert id, the SSE handler reads full evidence from the DB (documented risk R2).

### D4 — Hexagonal layering with DIP ports
- **Decision:** `domain/ports.py` declares `TurnRepository`, `AlertRepository`, `AlertBroadcaster` Protocols; infrastructure implements them; interface wires concretes.
- **Rationale:** `architecture.md § 2` (DIP at boundaries); enables hermetic unit tests with fakes.
- **Alternatives considered:** call psycopg directly from the orchestrator (rejected — couples domain to the driver, untestable without a DB).
- **Consequences:** clean unit tests; a small amount of port/adapter boilerplate.

### D5 — Per-Turn streaming insert into a Timescale hypertable with JSONB evidence
- **Decision:** `turns` and `alerts` are hypertables; each Turn/Alert is inserted as it is produced; `alerts.evidence` is JSONB; both carry `created_at timestamptz default now()`.
- **Rationale:** keeps the useful part of `duckdb_loader.py:17` (evidence-JSON + created_at), rejects its file-batch model; ADR-005 storage spine.
- **Alternatives considered:** plain Postgres table (rejected — M0 must exercise real hypertable behavior per ADR-005); batch insert (rejected — real-time requires per-Turn).
- **Consequences:** M0 exercises `create_hypertable`; chunking/retention deferred to M1 (honest gap G1).

### D6 — Two-tier tests (hermetic unit + real-Timescale integration)
- **Decision:** unit tests over fakes assert the ordered event/state stream + DSL evaluation; an integration tier runs pytest against a docker-compose Timescale.
- **Rationale:** livekit hermetic session tests (`test_agent_session.py:51`) + chatwoot real-DB channel spec (`room_channel_spec.rb:12`); `.claude/rules/testing.md § 2` (integration tests use real deps).
- **Alternatives considered:** sqlite/in-memory for the repo (rejected — cannot test hypertable behavior).
- **Consequences:** integration tests require Docker; gated behind a marker so unit runs need no services.

## Drawbacks & Risks

| Drawback / Risk | Severity | Mitigation | Owner |
|---|---|---|---|
| R1 — Integration tests require a running Timescale container (slower, env-dependent) | Medium | `@pytest.mark.integration` marker; unit tier needs no services; docker-compose one-liner in DoD | dev |
| R2 — LISTEN/NOTIFY payload is capped at 8000 bytes; full evidence may exceed it | Medium | NOTIFY carries `alert_id` only; SSE handler reads full evidence from the DB by id | dev |
| R3 — ingest×query contention on one Postgres under load | Medium | M0 validates functionally; load/p95 + replica split deferred to M1 pilot (ADR-005) | dev |
| R4 — new `[monitoring]` extra pulls fastapi/uvicorn/psycopg into the env | Low | optional extra; core `talkex` unaffected; pinned ranges | dev |

## Unresolved Questions

- Q1 — Does `timescale/timescaledb-ha:pg16` bundle the `timescaledb` extension enabled by default, or does the migration need `CREATE EXTENSION IF NOT EXISTS timescaledb`? (resolve in T0.1 by asserting `create_hypertable` succeeds in the integration test).
- Q2 — SSE vs WebSocket for the supervisor page: M0 picks SSE (simpler, one-way). Revisit if bidirectional control is needed (out of M0 scope).
- Q3 — Exact DSL rule for M0's "critical" alert: a `contains_any(["cancelar","cancelamento"])` rule is sufficient to prove the path; final rule set is M2/M3.

## Dependency Graph

```
Phase 0 (infra: compose + migration + package skeleton + extra)
   │
   ▼
Phase 1 (domain: ports + channel + Alert model)  ──▶ Phase 2 (application: session + orchestrator, fakes)
   │                                                        │
   ▼                                                        │
Phase 3 (infra: timescale_repo + notify_broadcaster, integration) 
   │                                                        │
   └───────────────────────────┬────────────────────────────┘
                               ▼
                     Phase 4 (interface: ingest_api + supervisor_sse)
                               ▼
                 Final Phase: Integration Validation (4 DoDs end-to-end)
```

Phase 1 and (the fake-based) Phase 2 can proceed in parallel with Phase 3's schema work; Phase 4 blocks on 1–3.

---

## Phase 0: Infra scaffold

**Objective:** stand up a real TimescaleDB for dev/tests and create the empty layered package + optional extra.

### T0.1 — docker-compose Timescale + hypertable migration

#### Objective
Boot `timescale/timescaledb-ha:pg16` on port 5433 and create `turns`/`alerts` hypertables.

#### Why this step
1. **What:** add `deploy/monitoring/docker-compose.yml` + `deploy/monitoring/migrations/0001_m0_hypertable.sql`.
2. **Why now:** every downstream integration test (D5, D6) needs a real hypertable; port 5433 because 5432 is occupied locally (verified `ss -ltnp`). Cites ADR-005 + blueprint D5.

#### Evidence
Local `docker ps` shows no Timescale container and 5432 in use (verified this session). ADR-005 mandates Timescale + pgvector.

#### Files to edit
```
deploy/monitoring/docker-compose.yml (NEW) — timescale service, port 5433, healthcheck
deploy/monitoring/migrations/0001_m0_hypertable.sql (NEW) — CREATE EXTENSION timescaledb; turns/alerts tables; create_hypertable(...)
```

#### Deep Dives
- `turns(turn_id text pk, conversation_id text, speaker text, raw_text text, normalized_text text, start_offset int, end_offset int, metadata jsonb, created_at timestamptz default now())`, `create_hypertable('turns','created_at')`.
- `alerts(alert_id text pk, conversation_id text, window_id text, rule_name text, evidence jsonb, created_at timestamptz default now())`, `create_hypertable('alerts','created_at')`.
- Invariant: migration is idempotent (`IF NOT EXISTS`).

#### TDD
```
RED:     (integration) test_migration_creates_hypertables — asserts turns/alerts are hypertables (query timescaledb_information.hypertables)
GREEN:   write the SQL + compose; bring up; apply migration
REFACTOR: None expected
VERIFY:  docker compose -f deploy/monitoring/docker-compose.yml up -d && psql -h localhost -p 5433 -f migrations/0001_m0_hypertable.sql
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `test_migration_creates_hypertables` asserts `timescaledb_information.hypertables` returns rows for `turns` and `alerts` (resolves Q1)
- [ ] `docker compose ps` reports the timescale service `healthy`
- [ ] `psql -h localhost -p 5433 -c "\dt"` lists `turns` and `alerts`

#### DoD
- [ ] `docker compose -f deploy/monitoring/docker-compose.yml up -d` yields a healthy container and the migration applies without error

### T0.2 — layered package skeleton + `[monitoring]` extra

#### Objective
Create empty `talkex.monitoring` layered package and add the optional dependency extra.

#### Why this step
1. **What:** create `__init__.py` files for `monitoring/{,domain,application,infrastructure,interface}` + `config.py`; add `[project.optional-dependencies] monitoring = [fastapi, uvicorn, psycopg[binary], sse-starlette]` to `pyproject.toml`.
2. **Why now:** establishes the DIP layering (D4) before code; keeps core `talkex` dependency-free (R4).

#### Files to edit
```
src/talkex/monitoring/__init__.py (NEW)
src/talkex/monitoring/{domain,application,infrastructure,interface}/__init__.py (NEW)
src/talkex/monitoring/config.py (NEW) — MonitoringConfig (dsn, queue_maxsize, notify_channel)
pyproject.toml — add [monitoring] extra
```

#### TDD
```
RED:     test_monitoring_config_defaults — asserts MonitoringConfig defaults (queue_maxsize>0, notify_channel str)
GREEN:   implement MonitoringConfig (frozen pydantic)
VERIFY:  pytest tests/unit/monitoring/test_config.py -q
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `pip install -e ".[monitoring]"` exits 0 and installs fastapi/uvicorn/psycopg
- [ ] `python -c "import talkex.monitoring"` exits 0
- [ ] `ruff check src/talkex/monitoring` and `mypy src/talkex/monitoring` report zero warnings/errors

#### DoD
- [ ] `pytest tests/unit/monitoring/test_config.py` passes and the import command above exits 0

---

## Phase 1: Domain (ports, bounded channel, Alert)

**Objective:** the pure-domain core with no I/O.

### T1.1 — bounded `TurnChannel` (backpressure primitive)

#### Objective
An async bounded channel whose `put` awaits when full and that closes deterministically.

#### Why this step
1. **What:** `TurnChannel` wrapping `asyncio.Queue(maxsize)` with `put`/`get`/`close`/async-iteration and a `ChannelClosed` error.
2. **Why now:** the backpressure lever (D1); mirrors `channel.py:49,:71,:174`.

#### Deep Dives
- Fields: `_queue: asyncio.Queue[Turn]`, `_closed: bool`. `put` awaits `_queue.put` (blocks when full); `close()` sets closed + sentinels getters; `__aiter__/__anext__` raise `StopAsyncIteration` on close.
- Invariant: after `close()`, `put` raises `ChannelClosed`; pending `get` drains remaining then stops.

#### TDD
```
RED:     test_put_blocks_when_full — with maxsize=1, second put does not complete until a get (assert via asyncio.wait_for timeout)
RED:     test_close_stops_iteration — async-for terminates after close
RED:     test_put_after_close_raises — ChannelClosed
GREEN:   implement TurnChannel
REFACTOR: None expected
VERIFY:  pytest tests/unit/monitoring/test_channel.py -q
```

#### Concurrency tests
```
Atomic-throughput invariant: N producers put M turns each into maxsize=4 channel; one consumer drains;
assert total consumed == N*M and no loss (asyncio.gather with a barrier). Cancellation: cancel the
consumer task; assert put unblocks / channel closes cleanly.
```

#### Acceptance Criteria
- [ ] `put` awaits when full (backpressure proven, not just "has a maxsize")
- [ ] no lost/duplicated turns under N producers
- [ ] Pass: ruff+mypy clean; file ≤ 500 LoC

#### DoD
- [ ] `test_channel.py` green incl. concurrency invariant

### T1.2 — ports + `Alert` domain model + domain events

#### Objective
DIP Protocols and the `Alert` value object emitted on a rule match.

#### Why this step
1. **What:** `domain/ports.py` (`TurnRepository`, `AlertRepository`, `AlertBroadcaster` Protocols) and `domain/models.py` (`Alert` frozen model + `SessionState` enum + `TurnIngested`/`AlertRaised` events).
2. **Why now:** D4 — the domain declares contracts; adapters implement later. `Alert.evidence` reuses `EvidenceItem` (`models/rule_execution.py`).

#### TDD
```
RED:     test_alert_is_frozen_and_carries_evidence — Alert(evidence=[EvidenceItem]) immutable
RED:     test_session_state_transitions_enum — INITIALIZING/LISTENING/CLOSING present
GREEN:   implement models + Protocols
VERIFY:  pytest tests/unit/monitoring/test_domain.py -q
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] Protocols are `typing.Protocol` (structural, no concrete import)
- [ ] `Alert` frozen; evidence typed as `list[EvidenceItem]`
- [ ] Pass: ruff+mypy clean

#### DoD
- [ ] domain tests green; no infra import in domain (verify: `grep -L psycopg src/talkex/monitoring/domain/*`)

---

## Phase 2: Application (session state machine + orchestrator)

**Objective:** orchestrate the reused pipeline over the channel, with fakes.

### T2.1 — session state machine with two-phase drain

#### Objective
`MonitoringSession` cycling `initializing→listening→closing`, draining the channel before close.

#### Why this step
1. **What:** `application/session.py` — owns a `TurnChannel`, a consumer task, and `start()`/`drain()`/`aclose()`.
2. **Why now:** D2; mirrors `agent_session.py:1013,:1049`.

#### TDD
```
RED:     test_session_state_sequence — start→listening; aclose→closing; assert ordered state events
RED:     test_drain_processes_inflight_before_close — enqueue 3 turns then aclose; assert all 3 processed
GREEN:   implement session
VERIFY:  pytest tests/unit/monitoring/test_session.py -q
```

#### Concurrency tests
```
Cancellation propagation: aclose() cancels the consumer only AFTER drain; assert no turn dropped and the
task is finished (asyncio.Task.done()).
```

#### Acceptance Criteria
- [ ] ordered state stream asserted (not internal fields)
- [ ] in-flight turns fully processed before close
- [ ] Pass: ruff+mypy clean

#### DoD
- [ ] `test_session.py` green

### T2.2 — orchestrator: segment → window → rule → alert

#### Objective
Consume a Turn, reuse `TurnSegmenter`/`SlidingWindowBuilder`, evaluate one DSL rule, emit `Alert` on match, persist via ports.

#### Why this step
1. **What:** `application/orchestrator.py` — injected `TurnRepository`/`AlertRepository`/`AlertBroadcaster` (fakes in unit tests); reuses `segment` (`segmenter.py:39`), `build` (`builder.py:38`), `parse_dsl`+evaluator.
2. **Why now:** the core "decide" step (D1/D4); DSL is the online decision (not an LLM — rejecting `main.py:49`).

#### Pseudo-code / Signatures
```pseudocode
class Orchestrator:
  def __init__(self, turn_repo, alert_repo, broadcaster, rule_ast): ...
  async def handle(self, turn: Turn) -> None:
    await self.turn_repo.save(turn)                       # D5 per-Turn insert
    windows = build(conversation_of(turn), turns_so_far, cfg)   # reuse builder.py:38
    for w in windows:
      result = evaluate(self.rule_ast, w)                 # reuse rules evaluator
      if result.matched:
        alert = Alert(window_id=w.window_id, evidence=result.to_evidence_items(), ...)
        await self.alert_repo.save(alert)                 # commit
        await self.broadcaster.notify(alert.alert_id)     # D3 after commit
# Example: turn "quero cancelar" → rule contains_any(["cancelar"]) matches → Alert with matched_text evidence
```

#### TDD
```
RED:     test_matching_turn_raises_alert_with_evidence — fake repos capture Alert; evidence non-empty
RED:     test_non_matching_turn_raises_no_alert
RED:     test_turn_is_persisted_before_alert — ordering: turn_repo.save called before alert_repo.save
GREEN:   implement orchestrator
VERIFY:  pytest tests/unit/monitoring/test_orchestrator.py -q
```

#### Concurrency tests
```
(none — single-threaded)
```
(The channel/session own concurrency, tested in T1.1/T2.1; `handle()` processes one turn at a time.)

#### Acceptance Criteria
- [ ] alert carries real `EvidenceItem`s from the DSL match
- [ ] broadcaster called only after alert persisted (D3 commit-then-notify)
- [ ] Pass: ruff+mypy clean

#### DoD
- [ ] `test_orchestrator.py` green over fakes

---

## Phase 3: Infrastructure (Timescale repo + LISTEN/NOTIFY)

**Objective:** real adapters, integration-tested against the container from Phase 0.

### T3.1 — Timescale repository (per-Turn / per-Alert insert)

#### Objective
`TimescaleTurnRepository` / `TimescaleAlertRepository` implementing the ports via psycopg.

#### Why this step
1. **What:** `infrastructure/timescale_repo.py` — async psycopg inserts into the hypertables (D5).
2. **Why now:** the persistence adapter; integration test proves DoD-1 (segment → Turn in hypertable).

#### TDD
```
RED (integration): test_save_turn_lands_in_hypertable — save(turn); SELECT by turn_id returns it with created_at
RED (integration): test_save_alert_evidence_roundtrips_jsonb — evidence JSONB read back equals input
GREEN:   implement repo
VERIFY:  pytest tests/integration/monitoring/test_timescale_repo.py -m integration -q
```

#### Failure scenarios
Covered in the `## Failure scenarios` section (connection reset, insert timeout).

#### Concurrency tests
```
(none — single-threaded)
```
(Each `save()` uses its own connection/transaction; no shared mutable state in the repo.)

#### Acceptance Criteria
- [ ] `test_save_turn_lands_in_hypertable` asserts a SELECT by turn_id returns the row with a non-null `created_at`
- [ ] `test_save_alert_evidence_roundtrips_jsonb` asserts the read-back JSONB equals the input evidence
- [ ] `ruff check src/talkex/monitoring/infrastructure/timescale_repo.py` reports zero warnings and `mypy` reports zero errors

#### DoD
- [ ] integration repo tests green with the container up

### T3.2 — LISTEN/NOTIFY broadcaster

#### Objective
`NotifyAlertBroadcaster.notify(alert_id)` issues `NOTIFY <channel>, alert_id`; an async `listen()` yields ids.

#### Why this step
1. **What:** `infrastructure/notify_broadcaster.py` — psycopg async NOTIFY + LISTEN (D3).
2. **Why now:** the live-push transport; R2 payload = id only.

#### TDD
```
RED (integration): test_notify_delivers_alert_id_to_listener — listen() receives the id that notify() sent
GREEN:   implement broadcaster
VERIFY:  pytest tests/integration/monitoring/test_notify_broadcaster.py -m integration -q
```

#### Concurrency tests
```
Happens-before: start listener (barrier), then notify from another connection; assert the id is observed
within a timeout (asyncio.wait_for).
```

#### Acceptance Criteria
- [ ] id delivered to a separate LISTEN connection
- [ ] Pass: ruff+mypy clean

#### DoD
- [ ] integration notify test green

---

## Phase 4: Interface (ingest API + supervisor SSE)

**Objective:** the thin edges wiring concretes.

### T4.1 — FastAPI streaming ingest endpoint

#### Objective
`POST /ingest` enqueues a Turn onto the session channel and returns immediately.

#### Why this step
1. **What:** `interface/ingest_api.py` — FastAPI route; only `await channel.put(turn)`; composition root wires session + repo + broadcaster.
2. **Why now:** D1 — interface stays thin; no inline work (rejecting `main.py:49`).

#### TDD
```
RED (integration): test_ingest_enqueues_and_returns_fast — POST returns 202 without waiting on processing
GREEN:   implement route + app factory
VERIFY:  pytest tests/integration/monitoring/test_ingest_e2e.py::test_ingest_enqueues -m integration -q
```

#### Concurrency tests
```
(none — single-threaded)
```
(The route only `await channel.put`; the channel's own concurrency is tested in T1.1.)

#### Acceptance Criteria
- [ ] `test_ingest_enqueues_and_returns_fast` asserts the POST returns 202 and no `turn_repo.save` was called synchronously in the handler
- [ ] `ruff check` and `mypy` report zero warnings/errors on `interface/ingest_api.py`

#### DoD
- [ ] `pytest tests/integration/monitoring/test_ingest_e2e.py::test_ingest_enqueues -m integration` passes

### T4.2 — supervisor SSE page

#### Objective
`GET /supervisor/stream` (SSE) LISTENs and pushes alerts; `GET /supervisor` serves a minimal HTML page.

#### Why this step
1. **What:** `interface/supervisor_sse.py` — SSE endpoint reading the broadcaster + reading full evidence from the repo by id (R2); minimal HTML with an EventSource.
2. **Why now:** DoD-4 — supervisor sees the live alert.

#### TDD
```
RED (integration): test_alert_reaches_sse_stream — after an ingest that matches, the SSE stream yields the alert event with evidence
GREEN:   implement SSE endpoint + page
VERIFY:  pytest tests/integration/monitoring/test_ingest_e2e.py::test_ingest_to_supervisor_alert_e2e -m integration -q
```

#### Concurrency tests
```
(none — single-threaded)
```
(The LISTEN loop is a single asyncio task per SSE connection; broadcaster concurrency is tested in T3.2.)

#### Acceptance Criteria
- [ ] `test_ingest_to_supervisor_alert_e2e` asserts the SSE stream emits an event whose payload contains the alert_id and non-empty evidence read from the DB
- [ ] `ruff check` and `mypy` report zero warnings/errors on `interface/supervisor_sse.py`

#### DoD
- [ ] `pytest tests/integration/monitoring/test_ingest_e2e.py::test_ingest_to_supervisor_alert_e2e -m integration` passes

---

## Coverage Matrix

| # | Gap / Requirement (M0 DoD) | Task(s) | Resolution |
|---|---|---|---|
| 1 | Segment posted → Turn in hypertable with timestamp | T0.1, T3.1, T4.1 | streaming route enqueues; repo inserts; hypertable holds created_at |
| 2 | Segmentation + context window from live stream | T2.2 | orchestrator reuses `segment`/`build` on each Turn |
| 3 | One DSL rule fires on a window → alert with evidence | T1.2, T2.2 | orchestrator evaluates rule; Alert carries EvidenceItems |
| 4 | Supervisor page shows active conversation + live alert | T3.2, T4.2 | LISTEN/NOTIFY → SSE stream |
| 5 | Backpressure (bottleneck: ingest fan-in) | T1.1 | bounded channel awaiting put |
| 6 | Deterministic shutdown (no dropped turns) | T2.1 | two-phase drain |
| 7 | Two-tier tests (unit + real Timescale) | T1.1, T1.2, T2.1, T2.2, T3.1, T3.2, T4.1, T4.2 | hermetic unit + integration marker |
| 8 | Layered package skeleton + optional `[monitoring]` extra (DIP scaffold) | T0.2 | package-by-layer dirs + pyproject extra |

**Coverage: 8/8 gaps covered (100%)**

## Global Definition of Done

- [ ] All phases completed
- [ ] All tests passing — `pytest tests/unit/monitoring -q` and `pytest tests/integration/monitoring -m integration -q` green
- [ ] Zero type errors — `mypy src/talkex/monitoring tests`
- [ ] Zero lint warnings — `ruff check src/talkex/monitoring tests` + `ruff format --check`
- [ ] File-size budget respected (≤ 500 LoC per file, `.claude/rules/architecture.md`)
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] No modification to reused `talkex` public APIs (backward compat)
- [ ] Runtime-metric proof — the DoD-4 e2e test observes a real alert on the SSE stream (not just compiles)
- [ ] Plan archived to `knowledge-base/plans/completed/` after `/review` READY_TO_MERGE + merge

## Failure scenarios (external I/O)

| Dependency | Failure mode | How the test reproduces it | Expected behavior |
|---|---|---|---|
| `timescale:turns` (DB) | insert timeout / connection reset mid-insert | integration test kills the connection (or a short statement_timeout) | orchestrator surfaces a typed error; the session does not silently drop the turn; no partial commit |
| `postgres LISTEN/NOTIFY` | listener connection dropped | close the LISTEN connection then notify | SSE endpoint reconnects the LISTEN; no crash; next notify is delivered |
| `POST /ingest` (channel full) | producer floods a full channel | flood maxsize=1 channel | `put` awaits (backpressure), request does not error; no OOM |

## Final Phase: Integration Validation (MANDATORY)

**Objective:** prove the four M0 DoDs work in a real workload against a real Timescale.

### Execution
```
docker compose -f deploy/monitoring/docker-compose.yml up -d           # real Timescale on 5433
pytest tests/unit/monitoring -q                                        # hermetic unit tier
pytest tests/integration/monitoring -m integration -q                  # real-DB integration tier (incl. e2e + failure)
mypy src/talkex/monitoring tests
ruff check src/talkex/monitoring tests && ruff format --check src/talkex/monitoring tests
```

### Acceptance Criteria
- [ ] All test suites green (unit + integration)
- [ ] `test_ingest_to_supervisor_alert_e2e` green (the Goal metric)
- [ ] Zero type errors; zero lint warnings
- [ ] Runtime-metric proof — a real alert observed on the SSE stream
- [ ] Failure scenarios exercised (DB timeout, listener drop, channel-full backpressure)

### If Validation Fails
1. Classify plan-caused vs pre-existing.
2. Fix all plan-caused failures.
3. Re-run the chain.
4. Log pre-existing issues in the PR description.
