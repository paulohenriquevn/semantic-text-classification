# Blueprint: M0 Walking Skeleton — Real-Time Attendance Monitoring

> **Version 1.0** — Technical blueprint locking the M0 architecture (streaming ingestion → segmentation →
> DSL rule → Timescale persistence → live supervisor alert) by synthesizing three cloned reference
> projects against ADR-005 and `.claude/rules/architecture.md`. Produced by the `cycle-discover` execute
> phase from `knowledge-base/discoveries/plans/m0-walking-skeleton-realtime-monitoring-plan.md`.

**Slug:** `m0-walking-skeleton-realtime-monitoring`
**Created:** 2026-07-29
**discover-confidence verdict:** recorded at the end after scoring.

## Executive summary

M0 is a bounded producer→consumer streaming pipeline with an explicit session state machine and a
commit-time domain-event broadcast to the supervisor. The references give citable precedent for the
backpressure primitive (livekit bounded channel), the real-time fan-out (chatwoot domain-event →
listener → async broadcast), and the end-to-end pipeline shape plus its anti-patterns
(ai-powered-call-center-intelligence, whose inline online-LLM call is exactly what TalkEx's "LLMs
offline only" axiom forbids). No peer uses TimescaleDB — hypertable specifics come from ADR-005.

## Context

M0 must prove the architecture end-to-end (ROADMAP § M0). Domain primitives already exist in
`src/talkex/models/`, `src/talkex/segmentation/`, `src/talkex/context/`, `src/talkex/rules/`. Undesigned
and locked here: the streaming ingestion boundary, the live supervisor push, and the integration-test
harness against a real TimescaleDB — within `.claude/rules/architecture.md` layering (interface →
application → domain ← infrastructure, DIP at borders) and `docs/adr/ADR-005-online-storage-realtime-monitoring.md`.

## Objective

Lock, with cited evidence, how to structure the ingestion boundary (backpressure), the live supervisor
push (commit-time broadcast), the online decide step (DSL rules, not LLM), the persistence write
(per-Turn hypertable insert), and the two-tier test strategy — so M0 implementation proceeds without
rework.

## Coverage Corner 1 — Integration Tests

**livekit-agents — session tests + fault injection.** Hermetic unit tests over fakes with virtual time:
`knowledge-base/references/livekit-agents/tests/test_agent_session.py:51` (markers), assertion by ordered
state transitions `:219`. Fault injection via Toxiproxy: `knowledge-base/references/livekit-agents/tests/toxic_proxy.py:89`
(`Proxy.down()`), `add_toxic` `:112`; concrete timeout-toxic test with elapsed-time bounds
`knowledge-base/references/livekit-agents/tests/test_tts.py:358`. Real deps in Docker for fault tests:
`knowledge-base/references/livekit-agents/tests/docker-compose.yml:2`.

**chatwoot — real-time channel contract.** Spec drives the real channel with `stub_connection` + real DB
factories: `knowledge-base/references/chatwoot/spec/channels/room_channel_spec.rb:8`; asserts subscription
confirmed + streams for expected topics (per-user and per-account): `:12` and `:18`.

**M0 test decision:** two tiers — fast hermetic unit tests asserting the ordered event/state stream and
DSL evaluation; a fault-injection integration tier with a proxy between app and Timescale to assert
ingestion behavior under DB timeout/backpressure; a channel contract test at the subscribe/broadcast
boundary. (ai-powered-call-center-intelligence has no test suite — the tests corner draws only on livekit
and chatwoot, which is sufficient.)

## Coverage Corner 2 — Dependencies

| Peer | Dependency | Version | Role | Source |
|---|---|---|---|---|
| livekit-agents | livekit | ==1.1.13 | realtime transport | `knowledge-base/references/livekit-agents/livekit-agents/pyproject.toml:30` |
| livekit-agents | pydantic | >=2.0,<3 | typed models | `knowledge-base/references/livekit-agents/livekit-agents/pyproject.toml:46` |
| chatwoot | rails | ~> 7.1 | ActionCable framework | `knowledge-base/references/chatwoot/Gemfile:7` |
| chatwoot | redis | unpinned | ActionCable pub/sub backend | `knowledge-base/references/chatwoot/Gemfile:67` |
| chatwoot | sidekiq | ~> 7.3 | async broadcast fan-out | `knowledge-base/references/chatwoot/Gemfile:136` |
| ai-call-center | fastapi | ==0.111.0 | HTTP API | `knowledge-base/references/ai-powered-call-center-intelligence/requirements.txt:2` |
| ai-call-center | openai | ==0.28.0 | online LLM — to avoid | `knowledge-base/references/ai-powered-call-center-intelligence/requirements.txt:5` |

**M0 deps decision:** FastAPI+uvicorn + pydantic v2 are already in the TalkEx stack. No peer uses
TimescaleDB; chatwoot relies on Redis pub/sub + Sidekiq for live push, so the transport backend is a
decision the references frame but do not make for a Python stack (see D3).

## Coverage Corner 3 — Tools

chatwoot boots real Postgres + Redis + Sidekiq in one compose:
`knowledge-base/references/chatwoot/docker-compose.yaml:85` (`pgvector/pgvector:pg16`), redis `:97`,
sidekiq `:46`. livekit compose brings up real deps for fault tests:
`knowledge-base/references/livekit-agents/tests/docker-compose.yml:2`; CI is uv-based split ruff/type
jobs: `knowledge-base/references/livekit-agents/.github/workflows/ci.yml:14`. ai-call-center uses an
embedded/file DuckDB + shell run (no containerized DB):
`knowledge-base/references/ai-powered-call-center-intelligence/run_app.sh:13`.

**M0 tools decision:** adopt the compose-real-dependency-then-run-tests recipe — a `docker-compose.yml`
booting a real TimescaleDB (ADR-005) + pub/sub backend, and an integration tier running pytest against
that live DB. The embedded-DB shortcut is not adequate for hypertable behavior. CI: separate
lint/type/unit jobs (no services) + a slower integration job spinning up Timescale.

## Coverage Corner 4 — Techniques

**livekit-agents — bounded session + backpressure.** Backpressure primitive is a bounded asyncio channel:
`knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/utils/aio/channel.py:49`
(`class Chan`, `maxsize` `:52`, `full()` `:162`); typed contract `:17,:21,:25`. Backpressure lives in the
producer's awaiting `send()` `:71`; non-blocking `send_nowait` raises `:90`; drain-then-close `:133`;
`ChanClosed → StopAsyncIteration` `:174`. Session lifecycle is an explicit state machine with single
guards: `knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/voice/agent_session.py:468`,
`:555`, idempotent `_update_agent_state` `:1723`, two-phase `drain()` `:1013` then `_aclose_impl` `:1049`,
single consumer task `:1670`.

**chatwoot — live push + conversation domain.** Subscribe binds per-scope streams:
`knowledge-base/references/chatwoot/app/channels/room_channel.rb:2`, `stream_from` `:27`. Domain event →
broadcast decoupled through a listener: `knowledge-base/references/chatwoot/app/listeners/action_cable_listener.rb:41`,
events `:65,:79,:86`; de-duplicated token union `:202`, deferred to a background job `:222`. Conversation
aggregate + commit-time broadcast: `knowledge-base/references/chatwoot/app/models/conversation.rb:112`,
`has_many :messages` `:122`, `after_create_commit` `:134` → dispatch `:311`, supervisor-query index `:35`.

**ai-powered-call-center-intelligence — pipeline shape + anti-pattern.** FastAPI route orchestrates inline:
`knowledge-base/references/ai-powered-call-center-intelligence/backend/main.py:37`; the online-LLM step to
reject is the synchronous GPT call in the handler `:49` (and `:62`), OpenAI global `:15`, CORS `*` `:23`.
Persistence: `insights JSON, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
`knowledge-base/references/ai-powered-call-center-intelligence/analytics/duckdb_loader.py:17`, append insert
`:38`, but a file-batch loader `:28` (not streaming).

**M0 techniques decision:** bounded queue with awaiting put; explicit session state machine with two-phase
drain; commit-time domain-event → listener → async broadcast to a supervisor-scoped topic; online decide =
DSL rule engine (LLM offline); per-Turn streaming insert into a Timescale hypertable with a JSONB evidence
column.

## Cross-cutting Comparison

| Concern | livekit-agents | chatwoot | ai-call-center | M0 decision |
|---|---|---|---|---|
| Ingestion/backpressure | bounded `Chan` in `send()` | (Rails request) | inline in handler (bad) | bounded queue, awaiting put |
| Session lifecycle | explicit state machine + drain | commit callbacks | none | state machine + two-phase drain |
| Live push | — | event→listener→async broadcast on commit | — | same, Postgres LISTEN/NOTIFY |
| Online "decide" | — | — | GPT online (rejected) | DSL rule engine (LLM offline) |
| Persistence | — | PG + composite index | DuckDB file-batch | Timescale hypertable, per-Turn insert |
| Test strategy | hermetic + Toxiproxy fault tier | real-DB channel spec | none | unit (fakes) + integration (real Timescale) |

## ADRs

### D1 — Producer/Consumer ingestion with backpressure

FastAPI streaming route enqueues a `Turn` onto a bounded queue (`Chan`-semantics, `channel.py:49,:71`);
a separate consumer task drains → context window → rule engine. No rule/persistence work inline in the
socket handler (rejecting `main.py:49`). **Consequence:** a fast transcript producer cannot unbounded-buffer
ahead of the slower consumer; the awaiting `put` is the backpressure lever.

### D2 — Explicit session State pattern with graceful drain

`initializing → listening → closing` with a single closing guard and `drain()`-before-close
(`agent_session.py:555,:1013,:1049`). **Consequence:** an in-flight rule/alert evaluation finishes before
teardown; no half-processed turn on disconnect.

### D3 — Observer/Mediator live push on commit

`Alert`/`Turn` emit a domain event after the DB transaction commits (`conversation.rb:134,:311`); a listener
maps it to a supervisor-scoped topic (`room_channel.rb:27`) and enqueues the push off the request path
(`action_cable_listener.rb:222`). Transport for M0 = **Postgres LISTEN/NOTIFY** (dependency-light, aligns
with ADR-005 single-spine; Redis pub/sub is the chatwoot precedent if throughput later demands it).
**Consequence:** the supervisor never sees an alert that later rolls back.

### D4 — Hexagonal layering with DIP ports

interface (FastAPI route + SSE/WS channel) → application (session orchestrator, event listener) → domain
(`Turn`, `ContextWindow`, DSL `Rule`, `Alert`, domain events) ← infrastructure (Timescale
`TurnRepository`/`AlertRepository`, `AlertBroadcaster` adapters), per `.claude/rules/architecture.md § 1–2`.
**Consequence:** the domain stays free of FastAPI/DB/transport; adapters are injected at the composition root.

### D5 — Streaming per-Turn Timescale hypertable with JSONB evidence

Keep the evidence-JSON + `created_at` idea (`duckdb_loader.py:17`), reject the file-batch model; index for
the supervisor's live query the way chatwoot indexes open conversations by scope+status (`conversation.rb:35`).
**Consequence:** each Turn is queryable the instant it lands; hypertable chunking/retention deferred to M1
(ADR-005) since no peer attests it.

### D6 — Two-tier tests (hermetic unit + real-Timescale integration)

Hermetic unit tests asserting the ordered event/state stream + DSL (`test_agent_session.py:51,:219`);
integration tier via docker-compose real Timescale (`docker-compose.yaml:85`) with an optional
Toxiproxy-style fault layer (`toxic_proxy.py:89`); channel contract test at the subscribe/broadcast boundary
(`room_channel_spec.rb:12`). **Consequence:** hypertable behavior is tested against a real DB, not an
embedded shortcut.

## Recommendations

1. Implement M0 as: `interface/ingest_api.py` (FastAPI streaming route) → bounded `TurnChannel` (D1) →
   `application/session.py` state machine (D2) → reuse `TurnSegmenter` + `SlidingWindowBuilder` → reuse
   `rules` engine for one rule → `infrastructure/timescale_repo.py` per-Turn insert (D5) → domain event →
   `infrastructure/notify_broadcaster.py` LISTEN/NOTIFY (D3) → minimal supervisor SSE page.
2. Ship a `docker-compose.yml` with `timescale/timescaledb-ha:pg16` on a free port (5432 is taken locally)
   for dev + the integration tier (D6).
3. Sequence the DoDs: (1) ingest→Turn in hypertable; (2) segmentation→window from the live stream; (3) one
   DSL rule → alert row with evidence; (4) minimal supervisor page shows the live alert.
4. Defer to M1: hypertable chunking/retention/continuous-aggregates and the pgvector/BM25 indexes — M0 needs
   only the plain hypertable + insert path (honest gap G1).

## Honest gaps

1. **G1 — No peer uses TimescaleDB.** Hypertable chunking/retention/continuous-aggregates are unattested;
   only the real-Postgres-in-compose + integration-test pattern transfers. Source of truth = ADR-005 (M1).
2. **G2 — No peer runs a rule/alert DSL online.** The DSL→AST→evidence decide step has no external precedent
   beyond TalkEx's own `src/talkex/rules/`.
3. **G3 — Live-push transport** (LISTEN/NOTIFY vs Redis) is framed by chatwoot (Redis) but not decided for a
   Python/Timescale stack; D3 chooses LISTEN/NOTIFY for M0 with Redis as the scale escape hatch.

## discover-confidence verdict

**SHIPPABLE — score 99.4, hard_caps_triggered: none** (via
`.claude/skills/discover-confidence/scripts/run_blueprint_score.py`, 2026-07-29). Coverage 4/4 corners,
all citations resolve, 6 ADRs (D1–D6). Proceed to the `cycle-plan` phase for M0.
