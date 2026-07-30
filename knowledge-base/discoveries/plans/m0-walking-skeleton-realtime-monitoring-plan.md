# Discovery Plan: M0 Walking Skeleton — Real-Time Attendance Monitoring

> **Version 1.1** (edge-cases absorbed 2026-07-29; see `knowledge-base/reviews/m0-...-edge-cases-2026-07-29.md`) — This discovery investigates how three cloned reference projects solve the four
> concrete problems the M0 walking skeleton must solve — bounded streaming ingestion, real-time
> push to a supervisor, the end-to-end transcript→classify→persist pipeline shape, and how each is
> integration-tested — so the resulting blueprint can lock the M0 architecture (Design Patterns +
> System Design + OOP) against the ADR-005 storage spine before any code is written.

**Slug:** `m0-walking-skeleton-realtime-monitoring`
**Owner:** paulohenriquevn
**Created:** 2026-07-29
**Time budget:** 4.5h (per-project breakdown in ADR D1)

## Context

M0 is the walking skeleton of the `realtime-attendance-monitoring` roadmap (`ROADMAP.md § M0`): the
thinnest end-to-end slice — streaming ingest of one transcript segment → segmentation into
turns/windows → one cascade check (a DSL rule + sentiment stub) → persistence to Postgres/TimescaleDB
→ one evidence-backed alert on a minimal supervisor screen.

The storage spine is locked by `docs/adr/ADR-005-online-storage-realtime-monitoring.md` (single
Postgres/TimescaleDB with pgvector + BM25, 30-day hot window). The domain primitives already exist
in `talkex` (`src/talkex/models/`, `src/talkex/segmentation/`, `src/talkex/context/`,
`src/talkex/rules/`). What is NOT yet designed — and what this discovery must inform — is the
**streaming ingestion boundary**, the **real-time push to the supervisor**, and the **integration-test
harness against a real Timescale**. The `.claude/rules/architecture.md` layering (interface →
application → domain ← infrastructure, DIP at boundaries) and `.claude/rules/testing.md` pyramid
(integration tests use a real DB) constrain every pattern this discovery may borrow.

## Objective

Enable the M0 architecture blueprint to decide, with evidence, **how to structure the streaming
ingestion boundary, the real-time supervisor push, and the integration-test harness** — reusing the
existing `talkex` domain and respecting ADR-005 and `architecture.md`.

- [ ] All research questions in this plan answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison table populated for every in-scope reference project
- [ ] Recommendations section provides at least one concrete decision proposal per research question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope (per reference project)

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/livekit-agents/` | `livekit-agents/livekit/agents/voice/`, `livekit-agents/livekit/agents/utils/aio/`, `tests/` | Closest analog to bounded streaming ingest + live session state + backpressure |
| `knowledge-base/references/chatwoot/` | `app/channels/`, `app/listeners/`, `app/models/`, `spec/channels/` | Real-time fan-out to many operators + conversation/message domain + its spec pattern |
| `knowledge-base/references/ai-powered-call-center-intelligence/` | `backend/`, `analytics/` | End-to-end transcript→analysis→persistence pipeline shape (LLM-online step is prior art we deliberately reject) |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/livekit-agents/` — all `livekit-plugins-*` and TTS/STT/voice-model plugins | ASR/voice is Macaw's domain (ROADMAP out-of-scope); only the async streaming/session machinery is relevant |
| `knowledge-base/references/chatwoot/` — `app/javascript/`, `app/views/`, billing/enterprise dirs | V1 reuses the existing React demo frontend; only the server-side real-time push + domain is in scope |
| `knowledge-base/references/ai-powered-call-center-intelligence/` — `frontend/`, `churn_model/` | CRA default frontend and the offline churn model are not the M0 online slice |
| `knowledge-base/references/portuguese-bert/`, `knowledge-base/references/portuguese-nlp/` | Sentiment ML is M2, not M0 — deferred to the M2 discovery |
| Any project NOT under `knowledge-base/references/` | Cross-Project Rule: never claim a feature without reading its source |

## ADRs

### D1 — Time budget + stop conditions

**Decision:** livekit-agents: 2h · chatwoot: 1.5h · ai-powered-call-center-intelligence: 1h.

**Rationale:** livekit-agents is the closest analog to the hardest M0 problem (bounded streaming
ingest + backpressure), so it gets the deepest dive; chatwoot is the reference for real-time push +
domain modelling; ai-powered-call-center-intelligence is small and mainly confirms the end-to-end
pipeline shape (and the online-LLM anti-pattern we reject), so the shallowest budget.

**Alternatives considered:** equal split (rejected — livekit warrants more), single-project deep-dive
(rejected — the four corners span all three).

**Stop condition — per question:** When a question's Fase A returns empty matches after 3 consecutive
retries with different query variants, mark it BLOCKED with reason "Fase A exhausted" and continue.
Never fabricate a Fase B answer (Unbreakable Rule 3).

**Stop condition — per project:** When a project's budget is exhausted with questions pending, mark
them BLOCKED with reason "budget exhausted" and advance. If every remaining question is done or
honestly blocked, emit `<promise>BLUEPRINT_BLOCKED</promise>` — never `BLUEPRINT_COMPLETE` from a
blocked state.

**Consequences:** the halt-loop stops per-project on budget exhaustion; blocked questions surface in
the blueprint as next-discovery seeds.

### D2 — Investigation depth

**Decision:** Read the key technique files end-to-end (they carry the design intent we borrow); for
dependencies and tools, Grep/Read the manifest + config files (text-shape, no deep read needed).

**Rationale:** the technique files (`agent_session.py`, `room_channel.rb`, `backend/main.py`) encode
the pattern we adapt, so intent + edge-cases matter; manifests are declarative and need only the
version/pin extracted.

**Consequences:** deeper token spend on ~5 technique files; cheap scan on manifests/CI configs.

### D3 — Scaffold adaptation (project-specific)

**Decision:** The cycle framework's default `rules/` and `.claude/knowledge-base/` locations do not
match this repo — project rules live in `.claude/rules/` and reference clones live in
`knowledge-base/references/` (repo root). This plan cites those real locations.

**Rationale:** bending the repo to the tool's default paths would be a workaround; citing the real
paths is faithful and keeps the `discover-plan-confidence` reference-existence check green.

**Consequences:** citations use `knowledge-base/references/...` (root) and `.claude/rules/...`.

## Research Questions

Each question maps to one Coverage Corner. Fase A = broad hotspot map (ast-grep/grep); Fase B = deep
Read at each hotspot producing line-exact citations.

| # | Question | Corner | Reference project(s) | Fase A (broad map) | Fase B (deep Read) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | How does livekit-agents bound a live streaming session and apply backpressure between producer and consumer? | techniques | `knowledge-base/references/livekit-agents/` | `ast-grep run -p 'class $N: $$$' --lang python` over `livekit-agents/livekit-agents/livekit/agents/voice/agent_session.py` + `utils/aio/channel.py` to list the session + channel classes | Read `agent_session.py` (session lifecycle/state) and `utils/aio/channel.py` (bounded Chan + `ChanClosed`) end-to-end; capture the backpressure primitive | Prose + class sketch of the bounded-channel/session pattern, with `knowledge-base/references/livekit-agents/...:line` citations |
| Q2 | How does chatwoot fan out a real-time update to many connected operators, and how is the conversation/message domain shaped? | techniques | `knowledge-base/references/chatwoot/` | Grep `broadcast`/`ActionCable` in `app/channels/room_channel.rb` + `app/listeners/action_cable_listener.rb`; list associations in `app/models/conversation.rb` | Read the channel + listener + `conversation.rb`; capture the event→broadcast fan-out and the domain associations | Sequence of domain-event → broadcast; domain-model note; citations to `knowledge-base/references/chatwoot/...:line` |
| Q3 | What is the end-to-end transcript→analysis→persistence pipeline shape in ai-powered-call-center-intelligence, and where is the (rejected) online-LLM step? | techniques | `knowledge-base/references/ai-powered-call-center-intelligence/` | Grep the route + stage calls in `backend/main.py`; Grep the load path in `analytics/duckdb_loader.py` | Read `backend/main.py` (FastAPI route + stage orchestration) and `analytics/duckdb_loader.py` (persist to DuckDB); identify the GPT-online call we replace with the cascade | Pipeline stage list, the persistence write shape, and the exact online-LLM call to avoid, with citations |
| Q4 | How does livekit-agents integration-test the streaming/session layer, including fault + backpressure injection? | tests | `knowledge-base/references/livekit-agents/` | Glob `tests/` for `test_agent_session.py`, `docker-compose.yml`, `toxic_proxy.py` (text-shape — Fase A may skip) | Read `tests/test_agent_session.py` (session assertions) + `tests/toxic_proxy.py` + `tests/docker-compose.yml` (fault/real-dep harness) | How the boundary is tested against real deps + fault injection; a template for M0's ingest integration test |
| Q5 | How does chatwoot spec its real-time channel (and thus the DB/broadcast boundary)? | tests | `knowledge-base/references/chatwoot/` | Glob `spec/channels/room_channel_spec.rb` (text-shape) | Read `spec/channels/room_channel_spec.rb`; capture how the channel + broadcast is asserted | Assertion style for a real-time push test; informs M0's live-push test |
| Q6 | What real-time + persistence dependencies and versions do the peers pin (async stack; pg/pg_search/redis/sidekiq; fastapi/duckdb)? | deps | all three | Grep `livekit-agents/pyproject.toml`, `chatwoot/Gemfile`, `ai-powered-call-center-intelligence/requirements.txt` for the relevant deps (text-shape) | Read each match in context; extract version + role | Version table (peer → dep → version → role) confirming/adjusting M0's stack choices vs ADR-005 |
| Q7 | What is each peer's local-dev + test-run story (docker-compose for a real DB, CI shape)? | tools | all three | Glob `livekit-agents/tests/docker-compose.yml` + `.github/workflows/ci.yml`; `chatwoot/docker-compose.yaml`; `ai-powered-call-center-intelligence/run_app.sh` | Read each; capture the "bring up a real dependency + run tests" recipe | Step-by-step dev/test harness recipe; input to M0's docker-compose Timescale + integration-test command |

## Coverage Matrix

| Corner | Questions mapped | Status |
|---|---|---|
| Integration tests | Q4, Q5 | Covered |
| Dependencies | Q6 | Covered |
| Tools | Q7 | Covered |
| Techniques | Q1, Q2, Q3 | Covered |

**Coverage: 4/4 corners covered (100%)**

## Halt-loop Checkpoints

| Checkpoint | Assertion | Action if fails |
|---|---|---|
| Before answering Qx | Every `knowledge-base/references/{project}/{path}` declared in the question exists | Mark Qx BLOCKED "path not found", continue |
| Per-question Fase A budget | Fase A returned ≥1 hotspot OR 3 query-variant retries attempted | After 3 empty retries, mark Qx BLOCKED "Fase A exhausted"; continue |
| After answering Qx | Blueprint section under Qx has ≥1 citation | Re-iterate Qx (1 retry max) |
| Per-project time budget | Project budget (D1) not exhausted | When exhausted, mark remaining Qx BLOCKED "budget exhausted"; advance |
| Fase B scope cap (Q1, Q2) — from edge-case EC-2 | Fase B reads ONLY the files named in the RQ (`agent_session.py` + `channel.py`; `room_channel.rb` + `action_cable_listener.rb` + `conversation.rb`) | If tempted to read sibling modules, STOP — the bounded-channel + broadcast-listener is the whole answer |
| Before promising complete | All 4 coverage corners have populated blueprint sections | Refuse promise, continue iterating |

## Acceptance Criteria

- [ ] All research questions answered OR explicitly marked BLOCKED with reason
- [ ] All four coverage corners have populated sections in the blueprint
- [ ] Every citation in the blueprint points to a real `knowledge-base/references/{...}` path
- [ ] At least one ADR section in the blueprint synthesizes M0 architecture decisions
- [ ] Time budget respected per project
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m0-walking-skeleton-realtime-monitoring-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed → re-score)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations
- [ ] Coverage Matrix 100% covered
- [ ] ADRs reference at least one project rule — `.claude/rules/architecture.md` (layering/DIP), `.claude/rules/testing.md` (integration tests against real deps), and `docs/adr/ADR-005` (storage spine)
