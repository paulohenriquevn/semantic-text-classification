# Discovery Plan: M6 — Aggregated Dashboards (continuous aggregates)

> **Version 1.0** — Investigate how the cloned peers build INCREMENTAL reporting rollups (chatwoot's
> reporting-event rollup subsystem) and dashboard aggregations (ai-powered's analytics layer), so we can
> lock the M6 architecture: Timescale continuous aggregates for business KPIs (cancellation/alert rate per
> queue per 5 min, sentiment trend, script adherence) that refresh incrementally with no full scan, are
> retained beyond the 30-day raw purge, and back a manager dashboard with honest freshness SLOs. Reference
> projects in scope: `chatwoot` (a production incremental reporting-rollup + report-builder subsystem with
> real-DB specs) and `ai-powered-call-center-intelligence` (a columnar dashboard-aggregation contrast). The
> blueprint must let us decide the CA definitions, the refresh/retention policy, the KPI set, and the query
> shape the dashboard reads.

**Slug:** `m6-aggregated-dashboards`
**Owner:** tomas-herrera (Data & Eval Engineer)
**Created:** 2026-07-30
**Time budget:** 5h (per-project breakdown in ADR D1)

## Context

M6 (`ROADMAP.md § M6`) must give managers dashboards from Timescale continuous aggregates: rollups that
compute incrementally with no full scan (e.g. cancellation-intent rate per queue per 5 min), render with
acceptable freshness/latency, and are retained beyond 30 days (long-term business memory) while raw data
purges. M1 already shipped ONE continuous aggregate — `turns_per_min` (`deploy/monitoring/migrations/0002_m1_retention_indexes.sql:44-52`,
per-minute turn count) — plus 30-day retention on `turns`/`alerts` (`:34-35`) and a `src/talkex/analytics/`
module (aggregators/metrics/query_runner). The open gap M6 closes: how mature peers model INCREMENTAL
business rollups (so a new alert bumps a counter rather than triggering a full re-scan), how they keep the
rollup when the raw event ages out, how they define KPIs that are operationally meaningful (ROADMAP risk 2),
and how they test the increment. This is a data-governance decision (tomas-herrera's domain) — respecting
`.claude/rules/architecture.md § 1` (the dashboard read is an infrastructure adapter behind a domain port)
and `.claude/rules/testing.md § 2` (rollup correctness proven against a real DB).

## Objective

Decide the M6 continuous-aggregate architecture (CA definitions, refresh + retention policy, KPI set,
dashboard query shape) from evidence in the peers. Success criteria for the blueprint:

- [ ] All 7 research questions answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison table populated for `chatwoot` and `ai-powered-call-center-intelligence`
- [ ] Recommendations section provides at least one concrete decision proposal per research question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope (per reference project)

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/chatwoot/` | `app/models/reporting_event*.rb`, `app/builders/v2/report_builder.rb`, `app/listeners/reporting_event_listener.rb`, `spec/models/reporting_event*`, `spec/listeners/` | A production incremental reporting-rollup subsystem + report builder + real-DB specs — the closest analog to M6's KPI rollups. |
| `knowledge-base/references/ai-powered-call-center-intelligence/` | `analytics/powerdash_components.py`, `analytics/duckdb_loader.py` | A columnar dashboard-aggregation contrast (DuckDB) — validates the Timescale-CA choice and informs KPI/dashboard shape. |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/chatwoot/app/javascript/` | Front-end chart rendering belongs to a later UI slice, not the M6 rollup backend. |
| `knowledge-base/references/ai-powered-call-center-intelligence/analytics/*.ipynb`, `*.db` | Exploratory notebooks + a sample DB, not architecture source. |
| `knowledge-base/references/livekit-agents/`, `portuguese-bert/`, `portuguese-nlp/` | Streaming/embedding peers — irrelevant to reporting rollups (ADR D3). |
| `knowledge-base/references/*/` build artifacts, `node_modules/`, `vendor/` | Not source of truth. |

## ADRs

### D1 — Time budget + stop conditions

**Decision:** chatwoot: 3.5h; ai-powered-call-center-intelligence: 1.5h. Total 5h.

**Rationale:** chatwoot has a real incremental-rollup subsystem with specs (the deepest analog), so it gets
the most time; ai-powered is a short columnar-aggregation contrast that mainly informs KPI/dashboard shape.

**Alternatives considered:** equal split (rejected — ai-powered's analytics is thin); chatwoot-only (rejected
— loses the columnar-vs-CA contrast that validates our storage choice).

**Stop condition — per question (mandatory):** After 3 empty Fase-A query-variant retries, mark the question
BLOCKED ("Fase A exhausted") and continue. Do NOT pad with unrelated hotspots.

**Stop condition — per project (mandatory):** On budget exhaustion, mark remaining questions BLOCKED ("budget
exhausted"). If every remaining question is `done` or honestly `blocked`, emit `<promise>BLUEPRINT_BLOCKED</promise>`
(never `BLUEPRINT_COMPLETE` from a blocked state).

**Anti-pattern:** NEVER fabricate Fase B answers to close a Fase-A-exhausted question (Unbreakable Rule 3).

**Consequences:** the halt-loop stops per-project on budget exhaustion; blocked questions surface in the
blueprint's `## Blocked questions` section as next-discovery seed.

### D2 — Investigation depth

**Decision:** Read the reporting model/listener/builder/spec files end-to-end (short, behavior-dense Ruby);
for dependency questions, Grep the manifest/model then Read the matched lines. ast-grep Fase A only where a
Ruby method/class map helps.

**Rationale:** the rollup increment logic lives in a handful of files; reading them whole captures the
"increment vs full-scan" intent that a symbol grep misses.

**Consequences:** deeper per-file cost, but the files are small (≤ 187 LoC); total stays within budget.

### D3 — Defer the streaming + embedding-corpus peers

**Decision:** Exclude `livekit-agents`, `portuguese-bert`, `portuguese-nlp`.

**Rationale:** none has a reporting-rollup or dashboard-aggregation surface; the four corners are covered by
the two in-scope peers.

**Consequences:** no corner is left to a peer that cannot answer it.

## Research Questions

| # | Question | Corner | Reference project(s) | Fase A (broad — map) | Fase B (deep — Read at each hotspot) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | How does chatwoot compute reporting rollups INCREMENTALLY (a new event bumps a counter) rather than re-scanning raw events? | techniques | `knowledge-base/references/chatwoot/app/models/reporting_events_rollup.rb`, `app/listeners/reporting_event_listener.rb` | `ast-grep run -p 'def $NAME($$$)' --lang ruby knowledge-base/references/chatwoot/app/models/reporting_events_rollup.rb` to map the rollup methods | Read the rollup model + the listener that writes it; capture how an event increments a bucket without a full scan | Prose + a decomposition: event → increment mechanism → bucket key → `path:line`, mapped to our CA-per-5-min KPIs |
| Q2 | How does ai-powered aggregate call data for its dashboard components (grouping, columnar rollup)? | techniques | `knowledge-base/references/ai-powered-call-center-intelligence/analytics/powerdash_components.py`, `analytics/duckdb_loader.py` | `ast-grep run -p 'def $NAME($$$): $$$' --lang python knowledge-base/references/ai-powered-call-center-intelligence/analytics/powerdash_components.py` | Read both files; capture the aggregation grouping + the columnar (DuckDB) load path | Aggregation inventory: metric → grouping → engine → `path:line`; contrast vs Timescale CA |
| Q3 | What does chatwoot depend on to STORE + SCHEDULE its rollups (table/model, job/cron)? | deps | `knowledge-base/references/chatwoot/app/models/reporting_events_rollup.rb`, `app/models/reporting_event.rb` | SKIP Fase A (small files). Grep `belongs_to|scope|enum|self\.` in the two models; Grep for a scheduling job referencing them | Read the model bodies + any referenced job; capture storage shape + refresh cadence | Dependency/table table: model → columns → refresh mechanism → citation |
| Q4 | What engine does ai-powered use for dashboard aggregation (DuckDB? pandas?) and what does that choice imply vs a Timescale CA? | deps | `knowledge-base/references/ai-powered-call-center-intelligence/analytics/duckdb_loader.py`, `analytics/requirements`-adjacent | SKIP Fase A (text-shape). Grep `duckdb|pandas|import` in `analytics/duckdb_loader.py`; confirm the engine | Read `duckdb_loader.py` to confirm the columnar load + query engine | Engine verdict + a one-line implication for M6's CA-vs-columnar decision, with citation |
| Q5 | How does chatwoot TEST that a rollup increments correctly (real DB, factories, the listener path)? | tests | `knowledge-base/references/chatwoot/spec/models/reporting_events_rollup_spec.rb`, `spec/listeners/reporting_event_listener_spec.rb` | `ast-grep run -p 'it $$$ do $$$ end' --lang ruby knowledge-base/references/chatwoot/spec/models/reporting_events_rollup_spec.rb` to map example blocks | Read each spec + its factory usage; capture DB posture (real vs stub), seeded events, and the increment assertion | Table: spec example → factory → DB posture → assertion → `path:line`; informs M6's CA integration-test tier |
| Q6 | How does chatwoot SCHEDULE/refresh its rollups (a job? a listener on event creation?) and at what cadence? | tools | `knowledge-base/references/chatwoot/app/listeners/reporting_event_listener.rb` | `ast-grep run -p 'def $NAME($$$)' --lang ruby knowledge-base/references/chatwoot/app/listeners/reporting_event_listener.rb` to map the listener hooks | Read the listener fully; capture which events trigger a rollup write and whether it is synchronous or deferred | Trigger inventory: event → rollup write → sync/deferred → citation; informs our CA `refresh_continuous_aggregate` policy cadence |
| Q7 | How does chatwoot's report builder QUERY the rollups to serve a dashboard (grouping, time range, dimensions)? | tools | `knowledge-base/references/chatwoot/app/builders/v2/report_builder.rb` | `ast-grep run -p 'def $NAME($$$)' --lang ruby knowledge-base/references/chatwoot/app/builders/v2/report_builder.rb` to map the query-building methods | Read the builder; capture the group-by dimensions (queue/agent/time), the time-range handling, and the metric list | Query-shape table: dimension → SQL/scope → time range → citation; informs M6's dashboard read query over the CA |

## Coverage Matrix

| Corner | Questions mapped | Status |
|---|---|---|
| Integration tests | Q5 | Covered |
| Dependencies | Q3, Q4 | Covered |
| Tools | Q6, Q7 | Covered |
| Techniques | Q1, Q2 | Covered |

**Coverage: 4/4 corners covered (100%)**

## Halt-loop Checkpoints

| Checkpoint | Assertion | Action if fails |
|---|---|---|
| Before answering Qx | Every `knowledge-base/references/{project}/{path}` declared in Qx's Fase A exists | Mark Qx BLOCKED ("path not found"), continue |
| Per-question Fase A budget | Fase A returned ≥ 1 hotspot OR 3 retries attempted | After 3 empty retries, mark Qx BLOCKED ("Fase A exhausted"); continue |
| After answering Qx | Blueprint section under Qx has ≥ 1 citation | Re-iterate Qx (1 retry max) |
| Mid-loop sanity | Total `knowledge-base/references/` citations ≥ prose-words / 200 | Add citations to under-cited paragraphs (1 retry max) |
| Per-project time budget | Project time budget not exhausted | When exhausted, mark remaining Qx BLOCKED ("budget exhausted"); advance |
| Before promising complete | All 4 coverage corners have populated sections | Refuse promise, continue iterating |

## Acceptance Criteria

- [ ] All 7 research questions answered OR explicitly marked BLOCKED with reason
- [ ] All four coverage corners have populated sections in the blueprint
- [ ] Every citation points to a real `knowledge-base/references/{...}` path
- [ ] At least one ADR section in the blueprint synthesizes the M6 CA/refresh/retention/KPI/query decisions
- [ ] Time budget respected per project
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m6-aggregated-dashboards-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed → confidence re-score)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations
- [ ] Coverage Matrix 100% covered
- [ ] ADRs reference at least one principle from project rules — here `.claude/rules/architecture.md § 1` (the dashboard read is an infrastructure adapter behind a domain port) + the ADR-005 hot/purge split (rollups outlive raw) + KISS (extend the shipped `turns_per_min` CA pattern, don't add a columnar store)
