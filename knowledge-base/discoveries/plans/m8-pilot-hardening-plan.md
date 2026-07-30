# Discovery Plan: M8 — Pilot Hardening & V1 Ship

> **Version 1.0** — Investigate how the cloned peers expose health/readiness (chatwoot's health controller),
> collect self-metrics (livekit's usage metrics), and track engagement/activity (chatwoot's online-status
> tracker), so we can lock the M8 hardening architecture: a `/health` readiness endpoint for the monitor
> itself, an alert-engagement metric (the north-star proxy), a V1-acceptance harness that runs every V1 ship
> criterion and emits a single pass/fail report from the real synthetic-load evidence, and an operational
> runbook + failure-recovery posture. Reference projects in scope: `chatwoot` (a production health endpoint +
> engagement tracker with real specs) and `livekit-agents` (a self-metrics collection pattern). The blueprint
> must let us decide the health-probe set, the engagement metric shape, the acceptance-report contract, and
> the failure-recovery posture.

**Slug:** `m8-pilot-hardening`
**Owner:** tomas-herrera (Data & Eval Engineer)
**Created:** 2026-07-30
**Time budget:** 4h (per-project breakdown in ADR D1)

## Context

M8 (`ROADMAP.md § M8`) is the capstone: prove every V1 ship criterion (100% coverage, alert p95 < ~2 s,
retrieval p95 < 200 ms, sentiment macro-F1 ≥ 0.70 > baseline, critical-alert precision ≥ 0.8, purge working),
track supervisor alert-engagement (north-star proxy), and put an operational runbook + health/latency
self-monitoring + failure recovery in place. It depends on ALL prior milestones (M0-M7), whose evidence
already exists (alert p95 < 2 s and precision ≥ 0.89 from M3; retrieval p95 = 160.95 ms from M5; sentiment
macro-F1 0.856 from M2; purge/rollup-survives-drop from M6). The open gap M8 closes: how mature peers expose
a readiness endpoint (`chatwoot/app/controllers/health_controller.rb`), collect self-metrics
(`livekit-agents/.../metrics/usage_collector.py`), and track engagement (`chatwoot/lib/online_status_tracker.rb`)
— so the monitor can be operated, not just built. Honest scope: there is no real 8 kHz call-center pilot data
in this environment, so the V1 criteria are validated on synthetic load (as M2-M6 already did) and aggregated
into a readiness report; real-world drift (ROADMAP risk 1) is documented as unvalidated-until-real-data.
Constrained by `.claude/rules/architecture.md § 1` (the health endpoint is an interface adapter) and
`.claude/rules/testing.md` (the acceptance harness re-runs real checks, not a hard-coded PASS).

## Objective

Decide the M8 hardening architecture (health-probe set, engagement metric, V1-acceptance contract, failure
recovery) from evidence in the peers. Success criteria for the blueprint:

- [ ] All 7 research questions answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison table populated for `chatwoot` and `livekit-agents`
- [ ] Recommendations section provides at least one concrete decision proposal per research question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope (per reference project)

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/chatwoot/` | `app/controllers/health_controller.rb`, `spec/controllers/health_controller_spec.rb`, `lib/online_status_tracker.rb`, `spec/lib/online_status_tracker_spec.rb` | A production health endpoint + an engagement/activity tracker, both with real specs — the closest analog to M8's `/health` + engagement metric. |
| `knowledge-base/references/livekit-agents/` | `livekit-agents/livekit/agents/metrics/usage_collector.py`, `livekit-agents/livekit/agents/metrics/base.py` | A self-metrics collection + aggregation pattern — informs the monitor's own latency/health metrics. |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/chatwoot/app/javascript/` | Front-end status widgets, not the health backend. |
| `knowledge-base/references/livekit-agents/examples/` | Example agents, not the metrics core. |
| `knowledge-base/references/ai-powered-call-center-intelligence/`, `portuguese-*` | No health/engagement surface (ADR D3). |
| `knowledge-base/references/*/` build artifacts, `node_modules/`, `vendor/` | Not source of truth. |

## ADRs

### D1 — Time budget + stop conditions

**Decision:** chatwoot: 2.5h; livekit-agents: 1.5h. Total 4h.

**Rationale:** chatwoot has both the health endpoint AND the engagement tracker with specs (the primary analog);
livekit contributes the self-metrics-collection pattern in 1.5h.

**Alternatives considered:** equal split (rejected — chatwoot is denser for M8); chatwoot-only (rejected — loses
the metrics-aggregation pattern).

**Stop condition — per question (mandatory):** After 3 empty Fase-A query-variant retries, mark the question
BLOCKED ("Fase A exhausted") and continue. Do NOT pad with unrelated hotspots.

**Stop condition — per project (mandatory):** On budget exhaustion, mark remaining questions BLOCKED
("budget exhausted"). If every remaining question is `done` or honestly `blocked`, emit
`<promise>BLUEPRINT_BLOCKED</promise>` (never `BLUEPRINT_COMPLETE` from a blocked state).

**Anti-pattern:** NEVER fabricate Fase B answers to close a Fase-A-exhausted question (Unbreakable Rule 3).

**Consequences:** the halt-loop stops per-project on budget exhaustion; blocked questions surface in the
blueprint's `## Blocked questions` section.

### D2 — Investigation depth

**Decision:** Read the health controller + spec + status tracker + metrics files end-to-end (all ≤ 222 LoC);
Grep for the readiness probes / metric fields. ast-grep Fase A only where a method map helps.

**Rationale:** the readiness-probe set + the engagement-metric shape live in these small files whole.

**Consequences:** deeper per-file cost, tiny files, within budget.

### D3 — Defer the non-operational peers

**Decision:** Exclude `ai-powered-call-center-intelligence`, `portuguese-bert`, `portuguese-nlp`.

**Rationale:** none has a health/engagement/metrics surface; the four corners are covered by the two in-scope peers.

**Consequences:** no corner is left to a peer that cannot answer it.

## Research Questions

| # | Question | Corner | Reference project(s) | Fase A (broad — map) | Fase B (deep — Read at each hotspot) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | What does chatwoot's health controller PROBE to signal readiness (DB? cache? version?)? | techniques | `knowledge-base/references/chatwoot/app/controllers/health_controller.rb` | SKIP Fase A (7 LoC). Read the controller fully | Read the controller; capture what it returns + what it checks | Probe inventory: check → mechanism → response → `path:line`; informs our `/health` probe set |
| Q2 | How does livekit COLLECT + AGGREGATE self-metrics (counters, latency)? | techniques | `knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/metrics/usage_collector.py` | `ast-grep run -p 'def $NAME($$$): $$$' --lang python knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/metrics/usage_collector.py` | Read the collector; capture how metrics are accumulated + summarized | Prose + a collect→aggregate flow with citations; informs the monitor's own latency/health metrics |
| Q3 | What does chatwoot's health path DEPEND on (a base controller, a status lib, redis)? | deps | `knowledge-base/references/chatwoot/app/controllers/health_controller.rb`, `lib/online_status_tracker.rb` | SKIP Fase A. Grep `Redis|ActiveRecord|include|<` in both files | Read the matched dependency lines | Dependency table: dep → role → citation; informs our health/engagement backing (Timescale/pool) |
| Q4 | What data structures does livekit's metrics module use (dataclasses, fields)? | deps | `knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/metrics/base.py` | `ast-grep run -p 'class $NAME: $$$' --lang python knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/metrics/base.py` to map the metric classes | Read each metric dataclass; capture the fields (counts, durations) | Metric-shape table: class → fields → citation; informs our metric value objects |
| Q5 | How does chatwoot TEST the health endpoint (status code, body)? | tests | `knowledge-base/references/chatwoot/spec/controllers/health_controller_spec.rb` | SKIP Fase A (11 LoC). Read the spec fully | Read the spec; capture the request + the asserted response | Table: example → request → assertion → `path:line`; informs M8's `/health` test |
| Q6 | How does chatwoot's online-status tracker TRACK engagement/activity (presence, last-seen)? | tools | `knowledge-base/references/chatwoot/lib/online_status_tracker.rb` | `ast-grep run -p 'def $NAME($$$)' --lang ruby knowledge-base/references/chatwoot/lib/online_status_tracker.rb` to map the tracker methods | Read the tracker + its spec; capture how activity is recorded + queried | Prose + a track→query flow with citations; informs M8's alert-engagement metric (acted-on rate) |
| Q7 | How is livekit's metrics collector STRUCTURED for reuse (entrypoint, summarize)? | tools | `knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/metrics/usage_collector.py` | `ast-grep run -p 'def summarize($$$): $$$' --lang python knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/metrics/usage_collector.py` | Read the collector's public surface; capture the collect + summarize entrypoints | API-shape table: method → role → citation; informs our engagement/health collector structure |

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
- [ ] At least one ADR section in the blueprint synthesizes the M8 health/engagement/acceptance/recovery decisions
- [ ] Time budget respected per project
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m8-pilot-hardening-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed → confidence re-score)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations
- [ ] Coverage Matrix 100% covered
- [ ] ADRs reference at least one principle from project rules — here `.claude/rules/architecture.md § 1` (the health endpoint is an interface adapter) + `testing.md` (the acceptance harness re-runs real checks) + honesty (Rule 3 — synthetic-load validation is labeled, real-drift risk documented)
