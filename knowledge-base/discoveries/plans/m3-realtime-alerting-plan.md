# Discovery Plan: M3 Real-Time Rule Engine & Alerting

> **Version 1.0** — Investigate how the reference project chatwoot structures event→broadcast alerting,
> tests the channel boundary, and pins its async-broadcast deps, so the M3 blueprint can harden the M0
> alerting (already shipped: DSL rule → LISTEN/NOTIFY → SSE) with a critical-rule catalogue, an
> alert-precision evaluation (≥ 0.8 on critical categories, against the internal topic labels), and a
> turn→alert latency measurement (p95 < 2s). Honest scope: the rule-engine + push are internal and SHIPPED
> in M0; the critical-rule catalogue + precision/latency evaluation are internal methodology (ADR D3).

**Slug:** `m3-realtime-alerting`
**Owner:** paulohenriquevn
**Created:** 2026-07-30
**Time budget:** 1.5h (chatwoot alerting only — see ADR D1)

## Context

M3 hardens the alerting that M0 shipped (`ROADMAP.md § M3`): the DSL rule engine, `LISTEN/NOTIFY` push, and
SSE supervisor stream already exist (`src/talkex/monitoring/`, v0.2.0). What is undone: a catalogue of
critical rules (cancellation, escalation, negative-sentiment drop — the last reusing M2's SentimentDetector),
an **alert-precision** measurement (≥ 0.8, using the internal `topic` labels as ground truth: e.g.
`cancelamento`/`reclamacao` windows SHOULD alert), and a **turn→alert latency** measurement (p95 < 2s). This
discovery re-mines chatwoot's event→broadcast alerting (the pattern M0 borrowed) and defers the
catalogue + precision/latency method to internal work (ADR D3). Constrained by `.claude/rules/architecture.md`
and `.claude/rules/testing.md`.

## Objective

Enable the M3 blueprint to decide, with evidence, **the critical-rule catalogue structure, the alert-precision
evaluation method, and the latency-measurement approach** — hardening the shipped M0 alerting.

- [ ] All research questions answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison populated
- [ ] At least one concrete decision proposal per question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/chatwoot/` | `app/listeners/`, `app/channels/`, `app/models/`, `spec/channels/`, `config/` | Event→broadcast alerting pattern + its channel spec + async-broadcast deps |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/chatwoot/` — frontend, enterprise | Not alerting-server |
| Other peers | No real-time alerting pattern relevant to M3 |
| The M0 rule engine + LISTEN/NOTIFY + SSE (shipped) | Internal, already implemented (v0.2.0) — reused, not re-researched |
| The critical-rule catalogue + precision/latency eval | Internal methodology (ADR D3) |

## ADRs

### D1 — Time budget + stop conditions
**Decision:** chatwoot 1.5h (single-project — the only alerting peer; the pattern was already partly mined in M0).
**Stop condition — per question:** after 3 empty retries, mark BLOCKED "Fase A exhausted"; never fabricate (Rule 3).
**Stop condition — per project:** on budget exhaustion, mark remaining BLOCKED; emit `<promise>BLUEPRINT_BLOCKED</promise>` if any remain.

### D2 — Investigation depth
**Decision:** Read the listener/channel/spec in full; Grep the async deps.
**Consequences:** deeper read on the broadcast fan-out; cheap scan on deps.

### D3 — Critical-rule catalogue + precision/latency are internal (honest deferral)
**Decision:** the critical-rule DSL catalogue, the alert-precision eval (using internal `topic` labels), and the
turn→alert latency measurement are internal; chatwoot informs only the event→broadcast alerting structure.
**Rationale:** no peer measures alert precision against labels or catalogues sentiment-drop rules — fabricating
peer citations for those would fail the fabricated-citation cap (Rule 3).
**Consequences:** the blueprint synthesizes the catalogue + eval internally; chatwoot is the alerting-structure evidence.

## Research Questions

| # | Question | Corner | Reference project(s) | Fase A (broad map) | Fase B (deep Read) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | How does chatwoot map a domain event to a broadcast (the alert fan-out)? | techniques | `knowledge-base/references/chatwoot/` | Grep `broadcast`/`perform_later` in `app/listeners/action_cable_listener.rb` | Read `app/listeners/action_cable_listener.rb:41` (event→broadcast), `:222` (async job fan-out), `:202` (token union) | Alerting structure → informs the M3 critical-rule → alert mapping |
| Q2 | How is the alert channel + scope-topic structured (per-supervisor routing)? | techniques | `knowledge-base/references/chatwoot/` | Grep `stream_from` in `app/channels/room_channel.rb` | Read `app/channels/room_channel.rb:27`; `app/models/conversation.rb:134` (commit-time trigger) | Topic routing → informs multi-rule alert routing |
| Q3 | How does chatwoot spec the alert channel boundary (test method for M3's precision-style assertions)? | tests | `knowledge-base/references/chatwoot/` | Glob `spec/channels/room_channel_spec.rb` | Read `spec/channels/room_channel_spec.rb:12` (subscribe + broadcast assertions) | Test template → informs the M3 alert-precision + push tests |
| Q4 | What async-broadcast dependencies does chatwoot pin (the alternative to LISTEN/NOTIFY at scale)? | deps | `knowledge-base/references/chatwoot/` | Grep `redis`/`sidekiq` in `Gemfile` | Read `Gemfile:67` (redis), `:136` (sidekiq) | Dep note → M3 stays on LISTEN/NOTIFY; Redis is the scale escape hatch |
| Q5 | What is the run/dev story for the real-time broadcast stack? | tools | `knowledge-base/references/chatwoot/` | Read `config/database.yml` + `docker-compose.yaml` | Read `docker-compose.yaml:97` (redis service) | Run recipe → M3 needs no new infra (LISTEN/NOTIFY on the existing Timescale) |

## Coverage Matrix

| Corner | Questions mapped | Status |
|---|---|---|
| Integration tests | Q3 | Covered |
| Dependencies | Q4 | Covered |
| Tools | Q5 | Covered |
| Techniques | Q1, Q2 | Covered |

**Coverage: 4/4 corners covered (100%)**

## Halt-loop Checkpoints

| Checkpoint | Assertion | Action if fails |
|---|---|---|
| Before answering Qx | Every cited `knowledge-base/references/chatwoot/{path}` exists | Mark Qx BLOCKED "path not found", continue |
| Per-question Fase A budget | ≥1 hotspot OR 3 retries | Mark BLOCKED "Fase A exhausted"; continue |
| Precision/latency/catalogue question | If a question drifts into the internal eval/catalogue with no peer source | STOP — that is ADR D3 territory (internal), not a peer citation |
| Before promising complete | All 4 corners populated | Refuse promise, continue |

## Acceptance Criteria

- [ ] All research questions answered OR explicitly BLOCKED with reason
- [ ] All four corners populated in the blueprint
- [ ] Every citation resolves to a real `knowledge-base/references/{...}` path
- [ ] ≥1 ADR synthesizes M3 decisions; the internal-methodology deferral (D3) is explicit
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m3-realtime-alerting-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations (esp. no fake peer citation for alert precision — ADR D3)
- [ ] Coverage Matrix 100%
- [ ] ADRs cite `.claude/rules/architecture.md`, `.claude/rules/testing.md`
