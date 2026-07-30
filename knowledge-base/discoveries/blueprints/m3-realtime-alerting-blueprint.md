# Blueprint: M3 Real-Time Rule Engine & Alerting

> **Version 1.0** — Locks the M3 hardening of the M0-shipped alerting: a critical-rule catalogue
> (cancellation, escalation, negative-sentiment — the last reusing M2's SentimentDetector), an
> alert-precision evaluation (≥ 0.8 on critical categories, using the internal `topic` labels as ground
> truth), and a turn→alert latency measurement (p95 < 2s). chatwoot supplies the event→broadcast structure
> (already borrowed in M0); the catalogue + precision/latency method are internal (ADR D5). Produced by
> `cycle-discover` execute from `m3-realtime-alerting-plan.md`.

**Slug:** `m3-realtime-alerting`
**Created:** 2026-07-30
**discover-confidence verdict:** recorded at the end after scoring.

## Executive summary

M3 does not rebuild alerting — M0 shipped the DSL rule engine, `LISTEN/NOTIFY` push, and SSE stream
(v0.2.0). M3 adds the missing hardening: a **catalogue** of critical rules (each a DSL expression producing
evidence), a **precision evaluation** proving critical alerts fire on the right windows (≥ 0.8, ground-truthed
by the internal `topic` labels), and a **latency measurement** (turn→alert p95 < 2s). The negative-sentiment
rule reuses M2's `SentimentDetector` in the cascade. chatwoot is the alerting-structure baseline; the
catalogue + evaluation are internal method (ADR D5).

## Context

`ROADMAP.md § M3` wants critical-event alerting with precision ≥ 0.8 and low latency. The infrastructure is
shipped (`src/talkex/monitoring/`, v0.2.0); M2 added the sentiment feature (v0.4.0). M3 composes them into a
rule catalogue + an evidence-backed precision/latency eval, within `.claude/rules/architecture.md` and
`.claude/rules/testing.md`.

## Objective

Lock, with cited evidence, the critical-rule catalogue, the alert-precision evaluation (≥ 0.8 vs topic
labels), and the turn→alert latency method — so M3 implementation hardens the shipped alerting without rework.

## Coverage Corner 1 — Integration Tests

chatwoot specs the alert channel at the subscribe/broadcast boundary with real fixtures:
`knowledge-base/references/chatwoot/spec/channels/room_channel_spec.rb:12` (subscription confirmed + streams
the expected topic). **M3 test decision:** reuse M0's channel/e2e tests (shipped) and ADD (a) an
alert-precision test — feed labeled windows through the catalogue and assert precision ≥ 0.8 on critical
categories against the `topic` ground truth; (b) a latency test asserting turn→alert elapsed < 2s in-process.

## Coverage Corner 2 — Dependencies

chatwoot uses Redis + Sidekiq for async broadcast fan-out at scale:
`knowledge-base/references/chatwoot/Gemfile:67` (redis), `:136` (sidekiq). **M3 deps decision:** M3 stays on
Postgres `LISTEN/NOTIFY` (shipped in M0, dependency-light, ADR-005 single-spine); Redis pub/sub is the scale
escape hatch if fan-out throughput ever demands it — no new dependency in M3.

## Coverage Corner 3 — Tools

chatwoot runs the real-time broadcast stack via docker-compose with a redis service:
`knowledge-base/references/chatwoot/docker-compose.yaml:97`. **M3 tools decision:** M3 needs NO new infra —
alerting runs on the existing Timescale `LISTEN/NOTIFY` (M0 compose); the precision/latency evals run in-process
against the internal corpus.

## Coverage Corner 4 — Techniques

chatwoot maps a domain event to an async broadcast on commit:
`knowledge-base/references/chatwoot/app/listeners/action_cable_listener.rb:41` (event→broadcast), `:222`
(async job fan-out), `:202` (de-duplicated token union); the channel streams a scope topic
`knowledge-base/references/chatwoot/app/channels/room_channel.rb:27`; the trigger is a commit callback
`knowledge-base/references/chatwoot/app/models/conversation.rb:134`. **M3 technique decision:** this pattern is
already implemented in M0 (orchestrator → alert → LISTEN/NOTIFY → SSE). M3 ADDS a **critical-rule catalogue**:
multiple DSL rules (cancellation, escalation, negative-sentiment) each producing evidence; the
negative-sentiment rule reuses M2's `SentimentDetector` as a cascade check (fire when label == negative).

## Cross-cutting Comparison

| Concern | chatwoot (peer) | M0/M2 (shipped) | M3 decision |
|---|---|---|---|
| Event→broadcast | listener→async job | orchestrator→LISTEN/NOTIFY→SSE | reuse (shipped); no rebuild |
| Rule set | n/a (no DSL) | one cancellation rule | catalogue: cancellation + escalation + negative-sentiment |
| Sentiment rule | n/a | M2 SentimentDetector | negative-sentiment fires an alert (reuse M2) |
| Precision eval | n/a | none | precision ≥ 0.8 vs `topic` labels (internal) |
| Latency | async (Sidekiq) | LISTEN/NOTIFY | measure turn→alert p95 < 2s |
| Scale backend | Redis + Sidekiq | Postgres LISTEN/NOTIFY | stay on LISTEN/NOTIFY; Redis = escape hatch |

## ADRs

### D1 — Critical-rule catalogue (DSL rules producing evidence)
A small catalogue: `cancellation` (contains_any cancelar/cancelamento), `escalation` (contains_any
supervisor/gerente/reclamação-formal), `negative_sentiment` (M2 detector == negative). **Rationale:** reuses
M0's DSL engine + M2's detector; each rule carries evidence. **Alternative rejected:** one mega-rule
(rejected — un-tunable, no per-category precision). **Consequence:** per-category precision is measurable.

### D2 — Alert-precision eval against the internal `topic` labels
Ground-truth: a window from a `cancelamento`/`reclamacao` conversation SHOULD alert for the matching rule;
precision = TP/(TP+FP) per critical category, target ≥ 0.8. **Rationale:** the `topic` field is a real label
(no new annotation). **Alternative rejected:** manual spot-check (rejected — not reproducible). **Consequence:**
the DoD is an automated precision test.

### D3 — Latency measurement (turn→alert p95 < 2s)
Measure in-process elapsed from `orchestrator.handle(turn)` start to alert emission over N turns; assert p95 < 2s.
**Rationale:** M0's cascade is in-process + LISTEN/NOTIFY (fast); prove it. **Alternative rejected:** wall-clock
over the SSE round-trip (rejected — flaky, network-dependent). **Consequence:** deterministic latency evidence.

### D4 — Negative-sentiment rule reuses M2 (no new model)
The `negative_sentiment` rule calls M2's `SentimentDetector` (already shipped, macro-F1 0.856). **Rationale:**
composition over duplication. **Alternative rejected:** a new sentiment rule predicate (rejected — duplicates M2).
**Consequence:** the sentiment cascade feeds alerting directly.

### D5 — Internal catalogue + eval, chatwoot = structure (honest gap)
The catalogue + precision/latency method are internal; chatwoot informs only the event→broadcast structure.
**Rationale:** Rule 3 — no fabricated peer citations for precision/latency. **Consequence:** blueprint synthesizes
internally.

## Recommendations

1. Add a `critical_rules` catalogue module (3 compiled DSL rules + the sentiment check) reusing M0's compiler.
2. Extend the orchestrator to evaluate the catalogue (not a single rule), emitting one alert per matched rule with evidence.
3. Add an `experiments/scripts/eval_alert_precision.py`: feed labeled windows, compute per-category precision vs `topic`, assert ≥ 0.8.
4. Add a latency test: N synthetic turns through `handle`, assert p95 elapsed < 2s.
5. Reuse M0's channel/e2e + M1's Timescale integration tiers (no new infra).

## Honest gaps

1. **G1 — precision ground-truth is topic-label-derived** — `topic` is a proxy for "should-alert"; edge cases (a resolved complaint) may be imperfectly labeled. Documented; the target ≥ 0.8 tolerates some proxy noise.
2. **G2 — latency measured in-process** — excludes network/SSE round-trip (that is a separate, environment-dependent measurement deferred to the M8 pilot).

## discover-confidence verdict

Scored below via `run_blueprint_score.py`; verdict recorded after the run.
