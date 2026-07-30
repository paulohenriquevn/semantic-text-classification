# Plan: M3 Real-Time Rule Engine & Alerting

> **Version 1.0** — Harden the M0-shipped alerting with a critical-rule catalogue (cancellation, escalation,
> negative-sentiment reusing M2), evaluate alert precision ≥ 0.8 on critical categories against the internal
> `topic` labels, and measure turn→alert latency (p95 < 2s in-process). Reuses M0's DSL engine + LISTEN/NOTIFY
> and M2's SentimentDetector — no rebuild, no new infra.

## Goal

> "Enable the monitoring cascade to fire evidence-backed critical alerts so that a cancellation/escalation/
> negative window is flagged, measured by `test_alert_precision_meets_bar` passing (per-category alert
> precision ≥ 0.8 on the internal topic-labeled test set)."

## Context

M3 (`ROADMAP.md § M3`) composes shipped pieces: M0's DSL rule engine + LISTEN/NOTIFY alerting (v0.2.0) and
M2's SentimentDetector (v0.4.0). It adds a critical-rule catalogue, a precision eval, and a latency
measurement, per the SHIPPABLE 98.3 blueprint `knowledge-base/discoveries/blueprints/m3-realtime-alerting-blueprint.md`.
Constrained by `.claude/rules/architecture.md` + `.claude/rules/testing.md`.

## Baseline Context (deep review of current state)

### Files that will be touched

| File | LoC today | Last commit (sha + date) | Why it exists today | Invariants to preserve |
|---|---|---|---|---|
| `src/talkex/monitoring/application/orchestrator.py` | 108 | `1382fde` (2026-07-30) | cascade: turn→window→rule→alert (+ optional sentiment) | evaluate a rule CATALOGUE, keep single-rule behavior for M0 e2e |
| `src/talkex/monitoring/domain/critical_rules.py` (NEW) | 0 | — | compiled critical-rule catalogue | — |
| `experiments/scripts/eval_alert_precision.py` (NEW) | 0 | — | per-category precision vs topic labels | — |
| `tests/unit/monitoring/test_critical_rules.py` (NEW) | 0 | — | catalogue + precision + latency tests | — |

### Current callers / dependents

- **Symbol:** `TurnOrchestrator` (`orchestrator.py`) — Callers: `interface/app.py`, tests. M3 changes it to accept a rule catalogue (backward-compatible: a single-rule list preserves M0 behavior).
- New `critical_rules.py`: first-of-its-kind.

### Domain glossary

- **Critical rule** — a DSL rule whose match warrants a supervisor alert (cancellation/escalation/negative).
- **Alert precision** — TP/(TP+FP) per critical category; ground-truthed by the `topic` label.
- **p95 latency** — 95th percentile of turn→alert elapsed time.

### Architecture boundaries affected

`critical_rules.py` is a domain catalogue; the orchestrator (application) consumes it; no new infra boundary.

## Prior Art & Related Work

- **Internal blueprint** — `m3-realtime-alerting-blueprint.md` §"Coverage Corner 4" + its ADR set.
- **Reference — chatwoot** — event→broadcast `knowledge-base/references/chatwoot/app/listeners/action_cable_listener.rb:41`.
- **Internal (shipped)** — M0 DSL engine + LISTEN/NOTIFY; M2 `src/talkex/classification/sentiment.py`.

## Objective

- [ ] `critical_rules` catalogue (cancellation, escalation, negative-sentiment)
- [ ] Orchestrator evaluates the catalogue, one evidence-backed alert per matched rule
- [ ] Alert precision ≥ 0.8 per critical category on the internal topic-labeled set (automated test)
- [ ] turn→alert latency p95 < 2s (automated test); M0 e2e unchanged

## ADRs

### D1 — Rule catalogue over a single rule
- **Decision:** the orchestrator evaluates a list of compiled critical rules.
- **Rationale:** per-category precision needs distinct rules (blueprint D1).
- **Alternatives considered:** one mega-rule (rejected — un-tunable, no per-category precision).
- **Consequence:** each category is independently measurable; a single-item list preserves M0 behavior.

### D2 — Precision ground-truthed by the internal `topic` label
- **Decision:** a `cancelamento`/`reclamacao` window SHOULD alert; precision measured per category, target ≥ 0.8.
- **Rationale:** `topic` is a real label (no new annotation).
- **Alternatives considered:** manual spot-check (rejected — not reproducible).
- **Consequence:** the DoD is an automated precision test.

### D3 — In-process latency measurement
- **Decision:** measure elapsed inside `handle` over N turns; assert p95 < 2s.
- **Rationale:** the cascade is in-process; deterministic, no network flakiness.
- **Alternatives considered:** SSE round-trip timing (rejected — environment-dependent).
- **Consequence:** deterministic latency evidence; network round-trip deferred to M8 pilot.

## Drawbacks & Risks

| Drawback / Risk | Severity | Mitigation | Owner |
|---|---|---|---|
| R1 — `topic` is a proxy for "should-alert"; some labels imperfect | Medium | target ≥ 0.8 tolerates proxy noise; documented gap G1 | dev |
| R2 — evaluating N rules per window adds latency | Low | catalogue is small (3 rules); latency test guards p95 < 2s | dev |

## Unresolved Questions

- Q1 — Exact escalation lexicon (which terms signal escalation)? M3 uses `supervisor`/`gerente`/`reclamação formal`; tunable.
- Q2 — Should negative-sentiment always alert or only above a margin threshold? M3: alert on label==negative; margin threshold deferred.

## Dependency Graph

```
Phase 0 (critical_rules catalogue)  ──▶  Phase 1 (orchestrator evaluates catalogue)
                                                 ▼
                                   Phase 2 (precision + latency eval/tests)
                                                 ▼
                                     Final: Integration Validation
```

---

## Phase 0: Critical-rule catalogue

**Objective:** the compiled critical rules.

### T0.1 — critical_rules catalogue

#### Objective
A module exposing compiled cancellation/escalation rules (+ the negative-sentiment marker).

#### Why this step
1. **What:** `src/talkex/monitoring/domain/critical_rules.py` — compile the DSL rules via M0's compiler.
2. **Why now:** the catalogue is M3's core (blueprint D1).

#### Files to edit
```
src/talkex/monitoring/domain/critical_rules.py (NEW) — build_critical_rules() -> list[RuleDefinition]
tests/unit/monitoring/test_critical_rules.py (NEW) — catalogue + precision + latency
```

#### TDD
```
RED:   test_catalogue_has_critical_rules — build_critical_rules() returns >= 2 named rules
GREEN: implement the catalogue
VERIFY: pytest tests/unit/monitoring/test_critical_rules.py -k catalogue -q
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `test_catalogue_has_critical_rules` asserts the catalogue contains named `cancellation` and `escalation` rules
- [ ] `ruff check` and `mypy` report zero warnings/errors on `critical_rules.py`

#### DoD
- [ ] catalogue test passes

---

## Phase 1: Orchestrator evaluates the catalogue

**Objective:** one evidence-backed alert per matched critical rule.

### T1.1 — evaluate the catalogue in handle()

#### Objective
Change the orchestrator to accept a list of rules and emit one alert per match.

#### Why this step
1. **What:** `orchestrator.py` accepts `rules: list[RuleDefinition]` (a single-item list preserves M0).
2. **Why now:** per-category alerts (blueprint D1); backward-compatible.

#### Files to edit
```
src/talkex/monitoring/application/orchestrator.py — evaluate a rule list; one alert per matched rule
tests/unit/monitoring/test_critical_rules.py — test_two_rules_two_alerts
```

#### TDD
```
RED:   test_two_rules_two_alerts — a turn matching two rules yields two alerts with distinct rule_name
GREEN: iterate rules in handle()
VERIFY: pytest tests/unit/monitoring/test_critical_rules.py -q
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `test_two_rules_two_alerts` asserts two alerts with the two distinct rule names when a turn matches both
- [ ] existing single-rule orchestrator tests still pass (M0 e2e unchanged)
- [ ] `ruff` + `mypy` clean on `orchestrator.py`

#### DoD
- [ ] catalogue evaluation tests pass; M0 e2e unaffected

---

## Phase 2: Precision + latency evaluation

**Objective:** prove precision ≥ 0.8 and latency p95 < 2s.

### T2.1 — alert precision + latency tests

#### Objective
Measure per-category alert precision (vs `topic`) and turn→alert p95 latency.

#### Why this step
1. **What:** `experiments/scripts/eval_alert_precision.py` + tests in `test_critical_rules.py`.
2. **Why now:** the M3 DoDs (blueprint D2/D3).

#### Files to edit
```
experiments/scripts/eval_alert_precision.py (NEW) — per-category precision vs topic labels
tests/unit/monitoring/test_critical_rules.py — test_alert_precision_meets_bar, test_latency_p95
```

#### TDD
```
RED:   test_alert_precision_meets_bar — cancellation-rule precision on `cancelamento` windows >= 0.8 (topic ground truth)
RED:   test_latency_p95 — N turns through handle; p95 elapsed < 2.0s
GREEN: run the catalogue over labeled windows; time handle()
VERIFY: pytest tests/unit/monitoring/test_critical_rules.py -q
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `test_alert_precision_meets_bar` asserts per-category precision ≥ 0.8 on the labeled set
- [ ] `test_latency_p95` asserts p95 turn→alert elapsed < 2.0 seconds
- [ ] `ruff` + `mypy` clean

#### DoD
- [ ] precision + latency tests pass

---

## Coverage Matrix

| # | Gap / Requirement | Task(s) | Resolution |
|---|---|---|---|
| 1 | Critical-rule catalogue | T0.1 | build_critical_rules() |
| 2 | One alert per matched rule | T1.1 | orchestrator iterates the catalogue |
| 3 | Alert precision ≥ 0.8 | T2.1 | precision test vs topic labels |
| 4 | turn→alert p95 < 2s | T2.1 | latency test |
| 5 | Backward compatibility (M0) | T1.1 | single-item rule list preserves M0 |

**Coverage: 5/5 gaps covered (100%)**

## Global Definition of Done

- [ ] `pytest tests/unit/monitoring -q` green (incl. precision + latency)
- [ ] `mypy` + `ruff` clean
- [ ] File-size ≤ 500 LoC per file
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] M0 e2e unchanged (backward-compatible)
- [ ] Runtime-metric proof — the precision + latency tests observe real numbers, not just compile

## Failure scenarios (external I/O)

```
(none — no external I/O touched; the catalogue eval + latency run in-process over local data)
```

## Final Phase: Integration Validation (MANDATORY)

**Objective:** prove the M3 DoDs + no regression.

### Execution
```
pytest tests/unit/monitoring -q
mypy src/talkex/monitoring ; ruff check src/talkex tests
```

### Acceptance Criteria
- [ ] `test_alert_precision_meets_bar` + `test_latency_p95` green (the Goal metrics)
- [ ] M0 monitoring + e2e tests green (no regression)
- [ ] zero type/lint errors
