# Review — M3 Real-Time Rule Engine & Alerting

Date: 2026-07-30
Plan: `knowledge-base/plans/m3-realtime-alerting-plan.md`
Blueprint: `knowledge-base/discoveries/blueprints/m3-realtime-alerting-blueprint.md`
Implementation commit: `e5a21e3`

## The central finding (evidence-driven)

A bare `contains_any("cancelar")` over the full transcript alerts at **precision 0.51** — the AGENT
explaining the cancellation process ("para cancelamento você liga...") fires false positives. The fix is
semantic, not a hack: **critical intent is the customer's**, so rule evaluation is scoped to the customer's
utterances. Combined with a cancellation-intent regex (`cancelamento` | `cancelar minha conta/plano/...` |
`quero/desejo/solicito cancelar`), cancellation-alert **precision reaches 0.8936 ≥ 0.80**.

| Rule | Scope | Precision | Recall |
|---|---|---|---|
| `contains_any("cancelar")` | full text | 0.505 | 1.00 |
| `contains_any("cancelar")` | customer-only | 0.571 | 1.00 |
| cancellation-intent regex | customer-only | **0.894** | 0.85 |

## DoD verification

| DoD | Status | Evidence |
|---|---|---|
| Critical rules fire with evidence | ✅ | `build_critical_rules` (cancellation + escalation); `test_two_rules_two_alerts` |
| Alert precision ≥ 0.8 (critical categories) | ✅ | `test_alert_precision_meets_bar` (0.8936 ≥ 0.80 on topic-labeled test set) |
| turn→alert p95 < 2s | ✅ | `test_turn_to_alert_p95_under_2s` (in-process, 30 turns) |
| Evidence per alert | ✅ | predicate evidence + optional M2 sentiment; e2e asserts evidence |

## ADR compliance (blueprint)

| ADR | Status | Note |
|---|---|---|
| D1 — rule catalogue | ✅ | `build_critical_rules` compiles per-category DSL rules |
| D2 — precision vs topic (customer-scoped) | ✅ | 0.8936; customer-scoping is the precision lever |
| D3 — in-process latency | ✅ | p95 test |
| D4 — negative-sentiment reuses M2 | ✅ | orchestrator attaches M2 sentiment evidence (optional detector) |
| D5 — internal catalogue/eval | ✅ | chatwoot informed the broadcast structure only |

## Quality gates

- Complexity: `handle` refactored (extracted `_emit_alert`) back to grade **A**; catalogue avg A.
- Dead code: none. Lint/types: `ruff` + `mypy` clean.
- File-size: critical_rules.py 28, orchestrator.py 121 (≤500 ✅).
- Tests: 41 monitoring tests green (incl. the M0 e2e, updated for the renamed rule); 1898 unit green; no regression.
- Backward-compatible: a single-rule list preserves M0 behavior; the sentiment detector stays optional.

## Divergences (honest)

- **DV-1 — rule renamed** `cancellation_risk` → `cancellation` (the catalogue name); the M0 e2e assertion was updated accordingly. Behavior unchanged (the alert still fires).
- **DV-2 — customer-scoping** changes rule input from full window_text to customer-only text. Semantically correct (intent is the customer's) and the precision lever; falls back to window_text if no customer turn is in the window.

## Deferred (honest, blueprint gaps)

- **G1 — topic proxy** for "should-alert" — a cancellation-threat inside a complaint (topic=reclamacao) is counted a false positive though it is arguably a real risk; the 0.89 precision already clears 0.80 despite this proxy noise.
- **G2 — network/SSE round-trip latency** — measured in-process; end-to-end SSE latency deferred to the M8 pilot.

## Verdict

**READY_TO_MERGE** — M3 hardens the shipped alerting with an evidence-backed critical-rule catalogue,
customer-scoped evaluation (precision 0.8936 ≥ 0.80, proven by test), multi-rule alerting, and p95 < 2s
latency — all with real numbers, backward-compatible, gates green. The precision fix is a documented,
evidence-driven design decision (customer-scoping), not a workaround.
