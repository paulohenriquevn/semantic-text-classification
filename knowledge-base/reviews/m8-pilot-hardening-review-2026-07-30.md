# Review — M8 Pilot Hardening & V1 Ship

Date: 2026-07-30
Plan: `knowledge-base/plans/completed/m8-pilot-hardening-plan.md` (SHIPPABLE_WITH_CAVEATS 70)
Slice commits: Phase 0 `feat(api)`, Phase 1 `feat(api)`, Phases 2+Final `feat(pipeline)`
Reviewer: cycle self-review

## Scope reviewed (all ≤ 500 LoC)

| File | LoC | Verdict |
|---|---|---|
| `domain/health.py` (HealthReport/EngagementMetric) | 36 | OK |
| `domain/acceptance.py` (Criterion/ReadinessReport) | 54 | OK |
| `application/health_service.py` (readiness probe) | 49 | OK |
| `infrastructure/db_probe.py` (fail-soft DB ping) | 25 | OK |
| `infrastructure/engagement_repo.py` (acted_on_rate) | 37 | OK |
| `interface/app.py` (+/health,/ready,/dashboard/engagement) | +30 | OK |
| `experiments/scripts/v1_acceptance.py` (harness) | 78 | OK |
| `docs/runbook.md` (operational runbook) | doc | OK |

## Plan coverage (Coverage Matrix 6/6)

| # | Requirement (DoD) | Task | Status + evidence |
|---|---|---|---|
| 1 | Health self-monitoring | T0.1 | ✅ `/health` + `/ready`; `test_health_api` + `test_health_service` |
| 2 | Readiness surfaces degradation | T0.1 | ✅ 503 on db-down / not-listening / queue ≥ 90%; unit-proven |
| 3 | Alert-engagement tracked | T1.1 | ✅ `acted_on_rate` + `/dashboard/engagement`; `test_engagement` (0.5 on 2/4) |
| 4 | All V1 criteria pass, RE-RUN | T2.1/T3.1 | ✅ **real run `v1_readiness.json` verdict PASS** — retrieval p95 160.95ms, alert p95<2s, precision≥0.8, sentiment F1≥0.70, purge working |
| 5 | Operational runbook + recovery | T2.1 | ✅ `docs/runbook.md` (startup, probes, gate, recovery posture) |
| 6 | Honest synthetic-load caveat | T2.1/T3.1 | ✅ caveat embedded in the report + runbook |

## Global DoD

- [x] `ruff format --check` + `ruff check` + `mypy src/ tests/` clean
- [x] `pytest tests/unit` (2086 pass, 1 skip) + `pytest tests/integration` green
- [x] The V1-acceptance harness runs and emits a readiness report — a criterion FAIL fails the verdict (`test_missing_artifact_is_fail`, `test_report_fails_when_a_criterion_fails`)
- [x] `/ready` flips to 503 under queue saturation — `test_ready_false_when_queue_saturated`
- [x] File-size ≤ 500 LoC per file (largest 78)
- [x] The readiness report carries the synthetic-load caveat (no unqualified production claim)

## Findings

### F1 — the acceptance harness RE-RUNS, it does not hard-code — INFO (the M8 crux)
`v1_acceptance.py` sources retrieval-p95 from the fresh `m5_hybrid_bench.json` and RE-RUNS the actual M2/M3/M6 pytest checks (`test_turn_to_alert_p95_under_2s`, `test_alert_precision_meets_bar`, `test_macro_f1_meets_dod`, `test_rollup_survives_raw_chunk_drop`) — a failing/absent check yields `measured=0.0` → FAIL. `test_missing_artifact_is_fail` proves a missing measurement is never a silent PASS. This directly satisfies edge-case EC-1 (no acceptance theatre).

### F2 — readiness is a real probe over shipped state, not a stub — INFO
`/ready` reads `session.state`, `channel.qsize()`/`maxsize`, and a live `SELECT 1` (fail-soft to not-ready). The saturation + db-down + not-listening paths are each unit-tested. `PoolDbProbe` returns False on any exception rather than crashing the endpoint (fail-soft at the boundary — a probe reports, it does not raise).

### F3 — engagement is honestly framed as a proxy — INFO (accepted, EC-2)
`acted_on_rate` reuses the M5 labels (a label = acting on an alert) — DRY, no new instrumentation. Both the value-object docstring and the runbook state its production value awaits a real pilot. The plumbing is real and tested; the rate on synthetic data is not claimed as a production signal.

### F4 — V1 is claimed on synthetic load, honestly — INFO (accepted, EC-3 / Rule 3)
The readiness report's verdict is PASS, but the mandatory caveat ("validated on synthetic load; real 8 kHz drift NOT validated — re-run against real pilot data") is embedded in both the report and the runbook. No unqualified "production-ready" claim is made — the V1 ship is scoped to what the evidence supports.

### F5 — failure recovery is documented, not newly built — INFO (YAGNI)
The runbook documents the shipped mechanisms (bounded-channel backpressure, pool queueing, SSE auto-reconnect, export-before-purge, CA-outlives-raw) rather than adding a new circuit breaker. Per ADR D5 — the mechanisms exist and are tested (M0/M1/M4/M6/M7); M8 makes them operable + probe-observable.

No correctness, security, or resource findings.

## Verdict

**READY_TO_MERGE** — all 6 Coverage-Matrix requirements and the full Global DoD are met with real evidence. The V1-acceptance gate produces a **PASS** verdict on a live re-run of every ship criterion, with an honest synthetic-load caveat; the readiness probe is real and fail-soft; the engagement metric and recovery posture are honestly framed. This milestone closes the ROADMAP (M0–M8 all shipped). Findings are design notes and accepted honesty caveats; none blocking.
