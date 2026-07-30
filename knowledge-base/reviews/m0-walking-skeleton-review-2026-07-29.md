# Review — M0 Walking Skeleton

Date: 2026-07-29
Plan: `knowledge-base/plans/m0-walking-skeleton-plan.md`
Blueprint: `knowledge-base/discoveries/blueprints/m0-walking-skeleton-realtime-monitoring-blueprint.md`
Implementation commits: `a8c3105` → `cd57078` (workspace)

## DoD verification (evidence-backed)

| DoD | Status | Evidence |
|---|---|---|
| 1 — segment → Turn in hypertable w/ timestamp | ✅ | `test_ingest_to_supervisor_alert_e2e`, `test_save_turn_lands_in_hypertable_with_created_at` |
| 2 — segmentation + context window from live stream | ✅ | e2e posts a labeled transcript → `TurnSegmenter` → `enqueued≥1` → orchestrator builds windows |
| 3 — one DSL rule fires → alert with evidence | ✅ | `test_matching_turn_raises_alert_with_evidence`, e2e asserts `matched_text` evidence |
| 4 — supervisor page shows live alert | ✅ | e2e: alert id reaches a LISTEN subscriber (SSE data source); `GET /supervisor` returns 200 with `EventSource` |

## ADR compliance

| ADR | Status | Evidence |
|---|---|---|
| D1 — bounded channel backpressure | ✅ | `TurnChannel`; `test_put_blocks_when_full` |
| D2 — session state machine + two-phase drain | ✅ | `MonitoringSession`; `test_state_sequence_*`, `test_inflight_turns_processed_before_close` |
| D3 — commit-then-notify | ✅ | orchestrator saves alert then notifies (`orchestrator.py:94-95`); `test_turn_persisted_before_alert_and_notify` |
| D4 — hexagonal DIP | ✅ | domain+application import no psycopg/fastapi (grep clean); ports are `Protocol`s |
| D5 — per-Turn hypertable + JSONB evidence | ✅ | `TimescaleTurnRepository`; `test_alert_evidence_roundtrips_jsonb` |
| D6 — two-tier tests | ✅ | 18 unit + 6 integration |

## Quality gates

- Cyclomatic complexity: average **A (1.62)**, all 45 blocks grade A (target ≤10). `radon cc`.
- Dead code: none (`vulture --min-confidence 80`).
- Lint: `ruff check` clean; `ruff format --check` clean.
- Types: `mypy src/talkex/monitoring` clean (14 files).
- File-size budget: largest is `interface/app.py` at 137 LoC (≤500 budget ✅).
- Tests: 24 monitoring tests green; no baseline regression (1884 unit green).

## Divergences from plan (honest)

- **DV-1 — interface files consolidated.** The plan named `interface/ingest_api.py` + `interface/supervisor_sse.py`; the implementation consolidates both into `interface/app.py` (the composition root). Rationale: the ingest route, SSE route, and wiring are one cohesive interface-layer concern for the thin M0 slice; splitting added indirection without value (KISS). No behavior change. Accepted.

## Failure scenarios (plan `## Failure scenarios`)

| Scenario | Status | Evidence / decision |
|---|---|---|
| channel full → backpressure (no error, no OOM) | ✅ tested | `test_put_blocks_when_full`, `test_no_loss_under_many_producers` |
| DB connection reset mid-insert → fail-fast typed error | ✅ tested | `test_save_on_broken_connection_raises_not_silent` (raises `psycopg.OperationalError`) |
| LISTEN connection dropped → SSE reconnects | ⚠️ **deferred to M8 (pilot hardening)** | The SSE endpoint opens its own LISTEN connection; auto-reconnect-on-drop is a resilience-hardening concern beyond the M0 thin functional slice. Documented, not hidden. |

## Verdict

**READY_TO_MERGE** — M0 is functionally complete: all four DoDs proven end-to-end against a real
TimescaleDB, all six ADRs honored, quality gates green, one accepted divergence (DV-1) and one
explicitly-deferred resilience item (LISTEN reconnect → M8). No workarounds; no fabricated evidence.
