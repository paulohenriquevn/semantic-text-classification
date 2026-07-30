# Review — M6 Aggregated Dashboards (continuous aggregates)

Date: 2026-07-30
Plan: `knowledge-base/plans/completed/m6-aggregated-dashboards-plan.md` (SHIPPABLE_WITH_CAVEATS 70)
Slice commits: Phase 0 `feat(pipeline)`, Phases 1-2 `feat(api)`
Reviewer: cycle self-review

## Scope reviewed (all ≤ 500 LoC)

| File | LoC | Verdict |
|---|---|---|
| migration `0004_m6_kpi_rollup.sql` (CA + queue/sentiment cols + refresh policy, no CA retention) | 44 | OK |
| `domain/dashboard.py` (KpiQuery/KpiBucket) | 34 | OK |
| `infrastructure/kpi_repo.py` (CA query, bound whitelist filters) | 38 | OK |
| `domain/models.py` (Alert +queue +sentiment) | +2 fields | OK (additive/back-compat) |
| `infrastructure/timescale_repo.py` (alert INSERT/get +2 cols) | +4 | OK |
| `application/orchestrator.py` (`_emit_alert` populates queue+sentiment) | +6 | OK |
| `interface/app.py` (ingest queue, /dashboard/kpis) | 223 | OK |

## Plan coverage (Coverage Matrix 6/6)

| # | Requirement (DoD) | Task | Status + evidence |
|---|---|---|---|
| 1 | Incremental CA rollup, no full scan | T0.1 | ✅ `alerts_kpi_5min` CA + refresh policy; `test_incremental_refresh_bumps_bucket` (bucket += 1 on refresh) |
| 2 | queue/sentiment dimensions real | T0.1 | ✅ ingest→metadata→alert; `test_alert_dimensions` (queue promoted, sentiment promoted, honest 'default'/NULL) |
| 3 | Rollup retained beyond 30 days | T0.1/T3.1 | ✅ no CA retention (`test_no_retention_policy_on_ca`); **`test_rollup_survives_raw_chunk_drop`** — raw purged via drop_chunks, KPI bucket remains |
| 4 | KPI read | T1.1 | ✅ `KpiReadPort` + CA query; `test_kpi_read` (grouped by queue, filter narrows) |
| 5 | Dashboard renders trends | T2.1 | ✅ `GET /dashboard/kpis`; `test_dashboard_api` (buckets + queue filter) |
| 6 | Incremental refresh bumps rollup | T3.1 | ✅ `test_incremental_refresh_bumps_bucket` |

## Global DoD

- [x] `ruff format --check` + `ruff check` + `mypy src/ tests/` clean
- [x] `pytest tests/unit` (2060 pass, 1 skip) + `pytest tests/integration` green against real Timescale
- [x] CA increments incrementally (refresh bumps the bucket) — proven by a test
- [x] Rollup survives the raw chunk drop — proven by a `drop_chunks` test (the strongest M6 evidence)
- [x] File-size ≤ 500 LoC per file (largest 223)
- [x] Embeddings/dimensions honest — queue defaults to 'default' when ingest omits it; sentiment NULL when no detector

## Findings

### F1 — CA over `alerts` extends the shipped `turns_per_min` pattern — INFO (design strength, KISS)
M6 reuses the exact continuous-aggregate idiom shipped in M1 (`time_bucket`, `WITH (timescaledb.continuous)`, `WITH NO DATA`) rather than an app-level rollup table + listener (chatwoot's Rails path). The increment is DB-native — no reinvention (Rule 9). The edge-case EC-1 mapping (Ruby listener → Timescale CA) was honored.

### F2 — Rollup-outlives-raw is proven, not asserted — INFO (the M6 crux)
`test_rollup_survives_raw_chunk_drop` inserts into a 40-day-old chunk, materializes, `drop_chunks('alerts')` (raw count → 0), and asserts the KPI bucket still returns. This is the concrete proof of the ADR-005 hot/purge split — the exact synthesis point edge-case EC-2 flagged.

### F3 — "per queue" is genuinely populated end-to-end — INFO (avoided a workaround)
The baseline reality check (ingest didn't capture queue) was addressed by threading `queue` through ingest → `turn.metadata` → `alert.queue` → the CA dimension, not by hardcoding a constant. The default `'default'` is honest (documented), and the tests seed multiple real queues.

### F4 — sentiment promoted from evidence, not recomputed — INFO
The orchestrator already computed the sentiment label; M6 surfaces it as an `alerts.sentiment` column (accessed via the `EvidenceItem` TypedDict key). No new model inference on the write path.

### F5 — refresh cadence is policy-driven; tests force a full refresh — INFO, honest caveat (accepted)
Production refreshes via `add_continuous_aggregate_policy` (5-min schedule, closed-buckets-only). The integration tests call `refresh_continuous_aggregate(NULL,NULL)` to make assertions deterministic without waiting for the scheduler. The policy's existence + `end_offset` are set by the migration; the freshness SLO (≤ ~5 min) is documented, not measured live (would need a scheduler wait). Accepted.

No correctness, security, or resource findings.

## Verdict

**READY_TO_MERGE** — all 6 Coverage-Matrix requirements and the full Global DoD are met with real-DB evidence. The two hardest points — incremental refresh bumping the bucket, and the rollup surviving `drop_chunks` — are proven by tests, not asserted. Findings are design notes and one accepted refresh-cadence caveat; none blocking.
