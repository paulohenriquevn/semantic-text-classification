# Discover Edge Case Review — M6 Aggregated Dashboards

Date: 2026-07-30
Discovery plan analyzed: knowledge-base/discoveries/plans/m6-aggregated-dashboards-plan.md
Research questions analyzed: 7
Edge cases found: 3 (MUST FIX: 0, SHOULD TEST: 2, DOCUMENT: 1)

## MUST FIX

(none — all 7 questions map to a method and every cited path was verified to exist before the plan was written)

## SHOULD TEST

### EC-1: chatwoot rollup is a Rails/ActiveRecord model — translate the pattern, not the tech
- **Affected question:** Q1, Q3, Q6
- **Family:** Interpretation
- **Scenario:** chatwoot's `reporting_events_rollup` is an application-level rollup (a table + a listener that increments on event creation, driven by Sidekiq/ActiveRecord). M6 uses a Timescale **continuous aggregate** (DB-native incremental materialization) — a different mechanism for the same goal. The execute must extract the *pattern* (incremental increment keyed by (bucket, dimension), no full scan) and map it to a CA, NOT propose porting a Rails listener.
- **Suggested halt-loop checkpoint:** before marking Q1/Q6 done, assert the blueprint's recommendation names the Timescale CA mechanism (`time_bucket` + `WITH (timescaledb.continuous)` + `refresh_continuous_aggregate`/policy) as the M6 realization of the borrowed increment pattern — not a Ruby listener.

### EC-2: "rollup retained beyond 30 days while raw purges" is a CA-retention nuance — verify it is addressed
- **Affected question:** Q1, Q3
- **Family:** Coverage
- **Scenario:** The M6 DoD requires the rollup to survive the 30-day raw purge. In Timescale this means NOT adding a retention policy to the CA (or a longer one) even though `turns`/`alerts` have a 30-day retention. Neither chatwoot nor ai-powered will state this Timescale-specific fact — it is a synthesis point the blueprint must record in an ADR, not leave to the reader.
- **Suggested halt-loop checkpoint:** before promising complete, assert the blueprint has an ADR explicitly stating the CA retention posture (rollup outlives raw) with the ADR-005 hot/purge split as the rationale.

## DOCUMENT

### EC-3: ai-powered's DuckDB path is a contrast, not a template
- **Accepted risk:** ai-powered aggregates in a separate DuckDB (columnar) store. M6 deliberately rejects a separate store (single-engine, ADR-005). Q2/Q4 exist to CONTRAST, not to adopt — the blueprint should frame ai-powered as "why NOT a separate columnar store for a 30-day window" rather than a pattern to copy. No plan change needed; the framing is the execute's job.

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1 | 2 | 0 | 2 | 0 |
| Q2 | 1 | 0 | 0 | 1 |
| Q3 | 1 | 0 | 1 | 0 |
| Q4 | 0 | 0 | 0 | 0 |
| Q5 | 0 | 0 | 0 | 0 |
| Q6 | 1 | 0 | 1 | 0 |
| Q7 | 0 | 0 | 0 | 0 |

**Verdict:** DISCOVERY PLAN OK — the 2 SHOULD-TEST items are execution-time synthesis assertions (map the increment pattern to a Timescale CA; record the rollup-outlives-raw retention posture in an ADR), not plan-structure defects. No v1.1 bump required; the checkpoints are carried into execution.
