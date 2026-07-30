# Blueprint: M6 Aggregated Dashboards

**Slug:** `m6-aggregated-dashboards`
**Date:** 2026-07-30
**Plan reference:** `knowledge-base/discoveries/plans/m6-aggregated-dashboards-plan.md`
**Edge-cases reference:** `knowledge-base/reviews/m6-aggregated-dashboards-edge-cases-2026-07-30.md`

## Executive summary

M6 must serve manager dashboards from Timescale **continuous aggregates** (CAs): business KPIs
(cancellation/escalation rate per queue per 5 min, sentiment trend, script adherence) that (a) refresh
INCREMENTALLY with no full re-scan, (b) survive the 30-day raw purge as long-term business memory, and
(c) back a dashboard read with honest freshness. The two peers answer this from opposite ends:

- **chatwoot** ships a production incremental reporting-rollup subsystem. Its core lesson is the *pattern*,
  not the tech: a rollup row keyed by `(date, dimension_type, dimension_id, metric)` that a **DB upsert
  increments** (`count = existing + EXCLUDED`) on each new event — never a full re-scan
  (`app/services/reporting_events/rollup_service.rb:53-79`). Its read path groups pre-aggregated rows by
  period behind a report builder (`app/builders/v2/report_builder.rb:18-36,103-112`), and it proves the
  increment against a **real DB with factories** (`spec/listeners/reporting_event_listener_spec.rb:13-19`).
- **ai-powered-call-center-intelligence** is the **contrast**: it aggregates in a *separate* DuckDB
  columnar file (`analytics/duckdb_loader.py:11-25`) via in-memory pandas `value_counts` full scans
  (`analytics/powerdash_components.py:27,43`). This is exactly the second-store / full-scan design M6
  rejects (ADR-005 single-engine), so it validates — by counter-example — the Timescale-CA choice.

The synthesis (both peers are silent on the Timescale specifics): map chatwoot's `(bucket, dimension)`
incremental-increment pattern onto a `time_bucket('5 minutes', …) WITH (timescaledb.continuous)` CA
(extending the shipped `turns_per_min` CA at `deploy/monitoring/migrations/0002_m1_retention_indexes.sql:44-52`);
refresh it with a continuous-aggregate policy whose `end_offset` excludes the still-forming bucket
(the Timescale realization of chatwoot's "closed days only" backfill posture at
`knowledge-base/references/chatwoot/lib/tasks/reporting_events_rollup.rake` header, lines 4-6 of the printed plan);
and — critically — attach **NO retention policy** to the CA so the rollup outlives the raw purge
(ADR-005 hot/purge split).

## Context

M1 already shipped ONE continuous aggregate (`turns_per_min`, per-minute turn count) plus 30-day retention
on `turns`/`alerts` and a `src/talkex/analytics/` module. M6 closes the gap of INCREMENTAL business
rollups: how a new alert bumps a KPI counter (not a re-scan), how the rollup persists past the raw purge,
what KPIs are operationally meaningful, and how the increment is tested. This is a data-governance decision
(tomas-herrera's domain), respecting `.claude/rules/architecture.md § 1` (the dashboard read is an
infrastructure adapter behind a domain port) and `.claude/rules/testing.md § 2` (rollup correctness proven
against a real DB).

The two in-scope peers were read end-to-end (Ruby files ≤ 187 LoC each, Python files ≤ 73 LoC). All cited
paths were verified to exist before reading; nothing in this blueprint is inferred beyond a read line.

## Objective

Lock the M6 CA architecture from peer evidence: (1) CA definitions + dimensions, (2) refresh/retention
policy, (3) KPI set, (4) dashboard query shape. Each is decided in an ADR below with the peer citation that
grounds it and the Timescale synthesis that neither peer states.

## Coverage Corner 1 — Integration Tests

**Q5 — How does chatwoot TEST that a rollup increments correctly (real DB, factories, the listener path)?**

chatwoot proves rollup correctness at two tiers, both against a **real DB** (RSpec + FactoryBot, not stubs):

| Tier | Spec | DB posture | What it asserts | Citation |
|---|---|---|---|---|
| Listener → event creation | `reporting_event_listener_spec.rb` | Real DB; `create(:account/:user/:inbox/:conversation/:message)` factories | Firing a domain event through the listener writes exactly one raw reporting event: count `0 → 1` | `knowledge-base/references/chatwoot/spec/listeners/reporting_event_listener_spec.rb:13-19` |
| Rollup failure isolation | `reporting_event_listener_spec.rb` (context "when rollup creation fails") | Real DB; `RollupService.perform` stubbed to raise | The raw event is still persisted (count `= 1`) and the error is captured, not propagated | `…/spec/listeners/reporting_event_listener_spec.rb:21-36` |
| Idempotent event (bot handoff) | `reporting_event_listener_spec.rb` (`#conversation_bot_handoff`) | Real DB | Firing the same event twice yields count `= 1` (dedup) | `…/spec/listeners/reporting_event_listener_spec.rb:271-282` |
| Rollup model contract | `reporting_events_rollup_spec.rb` | Real DB; `create(:reporting_events_rollup, …)` | Scopes (`for_date_range`, `for_dimension`, `for_metric`) return the right subset; enum values persist as strings; all columns present | `knowledge-base/references/chatwoot/spec/models/reporting_events_rollup_spec.rb:83-181` |

Key observations for M6's test tier:
- The **increment** itself is asserted at the seam of "fire event → count changes" (`:14-19`), i.e. a
  behavioral before/after count, not an internal-structure assertion. This is the pattern to copy: seed a
  known set of alerts/turns, refresh the CA, assert the bucket count/avg equals the hand-computed expected.
- The **failure-isolation** test (`:21-36`) encodes a fail-soft policy: a rollup write failure must not lose
  the source event. For M6 the analogue is: a CA refresh failure must not block ingest — the CA is derived,
  the raw hypertable is the source of truth.
- All setup uses `let`/factories with **no shared mutable helpers** (chatwoot `CLAUDE.md`: "avoid custom
  helper methods for setup"), matching `.claude/rules/testing.md § 3` (independent tests).

**M6 integration-test shape (synthesis):** a `tests/integration/` test on a real TimescaleDB — insert N
`alerts` rows across two 5-min buckets and two queues, run `CALL refresh_continuous_aggregate(...)`, then
`SELECT` from the CA and assert the per-(bucket, queue) counts equal the seeded distribution. A second test
asserts the rollup survives a simulated raw purge (drop the raw chunk, re-query the CA, rows still present).

## Coverage Corner 2 — Dependencies

**Q3 — What does chatwoot depend on to STORE + SCHEDULE its rollups (table/model, job/cron)?**

Storage is a **plain relational table** `reporting_events_rollups`, defined by an ActiveRecord migration and
modelled by `ReportingEventsRollup`:

| Concern | Evidence | Citation |
|---|---|---|
| Table columns | `account_id, date, dimension_type, dimension_id, metric, count(bigint, default 0), sum_value(float), sum_value_business_hours(float)` | `knowledge-base/references/chatwoot/db/migrate/20260211145813_create_reporting_events_rollup.rb:10-21` |
| Grain / uniqueness | UNIQUE `(account_id, date, dimension_type, dimension_id, metric)` — the rollup key | `…/db/migrate/20260211145813_create_reporting_events_rollup.rb:25-27`; mirrored in model header `index_rollup_unique_key` at `app/models/reporting_events_rollup.rb:21` |
| Read indexes | `index_rollup_timeseries (account_id, metric, date)` + `index_rollup_summary (account_id, dimension_type, date)` | `…/db/migrate/20260211145813_create_reporting_events_rollup.rb:29-35` |
| Dimensions & metrics as enums | `dimension_type ∈ {account, agent, inbox, team}`; `metric ∈ {resolutions_count, first_response, resolution_time, reply_time, bot_resolutions_count, bot_handoffs_count}` | `app/models/reporting_events_rollup.rb:28-36` |
| Raw source table | `reporting_events (name, value, account_id, conversation_id, inbox_id, user_id, event_start/end_time)` | `knowledge-base/references/chatwoot/app/models/reporting_event.rb:1-56` |
| Backfill / schedule | Rake task `reporting_events_rollup:backfill` — "closed days only (today is skipped by default)", then "enable read path" | `knowledge-base/references/chatwoot/lib/tasks/reporting_events_rollup.rake` (namespace + `ReportingEventsRollupBackfill#run`, printed-plan lines 3-4) |

**Dependency implication for M6:** chatwoot needs an application table + a Sidekiq/ActiveRecord upsert path +
a manual rake backfill because Rails has no DB-native incremental materialization. **M6 needs none of that**:
TimescaleDB's continuous aggregate is the table, the incremental engine, and the scheduler in one. The M6
"dependency" is TimescaleDB's `timescaledb.continuous` materialized view + `add_continuous_aggregate_policy`
(the same extension M1 already depends on — `deploy/monitoring/migrations/0002_m1_retention_indexes.sql:44-52`).
No new dependency is added (KISS / Rung 4 of `.claude/rules/parsimony-ladder.md` — reuse the installed engine).

**Q4 — What engine does ai-powered use for dashboard aggregation, and what does that imply vs a Timescale CA?**

Engine verdict: **DuckDB (a separate, on-disk columnar store) + pandas in-memory**.

- It opens a *distinct* database file `data/call_summary.db` and `CREATE TABLE IF NOT EXISTS calls`
  (`knowledge-base/references/ai-powered-call-center-intelligence/analytics/duckdb_loader.py:11-25`), then
  ingests per-call rows with `INSERT` (`…/duckdb_loader.py:38-43`) and exposes an ad-hoc `query(sql)` that
  returns a DataFrame (`…/duckdb_loader.py:46-48`).
- Its dashboard "aggregation" is **pandas `value_counts()` over a DataFrame** — a full in-memory scan each
  render (`…/analytics/powerdash_components.py:27` for issue-category counts, `:43` for resolution-tactic
  counts), charted with Altair.

Implication for M6's CA-vs-columnar decision (EC-3 — contrast, not template): the DuckDB path is a
*second engine* that recomputes aggregates by full scan on demand over a per-call export. For a live,
continuously-updated 30-day dashboard that is precisely the anti-pattern ADR-005 rejects: it duplicates the
data into a separate store and re-scans instead of incrementally materializing. M6 keeps ONE engine
(Timescale) and pushes the aggregation into a CA. **DuckDB validates the CA choice by being the counter-example.**

## Coverage Corner 3 — Tools

**Q6 — How does chatwoot SCHEDULE/refresh its rollups, and at what cadence?**

The refresh is **event-driven, synchronous within the listener, best-effort**:

| Trigger event | Rollup write | Sync / deferred | Citation |
|---|---|---|---|
| `conversation_resolved` | `safe_rollup(reporting_event)` after `reporting_event.save!` | Synchronous in the listener, wrapped for failure isolation | `knowledge-base/references/chatwoot/app/listeners/reporting_event_listener.rb:23-24` |
| `first_reply_created` | `safe_rollup(...)` | Synchronous | `…/app/listeners/reporting_event_listener.rb:45-46` |
| `reply_created` | `safe_rollup(...)` | Synchronous | `…/app/listeners/reporting_event_listener.rb:70-71` |
| `conversation_bot_handoff` (dedup-guarded) | `safe_rollup(...)` | Synchronous | `…/app/listeners/reporting_event_listener.rb:97-98` |
| — the write itself | `RollupService.perform` → `upsert_all(..., on_duplicate: 'count = count + EXCLUDED.count …')` | DB upsert, one round-trip, no scan | `…/app/services/reporting_events/rollup_service.rb:53-79` |

Cadence & posture:
- **Per-event, not batched:** each new event bumps the rollup immediately. There is no cron; the "schedule"
  is the event stream itself.
- **Read-path is separately toggled:** the rollup is *always collected* for accounts with a valid reporting
  timezone, but a feature flag controls only whether reports *read* rollups vs raw events
  (`…/app/services/reporting_events/rollup_service.rb:22-27`). This is the "collect early, cut read over
  later" migration posture — mirrored by the rake backfill's step 4 "verify parity, then enable read path".
- **Closed-buckets-only for backfill:** the backfill task skips *today* by default and rolls up closed days
  only (rake header printed-plan line 3), because an in-progress bucket is not yet final.

**M6 refresh policy (synthesis — the Timescale realization):**
- Use `add_continuous_aggregate_policy(<ca>, start_offset => INTERVAL '2 hours', end_offset => INTERVAL '5 minutes', schedule_interval => INTERVAL '5 minutes')`.
- `schedule_interval` = 5 min matches the bucket width (chatwoot's "per-event" becomes "per-bucket-close").
- `end_offset` = 5 min is the direct analogue of chatwoot's "today is skipped" — **never materialize the
  still-forming bucket**; only refresh buckets that have closed.
- Timescale's incremental refresh reads only the invalidated ranges (buckets touched by new rows), so a new
  alert bumps its bucket without re-scanning the hypertable — the DB-native equivalent of the
  `count = count + EXCLUDED.count` upsert. **EC-1 satisfied:** the borrowed pattern is realized as a CA
  policy, not a ported Ruby listener.

**Q7 — How does chatwoot's report builder QUERY the rollups to serve a dashboard?**

`V2::ReportBuilder` is the read/consumption boundary:

| Aspect | Behavior | Citation |
|---|---|---|
| Entry points | `timeseries` (dispatches to a per-metric method), `build` (shapes `{value, timestamp, count}`), `summary`/`short_summary`/`bot_summary` (scalar cards) | `knowledge-base/references/chatwoot/app/builders/v2/report_builder.rb:18-63` |
| Grouping dimension (time) | `group_by_period(params[:group_by] || 'day', :created_at, range:, permit: %w[day week month year hour], time_zone:)` | `…/app/builders/v2/report_builder.rb:103-112` |
| Grouping dimension (entity) | Filtered per `inbox` / `user` / `label` / `team` via `account.<assoc>.find(params[:id])` | `…/app/builders/v2/report_builder.rb:87-101` |
| Time range | `range` (from `DateRangeHelper`), timezone-aware via `timezone_offset` | `…/app/builders/v2/report_builder.rb:13-16,108` |
| Metric whitelist | `metric_valid?` gate before dispatch (rejects unknown metrics) | `…/app/builders/v2/report_builder.rb:75-85` |
| Default grain | `DEFAULT_GROUP_BY = 'day'`, agents paginated 25/page | `…/app/builders/v2/report_builder.rb:7-8` |

**M6 dashboard query shape (synthesis):** the builder's structure maps cleanly onto a CA read —
`group_by_period(created_at)` becomes "the CA already bucketed at `time_bucket('5 minutes')`", so the M6
dashboard read is a `SELECT bucket, dimension, metric FROM <ca> WHERE bucket BETWEEN :from AND :to AND
dimension = :x ORDER BY bucket` — no live grouping, because the bucket is materialized. The entity filter
(inbox/user) becomes the M6 dimension filter (queue/domain/rule_name). The **metric whitelist** (`:75-85`)
is a pattern to keep: validate the requested KPI against an enum before building SQL. This read is served
behind a domain port per `.claude/rules/architecture.md § 1`, reusing the existing `AnalyticsQueryRunner`
consumption boundary (`src/talkex/analytics/query_runner.py` docstring: "the query runner is the consumption
boundary") rather than letting the API touch SQL directly.

## Coverage Corner 4 — Techniques

**Q1 — How does chatwoot compute reporting rollups INCREMENTALLY (a new event bumps a counter) rather than re-scanning?**

The increment is a **single DB upsert with an additive `on_duplicate` clause** — the crux of the whole
subsystem. Decomposition:

1. **Event → increment.** The listener saves the raw event, then calls `RollupService.perform`
   (`app/listeners/reporting_event_listener.rb:23-24`). The service builds one row per `(dimension, metric)`
   from *this event only* (`app/services/reporting_events/rollup_service.rb:41-51`) — it never queries the
   raw table to recompute.
2. **Increment mechanism.** `upsert_all(rows, unique_by: [:account_id, :date, :dimension_type, :dimension_id,
   :metric], on_duplicate: 'count = reporting_events_rollups.count + EXCLUDED.count, sum_value = … + EXCLUDED.sum_value, …')`
   (`…/rollup_service.rb:53-79`). On key collision Postgres **adds** the new event's contribution to the
   existing row; on a new key it inserts. **No `SELECT` over raw events, no full scan** — O(1) per event.
3. **Bucket key.** `(account_id, date, dimension_type, dimension_id, metric)` — a daily bucket (`date`) ×
   dimension × metric (`…/rollup_service.rb:63-70`, `event_date` at `:29-31` truncates the event timestamp to
   the account's reporting-timezone date).
4. **Multi-dimension fan-out.** One event increments the `account`, `agent`, and `inbox` rows in the same
   upsert batch (`…/rollup_service.rb:33-39,44-45`) — each dimension is a separate rollup row sharing the count.

**Mapping to M6 (EC-1 — extract the pattern, not the tech):**

| chatwoot (app-level) | M6 (Timescale CA, DB-native) |
|---|---|
| `date` truncation (`event_date`, `:29-31`) | `time_bucket('5 minutes', created_at) AS bucket` |
| `(dimension_type, dimension_id, metric)` key | `GROUP BY bucket, queue, rule_name` (dimensions from `alerts.rule_name`, `turns.metadata` JSONB) |
| `count = count + EXCLUDED.count` upsert | `count(*)` / `avg(...)` inside a `WITH (timescaledb.continuous)` CA, refreshed incrementally over invalidated buckets |
| `safe_rollup` on each event | `add_continuous_aggregate_policy` (schedule 5 min, `end_offset` skips the open bucket) |
| Manual `backfill` rake for history | `CALL refresh_continuous_aggregate(<ca>, <start>, <end>)` one-off for backfill; `WITH NO DATA` at create (M1 precedent `0002_…:51`) |

The pattern is: **keyed incremental accumulation over (bucket, dimension), never a re-scan.** chatwoot does
it with an app-level additive upsert; M6 does it with a continuous aggregate. The blueprint's recommendation
therefore names `time_bucket` + `WITH (timescaledb.continuous)` + `add_continuous_aggregate_policy` /
`refresh_continuous_aggregate` as the M6 realization — **not** a ported Ruby listener.

**Q2 — How does ai-powered aggregate call data for dashboard components?**

ai-powered aggregates **in Python/pandas at chart-build time**, per component:

| Dashboard component | Aggregation | Grouping | Engine | Citation |
|---|---|---|---|---|
| Sentiment trend | none (raw line) | per `call_id` over `utterance_index` | Altair over a DataFrame | `knowledge-base/references/ai-powered-call-center-intelligence/analytics/powerdash_components.py:10-22` |
| Issue-category distribution | `df['issue_category'].value_counts()` | by category | pandas in-memory full scan | `…/analytics/powerdash_components.py:25-39` (`:27`) |
| Resolution-tactic usage | `df['resolution_tactic'].value_counts()` | by tactic | pandas in-memory full scan | `…/analytics/powerdash_components.py:42-55` (`:43`) |
| Satisfaction by agent | boxplot (distribution) | by `agent_id` | Altair over a DataFrame | `…/analytics/powerdash_components.py:58-72` |

Contrast vs Timescale CA: every aggregate here is a **full re-scan of an in-memory DataFrame at render time**
— the opposite of incremental. It works because ai-powered's dataset is a small per-call export, not a live
hundreds-of-millions-of-turns stream. For M6's scale + freshness requirement this does not hold; but the
component *taxonomy* is useful input to the KPI set: sentiment-over-time, category distribution, and
tactic/adherence counts are exactly the manager-facing KPIs — so ai-powered informs **what to show**, while
chatwoot informs **how to compute it incrementally**.

## Cross-cutting Comparison

| Dimension | chatwoot | ai-powered-call-center-intelligence | M6 decision |
|---|---|---|---|
| Aggregation engine | Postgres relational table + additive upsert | Separate DuckDB file + pandas in-memory | TimescaleDB continuous aggregate (single engine) |
| Increment model | O(1) additive upsert per event, no scan (`rollup_service.rb:73-79`) | Full `value_counts` re-scan per render (`powerdash_components.py:27,43`) | CA incremental refresh over invalidated buckets |
| Bucket grain | daily × dimension × metric (`rollup_service.rb:63-70`) | per-call rows (`duckdb_loader.py:18-25`) | 5-min `time_bucket` × queue × rule_name |
| Dimensions | account / agent / inbox / team (`reporting_events_rollup.rb:28`) | call_id / issue_category / agent_id (`powerdash_components.py`) | queue, domain (`turns.metadata`), rule_name (`alerts`) |
| Scheduling | event-driven synchronous (`reporting_event_listener.rb:23-24`) | manual notebook run | CA policy, 5-min schedule, `end_offset` skips open bucket |
| Read path | `V2::ReportBuilder` group-by-period + metric whitelist (`report_builder.rb:103-112,75-85`) | ad-hoc `query(sql)` → DataFrame (`duckdb_loader.py:46-48`) | `SELECT` over CA behind `AnalyticsQueryRunner` port (DIP) |
| Retention posture | rollup table is permanent relational data | separate store, no purge coupling | **NO retention on CA**; raw purges at 30d, rollup outlives (ADR-005) |
| Correctness tests | real-DB RSpec + factories, before/after count (`spec/listeners/…:14-19`) | none in scope | real-Timescale integration test: seed → refresh → assert bucket counts + survives-purge |
| Relevance to M6 | **template** for the incremental pattern + test posture | **contrast** — why NOT a second columnar store | — |

## ADRs

### D1 — CA definition & dimensions: 5-min `time_bucket` continuous aggregate keyed by (bucket, dimension)

**Decision:** Define each M6 KPI as a `MATERIALIZED VIEW … WITH (timescaledb.continuous) … WITH NO DATA`
selecting `time_bucket('5 minutes', created_at) AS bucket` plus dimension columns and aggregate metrics,
`GROUP BY bucket, <dimensions>`. Dimensions: `queue` and `domain` (from `turns.metadata` JSONB) and
`rule_name` (from `alerts`). Extend the shipped `turns_per_min` CA pattern, do not invent a new mechanism.

**Rationale:** chatwoot proves the winning shape is keyed incremental accumulation over `(bucket, dimension,
metric)` with no re-scan (`app/services/reporting_events/rollup_service.rb:63-79`). Timescale's CA is the
DB-native realization of exactly that key, and M1 already runs one (`deploy/monitoring/migrations/0002_m1_retention_indexes.sql:44-52`).
KISS + Rung 4 of `.claude/rules/parsimony-ladder.md`: reuse the installed engine and the shipped pattern.

**Alternatives considered:** (a) an application-level rollup table + upsert path like chatwoot — rejected: it
reimplements what Timescale does natively and adds a Sidekiq-equivalent dependency (Rule 9 — don't reinvent).
(b) a separate DuckDB/columnar store like ai-powered (`analytics/duckdb_loader.py:11-25`) — rejected by
ADR-005 (single engine; see D6).

**Consequence:** one CA per KPI family (or one wide CA with several aggregate columns); created empty and
materialized forward; dimensions must be projectable from `alerts`/`turns` at bucket time.

### D2 — Refresh policy: continuous-aggregate policy at 5-min cadence, `end_offset` excludes the open bucket

**Decision:** Refresh each CA with `add_continuous_aggregate_policy(<ca>, start_offset => INTERVAL '2 hours',
end_offset => INTERVAL '5 minutes', schedule_interval => INTERVAL '5 minutes')`. Never materialize the
still-forming bucket. Backfill history once via `CALL refresh_continuous_aggregate(<ca>, <start>, <end>)`.

**Rationale:** chatwoot's backfill materializes **closed days only** ("today is skipped by default" — rake
`reporting_events_rollup.rake` printed plan, step 3) precisely because an in-progress bucket is not final;
`end_offset` is the Timescale equivalent. Its per-event write (`app/listeners/reporting_event_listener.rb:23-24`)
becomes a per-bucket-close refresh. EC-1: the pattern is realized as a policy, not a ported listener.

**Alternatives considered:** (a) real-time aggregation only (no policy) — rejected: unbounded live union over
raw data conflicts with the 30-day purge (see D3). (b) refresh on every ingest via a trigger — rejected as
over-engineering for a 5-min-freshness dashboard (KISS).

**Consequence:** dashboard freshness SLO ≈ one bucket + one schedule interval (~5-10 min) — an honest,
documented lag, not "real-time". A CA refresh failure must fail-soft (not block ingest), mirroring chatwoot's
`safe_rollup` isolation (`app/listeners/reporting_event_listener.rb:178-186`) and `.claude/rules/error-handling.md`.

### D3 — Retention posture: the rollup OUTLIVES the raw purge — NO retention policy on the CA

**Decision:** Attach **no** `add_retention_policy` to the M6 CAs (or, if bounded storage is later required, a
retention far longer than 30 days). The `turns`/`alerts` hypertables keep their 30-day retention
(`deploy/monitoring/migrations/0002_m1_retention_indexes.sql:34-35`); the CA is the long-term business memory
and must survive the raw drop. Ensure each bucket is fully materialized (`end_offset` in D2 well under 30
days) **before** its raw chunk is dropped, so no data is lost when `add_retention_policy` drops a raw chunk.

**Rationale:** ADR-005 states continuous-aggregate rollups are "kept well beyond 30 days as the long-term
business memory — derived, not raw, so retention of the raw data is unaffected"
(`docs/adr/ADR-005-online-storage-realtime-monitoring.md`). This is the EC-2 synthesis point: **neither peer
states it** — chatwoot's rollup is permanent relational data with no purge coupling; ai-powered has a separate
store. In Timescale the nuance is that a CA with real-time aggregation unions materialized buckets with live
raw rows, so once raw purges only the materialized portion remains — hence the "materialize before purge"
ordering. Honors ADR-005's hot/purge split and `.claude/rules/architecture.md § 1` (business memory is a
first-class governed artifact).

**Alternatives considered:** (a) inherit the 30-day retention onto the CA "for symmetry" — rejected: it would
destroy the long-term business trend M6 exists to provide (ROADMAP risk). (b) copy rollups to a Parquet object
store immediately — deferred (YAGNI): ADR-005 allows Parquet offload beyond a longer window, but M6 does not
need it on day one; the CA itself is kilobytes/megabytes.

**Consequence:** the CA grows unbounded over time (slowly — pre-aggregated). A later slice may add Parquet
offload per ADR-005; documented as a known future, not built now.

### D4 — KPI set: business KPIs keyed to `alerts.rule_name` + `turns.metadata` dimensions

**Decision:** M6 ships a small, operationally-meaningful KPI set: cancellation-intent rate per queue per
5 min, escalation rate, sentiment trend (avg sentiment score per bucket), and script/adherence rate. Counts
come from `alerts` (filtered by `rule_name`); the dimension breakdown (queue/domain) from `turns.metadata`
JSONB; sentiment from alert evidence JSONB. Gate every requested KPI against an enum before building SQL.

**Rationale:** chatwoot's metric enum (`resolutions_count, first_response, resolution_time, reply_time,
bot_resolutions_count, bot_handoffs_count` — `app/models/reporting_events_rollup.rb:29-36`) shows a *bounded,
named* KPI set with a `count` + `sum_value` shape and a read-side whitelist (`report_builder.rb:75-85`).
ai-powered's dashboard taxonomy (sentiment-over-time, category distribution, tactic usage —
`analytics/powerdash_components.py:10-72`) corroborates that these are the manager-facing shapes. Combining
both: bounded KPI enum (chatwoot rigor) over the business events M6 already computes (alerts/turns).

**Alternatives considered:** (a) expose arbitrary group-by/metric combos — rejected: unbounded query surface,
no whitelist, matches neither peer. (b) copy chatwoot's contact-center metrics verbatim — rejected: M6's
domain is conversation-intelligence KPIs (cancellation/escalation/sentiment/adherence), not resolution times.

**Consequence:** the KPI enum is the contract between the CA columns and the dashboard read; adding a KPI is a
new CA aggregate + an enum entry, an additive change (OCP).

### D5 — Dashboard query shape: pre-bucketed `SELECT` over the CA, behind a domain port

**Decision:** The dashboard read is `SELECT bucket, <dimension>, <metric> FROM <ca> WHERE bucket BETWEEN
:from AND :to AND <dimension> = :x ORDER BY bucket`, served behind a domain port and routed through the
existing `AnalyticsQueryRunner` consumption boundary. The API/interface layer never touches SQL directly.

**Rationale:** chatwoot centralizes the read in `V2::ReportBuilder` with time-range + entity-filter +
metric-whitelist (`app/builders/v2/report_builder.rb:75-112`); the CA moves the `group_by_period` grouping to
write time, so the read is a filtered scan of already-bucketed rows. `.claude/rules/architecture.md § 1`: the
read is an infrastructure adapter behind a domain port (DIP); `src/talkex/analytics/query_runner.py` already
declares itself "the consumption boundary", so M6 extends it rather than adding a parallel read path (DRY).

**Alternatives considered:** (a) let the FastAPI analytics endpoint query Timescale directly — rejected (DIP
violation, `.claude/rules/architecture.md § 2`). (b) build a new report builder — rejected: `AnalyticsQueryRunner`
+ `SimpleAnalyticsEngine` already exist (`src/talkex/analytics/__init__.py:9-49`) (DRY / Rung 4 parsimony).

**Consequence:** a thin CA-backed query implementation slots behind the existing analytics port; the read
indexes to add on the CA mirror chatwoot's `index_rollup_timeseries` / `index_rollup_summary`
(`db/migrate/20260211145813_create_reporting_events_rollup.rb:29-35`).

### D6 — Reject a separate columnar store: single Timescale engine, DuckDB is contrast not template

**Decision:** M6 does NOT introduce a separate columnar analytics store. All aggregation stays inside the one
TimescaleDB instance as continuous aggregates.

**Rationale:** ai-powered's DuckDB path (`analytics/duckdb_loader.py:11-25`, `powerdash_components.py:27,43`)
is a second engine that duplicates data and re-scans on render — exactly the design ADR-005 rejects
("Introducing ClickHouse just for a 30-day window is rejected as over-engineering"). For a 30-day live
window, single-engine CAs are simpler and incremental. EC-3: Q2/Q4 exist to contrast, not to adopt. KISS +
YAGNI.

**Alternatives considered:** (a) DuckDB/columnar side-store for heavy analytical queries — rejected by ADR-005
and by scale mismatch (ai-powered's per-call export vs M6's live stream). (b) ClickHouse — same rejection.

**Consequence:** heavy historical/ad-hoc analytics beyond the CA (should they ever be needed) would use the
Parquet offload path from ADR-005, not a live second engine — deferred (YAGNI).

## Recommendations

Concrete, per-question proposals for `/to-plan`:

1. **(Q1/Q3) CA over `alerts`:** create `alert_kpis_5min WITH (timescaledb.continuous) AS SELECT
   time_bucket('5 minutes', created_at) AS bucket, (metadata->>'queue') AS queue, rule_name,
   count(*) AS alert_count FROM alerts GROUP BY bucket, queue, rule_name WITH NO DATA` — extending the
   `turns_per_min` precedent (`deploy/monitoring/migrations/0002_m1_retention_indexes.sql:44-52`). Add a
   second CA for sentiment (avg over alert evidence). Verify `queue`/`domain` are reachable from
   `turns.metadata` / `alerts.evidence` JSONB before finalizing column projections (see Honest gaps).
2. **(Q6) Refresh policy:** `add_continuous_aggregate_policy` at 5-min `schedule_interval`, 5-min `end_offset`,
   2-hour `start_offset`; one-off `refresh_continuous_aggregate` for backfill. Document the ~5-10 min freshness SLO.
3. **(Q1/Q3) Retention:** add NO retention policy to the CAs; keep the 30-day policy on `turns`/`alerts`
   (`0002_…:34-35`); assert (integration test) the rollup survives a raw-chunk drop.
4. **(Q4/Q2) Storage:** single Timescale engine; no DuckDB/columnar side-store. Frame ai-powered as the
   counter-example in the plan's rationale.
5. **(Q7/Q2) KPI set + read:** a bounded KPI enum (cancellation/escalation/sentiment/adherence rate per queue
   per 5 min), a metric-whitelist gate (chatwoot `report_builder.rb:75-85`), served behind `AnalyticsQueryRunner`
   (`src/talkex/analytics/query_runner.py`). Add CA read indexes mirroring `index_rollup_timeseries`/`index_rollup_summary`.
6. **(Q5) Tests:** real-TimescaleDB integration tests — (a) seed alerts across 2 buckets × 2 queues, refresh,
   assert per-(bucket, queue) counts; (b) drop a raw chunk, re-query CA, assert rows persist; (c) refresh-failure
   does not block ingest. Mirror chatwoot's before/after-count assertion style (`spec/listeners/…:14-19`).

## Honest gaps

- **Not verified: exact JSONB paths for `queue`/`domain`/`sentiment`.** The blueprint assumes `queue`/`domain`
  live in `turns.metadata` and sentiment/rule signals in `alerts` (`rule_name`, `evidence`), per the task's
  internal-context brief. The actual column/JSONB keys were NOT read from the M1 schema in this investigation —
  `/to-plan` must confirm the projection paths against the live `turns`/`alerts` DDL before writing the CA SELECT.
- **Not verified in-repo: Timescale CA refresh-vs-retention interaction.** The "materialize each bucket before
  its raw chunk is dropped" ordering (D3) is a correct Timescale property but is a synthesis from ADR-005 +
  Timescale semantics, not read from a peer or a project file. It should be proven by the survives-purge
  integration test (Recommendation 6b), not assumed.
- **`disable_report_rollup_for_all_accounts` migration not read.** The read-path soft-toggle claim (Q6) is
  grounded in the in-code comment `app/services/reporting_events/rollup_service.rb:22-27`, which is sufficient;
  the disable migration file exists but its body was not read, so no claim rests on it.
- **ai-powered `requirements`-adjacent (Q4 Fase A) not separately opened.** The engine verdict (DuckDB +
  pandas) is established directly from `duckdb_loader.py` imports (`:5-6`) and `powerdash_components.py`
  (`:4-5`), which is conclusive; a requirements file read would only re-confirm it.

No question was BLOCKED. All 7 were fully answered from read source lines; the gaps above are downstream
verification items for `/to-plan`, not unanswered questions.

## discover-confidence verdict

**SHIPPABLE_WITH_CAVEATS** — all four coverage corners are populated with real `path:line` citations to
existing reference files; all 7 questions answered; both mandatory synthesis checkpoints (EC-1 pattern→CA
mapping in D1/D2/Q1; EC-2 rollup-outlives-raw retention posture in D3) are recorded as ADRs. The caveats are
the three Honest-gaps verification items (JSONB projection paths, CA refresh-vs-retention ordering proof, and
the deferred Parquet offload) — each is an explicit downstream check for `/to-plan`, not a hole in the
evidence base.
