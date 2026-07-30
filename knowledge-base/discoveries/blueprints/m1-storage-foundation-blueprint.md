# Blueprint: M1 Storage Foundation — Productionizing ADR-005

> **Version 1.0** — Locks the M1 storage design (index strategy, BM25/full-text, indexed query path,
> pool sizing, and the Timescale/pgvector productionization) by synthesizing chatwoot's production
> Postgres patterns against ADR-005. Produced by `cycle-discover` execute from
> `knowledge-base/discoveries/plans/m1-storage-foundation-plan.md`.

**Slug:** `m1-storage-foundation`
**Created:** 2026-07-29
**discover-confidence verdict:** recorded at the end after scoring.

## Executive summary

M1 productionizes the ADR-005 storage the M0 skeleton stubbed: query-aligned indexes on
`turns`/`alerts`, a BM25/full-text strategy, an indexed+paginated query path that survives
ingest×query contention, a pool sized to ingest concurrency, and the Timescale lifecycle
(30-day retention, compression, continuous aggregates) + pgvector index. chatwoot supplies strong,
citable precedent for the *Postgres* patterns (composite + GIN trigram indexes, `pg_search`/ILIKE,
filter-aggregate counting, pool==concurrency). The Timescale/pgvector core is NOT attested by any
clone — it is sourced from ADR-005 + official docs (honest gap G1, per plan ADR D3).

## Context

M0 shipped a plain `turns`/`alerts` hypertable with a `(conversation_id, created_at DESC)` index
(`deploy/monitoring/migrations/0001_m0_hypertable.sql`). M1 hardens it for the 30-day hot window and
the supervisor/QA query load (`ROADMAP.md § M1`), within `.claude/rules/architecture.md` (infra
adapters behind DIP ports) and `.claude/rules/testing.md` (integration tests vs a real DB).

## Objective

Lock, with cited evidence, the M1 index design, BM25 strategy, indexed query path, pool sizing, and
the Timescale/pgvector productionization — so M1 implementation proceeds without rework.

## Coverage Corner 1 — Integration Tests

chatwoot specs the query layer against a real DB with fixtures: the finder spec
`knowledge-base/references/chatwoot/spec/finders/conversation_finder_spec.rb` and the model spec
`knowledge-base/references/chatwoot/spec/models/conversation_spec.rb` exercise real records and assert
filtered/paginated results. **M1 test decision:** translate the pattern (real-DB fixtures + assertion on
filtered/paginated/indexed results) to pytest against a real TimescaleDB (extending the M0 integration
tier), asserting retention/compression/continuous-aggregate behavior and index usage — NOT copying the
RSpec syntax (edge-case EC-2).

## Coverage Corner 2 — Dependencies

chatwoot enables its Postgres extensions in the init migration:
`knowledge-base/references/chatwoot/db/migrate/20230426130150_init_schema.rb:4` (`enable_extension` for
`pg_stat_statements`, `pg_trgm`, `pgcrypto`, `plpgsql`), and declares `pg` + `pg_search` in
`knowledge-base/references/chatwoot/Gemfile`. **M1 deps decision:** M1 enables `timescaledb` (already in
the M0 migration), `vector` (pgvector — bundled in `timescaledb-ha:pg16`), and `pg_trgm`; the BM25 path
uses a generated `tsvector` column + GIN (or `pg_search`/ParadeDB if adopted — see D2). pgvector version
is dictated by the `timescaledb-ha:pg16` image, not a peer (gap G1).

## Coverage Corner 3 — Tools

chatwoot sizes its connection pool to the async concurrency:
`knowledge-base/references/chatwoot/config/database.yml:2` — `pool: <%= Sidekiq.server? ?
ENV.fetch('SIDEKIQ_CONCURRENCY', 10) : ENV.fetch('RAILS_MAX_THREADS', 5) %>` with a `reaping_frequency`;
the concurrency source is `knowledge-base/references/chatwoot/config/sidekiq.yml`. **M1 tools decision:**
adopt pool==concurrency — the M1 ingest worker pool and the query/read pool are sized to their respective
concurrency, with a reaping frequency; this is the concrete mitigation for the ADR-005 ingest×query
contention risk (R3), ahead of a read-replica split (deferred).

## Coverage Corner 4 — Techniques

**Index design.** chatwoot's canonical query-aligned composite index:
`knowledge-base/references/chatwoot/db/schema.rb:799` (`["account_id","inbox_id","status","assignee_id"]`,
`conv_acid_inbid_stat_asgnid_idx`), plus GIN trigram on message bodies
`knowledge-base/references/chatwoot/db/schema.rb:1171` (`messages.content gin_trgm_ops`) and a GIN trigram
composite `knowledge-base/references/chatwoot/db/migrate/20230426130150_init_schema.rb:370`. **M1 decision:**
index `turns`/`alerts` to match the supervisor/QA query — `(conversation_id, created_at DESC)` (M0
baseline) + a GIN trigram on `turns.raw_text` for substring/lexical scans + a pgvector HNSW index on the
embedding column (M2 populates it; M1 creates the index path).

**BM25 / full-text.** chatwoot uses `pg_search` only on Article:
`knowledge-base/references/chatwoot/app/models/article.rb:37` (`include PgSearch::Model`), `:89`
(`pg_search_scope` with weighted `tsearch`, `normalization: 2`, prefix); elsewhere it falls back to ILIKE +
GIN trigram (`knowledge-base/references/chatwoot/app/finders/conversation_finder.rb:156`). **M1 decision:**
for the 30-day window, use a generated `tsvector` column + GIN for BM25-adjacent lexical ranking
(dependency-light, aligns with ADR-005 single-spine); `pg_search`/ParadeDB is the richer BM25 option flagged
for a measured pilot. Honest: chatwoot does NOT run BM25 on high-volume message rows — a caution for M1's
throughput.

**Indexed query path under load.** chatwoot's finder aligns `where` to the composite index, paginates, and
counts with a single filter-aggregate:
`knowledge-base/references/chatwoot/app/finders/conversation_finder.rb:108` (account/inbox/status/assignee
`where`), `:188` (`COUNT(*) FILTER (...)` instead of N queries), `:212` (eager-load to avoid N+1), `:225`
(pagination). **M1 decision:** the supervisor/QA read path uses index-aligned `WHERE (conversation_id,
created_at)` + keyset pagination + a single filter-aggregate count, never N+1.

**M1 techniques decision (synthesis):** query-aligned composite + GIN trigram + pgvector HNSW indexes;
generated `tsvector` for lexical ranking; index-aligned keyset-paginated reads with filter-aggregate counts.

## Cross-cutting Comparison

| Concern | chatwoot (peer) | ADR-005 (source) | M1 decision |
|---|---|---|---|
| Time partitioning | none (plain PG) | Timescale hypertable | hypertable + 30-day retention/compression (gap G1) |
| Index strategy | composite + GIN trigram | — | `(conversation_id,created_at)` + GIN trigram + pgvector HNSW |
| BM25/full-text | pg_search on Article; ILIKE+trigram elsewhere | BM25 (pg_search/tsvector) | generated tsvector + GIN; pg_search flagged for pilot |
| Query path | index-aligned + filter-aggregate + eager-load | — | keyset pagination + filter-aggregate count |
| Pool | pool==concurrency (database.yml) | single-spine | pool==ingest/query concurrency + reaping |
| Retention/rollups | none | 30-day purge + continuous aggregates | add_retention_policy + continuous aggregate (gap G1) |

## ADRs

### D1 — Query-aligned indexes on turns/alerts
Index `(conversation_id, created_at DESC)` (M0 baseline) + GIN trigram on `turns.raw_text` + a pgvector HNSW
index path for the embedding column. **Rationale:** mirrors chatwoot's composite + GIN trigram design
(`schema.rb:799,:1171`). **Consequence:** the supervisor/QA query is index-covered; HNSW build cost is bounded
by the 30-day window (ADR-005).

### D2 — BM25 via generated tsvector + GIN (pg_search as pilot option)
A generated `tsvector` column + GIN for lexical ranking; `pg_search`/ParadeDB flagged for a measured pilot.
**Rationale:** chatwoot proves `pg_search` weighting (`article.rb:89`) but only on low-volume Article; on
high-volume rows it uses ILIKE+trigram (`conversation_finder.rb:156`) — a throughput caution. **Alternative
rejected:** `pg_search` on all turns now (unproven at ingest throughput). **Consequence:** dependency-light,
ADR-005-aligned; richer BM25 deferred to a measured decision.

### D3 — Timescale lifecycle sourced from ADR-005, not peers (honest gap)
`add_retention_policy(30 days)`, compression on chunks older than N days, and a continuous aggregate for
supervisor rollups come from `docs/adr/ADR-005-online-storage-realtime-monitoring.md` + official Timescale
docs — NO peer attests them (gap G1). **Rationale:** Unbreakable Rule 3 — no fabricated peer citations.
**Consequence:** M1 implementation validates these against a real Timescale in the integration tier.

### D4 — Ingest×query contention: pool==concurrency + filter-aggregate reads
Size the ingest pool and the read pool to their concurrency (chatwoot `database.yml:2`), use filter-aggregate
counts and eager-loading (`conversation_finder.rb:188,:212`). **Alternative rejected:** immediate read-replica
split (premature — ADR-005 R3 defers it to the measured pilot). **Consequence:** contention mitigated in one
instance first; replica is the escape hatch.

### D5 — Multi-stream fan-in on the M0 bounded channel + pool
Reuse the M0 bounded `TurnChannel` per session with the pooled repo; ingest stays async and backpressured.
**Rationale:** builds on M0 D1. **Consequence:** fan-in scales with the pool; backpressure protects the DB.

## Recommendations

1. M1 migration `0002`: `add_retention_policy('turns',INTERVAL '30 days')` + `add_retention_policy('alerts',...)`;
   enable compression + a compression policy; create a continuous aggregate for per-queue rollups; add the GIN
   trigram + pgvector HNSW indexes (D1/D3).
2. Add a generated `tsvector` column on `turns` + GIN (D2).
3. Introduce a psycopg connection pool (psycopg_pool) sized to ingest/query concurrency; wire it into the
   infrastructure adapters (D4).
4. Extend the integration tier: tests asserting retention drop, compression, continuous-aggregate refresh,
   index usage (EXPLAIN), and keyset-paginated reads against real Timescale (Corner 1).
5. Benchmark ingest (300 turns/s) + concurrent query p95 < 200 ms — the ADR-005 pilot gate.

## Honest gaps

1. **G1 — Timescale lifecycle + pgvector unattested by peers** (retention/compression/continuous-aggregates,
   HNSW tuning). Sourced from ADR-005 + official docs; validated in M1's integration tier, not by a peer citation.
2. **G2 — BM25 throughput at ingest rate** is unproven; D2 chooses the dependency-light tsvector path and flags
   pg_search for a measured pilot.

## discover-confidence verdict

**SHIPPABLE — score 100.0, hard_caps_triggered: none** (via `run_blueprint_score.py`, 2026-07-30).
Coverage 4/4 corners, all citations resolve, 5 ADRs (D1–D5). Proceed to the `cycle-plan` phase for M1.
