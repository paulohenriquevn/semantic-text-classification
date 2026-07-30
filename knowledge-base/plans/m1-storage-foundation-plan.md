# Plan: M1 Storage Foundation — Productionizing ADR-005

> **Version 1.0** — Harden the M0 `turns`/`alerts` hypertables for the 30-day hot window and the
> supervisor/QA query load: add retention + compression + a continuous aggregate, query-aligned GIN
> trigram + pgvector HNSW indexes + a generated `tsvector` column, a psycopg connection pool sized to
> concurrency, and index-aligned keyset-paginated reads — all validated against a real TimescaleDB.

## Goal

> "Enable the storage layer to hold the 30-day hot window under ingest×query load so that old data is
> auto-purged and supervisor reads stay index-covered, measured by the integration test
> `test_retention_drops_old_chunks_and_query_uses_index` passing (retention policy drops >30-day data;
> the supervisor read uses the composite index via EXPLAIN)."

## Context

M0 shipped plain `turns`/`alerts` hypertables with a `(conversation_id, created_at DESC)` index
(`deploy/monitoring/migrations/0001_m0_hypertable.sql`). M1 productionizes ADR-005 per the blueprint
`knowledge-base/discoveries/blueprints/m1-storage-foundation-blueprint.md` (discover-confidence SHIPPABLE
100.0): retention/compression/continuous-aggregates + BM25/pgvector index paths + pool sizing, within
`.claude/rules/architecture.md` (infra behind DIP) and `.claude/rules/testing.md` (real-DB integration).

## Baseline Context (deep review of current state)

### Files that will be touched

| File | LoC today | Last commit (sha + date) | Why it exists today | Invariants to preserve |
|---|---|---|---|---|
| `deploy/monitoring/migrations/0001_m0_hypertable.sql` | ~40 | `a8c3105` (2026-07-29) | M0 turns/alerts hypertables + base index | Do not drop; 0002 is additive/idempotent |
| `src/talkex/monitoring/infrastructure/timescale_repo.py` | 77 | `d57e4b5` (2026-07-29) | per-Turn/Alert insert + alert get | Keep `TurnRepository`/`AlertRepository` port compat |
| `src/talkex/monitoring/config.py` | 21 | `a8c3105` (2026-07-29) | MonitoringConfig (dsn, queue, channel) | Keep existing fields; add pool/retention additively |
| `src/talkex/monitoring/domain/ports.py` | 33 | `a8c3105` (2026-07-29) | DIP ports | Additive only (new read port) |
| `deploy/monitoring/migrations/0002_m1_retention_indexes.sql` (NEW) | 0 | — | retention/compression/CA + GIN/HNSW/tsvector | idempotent |
| `src/talkex/monitoring/infrastructure/pool.py` (NEW) | 0 | — | psycopg_pool wrapper sized to concurrency | — |
| `src/talkex/monitoring/infrastructure/read_repo.py` (NEW) | 0 | — | index-aligned keyset-paginated supervisor/QA reads | — |
| `tests/integration/monitoring/test_retention_compression.py` (NEW) | 0 | — | retention/compression/CA integration tests | — |
| `tests/integration/monitoring/test_indexes_and_query.py` (NEW) | 0 | — | index-usage (EXPLAIN) + keyset pagination tests | — |
| `tests/unit/monitoring/test_pool.py` (NEW) | 0 | — | pool sizing unit test | — |

### Current callers / dependents

- **Symbol:** `TimescaleTurnRepository`/`TimescaleAlertRepository` (`infrastructure/timescale_repo.py`) — Callers: `interface/app.py` (composition root), `tests/integration/monitoring/`. M1 adds a pool option (backward-compatible: still accepts a plain `AsyncConnection`).
- **Symbol:** `MonitoringConfig` (`config.py`) — Callers: `interface/app.py`, `infrastructure/*`, tests. M1 adds fields with defaults (backward-compatible).
- New `read_repo.py` + `pool.py`: first-of-its-kind, no callers yet.

### Domain glossary

- **Hypertable** — Timescale time-partitioned table (chunks by `created_at`).
- **Retention policy** — background job dropping chunks older than an interval (`add_retention_policy`).
- **Continuous aggregate** — incrementally-maintained materialized rollup over a hypertable.
- **Keyset pagination** — `WHERE (created_at, id) < (:cursor)` ordered pagination (index-aligned, no OFFSET).
- **HNSW** — pgvector approximate-nearest-neighbor index.

### Architecture boundaries affected

Per `.claude/rules/architecture.md § 2`: M1 adds infrastructure adapters (`pool.py`, `read_repo.py`) behind
a new domain read port; the pool/Timescale specifics stay in infrastructure, injected at the composition root.

## Prior Art & Related Work

- **Internal blueprint** — `knowledge-base/discoveries/blueprints/m1-storage-foundation-blueprint.md` §"Coverage Corner 4 — Techniques" and §"ADRs" D1–D5.
- **Reference — chatwoot** — composite index `knowledge-base/references/chatwoot/db/schema.rb:799`; GIN trigram `db/schema.rb:1171`; `pg_search` weighting `app/models/article.rb:89`; filter-aggregate count `app/finders/conversation_finder.rb:188`; pool==concurrency `config/database.yml:2`.
- **ADR** — `docs/adr/ADR-005-online-storage-realtime-monitoring.md` (retention/compression/CA + pgvector — the Timescale specifics, gap G1).

## Objective

- [ ] Migration 0002: retention + compression + continuous aggregate + GIN trigram + pgvector HNSW + generated tsvector, idempotent
- [ ] psycopg connection pool sized to concurrency, unit-tested
- [ ] Index-aligned keyset-paginated read repo (supervisor/QA reads)
- [ ] Integration tests: retention drop, compression, CA refresh, index usage via EXPLAIN, keyset pagination
- [ ] No regression to M0 (writes + e2e still green)

## ADRs

### D1 — Query-aligned indexes (composite + GIN trigram + pgvector HNSW)
- **Decision:** keep `(conversation_id, created_at DESC)`; add GIN trigram on `turns.raw_text`; create a pgvector HNSW index path on an embedding column (populated in M2).
- **Rationale:** mirrors chatwoot's composite + GIN trigram (`schema.rb:799,:1171`); ADR-005 pgvector.
- **Alternatives considered:** btree-only (rejected — no substring/lexical acceleration); IVFFlat over HNSW (rejected — HNSW better recall/latency for the bounded 30-day set).
- **Consequences:** supervisor/QA reads are index-covered; HNSW build bounded by the 30-day window.

### D2 — BM25 via generated tsvector + GIN (pg_search as pilot option)
- **Decision:** a generated `tsvector` column on `turns` + GIN; `pg_search`/ParadeDB flagged for a measured pilot.
- **Rationale:** chatwoot proves `pg_search` weighting but only on low-volume Article (`article.rb:89`); high-volume rows use ILIKE+trigram (`conversation_finder.rb:156`).
- **Alternatives considered:** `pg_search` on all turns now (rejected — unproven at ingest throughput, drawback R2).
- **Consequences:** dependency-light, ADR-005-aligned; richer BM25 deferred to a measured decision.

### D3 — Timescale lifecycle from ADR-005 (honest gap, validated by tests)
- **Decision:** `add_retention_policy(30 days)`, compression policy on old chunks, a continuous aggregate for rollups — sourced from ADR-005 + Timescale docs, validated in the integration tier.
- **Rationale:** no peer attests Timescale lifecycle (gap G1); Unbreakable Rule 3 forbids fabricated peer citations.
- **Alternatives considered:** manual DELETE-based purge (rejected — chunk drop is O(1); DELETE causes vacuum bloat).
- **Consequences:** purge is a partition drop; M1 tests prove it against real Timescale.

### D4 — Pool sized to concurrency for ingest×query contention
- **Decision:** psycopg connection pool sized to ingest/query concurrency, with reaping.
- **Rationale:** chatwoot `database.yml:2` sizes pool==concurrency; ADR-005 R3.
- **Alternatives considered:** one connection per operation (rejected — connection churn under 300 turns/s); immediate read-replica (rejected — premature per R3).
- **Consequences:** bounded connections; replica is the later escape hatch.

### D5 — Multi-stream fan-in reuses the M0 bounded channel + pool
- **Decision:** ingest fan-in stays on the M0 per-session bounded `TurnChannel` (backpressure) with the pooled repo; no new ingestion transport.
- **Rationale:** builds on M0 D1 (bounded channel); the pool (D4) absorbs concurrent writes.
- **Alternatives considered:** an external queue (Kafka/NATS) for fan-in (rejected — premature for the 30-day window; YAGNI until measured throughput demands it).
- **Consequences:** fan-in scales with the pool; backpressure protects the DB; a queue is the later escape hatch.

## Drawbacks & Risks

| Drawback / Risk | Severity | Mitigation | Owner |
|---|---|---|---|
| R1 — retention/compression are background jobs; timing is not instantaneous in tests | Medium | tests trigger the policy explicitly (`call run_job` / manual drop_chunks) rather than waiting on the scheduler | dev |
| R2 — generated tsvector + GIN adds write overhead per Turn at ingest rate | Medium | benchmark write throughput; tsvector is generated (no app cost); pg_search deferred | dev |
| R3 — pgvector HNSW index build/memory grows within the 30-day window | Medium | bounded by retention; monitor; tune `m`/`ef_construction` in the pilot | dev |
| R4 — pool exhaustion under burst ingest | Medium | pool sized to concurrency + reaping + backpressure from the M0 bounded channel | dev |

## Unresolved Questions

- Q1 — Does `timescaledb-ha:pg16` bundle pgvector with HNSW support enabled? (resolve in T1.1 by `CREATE INDEX ... USING hnsw` succeeding in the integration test).
- Q2 — Exact continuous-aggregate granularity (per-minute vs per-5-minute rollup)? M1 uses 1-minute buckets; tunable later.
- Q3 — tsvector language config (`portuguese` vs `simple`)? M1 uses `portuguese` for PT-BR stemming.

## Dependency Graph

```
Phase 0 (migration 0002: retention/compression/CA + indexes + tsvector)
   │
   ▼
Phase 1 (infra: connection pool)  ──▶  Phase 2 (infra: index-aligned keyset read repo)
   │                                        │
   └────────────────────┬───────────────────┘
                        ▼
             Final: Integration Validation (retention/compression/CA/EXPLAIN/keyset)
```

---

## Phase 0: Migration 0002 — lifecycle + indexes

**Objective:** productionize the schema per D1/D2/D3.

### T0.1 — retention + compression + continuous aggregate + indexes + tsvector

#### Objective
Add retention/compression/CA + GIN trigram + pgvector HNSW + generated tsvector to `turns`/`alerts`.

#### Why this step
1. **What:** `deploy/monitoring/migrations/0002_m1_retention_indexes.sql` — `add_retention_policy`, compression, a continuous aggregate, GIN trigram on `turns.raw_text`, a generated `search_vector tsvector` + GIN, and an embedding column + HNSW index path.
2. **Why now:** the storage core of M1 (D1/D2/D3); every downstream test needs it. Cites blueprint D1-D3 + ADR-005.

#### Evidence
M0 migration `deploy/monitoring/migrations/0001_m0_hypertable.sql` (hypertables). Blueprint D3 (retention/compression/CA). chatwoot GIN trigram `knowledge-base/references/chatwoot/db/schema.rb:1171`.

#### Files to edit
```
deploy/monitoring/migrations/0002_m1_retention_indexes.sql (NEW) — retention/compression/CA + GIN/HNSW/tsvector
```

#### Deep Dives
- `ALTER TABLE turns ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', raw_text)) STORED;` + `CREATE INDEX ... USING gin (search_vector)`.
- `ALTER TABLE turns ADD COLUMN embedding vector(384);` + `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` (populated in M2).
- `SELECT add_retention_policy('turns', INTERVAL '30 days', if_not_exists => true);` (+ alerts).
- `ALTER TABLE turns SET (timescaledb.compress); SELECT add_compression_policy('turns', INTERVAL '3 days', if_not_exists => true);`.
- `CREATE MATERIALIZED VIEW turns_per_min WITH (timescaledb.continuous) AS SELECT time_bucket('1 minute', created_at) b, conversation_id, count(*) FROM turns GROUP BY b, conversation_id;`.
- Invariant: every statement `IF NOT EXISTS`/`if_not_exists` (idempotent).

#### TDD
```
RED (integration): test_retention_policy_registered — timescaledb_information.jobs lists a retention job for turns
RED (integration): test_hnsw_and_gin_indexes_exist — pg_indexes shows the hnsw + gin indexes
GREEN:   write the SQL; apply against a fresh container
REFACTOR: None expected
VERIFY:  psql -h localhost -p 5433 -f deploy/monitoring/migrations/0002_m1_retention_indexes.sql
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `test_retention_policy_registered` asserts a retention job exists in `timescaledb_information.jobs` for `turns` and `alerts`
- [ ] `test_hnsw_and_gin_indexes_exist` asserts `pg_indexes` contains the hnsw + gin trigram + tsvector-gin indexes
- [ ] `psql -f deploy/monitoring/migrations/0002_m1_retention_indexes.sql` exits 0 and re-applies without error (idempotent)

#### DoD
- [ ] migration applies idempotently; retention job + indexes present (resolves Q1)

---

## Phase 1: Connection pool

**Objective:** a psycopg pool sized to concurrency (D4).

### T1.1 — psycopg connection pool

#### Objective
`MonitoringPool` wrapping `psycopg_pool.AsyncConnectionPool` sized from config.

#### Why this step
1. **What:** `src/talkex/monitoring/infrastructure/pool.py` + `MonitoringConfig` pool fields.
2. **Why now:** the ingest×query contention mitigation (D4); read repo (Phase 2) uses it.

#### Files to edit
```
src/talkex/monitoring/infrastructure/pool.py (NEW) — AsyncConnectionPool wrapper (min/max sized to concurrency)
src/talkex/monitoring/config.py — add pool_min_size, pool_max_size, retention_days (defaults)
```

#### Deep Dives
- `MonitoringPool(dsn, max_size)` → `AsyncConnectionPool(dsn, max_size=max_size)`; `async connection()` context manager; `open()`/`close()`.
- Invariant: `max_size == config.pool_max_size`; pool is not opened at import.

#### TDD
```
RED:     test_pool_config_defaults — MonitoringConfig exposes pool_max_size > 0
RED (integration): test_pool_gives_working_connections — acquire a connection, SELECT 1 returns 1
GREEN:   implement MonitoringPool + config fields
VERIFY:  pytest tests/unit/monitoring/test_pool.py -q ; pytest tests/integration/monitoring -k pool -m integration -q
```

#### Concurrency tests
```
Atomic acquire invariant: N concurrent tasks each acquire a connection from a max_size=4 pool and run
SELECT 1; assert all N succeed (queueing, not error) and the pool never exceeds max_size.
```

#### Acceptance Criteria
- [ ] `test_pool_config_defaults` asserts `MonitoringConfig().pool_max_size > 0`
- [ ] `test_pool_gives_working_connections` asserts an acquired connection returns `1` from `SELECT 1`
- [ ] `ruff check src/talkex/monitoring/infrastructure/pool.py` and `mypy` report zero warnings/errors

#### DoD
- [ ] pool unit + integration tests pass; ruff+mypy clean

---

## Phase 2: Index-aligned keyset read repo

**Objective:** supervisor/QA reads that use the index (D1) and paginate by keyset.

### T2.1 — read repository (keyset pagination + index-aligned WHERE)

#### Objective
`TimescaleReadRepository.recent_turns(conversation_id, before_cursor, limit)` using `(conversation_id, created_at DESC)`.

#### Why this step
1. **What:** `src/talkex/monitoring/infrastructure/read_repo.py` + a domain read port in `ports.py`.
2. **Why now:** the supervisor/QA query path (blueprint Corner 4); proves index usage (D1).

#### Files to edit
```
src/talkex/monitoring/infrastructure/read_repo.py (NEW) — keyset-paginated reads via the pool
src/talkex/monitoring/domain/ports.py — add TurnReadPort protocol
```

#### Deep Dives
- Query: `SELECT ... FROM turns WHERE conversation_id=%s AND created_at < %s ORDER BY created_at DESC LIMIT %s` — matches `(conversation_id, created_at DESC)` index (no OFFSET, no N+1).
- Invariant: no `OFFSET`; cursor is `created_at` of the last row.

#### TDD
```
RED (integration): test_recent_turns_keyset_paginates — insert 5 turns; page size 2 returns 2 then 2 then 1, no overlap
RED (integration): test_recent_turns_uses_composite_index — EXPLAIN shows an Index Scan using the composite index
GREEN:   implement read repo + port
VERIFY:  pytest tests/integration/monitoring/test_indexes_and_query.py -m integration -q
```

#### Concurrency tests
```
(none — single-threaded per query; the pool's concurrency is tested in T1.1)
```

#### Acceptance Criteria
- [ ] `test_recent_turns_keyset_paginates` asserts non-overlapping pages of size 2,2,1 for 5 turns
- [ ] `test_recent_turns_uses_composite_index` asserts the `EXPLAIN` output contains `Index Scan` (not `Seq Scan`)
- [ ] `ruff check` and `mypy` report zero warnings/errors on `read_repo.py`

#### DoD
- [ ] read-repo integration tests pass; ruff+mypy clean

---

## Coverage Matrix

| # | Gap / Requirement | Task(s) | Resolution |
|---|---|---|---|
| 1 | 30-day retention (purge) | T0.1 | add_retention_policy + test |
| 2 | Compression of old chunks | T0.1 | compression policy + test |
| 3 | Continuous aggregate rollup | T0.1 | continuous aggregate + test |
| 4 | BM25/lexical index (tsvector+GIN) | T0.1 | generated tsvector + GIN + test |
| 5 | pgvector HNSW index path | T0.1 | hnsw index + test |
| 6 | Pool sized to concurrency (ingest×query) | T1.1 | MonitoringPool + tests |
| 7 | Index-aligned keyset reads | T2.1 | read repo + EXPLAIN test |

**Coverage: 7/7 gaps covered (100%)**

## Global Definition of Done

- [ ] All phases completed
- [ ] `pytest tests/unit/monitoring -q` and `pytest tests/integration/monitoring -m integration -q` green
- [ ] `mypy src/talkex/monitoring` clean; `ruff check` + `ruff format --check` clean
- [ ] File-size budget ≤ 500 LoC per file
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] No regression to M0 (writes + e2e still green)
- [ ] Runtime-metric proof — the retention/EXPLAIN tests observe real DB behavior, not just compile
- [ ] Plan archived to `completed/` after `/review` READY_TO_MERGE + merge

## Failure scenarios (external I/O)

| Dependency | Failure mode | How the test reproduces it | Expected behavior |
|---|---|---|---|
| `timescale` (DB) | pool exhausted under burst | acquire max_size+1 concurrently | extra acquirer queues (does not error); backpressure holds; no crash |
| `timescale` (DB) | connection reset mid-read | close a pooled connection then query | typed error surfaced (fail-fast, Rule 8), pool recovers next acquire |

## Final Phase: Integration Validation (MANDATORY)

**Objective:** prove retention/compression/CA/index-usage/keyset against a real Timescale.

### Execution
```
docker compose -f deploy/monitoring/docker-compose.yml up -d
psql -h localhost -p 5433 -f deploy/monitoring/migrations/0002_m1_retention_indexes.sql
pytest tests/unit/monitoring -q
pytest tests/integration/monitoring -m integration -q
mypy src/talkex/monitoring ; ruff check src/talkex/monitoring tests ; ruff format --check src/talkex/monitoring tests
```

### Acceptance Criteria
- [ ] All suites green (unit + integration), including `test_retention_drops_old_chunks_and_query_uses_index` (the Goal metric)
- [ ] Zero type errors; zero lint warnings
- [ ] M0 e2e still green (no regression)
- [ ] Failure scenarios exercised (pool exhaustion queues; reset surfaces typed error)

### If Validation Fails
1. Classify plan-caused vs pre-existing.
2. Fix plan-caused failures.
3. Re-run the chain.
