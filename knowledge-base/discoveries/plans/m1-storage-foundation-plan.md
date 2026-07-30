# Discovery Plan: M1 Storage Foundation — Productionizing ADR-005

> **Version 1.0** — Investigate how the reference project `chatwoot` designs query-aligned Postgres
> indexing, full-text search, paginated/indexed query paths under load, and connection-pool sizing, so
> the M1 blueprint can productionize the ADR-005 storage spine (30-day retention/compression, pgvector +
> BM25 indexes, multi-stream fan-in). Honest scope note: the peers do NOT use TimescaleDB or pgvector —
> those specifics are sourced from ADR-005 + official docs, not the clones (recorded in ADR D3).

**Slug:** `m1-storage-foundation`
**Owner:** paulohenriquevn
**Created:** 2026-07-29
**Time budget:** 2h (chatwoot only — see ADR D1)

## Context

M1 productionizes the storage the M0 walking skeleton stubbed (`ROADMAP.md § M1`): Timescale hypertables
with 30-day retention + compression, pgvector + BM25 (`pg_search`/`tsvector`) indexes, and multi-stream
fan-in ingest. M0's blueprint recorded honest gap **G1** — no peer uses TimescaleDB, so hypertable
lifecycle is unattested by the clones (`knowledge-base/discoveries/blueprints/m0-walking-skeleton-realtime-monitoring-blueprint.md`
§"Honest gaps"). This discovery mines `chatwoot` for the *Postgres* patterns that DO transfer (indexing,
full-text, indexed query paths, pool sizing) and explicitly defers the Timescale/pgvector specifics to
ADR-005 + official docs. Constrained by `.claude/rules/architecture.md` (DIP: infra adapters) and
`.claude/rules/testing.md` (integration tests against a real DB).

## Objective

Enable the M1 blueprint to decide, with evidence, **the index design for `turns`/`alerts`, the BM25/full-text
strategy, the paginated indexed query path, and the connection-pool sizing** — reusing chatwoot's Postgres
patterns and deferring Timescale/pgvector specifics to ADR-005.

- [ ] All research questions answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison populated
- [ ] At least one concrete decision proposal per question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/chatwoot/` | `app/models/`, `app/finders/`, `db/`, `config/` | The only peer with rich production Postgres indexing / full-text / pool patterns |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/chatwoot/` — `app/javascript/`, `app/views/`, enterprise dirs | Frontend/enterprise, not storage |
| `knowledge-base/references/livekit-agents/`, `ai-powered-call-center-intelligence/` | No production RDBMS indexing/pool patterns relevant to M1 |
| `knowledge-base/references/portuguese-bert/`, `portuguese-nlp/` | Sentiment ML (M2), not storage |
| TimescaleDB hypertable lifecycle + pgvector index specifics | Unattested in any clone (ADR D3) — sourced from ADR-005 + official docs |

## ADRs

### D1 — Time budget + stop conditions

**Decision:** chatwoot: 2h (single-project deep dive — it is the only peer with relevant patterns).

**Rationale:** the other clones have no production RDBMS indexing/pool patterns; concentrating the budget on chatwoot maximizes signal.

**Stop condition — per question:** after 3 empty query-variant retries, mark BLOCKED "Fase A exhausted"; never fabricate a Fase B answer (Unbreakable Rule 3).

**Stop condition — per project:** on budget exhaustion, mark remaining questions BLOCKED "budget exhausted"; emit `<promise>BLUEPRINT_BLOCKED</promise>` if any remain blocked.

### D2 — Investigation depth

**Decision:** Read the index/finder/pool definitions in full (they encode the pattern); Grep the deps.

**Consequences:** deeper read on `conversation_finder.rb`, `article.rb`, `schema.rb`; cheap scan on `Gemfile`/`database.yml`.

### D3 — Timescale/pgvector peer-coverage gap (honest deferral)

**Decision:** the Timescale hypertable lifecycle (chunking, `add_retention_policy`, compression, continuous
aggregates) and pgvector index tuning are NOT attested by any clone; they are sourced from
`docs/adr/ADR-005-online-storage-realtime-monitoring.md` + official Timescale/pgvector docs, NOT this discovery.

**Rationale:** honesty (Unbreakable Rule 3) — fabricating peer citations for Timescale features would fail the
`discover-confidence` fabricated-citation cap and mislead the blueprint.

**Consequences:** this discovery covers the transferable Postgres patterns; the Timescale/pgvector core is an
explicit blueprint input from ADR-005, flagged as a residual gap.

## Research Questions

| # | Question | Corner | Reference project(s) | Fase A (broad map) | Fase B (deep Read) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | How does chatwoot design query-aligned composite + GIN trigram indexes for a filtered supervisor query? | techniques | `knowledge-base/references/chatwoot/` | Grep `t.index`/`gin` in `db/schema.rb` + `db/migrate/20230426130150_init_schema.rb` | Read `db/schema.rb` conversations block (`:799` composite `account_id,inbox_id,status,assignee_id`; `:1171` GIN trigram on `messages.content`) + `init_schema.rb:370` | Index design table → informs `turns`/`alerts` index design |
| Q2 | How does chatwoot implement full-text/BM25 (pg_search on Article) vs the ILIKE+gin_trgm fallback? | techniques | `knowledge-base/references/chatwoot/` | Grep `pg_search`/`pg_search_scope`/`ILIKE` in `app/models/` + `app/finders/` | Read `app/models/article.rb:37,:89` (`pg_search_scope` tsearch weighting) + `app/finders/conversation_finder.rb:156` (ILIKE fallback) | BM25 strategy decision (pg_search vs tsvector vs trigram) with honest tradeoff |
| Q3 | How does the paginated indexed query path avoid N+1 and count efficiently under load (ingest×query)? | techniques | `knowledge-base/references/chatwoot/` | Grep `page`/`per`/`includes`/`FILTER` in `app/finders/conversation_finder.rb` | Read `conversation_finder.rb:108-225` (index-aligned `where` + pagination), `:188-192` (filter-aggregate count), `:212` (eager-load) | Query-path pattern → informs M1 ingest×query contention mitigation |
| Q4 | How does chatwoot spec the finder/model query behavior against a real DB? | tests | `knowledge-base/references/chatwoot/` | Glob `spec/finders/conversation_finder_spec.rb`, `spec/models/conversation_spec.rb` | Read both specs; capture DB-fixture + assertion style | Test template for M1's repo/query integration tests |
| Q5 | What Postgres extensions + search deps does chatwoot enable/pin? | deps | `knowledge-base/references/chatwoot/` | Grep `enable_extension` in `init_schema.rb:4-7`; `pg`/`pg_search` in `Gemfile` | Read each match; extract extension + version | Extension/dep table → informs M1 extension enablement (pg_trgm, pgvector, timescaledb) |
| Q6 | How is the connection pool sized vs async concurrency? | tools | `knowledge-base/references/chatwoot/` | Read `config/database.yml:2-10` (pool) + `config/sidekiq.yml` (concurrency) | Read both; capture the pool==concurrency alignment | Pool-sizing recipe → informs M1 ingest×query pool tuning (drawback R3 from ADR-005) |

## Coverage Matrix

| Corner | Questions mapped | Status |
|---|---|---|
| Integration tests | Q4 | Covered |
| Dependencies | Q5 | Covered |
| Tools | Q6 | Covered |
| Techniques | Q1, Q2, Q3 | Covered |

**Coverage: 4/4 corners covered (100%)**

## Halt-loop Checkpoints

| Checkpoint | Assertion | Action if fails |
|---|---|---|
| Before answering Qx | Every cited `knowledge-base/references/chatwoot/{path}` exists | Mark Qx BLOCKED "path not found", continue |
| Per-question Fase A budget | ≥1 hotspot OR 3 retries | Mark BLOCKED "Fase A exhausted"; continue |
| Timescale/pgvector question | If a question drifts into Timescale/pgvector specifics with no peer source | STOP — that is ADR D3 territory (ADR-005 + docs), not a peer citation |
| Before promising complete | All 4 corners populated | Refuse promise, continue |

## Acceptance Criteria

- [ ] All research questions answered OR explicitly BLOCKED with reason
- [ ] All four corners populated in the blueprint
- [ ] Every citation resolves to a real `knowledge-base/references/{...}` path
- [ ] ≥1 ADR synthesizes M1 storage decisions; the Timescale/pgvector gap (D3) is explicit
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m1-storage-foundation-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations (esp. no fake Timescale/pgvector peer citations — ADR D3)
- [ ] Coverage Matrix 100%
- [ ] ADRs cite `.claude/rules/architecture.md` (DIP), `.claude/rules/testing.md` (real-DB integration), `docs/adr/ADR-005`
