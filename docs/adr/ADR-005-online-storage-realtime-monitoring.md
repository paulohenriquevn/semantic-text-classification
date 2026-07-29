# ADR-005: Online Storage for Real-Time Fleet Monitoring

## Status

Accepted — pending a measured pilot (see *Validation*). Applies to the **product /
fleet-monitoring platform**, not to the dissertation experiments, which keep their
current JSONL + in-memory stack unchanged.

## Context

TalkEx is the **fleet platform** that consumes transcriptions produced on-device by
Macaw Voice (`../jvscribe`), whose own PRD explicitly scopes the fleet platform, LGPD
compliance, and UI *out*. The product is **real-time monitoring of live call-center
transcriptions**, not batch analysis of a static corpus.

Requirements that drive the storage decision:

- **Streaming ingest** from ~1,000 attendant laptops. Estimated sustained rate
  `[ESTIMATED]`: ~150–300 turns/second, bursty; hundreds of millions of turns/month.
  Fan-in of many machines implies backpressure and offline buffering when a laptop drops.
- **Online hybrid retrieval** (the existing BM25 + ANN + fusion design) with a p95
  target < 200 ms (Web-Demo PRD).
- **Windowed real-time analytics** — live rates/counts per queue/domain over recent time.
- **Retention: 30 days for raw data, then purge.** This is a deliberate constraint (also
  a good LGPD data-minimization posture), not an oversight.
- **Model retraining** and long-term business trends still need *some* data to survive
  the purge.

## Decision

**One Postgres/TimescaleDB instance is the online spine for the 30-day hot window; a
cheap object store holds what must survive the purge.** Concretely:

1. **Single online engine — Postgres + extensions:**
   - **Semantic:** `pgvector` (HNSW) for ANN.
   - **Lexical:** real BM25 via `pg_search` (ParadeDB/tantivy) *or* native `tsvector`/GIN
     as the conservative fallback. `ts_rank` is **not** BM25 — this is called out so the
     choice is made with eyes open.
   - **Metadata + relational:** filter (`domain`, `channel`, `speaker`) in the same query,
     cheaply, before the vector step.
   - **Time-series / monitoring:** TimescaleDB hypertables + **continuous aggregates** for
     live dashboards (incremental rollups, no full scan).
   - This collapses the 3-piece hybrid retriever into a single engine and query.

2. **Hot/cold is a chunk lifecycle inside ONE engine, not two databases:**
   recent chunks uncompressed (fast ingest + live point queries = *hot*); older chunks
   compressed columnar (days 3–30, analytical = *warm*); `add_retention_policy` **drops
   whole chunks** at 30 days (purge = drop partition, not row-by-row `DELETE` — no vacuum
   bloat). Introducing ClickHouse just for a 30-day window is rejected as over-engineering.

3. **What survives the purge → object store (Parquet):**
   - **Continuous-aggregate rollups** (kilobytes/megabytes) kept well beyond 30 days as the
     long-term business memory — derived, not raw, so retention of the raw data is unaffected.
   - **An anonymized/pseudonymized sample** destined for labeling/retraining, exported
     *before* the chunk is dropped. Raw transcripts are **not** retained long-term (LGPD
     minimization); only anonymized retraining data leaves the hot store.

4. **The cascade keeps ANN load low.** Cheap lexical/rule predicates (the existing DSL
   engine) fire on each incoming turn with ~zero cost and no embedding; the vector index is
   hit only for semantic categories and "similar past calls" — a QPS far below the raw turn
   rate. This is what makes a single Postgres viable under the ingest rate.

The earlier "one column per intent/feature" idea (Leitura A) is adopted as a **wide feature
hypertable** with continuous aggregates. Splitting a dense embedding into "semantic columns"
(Leitura B) is **rejected** — embedding dimensions are not individually interpretable and
distance is computed over the whole vector.

## Alternatives Considered

- **Qdrant (embedded/server) + ClickHouse + separate BM25 service** — best-in-class at each
  axis, but three operational systems for a 30-day window. Rejected: complexity unjustified
  at this retention/scale.
- **Full polyglot from day one** — deferred until the pilot shows Postgres cannot hold p95.
- **ClickHouse as the single store** (see ADR discussion) — excellent OLAP, but brute-force
  vector search and immature ANN indexing make online p95 < 200 ms risky. Rejected as the
  *online* store.

## Consequences

**Positive**

- One operational surface for metadata + lexical + semantic + time-series; simpler, cheaper.
- Purge is native and O(drop-partition); no `DELETE`/vacuum bloat.
- 30-day retention **bounds the pgvector HNSW index** → predictable memory and latency.
- Strong LGPD posture (raw data minimized; only anonymized derivatives persist).
- Retraining and long-term trends preserved via the object-store carve-out.

**Negative / risks (must be measured, not assumed)**

- Postgres row-store is **not** ClickHouse for heavy *ad-hoc* OLAP over the full window;
  Timescale compression + continuous aggregates narrow but do not close the gap.
- **Extension coexistence is unverified** `[TO VERIFY]`: `pg_search` + TimescaleDB + `pgvector`
  in one instance is not asserted to work cleanly. If they conflict, `tsvector` is the BM25
  fallback and/or lexical stays in-app.
- High-rate **ingest contends with concurrent ANN queries** on the same instance; may force
  primary-for-ingest + read-replica-for-query. Benchmark before committing.
- `pg_search`/ParadeDB is younger than the incumbent BM25 path — maturity risk.

## Validation (pilot — decision is not final without it)

Aligned with both repos' evidence discipline ("architecture is measured, not chosen by
conviction"):

1. Synthetic ingest at 300 turns/s into a Timescale hypertable with a 30-day retention policy.
2. `pgvector` HNSW over ~10M vectors × 384 dims.
3. Measure **hybrid-retrieval p95** and **continuous-aggregate latency** under concurrent
   ingest + query.
4. Compare against the current baseline (embedded Qdrant + in-memory BM25) — the mandatory
   BM25/baseline comparison.

If p95 < 200 ms holds under concurrent load, the single-Postgres design wins on operational
simplicity. If it degrades, split ingest/query onto a replica before escalating to polyglot.
