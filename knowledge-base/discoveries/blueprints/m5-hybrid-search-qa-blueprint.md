# Blueprint: M5 Hybrid Search & QA over 30 days

**Slug:** `m5-hybrid-search-qa`
**Date:** 2026-07-30
**Plan reference:** `knowledge-base/discoveries/plans/m5-hybrid-search-qa-plan.md`
**Owner:** kael-okonkwo (NLP Engineer)

## Executive summary

Two peers were mined to lock the M5 DB-side hybrid-retrieval architecture. **chatwoot** is the production analog: it runs criterion/metadata-filtered conversation search **entirely on Postgres** — raw `ILIKE` for conversations, `tsvector`/GIN (`content @@ to_tsquery`) for messages, with OpenSearch (searchkick) as an **opt-in advanced tier that gracefully falls back to SQL** (`app/services/search_service.rb:59-64`). Its structured filter model is a JSONB-persisted predicate DSL (`custom_filters.query :jsonb`, `app/models/custom_filter.rb:8`) compiled into parameterized SQL fragments (`app/services/filter_service.rb:24-44,181-186`), and it is tested against a **real Postgres with FactoryBot factories** — never a mocked engine for the SQL paths (`spec/services/conversations/filter_service_spec.rb`). Its search indexes are **GIN (`gin_trgm_ops` + tsvector) built `CONCURRENTLY`** (`app/jobs/migration/add_search_indexes_job.rb:6-15`).

**ai-powered-call-center-intelligence** is the deliberate no-semantic-retrieval contrast: it has **no vector/ANN store** (bounded grep verdict in Q4) — retrieval is entirely LLM-mediated via a single GPT call (`backend/gpt_analysis.py:28-35`), and it offloads all query/analytics load to a **separate DuckDB columnar file** decoupled from the ingest path (`analytics/duckdb_loader.py:13-25`).

The synthesis: M5 keeps the **single-engine Postgres+extensions** design (ADR-005), runs a **DB-side hybrid** of `ts_rank` (GIN) ⊕ pgvector HNSW (`<=>`) fused with the already-shipped `reciprocal_rank_fusion` (`src/talkex/retrieval/fusion.py:30`), expresses criterion filters as a **JSONB predicate model compiled to parameterized SQL** (chatwoot's proven pattern), tests it against a **real Timescale instance with factories**, and maintains **GIN + HNSW indexes built `CONCURRENTLY`** — always benchmarked against a BM25-only baseline (`docs/KB.md:439`).

## Context

M5 (`ROADMAP.md § M5`) must deliver QA hybrid BM25 + ANN search over the 30-day Timescale window with metadata filters (p95 < 200 ms under concurrent ingest), search-by-criterion (compliance/script/sentiment/intent) with evidence, and a label-persist action for retraining. The substrate shipped in M1: `TimescaleReadRepository` with keyset pagination (`src/talkex/monitoring/infrastructure/read_repo.py:22-38`), a generated PT-BR `tsvector`+GIN, a trigram GIN on `raw_text`, and an embedding column + pgvector HNSW path (`docs/adr/ADR-005-online-storage-realtime-monitoring.md:34-43`). The fusion math already exists in-repo (`src/talkex/retrieval/fusion.py:30,75`), but `SimpleHybridRetriever` sources candidates **in-memory** (`src/talkex/retrieval/hybrid.py:129,139`). The open gap M5 closes: **how to run hybrid retrieval at the database level** (candidates from `ts_rank` + pgvector, fused, metadata-filtered, keyset-paginated) under ingest×query contention, and how to test + index it. This discovery closes the "how do mature peers do criterion-filtered conversation search + test it + index it" gap before we lock the M5 architecture.

## Objective

Decide the M5 hybrid-search architecture (query strategy, index set, fusion, filter model, test tier, index maintenance) from peer evidence. This blueprint answers all 7 research questions with `path:line` citations, populates the cross-cutting comparison, and issues ADRs synthesizing the query/index/fusion/filter/test decisions for `/to-plan`.

## Coverage Corner 1 — Integration Tests

**(Q5) How chatwoot tests search/filter against a real Postgres.**

chatwoot's search/filter specs run against a **real Postgres instance with FactoryBot factories seeding rows** — the SQL paths are exercised end-to-end, never stubbed. Only the external OpenSearch engine is stubbed.

Evidence:

| Spec example | Fixture/factory | DB posture | Assertion style | Citation |
|---|---|---|---|---|
| "searches across message content and return in created_at desc" | `create(:message, ... content: 'harry is cool')` real rows | **Real Postgres** — runs the actual `ILIKE`/`to_tsquery` | asserts exact ordered id list `eq([message2.id, message.id])` | `spec/services/search_service_spec.rb:80-91` |
| "uses GIN search when search_with_gin feature is enabled" | factory messages + `feature_enabled?('search_with_gin')` stub | **Real Postgres** GIN path; `and_call_original` | asserts the GIN method is actually called | `spec/services/search_service_spec.rb:108-117` |
| "returns same results regardless of search type" | factory messages | **Real Postgres** — runs GIN then LIKE, compares | `expect(gin_results).to match_array(like_results)` (behavioral equivalence) | `spec/services/search_service_spec.rb:119-138` |
| "filter conversations by additional_attributes and status" | `create(:conversation, ... additional_attributes:, status:)` | **Real Postgres** — runs the compiled JSONB SQL | count-equals against an independent AR query | `spec/services/conversations/filter_service_spec.rb:72-77` |
| "rejects invalid created_at comparison values" (SQLi guard) | malicious `"...OR (SELECT pg_sleep(5))..."` value | **Real Postgres** | `raise_error(CustomExceptions::CustomFilter::InvalidValue)` — a **negative-case** test asserting the typed error | `spec/services/conversations/filter_service_spec.rb:530-543` |
| "base_relation returns all for admins / filters by membership" | factory conversations across 2 inboxes | **Real Postgres** | count-equals on permission-scoped result | `spec/services/conversations/filter_service_spec.rb:649-679` |
| custom_filter invalidation | `create(:custom_filter, ...)` | **Real Postgres + Redis store** | `change { store.filter_version(...) }.by(1)` | `spec/models/custom_filter_spec.rb:12-57` |

Only the OpenSearch tier is mocked (`allow(Message).to receive(:search).and_return([])`, `spec/services/search_service_spec.rb:463`) and the fallback path is tested by **raising** a real `Faraday::ConnectionFailed` then asserting SQL results come back non-empty (`spec/services/search_service_spec.rb:737-753`). Filter combination is proven by seeding time/sender/inbox variants and asserting inclusion/exclusion (`spec/services/search_service_spec.rb:228-239`).

**M5 implication:** the integration tier runs against a **real Timescale/Postgres** (matching `.claude/rules/testing.md` § 2 — "repositories against a real DB"). Seed turns with factories, assert (a) exact hybrid ranking on a known corpus, (b) GIN-vs-baseline behavioral equivalence on the lexical leg, (c) a **negative-case** test that a malformed/injection criterion raises a typed `EngineError` subclass, (d) filter-combination inclusion/exclusion. Mock nothing on the SQL path; if a reranker/embedder boundary exists, stub only that external call and test its failure→fallback like chatwoot's searchkick fallback.

## Coverage Corner 2 — Dependencies

**(Q3) chatwoot's full-text stack — which engine searches conversations/messages, and version.**

Both gems exist in the Gemfile — `gem 'searchkick'` (`Gemfile:72`) and `gem 'pg_search'` (`Gemfile:166`) — but they serve **different entities**. **CHECKPOINT EC-1 (named engine + path:line):**

- **Conversation search → raw Postgres `ILIKE`** over joined contact fields + `display_id`, no gem, no tsvector: `app/services/search_service.rb:33-37` (`contacts.name ILIKE :search OR ... conversations.display_id ... ILIKE`).
- **Message search → raw Postgres `tsvector`/GIN** via `content @@ to_tsquery(?)` (`app/services/search_service.rb:87`), with a plain `ILIKE` fallback (`:101`) and an **opt-in** OpenSearch tier. The OpenSearch tier is the **searchkick** gem — `Message` declares `searchkick callbacks: false if ChatwootApp.advanced_search_allowed?` (`app/models/message.rb:42`), invoked as `Message.search` only under the `advanced_search` feature flag (`app/services/search_service.rb:66-70`), and it **falls back to GIN/LIKE on any Searchkick/Elasticsearch error** (`app/services/search_service.rb:59-64`).
- **The `pg_search` gem is NOT on the conversation/message path.** `pg_search_scope` is declared **only** on the Article/help-center model (`app/models/article.rb:89`); `grep` finds no `pg_search`/`PgSearch` in `app/models/message.rb` or `app/models/conversation.rb`. So the conversation/message full-text stack is **raw SQL (`ILIKE` / `to_tsquery` on GIN) + optional searchkick/OpenSearch**, never the `pg_search` gem.

Versions: the Gemfile pins both gems **without a version constraint** (`gem 'searchkick'` and `gem 'pg_search'` are bare — `Gemfile:72,166`); the concrete resolved version lives in `Gemfile.lock` (not read in this pass — see Honest gaps).

**M5 implication:** validates ADR-005's pg-native default (`docs/adr/ADR-005-online-storage-realtime-monitoring.md:35-43`). The closest production peer runs conversation/message full-text **on Postgres itself** and treats the external engine as an optional, fail-open accelerator. M5 should ship the pg-native `ts_rank`/GIN leg first and keep any external/reranker tier behind a flag with SQL fallback — never a hard dependency.

**(Q4) Does ai-powered use a vector store / ANN library, or is retrieval purely LLM-mediated?**

**CHECKPOINT EC-2 (bounded grep verdict):** *No local vector/ANN dependency in `requirements.txt` and no embedding/vector call found in `backend/`.* Greps run: `faiss|pgvector|chroma|qdrant|azure.*search|embedding|weaviate|milvus|pinecone|annoy|hnsw|sentence-transformers` over `requirements.txt` → **NO MATCH**; `faiss|pgvector|chroma|qdrant|azure.*search|embedding|vector` over `backend/` → **NO MATCH**. `requirements.txt` contains only `openai==0.28.0`, `transformers` (for sentiment, not retrieval), `scikit-learn`/`xgboost` (classifiers), and `duckdb` — no ANN/vector-store dependency (`requirements.txt:1-30`).

Retrieval is **entirely LLM-mediated**: a transcript is passed whole to a single GPT chat completion with a system+telecom prompt, returning insights as a natural-language bullet list (`backend/gpt_analysis.py:22-35`). There is no embedding step, no similarity search, no index — `backend/main.py` transcribes with Whisper then calls `analyze_transcript` directly (`backend/main.py:45-50`).

**M5 implication:** this is the anti-pattern M5 rejects. A "just ask the LLM" design has no sub-second retrieval, no ranking, no evidence trace, and no scalable filter — it cannot meet M5's p95 < 200 ms or "every prediction carries evidence" axiom. It confirms the value of a real hybrid index: M5 keeps embeddings in pgvector HNSW and does ANN in-DB (`docs/adr/ADR-005-online-storage-realtime-monitoring.md:35`), reserving LLMs for offline labeling only (CLAUDE.md § Key Design Axioms).

## Coverage Corner 3 — Tools

**(Q6) How chatwoot builds/maintains search indexes, and what shape.**

A single background job creates the search indexes, all built **`CONCURRENTLY`** (no table lock) — `app/jobs/migration/add_search_indexes_job.rb`:

| Index | Table.column(s) | Type | Concurrent? | Citation |
|---|---|---|---|---|
| composite btree | `messages(account_id, inbox_id)` | default btree (tenant+scope filter) | `algorithm: :concurrently` | `add_search_indexes_job.rb:6` |
| trigram GIN | `messages(content)` | `using: 'gin', opclass: :gin_trgm_ops` | `algorithm: :concurrently` | `add_search_indexes_job.rb:7` |
| trigram GIN | `contacts(name,email,phone_number,identifier)` | `using: 'gin', opclass: :gin_trgm_ops`, named | `algorithm: :concurrently` | `add_search_indexes_job.rb:8-15` |

Notes: the job is a scheduled ActiveJob (`queue_as :scheduled_jobs`, `:2`) marked self-retiring ("Delete migration and spec after 2 consecutive releases", `:1`). The message full-text uses **`gin_trgm_ops`** (pg_trgm) for the `ILIKE`/substring path; the `tsvector`/`to_tsquery` path in `search_service.rb:87` relies on a GIN over the tsvector (the composite index + trgm here accelerate the filter/LIKE legs).

**M5 implication:** matches ADR-005's index set (`docs/adr/ADR-005-online-storage-realtime-monitoring.md:34-43`). M5's maintenance plan: build **GIN over the generated PT-BR `tsvector`** (for `ts_rank`) + **HNSW over the embedding column** (for `<=>`) + a **tenant/time composite btree** aligned to the keyset order (`(conversation_id, created_at DESC)`, already precedent in `read_repo.py:4-5,25`), all `CREATE INDEX CONCURRENTLY` so index builds never block ingest. Validate each with `EXPLAIN (ANALYZE, BUFFERS)` to confirm index usage under the p95 target.

**(Q7) How ai-powered separates the analytics/query path from ingest.**

ai-powered **physically separates query from write**: analytics reads from a **standalone DuckDB columnar file** (`data/call_summary.db`) that is loaded from finished ingest artifacts (transcript + redactions + GPT insights JSON), not from a live transactional store — `analytics/duckdb_loader.py:11-25`. The loader `INSERT`s a full call package after processing (`duckdb_loader.py:38-41`), and the query surface is a thin `query(sql)` helper returning a DataFrame for notebooks/charts (`duckdb_loader.py:46-47`), consumed by Altair chart builders (`analytics/powerdash_components.py:10-22`). The ingest path (`backend/main.py`) never touches DuckDB — analytical read load is fully offloaded from the write path.

**M5 implication:** the same *principle* (offload analytical/query load from the hot write path) applies, but M5 does it **within one Postgres/Timescale engine** rather than a second store — ADR-005 rejects a separate analytics DB as over-engineering for a 30-day window and instead uses **Timescale hot→warm chunk lifecycle** (compressed columnar for days 3-30) as the read-offload mechanism (`docs/adr/ADR-005-online-storage-realtime-monitoring.md:45-49`). Concretely, M5 mitigates ingest×query contention with **pool separation / read sizing** on `MonitoringPool` (`read_repo.py:13,19`) and reads over the compressed chunks, not a copied columnar store.

## Coverage Corner 4 — Techniques

**(Q1) How chatwoot composes a full-text query with structured metadata filters in one search.**

Two complementary techniques:

1. **Inline text + scoped filters (search path).** `SearchService#filter_messages` builds a base relation scoped by tenant + time (`current_account.messages.where('created_at >= ?', 3.months.ago)`, `search_service.rb:108`) + inbox ACL (`:109`), then layers filters (`apply_message_filters` → time/sender/inbox, `:113-119`), then appends the text predicate: GIN `content @@ to_tsquery(?)` with a phrase-distance tsquery (`search_query.split.join(' <-> ')`, `:84-87`) or `ILIKE` (`:101`). Text match and filter predicates are **AND-composed on one ActiveRecord relation** and keyset-ordered + paginated (`.reorder('created_at DESC').page(...).per(15)`, `:88-90`).

2. **JSONB predicate DSL (filter path).** `Conversations::FilterService` compiles a client-supplied `payload` array of `{attribute_key, filter_operator, values, query_operator}` hashes into a single parameterized SQL string:

| Filter dimension | Predicate mechanism | Citation | Maps to M5 criterion |
|---|---|---|---|
| enum status/priority/message_type | value coercion → `IN (:value_n)` | `filter_service.rb:50-52,163-167` | compliance state, disposition |
| free-text contains | `ILIKE ANY (ARRAY[:value_n])` | `filter_service.rb:29-31,169-173` | script/keyword evidence |
| presence | `IS NOT NULL` / `IS NULL` | `filter_service.rb:32-35` | "has sentiment", "has intent" |
| numeric/date range | `< / > :value_n` with typed coercion | `filter_service.rb:36-37,85-98` | sentiment score threshold, time window |
| `days_before` | rewrites to `is_less_than` on a computed date | `filter_service.rb:100-108` | 30-day window slices |
| labels/tags | correlated `EXISTS (SELECT ... FROM taggings ...)` subquery | `filter_service.rb:118-139` | intent/label tags |
| JSONB custom attr | `custom_attributes ->> 'key'` compared | (spec) `filter_service_spec.rb:272-281` | arbitrary analyzed fields |

Composition + safety: predicates are joined per-`query_operator` (`AND`/`OR`) in `query_builder` (`filter_service.rb:181-186`) and passed as `base_relation.where(@query_string, @filter_values)` — **values are always bound parameters, never interpolated**, and range values are type-coerced (`coerce_lt_gt_value`, `:150-161`) raising `InvalidValue` on bad input (the SQLi guard proven in `filter_service_spec.rb:530-543`). The base relation is permission-filtered first (`Conversations::PermissionFilterService`, `conversations/filter_service.rb:26-36`).

**M5 implication:** adopt the **JSONB predicate-DSL model** for search-by-criterion. M5 already has a rules DSL (`src/talkex/rules/`); the QA search filter can reuse that shape — a criterion (compliance/script/sentiment/intent) becomes a bound predicate AND-composed with the `ts_rank`/pgvector candidate query. Critically: **bind every value as a parameter and type-coerce ranges**, replicating chatwoot's injection defense (`.claude/rules/error-handling.md` — typed errors at the boundary). Store saved criteria as JSONB like `custom_filters.query` (`custom_filter.rb:8`).

**(Q2) What analyzed fields ai-powered persists/queries, and through what surface.**

| Field | Producer | Query surface | Citation |
|---|---|---|---|
| `transcript` (raw) | Whisper `whisper_model.transcribe` | DuckDB `calls.transcript` TEXT | `backend/main.py:45-46`; `duckdb_loader.py:18-25` |
| `redacted_transcript` | Presidio (redaction step, loaded from file) | DuckDB `calls.redacted_transcript` | `duckdb_loader.py:32-33` |
| `insights` (intent, tone, churn risk, sentiment, issue category, resolution tactic, satisfaction) | single GPT call, free-form bullet text | DuckDB `calls.insights` JSON column | `backend/gpt_analysis.py:22-35`; `duckdb_loader.py:22` |
| derived: `sentiment_score`, `issue_category`, `resolution_tactic`, `satisfaction_score`, `agent_id` | assumed columns in the analytics DataFrame (chart inputs) | Altair charts over DuckDB query results | `analytics/powerdash_components.py:10-22,25-39,42-55,58-72` |

The analysis surface is **coarse**: GPT returns *unstructured natural-language bullets* (`gpt_analysis.py:18-20,35`), and the "fields" (`sentiment_score`, `issue_category`, …) only become queryable columns downstream in the analytics DataFrame — there is no per-turn structured schema at the API. The API returns `{transcript, insights}` as free text (`backend/main.py:53-56`).

**M5 implication:** the *taxonomy* is a useful checklist of QA-relevant fields (sentiment, intent, issue category, resolution tactic, satisfaction/compliance) — M5 should expose exactly these as **structured, per-turn, queryable criteria with scores and text evidence**, not free-text bullets. Where ai-powered stops at "ask GPT and eyeball bullets", M5 persists typed classification outputs (label, score, confidence, threshold, model version, text evidence — CLAUDE.md § axiom) filterable via the Q1 predicate DSL and rankable via the hybrid query.

## Cross-cutting Comparison

| Dimension | chatwoot | ai-powered-call-center-intelligence | M5 decision |
|---|---|---|---|
| Full-text engine (conv/msg) | Postgres `ILIKE` (conv) + `tsvector`/GIN `to_tsquery` (msg), searchkick optional w/ fallback (`search_service.rb:36,87,59-64`) | none — LLM only (`gpt_analysis.py:28`) | pg-native `ts_rank`/GIN, external tier optional (ADR-005:35-43) |
| Semantic/ANN store | none (lexical only) | none — no vector dep (Q4 grep) | pgvector HNSW `<=>` in-DB |
| Fusion | n/a (single leg) | n/a | reuse `reciprocal_rank_fusion` (`fusion.py:30`), DB candidates |
| Filter model | JSONB predicate DSL → bound SQL (`filter_service.rb:24-44,181-186`) | none (free-text bullets) | adopt JSONB predicate DSL, reuse `rules/` shape |
| Injection defense | bound params + type coercion (`filter_service.rb:150-161`) | n/a | replicate: bind + coerce + typed error |
| Query/write separation | single Postgres | separate DuckDB columnar file (`duckdb_loader.py:13`) | single Timescale, hot→warm chunk offload (ADR-005:45-49) |
| Index build | GIN `gin_trgm_ops` + composite btree, `CONCURRENTLY` (`add_search_indexes_job.rb:6-15`) | none | GIN(tsvector) + HNSW + composite btree, all `CONCURRENTLY` |
| Test tier | real Postgres + factories; stub only external engine (`search_service_spec.rb`) | none shipped | real Timescale + factories; exact-ranking + negative-case + fallback |
| Pagination | offset `.page.per(15)` (`search_service.rb:89`) | none | **keyset** (already in `read_repo.py:25`) — better than chatwoot's OFFSET |

## ADRs

### D1 — Query strategy: DB-side hybrid, candidates from `ts_rank` + pgvector, fused in `reciprocal_rank_fusion`

**Decision:** M5 runs the two retrieval legs **inside Postgres** — a lexical leg (`ts_rank` over the generated PT-BR tsvector + GIN, phrase/prefix `to_tsquery`) and a semantic leg (pgvector HNSW `<=>` over the embedding column), each returning a top-K candidate list with rank. The two ranked lists are fused by the already-shipped `reciprocal_rank_fusion` (`src/talkex/retrieval/fusion.py:30`), not re-implemented. The DB query lives behind a domain port implemented by an infrastructure adapter extending `TimescaleReadRepository` (`src/talkex/monitoring/infrastructure/read_repo.py:16-38`).

**Rationale:** chatwoot proves message full-text on pg `to_tsquery`/GIN is production-viable (`app/services/search_service.rb:87`); ADR-005 already collapses the 3-piece retriever into one engine (`docs/adr/ADR-005-online-storage-realtime-monitoring.md:43`). RRF is scale-free (needs no score normalization, `fusion.py:9`), so lexical `ts_rank` and cosine distance can be fused without calibration. Reusing `fusion.py` honors **DRY**; the port/adapter split honors **DIP** (`.claude/rules/architecture.md § 2` — domain defines the interface, infra implements it).

**Alternatives considered:** (a) keep `SimpleHybridRetriever`'s in-memory candidate sourcing (`hybrid.py:129,139`) — rejected: cannot scale to the 30-day window or meet p95 under ingest, materializes rows into app memory. (b) External engine (OpenSearch/searchkick) as primary — rejected: chatwoot itself keeps it optional with SQL fallback (`search_service.rb:59-64`); a hard external dep violates the single-engine ADR-005 and adds an ingest sync burden. (c) `linear_fusion` — deferred: needs min-max normalization (`fusion.py:98-99`), more tuning; RRF is the safer default.

**Consequence:** one round-trip per leg (or one CTE), candidates fused app-side; component `lexical_score`/`semantic_score` preserved for evidence (`fusion.py:19-21`). The M5 plan must decide single-CTE vs two-query; EXPLAIN-validate both.

### D2 — Filter model: JSONB predicate DSL compiled to parameterized SQL, AND-composed with the candidate query

**Decision:** search-by-criterion (compliance/script/sentiment/intent) is expressed as a JSONB predicate list (`{attribute_key, filter_operator, values, query_operator}`), compiled to a **bound** SQL fragment and AND/OR-composed with the D1 candidate query — mirroring chatwoot's `FilterService` (`app/services/filter_service.rb:24-44,181-186`). Saved criteria persist as JSONB, like `custom_filters.query :jsonb` (`app/models/custom_filter.rb:8`). Reuse the existing `src/talkex/rules/` DSL shape where it fits.

**Rationale:** chatwoot's model covers exactly M5's dimensions — enums (`IN`), presence (`IS [NOT] NULL`), ranges (typed `< / >`), JSONB attrs (`->>`), tag `EXISTS` subqueries (`filter_service.rb:50-52,32-35,85-98,118-139`) — a superset of compliance/score/intent filters. **KISS/Don't-Reinvent** (`.claude/rules/parsimony-ladder.md`): a battle-tested predicate model beats a bespoke one.

**Alternatives considered:** (a) hard-coded per-criterion query methods — rejected: N criteria × M operators explodes, violates OCP. (b) accept raw SQL from the client — rejected: injection surface; chatwoot's own SQLi regression test (`filter_service_spec.rb:530-543`) shows why binding+coercion is mandatory.

**Consequence:** M5 must implement value coercion + a typed `EngineError` subclass raised on malformed criteria (`.claude/rules/error-handling.md` — typed errors at the boundary), and never string-interpolate values.

### D3 — Index set: GIN(tsvector) + HNSW(embedding) + composite btree, all built `CONCURRENTLY`

**Decision:** M5 maintains three indexes: a **GIN over the generated PT-BR `tsvector`** (lexical leg), an **HNSW over the embedding column** (semantic leg, `<=>`), and a **composite btree aligned to the keyset order** (`(conversation_id, created_at DESC)` / tenant+time). Every index is `CREATE INDEX CONCURRENTLY`. Each hot query path is EXPLAIN-validated against the p95 < 200 ms target.

**Rationale:** chatwoot builds all search indexes `algorithm: :concurrently` so builds never lock the table under load (`app/jobs/migration/add_search_indexes_job.rb:6-15`) — essential under M5's concurrent ingest. The composite-btree keyset alignment already has in-repo precedent (`read_repo.py:4-5,25`, citing chatwoot's `conversation_finder.rb:108`). ADR-005 already specifies GIN + HNSW (`docs/adr/ADR-005-online-storage-realtime-monitoring.md:34-43`).

**Alternatives considered:** (a) trigram `gin_trgm_ops` for the lexical leg (chatwoot's choice for `ILIKE`, `add_search_indexes_job.rb:7`) — kept as a secondary option for substring/typo search, but the primary leg is tsvector GIN for `ts_rank` relevance. (b) IVFFlat instead of HNSW — rejected per ADR-005 (HNSW chosen). (c) non-concurrent build — rejected: locks ingest.

**Consequence:** index-build and retention (`add_retention_policy` chunk-drop, ADR-005:47-48) run as background maintenance; the plan needs an index-maintenance + EXPLAIN-validation task.

### D4 — Test tier: real Timescale + factories, with exact-ranking, negative-case, and fallback tests

**Decision:** M5's retrieval is tested at the **integration tier against a real Timescale/Postgres**, seeded with factory turns. Required cases: (a) exact hybrid ranking on a known corpus, (b) lexical-leg behavioral equivalence vs a BM25-only baseline, (c) a **negative-case** test asserting a malformed/injection criterion raises the typed error, (d) filter-combination inclusion/exclusion, (e) if any external tier is added, a failure→fallback test. Mock nothing on the SQL path.

**Rationale:** chatwoot tests every SQL path against real Postgres and only stubs the external engine (`spec/services/search_service_spec.rb:80-138,463,737-753`; `spec/services/conversations/filter_service_spec.rb`). Its SQLi negative-case (`filter_service_spec.rb:530-543`) and GIN-vs-LIKE equivalence (`search_service_spec.rb:119-138`) are the exact assertions M5 needs. Matches `.claude/rules/testing.md` § 2 (integration = real DB) and § 4.1 (edge **and** negative cases).

**Alternatives considered:** (a) mock the DB/return canned hits — rejected: ranking + index behavior + injection defense are only real against a live planner. (b) test only happy-path ranking — rejected: § 4.1 requires the negative lens; chatwoot's own suite covers both.

**Consequence:** CI needs a Timescale service container; the BM25-baseline benchmark (`docs/KB.md:439`) becomes an integration-tier fixture, satisfying the "always benchmark hybrid vs BM25-only" axiom (`docs/KB.md:353,439`).

### D5 — Ingest×query contention: single-engine read offload via Timescale chunk lifecycle + pool separation

**Decision:** M5 mitigates ingest×query contention **within one Timescale engine** — reads target the compressed warm chunks (days 3-30) and use a **separate/sized read pool** on `MonitoringPool`, rather than introducing a second analytics store.

**Rationale:** ai-powered offloads all query load to a separate DuckDB file (`analytics/duckdb_loader.py:13-25`) — the right *principle* (decouple query from write) but the wrong *mechanism* for a 30-day hot window. ADR-005 achieves the same offload via hot→warm chunk lifecycle inside one engine and rejects a second store as over-engineering (`docs/adr/ADR-005-online-storage-realtime-monitoring.md:45-49`) — **YAGNI**. `TimescaleReadRepository` already isolates reads behind a pool (`read_repo.py:13,19`).

**Alternatives considered:** (a) a separate columnar analytics DB (ai-powered's literal design) — rejected per ADR-005. (b) single shared pool — risk: query bursts starve ingest connections; sizing/separating the read pool is the cheaper mitigation.

**Consequence:** the plan needs a pool-sizing task and a concurrency/contention test (ingest running while queries execute), measuring p95 under load — the M5 SLO gate.

## Recommendations

1. **Q1/D2 — filter model:** implement a JSONB predicate compiler for QA criteria, reusing `src/talkex/rules/` and chatwoot's operator set (`filter_service.rb:24-44`). Bind all values; type-coerce ranges; raise a typed `EngineError` on malformed input.
2. **Q2 — analyzed fields:** expose sentiment/intent/issue-category/resolution-tactic/compliance as **structured per-turn criteria with score + text evidence** (CLAUDE.md axiom), not free-text — the gap ai-powered leaves open (`gpt_analysis.py:35`).
3. **Q3/D1 — engine:** ship the pg-native `ts_rank`/GIN lexical leg + pgvector HNSW semantic leg; keep any external/reranker tier optional with SQL fallback (chatwoot pattern, `search_service.rb:59-64`).
4. **Q4/D1 — reject LLM-only retrieval:** no ANN/vector dep in ai-powered confirms hybrid-index value; LLMs stay offline-only.
5. **Q5/D4 — tests:** real Timescale + factories; add exact-ranking, BM25-equivalence, negative-case injection, filter-combination, and (if external) fallback tests. Mock nothing on SQL.
6. **Q6/D3 — indexes:** GIN(tsvector) + HNSW(embedding) + keyset-aligned composite btree, all `CONCURRENTLY`; EXPLAIN-validate each hot path vs p95 < 200 ms.
7. **Q7/D5 — contention:** single-engine read offload (warm chunks + sized read pool on `MonitoringPool`); add a concurrent ingest×query p95 load test.
8. **Fusion (D1):** reuse `reciprocal_rank_fusion` (`fusion.py:30`) as the default; benchmark `linear_fusion` only if RRF underperforms. Always benchmark hybrid vs BM25-only baseline (`docs/KB.md:439`).

## Honest gaps

- **Q3 exact gem versions:** the Gemfile pins `searchkick` and `pg_search` **without version constraints** (`Gemfile:72,166`); the resolved versions are in `Gemfile.lock`, which was not read this pass. The engine-selection answer (which gem searches what) is fully resolved; only the numeric version is unread. Low impact — M5 does not depend on chatwoot's gem versions.
- **Q6 tsvector-GIN migration:** `add_search_indexes_job.rb` creates a **trigram** GIN (`gin_trgm_ops`) on `messages.content` (`:7`), which serves the `ILIKE` path; the `to_tsquery`/tsvector GIN used at `search_service.rb:87` is created by a **separate migration in `db/migrate/`** not read in this pass (the job is a one-off backfill, not the canonical schema). The index *shape* (GIN + concurrent) is confirmed; the specific tsvector-column migration file is uncited. Does not block M5 (ADR-005 already specifies the tsvector GIN).
- **Q2 structured schema:** ai-powered's `insights` is free-text (`gpt_analysis.py:35`); the "fields" table's derived columns (`sentiment_score`, etc.) are *assumed* by the chart code (`powerdash_components.py:11`), never persisted as a typed schema. Reported honestly as a coarse/unstructured surface — this is a limitation of the peer, correctly captured, not a gap in evidence.
- No question was BLOCKED; all 7 were answered against the cited files.

## discover-confidence verdict

**SHIPPABLE_WITH_CAVEATS.** All 7 research questions answered with real `path:line` citations across both peers; all 4 coverage corners populated with non-placeholder evidence; both edge-case checkpoints (EC-1 named engine with `path:line`; EC-2 bounded-grep verdict, greps actually run) satisfied; 5 ADRs synthesize the query/index/fusion/filter/test/contention decisions, each citing a project principle (DIP/DRY/KISS/YAGNI) or rule/ADR. Caveats: two minor unread-file gaps (Gemfile.lock versions; the exact tsvector-column migration in `db/migrate/`) — both honestly logged and neither blocks the M5 architecture, which ADR-005 already locks. Feeds `/to-plan` directly.
