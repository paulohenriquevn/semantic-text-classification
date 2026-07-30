# Discovery Plan: M5 — Hybrid Search & QA over 30 days

> **Version 1.0** — Investigate how the cloned peers implement criterion/metadata-filtered conversation search, how they combine lexical full-text with (or without) semantic retrieval, how they test search against a real database, and how they build/maintain search indexes — so we can lock the architecture for M5's DB-side hybrid retrieval (Postgres `ts_rank` GIN + pgvector HNSW `<=>`, fused with our existing `reciprocal_rank_fusion`) over the 30-day Timescale window with p95 < 200 ms under concurrent ingest, plus a QA search-by-criterion surface and a label-persist action for retraining. Reference projects in scope: `chatwoot` (production conversation search/filter + real-DB specs + index tooling) and `ai-powered-call-center-intelligence` (end-to-end transcription→analysis→query shape; a deliberate no-semantic-retrieval contrast). The blueprint output must let us decide the exact query strategy, index set, fusion, test tier, and index-maintenance plan.

**Slug:** `m5-hybrid-search-qa`
**Owner:** kael-okonkwo (NLP Engineer)
**Created:** 2026-07-30
**Time budget:** 5h (per-project breakdown in ADR D1)

## Context

M5 (`ROADMAP.md § M5`) must give QA hybrid BM25 + ANN search over the 30-day window with metadata filters (p95 < 200 ms under concurrent ingest), search-by-criterion (compliance/script/sentiment/intent) with evidence, and a label-persist action for retraining. Much of the substrate already shipped in M1 (`chore(release): v0.3.0`): `src/talkex/monitoring/infrastructure/read_repo.py` (`TimescaleReadRepository`, keyset pagination), a generated PT-BR `tsvector` column + GIN, a GIN trigram index on `raw_text`, and an embedding column + pgvector HNSW index path (per ADR-005, `docs/adr/ADR-005-online-storage-realtime-monitoring.md`). The in-repo `src/talkex/retrieval/` module already implements the fusion algorithms (`fusion.py`: `reciprocal_rank_fusion`, `linear_fusion`) and an in-memory hybrid retriever (`hybrid.py`: `SimpleHybridRetriever`) — but sourcing candidates **in-memory**, not from Postgres. The open gap M5 must close: **how to run the hybrid retrieval at the database level** (candidates from `ts_rank` + pgvector, fused, metadata-filtered, keyset-paginated) under ingest×query contention, and how to test + index it. `docs/KB.md § Hybrid Retrieval` (line 353) and § BM25 baseline (line 439) lock the axiom: always benchmark hybrid against a BM25-only baseline. This discovery closes the "how do mature peers do criterion-filtered conversation search + test it + index it" gap before we lock the M5 architecture. Respects `.claude/rules/architecture.md § 1` (layered boundaries — the DB query is an infrastructure adapter behind a domain port) and `.claude/rules/testing.md` (integration tier against a real DB).

## Objective

Decide the M5 hybrid-search architecture (query strategy, index set, fusion, filter model, test tier, index maintenance) from evidence in the peers. Success criteria for the blueprint:

- [ ] All 7 research questions answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison table populated for `chatwoot` and `ai-powered-call-center-intelligence`
- [ ] Recommendations section provides at least one concrete decision proposal per research question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope (per reference project)

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/chatwoot/` | `app/services/` (search + filter services), `app/models/custom_filter.rb`, `app/jobs/migration/`, `spec/services/`, `spec/models/`, `Gemfile` | Production conversation search/filter, its full-text stack, its real-DB specs, and its index-migration tooling are the closest analog to M5's QA search. |
| `knowledge-base/references/ai-powered-call-center-intelligence/` | `backend/` (main, gpt_analysis), `analytics/` (duckdb_loader, powerdash_components), `requirements.txt` | End-to-end transcription→analysis→query shape; a deliberate contrast (LLM analysis + DuckDB analytics, no semantic retrieval store) to validate our hybrid decision and the ingest×query separation idea. |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/chatwoot/app/javascript/` | Front-end search UX belongs to a later UI slice, not the M5 retrieval backend. |
| `knowledge-base/references/livekit-agents/` | Streaming/ingest peer — already mined for M0 backpressure; irrelevant to search relevance/indexing (deferred via ADR D3). |
| `knowledge-base/references/portuguese-bert/`, `portuguese-nlp/` | Embedding-model corpora, not retrieval-architecture sources; the embedding path is already locked by M1/M2. |
| `knowledge-base/references/*/` build artifacts, `node_modules/`, `vendor/`, `.git/` | Not source of truth. |

## ADRs

### D1 — Time budget + stop conditions

**Decision:** chatwoot: 3h; ai-powered-call-center-intelligence: 2h. Total 5h.

**Rationale:** chatwoot is the closest analog (real conversation search + filter model + real-DB specs + index migration tooling), so it gets the deepest dive; ai-powered is informational — it shapes the end-to-end contract and provides a no-semantic-retrieval contrast, so 2h suffices.

**Alternatives considered:** equal split (rejected — ai-powered has no retrieval store to dig into); chatwoot-only (rejected — loses the end-to-end + contention-separation contrast).

**Stop condition — per question (mandatory):** When a question's Fase A returns empty matches after 3 consecutive retries with different query variants (pattern → kind-based → alternate path → broader scope), mark the question BLOCKED with reason "Fase A exhausted — no hotspots found" and continue. Do NOT pad with unrelated hotspots from another question's scope.

**Stop condition — per project (mandatory):** When a project's time budget is exhausted with N questions pending, mark them BLOCKED with reason "budget exhausted" and continue. If every remaining question is `done` or honestly `blocked`, emit `<promise>BLUEPRINT_BLOCKED</promise>` (never `BLUEPRINT_COMPLETE` from a blocked state).

**Anti-pattern:** NEVER fabricate Fase B answers to close a Fase-A-exhausted question. Honest BLOCKED is required (Unbreakable Rule 3).

**Consequences:** the halt-loop stops per-project on budget exhaustion; blocked questions surface in the blueprint's `## Blocked questions` section as next-discovery seed.

### D2 — Investigation depth

**Decision:** Read service/spec/job files end-to-end (they are short, behavior-dense Ruby/Python); for dependency questions, Grep the manifest then Read the matched lines in context. ast-grep Fase A only where a code-shape map helps (Ruby method/class enumeration); text-shape questions (Gemfile, requirements.txt) skip Fase A.

**Rationale:** search/filter logic lives in a handful of services and their specs; reading them whole captures intent + edge cases that a symbol grep misses. Manifests are text-shape.

**Consequences:** deeper per-file cost, but the files are small; total stays within the 5h budget.

### D3 — Defer livekit-agents and the embedding-corpus peers

**Decision:** Exclude `livekit-agents`, `portuguese-bert`, `portuguese-nlp` from this discovery.

**Rationale:** livekit is a streaming/ingest source (mined for M0 backpressure), not a search-relevance/indexing source; the Portuguese corpora are embedding-model inputs, and the embedding path is already locked by M1/M2. None answers a search-architecture question.

**Consequences:** the four corners are covered by two peers; no corner is left to a peer that cannot answer it.

## Research Questions

| # | Question | Corner | Reference project(s) | Fase A (broad — map) | Fase B (deep — Read at each hotspot) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | How does chatwoot combine a full-text query with structured metadata filters (status, label, assignee, custom attributes) in one conversation search? | techniques | `knowledge-base/references/chatwoot/app/services/search_service.rb`, `app/services/conversations/filter_service.rb`, `app/models/custom_filter.rb` | `ast-grep run -p 'def $NAME($$$)' --lang ruby knowledge-base/references/chatwoot/app/services/search_service.rb` to map the query-building methods; Grep `filter` in `conversations/filter_service.rb` | Read `search_service.rb`, `conversations/filter_service.rb`, `custom_filter.rb` fully; capture how text match + filter predicates are composed into one query and how the filter model is expressed | Prose + a decomposition table: filter dimension → SQL/predicate mechanism → `path:line`, mapped to our compliance/script/sentiment/intent criteria |
| Q2 | In ai-powered, what analyzed fields per conversation are persisted and made queryable (GPT analysis output, sentiment, intents), and through what surface (API/analytics)? | techniques | `knowledge-base/references/ai-powered-call-center-intelligence/backend/main.py`, `backend/gpt_analysis.py`, `analytics/duckdb_loader.py` | `ast-grep run -p 'def $NAME($$$): $$$' --lang python knowledge-base/references/ai-powered-call-center-intelligence/backend/gpt_analysis.py` to map analysis functions | Read `main.py`, `gpt_analysis.py`, `duckdb_loader.py`; capture which fields are produced and how they reach a query surface | Field inventory: field → producer → query surface → `path:line`; informs what evidence/criteria M5 exposes to QA |
| Q3 | What full-text search stack does chatwoot use for conversation/message search — Postgres-native (`pg_search`, GIN/tsvector) vs external (`searchkick`/OpenSearch) — and when is each used? | deps | `knowledge-base/references/chatwoot/Gemfile`, `app/services/search_service.rb` | SKIP Fase A (text-shape). Grep `pg_search`, `searchkick`, `elasticsearch` in `Gemfile`; Grep `search` usage in `search_service.rb` | Read the Gemfile matches (lines 72 `searchkick`, 166 `pg_search`) + how `search_service.rb` invokes them | Stack table: engine → gem+version → which entity it searches → citation; validates our pg-native tsvector choice vs an external engine |
| Q4 | Does ai-powered use any vector store / ANN library for semantic retrieval, or is retrieval purely LLM-mediated — and what does that design choice imply? | deps | `knowledge-base/references/ai-powered-call-center-intelligence/requirements.txt`, `backend/gpt_analysis.py`, `analytics/duckdb_loader.py` | SKIP Fase A (text-shape). Grep `faiss|pgvector|chroma|qdrant|numpy|sklearn|azure.*search` in `requirements.txt`; Grep `embedding|vector|search` in `backend/` | Read `requirements.txt` + the retrieval-relevant backend/analytics files to confirm presence/absence of a semantic store | Dependency verdict (present/absent) + a short design implication for M5's hybrid decision, with citations |
| Q5 | How does chatwoot test its search/filter services against a real Postgres — factories, fixtures, real DB or a mocked engine? | tests | `knowledge-base/references/chatwoot/spec/services/search_service_spec.rb`, `spec/services/conversations/filter_service_spec.rb`, `spec/models/custom_filter_spec.rb` | `ast-grep run -p 'it $$$ do $$$ end' --lang ruby knowledge-base/references/chatwoot/spec/services/search_service_spec.rb` to map example blocks | Read each spec + its setup/factory usage; capture DB posture (real pg vs stub), seeded data, and assertion style | Table: spec example → fixture/factory → DB posture → assertion type → `path:line`; informs M5's integration-test tier against real Timescale |
| Q6 | How does chatwoot build and maintain its search indexes (the migration job that adds GIN/pg_trgm/tsvector indexes), and what index shape does it use? | tools | `knowledge-base/references/chatwoot/app/jobs/migration/add_search_indexes_job.rb` | `ast-grep run -p 'def $NAME($$$)' --lang ruby knowledge-base/references/chatwoot/app/jobs/migration/add_search_indexes_job.rb` to map the index-creation steps | Read the job fully; capture each index (column, type — GIN/trigram/tsvector, concurrency) | Index inventory: index name → column → type → concurrent? → citation; informs M5's index-maintenance + EXPLAIN-validated p95 plan |
| Q7 | How does ai-powered separate the analytics/query path (DuckDB columnar loader) from the ingest/transcription path — is query load offloaded from the write path? | tools | `knowledge-base/references/ai-powered-call-center-intelligence/analytics/duckdb_loader.py`, `analytics/powerdash_components.py` | `ast-grep run -p 'def $NAME($$$): $$$' --lang python knowledge-base/references/ai-powered-call-center-intelligence/analytics/duckdb_loader.py` to map loader functions | Read both files; capture whether analytics reads from a separate store/columnar copy vs the live ingest DB | Prose + a data-path diagram (ingest store vs query store) + citation; informs M5's ingest×query contention mitigation (pool sizing / read offload) |

## Coverage Matrix

| Corner | Questions mapped | Status |
|---|---|---|
| Integration tests | Q5 | Covered |
| Dependencies | Q3, Q4 | Covered |
| Tools | Q6, Q7 | Covered |
| Techniques | Q1, Q2 | Covered |

**Coverage: 4/4 corners covered (100%)**

## Halt-loop Checkpoints

| Checkpoint | Assertion | Action if fails |
|---|---|---|
| Before answering Qx | Every `knowledge-base/references/{project}/{path}` declared in Qx's Fase A exists | Mark Qx BLOCKED with reason "path not found", continue |
| Per-question Fase A budget | Fase A returned ≥ 1 hotspot OR 3 query-variant retries attempted | After 3 empty retries, mark Qx BLOCKED "Fase A exhausted"; continue |
| After answering Qx | Blueprint section under Qx has ≥ 1 citation | Re-iterate Qx (1 retry max) |
| Mid-loop sanity | Total `knowledge-base/references/` citations ≥ blueprint-prose-words / 200 | Add citations to under-cited paragraphs (1 retry max) |
| Per-project time budget | Project time budget not exhausted | When exhausted, mark remaining Qx for that project BLOCKED "budget exhausted"; advance |
| Before promising complete | All 4 coverage corners have populated sections | Refuse promise, continue iterating |

## Acceptance Criteria

- [ ] All 7 research questions answered OR explicitly marked BLOCKED with reason
- [ ] All four coverage corners have populated sections in the blueprint
- [ ] Every citation points to a real `knowledge-base/references/{...}` path
- [ ] At least one ADR section in the blueprint synthesizes the M5 query/index/fusion/test decisions
- [ ] Time budget respected per project
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m5-hybrid-search-qa-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed → confidence re-score)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations
- [ ] Coverage Matrix 100% covered
- [ ] ADRs reference at least one principle from project rules — here `.claude/rules/architecture.md § 1-2` (the DB query is an infrastructure adapter behind a domain port; DIP) + DRY (reuse `src/talkex/retrieval/fusion.py`) + the `docs/KB.md` benchmark-against-BM25 axiom
