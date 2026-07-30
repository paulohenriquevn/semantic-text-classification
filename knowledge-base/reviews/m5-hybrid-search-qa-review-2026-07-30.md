# Review — M5 Hybrid Search & QA over 30 days

Date: 2026-07-30
Plan: `knowledge-base/plans/completed/m5-hybrid-search-qa-plan.md` (SHIPPABLE_WITH_CAVEATS 70)
Slice commits: `334dc96`..`HEAD` (Phase 0 `feat(pipeline)`, Phase 1 `feat(retrieval)`, Phase 2 `feat(retrieval)`, Phase 3 `feat(api)`, Phase 4 `test(retrieval)`)
Reviewer: cycle self-review

## Scope reviewed (all ≤ 500 LoC)

| File | LoC | Verdict |
|---|---|---|
| `domain/search.py` (SearchQuery/SearchHit/Criterion/Label) | 61 | OK |
| `infrastructure/embedder.py` (Deterministic + ST adapters) | 75 | OK |
| `infrastructure/read_repo.py` (hybrid candidates + criterion compile) | 131 | OK |
| `infrastructure/label_repo.py` | 36 | OK |
| `application/search_service.py` (RRF fusion) | 45 | OK |
| `interface/app.py` (+/search +/label routes, +read pool) | 207 | OK |
| `experiments/scripts/backfill_embeddings.py` | 75 | OK |
| `experiments/scripts/bench_hybrid_search.py` | 203 | OK |
| migration `0003_m5_search.sql` (labels table) | 20 | OK |

## Plan coverage (Coverage Matrix 6/6)

| # | Requirement (DoD) | Task | Status + evidence |
|---|---|---|---|
| 1 | Embedding column empty → ANN inert | T0.1 | ✅ backfill + ingest write; `test_embedding_population` (3 tests) |
| 2 | Hybrid retrieval over 30 days | T1.1 | ✅ ts_rank⊕pgvector fused by RRF; `test_hybrid_search` + `test_search_service` (6) |
| 3 | Search by criterion with evidence | T2.1+T3.1 | ✅ bound-SQL criterion + `/search`; `test_criterion_compile`, `test_labels`, `test_search_api` |
| 4 | Label persists for retraining | T2.1+T3.1 | ✅ labels table + `LabelRepository` + `/label`; `test_labels`, `test_search_api` |
| 5 | p95 < 200 ms under concurrent ingest | T4.1 | ✅ **real run p95=160.95ms**; `test_bench_hybrid_search` asserts <200ms |
| 6 | Hybrid ≥ BM25 baseline (KB axiom) | T4.1 | ✅ **real run hybrid MRR=1.0 vs BM25 0.833** (ANN catches paraphrase queries) |

## Global DoD

- [x] `ruff format --check` + `ruff check` + `mypy src/ tests/` clean
- [x] `pytest tests/unit` (1906 pass, 1 skip) + `pytest tests/integration` (146 pass) green against real Timescale
- [x] p95 < 200 ms under concurrent ingest — `experiments/results/m5_hybrid_bench.json` (160.95ms)
- [x] hybrid ≥ BM25 baseline — 1.0 vs 0.833 (real multilingual-MiniLM); honest-negative path documented in bench for the deterministic case
- [x] File-size ≤ 500 LoC per file (largest 207)
- [x] Embeddings populated (no BM25-in-disguise) — the critical baseline gap is closed

## Findings

### F1 — DIP split of fusion vs candidate SQL — INFO (design strength)
`SearchService` (application) owns RRF; the `TurnSearchPort` adapter owns candidate SQL. This makes the fusion unit-testable with a fake port (`test_search_service`) and keeps the SQL in infrastructure — a faithful reading of the plan's D1 (the plan named a single `hybrid_search`; the split is a testability-driven refinement, same behavior, documented here).

### F2 — Injection defense is structural, not incidental — INFO
`_criteria_clause` sources columns from a fixed whitelist and always binds values; `test_injection_value_is_bound_never_interpolated` proves a malicious value never reaches the SQL string. Unknown fields fail fast (typed error) — no silent pass-through (`error-handling.md`).

### F3 — p95 test uses the deterministic embedder — INFO, honest caveat (accepted)
The automated `test_search_p95_under_concurrent_ingest` runs with `DeterministicEmbedder` (hermetic, no download) and asserts p95 < 200 ms on the SQL+fusion path. The **real** production number (which includes the ~real ST query-embedding cost) is measured by the manual benchmark run and is **160.95ms < 200ms** — so the DoD holds even with the real embedding cost included. This split (hermetic CI test + real evidence artifact) is deliberate and recorded.

### F4 — Criterion whitelist is intentionally small for M5 — INFO (YAGNI)
`speaker` + `conversation_id` + `metadata.<key>`. Sentiment/intent are computed at alert-time (M2/M3), not persisted per-turn, so they are not yet filterable columns; a full criterion catalogue is deferred (plan Unresolved Q, YAGNI) rather than faked.

No correctness, security, or resource findings.

## Verdict

**READY_TO_MERGE** — all 6 Coverage-Matrix requirements and the full Global DoD are met with real-DB evidence (p95=160.95ms, hybrid MRR 1.0 > BM25 0.833). The critical baseline gap (empty embedding column) is closed. Findings are design notes and one accepted, documented latency-measurement caveat — none blocking.
