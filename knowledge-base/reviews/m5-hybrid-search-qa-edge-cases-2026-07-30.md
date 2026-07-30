# Discover Edge Case Review — M5 Hybrid Search & QA

Date: 2026-07-30
Discovery plan analyzed: knowledge-base/discoveries/plans/m5-hybrid-search-qa-plan.md
Research questions analyzed: 7
Edge cases found: 3 (MUST FIX: 0, SHOULD TEST: 2, DOCUMENT: 1)

## MUST FIX

(none — all 7 questions map to a method and every cited path was verified to exist before the plan was written)

## SHOULD TEST

### EC-1: chatwoot has TWO search engines — Q3 must attribute per-entity
- **Affected question:** Q3 (and indirectly Q1)
- **Family:** Interpretation
- **Scenario:** `Gemfile` carries both `searchkick` (line 72, OpenSearch) and `pg_search` (line 166). During `/discover-execute`, Q3 could report "chatwoot uses OpenSearch" or "chatwoot uses Postgres FTS" without stating **which entity each engine searches** (conversations vs contacts vs articles). The M5 decision hinges on the *conversation-search* path specifically.
- **Suggested halt-loop checkpoint:** before marking Q3 done, assert `search_service.rb` was read and the engine used for the **conversation/message** search path is named with a `path:line` (not just "both gems exist in the Gemfile").

### EC-2: Q4 proves a negative — bound it to the manifest + imports, not to a claim
- **Affected question:** Q4
- **Family:** Interpretation
- **Scenario:** Q4 asks whether ai-powered uses a vector store. Absence of `faiss`/`pgvector` in `requirements.txt` does not by itself prove "retrieval is purely LLM-mediated" — a hosted service (e.g., Azure Cognitive Search) could be called over HTTP with no Python dep.
- **Suggested halt-loop checkpoint:** before marking Q4 done, assert the verdict is phrased as "no local vector/ANN dependency found in requirements.txt + no embedding/vector call found in backend/ (greps: faiss|pgvector|chroma|qdrant|azure.*search|embedding)" with the greps cited — not as an unqualified "purely LLM" claim.

## DOCUMENT

### EC-3: soft ordering — Q5 (specs) reads best after Q1 (service)
- **Accepted risk:** Q5 audits `search_service_spec.rb`; understanding it is easier once Q1 has mapped `search_service.rb`. This is a readability preference, not a hard dependency — the specs are self-contained enough to read standalone. `/discover-execute` may answer them in plan order; no plan change needed.

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1 | 0 | 0 | 0 | 0 |
| Q2 | 0 | 0 | 0 | 0 |
| Q3 | 1 | 0 | 1 | 0 |
| Q4 | 1 | 0 | 1 | 0 |
| Q5 | 1 | 0 | 0 | 1 |
| Q6 | 0 | 0 | 0 | 0 |
| Q7 | 0 | 0 | 0 | 0 |

**Verdict:** DISCOVERY PLAN OK — the 2 SHOULD-TEST items are execution-time assertions (attribute the chatwoot engine per-entity; phrase Q4 as a bounded grep result), not plan-structure defects. No v1.1 bump required; the two checkpoints are carried into execution.
