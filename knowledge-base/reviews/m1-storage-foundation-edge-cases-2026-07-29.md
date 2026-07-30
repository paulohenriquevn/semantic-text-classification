# Discover Edge Case Review — M1 Storage Foundation

Date: 2026-07-29
Discovery plan analyzed: knowledge-base/discoveries/plans/m1-storage-foundation-plan.md
Research questions analyzed: 6
Edge cases found: 3 (MUST FIX: 0, SHOULD TEST: 1, DOCUMENT: 2)

## MUST FIX

_None._ All cited `knowledge-base/references/chatwoot/` paths were verified to exist (7/7 via `ls`).

## SHOULD TEST

### EC-1: Q1 scope-creep into the full `db/schema.rb`
- **Affected question:** Q1 (index design)
- **Suggested halt-loop checkpoint:** cap Fase B reading of `db/schema.rb` to the `conversations` block (`:770-814`) and the named indexes cited (`:799`, `:1171`); do NOT read the whole 1400-line schema. The composite + GIN trigram indexes are the whole answer.

## DOCUMENT

### EC-2: chatwoot is Ruby/RSpec; M1 implements in Python/pytest (test corner)
- **Accepted risk:** Q4 reads chatwoot's `spec/finders/conversation_finder_spec.rb` for the *pattern* (real-DB fixtures + assertion style), not the code. The blueprint must translate the RSpec pattern to pytest against real Timescale — the technique transfers, the syntax does not. No action; noted so the blueprint does not copy Ruby verbatim.

### EC-3: the core M1 tech (Timescale lifecycle + pgvector) has no peer citation
- **Accepted risk:** already captured by plan ADR D3. Q1–Q6 cover the *transferable Postgres* patterns; hypertable chunking/retention/compression/continuous-aggregates + pgvector index tuning come from ADR-005 + official docs, and the blueprint will flag them as a residual gap (not fabricate peer citations). This is honest deferral, not a coverage hole.

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1 | 1 | 0 | 1 | 0 |
| Q4 | 1 | 0 | 0 | 1 |
| (cross) | 1 | 0 | 0 | 1 |

**Verdict:** DISCOVERY PLAN OK (0 MUST FIX; EC-1 checkpoint absorbed into the plan's halt-loop; EC-2/EC-3 documented as accepted risks)
