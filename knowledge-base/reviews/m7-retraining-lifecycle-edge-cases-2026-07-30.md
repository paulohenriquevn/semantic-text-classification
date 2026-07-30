# Discover Edge Case Review — M7 Retraining Loop & Data Lifecycle

Date: 2026-07-30
Discovery plan analyzed: knowledge-base/discoveries/plans/m7-retraining-lifecycle-plan.md
Research questions analyzed: 7
Edge cases found: 3 (MUST FIX: 0, SHOULD TEST: 2, DOCUMENT: 1)

## MUST FIX

(none — all 7 questions map to a method and every cited path was verified to exist before the plan was written)

## SHOULD TEST

### EC-1: ai-powered's Presidio redactor is English/US-centric — PT-BR PII differs
- **Affected question:** Q1, Q3
- **Family:** Interpretation
- **Scenario:** ai-powered redacts US entities (SSN etc.) via Presidio + spaCy `en`. Brazilian PII is CPF, CNPJ, RG, DDD phone, e-mail, and PT names — Presidio's PT support is immature. The execute must extract the PATTERN (detect entity → redact-vs-preserve → keep account IDs for supervisor review), not recommend porting the English Presidio config. The blueprint's recommendation should weigh a focused PT-BR regex redactor (CPF/phone/email/name-heuristic) against Presidio-PT, honestly.
- **Suggested halt-loop checkpoint:** before marking Q1/Q3 done, assert the blueprint names the PT-BR PII entity set (CPF, phone, email, …) and makes an explicit build-vs-adopt call for the redactor — not a blind "use Presidio".

### EC-2: "export BEFORE the 30-day drop" is an ordering guarantee — the test must prove it
- **Affected question:** Q6
- **Family:** Coverage
- **Scenario:** The M7 DoD requires the anonymized sample to be exported BEFORE the raw chunk is purged. chatwoot's export job is on-demand, not tied to a retention purge. The execute must synthesize the pre-purge trigger (export the window, THEN allow drop_chunks) — neither peer states this ordering.
- **Suggested halt-loop checkpoint:** before promising complete, assert the blueprint has an ADR on the export-before-purge ordering (export the to-be-dropped window first; the drop is safe only after a successful export), with the ADR-005 hot/purge split as rationale.

## DOCUMENT

### EC-3: label volume may be too low to show retraining gains (ROADMAP risk 2)
- **Accepted risk:** M5's `labels` table may hold too few QA labels to beat the M2 pre-trained baseline. This is an honest-negative possibility the blueprint should frame (benchmark new-vs-deployed; if no gain, record it and keep the deployed model) — not a plan defect. No plan change needed; the benchmark IS the gate.

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1 | 1 | 0 | 1 | 0 |
| Q2 | 0 | 0 | 0 | 0 |
| Q3 | 1 | 0 | 1 | 0 |
| Q4 | 0 | 0 | 0 | 0 |
| Q5 | 0 | 0 | 0 | 0 |
| Q6 | 1 | 0 | 1 | 0 |
| Q7 | 0 | 0 | 0 | 0 |
| (retrain gains) | 1 | 0 | 0 | 1 |

**Verdict:** DISCOVERY PLAN OK — the 2 SHOULD-TEST items are execution-time synthesis assertions (name the PT-BR PII set + a build-vs-adopt call; record the export-before-purge ordering ADR), not plan-structure defects. No v1.1 bump required.
