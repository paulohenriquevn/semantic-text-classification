# Discover Edge Case Review — M8 Pilot Hardening & V1 Ship

Date: 2026-07-30
Discovery plan analyzed: knowledge-base/discoveries/plans/m8-pilot-hardening-plan.md
Research questions analyzed: 7
Edge cases found: 3 (MUST FIX: 0, SHOULD TEST: 2, DOCUMENT: 1)

## MUST FIX

(none — all 7 questions map to a method and every cited path was verified to exist before the plan was written)

## SHOULD TEST

### EC-1: the V1-acceptance harness must RE-RUN checks, not hard-code the numbers
- **Affected question:** Q1, Q5
- **Family:** Interpretation
- **Scenario:** M8's acceptance report aggregates the V1 criteria (alert p95, retrieval p95, sentiment F1, precision, purge). The blueprint must recommend that the harness RE-RUN the underlying checks (or read fresh metrics JSONs produced by the existing benchmarks) — NOT embed the historical numbers as constants. A hard-coded PASS is acceptance theatre (the same failure the plan-confidence golden rule guards against).
- **Suggested halt-loop checkpoint:** before marking Q1/Q5 done, assert the blueprint's acceptance-report recommendation sources each criterion from a live check or a freshly-produced metrics artifact, with an explicit "no hard-coded PASS" note.

### EC-2: engagement is a north-star PROXY on synthetic data — label it honestly
- **Affected question:** Q6
- **Family:** Coverage
- **Scenario:** The alert-engagement metric (acted-on rate) is the north-star proxy, but in this environment there is no real supervisor acting on alerts. The metric plumbing (record an ack/label → compute a rate) is real and testable; the *rate itself* on synthetic data is not a production signal. The blueprint must frame the metric as instrumentation-ready, with the real rate pending a live pilot.
- **Suggested halt-loop checkpoint:** before promising complete, assert the blueprint records that the engagement metric is instrumented-and-tested but its production value awaits a real pilot (Rule 3 honesty).

## DOCUMENT

### EC-3: real-world 8 kHz drift is unvalidated (ROADMAP risk 1)
- **Accepted risk:** the V1 criteria are proven on synthetic load; live 8 kHz call-center speech may degrade them. This is a documented pilot risk, not a plan defect — the acceptance report must carry an explicit "validated on synthetic load; real-drift pending" caveat. No plan change needed.

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1 | 1 | 0 | 1 | 0 |
| Q2 | 0 | 0 | 0 | 0 |
| Q3 | 0 | 0 | 0 | 0 |
| Q4 | 0 | 0 | 0 | 0 |
| Q5 | 1 | 0 | 1 | 0 |
| Q6 | 1 | 0 | 1 | 0 |
| Q7 | 0 | 0 | 0 | 0 |
| (drift) | 1 | 0 | 0 | 1 |

**Verdict:** DISCOVERY PLAN OK — the 2 SHOULD-TEST items are honesty guards on the acceptance harness (re-run, don't hard-code; label the engagement proxy), not plan-structure defects. No v1.1 bump required.
