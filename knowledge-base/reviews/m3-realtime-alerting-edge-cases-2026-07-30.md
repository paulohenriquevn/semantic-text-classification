# Discover Edge Case Review — M3 Real-Time Alerting

Date: 2026-07-30
Discovery plan analyzed: knowledge-base/discoveries/plans/m3-realtime-alerting-plan.md
Research questions analyzed: 5
Edge cases found: 2 (MUST FIX: 0, SHOULD TEST: 0, DOCUMENT: 2)

## MUST FIX

_None._ All cited `knowledge-base/references/chatwoot/` paths were verified in M0/M1 (reused).

## DOCUMENT

### EC-1: alerting technique heavily overlaps M0 (already shipped)
- **Accepted risk:** Q1/Q2 re-mine chatwoot's event→broadcast pattern that M0 already borrowed and shipped (v0.2.0). The blueprint must FOCUS on the NEW M3 work (critical-rule catalogue, precision eval, latency), citing chatwoot only for the alerting-structure baseline — not re-deriving M0. No action; noted so the blueprint does not re-plan M0.

### EC-2: alert-precision uses internal `topic` labels as ground truth (no peer)
- **Accepted risk:** already captured by plan ADR D3. The precision ≥ 0.8 eval maps `topic` labels (e.g. `cancelamento`/`reclamacao` → should-alert for the cancellation rule) to a confusion-based precision on the internal corpus. No peer does this; it is internal methodology, honestly deferred — not a coverage hole.

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1/Q2 | 1 | 0 | 0 | 1 |
| (precision) | 1 | 0 | 0 | 1 |

**Verdict:** DISCOVERY PLAN OK (0 MUST FIX; EC-1/EC-2 documented as accepted risks — the blueprint focuses on the new catalogue + precision/latency, reusing M0's shipped alerting)
