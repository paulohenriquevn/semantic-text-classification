# Discover Edge Case Review — M0 Walking Skeleton

Date: 2026-07-29
Discovery plan analyzed: knowledge-base/discoveries/plans/m0-walking-skeleton-realtime-monitoring-plan.md
Research questions analyzed: 7
Edge cases found: 2 (MUST FIX: 0, SHOULD TEST: 1, DOCUMENT: 1) — plus 1 investigated-and-withdrawn (EC-1)

## MUST FIX

_None._

## Withdrawn after verification (honest audit trail)

### EC-1 (WITHDRAWN — false positive): livekit In-Scope path
- Initial suspicion: the In-Scope subdir `livekit-agents/livekit/agents/voice/` looked like it omitted the repo's double `livekit-agents/` nesting.
- Verification: the In-Scope subdir column is **project-relative** (template convention), so it is appended to the project prefix `knowledge-base/references/livekit-agents/`. Combined = `knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/voice/`, which **resolves** (verified via `ls -d`). The `utils/aio/` path resolves too.
- Conclusion: the plan is correct. The earlier failing `ls` used a single-segment path that neither the plan nor the filesystem uses — a tester error, not a plan defect. No change to the plan.

## SHOULD TEST

### EC-2: Q1/Q2 scope-creep into large subsystems
- **Affected question:** Q1 (livekit voice/), Q2 (chatwoot app/)
- **Suggested halt-loop checkpoint:** Before iterating, cap Fase B to the exact files named in the RQ (`agent_session.py` + `channel.py`; `room_channel.rb` + `action_cable_listener.rb` + `conversation.rb`). Do NOT expand into sibling modules — the bounded-channel and the broadcast-listener are the whole answer.

## DOCUMENT

### EC-3: ai-powered-call-center-intelligence has no test suite
- **Accepted risk:** The tests corner is intentionally covered only by livekit (Q4) and chatwoot (Q5). The repo has no Python tests (only a CRA default smoke test). This is fine: ai-powered contributes the technique/deps/tools corners; forcing a test question onto a test-less repo would produce a BLOCKED with no value. No action needed.

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1 | 1 | 0 | 1 | 0 |
| Q2 | 1 | 0 | 1 | 0 |
| Q3–Q7 | 0 | 0 | 0 | 0 |
| (ai-powered tests) | 1 | 0 | 0 | 1 |

**Verdict:** DISCOVERY PLAN OK (0 MUST FIX after verification; EC-2 SHOULD-TEST checkpoint absorbed into plan v1.1 by the autonomous runner; EC-3 documented as accepted risk)
