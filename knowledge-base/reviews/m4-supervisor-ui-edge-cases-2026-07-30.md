# Discover Edge Case Review — M4 Supervisor UI

Date: 2026-07-30
Discovery plan analyzed: knowledge-base/discoveries/plans/m4-supervisor-ui-plan.md
Research questions analyzed: 5
Edge cases found: 3 (MUST FIX: 0, SHOULD TEST: 1, DOCUMENT: 2)

## MUST FIX

_None._ All cited `knowledge-base/references/` paths verified (chatwoot frontend + ai-powered App.test.tsx).

## SHOULD TEST

### EC-1: SSE proxy path not covered by the demo's Vite config
- **Affected question:** Q1
- **Suggested halt-loop checkpoint:** the blueprint MUST note that `demo/frontend/vite.config.ts` proxies only `/api` (stripping the prefix); `GET /supervisor/stream` needs its own proxy entry OR the frontend calls it under `/api/supervisor/stream`. Do not assume the SSE endpoint is reachable in dev without a proxy change.

## DOCUMENT

### EC-2: chatwoot is Vue; the pattern transfers, not the code
- **Accepted risk:** Q1/Q2 read Vue (ActionCable connector, `.vue` card). The blueprint copies the connect→subscribe→event→state PATTERN into a React `useSupervisorStream` hook + a React card — NOT the Vue code. Already captured by ADR D3. No action.

### EC-3: no component-test toolchain in the demo (vitest absent)
- **Accepted risk:** Q3 cites a CRA/jest example (ai-powered) that is not Vite-transferable. The demo has no vitest. M4 adds vitest + @testing-library/react + jsdom internally — a toolchain-setup task, honestly flagged (blueprint must budget it).

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1 | 2 | 0 | 1 | 1 |
| Q3 | 1 | 0 | 0 | 1 |

**Verdict:** DISCOVERY PLAN OK (0 MUST FIX; EC-1 checkpoint absorbed; EC-2/EC-3 documented — the blueprint must add the Vite proxy + the vitest toolchain as explicit tasks)
