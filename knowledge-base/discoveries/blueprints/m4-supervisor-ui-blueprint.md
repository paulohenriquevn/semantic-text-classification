# Blueprint: M4 Supervisor Live-Monitoring UI

> **Version 1.0** — Locks the M4 supervisor UI: a React page on the internal `demo/frontend` (React 18 +
> Vite + TS + Tailwind) with a `useSupervisorStream` hook (native `EventSource` over the shipped
> `GET /supervisor/stream`), an active-call list, an alert inbox, and a drill-down reusing `EvidenceBadge` /
> `HighlightedText` / `evidence.ts`. chatwoot supplies the realtime-connector + card UX pattern (Vue → React);
> a vitest toolchain + a Vite SSE proxy are added. Produced by `cycle-discover` execute from
> `m4-supervisor-ui-plan.md`.

**Slug:** `m4-supervisor-ui`
**Created:** 2026-07-30
**discover-confidence verdict:** recorded at the end after scoring.

## Executive summary

The backend is shipped: `GET /supervisor/stream` (SSE) emits evidence-backed alerts (M0 + M2 sentiment +
M3 critical rules). M4 builds the React consumer: a `useSupervisorStream` hook (native `EventSource`,
connect→event→state, no new dependency), a supervisor page composing an active-call list + an alert inbox +
a drill-down that reuses the demo's `EvidenceBadge`/`ConversationView`/`evidence.ts`. chatwoot's
ActionCable connector + conversation card are the UX pattern (Vue → React). A vitest + @testing-library
toolchain and a Vite proxy for the SSE path are added (both absent today).

## Context

`ROADMAP.md § M4` — the anchor supervisor surface. `demo/frontend` (React+Vite+TS+Tailwind+react-query) has
reusable pieces but no SSE client, no supervisor components, and no component-test toolchain. Constrained by
`.claude/rules/architecture.md` and `.claude/rules/testing.md`.

## Objective

Lock, with cited evidence, the SSE-client hook, the active-call/alert-inbox/drill-down composition, the
reuse strategy, the vitest toolchain, and the Vite SSE proxy — so M4 implementation builds the supervisor
page without rework.

## Coverage Corner 1 — Integration Tests

A React component test example: `knowledge-base/references/ai-powered-call-center-intelligence/frontend/src/App.test.tsx`
(CRA/jest — honest: NOT Vite-transferable). **M4 test decision:** add `vitest` + `@testing-library/react` +
`jsdom` (absent in the demo) and a component test that renders the supervisor page against a mocked
`EventSource`, asserting an incoming alert appears in the inbox with its evidence. This is the "100%
functional" evidence for the UI.

## Coverage Corner 2 — Dependencies

chatwoot pins frontend realtime deps in `knowledge-base/references/chatwoot/package.json` (an ActionCable/
websocket stack). **M4 deps decision:** use the browser-native `EventSource` for SSE (no new runtime dep) +
the demo's existing `@tanstack/react-query`; add dev deps `vitest` + `@testing-library/react` + `jsdom` for
tests.

## Coverage Corner 3 — Tools

chatwoot's frontend build/dev tooling is declared in `knowledge-base/references/chatwoot/package.json`
(scripts). **M4 tools decision:** reuse the demo's Vite (`dev`/`build`), add a `test` script (`vitest`), and
add a Vite proxy entry for `/supervisor/stream` — `demo/frontend/vite.config.ts` currently proxies only
`/api` (stripping the prefix), so the SSE endpoint needs its own proxy (edge-case EC-1).

## Coverage Corner 4 — Techniques

chatwoot's realtime connector shows the connect→subscribe→event→state pattern:
`knowledge-base/references/chatwoot/app/javascript/shared/helpers/BaseActionCableConnector.js` (base
subscribe + event dispatch) and `knowledge-base/references/chatwoot/app/javascript/dashboard/helper/actionCable.js`
(maps server events like `conversation.created`/`message.created` to store actions); the list-item UX is
`knowledge-base/references/chatwoot/app/javascript/dashboard/components-next/Conversation/ConversationCard/ConversationCard.vue`
(card with status/priority/unread indicators). **M4 technique decision:** a React `useSupervisorStream`
hook opens an `EventSource`, parses `alert` events, and updates state (an alerts list + a per-conversation
active-call map) — the direct React analog of the ActionCable connector; the active-call card mirrors
`ConversationCard.vue` (conversation id + latest rule + sentiment + a click-through to drill-down).

## Cross-cutting Comparison

| Concern | chatwoot (peer) | demo/frontend (internal) | M4 decision |
|---|---|---|---|
| Realtime transport | ActionCable (WebSocket) | none | native `EventSource` (SSE) |
| Connector pattern | BaseActionCableConnector | none | `useSupervisorStream` hook (same pattern) |
| List-item card | ConversationCard.vue | ResultCard.tsx (search) | active-call card (React) |
| Drill-down + evidence | Vue components | ConversationView + EvidenceBadge | reuse the demo's React components |
| Test toolchain | jest (Vue) | none | add vitest + @testing-library |
| Build/dev | webpack/vite | Vite | reuse Vite + add SSE proxy + test script |

## ADRs

### D1 — `useSupervisorStream` hook (EventSource, connect→event→state)
A React hook wrapping `EventSource("/api/supervisor/stream")`, parsing `alert` events into state (an alerts
array + an active-call map keyed by conversation_id). **Rationale:** the direct React analog of chatwoot's
connector (`BaseActionCableConnector.js`, `actionCable.js`); native SSE, no new dep. **Alternative rejected:**
polling (rejected — not real-time); a WebSocket lib (rejected — SSE is one-way + shipped). **Consequence:**
live updates without a runtime dependency.

### D2 — Supervisor page composition (active-call list + inbox + drill-down)
A `SupervisorPage` composing: an active-call list (one card per conversation with a live alert), an alert
inbox (chronological), and a drill-down reusing `ConversationView`/`EvidenceBadge`. **Rationale:** the M4 DoD
surfaces (blueprint Corner 4). **Alternative rejected:** one flat alert list (rejected — no per-call state).
**Consequence:** the supervisor sees active calls + can drill into evidence.

### D3 — Reuse the demo's evidence components
The drill-down reuses `EvidenceBadge` (colored per predicate family) + `HighlightedText` + `evidence.ts`
(`extractHighlightFragments`). **Rationale:** DRY — these are shipped and cover the M2/M3 evidence shape.
**Consequence:** consistent evidence rendering; less new code.

### D4 — Add a vitest component-test toolchain
Add `vitest` + `@testing-library/react` + `jsdom` + a `test` script; a component test renders the supervisor
page against a mocked `EventSource` and asserts an alert appears with evidence. **Rationale:** the demo has no
test toolchain (edge-case EC-3); the M4 DoD needs functional evidence. **Alternative rejected:** browser-only
manual check (rejected — not reproducible/CI-able). **Consequence:** the UI is CI-testable.

### D5 — Vite proxy for the SSE endpoint
Add a `/supervisor/stream` (or `/api/supervisor/stream`) proxy to `vite.config.ts` (today only `/api` is
proxied). **Rationale:** edge-case EC-1 — the SSE path is not reachable in dev otherwise. **Consequence:** the
stream works in `vite dev`.

### D6 — chatwoot is pattern-only; React is internal (honest gap)
chatwoot is Vue — the connector/card PATTERN transfers, not the code; the implementation is on the internal
`demo/frontend`. **Rationale:** Rule 3. **Consequence:** blueprint synthesizes React internally.

## Recommendations

1. `npm install` the demo/frontend; add dev deps `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `jsdom`; add a `test` script + a minimal vitest config.
2. `src/lib/useSupervisorStream.ts`: an `EventSource` hook → `{ alerts, activeCalls }` state (D1).
3. `src/components/SupervisorPage.tsx`: active-call list + alert inbox + drill-down reusing `ConversationView`/`EvidenceBadge` (D2/D3); add a "Supervisor" tab to `App.tsx`.
4. `src/types/api.ts`: add a `SupervisorAlertEvent` type.
5. `vite.config.ts`: add the SSE proxy (D5).
6. `SupervisorPage.test.tsx`: render with a mocked `EventSource`, dispatch an `alert` event, assert it appears with evidence (D4).

## Honest gaps

1. **G1 — chatwoot is Vue** — the connector/card pattern transfers; the React code is new (internal).
2. **G2 — component test uses a mocked EventSource** — a full browser/e2e SSE round-trip is deferred to the M8 pilot; the component test proves the render + event→UI wiring.

## discover-confidence verdict

**SHIPPABLE — score 99.7, hard_caps_triggered: none** (2026-07-30). Coverage 4/4, all citations resolve,
6 ADRs (D1–D6). Proceed to cycle-plan for M4.
