# Discovery Plan: M4 Supervisor Live-Monitoring UI

> **Version 1.0** — Investigate how chatwoot's frontend consumes a real-time stream and structures the
> conversation-list / card UX, so the M4 blueprint can build a React supervisor page (active-call list +
> drill-down with evidence + alert inbox) consuming the shipped `GET /supervisor/stream` (SSE), reusing the
> internal `demo/frontend` components. Honest scope: the strong peer (chatwoot) is Vue — pattern, not code;
> the React implementation is built on the internal `demo/frontend` (ADR D3), which today has no SSE client
> and no component-test toolchain.

**Slug:** `m4-supervisor-ui`
**Owner:** paulohenriquevn
**Created:** 2026-07-30
**Time budget:** 1.5h (chatwoot frontend + ai-powered — see ADR D1)

## Context

M4 is the anchor product surface (`ROADMAP.md § M4`): the supervisor's live-monitoring screen. The backend
is shipped — `GET /supervisor/stream` (SSE) and `GET /supervisor` (a minimal HTML page) from M0 (v0.2.0),
now carrying M2 sentiment + M3 critical-rule evidence per alert. The internal `demo/frontend` (React 18 +
Vite + TypeScript + Tailwind + react-query) has reusable pieces (`EvidenceBadge`, `HighlightedText`,
`ConversationView`, `evidence.ts`, `api.ts`) but NO SSE client, NO supervisor components, and NO
component-test toolchain. This discovery mines chatwoot's realtime-connector + list-card UX (the strong
peer pattern) and defers the React implementation to the internal demo (ADR D3). Constrained by
`.claude/rules/architecture.md` and `.claude/rules/testing.md`.

## Objective

Enable the M4 blueprint to decide, with evidence, **the SSE-client/stream-state pattern, the active-call
list-item UX, the drill-down reuse strategy, and the component-test toolchain** — building the React
supervisor page on the internal demo frontend.

- [ ] All research questions answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison populated
- [ ] At least one concrete decision proposal per question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/chatwoot/` | `app/javascript/shared/helpers/`, `app/javascript/dashboard/`, `package.json` | Realtime-connector + conversation-card/list UX (pattern) |
| `knowledge-base/references/ai-powered-call-center-intelligence/` | `frontend/src/` | A React component-test example (CRA/jest — conceptual only) |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/chatwoot/` — Ruby backend | Not frontend |
| chatwoot Vue component internals (SFC templates) | Vue-specific; only the connector + list pattern transfers to React |
| The internal `demo/frontend` React code | Internal (ADR D3) — the implementation base, not peer research |
| Other peers | No supervisor UI |

## ADRs

### D1 — Time budget + stop conditions
**Decision:** chatwoot 1.25h (the strong pattern peer) · ai-powered 0.25h (a thin React example).
**Stop condition — per question:** after 3 empty retries, mark BLOCKED "Fase A exhausted"; never fabricate (Rule 3).
**Stop condition — per project:** on budget exhaustion, mark remaining BLOCKED; emit `<promise>BLUEPRINT_BLOCKED</promise>` if any remain.

### D2 — Investigation depth
**Decision:** Read the realtime connector + list card in full; scan the deps/tooling.
**Consequences:** deeper read on the connector; cheap scan on package.json.

### D3 — React implementation is internal (honest deferral)
**Decision:** the supervisor page is built on the internal `demo/frontend` (reusing `EvidenceBadge`,
`ConversationView`, `evidence.ts`); chatwoot (Vue) supplies only the realtime-connector + list-UX pattern;
the demo has no SSE client and no vitest — M4 adds both internally.
**Rationale:** honesty (Rule 3) — chatwoot's Vue code is not a copyable React citation; the demo is internal.
**Consequences:** the blueprint synthesizes the React page internally; chatwoot is pattern evidence.

## Research Questions

| # | Question | Corner | Reference project(s) | Fase A (broad map) | Fase B (deep Read) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | How does chatwoot's frontend consume a real-time stream and dispatch events to state? | techniques | `knowledge-base/references/chatwoot/` | Grep `subscribe`/`received`/event map in `app/javascript/shared/helpers/BaseActionCableConnector.js` + `app/javascript/dashboard/helper/actionCable.js` | Read both; capture the connect → subscribe → event→state pattern | SSE-client hook design → informs `useSupervisorStream` |
| Q2 | How is a live conversation-list card structured (state + priority + unread)? | techniques | `knowledge-base/references/chatwoot/` | Glob `app/javascript/dashboard/components-next/Conversation/ConversationCard/ConversationCard.vue` | Read the card + its sibling status/priority indicators | Active-call list-item UX → informs the supervisor call list |
| Q3 | How is a React frontend component tested (toolchain to adopt)? | tests | `knowledge-base/references/ai-powered-call-center-intelligence/` | Read `frontend/src/App.test.tsx` (CRA/jest example — honest: not Vite) | Read the test + `setupTests.ts` | Test pattern → M4 adds vitest + @testing-library (the demo has none) |
| Q4 | What realtime/frontend dependencies does chatwoot pin? | deps | `knowledge-base/references/chatwoot/` | Grep realtime/frontend deps in `package.json` | Read the relevant deps | Dep note → M4 uses native `EventSource` (no new dep) + the demo's react-query |
| Q5 | What is chatwoot's frontend build/dev tooling? | tools | `knowledge-base/references/chatwoot/` | Grep `scripts`/build config in `package.json` | Read the scripts | Tooling note → M4 reuses the demo's Vite (dev/build), adds a `test` script |

## Coverage Matrix

| Corner | Questions mapped | Status |
|---|---|---|
| Integration tests | Q3 | Covered |
| Dependencies | Q4 | Covered |
| Tools | Q5 | Covered |
| Techniques | Q1, Q2 | Covered |

**Coverage: 4/4 corners covered (100%)**

## Halt-loop Checkpoints

| Checkpoint | Assertion | Action if fails |
|---|---|---|
| Before answering Qx | Every cited `knowledge-base/references/{path}` exists | Mark Qx BLOCKED "path not found", continue |
| Per-question Fase A budget | ≥1 hotspot OR 3 retries | Mark BLOCKED "Fase A exhausted"; continue |
| React-implementation question | If a question drifts into the internal demo React code with no peer source | STOP — that is ADR D3 territory (internal) |
| Before promising complete | All 4 corners populated | Refuse promise, continue |

## Acceptance Criteria

- [ ] All research questions answered OR explicitly BLOCKED with reason
- [ ] All four corners populated in the blueprint
- [ ] Every citation resolves to a real `knowledge-base/references/{...}` path
- [ ] ≥1 ADR synthesizes M4 decisions; the internal-React deferral (D3) is explicit
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m4-supervisor-ui-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations (esp. no fake React-code peer citation — ADR D3)
- [ ] Coverage Matrix 100%
- [ ] ADRs cite `.claude/rules/architecture.md`, `.claude/rules/testing.md`
