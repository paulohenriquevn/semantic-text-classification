# Plan: M4 Supervisor Live-Monitoring UI

> **Version 1.0** — Build the React supervisor page on `demo/frontend`: a `useSupervisorStream` hook
> (native `EventSource` over the shipped `GET /supervisor/stream`), a `SupervisorPage` (active-call list +
> alert inbox + drill-down reusing `EvidenceBadge`/`ConversationView`), a Vite SSE proxy, and a vitest
> component test proving an incoming alert renders with its evidence.

## Goal

> "Enable a supervisor to see live evidence-backed alerts so that an incoming alert appears in the inbox
> with its evidence, measured by `SupervisorPage.test.tsx` passing (render the page against a mocked
> EventSource, dispatch an alert event, assert it appears with rule name + evidence)."

## Context

M4 (`ROADMAP.md § M4`) builds the supervisor surface on the internal `demo/frontend` (React+Vite+TS+Tailwind),
consuming the shipped `GET /supervisor/stream` (M0, carrying M2 sentiment + M3 critical-rule evidence), per
the SHIPPABLE 99.7 blueprint `knowledge-base/discoveries/blueprints/m4-supervisor-ui-blueprint.md`. The demo
has no SSE client and no test toolchain — M4 adds both. Constrained by `.claude/rules/architecture.md`,
`.claude/rules/testing.md`.

## Baseline Context (deep review of current state)

### Files that will be touched

| File | LoC today | Last commit | Why it exists today | Invariants to preserve |
|---|---|---|---|---|
| `demo/frontend/src/App.tsx` | ~90 | (demo) | tab shell (search/categories/analytics/guide) | additive: add a "supervisor" tab, keep the rest |
| `demo/frontend/src/components/EvidenceBadge.tsx` | ~40 | (demo) | evidence badge per predicate family | reused read-only |
| `demo/frontend/src/lib/evidence.ts` | (demo) | (demo) | `extractHighlightFragments` | reused read-only |
| `demo/frontend/vite.config.ts` | ~20 | (demo) | Vite config; proxies `/api` | add an SSE proxy entry |
| `demo/frontend/package.json` | (demo) | (demo) | React+Vite deps, no `test` | add dev deps + `test` script |
| `demo/frontend/src/lib/useSupervisorStream.ts` (NEW) | 0 | — | EventSource hook → {alerts, activeCalls} | — |
| `demo/frontend/src/components/SupervisorPage.tsx` (NEW) | 0 | — | active-call list + alert inbox + drill-down | — |
| `demo/frontend/src/components/SupervisorPage.test.tsx` (NEW) | 0 | — | component test (mocked EventSource) | — |
| `demo/frontend/vitest.config.ts` (NEW) | 0 | — | vitest + jsdom config | — |
| `demo/frontend/src/types/api.ts` | (demo) | (demo) | API types | add `SupervisorAlertEvent` |

### Current callers / dependents

- **Symbol:** `App.tsx` tab shell — self-contained; M4 adds a tab (additive, no caller impact).
- New `useSupervisorStream`/`SupervisorPage`: first-of-its-kind.

### Domain glossary

- **SSE (Server-Sent Events)** — one-way server→client stream (`EventSource`); the supervisor's live channel.
- **Active call** — a conversation with a recent live alert (keyed by conversation_id).
- **Alert inbox** — the chronological list of incoming alert events.

### Architecture boundaries affected

Frontend only: a new hook (state) + a new page (view) in `demo/frontend`, reusing existing components; no
backend change (the SSE endpoint is shipped).

## Prior Art & Related Work

- **Internal blueprint** — `m4-supervisor-ui-blueprint.md` §"Coverage Corner 4" + its ADR set.
- **Reference — chatwoot** — realtime connector `knowledge-base/references/chatwoot/app/javascript/shared/helpers/BaseActionCableConnector.js`; card UX `knowledge-base/references/chatwoot/app/javascript/dashboard/components-next/Conversation/ConversationCard/ConversationCard.vue`.
- **Reference — ai-powered** — React test example `knowledge-base/references/ai-powered-call-center-intelligence/frontend/src/App.test.tsx`.

## Objective

- [ ] `useSupervisorStream` hook (EventSource → alerts + activeCalls)
- [ ] `SupervisorPage` (active-call list + alert inbox + drill-down reusing evidence components)
- [ ] vitest toolchain + a component test proving alert render
- [ ] Vite SSE proxy; a "Supervisor" tab in App.tsx

## ADRs

### D1 — useSupervisorStream hook (EventSource)
- **Decision:** a React hook wrapping `EventSource`, parsing `alert` events into `{alerts, activeCalls}`.
- **Rationale:** the React analog of chatwoot's connector; native SSE, no runtime dep.
- **Alternatives considered:** polling (rejected — not real-time); WebSocket lib (rejected — SSE is shipped/one-way).
- **Consequence:** live updates without a new dependency.

### D2 — Reuse the demo's evidence components
- **Decision:** the drill-down reuses `EvidenceBadge` + `evidence.ts`.
- **Rationale:** DRY; covers the M2/M3 evidence shape.
- **Alternatives considered:** new evidence UI (rejected — duplicates shipped components).
- **Consequence:** consistent rendering, less code.

### D3 — vitest component test against a mocked EventSource
- **Decision:** add vitest + @testing-library/react + jsdom; the test mocks `EventSource`, dispatches an alert, asserts it renders.
- **Rationale:** the demo has no test toolchain; the DoD needs functional UI evidence.
- **Alternatives considered:** browser-only manual check (rejected — not CI-able).
- **Consequence:** the UI is CI-testable.

## Drawbacks & Risks

| Drawback / Risk | Severity | Mitigation | Owner |
|---|---|---|---|
| R1 — adding a test toolchain to a demo without one | Medium | minimal vitest config + jsdom; one focused component test | dev |
| R2 — SSE not proxied in dev (only `/api`) | Medium | add a Vite proxy entry for the SSE path (per the blueprint SSE-proxy decision) | dev |

## Unresolved Questions

- Q1 — Should the active-call list persist closed calls or drop them? M4 keeps the last-alert-per-conversation; eviction policy deferred.
- Q2 — Exact SSE reconnect/backoff on drop? M4 relies on `EventSource`'s built-in auto-reconnect; custom backoff deferred to M8.

## Dependency Graph

```
Phase 0 (toolchain: deps + vitest config + Vite SSE proxy)  ──▶  Phase 1 (useSupervisorStream hook + test)
                                                                        ▼
                                                     Phase 2 (SupervisorPage + tab + component test)
                                                                        ▼
                                                        Final: Integration Validation (build + test)
```

---

## Phase 0: Toolchain

**Objective:** vitest + SSE proxy ready.

### T0.1 — add test toolchain + SSE proxy

#### Objective
Add vitest/@testing-library/jsdom + a `test` script + a Vite proxy for the SSE path.

#### Why this step
1. **What:** `demo/frontend/package.json` dev deps + `test` script; `vitest.config.ts`; `vite.config.ts` proxy.
2. **Why now:** every M4 test needs the toolchain; the SSE needs a dev proxy (blueprint toolchain + SSE-proxy decisions).

#### Files to edit
```
demo/frontend/package.json — dev deps + "test": "vitest run"
demo/frontend/vitest.config.ts (NEW) — jsdom env + setup
demo/frontend/vite.config.ts — add /supervisor/stream (or /api/supervisor/stream) proxy
```

#### TDD
```
RED:   a trivial vitest sanity test (expect(1+1).toBe(2)) runs green under `npm test`
GREEN: add the config + deps
VERIFY: cd demo/frontend && npm test
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `cd demo/frontend && npm test` runs vitest and reports a green run
- [ ] `demo/frontend/vite.config.ts` proxies the SSE path (grep shows the entry)

#### DoD
- [ ] vitest runs; SSE proxy present

---

## Phase 1: useSupervisorStream hook

**Objective:** the SSE state hook.

### T1.1 — useSupervisorStream

#### Objective
A hook that opens an `EventSource`, parses `alert` events, exposes `{alerts, activeCalls}`.

#### Why this step
1. **What:** `demo/frontend/src/lib/useSupervisorStream.ts` + `src/types/api.ts` `SupervisorAlertEvent`.
2. **Why now:** the page consumes it (blueprint D1).

#### Files to edit
```
demo/frontend/src/lib/useSupervisorStream.ts (NEW)
demo/frontend/src/types/api.ts — add SupervisorAlertEvent
demo/frontend/src/lib/useSupervisorStream.test.ts (NEW)
```

#### TDD
```
RED:   test_hook_collects_alerts — a mocked EventSource dispatching two alert events yields alerts.length===2
GREEN: implement the hook
VERIFY: npm test -- useSupervisorStream
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `test_hook_collects_alerts` asserts two dispatched alert events produce `alerts.length === 2`
- [ ] `test_hook_tracks_active_call` asserts `activeCalls` is keyed by conversation_id (latest alert wins)

#### DoD
- [ ] hook tests green

---

## Phase 2: SupervisorPage

**Objective:** the page + tab.

### T2.1 — SupervisorPage + tab + component test

#### Objective
Compose the active-call list + alert inbox + drill-down; add the tab; test the render.

#### Why this step
1. **What:** `SupervisorPage.tsx` (uses the hook + `EvidenceBadge`); `App.tsx` tab; `SupervisorPage.test.tsx`.
2. **Why now:** the M4 DoD surface (blueprint D2/D3).

#### Files to edit
```
demo/frontend/src/components/SupervisorPage.tsx (NEW)
demo/frontend/src/App.tsx — add a "supervisor" tab
demo/frontend/src/components/SupervisorPage.test.tsx (NEW)
```

#### TDD
```
RED:   test_alert_renders_with_evidence — render SupervisorPage with a mocked EventSource; dispatch an alert with a rule + evidence; assert the alert (rule name) and an EvidenceBadge appear in the inbox
GREEN: implement SupervisorPage
VERIFY: npm test -- SupervisorPage
```

#### Concurrency tests
```
(none — single-threaded)
```

#### Acceptance Criteria
- [ ] `test_alert_renders_with_evidence` asserts a dispatched alert's rule name is visible and an evidence badge renders
- [ ] `App.tsx` exposes a "supervisor" tab that mounts `SupervisorPage`

#### DoD
- [ ] SupervisorPage test green; the page mounts under the new tab

---

## Coverage Matrix

| # | Gap / Requirement | Task(s) | Resolution |
|---|---|---|---|
| 1 | Test toolchain (vitest) | T0.1 | vitest + jsdom + test script |
| 2 | SSE reachable in dev | T0.1 | Vite proxy |
| 3 | Live SSE state | T1.1 | useSupervisorStream hook |
| 4 | Active-call list + inbox + drill-down | T2.1 | SupervisorPage reusing EvidenceBadge |
| 5 | Alert renders with evidence (DoD) | T2.1 | SupervisorPage.test.tsx |

**Coverage: 5/5 gaps covered (100%)**

## Global Definition of Done

- [ ] `cd demo/frontend && npm test` green (hook + page component tests)
- [ ] `cd demo/frontend && npm run build` succeeds (tsc + vite build)
- [ ] File-size ≤ 500 LoC per file
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] No change to the shipped backend (SSE endpoint reused)
- [ ] Runtime-metric proof — the component test observes a real render of a dispatched alert, not just compiles

## Failure scenarios (external I/O)

| Dependency | Failure mode | How the test reproduces it | Expected behavior |
|---|---|---|---|
| SSE stream (EventSource) | connection error | the mocked EventSource fires `onerror` | the hook stays alive; `EventSource` auto-reconnect handles retry; no crash |

## Final Phase: Integration Validation (MANDATORY)

**Objective:** prove the UI renders alerts + builds.

### Execution
```
cd demo/frontend
npm test
npm run build
```

### Acceptance Criteria
- [ ] `SupervisorPage.test.tsx` + hook test green (the Goal metric)
- [ ] `npm run build` succeeds (tsc typecheck + vite build)
- [ ] no console errors in the test render
