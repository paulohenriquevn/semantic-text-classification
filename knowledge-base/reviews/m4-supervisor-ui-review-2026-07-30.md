# Review — M4 Supervisor Live-Monitoring UI

Date: 2026-07-30
Plan: `knowledge-base/plans/m4-supervisor-ui-plan.md` (v1.0)
Slice commit: `334dc96` (feat(analytics): M4 supervisor live-monitoring UI)
Reviewer: cycle self-review (frontend slice, ~256 LoC)

## Scope reviewed

| File | LoC | Verdict |
|---|---|---|
| `demo/frontend/src/lib/useSupervisorStream.ts` | 54 | OK |
| `demo/frontend/src/lib/useSupervisorStream.test.ts` | 60 | OK |
| `demo/frontend/src/components/SupervisorPage.tsx` | 91 | OK |
| `demo/frontend/src/components/SupervisorPage.test.tsx` | 51 | OK |
| `demo/frontend/src/App.tsx` | +tab | OK (additive) |
| `demo/frontend/vitest.config.ts`, `src/test/setup.ts`, `package.json` | toolchain | OK |

All files ≤ 500 LoC (largest 91). No backend change (SSE endpoint reused).

## Plan coverage (Coverage Matrix 5/5)

| # | Requirement | Task | Status |
|---|---|---|---|
| 1 | Test toolchain (vitest) | T0.1 | ✅ vitest+jsdom+`test` script; 4 tests green |
| 2 | SSE reachable in dev | T0.1 | ✅ via existing `/api` Vite proxy (`/api/supervisor/stream` → backend `/supervisor/stream`) |
| 3 | Live SSE state | T1.1 | ✅ `useSupervisorStream` (inbox + active-call map, latest-wins) |
| 4 | Active-call list + inbox + drill-down | T2.1 | ✅ `SupervisorPage` |
| 5 | Alert renders with evidence (DoD/Goal metric) | T2.1 | ✅ `SupervisorPage.test.tsx` renders a dispatched alert with rule name + evidence chips |

## Global DoD

- [x] `npm test` green — 4 tests (2 files) pass
- [x] `npm run build` succeeds — tsc typecheck + vite build (1641 modules)
- [x] File-size ≤ 500 LoC per file
- [x] CHANGELOG.md updated under `[Unreleased]`
- [x] No backend change
- [x] Runtime-metric proof — the component test renders a real dispatched alert (RED→GREEN, not compile-only)

## Findings

### F1 — Accepted divergence from ADR D2 (reuse EvidenceBadge) — INFO, accepted
- **Plan ADR D2** decided the drill-down would reuse `EvidenceBadge`.
- **As built:** `SupervisorPage` renders a local `EvidenceChip` instead.
- **Reason:** the shipped `EvidenceBadge` expects `PredicateEvidence` (`field_name`/`operator`/`score`), while the SSE alert's `SupervisorEvidence` shape is `predicate_type`/`matched_text`/`score`/`threshold` (M2 sentiment / M3 rule evidence). The shapes are genuinely incompatible; reusing `EvidenceBadge` would require an adapter layer for no visual gain. A compact chip is simpler (KISS) and is documented in-code (`SupervisorPage.tsx:15-16`).
- **Impact:** none on the DoD (the Goal metric — "alert renders with rule name + evidence" — is met by the chip). Recorded as an honest divergence, not silently.

### F2 — `key={i}` on the evidence map — INFO, acceptable
- Evidence lists are static per alert (never reordered/spliced), so index keys are safe here.

No correctness, security, or resource findings.

## Verdict

**READY_TO_MERGE** — all 5 Coverage-Matrix requirements and the full Global DoD are met with green-test evidence; the single divergence (F1) is a documented, DoD-neutral KISS choice.
