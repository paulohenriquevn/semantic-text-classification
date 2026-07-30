# Discovery Plan: M7 — Retraining Loop & Data Lifecycle

> **Version 1.0** — Investigate how the cloned peers (a) detect + redact PII before exporting data
> (ai-powered's Presidio-based redactor) and export/store a dataset (chatwoot's export job), and (b)
> train + benchmark + version a model (ai-powered's churn pipeline), so we can lock the M7 architecture:
> an anonymized retraining-sample export to Parquet/object-storage BEFORE the 30-day chunk drop, a
> retraining pipeline that consumes QA labels (M5) + exported samples to produce a benchmarked new model
> version, and model-version tracking on every prediction. LGPD-critical — the blueprint must let us decide
> the PII-redaction strategy for PT-BR, the export format + storage boundary, and the retrain/benchmark/version
> flow. Reference projects in scope: `ai-powered-call-center-intelligence` (PII redaction + a train/benchmark
> pipeline) and `chatwoot` (a production data-export job with real-DB specs).

**Slug:** `m7-retraining-lifecycle`
**Owner:** tomas-herrera (Data & Eval Engineer)
**Created:** 2026-07-30
**Time budget:** 5h (per-project breakdown in ADR D1)

## Context

M7 (`ROADMAP.md § M7`) must close the offline data lifecycle: export an anonymized/pseudonymized
retraining sample to Parquet/object storage BEFORE the 30-day chunk drop, retrain the sentiment/intent
models on accumulated QA labels + exported samples producing a benchmarked new model version, and track
model versions (every prediction records its version). It depends on M1 (retention/purge), M2
(`SentimentDetector` — `src/talkex/classification/sentiment.py`, train/evaluate + joblib), and M5 (the
`labels` table). Risk 1 is LGPD-critical: a leak of raw PII into the cold sample is unacceptable. The open
gap M7 closes: how mature peers redact PII (and what they deliberately preserve), how they export + store a
dataset, and how they structure a train→benchmark→version pipeline. This is a data-governance decision
(tomas-herrera's domain) — respecting `.claude/rules/architecture.md § 2` (storage is an infrastructure
adapter behind a domain port; DIP) and `.claude/rules/testing.md § 4.1` (the anonymization negative-case:
assert PII is ABSENT, a typed guarantee, not merely "it ran").

## Objective

Decide the M7 architecture (PT-BR PII redaction strategy, export format + storage boundary, retrain/benchmark/version
flow) from evidence in the peers. Success criteria for the blueprint:

- [ ] All 7 research questions answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison table populated for `ai-powered-call-center-intelligence` and `chatwoot`
- [ ] Recommendations section provides at least one concrete decision proposal per research question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope (per reference project)

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/ai-powered-call-center-intelligence/` | `backend/pii_redaction.py`, `churn_model/train.py`, `requirements.txt` | A PII redactor + a train/benchmark/version pipeline — the two technical cores of M7. |
| `knowledge-base/references/chatwoot/` | `app/jobs/account/contacts_export_job.rb`, `spec/jobs/account/contacts_export_job_spec.rb` | A production data-export job + its real-DB spec — the export/store + how-to-test analog. |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/ai-powered-call-center-intelligence/frontend/`, `analytics/*.ipynb` | UI + notebooks, not lifecycle source. |
| `knowledge-base/references/chatwoot/app/javascript/` | Front-end, irrelevant to the export backend. |
| `knowledge-base/references/livekit-agents/`, `portuguese-bert/`, `portuguese-nlp/` | No PII/export/retrain surface (ADR D3). |
| `knowledge-base/references/*/` build artifacts, `node_modules/`, `vendor/` | Not source of truth. |

## ADRs

### D1 — Time budget + stop conditions

**Decision:** ai-powered: 3h; chatwoot: 2h. Total 5h.

**Rationale:** ai-powered holds both technical cores (PII redaction + train/benchmark), so it gets the most
time; chatwoot contributes the export/store + test pattern in 2h.

**Alternatives considered:** equal split (rejected — ai-powered is denser for M7); ai-powered-only (rejected —
loses the export-job test pattern).

**Stop condition — per question (mandatory):** After 3 empty Fase-A query-variant retries, mark the question
BLOCKED ("Fase A exhausted") and continue. Do NOT pad with unrelated hotspots.

**Stop condition — per project (mandatory):** On budget exhaustion, mark remaining questions BLOCKED
("budget exhausted"). If every remaining question is `done` or honestly `blocked`, emit
`<promise>BLUEPRINT_BLOCKED</promise>` (never `BLUEPRINT_COMPLETE` from a blocked state).

**Anti-pattern:** NEVER fabricate Fase B answers to close a Fase-A-exhausted question (Unbreakable Rule 3).

**Consequences:** the halt-loop stops per-project on budget exhaustion; blocked questions surface in the
blueprint's `## Blocked questions` section as next-discovery seed.

### D2 — Investigation depth

**Decision:** Read the redactor + train + export-job + spec files end-to-end (short, ≤ 170 LoC each); for
dependency questions, Grep the manifest/imports then Read the matched lines. ast-grep Fase A only where a
function map helps.

**Rationale:** the PII entity-selection intent + the train/benchmark flow live in these files whole; a symbol
grep misses "what is preserved vs redacted".

**Consequences:** deeper per-file cost, small files, within budget.

### D3 — Defer streaming + embedding-corpus peers

**Decision:** Exclude `livekit-agents`, `portuguese-bert`, `portuguese-nlp`.

**Rationale:** none has a PII/export/retrain surface; the four corners are covered by the two in-scope peers.

**Consequences:** no corner is left to a peer that cannot answer it.

## Research Questions

| # | Question | Corner | Reference project(s) | Fase A (broad — map) | Fase B (deep — Read at each hotspot) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | How does ai-powered DETECT + REDACT PII, and what does it deliberately PRESERVE (vs redact)? | techniques | `knowledge-base/references/ai-powered-call-center-intelligence/backend/pii_redaction.py` | `ast-grep run -p 'def $NAME($$$): $$$' --lang python knowledge-base/references/ai-powered-call-center-intelligence/backend/pii_redaction.py` to map the detect/redact functions | Read the redactor fully; capture the entity list, redact-vs-preserve policy, and the returned span metadata | Prose + a table: entity type → redact/preserve → mechanism → `path:line`; informs our PT-BR PII strategy |
| Q2 | How does ai-powered's churn model TRAIN, BENCHMARK, and persist/version the artifact? | techniques | `knowledge-base/references/ai-powered-call-center-intelligence/churn_model/train.py` | `ast-grep run -p 'def $NAME($$$): $$$' --lang python knowledge-base/references/ai-powered-call-center-intelligence/churn_model/train.py` | Read `train.py`; capture the split, fit, metric set (accuracy/AUC/precision/recall), and the `dump()` artifact path | Pipeline inventory: step → tool → metric → artifact → `path:line`; informs our retrain+benchmark+version flow |
| Q3 | What does ai-powered depend on for PII redaction, and at what versions? | deps | `knowledge-base/references/ai-powered-call-center-intelligence/requirements.txt`, `backend/pii_redaction.py` | SKIP Fase A (text-shape). Grep `presidio|spacy` in `requirements.txt`; Grep the imports in `pii_redaction.py` | Read the matched lines to confirm the analyzer/anonymizer + spaCy model + versions | Dependency table: lib → version → role → citation; informs the build-vs-adopt decision for PT-BR |
| Q4 | What does ai-powered's churn train depend on (ML libs), and how is the artifact loaded back? | deps | `knowledge-base/references/ai-powered-call-center-intelligence/churn_model/train.py` | SKIP Fase A (text-shape). Grep `import|from` in `train.py`; find the `dump(`/`load(` calls | Read the imports + the dump/load calls in context | Dependency + artifact-format table with citations; informs reuse of our `SentimentDetector` + joblib |
| Q5 | How does chatwoot TEST its data-export job (real data, the produced artifact, any anonymization)? | tests | `knowledge-base/references/chatwoot/spec/jobs/account/contacts_export_job_spec.rb` | `ast-grep run -p 'it $$$ do $$$ end' --lang ruby knowledge-base/references/chatwoot/spec/jobs/account/contacts_export_job_spec.rb` to map example blocks | Read each example + its factory/setup; capture what is asserted about the exported artifact | Table: spec example → setup → assertion (artifact shape / content) → `path:line`; informs M7's export + anonymization test tier |
| Q6 | How does chatwoot's export job PRODUCE + STORE the export (format, storage backend, lifecycle)? | tools | `knowledge-base/references/chatwoot/app/jobs/account/contacts_export_job.rb` | `ast-grep run -p 'def $NAME($$$)' --lang ruby knowledge-base/references/chatwoot/app/jobs/account/contacts_export_job.rb` to map the job methods | Read the job fully; capture the serialization format (CSV?), the storage attach (ActiveStorage?), and any cleanup | Prose + a produce→store→cleanup flow with citations; informs our Parquet + storage-port + pre-purge trigger |
| Q7 | How is ai-powered's churn train structured as a runnable PIPELINE (entrypoint, artifacts, reproducibility)? | tools | `knowledge-base/references/ai-powered-call-center-intelligence/churn_model/train.py` | `ast-grep run -p 'def main($$$): $$$' --lang python knowledge-base/references/ai-powered-call-center-intelligence/churn_model/train.py` | Read `main()` + the artifact writes; capture the entrypoint shape, the artifacts produced, and the random_state/reproducibility posture | Pipeline-shape table: entrypoint → inputs → artifacts → reproducibility → citation; informs our retrain script structure |

## Coverage Matrix

| Corner | Questions mapped | Status |
|---|---|---|
| Integration tests | Q5 | Covered |
| Dependencies | Q3, Q4 | Covered |
| Tools | Q6, Q7 | Covered |
| Techniques | Q1, Q2 | Covered |

**Coverage: 4/4 corners covered (100%)**

## Halt-loop Checkpoints

| Checkpoint | Assertion | Action if fails |
|---|---|---|
| Before answering Qx | Every `knowledge-base/references/{project}/{path}` declared in Qx's Fase A exists | Mark Qx BLOCKED ("path not found"), continue |
| Per-question Fase A budget | Fase A returned ≥ 1 hotspot OR 3 retries attempted | After 3 empty retries, mark Qx BLOCKED ("Fase A exhausted"); continue |
| After answering Qx | Blueprint section under Qx has ≥ 1 citation | Re-iterate Qx (1 retry max) |
| Mid-loop sanity | Total `knowledge-base/references/` citations ≥ prose-words / 200 | Add citations to under-cited paragraphs (1 retry max) |
| Per-project time budget | Project time budget not exhausted | When exhausted, mark remaining Qx BLOCKED ("budget exhausted"); advance |
| Before promising complete | All 4 coverage corners have populated sections | Refuse promise, continue iterating |

## Acceptance Criteria

- [ ] All 7 research questions answered OR explicitly marked BLOCKED with reason
- [ ] All four coverage corners have populated sections in the blueprint
- [ ] Every citation points to a real `knowledge-base/references/{...}` path
- [ ] At least one ADR section in the blueprint synthesizes the M7 PII/export/retrain/version decisions
- [ ] Time budget respected per project
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m7-retraining-lifecycle-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed → confidence re-score)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations
- [ ] Coverage Matrix 100% covered
- [ ] ADRs reference at least one principle from project rules — here `.claude/rules/architecture.md § 2` (storage behind a domain port; DIP) + `testing.md § 4.1` (the anonymization negative-case asserts PII ABSENT) + the ADR-005 hot/purge split (export must precede the 30-day drop)
