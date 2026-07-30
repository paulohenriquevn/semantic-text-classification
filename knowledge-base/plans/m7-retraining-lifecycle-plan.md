# Plan: M7 Retraining Loop & Data Lifecycle

> **Version 1.0** — Close the offline data lifecycle: a PT-BR PII redactor behind a `Redactor` port, an
> anonymized retraining-sample export to Parquet behind a `SamplePort` (export-before-purge — the drop is
> gated on a verified export), and a retraining pipeline that consumes QA labels (M5) + exported samples to
> produce a benchmarked, versioned `SentimentDetector` (promote only on a macro-F1 gain). Per the SHIPPABLE
> 96.0 blueprint `knowledge-base/discoveries/blueprints/m7-retraining-lifecycle-blueprint.md`.

## Goal

> "Export an anonymized retraining sample and retrain a versioned model, measured by an integration test
> suite that asserts (a) the exported Parquet contains NO raw PII (CPF/phone/email absent), (b) the raw
> chunk is dropped only AFTER a verified export, and (c) the retrain produces a new model version with a
> macro-F1 benchmarked against the deployed one."

## Context

M7 (`ROADMAP.md § M7`) closes the offline lifecycle: export an anonymized/pseudonymized retraining sample to
Parquet/object storage BEFORE the 30-day chunk drop, retrain sentiment on accumulated QA labels + exported
samples producing a benchmarked new model version, and track model versions. It depends on M1 (retention/purge,
migration 0002), M2 (`SentimentDetector` — `src/talkex/classification/sentiment.py`, train/evaluate/predict +
joblib), and M5 (`labels` table). The blueprint (SHIPPABLE 96.0) locked 6 ADRs: a focused PT-BR regex redactor
behind a `Redactor` port — an explicit build-vs-adopt call against blindly porting the peer's English Presidio
config (D1); redact-vs-preserve keeping `turn_id`/`conversation_id` join keys (D2); Parquet behind a `SamplePort`
(D3); export-before-purge ordering — drop gated on a verified export (D4); retrain + benchmark reusing the
shipped `SentimentDetector`, promote on gain (D5); `model_version` on every artifact/prediction (D6). Risk 1 is
LGPD-critical: no raw PII may leak into the cold sample. Constrained by `.claude/rules/architecture.md § 2`
(storage/redaction behind domain ports; DIP) and `.claude/rules/testing.md § 4.1` (the anonymization
negative-case asserts PII ABSENT — a typed guarantee, not "it ran").

## Baseline Context (deep review of current state)

### Files that will be touched

| File | LoC today | Last commit | Why it exists today | Invariants to preserve |
|---|---|---|---|---|
| `src/talkex/classification/sentiment.py` (SentimentDetector) | ~100 | M2 | train/predict/evaluate + joblib | additive: add `model_version`; keep the fitted-guard + macro-F1 |
| `experiments/scripts/train_sentiment.py` | ~70 | M2 | the M2 train entrypoint (load→build→dump) | the pattern the retrain script extends (not edited) |
| `src/talkex/monitoring/domain/ports.py` | ~90 | M0-M6 | domain Protocols | additive: `Redactor`, `SamplePort` |
| `src/talkex/monitoring/domain/lifecycle.py` (NEW) | 0 | — | `RetrainingSample`, `ExportResult`, `BenchmarkResult` value objects | — |
| `src/talkex/monitoring/infrastructure/pii_redactor.py` (NEW) | 0 | — | PT-BR regex redactor (CPF/CNPJ/phone/email/name) | — |
| `src/talkex/monitoring/infrastructure/parquet_sample_store.py` (NEW) | 0 | — | Parquet SamplePort impl | — |
| `src/talkex/monitoring/application/lifecycle_exporter.py` (NEW) | 0 | — | reads turns+labels → redact → Parquet; export-before-purge | — |
| `src/talkex/classification/retraining_pipeline.py` (NEW) | 0 | — | load samples+labels → retrain → benchmark → version | — |

### Current callers / dependents

- **`SentimentDetector`** (`sentiment.py`) — trained in `experiments/scripts/train_sentiment.py`, used by the M2/M3 orchestrator cascade (`_sentiment_evidence`). Adding `model_version` is additive (defaulted). The retrain produces a NEW instance; the deployed one is unchanged unless promoted.
- **`labels` table** (M5, migration 0003) — the QA-label source; read-only here.
- **`turns.raw_text`** — carries PII; the export is the ONLY consumer that anonymizes it.

### Domain glossary

- **Redactor** — a port that maps text → text with PT-BR PII (CPF, CNPJ, phone, email, names) replaced by tokens; join keys preserved.
- **RetrainingSample** — one anonymized (text, label) row destined for Parquet.
- **Export-before-purge** — the ordering invariant: `drop_chunks` runs only after a verified successful export of the to-be-dropped window (ADR-005 hot/purge split).
- **model_version** — a semantic tag stamped on the artifact + on every prediction (evidence axiom).

### Architecture boundaries affected

Offline lifecycle: two new domain ports (`Redactor`, `SamplePort`), a redactor + Parquet adapter (infra), an
export application service, and a retraining pipeline in `classification`. DIP preserved (domain declares the
ports; infra implements). No online/ingest path touched.

### ⚠ Baseline reality checks

1. **PT-BR PII is greenfield.** The peer's Presidio redactor is English/US (`en`, SSN). Brazilian PII (CPF,
   CNPJ, phone, email, names) needs a focused regex redactor — Presidio-PT is immature (blueprint D1). This is
   a "the lib does not fit our language" build case (Rule 9 exception), documented in an ADR.
2. **`model_version` does not exist on `SentimentDetector`.** M2 shipped it without a version; D6 adds it (the
   evidence axiom — every prediction records its model version — is not yet honored for sentiment).
3. **pyarrow 25.0.0 + pandas 3.0.5 are installed** (added for M7); the export writes real Parquet.

## Prior Art & Related Work

- **Internal blueprint** — `m7-retraining-lifecycle-blueprint.md` (its ADR set + Cross-cutting Comparison).
- **Reference — ai-powered** — PII redact-vs-preserve pattern (`knowledge-base/references/ai-powered-call-center-intelligence/backend/pii_redaction.py`), a train/benchmark pipeline (`churn_model/train.py`). The English Presidio config is the anti-pattern for PT-BR (blueprint D1).
- **Reference — chatwoot** — a data-export job (`knowledge-base/references/chatwoot/app/jobs/account/contacts_export_job.rb`) + its real-DB spec (`spec/jobs/account/contacts_export_job_spec.rb`) — the export/store + how-to-test analog.
- **Internal reuse** — the shipped `SentimentDetector` (retrained, not rebuilt — DRY); the `embeddings/config.py:37` `model_version` precedent.

## Objective

- [ ] `Redactor` port + a PT-BR regex redactor (CPF/CNPJ/phone/email/name), join keys preserved
- [ ] `SamplePort` + a Parquet store; a `DataLifecycleExporter` (redact → Parquet) with export-before-purge
- [ ] A retraining pipeline: load samples + labels → retrain `SentimentDetector` → benchmark vs deployed
- [ ] `model_version` on the detector + on every prediction

## ADRs

### D1 — Focused PT-BR regex redactor (build, not blind-adopt)
- **Decision:** a `Redactor` port with a regex impl covering CPF, CNPJ, phone (DDD), email, and a name heuristic.
- **Rationale:** the peer's Presidio is English/US (`en`, SSN); Presidio-PT is immature. A focused, well-tested PT-BR regex is more reliable AND lighter (Rule 9 "the lib does not fit" exception; `architecture.md § 2` DIP).
- **Alternatives considered:** Presidio + a PT spaCy model (rejected — immature PT NER, heavy); no redaction (rejected — LGPD failure).
- **Consequence:** deterministic, testable redaction; the port lets us swap in Presidio later if PT support matures.

### D2 — Redact sensitive PII, preserve join keys
- **Decision:** redact PII in `raw_text`; preserve `turn_id`/`conversation_id` (pseudonymous keys) for label joins.
- **Rationale:** the peer preserves account IDs for supervisor review (`pii_redaction.py:13-20`); joins need stable keys.
- **Alternatives considered:** redact everything incl. keys (rejected — breaks the label join).
- **Consequence:** the sample is anonymized yet joinable to labels.

### D3 — Parquet behind a SamplePort
- **Decision:** a `SamplePort.write(rows) -> path`; a Parquet impl (pyarrow).
- **Rationale:** Parquet is the columnar retraining-sample standard; the port abstracts local-dir vs S3/MinIO (DIP).
- **Alternatives considered:** CSV (rejected — no schema/typing); a DB table (rejected — must survive the purge as a cold file).
- **Consequence:** typed columnar samples; storage swappable.

### D4 — Export-before-purge ordering (synthesis)
- **Decision:** `drop_chunks` runs ONLY after a verified successful export of the to-be-dropped window.
- **Rationale:** neither peer states this; derived from ADR-005 (raw purges; cold sample must precede). A purge before export = permanent data loss.
- **Alternatives considered:** independent schedules (rejected — a race can drop un-exported data).
- **Consequence:** no retraining data is ever lost to the purge.

### D5 — Retrain + benchmark, promote on gain
- **Decision:** the pipeline retrains a NEW `SentimentDetector` on samples + labels, benchmarks macro-F1 vs the deployed model on a held-out set, and promotes only on a gain.
- **Rationale:** reuse the shipped detector (DRY); the benchmark is the honest gate (ROADMAP risk 2 — low label volume may show no gain; then keep deployed).
- **Alternatives considered:** always promote the retrained model (rejected — could regress).
- **Consequence:** the deployed model never regresses; an honest-negative is recorded when there is no gain.

### D6 — model_version on artifact + prediction
- **Decision:** `SentimentDetector` carries a `model_version`; `predict` returns it; the retrain stamps a new one.
- **Rationale:** the evidence axiom (every prediction records its model version); mirrors `embeddings/config.py:37`.
- **Alternatives considered:** an external manifest only (rejected — the prediction itself must carry the version).
- **Consequence:** every sentiment prediction is traceable to a model version.

## Drawbacks & Risks

| Drawback / Risk | Severity | Mitigation | Owner |
|---|---|---|---|
| R1 — a PII regex miss leaks raw PII into the cold sample (LGPD) | High | D1 focused patterns + a negative-case test asserting CPF/phone/email ABSENT; conservative over-redaction preferred to a leak | tomas |
| R2 — low QA-label volume shows no retrain gain | Medium | D5 benchmark is the gate; record the honest-negative + keep the deployed model | tomas |
| R3 — a purge before export loses data | High | D4 ordering: drop only after a verified export (asserted in a test) | tomas |
| R4 — a name-heuristic over-redacts common words | Medium | keep the name heuristic conservative (capitalized bigrams after a title/greeting); documented; the label is unaffected (text-only) | kael |

## Unresolved Questions

- Q1 — Exact PT-BR name-detection recall target? M7 uses a conservative heuristic (title/greeting + capitalized token); full NER deferred (YAGNI until a leak is observed).
- Q2 — Object-storage backend for production (S3/MinIO)? M7 ships the local-dir Parquet impl behind `SamplePort`; the cloud impl is a later swap (DIP).

## Dependency Graph

```
Phase 0 (PT-BR PII Redactor)  ──▶  Phase 1 (SamplePort + Parquet export; export-before-purge)
                                          ▼
                              Phase 2 (retrain + benchmark + model_version)
                                          ▼
              Final: Integration Validation (anonymized Parquet has NO PII; retrain produces a benchmarked versioned model)
```

---

## Phase 0: PT-BR PII Redactor

**Objective:** `Redactor.redact(text)` removes PT-BR PII, preserves join keys.

### T0.1 — Redactor port + PT-BR regex impl

#### Objective
A `Redactor` port and a regex impl redacting CPF, CNPJ, phone (DDD), email, and heuristic names.

#### Why this step
1. **What:** `domain/ports.py` `Redactor`; `infrastructure/pii_redactor.py` `RegexRedactor`.
2. **Why now:** every exported row must be anonymized first (blueprint D1/D2; LGPD risk R1).

#### Files to edit
```
src/talkex/monitoring/domain/ports.py — Redactor Protocol
src/talkex/monitoring/infrastructure/pii_redactor.py (NEW)
tests/unit/monitoring/test_pii_redactor.py (NEW)
```

#### TDD
```
RED: test_redacts_cpf_phone_email — a text with a CPF, a DDD phone, and an email yields NONE of them
     test_preserves_non_pii — ordinary words survive; the label text is intact
GREEN: implement the regexes
VERIFY: pytest tests/unit/monitoring/test_pii_redactor.py -x
```

#### Concurrency tests

(none — single-threaded) — the redactor is a pure text function.

#### Acceptance Criteria
- [ ] `test_redacts_cpf_phone_email` asserts CPF, phone, and email are ABSENT after redaction (negative-case, typed guarantee)
- [ ] `test_preserves_non_pii` asserts ordinary text survives

#### DoD
- [ ] redactor green; no PII pattern survives

---

## Phase 1: Sample export

**Objective:** an anonymized Parquet sample, exported before purge.

### T1.1 — SamplePort + Parquet store + exporter

#### Objective
A `SamplePort` + a Parquet impl, and a `DataLifecycleExporter` that reads turns+labels in a window, redacts, writes Parquet, and gates the purge on a verified export.

#### Why this step
1. **What:** `domain/lifecycle.py` (`RetrainingSample`, `ExportResult`); `ports.py` `SamplePort`; `infrastructure/parquet_sample_store.py`; `application/lifecycle_exporter.py`.
2. **Why now:** the cold sample must exist before the 30-day drop (blueprint D3/D4; risk R3).

#### Files to edit
```
src/talkex/monitoring/domain/lifecycle.py (NEW) — RetrainingSample, ExportResult
src/talkex/monitoring/domain/ports.py — SamplePort
src/talkex/monitoring/infrastructure/parquet_sample_store.py (NEW)
src/talkex/monitoring/application/lifecycle_exporter.py (NEW)
tests/integration/monitoring/test_lifecycle_export.py (NEW)
tests/unit/monitoring/test_parquet_sample_store.py (NEW)
```

#### TDD
```
RED (unit):        test_parquet_roundtrip — write RetrainingSample rows, read the Parquet back, rows match
RED (integration): test_export_is_anonymized_and_precedes_purge — seed turns with PII + labels; export;
                   assert the Parquet has NO raw PII; then purge; assert the sample file still exists
GREEN: implement the store + exporter (export → verify → allow drop)
VERIFY: pytest tests/unit/monitoring/test_parquet_sample_store.py tests/integration/monitoring/test_lifecycle_export.py -x
```

#### Concurrency tests

(none — single-threaded) — the export is an offline batch job.

#### Acceptance Criteria
- [ ] the exported Parquet contains NO raw PII (the LGPD gate)
- [ ] the purge runs only after a verified export (ordering asserted)
- [ ] the Parquet round-trips (write→read)

#### DoD
- [ ] anonymized Parquet export green; export-before-purge proven

---

## Phase 2: Retraining + versioning

**Objective:** a benchmarked, versioned retrained model.

### T2.1 — retraining pipeline + model_version

#### Objective
Add `model_version` to `SentimentDetector`; a `RetrainingPipeline` that loads samples + labels, retrains, and benchmarks macro-F1 vs the deployed model.

#### Why this step
1. **What:** `sentiment.py` `model_version`; `classification/retraining_pipeline.py` (NEW).
2. **Why now:** DoD #2 (retrain + benchmark) + #3 (model versioning); blueprint D5/D6.

#### Files to edit
```
src/talkex/classification/sentiment.py — add model_version (predict returns it)
src/talkex/classification/retraining_pipeline.py (NEW)
tests/unit/classification/test_retraining_pipeline.py (NEW)
tests/unit/classification/test_sentiment_versioning.py (NEW)
```

#### TDD
```
RED: test_predict_records_model_version — a trained detector's prediction carries its model_version
     test_retrain_benchmarks_vs_deployed — the pipeline returns a BenchmarkResult(new_f1, deployed_f1, promoted)
GREEN: implement versioning + the pipeline
VERIFY: pytest tests/unit/classification/test_retraining_pipeline.py tests/unit/classification/test_sentiment_versioning.py -x
```

#### Concurrency tests

(none — single-threaded) — training is a batch fit.

#### Acceptance Criteria
- [ ] every prediction records the model_version (evidence axiom)
- [ ] the retrain returns a benchmark (new vs deployed macro-F1) and promotes only on a gain

#### DoD
- [ ] versioned retrain + benchmark green

---

## Coverage Matrix

| # | Gap / Requirement (DoD) | Task(s) | Resolution |
|---|---|---|---|
| 1 | Anonymized sample exported before purge (DoD #1) | T0.1, T1.1 | PT-BR redactor + Parquet export + ordering |
| 2 | No raw PII in the cold sample (LGPD risk R1) | T0.1, T1.1 | negative-case: PII absent |
| 3 | Retrain consumes labels + samples → new version (DoD #2) | T2.1 | RetrainingPipeline |
| 4 | Benchmarked vs deployed (DoD #2) | T2.1, T3.1 | BenchmarkResult new-vs-deployed |
| 5 | Model versioning; every prediction records version (DoD #3) | T2.1 | model_version on detector + prediction |
| 6 | Export-before-purge ordering (risk R3) | T1.1, T3.1 | drop only after verified export |

**Coverage: 6/6 gaps covered (100%)**

## Global Definition of Done

- [ ] `ruff format --check . && ruff check . && mypy src/ tests/` clean
- [ ] `pytest tests/unit -x && pytest tests/integration -x` green
- [ ] Exported Parquet contains NO raw PII — proven by the negative-case test (the LGPD gate)
- [ ] Retrain produces a benchmarked, versioned model — proven by a test
- [ ] File-size ≤ 500 LoC per file
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] pyarrow + pandas added to the project dependencies (M7 export extra)

## Failure scenarios (external I/O)

| Dependency | Failure mode | How the test reproduces it | Expected behavior |
|---|---|---|---|
| Parquet store (filesystem) | write fails / dir missing | point the store at an unwritable path | typed error surfaced; the purge is NOT allowed (export-before-purge holds) |
| TimescaleDB (export read) | empty window | export a window with no turns | writes an empty (schema-valid) Parquet, returns 0-row ExportResult, no crash |
| deployed model artifact | missing baseline for the benchmark | benchmark with no deployed model | the pipeline treats "no baseline" as promote-the-first-model (documented), not a crash |

## Final Phase: Integration Validation (MANDATORY)

**Objective:** prove the LGPD gate + the retrain benchmark with real evidence.

### T3.1 — LGPD + retrain-benchmark validation

#### Objective
Prove the exported Parquet leaks no PII and the retrain produces a benchmarked versioned model.

#### Why this step
1. **What:** integration assertions across the export + retrain; run against real Timescale + a real Parquet file.
2. **Why now:** DoD #1/#2/#3 are proven only by measurement (blueprint D4/D5/D6).

#### Files to edit
```
tests/integration/monitoring/test_lifecycle_export.py — add the end-to-end LGPD + ordering assertions
experiments/scripts/run_retraining.py (NEW) — a runnable pipeline entrypoint producing a benchmark JSON
```

#### TDD
```
RED: test_end_to_end_export_then_retrain — seed PII turns + labels; export (assert no PII); retrain from the
     Parquet; assert a BenchmarkResult + a stamped model_version
GREEN: wire the entrypoint
VERIFY: pytest tests/integration/monitoring/test_lifecycle_export.py -x && python experiments/scripts/run_retraining.py --help
```

#### Concurrency tests

(none — single-threaded) — the pipeline is an offline batch.

#### Acceptance Criteria
- [ ] the exported Parquet has NO raw PII (end-to-end LGPD gate)
- [ ] the purge follows a verified export
- [ ] the retrain emits a benchmark (new vs deployed) + a model_version

#### DoD
- [ ] LGPD + retrain-benchmark evidence recorded
