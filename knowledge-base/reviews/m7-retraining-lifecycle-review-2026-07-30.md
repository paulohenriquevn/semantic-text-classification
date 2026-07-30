# Review — M7 Retraining Loop & Data Lifecycle

Date: 2026-07-30
Plan: `knowledge-base/plans/completed/m7-retraining-lifecycle-plan.md` (SHIPPABLE_WITH_CAVEATS 70)
Slice commits: Phase 0 `feat(classification)`, Phase 1 `feat(pipeline)`, Phases 2+Final `feat(classification)`
Reviewer: cycle self-review

## Scope reviewed (all ≤ 500 LoC)

| File | LoC | Verdict |
|---|---|---|
| `domain/lifecycle.py` (RetrainingSample/ExportResult/BenchmarkResult) | 41 | OK |
| `infrastructure/pii_redactor.py` (PT-BR RegexRedactor) | 37 | OK |
| `infrastructure/parquet_sample_store.py` (Parquet SamplePort) | 51 | OK |
| `infrastructure/lifecycle_repo.py` (read window + purge) | 36 | OK |
| `application/lifecycle_exporter.py` (export-before-purge) | 44 | OK |
| `classification/retraining_pipeline.py` (retrain + benchmark) | 58 | OK |
| `classification/sentiment.py` (+model_version) | +8 | OK (additive) |
| `experiments/scripts/run_retraining.py` (entrypoint) | 84 | OK |

## Plan coverage (Coverage Matrix 6/6)

| # | Requirement (DoD) | Task | Status + evidence |
|---|---|---|---|
| 1 | Anonymized sample exported before purge | T0.1/T1.1 | ✅ PT-BR redactor + Parquet export + ordering; `test_lifecycle_export` |
| 2 | No raw PII in the cold sample (LGPD) | T0.1/T1.1 | ✅ `test_pii_redactor` + `test_export_is_anonymized`; **real run** m7_retrain.json — CPF/email ABSENT (`[CPF]` token) |
| 3 | Retrain consumes labels + samples → new version | T2.1 | ✅ `RetrainingPipeline`; `test_retraining_pipeline` |
| 4 | Benchmarked vs deployed | T2.1/T3.1 | ✅ `BenchmarkResult(new_f1, deployed_f1, promoted)`; **real run** new_f1=1.0, promoted |
| 5 | Model versioning; every prediction records version | T2.1 | ✅ `SentimentPrediction.model_version`; `test_sentiment_versioning` |
| 6 | Export-before-purge ordering | T1.1/T3.1 | ✅ `test_export_then_purge_ordering` + `test_failed_export_does_not_purge` (raw intact on failure) |

## Global DoD

- [x] `ruff format --check` + `ruff check` + `mypy src/ tests/` clean
- [x] `pytest tests/unit` (2075 pass, 1 skip) + `pytest tests/integration` green
- [x] Exported Parquet contains NO raw PII — the LGPD gate, proven by unit + integration + a real end-to-end run
- [x] Retrain produces a benchmarked, versioned model — `experiments/results/m7_retrain.json` (real run)
- [x] File-size ≤ 500 LoC per file (largest 84)
- [x] pyarrow + pandas added to project dependencies (mypy override)

## Findings

### F1 — PT-BR redactor is build-not-adopt, honestly justified — INFO (design strength)
The peer's Presidio config is English/US (SSN, spaCy `en`); a focused PT-BR regex (CPF/CNPJ/phone/email/name) is more reliable for our language and lighter. The `Redactor` port lets Presidio-PT swap in later if it matures. Over-redaction is preferred to a leak (documented) — the LGPD failure mode is a leak, not a false `[NOME]`.

### F2 — export-before-purge is an ordering PROOF, not a comment — INFO (the M7 crux)
`test_failed_export_does_not_purge` points the store at an unwritable path, asserts the export raises, and asserts the raw row is STILL present — proving the drop is unreachable without a successful export. `test_export_then_purge_ordering` proves the happy path (export → drop). This is the concrete realization of edge-case EC-2 / ADR-005.

### F3 — model_version recorded at the prediction, not yet in the alert evidence — INFO, scoped (accepted)
`SentimentPrediction.model_version` honors the evidence axiom at the model boundary (D6). The M3/M6 alert `evidence` JSONB + the `alerts.sentiment` column still store only the label — surfacing the version into the alert evidence would touch the M3 evidence shape and is deferred (YAGNI until an audit needs per-alert model provenance). The axiom is honored where predictions are produced; the propagation is a documented follow-up.

### F4 — retrain gate is honest about low label volume — INFO
`promoted` is strict-gain-only; the real run had no incumbent (`deployed_f1=null` → promote the first model). The pipeline records the benchmark rather than always promoting — the ROADMAP risk-2 honest-negative path is built in.

### F5 — the F1=1.0 is on a tiny hermetic eval — INFO, honest caveat (accepted)
The real run's F1=1.0 is on an 8-row seeded dataset with a 2-row eval — it proves the PIPELINE works end-to-end (export→retrain→benchmark→version), NOT a production relevance number. A production retrain would use the accumulated real labels; the number is deliberately framed as pipeline-evidence, not a model-quality claim.

No correctness, security, or resource findings.

## Verdict

**READY_TO_MERGE** — all 6 Coverage-Matrix requirements and the full Global DoD are met with real evidence. The LGPD gate (no PII in the cold sample) is proven at three levels (unit, integration, real run); export-before-purge is proven including the failure path. Findings are design notes and two accepted, documented scoping/measurement caveats; none blocking.
