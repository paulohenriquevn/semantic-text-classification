# Blueprint: M7 Retraining Loop & Data Lifecycle

**Slug:** `m7-retraining-lifecycle`
**Date:** 2026-07-30
**Plan reference:** `knowledge-base/discoveries/plans/m7-retraining-lifecycle-plan.md`
**Edge-cases reference:** `knowledge-base/reviews/m7-retraining-lifecycle-edge-cases-2026-07-30.md`

## Executive summary

M7 closes the offline data lifecycle for TalkEx: **export an anonymized retraining sample to Parquet/object
storage BEFORE the 30-day chunk drop, retrain the shipped `SentimentDetector` on accumulated QA labels (M5) +
exported samples, benchmark the candidate against the deployed model, and stamp a `model_version` on every
prediction.** Two cloned peers supply the technical patterns:

- **`ai-powered-call-center-intelligence`** — a Presidio-based PII redactor with an explicit
  redact-vs-preserve policy that *keeps account IDs for supervisor review* (`backend/pii_redaction.py`), plus a
  self-contained `train → benchmark → dump(joblib)` pipeline (`churn_model/train.py`).
- **`chatwoot`** — a production data-export job (`app/jobs/account/contacts_export_job.rb`) with a real-DB spec
  (`spec/jobs/account/contacts_export_job_spec.rb`) that asserts the produced artifact end-to-end and downloads
  it back to verify content.

All 7 research questions are fully answered from the cited files; there are **no fabricated citations** and
**no BLOCKED questions**. Two synthesis points are neither peer's — the blueprint promotes them to ADRs: the
**PT-BR PII entity set + build-vs-adopt redactor call** (peers are English/US-centric) and the
**export-before-purge ordering guarantee** (neither peer ties an export to a retention purge).

## Context

Per `docs/adr/ADR-005-online-storage-realtime-monitoring.md:45-56`, TalkEx keeps a 30-day hot window in one
Postgres/TimescaleDB instance and drops whole chunks at 30 days via `add_retention_policy` — confirmed in
`deploy/monitoring/migrations/0002_m1_retention_indexes.sql:34` (`add_retention_policy('turns', INTERVAL '30
days', ...)`). The `turns` table stores **raw PII** in `raw_text` (`deploy/monitoring/migrations/
0001_m0_hypertable.sql:11`) and purges it; ADR-005 mandates that *only an anonymized/pseudonymized sample*
survives, exported **before** the drop (`ADR-005:53-56`). The QA-label source is the M5 `labels` table
(`label_id, turn_id, conversation_id, label, labeled_by, created_at` —
`deploy/monitoring/migrations/0003_m5_search.sql:7-14`), indexed by recency for retraining export
(`labels_created_idx` — `0003_m5_search.sql:18`). M7 retrains the *already shipped* `SentimentDetector`
(`src/talkex/classification/sentiment.py:64`, TF-IDF word+char + LinearSVC, macro-F1 ≈ 0.856) using the M2
train entrypoint pattern (`experiments/scripts/train_sentiment.py:54`). This is a data-governance decision
(tomas-herrera's domain) constrained by `.claude/rules/architecture.md § 2` (storage behind a domain port; DIP)
and `.claude/rules/testing.md § 4.1` (the anonymization negative-case asserts PII **absent** — a typed
guarantee, not "it ran").

## Objective

Lock the M7 architecture from peer evidence: (1) the **PT-BR PII redaction strategy** (entity set +
redact-vs-preserve policy + build-vs-adopt call), (2) the **export format + storage boundary** (Parquet behind
a DIP port), (3) the **export-before-purge ordering**, and (4) the **retrain → benchmark → version** flow that
reuses `SentimentDetector` and refuses to promote a candidate that does not beat the deployed model.

## Coverage Corner 1 — Integration Tests

**Maps to Q5** — *How does chatwoot TEST its data-export job (real data, produced artifact, anonymization)?*

chatwoot's export-job spec is a textbook **integration test against a real DB + real storage**, not a mock
theatre. It seeds real `contact` rows via factories (`spec/.../contacts_export_job_spec.rb:50-58`), runs the
job synchronously with `described_class.perform_now(...)` (`:65`), then **downloads the produced artifact back
and parses it** to assert content — the artifact is the oracle, not an internal call count.

| Spec example (`spec/jobs/account/contacts_export_job_spec.rb`) | Setup | Assertion on the artifact | line |
|---|---|---|---|
| `generates CSV file and attach to account` | stub mailer double | `account.contacts_export` present; blob URL present; mailer received `contact_export_complete(url, email)` | `:60-72` |
| `generates valid data export file` | 10 seeded contacts | `CSV.parse(account.contacts_export.download)` row count == `account.contacts.count`; emails & phone numbers present in parsed rows | `:74-85` |
| `exports labels when requested` | add `vip` label to a contact | download → strip BOM → parse → row for that contact has `labels == 'vip'`; headers == requested | `:87-100` |
| `bulk loads labels while exporting` | 2 labels on every contact | asserts **exactly 1** `FROM "taggings"` SQL query via `ActiveSupport::Notifications` — an **N+1 regression guard** | `:102-122` |
| `prepends UTF-8 BOM` | default cols | `raw.bytes[0..2] == [0xEF,0xBB,0xBF]` — asserts byte-level artifact encoding | `:124-129` |
| filter variants (label / single / multiple filters) | filtered seeds | parsed row count matches the filter's expected subset | `:131-168` |

**Anonymization gap (honest):** chatwoot's export is **not anonymized** — it deliberately exports raw
`email`/`phone_number` for a contacts export (`:83-84` asserts the raw email/phone survive). So chatwoot gives
us the *artifact-download-and-assert* test shape and the *N+1 query guard*, but **not** an anonymization
assertion — M7 must add that tier itself.

**Implication for M7's test tier (synthesized):**
1. **Integration (real DB + real storage port):** seed `turns` + `labels`, run the export job, **read the
   Parquet back** (pyarrow), assert row count == expected window size — mirroring chatwoot `:74-85`.
2. **Negative-case anonymization (`testing.md § 4.1`):** assert the parsed Parquet column that came from
   `raw_text` contains **no** CPF/phone/email — a typed *PII-absent* guarantee. This is the tier chatwoot lacks
   and the LGPD-critical one (ROADMAP risk 1).
3. **N+1 / cost guard (optional, borrowed from `:102-122`):** the export reads labels + turns in bounded
   queries, not per-row.

## Coverage Corner 2 — Dependencies

**Maps to Q3 (PII deps) + Q4 (churn-train ML deps + artifact load-back).**

### Q3 — ai-powered's PII dependencies

| Dependency | Version | Role | Citation |
|---|---|---|---|
| `presidio-analyzer` | **unpinned** | entity detection engine (`AnalyzerEngine`) | `requirements.txt:8`; import `backend/pii_redaction.py:2` |
| `presidio-anonymizer` | **unpinned** | redaction/anonymization (`AnonymizerEngine`) | `requirements.txt:9`; import `pii_redaction.py:3` |
| `spacy` | `==3.7.5` | token parsing; loads `en_core_web_sm` | `requirements.txt:7`; `pii_redaction.py:1,10` |
| `re` (stdlib) | n/a | custom `ACCOUNT_ID` regex `\b[0-9]{10,12}\b` | `pii_redaction.py:4,14-16` |

**Critical finding for M7 (EC-1):** the model loaded is **`en_core_web_sm`** (`pii_redaction.py:10`) and the
analyzer is called with **`language="en"`** (`pii_redaction.py:27`); the only auto-redacted entity is
**`US_SSN`** (`pii_redaction.py:20`). This stack is **English/US-centric** and does not transfer to Brazilian
PII (CPF, CNPJ, RG, DDD phone, PT names) — Presidio's PT recognizers are immature. Presidio is *unpinned* here,
which is a supply-chain smell we would not copy. See **ADR D1**.

### Q4 — ai-powered's churn-train ML dependencies + artifact load-back

| Dependency | Version | Role | Citation |
|---|---|---|---|
| `pandas` | `==1.5.3` | CSV load, feature grouping | `requirements.txt:16`; `churn_model/train.py:2,48` |
| `numpy` | (transitive) | Sturges bins, `hstack` stacking | `train.py:3,40,85` |
| `scikit-learn` | **unpinned** | split, `ColumnTransformer`, `OneHotEncoder`, `LogisticRegression`, `DecisionTreeClassifier`, `GaussianNB`, metrics | `requirements.txt:23`; `train.py:4-12` |
| `xgboost` | **unpinned** | Level-1 `XGBClassifier` | `requirements.txt:24`; `train.py:11,75` |
| `joblib` | (via sklearn) | **`dump(...)` artifact persistence** — no `load()` in this file | `train.py:13,77-78,92-94` |

**Artifact format + load-back (honest gap):** `train.py` persists **six** joblib artifacts —
`preprocessor.joblib`, `xgb_model.joblib`, `lr/dt/nb_model.joblib` (`train.py:77-78,92-94`) — but the file
**only writes** them; there is **no `load()` call** in `train.py` (Fase B searched: `dump` appears, `load` does
not). Load-back happens elsewhere in ai-powered (not in scope). So the peer confirms the *persistence* half
(joblib `dump`) but not the *reload* half.

**Implication for M7 (DRY):** TalkEx already persists with **joblib** in the exact same idiom —
`joblib.dump(model, MODEL_PATH)` (`experiments/scripts/train_sentiment.py:66`) — and the shipped
`SentimentDetector` (`src/talkex/classification/sentiment.py:64`) is a single `Pipeline`. M7 should **reuse
`SentimentDetector` + joblib**, persisting **one** pipeline artifact (not six), and add the reload path the peer
lacks. Do **not** adopt xgboost/the 2-level stack — that is churn-specific and violates KISS for our binary
negative-detection task.

## Coverage Corner 3 — Tools

**Maps to Q6 (chatwoot export job: produce + store + lifecycle) + Q7 (ai-powered churn train as a runnable
pipeline).**

### Q6 — chatwoot's export job: produce → store → deliver

Flow (`app/jobs/account/contacts_export_job.rb`):

1. **Entrypoint** — an async job, `queue_as :low` (`:2`), `perform(account_id, user_id, column_names, params)`
   (`:7`). Work is done off the request path on a low-priority queue.
2. **Produce** — `generate_csv(headers)` builds an **in-memory CSV** via `CSV.generate` (`:23-28`); column
   selection is validated against real DB columns (`valid_headers` → `Contact.column_names.include?` `:70-77`).
3. **Encode** — prepends a **UTF-8 BOM** so spreadsheets read non-ASCII correctly (`:82-91`).
4. **Store** — attaches to an **ActiveStorage** attachment on the account:
   `@account.contacts_export.attach(io: StringIO.new(...), filename: "..._contacts.csv", content_type:
   'text/csv')` (`:87-91`). Storage backend is abstracted behind ActiveStorage (local disk / S3 / GCS by
   config) — **the job never names a concrete backend**. This is exactly the DIP/port posture M7 wants.
5. **Deliver + lifecycle** — emails a **blob URL** (`rails_blob_url`) to the operator (`:94-102`).

**Lifecycle / cleanup (honest gap):** the job **does not delete** the export and is **not tied to a retention
purge** — it is on-demand, and `contacts_export.attach` *replaces* the previous single attachment. So chatwoot
gives us the **produce → store-behind-a-port → notify** shape, but **no pre-purge trigger and no cleanup
policy** (EC-2). M7 must synthesize the ordering itself — see **ADR D4**.

**M7 mapping:** swap CSV→**Parquet** (pyarrow 25.0.0 + pandas 3.0.5 are installed), swap
ActiveStorage→**a TalkEx `SamplePort` domain interface** (`architecture.md § 2`) with a Parquet/object-store
adapter, and swap "email a URL" → "record the export manifest + only then permit `drop_chunks`".

### Q7 — ai-powered's churn train as a runnable pipeline

`churn_model/train.py` is a **single-file, `main()`-guarded, reproducible** pipeline:

| Aspect | Evidence | line |
|---|---|---|
| Entrypoint | `def main():` under `if __name__ == '__main__': main()` | `:43,112-113` |
| Inputs | reads a fixed CSV path `data/raw/churn-in-telecom-dataset.csv` | `:48` |
| Artifacts dir | `os.makedirs('churn_model/artifacts', exist_ok=True)` created up-front | `:45` |
| Split | `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` | `:61-63` |
| Reproducibility | **`random_state=42` on every stochastic step** (split, xgb, lr, dt) | `:63,75,89,90` |
| Benchmark | prints Accuracy / AUC / Precision / Recall on the held-out test set | `:106-109` |
| Persist | `dump(...)` each fitted artifact to `artifacts/*.joblib` | `:77-78,92-94` |

**Weaknesses to NOT copy (honest):** hard-coded I/O paths (`:48`), **metrics printed to stdout, not persisted
to JSON** (`:106-109`) → not machine-comparable across runs, and **no model-version stamp** on the artifacts.
TalkEx's own M2 entrypoint is already better on two of these: it **writes metrics to JSON with a DoD gate**
(`experiments/scripts/train_sentiment.py:66-81`). M7's retrain script should follow the *structure* of
ai-powered's `main()` (dirs up-front, `random_state`, held-out benchmark) but keep TalkEx's **JSON-metrics +
DoD-gate** discipline and add a **version stamp** (D6).

## Coverage Corner 4 — Techniques

**Maps to Q1 (PII detect + redact + what is PRESERVED) + Q2 (churn train → benchmark → persist/version).**

### Q1 — detect + redact + deliberately preserve

The core technique is a **redact-vs-preserve split**, not blanket redaction:

| Entity type | Detected? | Redacted? | Mechanism | line |
|---|---|---|---|---|
| `US_SSN` | yes (Presidio) | **YES** (only this) | `REDACT_ENTITIES = ["US_SSN"]` → `anonymizer.anonymize` | `pii_redaction.py:20,43-44` |
| `PHONE_NUMBER`, `EMAIL_ADDRESS`, `PERSON`, `ADDRESS`, `DATE_TIME` | yes (Presidio) | **NO** (flagged only) | detected + returned as span metadata, not anonymized | `:26,42-54` |
| `ACCOUNT_ID` | yes (custom regex `\b[0-9]{10,12}\b`) | **NO — deliberately preserved** | "Account IDs: allowed for supervisor review — do not redact" | `:12-16,30-40` |

Mechanism: `analyzer.analyze(...)` returns `RecognizerResult` spans (`:24-28`); custom account-number matches
are appended as flagged (non-redacted) spans (`:31-40`); **only** entities in `REDACT_ENTITIES` are passed to
`anonymizer.anonymize` (`:43-44`); the function returns **both** `redacted_text` and full `pii_detected` span
metadata (`:47-59`). The design intent — *keep operationally-necessary identifiers (account IDs) visible to a
supervisor while redacting the truly sensitive field* — is the **pattern M7 adopts**, re-targeted to PT-BR (D1,
D2).

### Q2 — train → benchmark → persist/version

ai-powered's technique (`churn_model/train.py`): a **2-level stacked ensemble** — Level-1 XGBoost whose
predicted probability is stacked as a feature (`:74-86`) into Level-2 LR/DT/NB combined by **weighted soft
voting** (`VOTE_WEIGHTS = {'lr':0.4,'dt':0.3,'nb':0.3}`, `:32-33,100-103`), thresholded at 0.5 (`:103`), then
benchmarked with Accuracy/AUC/Precision/Recall on a stratified hold-out (`:60-63,106-109`). Persistence is
per-component joblib `dump` (`:77-78,92-94`). **There is no versioning of the artifacts** — files are
overwritten in place, no version string, no metrics file (Fase B: no `version`/`json.dump` in the file).

**Contrast with TalkEx (what M7 keeps):** our shipped detector is a **single** TF-IDF+LinearSVC `Pipeline`
(`sentiment.py:49-61`) with `train`/`predict`/`evaluate` behind a domain interface (`sentiment.py:75-101`), and
the M2 entrypoint already emits a **benchmarked metrics JSON with a DoD pass/fail** (`train_sentiment.py:63-81`).
M7's technique = **retrain that same `Pipeline` on {M5 labels + exported anonymized samples}, benchmark
candidate-vs-deployed on a frozen hold-out, and only promote on strict improvement** (D5), stamping a
`model_version` (D6) — reusing `sentiment.py`, not rebuilding a stacked ensemble (KISS/YAGNI).

## Cross-cutting Comparison

| Dimension | `ai-powered-call-center-intelligence` | `chatwoot` | TalkEx M7 decision |
|---|---|---|---|
| PII detection | Presidio + spaCy `en`, `language="en"` (`pii_redaction.py:10,27`) | none (exports raw email/phone `spec:83-84`) | PT-BR entity set via focused regex redactor (D1) |
| Redact-vs-preserve | redact `US_SSN`; preserve `ACCOUNT_ID` for supervisor (`:20,13-16`) | n/a | redact CPF/CNPJ/RG/phone/email/name; preserve `conversation_id`/`turn_id` (D2) |
| Export format | n/a | in-memory **CSV** + BOM (`job:23-28,82-91`) | **Parquet** (pyarrow installed) (D3) |
| Storage boundary | local `artifacts/*.joblib` hard path (`train.py:45,77`) | **ActiveStorage** port, backend by config (`job:87-91`) | domain `SamplePort` + object-store adapter (DIP) (D3) |
| Export ↔ purge ordering | n/a | on-demand, no purge link (`job` — none) | **export window BEFORE `drop_chunks`** (D4) |
| Train/benchmark | 2-level stack, stdout metrics, `random_state=42` (`train.py:74-109`) | n/a | reuse `SentimentDetector`, JSON metrics + DoD gate, candidate-vs-deployed (D5) |
| Model versioning | none (artifacts overwritten) | n/a | `model_version` per artifact + per prediction (D6) |
| Test discipline | none (no spec in scope) | real-DB, download-and-assert, N+1 guard (`spec:60-122`) | integration + **PII-absent negative case** (`testing.md §4.1`) |
| Dependency hygiene | Presidio/sklearn/xgboost **unpinned** (`requirements.txt:8-9,23-24`) | (Gemfile, out of scope) | pin exact versions; reuse joblib idiom already in repo |

## ADRs

### D1 — PT-BR PII redaction: build a focused regex redactor, do NOT port the English Presidio config

**Decision.** Build a **TalkEx PT-BR redactor** targeting the Brazilian PII entity set — **CPF, CNPJ, RG,
DDD/mobile phone, e-mail, and a PT person-name heuristic** — as the M7 v1 redactor. Do **not** adopt
ai-powered's `en`-Presidio config; keep Presidio-PT as a *deferred* option behind the same domain interface.

**Rationale.** The peer redactor is hard-wired to English: `spacy.load("en_core_web_sm")`
(`pii_redaction.py:10`), `analyzer.analyze(..., language="en")` (`:27`), and the only auto-redacted entity is
`US_SSN` (`:20`) — none of which detect Brazilian PII (EC-1). CPF/CNPJ/phone are **regular, checksum-verifiable
formats**, so a small deterministic regex redactor is high-precision, has **no model download**, is CPU-instant,
and is directly testable — matching TalkEx's stated preference for efficient traditional methods over heavy
deps (`sentiment.py:5-8`). Presidio's PT support is immature; adopting it blindly would ship an
English-detector that silently passes Brazilian PII into the cold sample — an LGPD leak (ROADMAP risk 1). This
respects Unbreakable Rule 9 honestly: we *don't reinvent* the redact-vs-preserve **pattern** (we lift it from
the peer), but we *do* write the PT-BR detector because no mature lib resolves it without gambiarra.

**Alternatives considered.** (a) Port ai-powered's Presidio+spaCy with a PT model — rejected: immature PT
recognizers, heavy spaCy dep, and no CPF/CNPJ recognizer out of the box. (b) LLM-based redaction — rejected for
the online path (cost/latency; LLMs are offline-only per `CLAUDE.md` axioms). (c) No redaction, rely on 30-day
purge alone — rejected: the *exported cold sample* survives the purge, so it MUST be redacted (`ADR-005:53-56`).

**Consequence.** M7 owns a `PtBrRedactor` behind a `Redactor` port; recall gaps on free-text names are a known
limitation (documented, tested), and Presidio-PT can later be swapped in behind the same port without touching
callers (OCP).

### D2 — Redact-vs-preserve policy: redact sensitive PII, preserve internal identifiers for review

**Decision.** Redact **CPF, CNPJ, RG, phone, e-mail, person-name** in the exported text; **preserve** TalkEx
internal identifiers (`conversation_id`, `turn_id`, `labeled_by`) unredacted so QA/supervisors can trace a
sample back to its label. Return **both** the redacted text and a PII-span manifest (count/type per record),
never the raw PII text, in the export.

**Rationale.** Directly lifts the peer's intent — *"Account IDs: allowed for supervisor review — do not redact"*
(`pii_redaction.py:13-16`) and its return of `{redacted_text, pii_detected}` (`:56-59`). TalkEx's analog of the
account ID is the label-join key: the M5 `labels` table references `turn_id`/`conversation_id`
(`0003_m5_search.sql:9-10`), which the retraining join needs; these are opaque internal IDs, not personal data,
so preserving them is safe and operationally required. Redacting the genuinely-sensitive set satisfies
`testing.md § 4.1` (the export test asserts these are **absent**).

**Alternatives considered.** (a) Redact everything including IDs — rejected: breaks the label↔sample join and
supervisor traceability. (b) Preserve phone/email like chatwoot's contacts export (`spec:83-84`) — rejected:
that export is a CRM feature with a different lawful basis; the M7 retraining sample is data-minimized under
LGPD.

**Consequence.** The export carries redacted text + a non-identifying PII-span manifest; the join key survives;
the negative-case test (D-in-Corner-1) asserts CPF/phone/email are gone.

### D3 — Export format = Parquet behind a domain `SamplePort` (DIP)

**Decision.** Serialize the anonymized retraining sample to **Parquet** and write it through a **`SamplePort`
domain interface**, with a concrete object-store/Parquet adapter injected at the composition root — never a
concrete storage SDK inside the export use case.

**Rationale.** ADR-005 already names Parquet as the survive-the-purge format (`ADR-005:51-56`), and pyarrow
25.0.0 + pandas 3.0.5 are installed (Parquet is available now, no new dep — parsimony ladder rung 4). chatwoot
proves the port pattern: its job attaches through **ActiveStorage** and never names disk/S3/GCS
(`contacts_export_job.rb:87-91`), so the backend is swappable by config. This is exactly `architecture.md § 2`
(storage is an infrastructure adapter behind a domain port; DIP): the M7 export use case depends on `SamplePort`,
the adapter satisfies it. Parquet (columnar, typed, compressed) beats chatwoot's CSV for an ML retraining
sample — typed columns, no BOM hacks, cheap to read back with pyarrow in the integration test.

**Alternatives considered.** (a) CSV like chatwoot (`job:23-28`) — rejected: untyped, lossy for ML, needs BOM
workarounds (`:82-91`). (b) Write directly to S3 in the use case — rejected: couples domain to a concrete SDK,
violates DIP, breaks the integration test's local-adapter substitution.

**Consequence.** Tests inject a local-filesystem `SamplePort`; production injects an object-store adapter; the
export use case is unit-testable with no cloud.

### D4 — Export-before-purge ordering: export the to-be-dropped window, THEN allow `drop_chunks` (SYNTHESIS)

**Decision.** The 30-day retention purge is **gated on a successful export**. The lifecycle is: (1) select the
window about to age out, (2) redact + export it to Parquet via `SamplePort`, (3) **verify the export succeeded**
(row count + manifest written), (4) **only then** permit the chunk drop. `drop_chunks`/`add_retention_policy`
must never race ahead of the export.

**Rationale.** This ordering is **neither peer's** — chatwoot's export is on-demand with no purge link
(`contacts_export_job.rb` has no retention hook), and ai-powered has no retention concept. It is synthesized
from `ADR-005:45-56`: raw `turns.raw_text` is dropped whole-chunk at 30 days
(`0002_m1_retention_indexes.sql:34`), and *"only anonymized retraining data leaves the hot store … exported
before the chunk is dropped"* (`ADR-005:53-56`). If the drop runs first, the sample is **irrecoverably gone**
(chunk-drop, not soft delete) — an LGPD-and-data-loss double failure (EC-2). Fail-fast (`error-handling.md`): a
failed export must **block** the drop and raise a typed error, not silently let the purge proceed.

**Alternatives considered.** (a) Rely on Timescale's `add_retention_policy` alone (`0002:34`) — rejected: it
drops on a timer with no export coupling; the sample would be lost. (b) Export lazily after the drop from a
backup — rejected: raw backups of PII contradict LGPD minimization (`ADR-005:55-56`). (c) Lengthen retention to
buy time — rejected: 30 days is a deliberate minimization constraint (`ADR-005:25`), not a tunable.

**Consequence.** M7 needs an **export-then-purge orchestration** (a scheduled job that exports the aging window
before the retention policy would drop it), plus an ordering **integration test** that proves a failed export
leaves the chunk intact (asserts the typed error + that the window still exists).

### D5 — Retrain by reusing `SentimentDetector`; benchmark candidate-vs-deployed; promote only on strict gain

**Decision.** M7's retrain script **reuses the shipped `SentimentDetector`** (`sentiment.py:64`) and the M2
entrypoint structure (`train_sentiment.py:54`), trains a candidate on {M5 labels + exported anonymized samples},
benchmarks it against the **currently deployed** model on a frozen hold-out (macro-F1, negative-F1), and
**promotes the candidate only if it strictly beats the deployed model**; otherwise it records the result and
keeps the deployed model.

**Rationale.** DRY/KISS/YAGNI: the detector, the joblib idiom (`train_sentiment.py:66`), and the JSON-metrics +
DoD gate (`:67-81`) already exist — M7 extends them, it does not rebuild them, and it explicitly does **not**
adopt ai-powered's churn-specific 2-level XGBoost stack (`churn_model/train.py:74-103`), which is over-built for
binary negative-detection. The candidate-vs-deployed gate is the honest answer to **EC-3 / ROADMAP risk 2**:
M5's label volume may be too low to beat the M2 baseline; the benchmark **is** the gate — if there's no gain, we
record it and keep the deployed model rather than shipping a regression. The peer's `main()` supplies the
reproducibility posture to copy (`random_state=42` on every stochastic step — `train.py:63,75,89,90`).

**Alternatives considered.** (a) Always promote the newest model — rejected: can silently regress on thin
labels (EC-3). (b) Port ai-powered's stacked ensemble — rejected: YAGNI for our task; a single Pipeline hits
macro-F1 0.856 already (`sentiment.py:6-7`). (c) Retrain from scratch each time without a baseline comparison —
rejected: no gate, no evidence the retrain helped (violates the "always benchmark" axiom in `CLAUDE.md`).

**Consequence.** Retraining is safe-by-default (never ships a worse model); a benchmarks JSON records
new-vs-deployed each run; low-label runs produce an honest "no gain — kept deployed" record, not a silent
downgrade.

### D6 — Stamp `model_version` on every artifact and every prediction

**Decision.** Each retrained artifact carries an explicit **`model_version`** string; every prediction records
the version of the model that produced it (label, score, `model_version`, threshold — per the "every prediction
carries evidence" axiom).

**Rationale.** ai-powered's fatal gap is **no versioning** — artifacts are overwritten in place with no version
string or metrics file (`train.py:77-78,92-94`), so you cannot tell which model produced which prediction or
reproduce a benchmark. TalkEx already has the precedent to follow: `EmbeddingModelConfig.model_version` is a
required, non-empty, frozen field (`src/talkex/embeddings/config.py:37,51-57`) — M7 mirrors that for the
classifier. This satisfies `CLAUDE.md`'s design axiom that predictions carry `model version`, and makes the D5
candidate-vs-deployed comparison auditable.

**Alternatives considered.** (a) Overwrite in place like the peer (`train.py:77`) — rejected: no traceability,
no rollback. (b) Version by file mtime/path only — rejected: not carried onto the prediction, so online
predictions can't be attributed to a model. (c) Git SHA of training code only — rejected: doesn't capture the
data snapshot; pair the version string with the training-data window instead.

*(Every ADR above cites ≥1 project principle: D1 Rule 9 + `sentiment.py`; D2/D4 `testing.md §4.1` +
`error-handling.md`; D3 `architecture.md §2`; D4/D5 `ADR-005` + the "always benchmark" axiom; D6 the
"predictions carry evidence" axiom + `embeddings/config.py`.)*

## Recommendations

1. **Redactor (D1/D2).** Implement `talkex/governance/redactor.py` with a `Redactor` port and a `PtBrRedactor`
   adapter covering CPF/CNPJ/RG/phone/email/name; return `{redacted_text, pii_span_manifest}` (no raw PII).
   Preserve `turn_id`/`conversation_id`. Mirror the peer's redact-vs-preserve split (`pii_redaction.py:13-20,
   43-59`), not its `en` config.
2. **Export use case (D3/D4).** Implement an `ExportRetrainingSample` use case depending on `SamplePort`
   (Parquet adapter via pyarrow) and a `Redactor`. Source rows from `turns` (aging window) joined to `labels`
   (`0003_m5_search.sql:9-10`). Gate the retention purge on export success — export first, `drop_chunks` only
   after (`ADR-005:53-56`; `0002:34`).
3. **Retrain script (D5/D6).** Add `experiments/scripts/retrain_sentiment.py` (extends
   `train_sentiment.py:54`): load {labels + exported samples} → `SentimentDetector.train` → benchmark
   candidate-vs-deployed → write metrics JSON → promote only on strict gain → `joblib.dump` with a
   `model_version`. Copy the peer's `random_state`/`main()` reproducibility (`train.py:63,112-113`), keep
   TalkEx's JSON+DoD discipline (`train_sentiment.py:67-81`).
4. **Tests (Corner 1 + `testing.md §4.1`).** (a) Integration: seed `turns`+`labels`, run export, read Parquet
   back, assert row count (mirror `spec:74-85`). (b) **Negative-case:** assert exported text has **no**
   CPF/phone/email — a typed PII-absent guarantee. (c) Ordering: a failed export leaves the chunk intact +
   raises a typed error (D4). (d) Retrain gate: a candidate that doesn't beat deployed is **not** promoted (EC-3).
5. **Dependency hygiene.** Pin exact versions (the peer leaves Presidio/sklearn/xgboost unpinned —
   `requirements.txt:8-9,23-24`); reuse the joblib idiom already in the repo rather than adding artifact tooling.

## Honest gaps

- **Q4 load-back:** `churn_model/train.py` only `dump`s artifacts; there is **no `load()`** in the file
  (`train.py:13,77-94`) — the reload path lives elsewhere in ai-powered (out of scope). M7 must author its own
  reload, following the joblib idiom already in `train_sentiment.py:66`.
- **Q5 anonymization:** chatwoot's export is **not anonymized** — its spec asserts raw email/phone *survive*
  (`spec:83-84`). It gives the artifact-download-and-assert + N+1-guard test shape, but the PII-absent
  assertion is a TalkEx-only synthesis (`testing.md §4.1`).
- **Q6 cleanup/lifecycle:** chatwoot's job has **no deletion/retention hook** (`contacts_export_job.rb` — none);
  the export↔purge ordering (D4) is synthesized from `ADR-005`, not observed in a peer.
- **Q3 PT-BR coverage:** no cloned peer implements a Brazilian PII detector; D1's `PtBrRedactor` is
  greenfield (pattern-lifted, code-original). Free-text PT name recall is a known limitation to be measured.
- **EC-3 (retrain gains):** whether M5's label volume beats the M2 baseline is **unknowable until measured**;
  D5's candidate-vs-deployed gate is designed to surface this honestly rather than assume a gain.
- No peer showed **model versioning**; D6 is synthesized from TalkEx's own `embeddings/config.py:37` precedent.

## discover-confidence verdict

**SHIPPABLE_WITH_CAVEATS.** All 7 questions are answered from real cited files (no BLOCKED, no fabricated
paths), all 4 coverage corners are populated with evidence, and the two mandatory synthesis checkpoints (EC-1
PT-BR PII entity set + build-vs-adopt call in D1; EC-2 export-before-purge ordering in D4) are promoted to
ADRs. The caveats are the honest gaps above — chiefly that the PII-anonymization and export↔purge-ordering
patterns are **synthesized** (neither peer demonstrates them) and that the PT-BR redactor is greenfield — which
is inherent to M7's PT-BR/LGPD scope, not a research deficiency.
