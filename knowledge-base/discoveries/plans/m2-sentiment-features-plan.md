# Discovery Plan: M2 Online Sentiment & Feature Extraction

> **Version 1.0** — Investigate how the reference projects load a PT-BR encoder (BERTimbau), run a
> lightweight CPU sentiment stage, evaluate with F1, and which PT-BR sentiment resources exist — so the
> M2 blueprint can add an online sentiment classifier (macro-F1 ≥ 0.70, beating a BM25/lexical baseline,
> CPU-first) reusing the talkex-internal labeled dataset + classifiers. Honest scope: the core M2 assets
> (a sentiment-labeled PT-BR corpus + the talkex classification module) are INTERNAL, not from the peers;
> the peers inform the model/baseline landscape + the F1-evaluation method (ADR D3).

**Slug:** `m2-sentiment-features`
**Owner:** paulohenriquevn
**Created:** 2026-07-30
**Time budget:** 2h (three peers — see ADR D1)

## Context

M2 adds online sentiment as a cascade feature (`ROADMAP.md § M2`): a lightweight CPU-first PT-BR sentiment
classifier, per turn/window, feeding the wide feature table, benchmarked against the BM25/lexical baseline.
The eval data ALREADY exists internally — `experiments/data/{train,val,test}.jsonl` carry a 3-class
`sentiment` label (train 1250 / val 404 / test 468, positive/negative/neutral) — and the talkex
`src/talkex/classification/` module (logistic/lightgbm/mlp/similarity) can be reused over embedding features.
This discovery mines the peers for the *model landscape* (BERTimbau encoder, sentiment pipeline shape),
the *baselines* (pysentimiento, PT lexicons), and the *F1-evaluation* method, and explicitly defers the
core (internal data + classifiers) to talkex itself (ADR D3). Constrained by `.claude/rules/architecture.md`
(DIP) and `.claude/rules/testing.md`, and the axiom "embeddings represent, classifiers decide" + "always
benchmark against BM25".

## Objective

Enable the M2 blueprint to decide, with evidence, **the sentiment model/approach (encoder+classifier vs
lexicon baseline), the CPU inference shape, and the macro-F1 evaluation method** — reusing the internal
labeled data + talkex classifiers.

- [ ] All research questions answered with citations to `knowledge-base/references/`
- [ ] Cross-cutting comparison populated
- [ ] At least one concrete decision proposal per question
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS

## In-Scope / Out-of-Scope

### In-Scope

| Project | In-scope subdirectories | Reason |
|---|---|---|
| `knowledge-base/references/portuguese-bert/` | `README.md`, `ner_evaluation/` | BERTimbau load pattern + a fine-tuning/F1 example |
| `knowledge-base/references/ai-powered-call-center-intelligence/` | `backend/sentiment_analysis.py`, `requirements.txt` | a lightweight CPU sentiment-pipeline shape (anti-pattern: English, unintegrated) |
| `knowledge-base/references/portuguese-nlp/` | `README.md` | PT-BR sentiment datasets/models/lexicons (baseline landscape) |

### Out-of-Scope (explicit)

| Project / Subdir | Why excluded |
|---|---|
| `knowledge-base/references/portuguese-bert/` — NER-specific data/configs | NER task, not sentiment (only the load + F1 pattern transfers) |
| `knowledge-base/references/ai-powered-call-center-intelligence/` — Whisper/GPT/frontend | ASR/LLM/UI, not sentiment |
| `knowledge-base/references/chatwoot/`, `livekit-agents/` | No sentiment ML |
| The internal labeled corpus + talkex classifiers | Core M2 assets — internal, not a peer citation (ADR D3) |

## ADRs

### D1 — Time budget + stop conditions
**Decision:** portuguese-bert 1h · ai-powered 0.5h · portuguese-nlp 0.5h.
**Rationale:** BERTimbau is the richest model reference; the others are shallow (a pipeline shape, a resource list).
**Stop condition — per question:** after 3 empty retries, mark BLOCKED "Fase A exhausted"; never fabricate (Rule 3).
**Stop condition — per project:** on budget exhaustion, mark remaining BLOCKED; emit `<promise>BLUEPRINT_BLOCKED</promise>` if any remain.

### D2 — Investigation depth
**Decision:** Read the sentiment/load/eval files in full; Grep the resource lists + deps.
**Consequences:** deeper read on `sentiment_analysis.py`, `README` load block, `run_bert_harem.py` F1 lines.

### D3 — Core M2 assets are internal (honest deferral)
**Decision:** the sentiment-labeled corpus (`experiments/data/*.jsonl`) and the talkex classifiers
(`src/talkex/classification/`) are the M2 core; the peers inform only the model/baseline landscape + the
F1-eval method.
**Rationale:** honesty (Rule 3) — no peer does PT-BR sentiment classification with macro-F1; fabricating a
peer citation for that would fail the fabricated-citation cap.
**Consequences:** the blueprint synthesizes an internal-first design; peers are landscape + method evidence.

## Research Questions

| # | Question | Corner | Reference project(s) | Fase A (broad map) | Fase B (deep Read) | Expected answer shape |
|---|---|---|---|---|---|---|
| Q1 | How is BERTimbau loaded for downstream use (cased PT-BR tokenization)? | techniques | `knowledge-base/references/portuguese-bert/` | Grep `from_pretrained`/`BertTokenizer` in `README.md` | Read `README.md:44-63` (AutoModel/AutoTokenizer load block) | Load recipe → informs the M2 encoder option |
| Q2 | What is a lightweight CPU sentiment-pipeline shape, and its pitfalls? | techniques | `knowledge-base/references/ai-powered-call-center-intelligence/` | Read `backend/sentiment_analysis.py` | Read `sentiment_analysis.py:1-38` (transformers pipeline, chunking, labels) — note English + unintegrated pitfalls | Pipeline shape + anti-patterns to design against |
| Q3 | How does a BERTimbau downstream task evaluate with F1 (transposable to macro-F1)? | techniques | `knowledge-base/references/portuguese-bert/` | Grep `f1_score`/`classification_report` in `ner_evaluation/run_bert_harem.py` | Read `run_bert_harem.py:97-115` (seqeval F1) — adapt to `sklearn f1_score(average='macro')` for sentiment | Eval-method recipe → informs M2 macro-F1 harness |
| Q4 | Do the peers ship a test/eval suite for the sentiment/model layer? | tests | all three | Glob `*test*`/`conftest*` across the peers | Confirm presence/absence; capture the F1-metric eval as the closest analog | Honest test-corner answer (peers: eval-by-F1, no unit suites) |
| Q5 | What model/inference dependencies do the peers pin (transformers/torch)? | deps | `portuguese-bert/`, `ai-powered-call-center-intelligence/` | Grep in `portuguese-bert/ner_evaluation/requirements.txt` + `ai-powered-call-center-intelligence/requirements.txt` | Read each; extract version + role | Dep table → informs the M2 model dep choice (modern `transformers` vs old `pytorch-transformers`) |
| Q6 | What is the model setup/run story, and which PT-BR sentiment datasets/models/lexicons exist? | tools | `portuguese-bert/`, `portuguese-nlp/` | Read `portuguese-bert/ner_evaluation/README.md:16-36` (setup/run); Grep sentiment sections in `portuguese-nlp/README.md` | Read the setup block + `portuguese-nlp/README.md:26,:204,:289,:176` (datasets, FinBERT-PT-BR, pysentimiento, lexicons) | Setup recipe + baseline landscape (lexicon/pysentimiento to beat) |

## Coverage Matrix

| Corner | Questions mapped | Status |
|---|---|---|
| Integration tests | Q4 | Covered |
| Dependencies | Q5 | Covered |
| Tools | Q6 | Covered |
| Techniques | Q1, Q2, Q3 | Covered |

**Coverage: 4/4 corners covered (100%)**

## Halt-loop Checkpoints

| Checkpoint | Assertion | Action if fails |
|---|---|---|
| Before answering Qx | Every cited `knowledge-base/references/{path}` exists | Mark Qx BLOCKED "path not found", continue |
| Per-question Fase A budget | ≥1 hotspot OR 3 retries | Mark BLOCKED "Fase A exhausted"; continue |
| Core-asset question | If a question drifts into the internal corpus/classifiers with no peer source | STOP — that is ADR D3 territory (talkex-internal), not a peer citation |
| Before promising complete | All 4 corners populated | Refuse promise, continue |

## Acceptance Criteria

- [ ] All research questions answered OR explicitly BLOCKED with reason
- [ ] All four corners populated in the blueprint
- [ ] Every citation resolves to a real `knowledge-base/references/{...}` path
- [ ] ≥1 ADR synthesizes M2 decisions; the internal-core deferral (D3) is explicit
- [ ] `/discover-confidence` verdict ≥ SHIPPABLE_WITH_CAVEATS
- [ ] Blueprint saved at `knowledge-base/discoveries/blueprints/m2-sentiment-features-blueprint.md`

## Global Definition of Done

- [ ] All phases completed (plan → edge-cases → execute → confidence → improve if needed)
- [ ] Final `/discover-confidence` verdict recorded in the blueprint header
- [ ] No fabricated citations (esp. no fake peer citation for PT-BR sentiment F1 — ADR D3)
- [ ] Coverage Matrix 100%
- [ ] ADRs cite `.claude/rules/architecture.md`, `.claude/rules/testing.md`, and the axiom "always benchmark against BM25"
