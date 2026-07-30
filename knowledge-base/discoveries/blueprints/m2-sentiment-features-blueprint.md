# Blueprint: M2 Online Sentiment & Feature Extraction

> **Version 1.0** — Locks the M2 sentiment design: a CPU-first PT-BR sentiment classifier reusing the
> talkex classification module over embedding features, trained on the internal sentiment-labeled corpus,
> evaluated by macro-F1 (≥ 0.70) against a lexical baseline, wired into the online cascade. The peers
> supply the model/baseline landscape + the F1-evaluation method; the core (labeled data + classifiers) is
> internal (ADR D5). Produced by `cycle-discover` execute from `m2-sentiment-features-plan.md`.

**Slug:** `m2-sentiment-features`
**Created:** 2026-07-30
**discover-confidence verdict:** recorded at the end after scoring.

## Executive summary

M2 adds online sentiment as a cascade feature. The pragmatic, CPU-first, evidence-backed design reuses
the talkex `LogisticClassifier`/`MLPClassifier` over the existing multilingual embedding, trained on the
internal 3-class sentiment corpus (train 1250 / val 404 / test 468), evaluated by `sklearn` macro-F1
against a lexical baseline — no heavy model download, no online LLM. BERTimbau (a PT-BR encoder) is the
richer option flagged for a pilot; `ai-powered`'s pipeline is a shape reference (its English DistilBERT +
lack of integration are anti-patterns). No peer does PT-BR sentiment with macro-F1 — that method is built
internally (honest gap, ADR D5).

## Context

`ROADMAP.md § M2` wants online sentiment (macro-F1 ≥ 0.70, beating a BM25/lexical baseline, CPU-first),
feeding the wide feature table. Eval data + classifiers already exist internally; this blueprint locks how
to combine them, informed by the peers, under the axiom "embeddings represent, classifiers decide" +
"always benchmark against BM25" and `.claude/rules/architecture.md`/`testing.md`.

## Objective

Lock, with cited evidence, the M2 sentiment approach (embedding+classifier vs lexicon baseline), the CPU
inference shape, the macro-F1 evaluation method, and the cascade integration — so M2 implementation trains a
sentiment classifier reaching macro-F1 ≥ 0.70 over the internal test set, beating a lexical baseline, without
rework.

## Coverage Corner 1 — Integration Tests

The peers ship NO unit/integration test suite for the sentiment/model layer (`find *test*` empty in
`portuguese-bert/` and `ai-powered-call-center-intelligence/`). The closest analog is F1-based evaluation:
`knowledge-base/references/portuguese-bert/ner_evaluation/run_bert_harem.py:97` uses `seqeval`
`f1_score`/`classification_report`. **M2 test decision:** build a deterministic macro-F1 harness with
`sklearn.metrics.f1_score(average='macro')` over the internal held-out test set (a reproducible, seeded
evaluation — a real test asserting `macro_f1 >= 0.70` and `macro_f1 > baseline`), plus unit tests for the
feature/label mapping. Honest: this is eval-by-metric, not a peer-copied suite.

## Coverage Corner 2 — Dependencies

`knowledge-base/references/portuguese-bert/ner_evaluation/requirements.txt:1` pins a legacy stack
(`pytorch-transformers==1.1.0`, `scikit-learn==0.21.2`, Py3.6) — rejected as outdated (EC-1).
`knowledge-base/references/ai-powered-call-center-intelligence/requirements.txt:13` lists `transformers`
(unpinned) for its sentiment stage. **M2 deps decision:** reuse what talkex already installs —
`sentence-transformers`/`torch` (the multilingual embedding) + `scikit-learn` (classifier + macro-F1); no
new model dep for the primary path. BERTimbau via modern `transformers` is the pilot option, not M2 core.

## Coverage Corner 3 — Tools

`knowledge-base/references/portuguese-bert/ner_evaluation/README.md:16` documents the setup/run story
(conda, `pip install -r requirements.txt`, `run_inference.py`). `knowledge-base/references/portuguese-nlp/README.md`
catalogs the PT-BR sentiment landscape: labeled datasets `:26` (ptbr-sentiment-analysis-datasets), `:124`
(tweets), `:118` (SST-2 pt); models `:204` (FinBERT-PT-BR); tools `:289` (pysentimiento); lexicons `:176`
(OpLexicon), `:182` (SentiLex-PT). **M2 tools decision:** the baseline to beat is a PT lexicon
(OpLexicon/SentiLex-PT) or a majority classifier — ultra-light, no model; the primary classifier trains from
`experiments/scripts/` reusing the existing experiment tooling.

## Coverage Corner 4 — Techniques

**Encoder load (pilot option).** `knowledge-base/references/portuguese-bert/README.md:44` shows the BERTimbau
load: `AutoModel/AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')`, cased PT-BR
(`do_lower_case=False`). **M2 decision:** the primary path reuses the already-loaded multilingual embedding
(no BERTimbau download); BERTimbau is the fine-tune pilot.

**Sentiment pipeline shape.** `knowledge-base/references/ai-powered-call-center-intelligence/backend/sentiment_analysis.py:1`
uses `transformers.pipeline("sentiment-analysis")`, chunks text (max ~120 words), returns `{text,label,score}`
— "lightweight, fast, local". **M2 decision:** copy the shape (chunk long windows, return label+score),
reject the model (English) and the wiring gap (it is unintegrated — M2 wires sentiment INTO the cascade).

**F1 evaluation.** `knowledge-base/references/portuguese-bert/ner_evaluation/run_bert_harem.py:97`
(`f1_score`/`classification_report`). **M2 decision:** adapt to `sklearn f1_score(average='macro')` for 3-class
sentiment on the internal test set.

**M2 techniques decision (synthesis):** embedding features → talkex `LogisticClassifier` (3-class sentiment)
trained on the internal corpus; chunk long windows (ai-powered shape); macro-F1 eval vs a PT-lexicon baseline.

## Cross-cutting Comparison

| Concern | portuguese-bert | ai-powered | portuguese-nlp | M2 decision |
|---|---|---|---|---|
| Model | BERTimbau encoder (fine-tune) | English DistilBERT pipeline (reject) | FinBERT-PT-BR, pysentimiento | reuse multilingual embedding + talkex classifier |
| Language | PT-BR (cased) | English (wrong) | PT-BR resources | PT-BR (internal corpus) |
| Eval | seqeval F1 (NER) | none | none (metrics = readability) | sklearn macro-F1 (sentiment) |
| Baseline | — | — | OpLexicon/SentiLex-PT lexicons | PT lexicon / majority (beat by ≥ 0.70) |
| Integration | standalone | standalone (unwired) | n/a | wired INTO the online cascade |
| CPU-first | heavy fine-tune | pipeline (local) | lexicon (ultra-light) | embedding+classifier (light) |

## ADRs

### D1 — Primary sentiment classifier: talkex classifier over embeddings
Reuse `src/talkex/classification/` (`LogisticClassifier`, fallback `MLPClassifier`) over the multilingual
embedding, trained on the internal labeled corpus. **Rationale:** "embeddings represent, classifiers decide";
CPU-fast; no model download; reuses proven code. **Alternative rejected:** BERTimbau fine-tune now (heavier,
slower to iterate — pilot). **Consequence:** fast online inference; BERTimbau is the accuracy escape hatch.

### D2 — Baseline: PT lexicon / majority (the number to beat)
A PT-BR lexicon (OpLexicon/SentiLex-PT, `portuguese-nlp/README.md:176`) or a majority classifier as the BM25/lexical
baseline. **Rationale:** the axiom "always benchmark against BM25/lexical". **Alternative rejected:** no baseline
(rejected — the DoD requires beating one). **Consequence:** macro-F1 gain is measured, not asserted.

### D3 — Evaluation: sklearn macro-F1 on the internal test set
`f1_score(average='macro')` on the 468-row test split; a seeded, reproducible harness asserting `>= 0.70` and
`> baseline`. **Rationale:** peers eval by F1 (`run_bert_harem.py:97`); sklearn is installed. **Alternative
rejected:** accuracy (rejected — imbalanced-insensitive; macro-F1 is the DoD). **Consequence:** the DoD is a
real passing test with a number.

### D4 — Cascade integration: sentiment as a wide feature (online)
Sentiment runs per window in the online cascade (cheap embedding + classifier), writing `sentiment` +
`sentiment_score` into the wide feature table (ADR-005 "Leitura A" columns). **Rationale:** cascade axiom
(cheap filters first); M2 feeds monitoring. **Alternative rejected:** offline-only sentiment (rejected — M2 is
the online feature). **Consequence:** the supervisor sees live sentiment; feeds M6 dashboards.

### D5 — Internal core, peers as landscape (honest gap)
The labeled corpus (`experiments/data/*.jsonl`) + classifiers are internal; no peer does PT-BR sentiment
macro-F1. **Rationale:** Rule 3 — no fabricated peer citations. **Consequence:** M2 synthesizes internally;
peers are landscape + method evidence.

## Recommendations

1. Add `experiments/scripts/train_sentiment.py`: embed the labeled corpus, train `LogisticClassifier` (3-class),
   eval macro-F1 on test vs a lexicon baseline; persist the model + a metrics JSON (D1-D3).
2. Add `src/talkex/classification/sentiment.py` (or reuse the orchestrator) exposing an online
   `predict_sentiment(text) -> (label, score)` over the embedding (D1/D4).
3. Wire sentiment into the monitoring cascade (per window) writing `sentiment`/`sentiment_score` features (D4).
4. Tests: a macro-F1 eval test (`assert macro_f1 >= 0.70 and macro_f1 > baseline`) + unit tests for the mapping.
5. Benchmark CPU inference latency per window (cascade budget).

## Honest gaps

1. **G1 — No peer PT-BR sentiment with macro-F1** — the classifier + eval are internal; peers are landscape/method.
2. **G2 — BERTimbau fine-tune deferred** — the primary path uses embedding+classifier; BERTimbau is a pilot if macro-F1 falls short.

## discover-confidence verdict

**SHIPPABLE — score 99.1, hard_caps_triggered: none** (via `run_blueprint_score.py`, 2026-07-30).
Coverage 4/4 corners, all citations resolve, 5 ADRs (D1–D5). Proceed to `cycle-plan` for M2.
