# Discover Edge Case Review — M2 Sentiment & Features

Date: 2026-07-30
Discovery plan analyzed: knowledge-base/discoveries/plans/m2-sentiment-features-plan.md
Research questions analyzed: 6
Edge cases found: 3 (MUST FIX: 0, SHOULD TEST: 1, DOCUMENT: 2)

## MUST FIX

_None._ All cited `knowledge-base/references/` paths verified (7/7 via `ls`).

## SHOULD TEST

### EC-1: BERTimbau reference deps are an old stack (pytorch-transformers 1.1.0 / Py3.6)
- **Affected question:** Q1, Q5
- **Suggested halt-loop checkpoint:** when reading `portuguese-bert/ner_evaluation/requirements.txt`, capture the version but flag it explicitly as legacy; the blueprint MUST map the load pattern to modern `transformers` + the already-installed `sentence-transformers`/`torch`, NOT copy `pytorch-transformers==1.1.0`.

## DOCUMENT

### EC-2: ai-powered sentiment model is English DistilBERT, unintegrated
- **Accepted risk:** Q2 reads `sentiment_analysis.py` for the *pipeline shape* (chunking, label mapping, CPU-local), NOT the model (English, wrong language) nor the wiring (it is standalone, not in main.py). The blueprint copies the shape and rejects the model/integration gaps. No action.

### EC-3: no peer does PT-BR sentiment classification with macro-F1
- **Accepted risk:** already captured by plan ADR D3. The tests corner (Q4) is honestly "eval-by-F1, no unit suites"; the macro-F1 harness is built internally with `sklearn.metrics.f1_score(average='macro')` over the internal labeled corpus. Not a coverage hole — an explicit internal-core deferral.

## Summary

| Question | Edges found | MUST FIX | SHOULD TEST | DOCUMENT |
|----------|-------------|----------|-------------|----------|
| Q1/Q5 | 1 | 0 | 1 | 0 |
| Q2 | 1 | 0 | 0 | 1 |
| Q4 | 1 | 0 | 0 | 1 |

**Verdict:** DISCOVERY PLAN OK (0 MUST FIX; EC-1 checkpoint absorbed; EC-2/EC-3 documented as accepted risks)
