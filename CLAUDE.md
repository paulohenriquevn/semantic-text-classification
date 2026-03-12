# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TalkEx — Conversation Intelligence Engine — an NLP platform for large-scale conversation analysis in call centers and digital service channels. Transforms conversations into structured, searchable, actionable insights using hybrid NLP (lexical + semantic + rules).

**Current state:** Advanced alpha. 97 source files, 100 test files, complete experiment pipeline (H1-H4 + ablation), 7 dissertation chapters drafted, demo infrastructure (FastAPI + React). Package renamed from `semantic_conversation_engine` to `talkex` (RFC-005).

**Dissertation context:** TalkEx is the technical artifact of a master's dissertation. The thesis investigates whether a hybrid cascaded architecture (BM25 + semantic embeddings + deterministic rules) outperforms isolated paradigms in conversation intent classification. Dataset: 2,257 PT-BR customer service conversations, 9 intent classes.

## Commands

```bash
# Install (from project root, requires Python 3.11+)
source .venv/bin/activate
pip install -e ".[dev]"

# Quality gates (run ALL before completing any task)
ruff format --check .
ruff check .
mypy src/ tests/
pytest tests/unit/ -x
pytest tests/ -x              # includes integration tests

# Run a single test file
pytest tests/unit/test_foo.py -x -v

# Run a single test by name
pytest tests/unit/test_foo.py -x -v -k "test_name"

# Format and lint (auto-fix)
ruff format .
ruff check --fix .

# Run experiments (all hypotheses + ablation)
.venv/bin/python experiments/scripts/run_experiment.py
```

## Architecture

Multi-stage NLP pipeline with cascaded inference (cheap filters first, expensive models only when needed):

```
Ingestion → Turn Segmentation → Context Window Builder
  → Embedding Generation (paraphrase-multilingual-MiniLM-L12-v2, 384 dims)
  → [Vector Index + Lexical Index]
  → Hybrid Retrieval (BM25 + ANN + score fusion)
  → Classification (LightGBM 100t/31l + lexical + embedding features)
  → Semantic Rule Engine (DSL → AST → evaluation with evidence)
  → Analytics / APIs
```

### Package Layout

Package name: `talkex`. Imports: `from talkex.models import Conversation`.

```
src/talkex/
├── __init__.py              # Package root, exports __version__
├── exceptions.py            # Domain exception hierarchy (EngineError base)
├── text_normalization.py    # Text normalization utilities
├── models/                  # Shared pydantic data types (frozen, strict)
├── ingestion/               # Data ingestion from multiple sources
├── segmentation/            # Turn segmentation and normalization
├── context/                 # Context window builder (sliding window of N turns)
├── embeddings/              # Multi-level embedding generation (turn, window, conversation)
├── retrieval/               # Hybrid search: BM25 + ANN + score fusion + reranking
├── classification/          # LightGBM, LogReg, MLP classifiers
├── classification_eval/     # Classification evaluation metrics and reports
├── rules/                   # Semantic rule engine: DSL parser → AST → executor with evidence
├── evaluation/              # Cross-hypothesis evaluation framework
├── analytics/               # Event collection, aggregation, reporting
└── pipeline/                # System orchestration, benchmark, CLI
tests/
├── unit/                    # ~100 test files mirroring src/ structure
├── integration/             # Full pipeline and cross-module tests
├── conftest.py              # Shared fixtures (factory functions: make_conversation, make_turn, etc.)
└── fixtures/                # Test data files
experiments/
├── scripts/run_experiment.py  # Unified experiment runner (H1-H4 + ablation + statistical tests)
└── results/{H1,H2,H3,H4,ablation}/  # JSON results + statistical_tests.json
docs/
├── adr/                     # Architecture Decision Records (ADR-001 through ADR-004)
├── dissertacao/             # Dissertation chapters (cap1-cap7) + research-log.md
└── pesquisas/               # Research papers and references
demo/
├── backend/                 # FastAPI demo API
└── frontend/                # React demo frontend
```

### Core Domain Concepts

- **Conversation** — complete interaction (voice/chat/email) with metadata
- **Turn** — individual utterance attributed to a speaker; finest unit for embedding/classification
- **Context Window** — sliding window of N adjacent turns; primary unit for contextual analysis
- **Embedding** — versioned dense vector (model name + version + pooling strategy); represents, does NOT classify
- **Hybrid Retrieval** — BM25 top-K + ANN top-K → score fusion → optional cross-encoder rerank
- **Semantic Rule Engine** — DSL compiled to AST; combines lexical, semantic, structural, contextual predicates; produces traceable evidence

### Key Design Axioms

- **Embeddings represent, classifiers decide** — never treat cosine similarity as classification
- **Always benchmark against BM25 baseline** before investing in semantic approaches
- **LLMs offline only** — lightweight models for online inference, LLMs only for offline labeling/discovery
- **Every prediction carries evidence** — label, score, confidence, threshold, model version, text evidence

## Experiment Results (Current)

| Hypothesis | Best Config | Key Metric | Verdict |
|---|---|---|---|
| H1 — Hybrid Retrieval | Hybrid-RRF | MRR=0.826 vs BM25 0.802 | Refutada no critério primário (p=0.103) |
| H2 — Lexical + Embeddings | lexical+emb LightGBM | Macro-F1=0.715 vs 0.309 | Confirmada (9/9 classes significant) |
| H3 — Rules + ML | ML+Rules-feature | Macro-F1=0.714 | Confirmada (cancelamento F1=1.000) |
| H4 — Cascaded Inference | cascade t=0.90 | Macro-F1=0.707, cost ratio 1.5× | Refutada (cost reduction < 40%) |
| Ablation | full_pipeline | Macro-F1=0.714 | Embeddings: +35.0pp; Lexical: +1.5pp |

LightGBM unified config: `n_estimators=100, num_leaves=31`.

## Architecture Decisions

| ADR | Decision |
|---|---|
| ADR-001 | Package layout with `src/` and public API via `__init__.py` re-exports |
| ADR-002 | Frozen + strict Pydantic models; in-memory preserves types, boundaries use strict=False |
| ADR-003 | Embedding vectors as `list[float]` in models, `ndarray` at computation boundaries |
| ADR-004 | Context window structural fields design |

## Key Reference Docs

- `docs/KB.md` — foundational knowledge base (embeddings, search, classification theory)
- `docs/KB_Complementar.md` — architecture for millions of conversations, online/offline separation
- `docs/PRD.md` — product requirements document
- `docs/dissertacao/` — dissertation chapters, experimental design, research log
- `docs/pesquisas/` — research papers on lexical vs semantic search, text clustering, attention mechanisms

## Commit Convention

```
<type>(<scope>): <description>

Types: feat, fix, refactor, docs, test, chore
Scopes: ingestion, asr, segmentation, context, embeddings, retrieval, classification, rules, analytics, pipeline, api
```

## Git Rules

- **NEVER** `git checkout` or `git revert` (denied in settings.json)
- **NEVER** work directly on `main`
- Always create NEW commits; never amend unless explicitly asked
- Branch naming: `feat/`, `fix/`, `refactor/`, `docs/` prefixes
