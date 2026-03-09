# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Semantic Conversation Intelligence Engine — an NLP platform for large-scale conversation analysis in call centers and digital service channels. Transforms conversations into structured, searchable, actionable insights using hybrid NLP (lexical + semantic + rules).

**Current state:** Foundation phase. Project scaffolding complete with package structure, quality gates, and core model stubs. Building toward V1.

## Commands

```bash
# Install (from project root, inside venv)
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
```

## Architecture

Multi-stage NLP pipeline with cascaded inference (cheap filters first, expensive models only when needed):

```
Ingestion → ASR/Transcription → Turn Segmentation → Context Window Builder
  → Embedding Generation → [Vector Index + Lexical Index]
  → Hybrid Retrieval (BM25 + ANN + score fusion + optional reranking)
  → Classification (multi-label, multi-level)
  → Semantic Rule Engine (DSL → AST → evaluation with evidence)
  → Analytics / APIs
```

**Online pipeline:** low-latency classification, real-time search, immediate rules.
**Offline pipeline:** relabeling, clustering, intent discovery, LLM enrichment, retraining.

### Package Layout

Package name: `semantic_conversation_engine`. Imports: `from semantic_conversation_engine.models import Conversation`.

```
src/semantic_conversation_engine/
├── __init__.py       # Package root, exports __version__
├── exceptions.py     # Domain exception hierarchy (EngineError base)
├── models/           # Shared pydantic data types (frozen, strict)
├── ingestion/        # Data ingestion from multiple sources
├── segmentation/     # Turn segmentation and normalization
├── context/          # Context window builder (sliding window of N turns)
├── embeddings/       # Multi-level embedding generation (turn, window, conversation, role-aware)
├── retrieval/        # Hybrid search: BM25 + ANN + score fusion + reranking
├── classification/   # Multi-label, multi-level supervised classification
├── rules/            # Semantic rule engine: DSL parser → AST → executor with evidence
└── analytics/        # APIs and analytics endpoints (FastAPI)
tests/
├── unit/             # Mirrors src/ structure
├── integration/      # Full pipeline and cross-module tests
├── conftest.py       # Shared fixtures (factory functions: make_conversation, make_turn, etc.)
└── fixtures/         # Test data files
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

## Team Structure

Three roles coordinate via `.claude/agents/` and `.claude/teams/`:

| Role | Domain | Owns |
|------|--------|------|
| Technical Coordinator | Routing & verification | Task decomposition, final checks |
| NLP Engineer (kael-okonkwo) | Pipeline & Architecture | ingestion, segmentation, context, embeddings, retrieval, classification, rules |
| Data & Eval Engineer (tomas-herrera) | Quality & Governance | benchmarking, data quality, security, governance, evaluation |

## Key Reference Docs

- `docs/KB.md` — foundational knowledge base (embeddings, search, classification theory)
- `docs/KB_Complementar.md` — architecture for millions of conversations, online/offline separation
- `docs/PRD.md` — product requirements document
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
