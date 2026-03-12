# TalkEx — Conversation Intelligence Engine

An NLP platform for large-scale conversation analysis in call centers and digital service channels. Transforms raw conversations into structured, searchable, and actionable insights using a **hybrid architecture** combining lexical search (BM25), semantic embeddings, supervised classification, and deterministic rule engines.

**TalkEx is the technical artifact of a master's dissertation** investigating whether a hybrid cascaded architecture outperforms isolated paradigms in conversation intent classification.

## Key Results

Evaluated on a corpus of **2,257 PT-BR customer service conversations** across **9 intent classes**:

| Hypothesis | Question | Result |
|---|---|---|
| **H1** — Hybrid Retrieval | Does BM25 + ANN fusion beat individual retrievers? | MRR 0.802 (BM25) vs 0.826 (Hybrid-RRF); p=0.103 — not statistically significant |
| **H2** — Lexical + Embeddings | Do dense embeddings improve classification over lexical-only features? | Macro-F1 0.715 (lexical+emb) vs 0.309 (lexical-only); significant in 9/9 classes |
| **H3** — Rules + ML | Do deterministic rules complement ML classifiers? | Macro-F1 0.714 (ML+Rules-feature) vs 0.709 (ML-only); cancelamento F1=1.000 |
| **H4** — Cascaded Inference | Does confidence-based routing reduce cost? | Cost ratio 1.5× insufficient; F1 degradation minimal (Δ=0.003 at t=0.90) |

## Architecture

Multi-stage NLP pipeline with cascaded inference (cheap filters first, expensive models only when needed):

```
TranscriptInput
    ↓
Stage 1: Turn Segmentation + Normalization
    ↓
Stage 2: Context Window Builder (sliding window of N turns)
    ↓
Stage 3: Embedding Generation (paraphrase-multilingual-MiniLM-L12-v2, 384 dims)
    ↓
Stage 4: Index Building (Vector Index + Lexical Index)
    ↓
Stage 5: Hybrid Retrieval (BM25 + ANN + score fusion)
    ↓
Stage 6: Classification (LightGBM + lexical + embedding features)
    ↓
Stage 7: Semantic Rule Engine (DSL → AST → evaluation with evidence)
    ↓
Stage 8: Analytics Event Collection
    ↓
SystemPipelineResult (with PipelineRunManifest)
```

## Quick Start

```bash
# Requires Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run pipeline on a transcript
talkex run transcript.txt --channel voice --format labeled

# Run with DSL rules
talkex run transcript.txt --rule 'keyword("billing")' --rule 'keyword("cancel")'

# Run benchmark comparing configurations
talkex benchmark transcript.txt --output benchmark_output/

# Show version
talkex version
```

### Running Experiments

```bash
# Activate venv with all dependencies
source .venv/bin/activate

# Run all hypotheses (H1-H4 + ablation)
python experiments/scripts/run_experiment.py

# Results are saved to experiments/results/{H1,H2,H3,H4,ablation}/
```

## Dataset

- **Source:** `RichardSakaguchiMS/brazilian-customer-service-conversations` (HuggingFace)
- **Size:** 2,257 conversations (1,179 original + 1,078 expanded via LLM generation)
- **Language:** Portuguese (PT-BR)
- **Classes (9):** cancelamento, compra, duvida_produto, duvida_servico, elogio, outros, reclamacao, saudacao, suporte_tecnico
- **License:** Apache 2.0

## Pipeline Stages

| Stage | Module | Purpose |
|---|---|---|
| Segmentation | `segmentation/` | Parse raw text into attributed turns (speaker, text, offsets) |
| Context | `context/` | Build sliding windows of N adjacent turns |
| Embeddings | `embeddings/` | Generate versioned dense vectors (model + version + pooling) |
| Retrieval | `retrieval/` | Hybrid search: BM25 + ANN + score fusion + optional reranking |
| Classification | `classification/` | LightGBM, LogReg, MLP classifiers with evidence |
| Classification Eval | `classification_eval/` | Evaluation metrics, reports, and dataset utilities |
| Rules | `rules/` | DSL compiled to AST; lexical + semantic + structural predicates |
| Analytics | `analytics/` | Event collection, aggregation, and reporting |
| Evaluation | `evaluation/` | Cross-hypothesis evaluation framework |

## Rule Engine DSL

Rules combine multiple signal families into auditable, traceable evaluations:

```python
from talkex.rules.compiler import SimpleRuleCompiler

compiler = SimpleRuleCompiler()

# Lexical predicates
rule = compiler.compile('keyword("billing")', "billing_rule", "billing_issue")

# Regex predicates
rule = compiler.compile('regex("cancel|terminate")', "cancel_rule", "cancel_intent")

# Combined with boolean logic
rule = compiler.compile(
    'keyword("billing") AND keyword("charge")',
    "billing_charge",
    "double_charge"
)
```

## Project Structure

```
src/talkex/
├── models/               # Shared pydantic data types (frozen, strict)
├── ingestion/            # Data ingestion from multiple sources
├── segmentation/         # Turn segmentation and normalization
├── context/              # Context window builder (sliding window)
├── embeddings/           # Multi-level embedding generation
├── retrieval/            # Hybrid search: BM25 + ANN + score fusion
├── classification/       # LightGBM, LogReg, MLP classifiers
├── classification_eval/  # Classification evaluation metrics and reports
├── rules/                # Semantic rule engine: DSL → AST → executor
├── evaluation/           # Cross-hypothesis evaluation framework
├── analytics/            # Event collection, aggregation, reporting
├── text_normalization.py # Text normalization utilities
├── pipeline/             # System orchestration, benchmark, CLI
└── exceptions.py         # Domain exception hierarchy
tests/
├── unit/                 # ~100 test files mirroring src/ structure
├── integration/          # Full pipeline and cross-module tests
├── conftest.py           # Shared fixtures
└── fixtures/             # Test data files
experiments/
├── scripts/              # run_experiment.py — unified H1-H4 + ablation runner
└── results/              # JSON results per hypothesis with statistical tests
docs/
├── adr/                  # Architecture Decision Records (ADR-001 through ADR-004)
├── dissertacao/          # Dissertation chapters (cap1-cap7)
└── pesquisas/            # Research papers and references
demo/
├── backend/              # FastAPI demo API
├── frontend/             # React demo frontend
└── data/                 # Demo data
```

## Development

```bash
# Install with dev dependencies (requires Python 3.11+)
pip install -e ".[dev]"

# Quality gates (run ALL before submitting)
ruff format --check .
ruff check .
mypy src/ tests/
pytest tests/unit/ -x
pytest tests/ -x            # includes integration tests

# Run a single test
pytest tests/unit/test_foo.py -x -v -k "test_name"

# Format and lint (auto-fix)
ruff format .
ruff check --fix .
```

## Key Design Principles

- **Embeddings represent, classifiers decide** — never treat cosine similarity as classification
- **Always benchmark against BM25 baseline** before investing in semantic approaches
- **Every prediction carries evidence** — label, score, confidence, threshold, model version
- **Cascaded inference** — cheap filters first, expensive models only when needed
- **LLMs offline only** — lightweight models for online inference

## Architecture Decisions

| ADR | Decision |
|---|---|
| ADR-001 | Package layout with `src/` and public API via `__init__.py` re-exports |
| ADR-002 | Frozen + strict Pydantic models; in-memory preserves types, boundaries use strict=False |
| ADR-003 | Embedding vectors as `list[float]` in models, `ndarray` at computation boundaries |
| ADR-004 | Context window structural fields design |

## License

MIT License
