# TalkEx — Conversation Intelligence Engine

A scalable **NLP platform for large-scale conversation analysis** designed for call centers, customer support operations, and digital service channels.

Transforms raw conversations into **structured, searchable, and actionable insights** using a hybrid architecture combining lexical search, semantic embeddings, supervised classification, and deterministic rule engines.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run pipeline on a transcript
talkex run transcript.txt --channel voice --format labeled

# Run with DSL rules
talkex run transcript.txt --rule 'keyword("billing")' --rule 'keyword("cancel")'

# Run benchmark comparing configurations
talkex benchmark transcript.txt --output benchmark_output/

# Export config template
talkex config --export config.json

# Show version
talkex version
```

## Architecture

Multi-stage NLP pipeline with cascaded inference (cheap filters first, expensive models only when needed):

```
TranscriptInput
    ↓
Stage 1: Turn Segmentation + Normalization
    ↓
Stage 2: Context Window Builder (sliding window of N turns)
    ↓
Stage 3: Embedding Generation (turn, window, conversation)
    ↓
Stage 4: Index Building (Vector Index + Lexical Index)
    ↓
Stage 5: Classification (multi-label, multi-level)
    ↓
Stage 6: Semantic Rule Engine (DSL → AST → evaluation with evidence)
    ↓
Stage 7: Analytics Event Collection
    ↓
SystemPipelineResult (with PipelineRunManifest)
```

Each stage is optional — the pipeline degrades gracefully when components are not provided. Per-stage timing, warnings, and artifact lineage are built in.

## Pipeline Stages

| Stage | Module | Purpose |
|-------|--------|---------|
| Segmentation | `segmentation/` | Parse raw text into attributed turns (speaker, text, offsets) |
| Context | `context/` | Build sliding windows of N adjacent turns |
| Embeddings | `embeddings/` | Generate versioned dense vectors (model + version + pooling) |
| Retrieval | `retrieval/` | Hybrid search: BM25 + ANN + score fusion + optional reranking |
| Classification | `classification/` | Multi-label supervised classification with evidence |
| Rules | `rules/` | DSL compiled to AST; lexical + semantic + structural + contextual predicates |
| Analytics | `analytics/` | Event collection, aggregation, and reporting |

## CLI Commands

```bash
# Run pipeline on a transcript file
talkex run <file> [--channel voice|chat|email] [--format labeled|plain]
                [--config config.json] [--output output/]
                [--rule 'keyword("term")'] [--no-embeddings] [--no-rules]
                [--conversation-id conv_123]

# Run benchmark comparing pipeline configurations
talkex benchmark <file> [--config config.json] [--output output/]

# Export or validate configuration
talkex config [--export template.json] [--validate config.json]

# Show version
talkex version
```

### Output Structure

Each pipeline run creates a structured output directory:

```
output/
└── run_a1b2c3d4e5f6/
    ├── manifest.json    # Execution identity, component versions, config fingerprint
    └── summary.json     # Run statistics (turns, windows, embeddings, timing)
```

## Configuration

Pipeline configuration uses JSON files with Pydantic validation:

```json
{
  "segmentation": {
    "normalize_unicode": true,
    "min_turn_chars": 1,
    "speaker_label_pattern": "^(CUSTOMER|AGENT|SYSTEM|UNKNOWN)\\s*:"
  },
  "context": {
    "window_size": 5,
    "stride": 2,
    "include_partial_tail": true
  },
  "embedding": {
    "model": {
      "model_name": "intfloat/e5-base-v2",
      "model_version": "1.0"
    },
    "dimensions": 384
  },
  "rules": {
    "evaluation_mode": "all",
    "evidence_policy": "always"
  },
  "output_dir": "output"
}
```

## Transcript Format

**Labeled format** (default):
```
CUSTOMER: I have a billing issue with my credit card.
AGENT: I can help you with that. What is the issue?
CUSTOMER: I was charged twice for the same order.
AGENT: Let me look into that for you right away.
```

**Plain format** (no speaker labels):
```
I have a billing issue with my credit card.
I can help you with that. What is the issue?
```

## Rule Engine DSL

Rules combine four signal families into auditable, traceable evaluations:

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

## Programmatic Usage

```python
from talkex.pipeline.config import PipelineConfig
from talkex.pipeline.runner import PipelineRunner

# Configure and run
config = PipelineConfig.from_json("config.json")
runner = PipelineRunner(config=config)

summary = runner.run_file(
    "transcript.txt",
    channel="voice",
    rules_text=['keyword("billing")'],
)

print(f"Run ID: {summary.run_id}")
print(f"Turns: {summary.turns_count}")
print(f"Windows: {summary.windows_count}")

# Persist outputs
PipelineRunner.save_outputs(summary, "output/")
```

### Direct Pipeline Usage

```python
from talkex.pipeline.system_pipeline import SystemPipeline
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.segmentation.segmenter import TurnSegmenter
from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig

pipeline = SystemPipeline(
    text_pipeline=TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    ),
)

result = pipeline.run(
    transcript_input,
    context_config=ContextWindowConfig(window_size=3, stride=2),
)

print(f"Manifest: {result.manifest.run_id}")
print(f"Turns: {len(result.pipeline_result.turns)}")
print(f"Windows: {len(result.pipeline_result.windows)}")
```

## Benchmarking

Compare pipeline configurations with the built-in benchmark runner:

```python
from talkex.pipeline.benchmark import SystemBenchmarkRunner

runner = SystemBenchmarkRunner()
report = runner.compare({
    "full": lambda: full_pipeline.run(transcript, ...),
    "text_only": lambda: text_pipeline.run(transcript, ...),
})

report.save_json("benchmark.json")
report.save_csv("benchmark.csv")
```

Reports include per-stage latency, skip rates, artifact counts, and aggregated metrics.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Quality gates (run ALL before submitting)
ruff format --check .
ruff check .
mypy src/ tests/
pytest tests/unit/ -x
pytest tests/ -x            # includes integration tests

# Run a single test
pytest tests/unit/test_foo.py -x -v -k "test_name"
```

### Package Structure

```
src/talkex/
├── models/           # Shared pydantic data types (frozen, strict)
├── ingestion/        # Data ingestion from multiple sources
├── segmentation/     # Turn segmentation and normalization
├── context/          # Context window builder (sliding window)
├── embeddings/       # Multi-level embedding generation
├── retrieval/        # Hybrid search: BM25 + ANN + score fusion
├── classification/   # Multi-label, multi-level classification
├── rules/            # Semantic rule engine: DSL → AST → executor
├── analytics/        # Event collection, aggregation, reporting
├── pipeline/         # System orchestration, benchmark, CLI
└── exceptions.py     # Domain exception hierarchy
```

### Key Design Principles

- **Embeddings represent, classifiers decide** — never treat cosine similarity as classification
- **Always benchmark against BM25 baseline** before investing in semantic approaches
- **Every prediction carries evidence** — label, score, confidence, threshold, model version
- **Cascaded inference** — cheap filters first, expensive models only when needed
- **LLMs offline only** — lightweight models for online inference

## License

MIT License
