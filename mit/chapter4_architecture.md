# Chapter 4 — Proposed Architecture: TalkEx

This chapter describes the architecture of TalkEx, the technical artifact at the core of this thesis. We present the complete conversation processing pipeline, the conversational data model, text normalization mechanisms, multi-level representations, the hybrid retrieval system, supervised classification, the semantic rule engine, and the cascaded inference strategy. The level of technical detail is intended to enable full reproduction of the system.

The description follows the natural data flow: from ingestion of a raw conversation to production of classified, auditable insights. Throughout the chapter, we reference architectural decisions formalized in Architecture Decision Records (ADRs), ensuring traceability between design principles and concrete implementation.

---

## 4.1 Pipeline Overview

### 4.1.1 Architectural Diagram

TalkEx implements a multi-stage conversational NLP pipeline designed to transform raw conversations into classified, auditable insights:

```
 +--------------+     +------------------+     +--------------------+
 |  Ingestion   |---->|  Turn            |---->|  Normalization &   |
 |              |     |  Segmentation    |     |  Preprocessing     |
 +--------------+     +------------------+     +--------------------+
                                                        |
                                                        v
                                               +--------------------+
                                               |  Context Window    |
                                               |  Builder           |
                                               +--------------------+
                                                        |
                                                        v
                                               +--------------------+
                                               |  Embedding         |
                                               |  Generation        |
                                               |  (multi-level)     |
                                               +--------------------+
                                                        |
                                            +-----------+-----------+
                                            |                       |
                                            v                       v
                                   +----------------+     +------------------+
                                   |  Lexical Index |     |  Vector Index    |
                                   |  (BM25)        |     |  (ANN/FAISS)     |
                                   +----------------+     +------------------+
                                            |                       |
                                            +----------+------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Hybrid Retrieval |
                                              |  (fusion + rerank)|
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Supervised       |
                                              |  Classification   |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Semantic Rule    |
                                              |  Engine           |
                                              |  (DSL → AST)     |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Analytics /      |
                                              |  APIs / Feedback  |
                                              +-------------------+
```

### 4.1.2 Design Principles

The architecture rests on three interdependent principles:

**Modularity.** Each pipeline stage is an independent module with well-defined interfaces. Modules communicate exclusively through shared data types (immutable Pydantic models), with no coupling to concrete implementations. This permits replacing any component — for instance, swapping the vector index from FAISS to Qdrant — without modifying upstream or downstream modules. The package structure reflects this modularity:

```
src/talkex/
  __init__.py           # Package root, exports __version__
  exceptions.py         # Domain exception hierarchy (EngineError base)
  text_normalization.py # Shared text normalization utilities
  models/               # Pydantic data types (frozen, strict)
  ingestion/            # Multi-source data ingestion
  segmentation/         # Turn segmentation and normalization
  context/              # Sliding context window builder
  embeddings/           # Multi-level embedding generation
  retrieval/            # Hybrid search: BM25 + ANN + score fusion
  classification/       # Supervised multi-class classification
  rules/                # Rule engine: DSL → AST → executor with evidence
  analytics/            # Analytical APIs and endpoints
```

This organization follows the `src/` layout convention (ADR-001), where all imports use the `talkex` namespace:

```python
from talkex.models import Conversation, Turn, ContextWindow
from talkex.retrieval import InMemoryBM25Index
from talkex.rules import SimpleRuleEvaluator
```

**Cascaded inference.** The pipeline applies progressively more expensive processing. Cheap lexical filters (O(1) cost) precede hybrid retrieval (O(log n) for ANN), which precedes supervised classification (O(d) for feature dimension d), which precedes semantic rules with embedding predicates. Section 4.8 details the inter-stage decision logic.

**Online/offline separation.** TalkEx distinguishes two operational modes. The online pipeline prioritizes low latency: classification, search, and immediate rules. The offline pipeline prioritizes quality and coverage: relabeling, clustering, intent discovery with LLMs, model retraining, and threshold recalibration. LLMs are used exclusively in the offline pipeline, never for online inference, ensuring predictable cost and controlled latency in production.

### 4.1.3 Data Flow

The transformation of a raw conversation proceeds in seven stages:

1. **Ingestion**: the conversation is received from a source (API, file, message queue) and validated against the `Conversation` schema.
2. **Segmentation**: raw text is segmented into turns (`Turn`), each attributed to a speaker (customer, agent, system).
3. **Normalization**: each turn receives text normalization (lowercasing, diacritics removal, punctuation) for consumption by lexical components.
4. **Context windows**: adjacent turns are grouped into sliding windows (`ContextWindow`) of size N with stride S.
5. **Embeddings**: dense vectors are generated at multiple granularity levels (turn, window, conversation, by speaker role).
6. **Indexing and retrieval**: texts are simultaneously indexed in the lexical index (BM25) and the vector index (ANN). Hybrid queries combine both via score fusion.
7. **Classification and rules**: heterogeneous features feed supervised classifiers. The semantic rule engine applies deterministic rules compiled to ASTs, producing decisions with traceable evidence.

At the end of the pipeline, each conversation is associated with classification predictions (`Prediction`) and rule evaluation results (`RuleExecution`), both carrying evidence metadata, model version, and execution time.

---

## 4.2 Conversational Data Model

The TalkEx data model defines six domain entities that flow through the entire pipeline. All use Pydantic v2 with `ConfigDict(frozen=True, strict=True)` (ADR-002), ensuring immutability and strict typing.

### 4.2.1 Entities and Relations

Table 4.1 presents the six core entities and their roles in the pipeline.

| Entity           | Description                                                | Granularity       |
|------------------|------------------------------------------------------------|-------------------|
| `Conversation`   | Complete interaction between customer and agent             | Entire conversation |
| `Turn`           | Individual utterance attributed to a speaker                | Turn              |
| `ContextWindow`  | Sliding window of N adjacent turns                          | Context window    |
| `EmbeddingRecord`| Versioned vector representation of a text object            | Multi-level       |
| `Prediction`     | Classification result with score, confidence, and threshold | Multi-level       |
| `RuleExecution`  | Rule evaluation result with traceable evidence              | Multi-level       |

The relations follow a composition hierarchy:

```
Conversation (1)
  |
  +--< Turn (N)                  # One conversation contains N turns
  |
  +--< ContextWindow (M)         # One conversation generates M windows
         |
         +--< turn_ids[]         # Each window references turns
         |
         +--< EmbeddingRecord    # Each window may have embeddings
         |
         +--< Prediction         # Each window may have predictions
         |
         +--< RuleExecution      # Each window may have rule evaluations
```

`EmbeddingRecord`, `Prediction`, and `RuleExecution` are polymorphic in granularity: each carries a `source_type` field indicating whether the source object is a turn, context window, or conversation. This allows embeddings, classifications, and rules to operate at any granularity level without multiplying data types.

### 4.2.2 Design Decisions

**Why frozen models (ADR-002).** Immutable models prevent accidental mutation of shared data as it flows through pipeline stages. When a stage needs to enrich an object (e.g., adding normalized text to a turn), it creates a new object rather than mutating the original. This guarantees that upstream stages always see consistent data and simplifies debugging in concurrent pipelines.

**Why `list[float]` for vectors (ADR-003).** Embedding vectors are stored as `list[float]` in Pydantic models for serialization compatibility, and converted to `numpy.ndarray` at computation boundaries (similarity calculation, FAISS indexing). This avoids Pydantic serialization issues with numpy types while preserving computational efficiency where it matters.

**Why strict mode with lossless widening.** Pydantic v2's strict mode rejects type coercion at boundaries (API, file deserialization), but permits lossless widening (int → float) within in-memory operations. This catches data quality issues at ingestion without being overly restrictive during computation.

---

## 4.3 Text Normalization

Text normalization is a critical preprocessing step for the PT-BR conversational domain. The `normalize_for_matching()` function provides accent-aware normalization essential for lexical components:

```python
def normalize_for_matching(text: str) -> str:
    """Lowercase + Unicode NFD diacritics removal for PT-BR matching."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text
```

**Why this matters for PT-BR.** Brazilian Portuguese features extensive diacritical variation: "não"/"nao", "número"/"numero", "cancelamento"/"cancelámento". In customer service transcriptions, diacritics are inconsistently applied — particularly in ASR outputs and informal chat. Without normalization, BM25 and lexical rule predicates would treat "cancelamento" and "cancelamento" (with acute accent) as different terms, reducing recall.

**Design choice: normalize for matching, preserve for display.** The original text (`raw_text`) is always preserved for auditability. Normalization produces a parallel field (`normalized_text`) used exclusively by lexical components (BM25 indexing, rule predicates, feature extraction). Semantic components (embedding generation) operate on the original text, as transformer models handle diacritics internally.

---

## 4.4 Turn Segmentation

The `TurnSegmenter` module transforms raw conversation text into a sequence of attributed `Turn` objects. The segmentation strategy depends on the data source:

- **Structured sources** (chat platforms, ticketing systems): turns are pre-segmented by the source system, requiring only validation and speaker attribution.
- **Unstructured sources** (voice transcriptions): the segmenter applies a speaker-alternation heuristic, splitting text at speaker change boundaries identified by ASR diarization markers.

Each turn receives a `turn_id`, `speaker` role (customer, agent, system, unknown), positional offsets, and optional normalized text.

---

## 4.5 Context Window Builder

### 4.5.1 Motivation

Individual turns capture local intent ("I want to cancel"), but miss multi-turn dependencies. A complaint that escalates across turns — starting as a question, progressing through dissatisfaction, and culminating in a cancellation request — can only be detected by examining adjacent turns jointly. The `SlidingWindowBuilder` constructs overlapping windows that capture these dependencies.

### 4.5.2 Configuration

The window builder is parameterized by:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 5 | Number of turns per window |
| `stride` | 2 | Step size between consecutive windows |
| `min_turns` | 3 | Minimum turns required to form a window |

**Why 5-turn windows with stride 2.** Five turns typically span 2–3 conversational exchanges (customer-agent pairs), capturing sufficient context for intent disambiguation without diluting signal. A stride of 2 provides overlap between consecutive windows, ensuring no conversational transition falls between windows. These parameters were selected based on analysis of the dataset's turn distribution (median 8 turns per conversation) and validated empirically during development — a 5-turn window generates 2–3 windows per conversation, providing multiple classification opportunities per interaction.

### 4.5.3 Structural Features

Each context window carries structural features extracted during construction:

- **Turn count** and **speaker distribution** (% customer turns, % agent turns)
- **Window position** (beginning, middle, end of conversation)
- **Text statistics** (total word count, mean words per turn)
- **Speaker transitions** (number of speaker changes within the window)

These structural features serve as additional classification signals (tested in the ablation study, Section 6.6) and as predicates for the rule engine.

### 4.5.4 Weak Supervision

Context windows inherit the intent label of their parent conversation. This weak supervision assumption introduces noise: intermediate windows (e.g., a greeting window in a cancellation conversation) may lack explicit signals of the conversation-level intent. We acknowledge this as a limitation (Section 7.3) and note that it affects both training and evaluation consistently, biasing per-class F1 downward for classes with gradual onset patterns (e.g., "compra", "saudacao").

---

## 4.6 Multi-Level Embedding Generation

### 4.6.1 Architecture

TalkEx generates dense vector representations at four granularity levels:

| Level | Source Text | Purpose |
|-------|------------|---------|
| **Turn** | Individual utterance | Local intent, keyword detection |
| **Window** | Concatenated turns in context window | Multi-turn dependencies, escalation patterns |
| **Conversation** | All turns concatenated | Global resolution, dominant tone |
| **Role-aware** | Customer-only or agent-only turns | Speaker-specific behavior analysis |

The primary embedding model is `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions), a Sentence Transformer trained for multilingual semantic similarity. This model was selected for three reasons:

1. **Multilingual support**: native PT-BR capability without language-specific fine-tuning.
2. **Efficiency**: 384 dimensions (vs. 768 for BERT-base or 1024 for E5-large) reduces storage and computation cost by 2–4×.
3. **Frozen deployment**: the encoder is used as-is, without fine-tuning — a deliberate architectural decision that trades potential domain-specific accuracy for accessibility and reproducibility.

**Why a frozen encoder.** Fine-tuning requires substantial GPU infrastructure, domain-specific training data, and ongoing model management. By using a frozen pre-trained encoder, TalkEx demonstrates that competitive classification is achievable without dedicated ML training infrastructure — all experiments were run on Google Colab's free-tier Tesla T4 GPU, which accelerates embedding inference but requires no fine-tuning budget. The empirical results (Chapter 6) show that frozen embeddings combined with LightGBM achieve Macro-F1 = 0.722, validating this pragmatic choice. A fine-tuned comparison remains as future work (Section 7.4).

### 4.6.2 Embedding Versioning

Each embedding record carries versioning metadata:

```python
@dataclass(frozen=True)
class EmbeddingRecord:
    source_id: str          # ID of the source object (turn, window, conversation)
    source_type: str        # "turn" | "window" | "conversation"
    model_name: str         # e.g., "paraphrase-multilingual-MiniLM-L12-v2"
    model_version: str      # Semantic version of the model
    pooling_strategy: str   # "mean" | "max" | "cls"
    vector: list[float]     # Dense vector (384 dimensions)
    generated_at: datetime  # Timestamp for provenance
```

This versioning ensures reproducibility: if the embedding model changes, downstream consumers (classifiers, indices) can detect the incompatibility rather than silently degrading.

---

## 4.7 Hybrid Retrieval

### 4.7.1 Dual Indexing

TalkEx maintains two parallel indices over the same corpus:

**Lexical index (BM25).** An in-memory BM25 index built with the `rank-bm25` library, operating on normalized text. BM25 scores are computed with standard parameters (k₁ = 1.5, b = 0.75). The index supports exact-term matching, product codes, and compliance keywords — domains where lexical precision is essential.

**Vector index (ANN).** A FAISS index over embedding vectors, using IVF (Inverted File Index) with product quantization for scalability. The index supports approximate nearest neighbor search with configurable recall-speed trade-offs.

### 4.7.2 Score Fusion

Given a query q, the hybrid retrieval pipeline:

1. Retrieves top-K₁ results from BM25 (lexical)
2. Retrieves top-K₂ results from ANN (semantic)
3. Computes the union of result sets
4. Applies score fusion to produce a unified ranking

Two fusion strategies are supported:

**Linear weighted fusion:**

```
Score(d) = α · sim_semantic(q, d) + (1 - α) · score_BM25(q, d)
```

where α ∈ [0, 1] controls the semantic weight. Both scores are min-max normalized to [0, 1] before fusion. The optimal α is selected on the validation set.

**Reciprocal Rank Fusion (RRF):**

```
RRF_score(d) = Σᵢ 1 / (k + rankᵢ(d))
```

where k = 60 (standard constant) and the sum is over all ranking systems that retrieved document d. RRF is rank-based rather than score-based, making it robust to score scale differences between BM25 and cosine similarity.

### 4.7.3 Optional Cross-Encoder Reranking

For high-stakes queries, the top-N results from score fusion can be reranked by a cross-encoder model that jointly encodes the query-document pair. This improves precision at the cost of O(N) forward passes through a transformer model. In the experimental evaluation (Chapter 6), reranking was not applied, as the focus was on score fusion effectiveness.

---

## 4.8 Supervised Classification

### 4.8.1 Feature Construction

The classification pipeline constructs heterogeneous feature vectors combining four signal families:

| Family | Features | Dimension | Source |
|--------|----------|-----------|--------|
| **Embedding** | Dense vector from window-level embedding | 384 | Sentence Transformer |
| **Lexical** | TF-IDF scores, term counts, n-gram patterns | ~11 | Text analysis |
| **Structural** | Turn count, speaker ratio, word count, position | ~4 | Context window metadata |
| **Rule** | Binary flags from rule engine evaluation | ~2 | Rule engine |

The total feature vector has approximately 397 dimensions (384 embedding + 11 lexical + 4 structural + 2 rule). Features are constructed as Python dictionaries (`list[dict[str, float]]`), converted to numpy arrays at the classifier boundary.

**Design principle: embeddings represent, classifiers decide.** Following AnthusAI's principle, embedding vectors are never used directly for classification via cosine similarity. Instead, they serve as input features to supervised classifiers that learn decision boundaries from labeled data. This separation ensures that the classifier can learn non-linear decision boundaries and incorporate heterogeneous signals that raw similarity cannot capture.

### 4.8.2 Classifier Models

Three classifier architectures are evaluated:

**Logistic Regression (baseline).** A simple linear model that serves as the mandatory baseline. Its linear decision boundary reveals how much of the classification can be achieved with simple feature combinations.

**LightGBM (primary).** A gradient boosting framework (100 estimators, 31 leaves) that handles heterogeneous features natively — a critical advantage when combining dense embeddings with sparse lexical indicators. LightGBM is deterministic with fixed splits and random state, enabling reproducible experiments.

**MLP (Multi-Layer Perceptron).** A two-layer neural network (hidden sizes: 256, 128) that captures non-linear feature interactions. MLP exhibits seed sensitivity due to stochastic weight initialization, making it the only classifier that shows non-zero standard deviation across seeds.

### 4.8.3 Prediction Output

Each classification produces a `Prediction` object carrying:

- **label**: predicted intent class
- **score**: probability for the predicted class
- **confidence**: calibrated confidence (when available)
- **threshold**: decision threshold applied
- **model_version**: classifier version for provenance
- **evidence**: feature importance or top contributing features

This output format ensures that every prediction is traceable and auditable, supporting the explainability requirements of regulated domains.

---

## 4.9 Semantic Rule Engine

### 4.9.1 Motivation

Supervised classifiers learn patterns from data but provide no formal guarantees and limited evidence traceability. In regulated domains — telecommunications (Anatel), financial services (Bacen), supplementary health (ANS) — decisions must be auditable, contestable, and explainable. The rule engine addresses this gap by providing deterministic, traceable decision logic that complements statistical classification.

### 4.9.2 Domain-Specific Language (DSL)

Rules are expressed in a human-readable DSL that is compiled to Abstract Syntax Trees (ASTs) for efficient evaluation:

```
RULE detect_cancellation
  WHEN speaker == "customer"
   AND contains_any(["cancelar", "encerrar", "desistir"])
  THEN tag("cancelamento", confidence=1.0)
```

The DSL supports four predicate families:

| Family | Predicates | Cost | Examples |
|--------|-----------|------|---------|
| **Lexical** | `contains`, `contains_any`, `regex_match`, `bm25_score` | O(n) | Exact keyword detection |
| **Semantic** | `intent_score`, `embedding_similarity` | O(d) | Soft semantic matching |
| **Structural** | `speaker`, `turn_index`, `channel`, `duration` | O(1) | Metadata conditions |
| **Contextual** | `repeated_in_window`, `occurs_after`, `count_in_window` | O(w) | Multi-turn patterns |

### 4.9.3 AST Compilation and Execution

The DSL parser compiles rules into AST nodes. The executor traverses the AST with two optimizations:

1. **Short-circuit evaluation**: if the first predicate in an AND conjunction fails, remaining predicates are not evaluated.
2. **Cost-ordered execution**: predicates within a conjunction are sorted by cost (structural O(1) → lexical O(n) → semantic O(d)), ensuring cheap filters eliminate candidates before expensive operations execute.

### 4.9.4 Evidence Trail

Every AST node produces traceable evidence upon evaluation:

```python
@dataclass(frozen=True)
class PredicateResult:
    predicate_type: str     # "lexical" | "semantic" | "structural" | "contextual"
    matched: bool           # Whether the predicate was satisfied
    score: float            # Numeric score (0.0-1.0)
    evidence: dict          # {"matched_words": [...], "positions": [...], ...}
    execution_time_ms: float
```

The complete evaluation of a rule produces a `RuleExecution` object with the full evidence chain: which predicates fired, what text matched, what scores were computed, and what thresholds were applied. This evidence trail is the key differentiator from black-box ML predictions.

### 4.9.5 Integration with Classification

Rules can be integrated with ML classification in three modes:

1. **Rules-as-override**: rule decisions override classifier predictions (post-processing).
2. **Rules-as-feature**: rule match flags are added as binary features to the classifier input.
3. **Rules-as-postprocessing**: rules adjust classifier confidence or add secondary labels.

The experimental evaluation (Chapter 6) compares these integration modes, finding that rules-as-feature is the only mode that avoids catastrophic degradation.

---

## 4.10 Cascaded Inference

### 4.10.1 Design

The cascaded inference pipeline applies progressively expensive processing, allowing early resolution when confidence is sufficient:

| Stage | Processing | Expected Cost | Resolution Criterion |
|-------|-----------|---------------|---------------------|
| 1 | Cheap lexical rules + metadata filters | ~1 ms | Confidence ≥ threshold |
| 2 | Hybrid retrieval + light classifier (LogReg) | ~10-50 ms | Confidence ≥ threshold |
| 3 | Full classification (LightGBM) + semantic rules | ~50-200 ms | Always resolves |
| 4 | Exception review (LLM, offline only) | ~500 ms-2s | Manual review |

The confidence threshold at each stage determines the cost-quality trade-off. Lower thresholds resolve more conversations at cheaper stages but risk misclassification; higher thresholds route more conversations to expensive stages but preserve quality.

### 4.10.2 Implementation Constraint

In the current experimental setup, both the light classifier (Stage 2) and full classifier (Stage 3) operate on pre-computed embeddings and context windows. This limits the cost differential between stages to the classifier inference time alone (LogReg ~20ms vs LightGBM ~30ms, a ratio of ~1.1×). As the experimental results demonstrate (Section 6.5), this insufficient cost differential prevents the cascade from achieving meaningful cost reduction. A production-grade cascade would require a genuinely cheap Stage 1 (e.g., lexical-only features without embedding computation) to realize the theoretical benefits.

---

## 4.11 Software Engineering

### 4.11.1 Scale and Quality

TalkEx comprises approximately 170 source files and 15,773 lines of production code, organized across 11 pipeline modules. The test suite includes 100 test files with 1,883 tests covering unit, integration, and pipeline-level validation. All code passes continuous quality gates: `ruff format`, `ruff check`, `mypy` type checking, and `pytest` execution.

### 4.11.2 Architecture Decision Records

Four ADRs document irreversible architectural decisions:

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-001 | `src/` layout with public API via `__init__.py` re-exports | Standard Python packaging, clear public/private boundary |
| ADR-002 | Frozen + strict Pydantic models; boundary deserialization with `strict=False` | Immutability guarantee + practical serialization |
| ADR-003 | Vectors as `list[float]` in models, `ndarray` at computation | Serialization compatibility + computational efficiency |
| ADR-004 | Context window structural fields design | Explicit feature extraction at window construction time |

---

## 4.12 Summary

The TalkEx architecture implements the three paradigms investigated in this thesis — lexical retrieval, semantic retrieval, and deterministic rules — within a single modular pipeline. The design prioritizes reproducibility (frozen encoder, versioned models, deterministic classifiers), auditability (rule evidence trails, prediction provenance), and accessibility (CPU-trainable classifiers, GPU-accelerated embedding inference via Google Colab free tier). Chapter 5 describes how this architecture is evaluated experimentally, and Chapter 6 presents the results.
