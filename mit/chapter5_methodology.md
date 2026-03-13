# Chapter 5: Experimental Design and Methodology

## Abstract of Chapter

This chapter specifies the experimental protocol used to evaluate the four hypotheses of the
thesis. We describe the dataset (post-audit), the evaluation metrics for each hypothesis, the
experimental configurations compared, the statistical analysis framework, the ablation study
design, and the threats to validity. The level of detail is intended to enable full
reproduction by independent researchers. All experimental decisions — metrics, baselines,
confirmation criteria, statistical tests — were defined *a priori*, before any results were
observed, to prevent post-hoc interpretation bias.

---

## 5.1 Dataset

### 5.1.1 Data Sources

Empirical evaluation requires a corpus of labeled customer service conversations in Brazilian
Portuguese with intent annotations and structural metadata. Public conversational datasets in
PT-BR are scarce — a recognized gap in Portuguese-language NLP (Souza et al., 2020). We
address this limitation through controlled synthetic expansion of an existing public corpus.

**Base corpus.** We use the `RichardSakaguchiMS/brazilian-customer-service-conversations`
dataset, hosted on HuggingFace under the Apache 2.0 license. The original corpus contains
approximately 944 conversations annotated with intent labels and sentiment polarities. While
valuable as a starting point, the original dataset exhibits four limitations that compromise
its suitability for rigorous experimentation: (i) an approximately uniform class distribution
(~11% per class), distant from the naturally imbalanced distributions observed in production
contact centers; (ii) low variability in conversation length, with approximately 90% of
conversations containing exactly 8 turns; (iii) homogeneous lexical style without persona or
register variation; and (iv) sentiment distributions decoupled from intent, with uniform 33%
proportions for each polarity regardless of class.

**Controlled synthetic expansion.** To overcome these limitations, we performed a synthetic
expansion using a large language model (Claude Sonnet, Anthropic) in offline batch mode. The
expansion was conducted with rigorous controls over the following generation variables:

- *Turn variability:* generated conversations follow a log-normal distribution with mean 8
  turns and standard deviation 4, producing conversations with 4 to 20 turns — reflecting the
  natural variability observed in contact center operations.
- *Class distribution:* we adopted an intentionally imbalanced distribution that reproduces
  typical contact center patterns, where complaints and inquiries predominate over praise and
  greetings.
- *Lexical variability:* each generated conversation is conditioned on one of five linguistic
  personas — formal, informal, angry, elderly, and young/slang — introducing register and
  vocabulary diversity.
- *Intent-conditioned sentiment:* sentiment distributions are conditioned on the intent class.
  For example, complaint conversations exhibit 65% negative and 35% neutral sentiment, while
  praise conversations exhibit 75% positive and 25% neutral sentiment.

It is essential to acknowledge that the use of synthetic data imposes an epistemological
limitation: conclusions derived from this corpus are valid within the defined methodological
scope and cannot be unconditionally generalized to real conversational data. We mitigate this
limitation through the robustness validation protocol described in Section 5.1.4 and the
data audit described in Section 5.1.3.

### 5.1.2 Post-Audit Corpus Specification

The raw expanded corpus contained 2,257 conversations across 9 intent classes. Prior to any
experimentation, we conducted a comprehensive data audit (Phase 1) to identify and remove
problematic records. The audit process and its outcomes are described in Section 5.1.3. After
the audit, the corpus was reduced to **2,122 conversations** across **8 intent classes**. Table
5.1 summarizes the post-audit corpus specification.

**Table 5.1** — Post-audit corpus specification.

| Dimension | Value |
|---|---|
| Total conversations | 2,122 |
| Original conversations | 847 |
| Synthetic conversations | 1,275 |
| Turns per conversation | 4–20 (log-normal, $\mu=8$, $\sigma=4$) |
| Intent classes | 8 |
| Language | PT-BR (informal, with diacritics and slang) |
| Annotation per conversation | Intent + sentiment + sector |
| Random seeds | 5 seeds: [13, 42, 123, 2024, 999] |

The 8 intent classes retained after the audit are: `cancelamento` (cancellation),
`compra` (purchase), `duvida_produto` (product inquiry), `duvida_servico` (service inquiry),
`elogio` (praise), `reclamacao` (complaint), `saudacao` (greeting), and `suporte_tecnico`
(technical support). Table 5.2 presents the class distribution.

**Table 5.2** — Intent class distribution in the post-audit corpus.

| Intent | Count | Proportion |
|---|---|---|
| reclamacao | ~424 | ~20.0% |
| duvida_produto | ~382 | ~18.0% |
| duvida_servico | ~361 | ~17.0% |
| suporte_tecnico | ~318 | ~15.0% |
| compra | ~212 | ~10.0% |
| cancelamento | ~170 | ~8.0% |
| saudacao | ~149 | ~7.0% |
| elogio | ~106 | ~5.0% |

The imbalanced distribution is methodologically relevant for two objectives: (i) evaluating
classifier behavior on minority classes, a frequent scenario in production applications; and
(ii) testing the effectiveness of deterministic rules (H3) on critical low-frequency classes
such as `cancelamento` and `elogio`.

### 5.1.3 Data Audit (Phase 1)

Before executing any experiment, we conducted a rigorous data audit that removed 135 records
from the original 2,257-conversation corpus. The audit addressed three categories of data
quality issues:

1. **Duplicate removal.** Exact and near-duplicate conversations were identified and removed
   using text fingerprinting (MinHash) and conversation-ID analysis. Duplicates between the
   original and synthetic corpora were prioritized for removal to preserve corpus diversity.

2. **Contamination removal.** The synthetic expansion used few-shot exemplars drawn from the
   original corpus. Any record identified as a few-shot exemplar that leaked into the
   evaluation splits was removed to prevent train-test contamination. Contamination was
   detected by matching `conversation_id` prefixes (`conv_synth_*`) and `source_file`
   metadata against the generation logs.

3. **Taxonomy consolidation.** The `outros` (other) class — present in the original 9-class
   taxonomy — was eliminated after human review confirmed that its instances were either
   ambiguous, mislabeled, or representable by the remaining 8 classes. The human audit
   achieved a confirmation rate of $\geq 96.7\%$ for retained labels, establishing the
   post-audit corpus as the authoritative ground truth for all subsequent experiments.

The audit was completed on 2026-03-12 with human review approval. All experiments reported
in this thesis use exclusively the post-audit corpus of 2,122 records.

### 5.1.4 Preprocessing

The preprocessing pipeline follows five sequential stages, implemented in the TalkEx
framework (Chapter 4):

1. **Text normalization.** We apply the `normalize_for_matching()` function from the
   `talkex.text_normalization` module, which performs lowercase conversion and diacritic
   removal via Unicode NFD decomposition. This normalization is essential for Brazilian
   Portuguese, where variations such as "não"/"nao" and "cancelamento"/"cancelámento" must
   be treated as equivalent for lexical matching.

2. **Turn segmentation.** The dataset contains pre-structured turns with speaker attribution.
   When necessary, speaker alternation heuristics reconstruct the segmentation using the
   `TurnSegmenter` from `talkex.segmentation`.

3. **Context window construction.** Sliding windows with configurable size, stride, and
   speaker alignment are generated using the `SlidingWindowBuilder` from `talkex.context`.
   The default configuration uses 5-turn windows with stride 2.

4. **Embedding generation.** Dense vector representations are generated at multiple
   granularity levels (turn, window, conversation) using the
   `paraphrase-multilingual-MiniLM-L12-v2` encoder (384 dimensions). Embeddings are cached
   for reproducibility and computational efficiency.

5. **Indexing.** Parallel indices are constructed for lexical retrieval (BM25, via
   `rank-bm25`) and semantic retrieval (ANN, via FAISS with flat index).

### 5.1.5 Splits, Seeds, and Reproducibility

We adopt **five random seeds** [13, 42, 123, 2024, 999] for all experiments. For each seed,
a stratified train/validation/test split (70%/15%/15%) is generated, preserving the class
distribution in each partition. All results are reported as mean $\pm$ standard deviation
across the five seeds, with statistical significance assessed via Wilcoxon signed-rank tests
on the per-seed paired observations.

The decision to use five seeds rather than a single fixed seed (as in the original
experimental design) was motivated by the need to estimate variance due to random
partitioning and to enable non-parametric paired statistical tests. The five-seed protocol
provides five paired observations per comparison, which is the minimum required for a
two-sided Wilcoxon signed-rank test at $\alpha = 0.05$.

**Window-level split integrity.** In experiments H2–H4 and the ablation study, the
classification unit is the **context window** (5 turns, stride 2), not the complete
conversation. Overlapping windows from the same conversation could introduce data leakage if
windows from the same conversation appeared in both training and test sets. To prevent this,
partitioning is performed **at the conversation level** before window generation. Each
conversation belongs entirely to a single split; windows are generated only after
partitioning. No test window shares turns with any training window.

**Label inheritance and weak supervision.** Intent labels are assigned at the conversation
level. When generating sliding windows, each window **inherits** the label of its source
conversation. This constitutes a form of weak supervision: intermediate windows (e.g.,
resolution or clarification turns) may not contain direct lexical or semantic evidence of the
annotated intent. This is an acknowledged limitation, shared with prior work that transfers
document-level labels to segments (Rayo et al., 2024).

**Window-to-conversation aggregation.** Classifier training operates at the **window level**:
each window is a training example with its own features. Evaluation, however, operates at
the **conversation level**: class probabilities predicted for each window are averaged (mean
class probabilities), and the final class is determined by argmax over the averaged
distribution. This strategy prevents any single window with an extreme prediction from
dominating the result — a conservative choice aligned with the noisy nature of weak
supervision via label inheritance.

---

## 5.2 Evaluation Metrics

Evaluation requires distinct metrics for each pipeline component. We organize metrics into
four families corresponding to the dimensions evaluated by hypotheses H1–H4.

### 5.2.1 Retrieval Metrics (H1)

For hybrid retrieval evaluation, we adopt standard information retrieval metrics (Manning
et al., 2008):

**Table 5.3** — Retrieval evaluation metrics.

| Metric | Definition | Justification |
|---|---|---|
| Recall@K | Fraction of relevant documents retrieved in the top-K | Measures coverage for downstream pipeline stages |
| Precision@K | Fraction of top-K documents that are relevant | Measures result set quality |
| MRR | Mean Reciprocal Rank — average of $1/r_i$ where $r_i$ is the rank of the first relevant result | Measures how quickly the system returns the first useful result |
| nDCG@K | Normalized Discounted Cumulative Gain | Measures ranking quality considering relative positions |

All metrics are evaluated for $K \in \{5, 10, 20\}$. MRR is the primary metric for H1, as
it captures the user-facing quality of the retrieval system in operational scenarios where
the first relevant result is most important.

### 5.2.2 Classification Metrics (H2, H3)

**Table 5.4** — Classification evaluation metrics.

| Metric | Definition | Justification |
|---|---|---|
| Macro-F1 | Arithmetic mean of per-class F1 scores, with equal weighting | Sensitive to minority class performance; primary metric |
| Micro-F1 | F1 computed over all instances, weighted by volume | Reflects overall performance considering class imbalance |
| Per-class Precision | Proportion of correct positive predictions per class | Measures false positive cost per class |
| Per-class Recall | Proportion of positive instances correctly identified per class | Measures coverage per class |

The choice of Macro-F1 as the primary metric is deliberate: in an imbalanced corpus, Micro-F1
can mask poor performance on minority classes. Macro-F1 assigns equal weight to each class,
ensuring that classifiers must perform well across the full taxonomy — including rare intents
such as `elogio` (5%) and `cancelamento` (8%) — to achieve high scores.

### 5.2.3 Rule Metrics (H3)

**Table 5.5** — Rule evaluation metrics.

| Metric | Definition |
|---|---|
| Rule precision | Proportion of correct activations over total activations |
| Rule recall | Proportion of true positives captured by the rule |
| Rule F1 | Harmonic mean of rule precision and recall |
| Coverage | Percentage of conversations where at least one rule produced evidence |

### 5.2.4 Efficiency Metrics (H4)

**Table 5.6** — Efficiency evaluation metrics.

| Metric | Definition |
|---|---|
| Cost per window | Processing time in milliseconds per classification unit |
| $\Delta$F1 | Macro-F1 difference between uniform and cascaded pipelines |
| % resolved per stage | Proportion of windows resolved at each cascade stage |

The combination of cost per window and $\Delta$F1 enables construction of the Pareto frontier
that is central to H4 evaluation: identifying configurations that reduce cost with acceptable
quality degradation.

---

## 5.3 Experimental Protocol for H1 — Hybrid Retrieval

### 5.3.1 Hypothesis

> *Hybrid retrieval (BM25 + ANN with score fusion) surpasses both standalone BM25 and
> standalone semantic search in MRR when applied to PT-BR customer service conversations.*

This hypothesis rests on the theoretical complementarity between lexical and semantic search,
widely documented in the literature (Lin et al., 2021; Formal et al., 2021; Rayo et al.,
2025). BM25 excels at exact term matching — product names, codes, regulatory keywords —
while dense retrieval captures paraphrases, implicit intent, and linguistic variation. The
hypothesis posits that the combination outperforms both isolated approaches in the
conversational domain.

### 5.3.2 Systems Compared

We define retrieval systems spanning three categories:

**Table 5.7** — Retrieval systems compared in H1.

| System | Category | Description |
|---|---|---|
| BM25-base | Lexical | Vanilla BM25 with lowercase and stopword removal |
| ANN | Semantic | Approximate nearest neighbor search with paraphrase-multilingual-MiniLM-L12-v2 embeddings (384 dims) |
| Hybrid-LINEAR | Hybrid | BM25 + ANN, linear score fusion: $S = \alpha \cdot s_{bm25} + (1-\alpha) \cdot s_{ann}$ |
| Hybrid-RRF | Hybrid | BM25 + ANN, Reciprocal Rank Fusion (Cormack et al., 2009) |

### 5.3.3 Parameter Space

For the hybrid systems, we sweep the fusion weight $\alpha$ (BM25 weight) across the range
[0.05, 0.95] in increments of 0.05. This comprehensive sweep identifies the optimal
trade-off between lexical and semantic contributions. Based on prior work (Rayo et al., 2025),
we expect the optimal $\alpha$ to fall in the range [0.20, 0.40], weighting lexical
contribution moderately while allowing semantic search to dominate.

### 5.3.4 Ground Truth Construction

Retrieval evaluation requires queries with annotated relevant documents. We construct the
ground truth as follows: queries are derived from the intent taxonomy (e.g., "cancelamento",
"reclamação"), and relevance is defined by matching the query intent against the conversation-
level intent annotation of each indexed window. This enables automatic ground truth
construction without additional manual annotation.

### 5.3.5 Confirmation Criteria

H1 is **confirmed** if the best hybrid system surpasses all isolated systems in MRR with a
statistically significant difference ($p < 0.05$ on the Wilcoxon signed-rank test across 5
seeds).

H1 is **partially confirmed** if the hybrid system surpasses in some metrics but not others,
or if the difference does not reach statistical significance.

H1 is **refuted** if an isolated system (BM25 or ANN) matches or surpasses the best hybrid
system on the primary metric (MRR).

---

## 5.4 Experimental Protocol for H2 — Combined Feature Representations

### 5.4.1 Hypothesis

> *Classifiers combining lexical features with dense pretrained embeddings achieve superior
> Macro-F1 compared to classifiers using only lexical features, consistently across multiple
> classifier families.*

This hypothesis rests on the complementarity between sparse lexical representations (TF-IDF,
BM25 scores) and dense pretrained representations (sentence embeddings). Lexical features
capture exact term matches and frequency patterns; dense embeddings capture semantic
relationships, paraphrases, and linguistic variation. The hypothesis posits that the
combination surpasses lexical-only features regardless of the classifier architecture.

### 5.4.2 Feature Configurations

**Table 5.8** — Feature configurations compared in H2.

| Configuration | Features |
|---|---|
| lexical-only | TF-IDF vectors + BM25 scores against class prototypes |
| embedding-only | 384-dim frozen embeddings (paraphrase-multilingual-MiniLM-L12-v2) |
| lexical+embedding | Concatenation of lexical and embedding features |
| lexical+embedding+structural | Full feature set: lexical + embedding + structural conversation features (speaker ratio, turn count, position) |

### 5.4.3 Classifiers

For each feature configuration, we train and evaluate three classifiers:

**Table 5.9** — Classifiers used in H2.

| Classifier | Configuration | Justification |
|---|---|---|
| Logistic Regression | Default scikit-learn parameters | Linear baseline; evaluates feature separability |
| LightGBM | 100 estimators, 31 leaves | Gradient boosting optimized for heterogeneous features |
| MLP | 2 hidden layers (scikit-learn) | Neural baseline for dense feature interactions |

The LightGBM configuration (`n_estimators=100, num_leaves=31`) was fixed a priori based on
standard defaults and is used uniformly across all H2, H3, and ablation experiments. No
per-experiment hyperparameter tuning was performed on the test set.

### 5.4.4 Context Window Parameters

The context window is a central component of the multi-level representation. The default
configuration uses 5-turn windows with stride 2, based on the empirical observation that
most conversational intents manifest within 3–7 turns. Window-level embeddings are generated
by mean pooling the turn-level embeddings within each window.

### 5.4.5 Confirmation Criteria

H2 is **confirmed** if lexical+embedding configurations consistently surpass lexical-only
configurations in Macro-F1, with a statistically significant difference ($p < 0.05$), across
at least two of the three classifier families.

H2 is **refuted** if lexical-only configurations match or surpass lexical+embedding
configurations in the majority of classifiers.

---

## 5.5 Experimental Protocol for H3 — Deterministic Rules

### 5.5.1 Hypothesis

> *Adding a semantic rule engine (DSL → AST) to the hybrid ML pipeline improves Macro-F1
> while providing per-decision evidence traceability.*

This hypothesis is motivated by the observation that statistical classifiers, while effective
in the general case, exhibit limitations in scenarios where (i) the cost of false positives
is asymmetric, (ii) compliance requirements demand auditable decisions, and (iii) domain-
specific linguistic patterns are known a priori. Rule-based systems, despite limited
coverage, offer controllable precision and full traceability — properties complementary to
statistical models.

### 5.5.2 Rule Set

The H3 experiment uses two lexical rules implemented in the TalkEx rule engine:

1. **rule_cancel** — detects explicit cancellation intent via lexical patterns (keywords:
   "cancelar", "encerrar", "desistir", "rescindir" and variants). Targets the `cancelamento`
   class.

2. **rule_complaint** — detects complaint patterns via lexical patterns (keywords:
   "reclamação", "absurdo", "desrespeito", "procon" and variants). Targets the `reclamacao`
   class.

These rules were defined before any evaluation on the test set to prevent construction bias.
The rule set is deliberately minimal (2 lexical rules) to establish a lower bound on rule
contribution. The TalkEx rule engine supports additional predicate families — semantic
(`intent_score`, `embedding_similarity`), structural, and contextual (`repeated_in_window`,
`occurs_after`) — that were not exercised in H3. Expanding the rule set with semantic
predicates is identified as a primary direction for future work.

### 5.5.3 Integration Strategies Compared

**Table 5.10** — Rule integration strategies compared in H3.

| Configuration | Description |
|---|---|
| ML-only | Best LightGBM classifier from H2, without rules |
| Rules-only | Only rule activations, no ML classifier |
| ML+Rules-feature | ML classifier with binary rule activation flags as additional features |
| ML+Rules-override | ML classifier with rule decisions overriding ML predictions when rules activate |

The rules-as-features strategy treats rule activations as additional binary features in the
LightGBM feature space, allowing the classifier to learn the informativeness of each rule.
The rules-as-override strategy applies rule decisions directly when a rule fires, bypassing
the ML prediction. This comparison isolates whether rules are better used as input signals
or as decision overrides.

### 5.5.4 Confirmation Criteria

H3 is **confirmed** if any ML+Rules configuration surpasses ML-only in Macro-F1 with
statistical significance ($p < 0.05$).

H3 is **inconclusive** if ML+Rules shows a positive trend but does not reach statistical
significance at $\alpha = 0.05$.

H3 is **refuted** if rules degrade Macro-F1 across all integration strategies.

---

## 5.6 Experimental Protocol for H4 — Cascaded Inference

### 5.6.1 Hypothesis

> *A cascaded inference pipeline reduces the average computational cost per window compared
> to the uniform pipeline, with acceptable quality degradation ($\Delta$F1 < 2 pp).*

This hypothesis rests on the principle of cascaded inference (Viola & Jones, 2001; Matveeva
et al., 2006): applying increasingly expensive processing stages, resolving easy cases early
with cheap models and reserving expensive models for ambiguous cases.

### 5.6.2 Pipeline Configurations

**Uniform pipeline (baseline).** All windows pass through all processing stages regardless
of complexity:

$$\text{Window} \to \text{Normalization} \to \text{Embeddings} \to \text{BM25 + ANN} \to \text{Fusion} \to \text{Classification} \to \text{Rules} \to \text{Output}$$

**Cascaded pipeline.** A two-stage cascade where windows classified by a lightweight Stage 1
model (lexical-only LightGBM) with confidence above threshold $\theta$ are not escalated to
Stage 2 (full pipeline with embedding + lexical features):

- **Stage 1** — Lexical-only LightGBM classifier. If $\max(P(y|x)) \geq \theta$, the window
  is resolved. Cost: lexical feature extraction only.
- **Stage 2** — Full LightGBM classifier with lexical + embedding features. Applied only to
  windows not resolved in Stage 1.

### 5.6.3 Threshold Space

Confidence thresholds are varied across: $\theta \in \{0.50, 0.55, 0.60, ..., 0.90\}$, a
grid of 9 values that explores the full trade-off between early resolution rate and
classification quality.

### 5.6.4 Confirmation Criteria

H4 is **confirmed** if at least one threshold configuration achieves cost reduction $\geq 20\%$
with $\Delta$F1 $< 2$ percentage points.

H4 is **refuted** if no configuration achieves meaningful cost reduction without unacceptable
quality degradation, or if the cost model reveals structural impediments to cascade benefit.

---

## 5.7 Ablation Study Design

The ablation study quantifies the marginal contribution of each feature family to the full
pipeline's classification performance. Starting from the full configuration (lexical +
embedding + structural + rule activation features with LightGBM), we systematically remove
one feature family at a time and measure the resulting change in Macro-F1.

**Table 5.11** — Ablation configurations.

| Configuration | Features Removed | Features Retained |
|---|---|---|
| full_pipeline | None | Lexical + Embedding + Structural + Rules |
| no_embeddings | Embedding features | Lexical + Structural + Rules |
| no_lexical | Lexical features | Embedding + Structural + Rules |
| no_rules | Rule activation features | Lexical + Embedding + Structural |
| no_structural | Structural features | Lexical + Embedding + Rules |

For each configuration, the LightGBM classifier (100 estimators, 31 leaves) is retrained and
evaluated using the same 5-seed protocol as the main experiments. The ablation is additive:
each configuration removes exactly one family, enabling direct attribution of marginal
contribution. The expected ranking of contributions, based on theoretical considerations, is:
embeddings >> lexical > rules ≈ structural.

---

## 5.8 Statistical Analysis Framework

### 5.8.1 Significance Testing

All pairwise comparisons use the **Wilcoxon signed-rank test**, a non-parametric test for
paired samples that does not assume normality (Wilcoxon, 1945). With 5 paired observations
(one per seed), the minimum achievable p-value is $2/2^5 = 0.0625$ for a two-sided test,
which sets an inherent power limitation. When all 5 paired differences have the same sign,
$p = 0.0625$; statistical significance at $\alpha = 0.05$ requires that the observed
differences be consistently in the same direction across seeds AND sufficiently large in
magnitude for the signed-rank statistic to exceed the critical value.

**Correction note.** We acknowledge that the five-seed Wilcoxon test has limited statistical
power. When the test yields $p > 0.05$ with a consistent directional trend, we interpret
this as inconclusive rather than as evidence against the alternative hypothesis — following
the recommendation of Demšar (2006) for ML experiments with small sample sizes.

### 5.8.2 Confidence Intervals

For the primary metric of each hypothesis, we report **bootstrap 95% confidence intervals**
computed with 10,000 resamples over the per-seed paired differences. The bootstrap CI
provides a non-parametric estimate of the plausible range of the true effect size. When the
CI excludes zero, it provides additional evidence beyond the p-value alone.

### 5.8.3 Effect Size

For statistically significant comparisons, we report the effect size $r$ computed as
$r = Z / \sqrt{N}$, where $Z$ is the standardized Wilcoxon statistic and $N$ is the number
of paired observations. Effect sizes are classified following Cohen's conventions: small
($r \approx 0.1$), medium ($r \approx 0.3$), large ($r \approx 0.5$).

---

## 5.9 Threats to Validity

We explicitly document threats to validity organized in the three classical categories of
Cook and Campbell (1979): internal, external, and construct validity.

### 5.9.1 Internal Validity

**Table 5.12** — Threats to internal validity and mitigations.

| Threat | Description | Mitigation |
|---|---|---|
| Hyperparameter overfitting | Selection of hyperparameters that maximize test performance by chance | Fixed LightGBM configuration (100t/31l) used uniformly; no per-experiment tuning on test set |
| Rule construction bias | Rules defined with knowledge of the test set | Rules defined and finalized before any test evaluation; creation date logged |
| Metric selection bias | Post-hoc selection of metrics favorable to the proposed system | All metrics defined a priori in experimental design; metrics where the system loses are reported with equal prominence |
| Implementation bugs | Code errors that invalidate results | Comprehensive automated test suite (1,883+ unit tests in TalkEx); quality gates (ruff, mypy, pytest) enforced before every experiment |
| Random seed selection | Seeds chosen to favor certain outcomes | Seeds [13, 42, 123, 2024, 999] selected before any experiment; results reported for ALL seeds |
| Data leakage | Train-test contamination through overlapping windows | Conversation-level partitioning enforced before window generation; contamination audit in Phase 1 |
| Few-shot contamination | Synthetic generation exemplars leaking into test splits | Contamination detection via conversation_id prefixes and source_file metadata; contaminated records removed in audit |

### 5.9.2 External Validity

**Table 5.13** — Threats to external validity and mitigations.

| Threat | Description | Mitigation |
|---|---|---|
| Synthetic data | Corpus is partially synthetic; conclusions may not generalize to real conversations | Explicit disclosure; robustness analysis comparing original vs synthetic subsets; discussion of potential differences |
| Single language | Results are specific to PT-BR | Analysis of language-specific components (diacritic normalization ablation); identification of language-agnostic vs language-specific modules |
| Single domain | Customer service conversations have specific characteristics | Explicit discussion of domain assumptions; identification of portable vs domain-specific components |
| Scale | Experiments at research scale (2,122 conversations), not production scale (millions) | Theoretical computational complexity analysis; empirical throughput measurements; scalability discussion |
| Single dataset | All experiments use a single dataset source | Acknowledged as a primary limitation; LODO and k-fold cross-validation experiments planned as extensions |

### 5.9.3 Construct Validity

**Table 5.14** — Threats to construct validity and mitigations.

| Threat | Description | Mitigation |
|---|---|---|
| Metric-utility gap | F1 and MRR may not reflect perceived utility by human operators | Inclusion of qualitative analysis with concrete examples; discussion of metric-utility correspondence |
| Explainability not formally evaluated | Evidence traceability is claimed but not measured with formal explainability metrics | Qualitative criteria for evidence quality; complete output examples with metadata |
| Simplified cost model | Time-based cost (ms) does not capture GPU, memory, and infrastructure costs | Complementary discussion of memory and infrastructure costs where relevant |
| Weak supervision noise | Label inheritance from conversation to window introduces label noise | Acknowledged limitation; window-to-conversation aggregation mitigates the effect; per-class analysis identifies affected intents |
| Rule coverage | Only 2 lexical rules tested; does not represent the full rule engine capability | Acknowledged as the most significant limitation of H3; expanded ruleset with semantic predicates identified as future work |

---

## 5.10 Experimental Infrastructure

### 5.10.1 Software Stack

**Table 5.15** — Software stack.

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Sentence encoder | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| BM25 | rank-bm25 + TalkEx implementation (`talkex.retrieval`) |
| ANN | FAISS (faiss-cpu), flat index |
| Classification | scikit-learn (LogReg, MLP), LightGBM |
| Rule engine | TalkEx DSL → AST → executor (`talkex.rules`) |
| Evaluation | scikit-learn metrics, custom evaluation scripts |
| Statistical tests | scipy.stats (Wilcoxon), bootstrap (custom) |
| Visualization | matplotlib, seaborn |
| Reproducibility | Fixed seeds, `pyproject.toml` dependency pinning, 1,883+ automated tests |

### 5.10.2 Execution Environment

All experiments were executed on Google Colab with a Tesla T4 GPU runtime (15 GB VRAM).
Embedding generation leverages GPU acceleration via PyTorch/CUDA through the
sentence-transformers library, while LightGBM classification trains on CPU (~6 seconds).
The full experiment suite completes in under 1 hour. The use of Google Colab's free-tier
GPU runtime validates the thesis's claim of accessibility: the entire experimental pipeline
is reproducible without dedicated ML infrastructure, requiring only a web browser and a
Google account.

### 5.10.3 Code and Data Availability

The TalkEx source code is available as a Git repository, including the complete NLP pipeline,
rule engine (DSL, parser, AST, executor), and experimental scripts in the `experiments/`
directory. The dataset is referenced by its HuggingFace identifier
(`RichardSakaguchiMS/brazilian-customer-service-conversations`), and the synthetic expansion
procedure is documented for reproduction. The unified experiment runner
(`experiments/scripts/run_experiment.py`) executes all H1–H4 experiments and the ablation
study in a single invocation.

---

## 5.11 Summary

This chapter specified the complete experimental protocol for evaluating the four thesis
hypotheses. The design ensures: (i) fair comparison between systems, with appropriate
baselines and controlled variables; (ii) statistical rigor, with non-parametric significance
tests and bootstrap confidence intervals across five random seeds; (iii) full
reproducibility, with fixed seeds, documented software versions, and versioned scripts; and
(iv) transparency about limitations, with explicit documentation of threats to validity.

The four experiments — hybrid retrieval (H1), combined feature representations (H2),
deterministic rules (H3), and cascaded inference (H4) — share the same post-audit dataset
and infrastructure but are evaluated with hypothesis-specific metrics and protocols.
Confirmation criteria were defined a priori, before execution, to prevent post-hoc
interpretation bias. The ablation study complements the hypothesis testing by quantifying
the marginal contribution of each feature family to the full pipeline.

Chapter 6 presents the results obtained and their critical analysis.
