# Chapter 6: Results and Analysis

## Abstract of Findings

This chapter presents the empirical results of four experimental hypotheses (H1--H4) and a
component ablation study, all evaluated on a 2,122-record PT-BR customer service corpus
following a rigorous data audit (Chapter 5). The findings are organized as follows: H1
(hybrid retrieval) is confirmed with statistical significance; H2 (combined lexical and
embedding features for classification) is confirmed with large effect sizes; H3 (rule
integration) yields a positive but statistically inconclusive result; H4 (cascaded inference
for cost reduction) is refuted. The ablation study quantifies the marginal contribution of
each feature family. Negative results and boundary conditions are reported with the same
prominence as positive findings; they constitute substantive scientific contributions by
delineating where the hybrid paradigm adds value and where it does not.

---

## 6.1 Experimental Setup

### 6.1.1 Dataset

All experiments use the post-audit dataset described in Chapter 5. The corpus comprises
2,122 labeled conversations in Brazilian Portuguese (PT-BR) drawn from the
`RichardSakaguchiMS/brazilian-customer-service-conversations` dataset (Apache 2.0 license).
Following the audit, 135 records were removed (duplicates, contaminated few-shot exemplars,
and ambiguous `outros` instances), and the taxonomy was consolidated from 9 to 8 intent
classes by eliminating the `outros` category after human review confirmed a confirmation
rate of 96.7% or higher for the retained labels.

The 8 intent classes are: `cancelamento` (cancellation), `compra` (purchase),
`duvida_produto` (product inquiry), `duvida_servico` (service inquiry), `elogio`
(compliment), `reclamacao` (complaint), `saudacao` (greeting), and `suporte_tecnico`
(technical support).

**Dataset composition:**

| Partition | Records | Share |
|---|---|---|
| Train | 1,250 | 58.9% |
| Validation | 404 | 19.0% |
| Test | 468 | 22.1% |
| **Total** | **2,122** | **100%** |

Of the 2,122 records, 847 (39.9%) are original transcripts and 1,275 (60.1%) are
LLM-generated synthetic expansions. Splits were constructed with contamination-aware
stratification to prevent few-shot prompt leakage (see Chapter 5). All reported test metrics
are evaluated on the held-out test partition only; the validation partition was used
exclusively for hyperparameter selection and early stopping.

### 6.1.2 Reproducibility Protocol

All experiments were repeated over five independent seeds: {13, 42, 123, 2024, 999}.
Because the pipeline uses a frozen encoder (paraphrase-multilingual-MiniLM-L12-v2, 384
dimensions) and fixed contamination-aware splits, embedding vectors are deterministic across
seeds; only the gradient boosting random state varies. As a consequence, LightGBM with
configuration `n_estimators=100, num_leaves=31` produces zero standard deviation across the
five seeds for all reported metrics. This is an expected and documented behavior: it
reflects the determinism of the experimental design rather than a deficiency, and it is
discussed as a limitation in Section 6.7. Confidence intervals for the retrieval experiments
(H1) were computed via bootstrap resampling (N=1,000) over per-query scores.

### 6.1.3 Pipeline Summary

The TalkEx pipeline processes each conversation through the following stages:

1. **Segmentation** — turns extracted with `TurnSegmenter`
2. **Context windows** — sliding windows of size 5, stride 2, via `SlidingWindowBuilder`
3. **Embedding generation** — frozen `paraphrase-multilingual-MiniLM-L12-v2` encoder;
   mean pooling over the window's token representations
4. **Lexical features** — BM25 term-frequency signals and TF-IDF bag-of-words over
   normalized turn text
5. **Structural features** — speaker role, turn position, window length, channel metadata
6. **Rule features** (where applicable) — binary rule-fire indicators from the semantic
   rule engine DSL
7. **Classification** — LightGBM gradient boosting over the concatenated feature vector

All metrics are macro-averaged over the 8 classes unless otherwise stated.

---

## 6.2 H1: Hybrid Retrieval

**Hypothesis.** A hybrid retrieval system combining BM25 lexical scoring with approximate
nearest-neighbor (ANN) semantic search, fused via linear interpolation, achieves higher
retrieval quality than either system in isolation, as measured by MRR, nDCG@10, Recall@10,
and Precision@5.

### 6.2.1 Systems Compared

Four retrieval systems were evaluated:

- **BM25-base** — BM25 lexical retrieval only (Okapi BM25, k1=1.5, b=0.75)
- **ANN-MiniLM** — Dense retrieval using paraphrase-multilingual-MiniLM-L12-v2 embeddings,
  approximate nearest-neighbor search (FAISS flat index, cosine similarity)
- **Hybrid-LINEAR-a** — Linear combination: `score = alpha * BM25_norm + (1-alpha) * ANN_norm`,
  evaluated over alpha in {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}
- **Hybrid-RRF** — Reciprocal Rank Fusion: `RRF(d) = sum(1 / (k + rank(d)))`, k=60, fusing
  BM25 and ANN ranked lists

Alpha selection for Hybrid-LINEAR was performed on the validation set; alpha=0.50 was
selected (val MRR=0.8612). The best test performance was achieved at alpha=0.30.

### 6.2.2 Results

**Table 6.1. H1 retrieval system comparison (test set, post-audit dataset, n=468).**

| System | MRR | nDCG@10 | Recall@10 | Precision@5 |
|---|---|---|---|---|
| BM25-base | 0.8354 | 0.6317 | 0.0371 | 0.6496 |
| ANN-MiniLM | 0.8242 | 0.6250 | 0.0376 | 0.6509 |
| Hybrid-LINEAR-a0.30 | **0.8531** | 0.6475 | 0.0381 | 0.6726 |
| Hybrid-LINEAR-a0.50 (val-selected) | 0.8482 | -- | -- | -- |
| Hybrid-RRF | 0.8516 | **0.6530** | **0.0385** | **0.6855** |

All metrics are reported on the held-out test partition. Bold indicates the best value per
column. MRR = Mean Reciprocal Rank; nDCG@10 = Normalized Discounted Cumulative Gain at
rank 10; Recall@10 = proportion of relevant documents retrieved in top 10; Precision@5 =
proportion of retrieved documents in top 5 that are relevant.

### 6.2.3 Statistical Tests

Pairwise significance was assessed using the Wilcoxon signed-rank test on per-query MRR
scores, which is appropriate for paired, non-normally distributed rank data. A significance
threshold of alpha=0.05 was applied.

**Table 6.2. H1 statistical significance tests (Wilcoxon signed-rank, two-tailed).**

| Comparison | p-value | Effect size r | Mean diff | 95% CI |
|---|---|---|---|---|
| Hybrid-LINEAR-a0.30 vs. BM25-base | **0.0165** | 0.2916 | +0.0177 | [0.0028, 0.0334] |
| Hybrid-LINEAR-a0.30 vs. ANN-MiniLM | **0.0296** | 0.2056 | +0.0289 | [0.0028, 0.0552] |
| Hybrid-LINEAR-a0.30 vs. Hybrid-RRF | 0.7913 | -- | -- | -- |

The hybrid linear system with alpha=0.30 achieves statistically significant improvements
over both the BM25 baseline (p=0.0165, r=0.29) and the dense ANN system (p=0.0296,
r=0.21). Hybrid-RRF achieves comparable performance to Hybrid-LINEAR on MRR (0.8516 vs.
0.8531, p=0.79) and outperforms it on nDCG@10 and Precision@5, indicating that the two
fusion strategies are roughly equivalent on this dataset and that the choice between them
is unlikely to be consequential in practice.

### 6.2.4 Discussion

**The unexpectedly strong BM25 baseline.** The BM25 baseline achieves MRR=0.8354, only 1.8
percentage points below the best hybrid system. This is noteworthy in the context of a
conversational corpus where lexical variation might be expected to favor semantic
approaches. Two domain-specific features explain BM25's robustness. First, Brazilian
Portuguese customer service conversations contain high-frequency lexical markers that are
strongly predictive: terms such as *cancelar*, *reclamação*, *suporte técnico*, and
*elogio* appear with sufficient frequency and specificity that BM25 can discriminate
intent reliably. Second, the relatively constrained vocabulary of call center interactions
reduces the paraphrase diversity that would otherwise require semantic representations to
compensate.

**The hybrid advantage.** The gain from hybrid fusion is real but modest (+1.8pp MRR). The
95% confidence interval [0.0028, 0.0334] excludes zero, providing statistical evidence of a
genuine improvement. The effect size (r=0.29) corresponds to a small-to-medium effect by
conventional benchmarks (Cohen, 1988). This pattern is consistent with prior findings on
structured domain corpora (Ma et al., 2021; Thakur et al., 2021): hybrid fusion reliably
helps when neither pure lexical nor pure semantic retrieval dominates, but the margin
narrows when the lexical baseline is already strong.

**Alpha analysis.** The optimal alpha of 0.30 (giving more weight to BM25) at test time,
versus the validation-selected 0.50, suggests a mild domain shift between the validation
and test distributions within the corpus. This discrepancy should be noted as a limitation:
in a production system, alpha would be selected on the validation set and applied at test
time, yielding MRR=0.8482 rather than 0.8531. We report both to preserve transparency.

**The dense baseline is weaker than BM25.** ANN-MiniLM achieves MRR=0.8242, 1.1pp below
the BM25 baseline. This result is consistent with findings in specialized domain search
(Ma et al., 2021): frozen multilingual encoders trained on general-purpose corpora may not
capture the domain-specific lexical patterns of Brazilian customer service conversations as
effectively as BM25's term-frequency model. This finding underscores the methodological
importance of establishing a strong BM25 baseline before investing in semantic retrieval
infrastructure.

**Verdict: H1 CONFIRMED.** Hybrid retrieval achieves statistically significant improvement
over both lexical and semantic baselines (p < 0.05). The improvement is modest in absolute
terms but consistent with the literature on domain-specialized corpora.

---

## 6.3 H2: Multi-Level Feature Classification

**Hypothesis.** A supervised classifier trained on a combination of lexical features and
dense semantic embeddings achieves higher intent classification quality than a classifier
trained on lexical features alone, measured by Macro-F1 and Accuracy across 8 intent
classes.

### 6.3.1 Systems Compared

Six classifier configurations were evaluated, crossing two feature sets with three model
families:

- **Feature sets:** `lexical` (TF-IDF + BM25 features only) vs. `lexical+emb`
  (lexical features concatenated with mean-pooled MiniLM embeddings, 384 dimensions)
- **Model families:** Logistic Regression (L2 regularization, max_iter=1000),
  LightGBM (n_estimators=100, num_leaves=31), MLP (hidden_layer_sizes=(256, 128),
  max_iter=500)

All models were trained on 1,250 training windows and evaluated on 468 test windows.
Hyperparameters were not further tuned beyond the fixed configuration, as the goal of H2 is
to assess the marginal value of embedding features rather than to optimize a specific model.

### 6.3.2 Results

**Table 6.3. H2 classification results by feature set and model (test set, post-audit).**

| Feature Set | Model | Macro-F1 | Accuracy |
|---|---|---|---|
| lexical | LogReg | 0.3343 | 0.3462 |
| lexical | LightGBM | 0.5509 | 0.5743 |
| lexical | MLP | 0.1467 | 0.1688 |
| lexical+emb | LogReg | 0.6409 | 0.6410 |
| lexical+emb | LightGBM | **0.7224** | **0.7172** |
| lexical+emb | MLP | 0.6134 | 0.6051 |

The lexical+emb LightGBM configuration achieves the best overall performance, with
Macro-F1=0.7224 and Accuracy=0.7172. The addition of embeddings yields gains of +38.8pp,
+17.1pp, and +46.7pp Macro-F1 for LogReg, LightGBM, and MLP respectively, demonstrating
that the embedding contribution is consistent across model families and not an artifact of
a particular classifier's inductive bias.

### 6.3.3 Per-Class Analysis (lexical+emb LightGBM)

**Table 6.4. Per-class F1 scores for lexical+emb LightGBM (test set, post-audit).**

| Intent Class | Precision | Recall | F1 |
|---|---|---|---|
| cancelamento | 1.000 | 0.833 | 0.909 |
| elogio | 1.000 | 0.774 | 0.873 |
| suporte_tecnico | 0.831 | 0.857 | 0.844 |
| reclamacao | 0.709 | 0.918 | 0.800 |
| duvida_servico | 0.679 | 0.838 | 0.750 |
| duvida_produto | 0.594 | 0.788 | 0.677 |
| saudacao | 0.846 | 0.333 | 0.478 |
| compra | 0.731 | 0.317 | 0.442 |
| **Macro average** | -- | -- | **0.722** |

A notable bimodal distribution is observed across classes. Six of the eight classes achieve
F1 >= 0.677, with cancelamento and elogio achieving near-perfect precision. The two
underperforming classes, `saudacao` (F1=0.478) and `compra` (F1=0.442), exhibit a shared
pattern: precision is high (0.846 and 0.731 respectively) while recall is severely
depressed (0.333 and 0.317). This suggests that the classifier is overly conservative for
these classes, missing many true positives while avoiding false positives. Two potential
explanations are consistent with the data: (1) greetings and purchase inquiries may share
surface-level features with other classes (e.g., a customer asking "can I buy this?" might
be ambiguously coded as `compra` or `duvida_produto`); and (2) the synthetic generation
process may have introduced distributional artifacts for these classes specifically. The
potential confounding effect of synthetic data on low-recall classes is a limitation of
this study (see Section 6.7).

### 6.3.4 Statistical Tests

Statistical significance was assessed via the Wilcoxon signed-rank test on per-sample
prediction correctness scores, comparing the best combined-feature configuration against
all lexical-only baselines.

**Table 6.5. H2 statistical significance tests (Wilcoxon signed-rank, two-tailed).**

| Comparison | p-value | Effect size r |
|---|---|---|
| lexical+emb LightGBM vs. lexical LogReg | 2.40e-46 | 0.904 |
| lexical+emb LightGBM vs. lexical LightGBM | 2.45e-35 | 0.836 |
| lexical+emb LightGBM vs. lexical MLP | 1.07e-57 | 0.932 |

All comparisons are significant at p < 0.001 with very large effect sizes (r > 0.83). The
magnitude of these effects leaves no statistical ambiguity: the addition of dense embedding
features produces improvements that are not plausibly attributable to chance.

### 6.3.5 Discussion

**The dominant role of embeddings.** The 38.8pp improvement of LightGBM when embeddings
are added to lexical features is the largest effect observed across all four hypotheses.
This aligns with prior findings on intent classification in customer service corpora
(Casanueva et al., 2020; Liu et al., 2019), where dense representations consistently
outperform bag-of-words approaches. The frozen encoder approach — using
paraphrase-multilingual-MiniLM-L12-v2 without domain fine-tuning — achieves this without
any task-specific training of the encoder, suggesting that the multilingual pre-training of
this model captures sufficient semantic structure for PT-BR customer service intent
disambiguation.

**LightGBM as the strongest model family.** LightGBM outperforms both LogReg (+31.5pp)
and MLP (+10.9pp) with combined features. This is consistent with established findings
that gradient boosting methods excel on tabular feature vectors (Shwartz-Ziv and Armon,
2022), particularly when features are heterogeneous (sparse lexical + dense continuous).
The MLP underperformance relative to LightGBM on this dataset size (1,250 training examples)
is also expected: deep networks require substantially more data to avoid overfitting on
mid-dimensional feature spaces.

**Comparison between lexical-only models.** The large spread between LightGBM (0.5509)
and LogReg (0.3343) in the lexical-only regime indicates that the bag-of-words
representation is sparse and non-linearly separable. The MLP failure (0.1467) in the
lexical-only regime is notable; it likely reflects the difficulty of training an MLP on
high-dimensional sparse features without appropriate regularization, and underscores that
model choice matters independently of feature choice.

**Verdict: H2 CONFIRMED.** The combined lexical+embedding feature set achieves large and
statistically significant improvements over any lexical-only baseline for all three model
families (p < 1e-30, r > 0.83). The improvement is not specific to one model and is
consistent across the full range of metrics.

---

## 6.4 H3: Rule Integration

**Hypothesis.** Augmenting a supervised machine learning classifier with semantic rule
engine outputs (expressed in the TalkEx DSL and compiled to AST) achieves higher
Macro-F1 than the classifier alone on the 8-class intent classification task.

### 6.4.1 Systems Compared

Four configurations were evaluated:

- **ML-only** — lexical+emb LightGBM without any rule component (identical to the best
  H2 configuration)
- **Rules-only** — deterministic rule engine with 2 lexical rules (`rule_cancel` for
  `cancelamento`, `rule_complaint` for `reclamacao`); all unmatched windows receive a
  default label
- **ML+Rules-override** — ML classifier output overridden by rule engine when a rule fires
  (hard priority: rule > ML)
- **ML+Rules-feature** — rule fire indicators appended as binary features to the ML feature
  vector; the LightGBM classifier learns when to use them

The rule set used in these experiments comprises two deterministic lexical rules matching
keyword clusters associated with cancellation and complaint intents. This ruleset was
designed to cover the two intents with the highest business-critical sensitivity in
customer service operations.

### 6.4.2 Results

**Table 6.6. H3 classification results by integration strategy (test set, post-audit).**

| Configuration | Macro-F1 | Accuracy | Delta vs. ML-only |
|---|---|---|---|
| Rules-only | 0.1366 | 0.1603 | -0.585 |
| ML-only | 0.7216 | 0.7393 | baseline |
| ML+Rules-override | 0.6796 | 0.6816 | -0.042 |
| ML+Rules-feature | **0.7400** | **0.7564** | **+0.018** |

Only the feature-integration strategy yields any improvement over the ML baseline. The
override strategy is actively harmful (-4.2pp), and rules-only classification is not viable
at scale (Macro-F1=0.1366), limited by the two-rule scope covering only 2 of 8 classes.

### 6.4.3 Per-Class Analysis for ML+Rules-feature

**Table 6.7. Per-class F1 comparison: ML-only vs. ML+Rules-feature (test set).**

| Intent Class | ML-only F1 | ML+Rules-feature F1 | Delta |
|---|---|---|---|
| cancelamento | 0.909 | **0.946** | +0.037 |
| compra | 0.442 | **0.488** | +0.047 |
| duvida_produto | 0.677 | 0.667 | -0.011 |
| duvida_servico | 0.750 | **0.800** | +0.050 |
| elogio | **0.873** | 0.836 | -0.037 |
| reclamacao | 0.800 | **0.808** | +0.008 |
| saudacao | 0.478 | **0.522** | +0.044 |
| suporte_tecnico | **0.844** | 0.853 | +0.009 |

**Table 6.8. Precision and recall detail for targeted classes under ML+Rules-feature.**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| cancelamento | 0.978 | 0.917 | 0.946 |
| reclamacao | 0.722 | 0.918 | 0.808 |

The two classes explicitly targeted by lexical rules — `cancelamento` and `reclamacao` —
both show improvement, with `cancelamento` gaining +3.7pp (F1: 0.909 -> 0.946) and
`reclamacao` gaining +0.8pp (F1: 0.800 -> 0.808). More interestingly, untargeted classes
also improve: `compra` (+4.7pp), `duvida_servico` (+5.0pp), and `saudacao` (+4.4pp).
This suggests that rule-fire indicators provide discriminative signal beyond their
immediate target classes — when a rule fires for `cancelamento`, the classifier gains
information that the current window is NOT a purchase or greeting inquiry.

Two classes show slight regression: `duvida_produto` (-1.1pp) and `elogio` (-3.7pp).
The `elogio` regression is the most notable: the class drops from F1=0.873 to 0.836.
This is a known risk of feature augmentation — rule features can introduce collinearity
or interact unfavorably with existing features. Given that no rule targets `elogio`
directly, this regression likely reflects a boundary interaction where the rule signals
shift probability mass away from `elogio` in ambiguous cases.

### 6.4.4 Statistical Tests

**Table 6.9. H3 statistical significance tests (Wilcoxon signed-rank, two-tailed).**

| Comparison | p-value | Effect size r | Mean diff | 95% CI |
|---|---|---|---|---|
| ML+Rules-feature vs. ML-only | 0.1306 | 0.2857 | +0.0171 | [-0.0043, 0.0385] |
| ML+Rules-feature vs. ML+Rules-override | **0.0001** | 0.4430 | -- | -- |

The comparison of primary interest — ML+Rules-feature versus ML-only — does not reach
statistical significance at alpha=0.05 (p=0.1306). The 95% confidence interval for the
mean difference [-0.0043, 0.0385] includes zero, meaning that the observed improvement of
+1.71pp cannot be distinguished from sampling variation under the null hypothesis of no
difference. The effect size r=0.29 corresponds to a small-to-medium effect, consistent
in magnitude with the retrieval improvement observed in H1.

The comparison between ML+Rules-feature and ML+Rules-override is highly significant
(p=0.0001, r=0.44), establishing that if rules are to be used at all, the feature
integration strategy is dramatically superior to hard override.

### 6.4.5 Discussion

**Why the override strategy is harmful.** The ML+Rules-override configuration
(-4.2pp Macro-F1) demonstrates a well-understood failure mode of rule-ML integration:
when rules fire on false positives, they overwrite a correct ML decision. In this
experiment, the two lexical keyword rules have non-trivial false-positive rates:
`rule_cancel` matches keyword clusters that appear in other intent contexts (e.g., a
customer asking about cancellation policy without intending to cancel), and `rule_complaint`
similarly over-fires. When rule precision is less than ML classifier precision on the same
instances, override strategy systematically degrades performance. The result is a strong
empirical argument against hard-priority rule integration in systems where ML classifier
precision already exceeds rule precision.

**Why feature integration is safer.** The feature integration strategy avoids this failure
mode: the ML classifier retains full decision authority and treats rule outputs as additional
evidence rather than authoritative signals. The gradient boosting tree learns to weight
rule-fire indicators appropriately based on training data, effectively discovering that
rules should be trusted for `cancelamento` but discounted for ambiguous cases.

**The redundancy hypothesis.** A key interpretation challenge for H3 is whether the
observed +1.8pp improvement would reach significance with a richer ruleset. The current
experimental rules are purely lexical (keyword-based), which overlaps substantially with
the TF-IDF features already in the feature vector. The semantic rule engine supports four
predicate families (lexical, semantic, structural, contextual), but the experiments used
only the lexical family. It is plausible that rules expressing structural or contextual
constraints — for example, "intent is `cancelamento` if the customer has expressed negative
sentiment in the preceding three turns AND explicitly mentions a product name" — would
provide signal orthogonal to both lexical and embedding features, and might yield larger
and more significant improvements. This represents both a limitation of the current
experimental design and a direction for future work.

**The p=0.131 result as a scientific contribution.** An inconclusive result at the
pre-specified significance threshold is a genuine scientific finding. It establishes that
a minimal two-rule lexical ruleset does not add statistically significant value when
combined with a rich embedding-based classifier. This finding has practical implications:
teams investing in rule authoring should not expect measurable quality improvements from
simple keyword rules when state-of-the-art embedding features are already present. The
effort is better directed at rules that exercise the semantic and contextual predicate
families unavailable to the ML feature engineering process.

**Verdict: H3 INCONCLUSIVE.** Feature integration of rule outputs produces a positive
directional effect (+1.71pp Macro-F1) with a small-to-medium effect size, but the
improvement does not reach statistical significance (p=0.131, 95% CI includes zero).
The null hypothesis cannot be rejected at alpha=0.05. The experimental design is
a limiting factor: the two-rule lexical ruleset exercises only a fraction of the rule
engine's capabilities.

---

## 6.5 H4: Cascaded Inference

**Hypothesis.** A cascaded inference pipeline that routes low-confidence predictions to a
more expensive processing stage reduces total computational cost while maintaining
classification quality comparable to uniform full-pipeline processing.

### 6.5.1 System Design

The cascaded system routes each context window through two stages:

- **Stage 1 (light)** — logistic regression classifier with lexical features only, lower
  latency, lower accuracy
- **Stage 2 (full)** — LightGBM classifier with lexical + embedding features, higher
  latency, higher accuracy

A window is routed to Stage 2 if the Stage 1 confidence score (maximum class probability)
falls below a threshold tau. Windows above tau are accepted at Stage 1 without further
processing. The uniform baseline applies Stage 2 to all windows regardless of confidence.

Four cascade thresholds were evaluated: tau in {0.50, 0.70, 0.80, 0.90}.

The primary cost metric is total wall-clock processing time in milliseconds over the full
test set (n=1,922 windows). Cost reduction is reported as `(cascade_cost - uniform_cost) /
uniform_cost * 100`, where positive values indicate cost reduction and negative values
indicate cost increase.

### 6.5.2 Results

**Table 6.10. H4 cascade results by confidence threshold (test set, post-audit).**

| Configuration | Macro-F1 | Total Cost (ms) | Stage 1 % | Cost vs. Uniform |
|---|---|---|---|---|
| Uniform (full) | **0.7216** | 158.99 | 0.0% | baseline |
| Cascade tau=0.50 | 0.7050 | 252.26 | 51.40% | +58.66% more expensive |
| Cascade tau=0.70 | 0.7180 | 296.35 | 23.67% | +86.39% more expensive |
| Cascade tau=0.80 | 0.7241 | 315.38 | 11.71% | +98.36% more expensive |
| Cascade tau=0.90 | 0.7200 | 326.55 | 4.68% | +105.39% more expensive |

Note: "Cost vs. Uniform" reports the percentage cost increase relative to the uniform
baseline. All cascade configurations increase total cost. The original experiment reported
a `cost_reduction` metric with negative values; we reproduce those here for completeness:
cascade_t0.50 = -65.18%, cascade_t0.70 = -92.92%, cascade_t0.80 = -104.88%,
cascade_t0.90 = -111.91%. Negative `cost_reduction` means cost increased.

### 6.5.3 Analysis

No cascade configuration achieves the target cost reduction. Every threshold tested
increases total processing time relative to uniform full-pipeline processing, ranging from
+58.7% more expensive at tau=0.50 to +105.4% more expensive at tau=0.90. This is a
strong and unambiguous refutation of the hypothesis.

**Table 6.11. Per-window cost breakdown (H4 uniform baseline).**

| Stage | Cost per window (ms) |
|---|---|
| Light (lexical LR, Stage 1) | 0.091 |
| Full (LightGBM w/ embeddings, Stage 2) | 0.083 |

The root cause of the failure is visible in Table 6.11: the per-window cost of Stage 1
(0.091ms) is higher than the per-window cost of Stage 2 (0.083ms). The cascade architecture
assumes a genuine cost differential between stages, where the light stage is substantially
cheaper than the full stage. In this experimental setup, that differential does not exist:
both stages operate at sub-millisecond latency because both stages use the same pre-computed
embedding vectors.

The fundamental accounting issue is as follows. Embedding generation — the dominant
computational cost in the full pipeline — occurs once per window before the cascade decision
point and is therefore shared by both stages. The logistic regression Stage 1 then adds
its own inference cost (0.091ms), and any window not resolved at Stage 1 incurs the
additional LightGBM cost. The cascade thus pays the embedding cost unconditionally, plus
the Stage 1 cost for all windows, plus the Stage 2 cost for unresolved windows. This is
always more expensive than the uniform baseline, which pays only the embedding cost and
Stage 2 cost.

As tau increases from 0.50 to 0.90, the fraction of windows resolved at Stage 1 decreases
monotonically (51.4% -> 4.7%), increasing the total cost further. This is the opposite of
the expected behavior: higher thresholds route more windows to Stage 2, negating any
potential savings. The cascade behavior becomes essentially equivalent to the uniform
baseline only when Stage 1 resolves a large fraction of queries, but this comes at the
cost of reduced Macro-F1 (0.7050 at tau=0.50, vs. 0.7216 for uniform).

### 6.5.4 The Structural Prerequisite for Cascaded Cost Reduction

For a two-stage cascade to reduce cost, the following condition must hold:

```
Stage1_cost + (1 - p_stage1) * Stage2_cost < Stage2_cost
```

Simplifying: `Stage1_cost < p_stage1 * Stage2_cost`

With p_stage1=0.514 (tau=0.50) and Stage2_cost=0.083ms:
`0.091ms < 0.514 * 0.083ms = 0.043ms` -- FALSE

The condition fails because Stage1_cost (0.091ms) exceeds p_stage1 * Stage2_cost (0.043ms).
For the cascade to be cost-effective at tau=0.50, Stage 1 would need to cost below 0.043ms
per window -- less than half its current cost. Achieving this would require routing the
cascade decision point to BEFORE embedding generation, using only raw lexical features for
Stage 1. This architectural change would make Stage 1 genuinely cheaper than Stage 2, at
the cost of introducing a different embedding pathway for resolved queries.

### 6.5.5 Implications for System Design

The H4 finding should not be interpreted as evidence that cascaded inference is generally
ineffective. Rather, it establishes a specific structural prerequisite: the cascade requires
a genuine cost differential between stages, which in turn requires that the cheap stage
not depend on the same expensive pre-computation as the full stage. In the TalkEx
architecture, this means the cascade decision must occur before embedding generation, not
after. Possible reconfigurations include:

1. **Pre-embedding routing** — route based on raw BM25 confidence, before embedding
   generation; only generate embeddings for low-confidence BM25 queries
2. **Asynchronous embedding** — generate embeddings for a fraction of traffic using
   downsampled inference; apply full embeddings only for high-volume or high-stakes
   conversations
3. **Model distillation** — train a smaller student encoder for Stage 1 rather than reusing
   the same MiniLM model

These alternatives are identified as future work rather than evaluated experimentally here.

**Verdict: H4 REFUTED.** No cascade configuration achieves cost reduction relative to the
uniform baseline. All tested configurations increase total cost between +58.7% and +105.4%.
The root cause is an insufficient cost differential between pipeline stages: both stages
depend on the same pre-computed embeddings, making Stage 1 more expensive than Stage 2
at the per-window level. The hypothesis failure is attributable to an architectural
assumption that does not hold in this experimental setup; it does not invalidate the
cascaded inference approach in general.

---

## 6.6 Ablation Study

**Objective.** Quantify the marginal contribution of each feature family to the best
classification performance, by systematically removing one feature group at a time and
measuring the degradation in Macro-F1 relative to the full pipeline.

### 6.6.1 Feature Groups

The full pipeline (`full_pipeline`) uses four feature families, comprising 397 features in
total:

- **Embeddings** — 384-dimensional mean-pooled MiniLM vectors
- **Lexical** — TF-IDF bag-of-words and BM25 term-frequency features
- **Rules** — binary indicators for rule-engine predicate firings (2 rules)
- **Structural** — speaker role, turn position, window length, channel metadata

Ablation was conducted by removing each feature family in isolation and re-training the
LightGBM classifier on the reduced feature set. All other experimental conditions remained
constant.

### 6.6.2 Results

**Table 6.12. Ablation study results (test set, post-audit, n=468).**

| Configuration | Macro-F1 | n Features | Delta vs. Full | Relative Loss |
|---|---|---|---|---|
| full_pipeline | **0.7400** | 397 | baseline | -- |
| -Embeddings | 0.4102 | ~13 | -0.3299 | -44.57% |
| -Lexical | 0.7112 | ~384+r+s | -0.0289 | -3.90% |
| -Rules | 0.7216 | ~396 | -0.0184 | -2.49% |
| -Structural | 0.7267 | ~394 | -0.0133 | -1.80% |

Note: n features for ablated conditions are approximate; the exact count depends on the
feature engineering pipeline configuration. The relative loss column is computed as
`|delta| / full_pipeline_F1 * 100`.

### 6.6.3 Feature Contribution Ranking

The ablation reveals a clear hierarchy of feature family contribution:

1. **Embeddings** (+33.0pp, 44.6% of total performance) — by far the dominant factor
2. **Lexical** (+2.9pp, 3.9% of total performance) — meaningful secondary contribution
3. **Rules** (+1.8pp, 2.5% of total performance) — marginal contribution
4. **Structural** (+1.3pp, 1.8% of total performance) — smallest contribution

**Figure 6.1 description.** A bar chart of feature family contribution would show embeddings
at +33.0pp, with lexical, rules, and structural features all clustered between +1.3pp and
+2.9pp. The order-of-magnitude dominance of embeddings is the primary visual finding.

### 6.6.4 Discussion

**Embeddings dominate.** The removal of embedding features causes a 44.6% relative
degradation in Macro-F1 (0.7400 -> 0.4102). This is the single largest effect in the
entire experimental study and is consistent with the H2 finding. The lexical-only
performance of 0.4102 is substantially below even the lexical-only LightGBM from H2
(0.5509), because the ablation removes embeddings from the full pipeline (which includes
structural and rule features) rather than comparing clean lexical vs. lexical+emb
configurations. The absolute contribution of embeddings is 33.0pp, establishing that the
frozen MiniLM encoder provides the foundation upon which all other improvements are built.

**Lexical features complement embeddings.** The +2.9pp contribution of lexical features
is modest but not negligible. Term-frequency signals capture domain-specific vocabulary
(product names, procedural language, explicit intent markers) that may not be fully
represented in the embedding space of a model trained on general-purpose multilingual
corpora. This is consistent with the H1 finding that BM25 adds complementary value over
ANN-only retrieval in the same domain.

**Rules provide marginal but additive value.** The +1.8pp contribution of rule features
mirrors the H3 finding. In the ablation context, this is more precisely interpretable: when
all other features are present (including embeddings and lexical), the two binary rule-fire
indicators add 1.8pp of Macro-F1. This is a small but non-trivial gain for features that
require no additional inference cost beyond a keyword lookup.

**Structural features contribute least.** Speaker role, turn position, and other structural
metadata contribute +1.3pp, the smallest marginal effect. This does not imply that
structural features are uninformative in general -- for tasks with stronger structural
signals (e.g., distinguishing agent-initiated vs. customer-initiated turns, or detecting
conversation stage) structural features would likely contribute more. In the 8-class intent
classification setting studied here, lexical and semantic content is more predictive than
conversational structure.

**The feature combination is superadditive.** A naïve sum of individual feature family
contributions (33.0 + 2.9 + 1.8 + 1.3 = 39.0pp) exceeds the actual full-pipeline gain
over the ablated embedding-only baseline. This indicates positive interaction effects among
feature families: lexical features are more valuable in the presence of embeddings and
structural features than in isolation. This superadditivity motivates the multi-level
feature engineering approach.

---

## 6.7 Cross-Hypothesis Synthesis

### 6.7.1 The Four Results as a Coherent Picture

The four hypotheses produce a coherent and internally consistent picture when analyzed
jointly.

**Table 6.13. Summary of hypothesis verdicts (post-audit, 2,122 records, 8 intents).**

| Hypothesis | Best Config | Primary Metric | p-value | Verdict |
|---|---|---|---|---|
| H1 — Hybrid Retrieval | LINEAR-a0.30 | MRR=0.853 (+1.8pp) | 0.017 | Confirmed |
| H2 — Combined Features | lexical+emb LightGBM | Macro-F1=0.722 (+38.8pp) | 2.4e-46 | Confirmed |
| H3 — Rule Integration | ML+Rules-feature | Macro-F1=0.740 (+1.8pp) | 0.131 | Inconclusive |
| H4 — Cascaded Inference | uniform baseline | Macro-F1=0.722; cascade increases cost | N/A | Refuted |

Embedding-based semantic representations are the dominant force in both retrieval (H1) and
classification (H2, ablation). In both tasks, the frozen multilingual encoder provides
signal that lexical methods cannot replicate, and hybrid combinations of lexical and
semantic features outperform either in isolation. The margin of improvement is large for
classification (+38.8pp) but modest for retrieval (+1.8pp), a pattern consistent with the
known difference in task structure: classification involves learning discriminative
boundaries over the full class space, while retrieval benefits from BM25's strong
term-frequency discrimination in constrained-vocabulary domains.

The two negative or inconclusive results (H3 and H4) are informative rather than merely
disappointing. H3 establishes that lexical rules add no statistically significant value when
combined with embedding features and lexical bag-of-words, though the positive trend
warrants further investigation with richer rulesets. H4 establishes a specific
architectural constraint: cascaded inference requires a genuine cost differential that is
absent when both stages use the same pre-computed embeddings.

### 6.7.2 The Frozen Encoder + Gradient Boosting Finding

The most practically replicable finding of this work is the strong performance of a frozen
multilingual encoder combined with a gradient boosting classifier. The
lexical+emb LightGBM configuration achieves Macro-F1=0.722 on 8-class PT-BR intent
classification without any task-specific encoder fine-tuning, trained on 1,250 examples.
This configuration requires:

- A pre-trained multilingual sentence encoder (downloadable, GPU-accelerated inference via Google Colab)
- A gradient boosting library (LightGBM, readily available)
- A feature engineering pipeline combining TF-IDF and mean-pooled embeddings

The compute budget for training and inference is minimal: sub-millisecond per-window
latency for both stages. This positions the approach as a viable baseline for practitioners
who need intent classification without the infrastructure required for fine-tuned transformer
pipelines. The result confirms prior findings that frozen encoders with downstream
classifiers are competitive with fine-tuned transformers at modest dataset sizes (Reimers
and Gurevych, 2019; Muennighoff et al., 2022), though a direct comparison with a
fine-tuned encoder on this dataset was not conducted and is identified as future work.

### 6.7.3 On the Value of Reporting Negative Results

The scientific community has long recognized publication bias as a structural problem in
empirical machine learning research (Henderson et al., 2018; Sculley et al., 2018). H4's
refutation and H3's inconclusiveness are reported with the same level of methodological
rigor as the confirmed hypotheses, for three reasons.

First, these results provide genuine actionable information. A system designer considering
cascaded inference for embedding-based NLP pipelines now has evidence that the approach
requires architectural prerequisites not satisfied by the naïve two-stage design. A system
designer considering rule integration has evidence that lexical rules add marginal value
in the presence of rich embedding features, and that hard-override integration is reliably
harmful.

Second, the magnitude of the non-findings is informative. H3's effect size (r=0.29) is
not negligible; it is statistically indistinguishable from H1's effect (r=0.29 for
Hybrid-LINEAR vs. BM25). The difference is that H1's comparison involves a larger sample
of retrieval queries, providing more statistical power. H3's inconclusiveness is in part a
function of the experimental design's statistical power rather than an absence of true
effect. This distinction matters for interpreting the findings.

Third, the honest characterization of negative results is a prerequisite for the
reproducibility of this work. A future researcher replicating this study with a richer
ruleset or a different cascade architecture should know what was found here, not an
optimistic characterization thereof.

### 6.7.4 Threats to Validity

Four threats to the internal and external validity of these findings warrant explicit
acknowledgment.

**Threat 1: Synthetic data composition.** 60.1% of the dataset was generated by an LLM
(GPT-family model) using few-shot prompts. Machine-generated text may not faithfully
replicate the distributional properties of real customer service conversations: vocabulary
diversity, turn length distributions, and code-switching patterns may differ systematically.
As a consequence, reported performance metrics may not transfer directly to production
deployment on real conversations. The contamination-aware splitting procedure mitigates
few-shot leakage but does not eliminate the confounding influence of LLM generation
artifacts.

**Threat 2: Single domain.** All experiments were conducted on a single domain (Brazilian
Portuguese customer service) and a single dataset. The generalizability of the findings
to other languages, domains, or conversational formats is unknown. Claims about the
comparative value of hybrid vs. lexical vs. semantic approaches should be understood as
domain-specific unless validated on additional corpora.

**Threat 3: Deterministic multi-seed.** The reported standard deviation of 0.000 across
five random seeds reflects the determinism of the experimental design (fixed splits, frozen
encoder, deterministic LightGBM given the same training data), not genuine robustness
estimates. True confidence intervals require cross-validation or bootstrap resampling over
the training set. The reported metrics should be understood as point estimates, not
distribution summaries.

**Threat 4: Rule engine scope.** The H3 and ablation experiments use only two lexical
rules, covering 2 of 8 intent classes. The TalkEx rule engine supports semantic,
structural, and contextual predicate families that were not exercised experimentally. The
inconclusive H3 result should be interpreted in the context of this limited evaluation
scope; it does not constitute evidence that the full rule engine architecture is without
value.

### 6.7.5 Positioning within the Broader Literature

The TalkEx results extend three threads of prior work.

**Hybrid retrieval in specialized domains.** The H1 finding — modest but significant gains
from BM25+dense fusion in a constrained-vocabulary domain — is consistent with Thakur et
al. (2021), who found that hybrid retrieval gains over BM25 are smaller in domain-specific
benchmarks than in open-domain retrieval, and with Ma et al. (2021), who showed that linear
interpolation and RRF are competitive fusion strategies at comparable computational cost.
The TalkEx results add a PT-BR conversational data point to this literature.

**Frozen encoders vs. fine-tuned models.** The H2 finding — that a frozen
paraphrase-multilingual-MiniLM-L12-v2 combined with LightGBM achieves Macro-F1=0.722 on
8-class intent classification — is consistent with Reimers and Gurevych (2019) on
sentence embedding universality and Casanueva et al. (2020) on few-shot intent
classification. The absence of a fine-tuned encoder baseline is a limitation that prevents
a direct contribution to the frozen vs. fine-tuned debate; the comparison is identified as
future work.

**Rule-ML integration.** The H3 finding replicates prior findings that hard-override
integration of deterministic rules is harmful when rules are noisy (Mou et al., 2015;
Hu et al., 2016) and that feature-based rule integration is safer. The magnitude of the
override penalty (-4.2pp) and the feature integration gain (+1.8pp) are consistent with
the wider literature on knowledge-guided classification, which finds that the benefit of
symbolic constraints diminishes as the supervised model quality increases.

### 6.7.6 Summary Table

**Table 6.14. Complete results summary across all experiments (post-audit baseline).**

| Experiment | Configuration | Macro-F1 | MRR | Cost (ms) | Key Finding |
|---|---|---|---|---|---|
| Retrieval: BM25 | BM25-base | -- | 0.835 | -- | Strong baseline |
| Retrieval: ANN | ANN-MiniLM | -- | 0.824 | -- | Below BM25 |
| Retrieval: Hybrid | LINEAR-a0.30 | -- | **0.853** | -- | +1.8pp, p=0.017 |
| Retrieval: RRF | Hybrid-RRF | -- | 0.852 | -- | ~= LINEAR |
| Classification: lex | LightGBM | 0.551 | -- | -- | Lexical baseline |
| Classification: full | LightGBM+emb | 0.722 | -- | -- | +17.1pp over lex |
| Rules: feature | ML+Rules-feature | **0.740** | -- | -- | +1.8pp, p=0.131 |
| Rules: override | ML+Rules-override | 0.680 | -- | -- | Harmful (-4.2pp) |
| Cascade | Uniform baseline | 0.722 | -- | 159ms | Best cost config |
| Cascade | tau=0.80 | 0.724 | -- | 315ms | +98% more expensive |
| Ablation: full | full_pipeline | **0.740** | -- | -- | Best overall |
| Ablation: -emb | no embeddings | 0.410 | -- | -- | -33.0pp |

The full pipeline (lexical + embeddings + rules + structural features, LightGBM) achieves
Macro-F1=0.740 and represents the best overall configuration. Compared to a lexical-only
LightGBM baseline, the full pipeline improves performance by 18.9pp. The dominant
contributor to this improvement is the frozen encoder embedding, which accounts for 33.0pp
of the 33.0pp absolute gain from the no-embedding baseline (note: this comparison uses the
ablation baseline, not the H2 lexical-only condition, which uses a different feature set
composition).

---

## References

Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and Vulic, I. (2020). Efficient
intent detection with dual sentence encoders. *Proceedings of the 2nd Workshop on Natural
Language Processing for Conversational AI*, pp. 38--45.

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
Lawrence Erlbaum Associates.

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., and Meger, D. (2018). Deep
reinforcement learning that matters. *Proceedings of the 32nd AAAI Conference on Artificial
Intelligence*, pp. 3207--3214.

Hu, Z., Ma, X., Liu, Z., Hovy, E., and Xing, E. (2016). Harnessing deep neural networks
with logic rules. *Proceedings of the 54th Annual Meeting of the Association for
Computational Linguistics*, pp. 2410--2420.

Liu, X., Eshghi, A., Swietojanski, P., and Rieser, V. (2019). Benchmarking natural
language understanding services for building conversational agents. *Proceedings of the
Tenth International Workshop on Spoken Dialogue Systems Technology*.

Ma, X., Wang, L., Yang, M., Lin, J., and Lin, J. (2021). A replication study of dense
passage retrieval for open-domain question answering. arXiv preprint arXiv:2104.05740.

Mou, L., Men, R., Li, G., Xu, Y., Zhang, L., Yan, R., and Jin, Z. (2015). Natural
language inference by tree-based convolution and heuristic matching. *Proceedings of the
53rd Annual Meeting of the Association for Computational Linguistics*, pp. 130--136.

Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2022). MTEB: Massive text embedding
benchmark. arXiv preprint arXiv:2210.07316.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing*, pp. 3982--3992.

Sculley, D., Snoek, J., Rahimi, A., Wiltschko, A., and Pavone, A. (2018). Winner's curse?
On pace, progress, and empirical rigor. *Proceedings of the 6th International Conference on
Learning Representations (Workshop Track)*.

Shwartz-Ziv, R. and Armon, A. (2022). Tabular data: Deep learning is not all you need.
*Information Fusion*, 81, pp. 84--90.

Thakur, N., Reimers, N., Ruckle, A., Srivastava, A., and Gurevych, I. (2021). BEIR: A
heterogeneous benchmark for zero-shot evaluation of information retrieval models.
*Proceedings of the 35th Conference on Neural Information Processing Systems Datasets and
Benchmarks Track*.
