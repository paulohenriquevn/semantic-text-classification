# Chapter 1 — Introduction

## Abstract of Chapter

This chapter establishes the research context, formulates the central research problem, derives four falsifiable hypotheses, and states the dissertation's objectives and contributions. We investigate whether a hybrid cascaded architecture that combines lexical retrieval (BM25), semantic retrieval (dense embeddings from a frozen encoder), supervised classification over heterogeneous features, and a deterministic rule engine can improve conversation intent classification and retrieval quality over isolated paradigms while preserving operational explainability. Our experiments are conducted on a publicly available PT-BR customer service corpus of 2,122 conversations across 8 intent classes, following a rigorous data audit protocol. We report both confirmations and refutations with full statistical transparency.

---

## 1.1 Context and Motivation

### 1.1.1 The Scale of Conversational Data in Brazilian Contact Centers

Customer service operations in Brazil generate one of the largest bodies of natural language text produced in the country. Large-scale operations in telecommunications, financial services, retail, and healthcare individually process between five and fifteen million conversations per month across voice, chat, email, and social media channels (Associação Brasileira de Telesserviços, 2023). Taken collectively, this volume constitutes a significant corpus of authentic spoken and written Brazilian Portuguese — and one of the most systematically under-exploited.

The informational loss in this context is not incidental; it is structural. Industry estimates consistently indicate that fewer than five percent of customer service conversations receive any form of analysis beyond the manual post-call disposition entered by the agent — a categorization widely recognized as inconsistent, coarse-grained, and distorted by the pressure to minimize average handling time (AHT). The remaining ninety-five percent of conversations persists as unstructured raw data, inaccessible to search, classification, audit, or trend analysis at scale. The result is an operational paradox: organizations invest heavily in capturing conversations (telephony infrastructure, recording systems, omnichannel platforms) while extracting a negligible fraction of the informational value those conversations contain.

This gap between data volume and analytical utilization is not merely an efficiency problem. It has concrete operational consequences across at least four dimensions:

**Compliance and regulatory risk.** In regulated sectors — telecommunications (Anatel), financial services (Bacen, SUSEP), and supplemental health insurance (ANS) — regulators increasingly require demonstrable evidence that agents follow mandated disclosure scripts, that complaints are registered and resolved within statutory deadlines, and that prohibited sales practices are not occurring. Manual audit can cover only a small sample of interactions. An automated system capable of detecting mentions of regulatory bodies (Procon, Reclame Aqui, Anatel), non-disclosure of fees, or agent script deviations — and producing auditable evidence for each detection — would directly address a concrete regulatory compliance need.

**Customer retention and churn prediction.** Churn signals in customer service conversations are rarely explicit in a single utterance. A customer who will ultimately cancel a subscription typically traverses a multi-turn trajectory: an unresolved technical complaint, followed by a request for a supervisor, followed by a direct cancellation threat. Classifying individual turns in isolation misses this trajectory. A system capable of modeling multi-turn context and detecting escalation patterns enables proactive intervention before the cancellation request is made — precisely the operational capability that contact center managers most value and that current tools most consistently fail to deliver.

**Quality assurance and operational consistency.** Agent quality monitoring at scale requires the ability to classify every conversation against a consistent taxonomy of contact reasons. Current practice — agent self-reported disposition codes — is unreliable: agents from the same operation apply the same taxonomy differently, under-report complaint contacts, and systematically miscategorize ambiguous interactions. Automated classification with consistent criteria applied uniformly across all conversations is not merely a convenience; it is the prerequisite for any meaningful operational analytics.

**Customer experience (CX) research and training.** The ability to search for conversations similar to a given exemplar — "find me all conversations where the customer mentioned a specific billing error and was subsequently offered a discount" — enables both research and training workflows that are currently performed through manual labeling. Retrieval quality directly determines the utility of these workflows.

### 1.1.2 Limitations of Current Approaches

The analytical tools available for large-scale conversation analysis exhibit fundamental limitations that motivate the present research.

**Shallow single-turn intent classification.** The majority of commercial conversation intelligence platforms operate on individual turns in isolation, without modeling conversational context. This design choice reflects the historical dominance of chatbot intent detection frameworks, which are architecturally single-turn, over the more complex task of multi-turn conversation understanding. The consequence is a systematic inability to capture the structural patterns that define many high-value contact reasons: escalation sequences, objection-after-offer patterns, topic shifts, and the distributed emergence of intent across multiple turns. A customer who says "I want to know my balance" in turn 1 and "that's way more than I expected, this can't be right" in turn 3 and "I want to talk to a supervisor" in turn 5 is exhibiting a complaint-and-escalation pattern that no single-turn classifier can detect.

**The lexical-or-semantic false dichotomy.** Search over customer service conversations has historically defaulted to one of two approaches: lexical matching (BM25, TF-IDF, keyword search) or semantic search (dense vector retrieval over embeddings). Both have well-documented limitations in the conversational domain. Lexical approaches excel at exact term matching — product codes, plan names, regulatory keywords, unique vocabulary — but fail to capture paraphrases and implicit intent: a customer saying "I don't want this service anymore" will not be retrieved by a search for the keyword "cancelamento." Semantic approaches using pre-trained dense encoders generalize across lexical variations and paraphrases, but are computationally more expensive, less interpretable, and surprisingly brittle in specialized domains where technical vocabulary is dense and consistent — precisely the conditions under which lexical approaches perform best. Harris (2025) provided direct evidence of this: BM25 outperformed off-the-shelf semantic embeddings on structured medical document classification, a result that challenges the assumption of semantic universality. The appropriate response is not to choose one paradigm but to understand the conditions under which each excels and to build a system capable of leveraging both.

**Absence of explainability and auditability.** Systems based exclusively on statistical machine learning models produce predictions without traceable evidence. When a neural classifier labels a conversation as "churn risk," there is no mechanism by which a quality analyst or compliance auditor can verify why that classification was made — which words triggered it, which contextual pattern it matched, what confidence threshold was applied. In regulated domains, this opacity is an operational impediment. Decisions based on opaque models cannot be audited, challenged, or explained to consumers as required by growing regulatory frameworks, including Brazil's Lei Geral de Proteção de Dados (LGPD) and sector-specific disclosure requirements. A system intended for production deployment in regulated contact center environments must produce evidence alongside every prediction.

**Uniform computational cost.** Traditional NLP pipelines apply the same processing depth to every conversation regardless of complexity. A simple greeting interaction ("Hi, I'd like to check my balance") receives the same computational investment as a complex multi-turn retention negotiation with twenty turns across multiple topics. This uniformity is inefficient in any large-scale deployment, where the cost per inference is a concrete operational constraint. The principle of cascaded inference — applying progressively more expensive processing stages and resolving simple cases early — is established in computer vision (Viola and Jones, 2001) and information retrieval (Liu et al., 2011), but its application to conversational NLP classification remains under-studied.

### 1.1.3 The Research Opportunity

The limitations enumerated above define a coherent research opportunity: to investigate whether a single architecture can simultaneously address shallow intent modeling (through multi-level representations that capture turn, window, and conversation context), the lexical-semantic trade-off (through hybrid retrieval with score fusion), the explainability gap (through a deterministic rule engine with traceable evidence), and the computational cost problem (through cascaded inference with confidence-based early exit). No existing work has combined all four approaches in the conversational domain.

This dissertation presents TalkEx, a Conversation Intelligence Engine that operationalizes this integration. TalkEx is a complete NLP pipeline — from raw text ingestion through turn segmentation, context window construction, multi-level embedding generation, hybrid retrieval, supervised classification, and a semantic rule engine based on a domain-specific language compiled to an abstract syntax tree — implemented as a production-quality software artifact with 170 source files and approximately 1,900 unit and integration tests.

The key design insight motivating TalkEx is that the four approaches are not in tension but complementary. Embeddings capture semantic generalization; lexical features capture exact-match discriminative signals; supervised classifiers with heterogeneous features combine both families of signal; and deterministic rules provide predictable, auditable decision coverage for high-stakes cases. Cascaded inference controls the computational cost of combining them. The empirical question — whether this combination produces measurable improvements over isolated paradigms in a realistic conversational dataset — is the subject of this dissertation.

---

## 1.2 Research Problem

### 1.2.1 Central Research Question

The foregoing analysis motivates the following central research question:

> **Can a hybrid cascaded architecture that integrates lexical retrieval, frozen-encoder semantic representations, supervised classification over heterogeneous features, and a deterministic rule engine improve classification and retrieval quality over isolated paradigms in the domain of Brazilian Portuguese customer service conversations, while preserving operational explainability?**

This question is deliberately scoped. It asks whether the combination *can* improve over isolated paradigms — an investigative framing — rather than claiming the combination *does* improve universally. The experimental results in this dissertation confirm some components of the combination and call others into question; honest framing of the question at the outset reflects the epistemic status of the answers.

### 1.2.2 Three Fundamental Tensions

The research question encapsulates three fundamental tensions that pervade natural language processing applied to conversational data:

**Tension 1: Lexical coverage versus semantic generalization.** Lexical and semantic retrieval operate on complementary signals. BM25 is precise for exact terms, technical vocabulary, and regulatory keywords — critical in contact center data where plan names, product codes, and compliance phrases appear verbatim. Dense semantic embeddings generalize across lexical variants and paraphrases — essential for capturing intent when customers express the same underlying need in highly variable language. Rayo et al. (2025) demonstrated that a hybrid combination (BM25 plus fine-tuned embedding encoder) outperformed both isolated approaches on regulatory text retrieval, achieving Recall@10 of 0.833 against 0.761 for BM25 alone and 0.810 for semantic search alone. However, this result was obtained on long, formal regulatory texts — European obligation documents from the ObliQA dataset — a substantially different register from the short, informal, multi-turn PT-BR conversations examined in this work. The transferability of the hybrid advantage to the conversational domain is a non-trivial empirical question.

**Tension 2: Turn-level granularity versus contextual multi-turn modeling.** Customer service conversations have a natural hierarchical structure: individual turns, context windows (sequences of adjacent turns), and the complete conversation. Each level captures qualitatively different information. A single turn captures a local intent ("I want to cancel"). A context window of five to ten turns captures multi-turn dependencies — an escalation sequence, a topic transition, an objection pattern. The full conversation captures the dominant goal and the interaction's outcome. Classifiers operating at a single granularity lose structural information. Lyu et al. (2025) demonstrated that attention-based pooling over token-level representations improves classification (F1 of 0.89 versus 0.86 for BERT base), but their evaluation was conducted on the AG News short-text dataset — one sentence per sample — not on multi-turn conversations where contextual modeling is qualitatively more important. The value of multi-level conversational representations in the intent classification setting requires direct experimental investigation.

**Tension 3: Statistical learning versus deterministic auditability.** Machine learning models learn patterns from data but provide no formal guarantees and no interpretable decision trace. Deterministic rules provide predictability, auditability, and zero-cost inference for covered cases, but require explicit knowledge encoding and do not generalize to uncovered patterns. Huang and He (2025) demonstrated that large language models can perform competitive text classification without fine-tuning by leveraging clustering as a classification paradigm — but at the cost of requiring online LLM inference (prohibitive at the scale of millions of conversations per month) and producing no traceable evidence per decision. Chiticariu et al. (2013) argued that rule-based systems complement statistical approaches precisely in settings requiring transparency and control. No prior work integrates deterministic rules with traceable evidence into a hybrid retrieval and classification pipeline for multi-turn conversational data.

### 1.2.3 Domain-Specific Challenges

Beyond the three fundamental tensions, the domain of Brazilian Portuguese customer service conversations imposes specific challenges that amplify the difficulty of the problem:

**Noise from automatic speech recognition (ASR).** Voice interactions — the dominant channel in Brazilian contact centers — pass through ASR systems that introduce systematic errors: substituted words, incorrect turn boundaries, missing punctuation and capitalization, and name entity transcription errors. A classification and retrieval system must be robust to this noise, requiring aggressive text normalization without loss of discriminative signals. The TalkEx normalization pipeline applies Unicode NFKD normalization with accent stripping and lowercase transformation — critical for Brazilian Portuguese, where the presence or absence of diacritics is inconsistent in both ASR output and customer-written text ("não" versus "nao", "cancelamento" versus "cancelamênto").

**Multi-turn intent dependencies.** Customer intent frequently does not reside in a single turn. Intent patterns that define the highest-value contact reason categories — churn escalation, fraud reporting, regulatory complaint — are inherently multi-turn phenomena. A classifier that processes turns in isolation systematically misses these patterns. The sliding context window approach implemented in TalkEx, with configurable window size and stride, provides the contextual span needed to capture these dependencies, at the cost of introducing configuration choices (window size, stride, speaker alignment, recency weighting) that affect classification quality.

**Linguistic variability of Brazilian Portuguese.** PT-BR as spoken in contact centers exhibits high variability: abbreviations ("vc", "td", "blz", "pq"), regional expressions, inconsistent diacritics, colloquialisms, and code-switching with English terms ("upgrade", "app", "feedback"). This variability affects both the lexical retrieval component — where consistent normalization determines whether "cancelamento" and "cancelámento" are treated as the same token — and the classification component, where vocabulary distributional patterns differ from the formal Portuguese text on which most available multilingual models were primarily trained.

**Simultaneous requirements for quality, explainability, and cost.** In production environments processing millions of conversations, a system that merely classifies accurately is insufficient. It must also explain its classifications in terms that human analysts can verify and act on, and it must do so at a computational cost compatible with the throughput requirements of large-scale deployment. These three requirements frequently conflict: more sophisticated models improve accuracy but increase cost and reduce interpretability. The cascade architecture is designed to manage this trade-off, but doing so effectively requires empirical calibration.

---

## 1.3 Hypotheses

From the research question and the analysis of the literature, we derive four falsifiable hypotheses. Each is stated with quantitative success criteria that were specified before the experiments were conducted. The criteria are strict enough to permit refutation — and, as the results will show, two of the four hypotheses were not confirmed under these criteria.

### H1 — Hybrid Retrieval Outperforms Isolated Retrieval Paradigms

> The hybrid retrieval system (BM25 plus approximate nearest neighbor search over frozen-encoder embeddings, with parametric score fusion) achieves strictly higher Mean Reciprocal Rank (MRR) than both the BM25-only baseline and the ANN-only baseline on the PT-BR customer service corpus, with statistical significance at α = 0.05 under the Wilcoxon signed-rank test over per-query MRR scores.

This hypothesis operationalizes the complementarity thesis: that lexical and semantic signals are sufficiently different in their failure modes that combining them produces a measurable advantage over either alone. The success criterion is deliberately specific about the statistical test (Wilcoxon signed-rank, a non-parametric test appropriate for paired comparisons over non-normally distributed per-query scores) and the required significance level. H1 is confirmed only if the best hybrid configuration beats *all* isolated baselines at the specified significance level; a result significant against one baseline but not the other would constitute a partial result.

This hypothesis also serves a methodological purpose: it instantiates the principle that semantic approaches should always be benchmarked against a strong BM25 baseline before being accepted as improvements (Robertson et al., 1996; Luan et al., 2021). In the domain under study, BM25 is a non-trivial baseline because customer service conversations contain discriminative lexical markers — cancellation vocabulary, complaint vocabulary, greeting phrases — that BM25 captures directly and cheaply.

**Success criterion:** Best hybrid configuration MRR > BM25 MRR and > ANN MRR, with Wilcoxon signed-rank p < 0.05 for both comparisons, computed over per-query MRR scores on the held-out test set.

### H2 — Combined Lexical and Embedding Features Outperform Lexical-Only Features

> A supervised classifier (LightGBM) trained on combined lexical features and dense embedding features extracted from a frozen multilingual encoder achieves strictly higher Macro-F1 than the best classifier trained on lexical features alone, with statistical significance at α = 0.05, evaluated over five random seeds on the held-out test set.

This hypothesis tests the additive value of dense embeddings over lexical representations in a supervised classification setting. It operationalizes the "embeddings represent, classifiers decide" principle (AnthusAI, 2024): embeddings are used as feature inputs to a supervised classifier, not as a direct similarity-based classifier. This design choice preserves the discriminative capacity of supervised learning while leveraging the semantic generalization of pre-trained representations. The hypothesis specifically requires that the combined system outperform *all* lexical-only configurations (including Logistic Regression, LightGBM, and MLP trained on lexical features alone), not merely the weakest lexical baseline.

The choice of Macro-F1 as the primary metric reflects the multi-class nature of the problem and the importance of per-class performance: Macro-F1 weights each class equally regardless of frequency, ensuring that performance on minority classes (such as "saudacao" and "compra" in the post-audit dataset) is not hidden by aggregate accuracy on majority classes.

**Success criterion:** Best combined (lexical + embedding) configuration Macro-F1 > best lexical-only configuration Macro-F1, with Wilcoxon signed-rank p < 0.05, computed over five random seed runs on the held-out test set.

### H3 — Deterministic Rules as Features Improve Classification over ML-Only

> Adding deterministic rule activations as features to the supervised classification pipeline (rules-as-feature integration) achieves strictly higher Macro-F1 than the ML-only baseline (classifier without rule features), with statistical significance at α = 0.05, as evaluated over per-instance predictions on the held-out test set.

This hypothesis addresses the complementarity of deterministic rules and statistical classifiers. The "rules-as-feature" integration strategy — in which rule activation outputs are included as binary features in the classifier's feature vector — is a soft integration approach that allows the classifier to learn when to trust rule signals versus when to override them, as opposed to a hard override strategy (ML+Rules-override) that enforces rule outputs unconditionally. The hypothesis expects the soft integration to be measurably superior to no integration.

H3 also serves the dissertation's explainability objective: even a small positive effect of rules on classification accuracy, combined with the traceable evidence that every rule execution produces, would support the argument that deterministic rules provide value beyond what can be captured by the statistical pipeline alone.

**Success criterion:** ML+Rules-feature Macro-F1 > ML-only Macro-F1, with Wilcoxon signed-rank p < 0.05, computed over per-instance predictions on the held-out test set.

### H4 — Cascaded Inference Reduces Computational Cost with Bounded Quality Loss

> A two-stage cascaded inference pipeline — applying a lightweight classifier (lexical features only) in Stage 1 and the full feature classifier (lexical + embedding + rule features) in Stage 2, with confidence-based early exit — achieves simultaneously: (a) at least 40% reduction in average computational cost per window compared to the uniform full-pipeline baseline, and (b) Macro-F1 degradation of less than 2 percentage points compared to the uniform baseline.

This hypothesis tests the central premise of cascaded inference: that early exit at lower confidence thresholds allows the system to process a meaningful fraction of cases cheaply without proportionally degrading classification quality. The criterion is deliberately conjunctive — both the cost reduction and the quality preservation conditions must be satisfied simultaneously for H4 to be confirmed. A configuration that achieves 40% cost reduction with 5% F1 degradation does not confirm H4; nor does a configuration that achieves 0.5% F1 degradation with 5% cost reduction.

The 40% threshold reflects the operational economics of large-scale contact center deployment: a cost reduction smaller than this would not justify the added architectural complexity of maintaining and calibrating two separate classifiers. The 2 percentage point F1 degradation bound ensures that the efficiency gain is not achieved by sacrificing classification accuracy on a scale that would have operational consequences.

This is the most demanding of the four hypotheses, for a structural reason noted before the experiments began: if the Stage 1 lightweight classifier operates on the same data as Stage 2, and if Stage 2 performs substantially better than Stage 1 (as it must, to motivate the cascade design), then any meaningful fraction of cases resolved by Stage 1 introduces classification errors. The hypothesis predicts that these errors are bounded within the acceptable limit.

**Success criterion:** Exists at least one confidence threshold configuration such that: average cost per window < 0.6 × uniform baseline cost AND Macro-F1 > uniform baseline Macro-F1 − 0.02, both evaluated on the held-out test set.

---

## 1.4 Objectives

### 1.4.1 General Objective

Design, implement, and empirically evaluate a hybrid cascaded architecture for intent classification and retrieval in Brazilian Portuguese customer service conversations, integrating lexical retrieval, dense semantic retrieval, supervised classification with heterogeneous features, and a deterministic rule engine with traceable evidence, with transparent reporting of both confirmations and refutations.

### 1.4.2 Specific Objectives

**SO1 — Hybrid Retrieval System.** Implement a hybrid retrieval system that combines BM25 with accent-aware normalization for PT-BR and approximate nearest neighbor search over embeddings from a frozen multilingual encoder (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensions), with parametric score fusion strategies (weighted linear combination and Reciprocal Rank Fusion), and evaluate its retrieval performance against isolated lexical and semantic baselines on MRR, Precision@K, and nDCG metrics on the post-audit corpus.

**SO2 — Multi-Level Conversational Representations.** Design and implement a sliding context window approach that constructs multi-level conversational representations at turn, window, and conversation granularity, and evaluate the contribution of each feature family (embedding features, lexical features, rule features, structural features) through systematic ablation on the post-audit corpus.

**SO3 — Semantic Rule Engine with Traceable Evidence.** Design and implement a semantic rule engine based on a domain-specific language (DSL) compiled to an abstract syntax tree (AST), supporting lexical, semantic, structural, and contextual predicates with short-circuit evaluation ordered by predicate cost, where each rule execution produces traceable evidence metadata (matched terms, similarity scores, thresholds applied, model version), and evaluate the rule engine's contribution to classification performance in both soft (rules-as-features) and hard (rules-as-override) integration modes.

**SO4 — Cascaded Inference Analysis.** Design and evaluate a two-stage cascaded inference pipeline with confidence-based early exit, analyzing the trade-off between computational cost reduction and classification quality degradation across a range of confidence thresholds, and identifying whether any configuration achieves the conjunctive criterion of H4 (≥40% cost reduction with <2pp F1 degradation).

---

## 1.5 Contributions

This dissertation presents four contributions. We scope each contribution carefully to reflect what the evidence supports.

### C1 — A Complete, Production-Quality Hybrid NLP Pipeline for Conversational Data

TalkEx is a fully implemented, documented, and tested NLP pipeline for conversation intent classification and retrieval. The pipeline comprises: data ingestion with multi-source support; turn segmentation and text normalization with PT-BR-specific accent handling; sliding context window construction with configurable parameters; multi-level embedding generation using a frozen multilingual encoder; BM25 indexing with accent-aware normalization; approximate nearest neighbor indexing using FAISS; hybrid retrieval with score fusion; supervised classification with LightGBM, Logistic Regression, and MLP over heterogeneous features; and a semantic rule engine with DSL compilation to AST. The system is implemented in Python 3.11 with strict Pydantic data models, 170 source files, and approximately 1,900 unit and integration tests.

The architectural contribution is not merely the sum of these components but the specific design decisions that govern their integration: the multi-level representation scheme that maps conversations to feature vectors at turn, window, and conversation granularity; the score fusion parameterization that enables systematic comparison of linear and rank-based fusion strategies; and the frozen-encoder design choice that decouples semantic representation quality from the availability of domain-specific training data — a particularly important consideration for low-resource languages and specialized domains where fine-tuning data is scarce. These design decisions are documented in four Architecture Decision Records (ADRs) included as part of the dissertation artifacts.

To our knowledge, no prior open-source system combines hybrid retrieval, multi-level conversational embeddings, supervised classification with heterogeneous features, and a DSL-based rule engine in a single, tested, production-ready implementation for Brazilian Portuguese conversational data.

### C2 — A DSL-Based Semantic Rule Engine with Per-Decision Evidence

The TalkEx rule engine introduces a domain-specific language for expressing classification and detection rules, compiled to an abstract syntax tree evaluated with short-circuit execution ordered by predicate cost. The rule engine supports four families of predicates: lexical (contains, regex, BM25 score threshold), semantic (embedding similarity, intent score), structural (speaker role, turn position, conversation channel), and contextual (pattern repetition within a window, temporal sequencing of events). Each predicate evaluation produces structured evidence metadata — the specific text matched, the similarity scores computed, the thresholds applied, and the model version used — ensuring full auditability of every rule-based decision.

This contribution directly addresses the explainability gap identified in the problem formulation. Unlike statistical classifiers, which produce probability scores without interpretable decision traces, and unlike LLM-based classification approaches (Huang and He, 2025), which provide no per-instance evidence, the TalkEx rule engine produces a complete decision audit trail for every classification it influences. In compliance-sensitive deployment contexts, this property is operationally necessary rather than merely desirable.

The contribution is scoped honestly: in our experiments, the rule engine was evaluated with a limited set of rules primarily targeting lexical patterns in two high-frequency intent classes. The experiments therefore test a necessary but not sufficient condition for the full design: that even a small, predominantly lexical ruleset produces measurable classification benefits when integrated as soft features. The broader expressiveness of the rule engine — semantic similarity predicates, contextual sequencing predicates — is demonstrated through the implementation but not fully evaluated in the experimental results presented here.

### C3 — Empirical Evidence on Hybrid Retrieval and Feature Combination in PT-BR Conversational Classification

This dissertation contributes systematic empirical evidence on the relative effectiveness of lexical retrieval, semantic retrieval, and their combination in the PT-BR conversational classification domain. The post-audit experimental results — based on a 2,122-conversation corpus with 8 intent classes, evaluated over five random seeds, with Wilcoxon signed-rank statistical tests — provide the following findings:

- Hybrid retrieval (Hybrid-LINEAR-α=0.30, MRR=0.853) statistically significantly outperforms both BM25 (MRR=0.835, p=0.017) and ANN-only (MRR=0.824, p=0.030) retrieval, confirming H1.
- Combined lexical and embedding features (LightGBM, Macro-F1=0.722) statistically significantly outperform lexical-only features (LightGBM, Macro-F1=0.334), confirming H2 with a margin of 38.8 percentage points.
- An ablation study isolates the contribution of each feature family: embeddings contribute +33.0 percentage points to Macro-F1 (the dominant component), lexical features contribute an additional +2.9 percentage points, rule features contribute +1.8 percentage points, and structural features contribute +1.3 percentage points.
- Rules-as-features integration produces a positive direction (+1.8pp Macro-F1) that does not reach statistical significance (p=0.131), leaving H3 inconclusive.
- No cascade configuration achieves the conjunctive criterion of H4: all cascade configurations increase measured cost rather than reducing it, refuting H4.

These findings contribute both positive and negative evidence. The refutation of H4 and the inconclusiveness of H3 are reported with the same prominence as the confirmations of H1 and H2. Negative and inconclusive results are informative for the research community and provide a more accurate representation of the current state of the architecture than a selective reporting of confirmations.

### C4 — A Data Audit Protocol for Synthetic Conversational Corpora

The experimental results in this dissertation were obtained on a corpus that underwent a rigorous two-stage audit before being used for hypothesis testing. The audit protocol includes: exact deduplication with threshold 0.97 cosine similarity; near-duplicate detection with threshold 0.92; few-shot contamination detection between train and test splits; systematic human review of ambiguous class assignments with a confirmed label acceptance rate of ≥96.7%; and removal of an entire intent class ("outros") found to be too heterogeneous for reliable supervised classification.

The protocol reduced the corpus from 2,257 pre-audit records (9 classes) to 2,122 post-audit records (8 classes) and changed the experimental conclusions substantially: under the pre-audit data, H1 was not statistically significant (p=0.103); under the post-audit data, H1 is confirmed (p=0.017). This change in conclusion under otherwise identical experimental conditions illustrates the concrete impact of data quality on NLP evaluation results.

We document the full audit protocol in Chapter 5 (Methodology) and make the audit code and decision trail available as supplementary material. This protocol is reusable for any researcher working with synthetic or semi-synthetic conversational corpora and contributes to the reproducibility infrastructure of the field.

---

## 1.6 Dissertation Organization

The remainder of this dissertation is organized into six chapters.

**Chapter 2 — Theoretical Foundations** establishes the conceptual framework underlying the work. We present the representation of text as dense vectors and the progression from bag-of-words to sentence transformers; the BM25 lexical retrieval model with its mathematical formulation; approximate nearest neighbor search for dense vector retrieval; hybrid retrieval strategies and score fusion methods; supervised text classification with heterogeneous feature sets; multi-turn conversational structure and context window modeling; domain-specific language design and abstract syntax tree evaluation; and the principle of cascaded inference in NLP systems. This chapter is didactic — it provides the foundational vocabulary for the contributions described in subsequent chapters.

**Chapter 3 — Related Work** positions this research within the state of the art. We critically analyze the most relevant prior work: Harris (2025) on lexical versus semantic search in structured domains; Rayo et al. (2025) on hybrid retrieval for regulatory text; Huang and He (2025) on LLM-based text clustering as classification; Lyu et al. (2025) on attention mechanisms for text classification; the AnthusAI semantic text classification study on embeddings as features for supervised classifiers; and the literature on conversation intelligence in contact center settings, including commercial platforms (Observe.AI, CallMiner, Verint) and academic systems (BERTaú). We conclude the chapter with an explicit positioning table identifying the gap that this dissertation fills: no prior work combines hybrid retrieval, multi-level conversational representations, supervised classification with heterogeneous features, and a DSL-based deterministic rule engine in a single evaluated system for PT-BR conversational data.

**Chapter 4 — Proposed Architecture: TalkEx** describes the system in sufficient technical detail for reproduction. We present the complete pipeline with component boundaries and data flow; the conversational data model (Conversation, Turn, ContextWindow, EmbeddingRecord, Prediction, RuleExecution as frozen strict Pydantic models); the text normalization module with PT-BR-specific design; the multi-level representation scheme; the hybrid retrieval system with score fusion variants; the supervised classification pipeline with feature engineering; the rule engine (DSL grammar, parser, AST representation, predicate handlers, short-circuit executor, evidence generation); the cascade inference logic; and the architectural decisions documented in ADR-001 through ADR-004.

**Chapter 5 — Experimental Design and Methodology** details the experimental protocol for each hypothesis. We describe the post-audit corpus statistics and the audit protocol; the experimental splits and seed strategy; the evaluation metrics for retrieval (MRR, Precision@K, nDCG), classification (Macro-F1, per-class F1), and cascade efficiency (cost per window, cost reduction percentage); the experimental protocol for each of H1 through H4; the ablation study design; the statistical analysis approach (Wilcoxon signed-rank with bootstrap confidence intervals); and the threats to validity, including the synthetic nature of the corpus, the single-domain evaluation, the absence of fine-tuned encoder baselines, and the determinism artifact in multi-seed evaluation.

**Chapter 6 — Results and Analysis** presents the experimental results for each hypothesis and the ablation study, analyzes per-class performance, discusses the cascade results and why H4 was refuted, and contextualizes the findings relative to the related work. We include a per-class error analysis that identifies the two intent classes (saudacao and compra) where all configurations perform below 0.50 F1, and discuss the extent to which this may reflect properties of the synthetic training data.

**Chapter 7 — Conclusion** synthesizes the confirmed and refuted findings, articulates the limitations of the current work, and identifies the highest-priority directions for future investigation: fine-tuning of the embedding encoder for the conversational domain; expansion of the rule engine to include semantic similarity and contextual sequencing predicates; leave-one-domain-out evaluation to test generalization across domains; cross-validation in place of fixed splits for tighter confidence intervals; and calibration metrics (Brier score, Expected Calibration Error) to support threshold-based abstention in production deployment.

---

## References for This Chapter

The following references are cited in this chapter. Full bibliographic details are provided in the dissertation reference list.

Associação Brasileira de Telesserviços. (2023). *Relatório Anual do Setor de Contact Center*. ABT.

AnthusAI. (2024). *Semantic Text Classification: Text Classification with Various Embedding Techniques*. GitHub repository. Retrieved from https://github.com/AnthusAI/semantic-text-classification.

Chiticariu, L., Li, Y., and Reiss, F. R. (2013). Rule-based information extraction is dead! Long live rule-based information extraction systems! In *Proceedings of EMNLP 2013* (pp. 827–832).

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186).

Harris, L. (2025). Comparing lexical and semantic vector search methods when classifying medical documents. arXiv preprint arXiv:2505.11582v2.

Huang, C., and He, G. (2025). Text clustering as classification with LLMs. In *Proceedings of SIGIR-AP 2025*. arXiv preprint arXiv:2410.00927v3.

Liu, T.-Y. (2011). *Learning to Rank for Information Retrieval*. Springer.

Lyu, N., Wang, Y., Chen, F., and Zhang, Q. (2025). Advancing text classification with large language models and neural attention mechanisms. arXiv preprint arXiv:2512.09444v1.

Rayo, J., de la Rosa, R., and Garrido, M. (2025). A hybrid approach to information retrieval and answer generation for regulatory texts. In *Proceedings of COLING 2025*. arXiv preprint arXiv:2502.16767v1.

Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019* (pp. 3982–3992).

Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., and Gatford, M. (1996). Okapi at TREC-4. In *Proceedings of the Fourth Text REtrieval Conference (TREC-4)* (pp. 73–96).

Viola, P., and Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. In *Proceedings of CVPR 2001* (Vol. 1, pp. I–511–I–518).

---

*Chapter word count (approximate): 5,400 words, equivalent to approximately 18 pages at standard double-spaced ACL format.*
