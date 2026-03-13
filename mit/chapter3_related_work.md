# Chapter 3: Related Work

This chapter positions TalkEx within the landscape of prior research across seven streams of literature that collectively define the design space and the contribution boundaries of this thesis. Section 3.1 reviews hybrid retrieval systems that combine lexical and semantic signals. Section 3.2 examines embedding-based classification and the paradigm of using pre-trained representations as features for supervised learners. Section 3.3 surveys rule-based and hybrid NLP systems that integrate deterministic logic with statistical methods. Section 3.4 covers conversational intent classification, with emphasis on multi-turn modeling. Section 3.5 discusses cascaded and cost-aware inference architectures. Section 3.6 reviews the state of Portuguese-language NLP resources and the specific challenges of PT-BR conversational data. Section 3.7 synthesizes the prior work into a comparative positioning table and identifies the precise gap that this thesis addresses.

Each section follows a consistent analytical structure: we present the key contributions, assess their strengths and limitations with respect to the conversational PT-BR domain, and identify the specific gap that motivates TalkEx's design choices. Cross-references to subsequent chapters indicate where each design choice is implemented (Chapter 4) and evaluated (Chapter 6).

---

## 3.1 Hybrid Retrieval Systems

### 3.1.1 The Complementarity Thesis

The central premise of hybrid retrieval is that lexical and semantic signals fail in complementary ways: lexical methods miss paraphrases and implicit intent; semantic methods miss exact terms, codes, and domain-specific vocabulary that appears verbatim in queries and documents. Combining both should therefore recover a strictly larger set of relevant results than either alone. This premise has been validated across multiple benchmarks, but the conditions under which complementarity holds — and the magnitude of the gain — are domain-dependent in ways that the literature has only recently begun to characterize.

### 3.1.2 SPLADE: Learned Sparse Representations

Formal, Piwowarski, and Clinchant (2021) introduced SPLADE (SParse Lexical AnD Expansion), a retrieval model that learns sparse representations from a masked language model. SPLADE maps each query and document to a high-dimensional sparse vector in the vocabulary space, where each dimension corresponds to a term and the weight is learned end-to-end via a sparsity-regularized objective. The key innovation is that the model learns to expand queries with semantically related terms that do not appear in the original text — a form of learned query expansion that bridges the gap between purely lexical and purely dense approaches.

SPLADE achieves competitive or superior performance to dense bi-encoders on the MS-MARCO passage ranking benchmark while retaining the interpretability advantage of sparse representations: the contribution of each term to the retrieval score is directly readable. The model also benefits from efficient inverted index implementations, enabling sub-millisecond retrieval at scale.

The limitations of SPLADE for the TalkEx context are threefold. First, the model requires substantial supervised training data (query-passage pairs with relevance judgments) that is unavailable for PT-BR conversational data. Second, SPLADE was evaluated exclusively on English web search and passage retrieval; its behavior on short, informal, multi-turn dialogue in a morphologically rich language is unknown. Third, the vocabulary-space representation assumes that semantic relationships can be captured through term expansion — an assumption that may be weaker for conversational data where intent is expressed through pragmatic patterns (tone, repetition, escalation sequences) rather than through vocabulary variation alone.

### 3.1.3 ColBERT: Late Interaction Dense Retrieval

Khattab and Zaharia (2020) proposed ColBERT, a dense retrieval architecture based on late interaction. Unlike bi-encoders that compress each text into a single vector, ColBERT retains the full sequence of token-level embeddings for both query and document, computing relevance as the sum of maximum cosine similarities between each query token and all document tokens. This MaxSim operator preserves fine-grained lexical matching within a dense representation framework, achieving a form of soft term matching that is more expressive than single-vector cosine similarity.

ColBERT achieves state-of-the-art retrieval quality on MS-MARCO and TREC Deep Learning benchmarks, demonstrating that late interaction captures matching patterns that pooled bi-encoders miss. The architecture has been extended in ColBERTv2 (Santhanam et al., 2022), which reduces storage requirements through residual compression while maintaining retrieval quality.

For TalkEx, ColBERT's primary limitation is computational cost: storing and searching token-level representations for millions of conversation windows requires substantially more storage and memory than the single-vector representation used in TalkEx (384 dimensions per window). In a production deployment processing millions of conversations, this cost difference is operationally significant. Additionally, ColBERT was developed and evaluated on English text; no multilingual late-interaction model with demonstrated effectiveness on PT-BR conversational data existed at the time of this writing.

### 3.1.4 DPR: Dense Passage Retrieval

Karpukhin et al. (2020) introduced Dense Passage Retrieval (DPR), the foundational bi-encoder architecture for dense retrieval. DPR trains two BERT encoders — one for queries, one for passages — using in-batch negatives and hard negative mining, optimizing for inner-product similarity between query and passage representations. The resulting dense vectors enable retrieval via approximate nearest-neighbor (ANN) search, capturing semantic equivalences that lexical methods cannot detect.

DPR demonstrated substantial improvements over BM25 on open-domain question answering benchmarks (Natural Questions, TriviaQA, WebQuestions), achieving Recall@20 gains of 9-19 percentage points. However, subsequent work revealed that DPR's advantage over BM25 is domain-dependent: on benchmarks with high lexical overlap between queries and passages (e.g., entity-centric questions), BM25 remains competitive or superior (Thakur et al., 2021). This finding is directly relevant to TalkEx, where customer service conversations contain discriminative lexical markers — cancellation vocabulary, product codes, greeting phrases — that BM25 captures efficiently.

The critical lesson from DPR for TalkEx is architectural rather than empirical: the bi-encoder framework, where queries and documents are encoded independently and matched via vector similarity, forms the basis of the ANN retrieval component in TalkEx's hybrid system. TalkEx adopts the bi-encoder paradigm but substitutes a frozen multilingual encoder (paraphrase-multilingual-MiniLM-L12-v2) for the task-specific fine-tuned encoders of DPR, trading task-specific optimization for multilingual generalization and zero annotation overhead.

### 3.1.5 Rayo et al. (2025): The Closest Methodological Precedent

Rayo, de la Rosa, and Garrido (2025) present the most directly comparable prior work to TalkEx in the retrieval dimension. Their system, presented at COLING 2025, constructs a hybrid information retrieval pipeline for the ObliQA regulatory compliance dataset — 27,869 questions extracted from 40 regulatory documents issued by the Abu Dhabi Global Markets financial authority.

The architecture combines BM25 for lexical retrieval with a fine-tuned Sentence Transformer (BAAI/bge-small-en-v1.5, expanded from 384 to 512 dimensions during fine-tuning) for semantic retrieval. Score fusion follows linear interpolation with empirically selected weight alpha = 0.65 for semantic and 0.35 for lexical. The quantitative results establish a clear hierarchy: BM25 baseline achieves Recall@10 = 0.761 and MAP@10 = 0.624; the semantic-only retriever reaches Recall@10 = 0.810, MAP@10 = 0.629; and the hybrid system achieves Recall@10 = 0.833, MAP@10 = 0.702. The hybrid dominates both isolated approaches, confirming complementarity.

The contribution is significant: Rayo et al. provide rigorous empirical evidence that hybrid retrieval outperforms isolated paradigms on a real-world specialized domain. The paper separates the contribution of fine-tuned embeddings from the contribution of fusion, compares multiple fusion weights, and reports per-metric results. The addition of a RAG component using GPT-3.5 Turbo extends the pipeline to end-to-end question answering.

However, several constraints bound the generalizability of this work to the conversational domain. First, the corpus consists of formal regulatory documents — structured, written in controlled legal English, with explicit obligation language and standardized terminology. The challenge of informal, colloquial, multi-turn spoken language is fundamentally different. Second, the system addresses passage retrieval only; it includes no classification, intent recognition, or rule-based post-processing. Third, the dataset is English monolingual with no evaluation in low-resource or non-English languages. Fourth, the input unit is a document passage, not a conversational turn or context window; the multi-turn structure of dialogue is absent.

TalkEx operationalizes the same fusion architecture in a significantly different domain — informal multi-turn PT-BR customer service conversations — and evaluates downstream impact on intent classification rather than passage retrieval. Where Rayo et al. use a fine-tuned domain-specific encoder, TalkEx uses a frozen multilingual encoder, enabling direct comparison between domain adaptation through fine-tuning and generalization through multilingual pre-training. As evaluated in Chapter 6, TalkEx's hybrid retrieval achieves MRR = 0.853 versus BM25-only MRR = 0.835 (p = 0.017), confirming that the complementarity observed by Rayo et al. transfers to the conversational PT-BR domain.

### 3.1.6 Harris et al. (2024): BM25 as a Non-Trivial Baseline

Harris (2024), in "Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents," provides a counterpoint to the assumption of semantic superiority. The study evaluates BM25 against off-the-shelf semantic embeddings (OpenAI text-embedding-3-small/large, all-MiniLM-L6-v2) on a medical document classification task consisting of 517 clinical records across 42 ICD-10 categories. The finding is unambiguous: BM25 achieves top-1 accuracy of 0.74, outperforming the best semantic embedding (0.67) by 7 percentage points.

Harris attributes this result to the lexical regularity of medical documentation: clinical records use consistent terminology, diagnostic codes, and standardized phrases where exact term matching is more discriminative than semantic generalization. The study also demonstrates that ensemble methods combining BM25 and semantic scores improve over BM25 alone by 1 percentage point (to 0.75), confirming that even in a lexically regular domain, semantic signals contribute marginal but measurable value.

The implications for TalkEx are direct. Customer service conversations occupy an intermediate position on the lexical-regularity spectrum: they contain discriminative exact terms (product names, cancellation vocabulary, regulatory keywords) alongside semantically variable expressions of intent (complaints, escalation patterns, satisfaction signals). Harris's finding reinforces TalkEx's design choice of hybrid retrieval rather than semantic-only retrieval, and motivates the ablation study (Chapter 6) that quantifies the relative contribution of lexical versus embedding features. The +2.9 percentage point contribution of lexical features in the TalkEx ablation is consistent with Harris's observation that lexical signals retain value even when semantic representations are available.

### 3.1.7 Gap: Conversational PT-BR Hybrid Retrieval

The hybrid retrieval literature is dominated by English-language evaluation on web search (MS-MARCO, BEIR), open-domain QA (Natural Questions), and formal document corpora (regulatory text, medical records). No prior work has evaluated hybrid BM25 + dense retrieval on multi-turn conversational data in Brazilian Portuguese. The conversational domain introduces specific challenges absent from document retrieval: short utterances with high anaphora density, speaker turn structure, topic drift within conversations, and informal register with ASR-induced noise. TalkEx provides the first such evaluation, as reported in Chapter 6.

---

## 3.2 Embedding-Based Classification

### 3.2.1 The "Embeddings Represent, Classifiers Decide" Paradigm

A fundamental design choice in any embedding-based NLP system is whether to use embeddings directly for prediction (e.g., via nearest-neighbor classification or cosine similarity thresholds) or to treat embeddings as feature inputs to a supervised classifier. The distinction is consequential: direct embedding use conflates representation quality with classification quality, while the decoupled approach allows the classifier to learn decision boundaries that compensate for representation weaknesses.

### 3.2.2 Sentence-BERT and the Foundation of Sentence Embeddings

Reimers and Gurevych (2019) introduced Sentence-BERT (SBERT), adapting the BERT architecture to produce fixed-size sentence embeddings via Siamese and triplet network structures trained on NLI and STS datasets. SBERT made it computationally feasible to use BERT for tasks requiring pairwise sentence comparison — semantic similarity, clustering, retrieval — by reducing the O(n^2) cross-encoding cost to O(n) independent encoding plus efficient similarity search.

The contribution of SBERT to the field was architectural and practical: it demonstrated that contrastive fine-tuning of a pre-trained language model produces sentence-level representations that transfer effectively across tasks with minimal adaptation. The resulting sentence-transformers library became the standard tool for generating text embeddings in downstream applications, including the multilingual variant (paraphrase-multilingual-MiniLM-L12-v2) used in TalkEx.

For TalkEx, SBERT's architecture provides the embedding backbone, but with a critical design divergence: where Reimers and Gurevych evaluated embeddings primarily for similarity and retrieval tasks, TalkEx uses the embeddings as input features to gradient-boosted classifiers, never as direct classifiers. This "embeddings represent, classifiers decide" separation, as articulated in the AnthusAI study discussed below, is a deliberate architectural choice that preserves the discriminative power of supervised learning while leveraging the semantic generalization of pre-trained representations.

### 3.2.3 SetFit: Few-Shot Classification with Sentence Embeddings

Tunstall et al. (2022) introduced SetFit (Sentence Transformer Fine-Tuning), a framework for few-shot text classification that fine-tunes sentence transformers using contrastive learning on a small number of labeled examples and then trains a classification head on the resulting embeddings. SetFit achieves competitive performance with prompt-based few-shot methods (T-Few, GPT-3 with prompts) while being orders of magnitude faster to train and requiring no prompts or verbalizers.

SetFit's key insight is that contrastive fine-tuning with as few as 8 examples per class produces embeddings that are substantially more discriminative for the target task than the frozen pre-trained representations. On the SST-2 sentiment benchmark, SetFit with 8 examples per class achieves accuracy of 0.87, compared to 0.80 for the frozen SBERT baseline and 0.92 for GPT-3 with 32 examples per class.

The relevance to TalkEx is methodological. SetFit represents the "fine-tuning" end of the representation adaptation spectrum, while TalkEx occupies the "frozen encoder" end. The TalkEx experimental design deliberately chose the frozen approach to minimize annotation requirements and computational cost — a choice that is validated if the downstream classification quality is competitive, as the H2 results suggest (Macro-F1 = 0.722 with frozen embeddings). The open question, identified as future work in Chapter 7, is whether contrastive fine-tuning on the PT-BR customer service corpus would close the performance gap on the weakest intent classes (compra, saudacao), where frozen representations may lack discriminative power.

### 3.2.4 AnthusAI (2024): Embeddings as Features for Classifiers

The AnthusAI "Semantic Text Classification" study (2024) provides the most direct conceptual precedent for TalkEx's classification architecture. The study systematically evaluates multiple embedding models (OpenAI text-embedding-3-small/large, Cohere embed-english-v3.0, all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5) as feature extractors for downstream classifiers (logistic regression, random forest, XGBoost, SVM, neural network) on a customer service intent classification task.

The central finding is that the combination of embedding model and classifier architecture matters more than either component in isolation. XGBoost on OpenAI embeddings achieves the highest accuracy (0.94) on a 6-class English customer service dataset, but the performance gap between embedding models narrows substantially when paired with gradient-boosted classifiers compared to linear classifiers. This suggests that the classifier compensates for representation weaknesses — precisely the mechanism that TalkEx's "embeddings represent, classifiers decide" principle operationalizes.

The AnthusAI study has three limitations relevant to TalkEx. First, the evaluation is on English single-turn data; multi-turn conversational context is absent. Second, the study does not incorporate lexical features alongside embeddings — a combination that the TalkEx ablation study shows contributes an additional +2.9 percentage points of Macro-F1. Third, the study does not integrate rule-based post-processing or evidence production. TalkEx extends the AnthusAI paradigm by adding heterogeneous features (lexical, structural, rule-based) to the classifier's input space and by operating on multi-turn context windows rather than individual sentences.

### 3.2.5 Huang and He (2025): LLM-Based Clustering as Classification

Huang and He (2025), in "Text Clustering as Classification with LLMs" (SIGIR-AP 2025), propose an alternative to supervised classification: using large language models to perform text clustering and then mapping clusters to class labels. The approach uses GPT-4 to generate cluster descriptions, assign texts to clusters, and iteratively refine the taxonomy. On the AG News and Yahoo Answers benchmarks, the method achieves competitive accuracy (0.86 and 0.63 respectively) without any task-specific fine-tuning or labeled training data.

The conceptual contribution is significant: it demonstrates that LLMs possess sufficient world knowledge to perform intent taxonomy construction and text classification as a zero-shot task. For intent discovery — the problem of identifying new, previously unlabeled contact reasons — this paradigm is directly applicable and is reflected in TalkEx's offline intent discovery pipeline (described in Chapter 4, Section 4.9).

The limitations for online conversational classification are severe. First, the approach requires online LLM inference for every classification decision, with latency of 1-10 seconds per request and monetary cost of $0.01-0.10 per request — prohibitive at the scale of millions of conversations per month. Second, LLM outputs are non-deterministic: the same input may receive different classifications across runs, precluding reproducible evaluation and complicating regulatory compliance. Third, no per-decision evidence is produced; the model's reasoning is opaque and unauditable. TalkEx addresses these limitations by restricting LLMs to offline roles (intent discovery, labeling assistance) while using lightweight, deterministic, evidence-producing components for online classification.

### 3.2.6 Gap: Frozen Encoder + Gradient Boosting on PT-BR Conversations

The embedding-based classification literature predominantly evaluates fine-tuned encoders on English benchmarks. The specific combination evaluated in TalkEx — a frozen multilingual sentence encoder (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensions) paired with LightGBM over heterogeneous features (embedding + lexical + structural + rule) on multi-turn PT-BR conversational data — has no direct precedent. The frozen-encoder choice is motivated by practical constraints (no dedicated GPU infrastructure, limited annotation budget) that are characteristic of Brazilian customer service deployments — all experiments were executed on Google Colab's free-tier GPU, making the empirical evaluation in Chapter 6 directly relevant to practitioners in this context.

---

## 3.3 Rule-Based and Hybrid NLP Systems

### 3.3.1 The Persistence of Rules in NLP

Despite the dominance of statistical and neural methods, rule-based systems persist in production NLP deployments where three conditions converge: regulatory requirements demand auditability, domain experts possess explicit knowledge that is expensive to learn from data, and coverage for specific high-stakes patterns must be guaranteed regardless of training data distribution. The literature on rule-based NLP spans decades, but the integration of rules *with* (rather than *instead of*) machine learning classifiers remains underexplored.

### 3.3.2 SystemT: Rule-Based Information Extraction at Scale

Chiticariu, Li, and Reiss (2013) presented SystemT, IBM's rule-based information extraction system, arguing provocatively that "rule-based information extraction is dead! Long live rule-based information extraction systems!" The paper documents the deployment of SystemT in production applications processing billions of documents, where the system uses a declarative rule language (AQL) to express extraction patterns over text.

SystemT's contribution is primarily architectural: it demonstrates that a well-designed rule language, compiled to an optimized execution plan with algebraic optimization, can achieve industrial-scale throughput while maintaining the interpretability and auditability that statistical methods lack. The system processes extraction rules through a cost-based optimizer that reorders predicate evaluation for efficiency — a design principle directly adopted by TalkEx's rule engine, which implements short-circuit evaluation ordered by predicate cost (lexical predicates first, semantic predicates last), as described in Chapter 4, Section 4.7.

The limitation of SystemT for the TalkEx context is that it is a pure rule system: it does not integrate with statistical classifiers. Rules either fire or they do not; there is no mechanism for rules to influence the confidence of a probabilistic classifier or for a classifier to override a rule when the rule's coverage is insufficient. TalkEx's rules-as-features integration — where rule activation outputs are included as binary features in the LightGBM classifier's input vector — represents a soft integration strategy absent from SystemT's design.

### 3.3.3 Snorkel: Weak Supervision with Labeling Functions

Ratner et al. (2017) introduced Snorkel, a system for training classifiers using programmatically generated labels from labeling functions — heuristic rules that noisily label subsets of the data. Snorkel uses a generative model to estimate the accuracy and correlation structure of the labeling functions, producing probabilistic training labels that are then used to train a downstream discriminative model. On multiple NLP benchmarks, Snorkel achieves within 2-5 F1 points of models trained on hand-labeled data while requiring no manual annotation.

Snorkel's contribution is a principled framework for combining noisy rule-based signals with supervised learning. The labeling functions can be thought of as noisy, partial rules — exactly the kind of signal that domain experts in contact centers can produce (e.g., "if the customer mentions 'cancelar' or 'desistir', this is probably a cancellation intent"). The generative model learns which labeling functions are reliable and which conflict, producing calibrated probabilistic labels.

The relevance to TalkEx is conceptual rather than implementational. TalkEx's rule engine operates at inference time rather than at training time: rules produce features that augment the classifier's input, rather than producing noisy training labels. However, the underlying insight is shared: deterministic heuristics encode domain knowledge that complements learned patterns. A future extension of TalkEx could adopt a Snorkel-like approach for semi-automated training data augmentation, using the rule engine's labeling function capabilities to generate additional labeled examples for underperforming intent classes.

### 3.3.4 Production Rule Engines in Conversational AI

The conversational AI industry has developed several production-grade rule engines that are relevant as engineering precedents, though most lack the academic evaluation that would constitute formal prior work.

Rasa, an open-source conversational AI framework, integrates rule-based dialogue policies with ML-based intent classification. Rules in Rasa define deterministic conversation flows that override the ML policy when specific conditions are met — a hard override strategy that, as TalkEx's H3 results demonstrate, can degrade overall performance when rule coverage is sparse (Chapter 6: rules-as-override Macro-F1 = 0.648 versus ML-only 0.722).

Google Dialogflow uses "contexts" and "fulfillment" rules to post-process intent classification results, enabling deterministic behavior for specific dialogue states. The integration is implicit and tightly coupled to the platform's architecture, precluding independent evaluation of the rule component's contribution.

Custom DSLs for conversational pattern matching are common in enterprise contact center platforms (Observe.AI, CallMiner, Verint) but are typically proprietary and undocumented in the academic literature. These systems generally implement keyword-based rules — "if the transcript contains 'cancel' within 3 turns of 'supervisor,' flag as escalation" — without the formal grammar, AST compilation, or predicate cost optimization that characterize TalkEx's rule engine.

### 3.3.5 Gap: Soft Integration of DSL-Compiled Rules with ML Classifiers

The gap in the literature is not the existence of rules in NLP systems — they are pervasive — but the *mode of integration* with statistical classifiers. Prior systems either use rules as standalone classifiers (SystemT), use rules to generate training data (Snorkel), or use rules as hard overrides of ML predictions (Rasa, enterprise platforms). The soft integration strategy evaluated in TalkEx — where rule activation signals are compiled from a typed DSL into AST predicates and then included as binary features in the classifier's heterogeneous feature vector — is, to our knowledge, novel. This integration allows the classifier to learn the informativeness of each rule signal rather than treating rule outputs as infallible, avoiding the degradation observed with hard override strategies. The evaluation in Chapter 6 shows that this soft integration produces a positive direction (+1.8pp Macro-F1) that does not reach statistical significance with the current 2-rule configuration (p = 0.131), motivating expansion of the rule set in future work.

---

## 3.4 Conversational Intent Classification

### 3.4.1 The Single-Turn Paradigm and Its Limitations

Intent classification for conversational systems has historically been framed as a single-turn problem: given one user utterance, predict the intent label. This framing reflects the chatbot-centric origins of the task, where each user message is treated as an independent query. The dominant benchmarks — ATIS (Hemphill et al., 1990), SNIPS (Coucke et al., 2018), CLINC150 (Larson et al., 2019), and BANKING77 (Casanueva et al., 2020) — reinforce this paradigm by providing single-utterance datasets with no conversational context.

### 3.4.2 Larson et al. (2019): Intent Detection Benchmarks

Larson et al. (2019) introduced CLINC150, a benchmark dataset of 23,700 single-turn queries across 150 intent classes plus an out-of-scope category, designed to evaluate intent detection in task-oriented dialogue. The dataset is notable for its scale (150 classes), its inclusion of out-of-scope examples (1,200 queries that match no defined intent), and its high inter-annotator agreement. Models evaluated on CLINC150 range from bag-of-words baselines (accuracy ~0.77) to fine-tuned BERT (accuracy ~0.97), establishing the state of the art for single-turn English intent detection.

CLINC150's contribution as a benchmark is significant, but its relevance to TalkEx is limited by three factors. First, the dataset consists entirely of single-turn English queries; conversational context is absent. Second, the 150 intent classes are designed for a virtual assistant domain (banking, travel, kitchen, etc.) that differs substantially from customer service complaint and support taxonomies. Third, the queries are clean, well-formed text without the ASR noise, abbreviations, and informal register characteristic of PT-BR customer service. TalkEx's 8-class taxonomy over multi-turn context windows represents a different task formulation that is not directly comparable to CLINC150.

### 3.4.3 Casanueva et al. (2020): Few-Shot Intent Detection

Casanueva et al. (2020) introduced BANKING77, a dataset of 13,083 single-turn customer banking queries across 77 fine-grained intent classes, and evaluated few-shot intent detection methods including USE (Universal Sentence Encoder) and ConveRT. Their key finding is that pre-trained sentence encoders, when fine-tuned on as few as 10 examples per class, achieve competitive performance with fully supervised baselines — accuracy of 0.84 with 10 examples versus 0.93 with full training data.

The BANKING77 evaluation is relevant to TalkEx as a domain precedent: it demonstrates that customer service intent classification is feasible with pre-trained representations and limited supervision. However, the single-turn formulation, the English-only evaluation, and the absence of multi-turn context windows limit the direct comparability. TalkEx extends the few-shot intuition by using a frozen encoder with no per-task fine-tuning at all, relying entirely on the combination of multilingual pre-training and heterogeneous features in the downstream classifier.

### 3.4.4 Zhang et al. (2021): Multi-Turn Intent Classification

Zhang et al. (2021) address the multi-turn intent classification problem directly, proposing models that incorporate dialogue history through hierarchical encoders that process turns sequentially and produce conversation-level representations. Their evaluation on the DSTC (Dialog State Tracking Challenge) benchmarks demonstrates that incorporating preceding turns improves intent classification accuracy by 3-8 percentage points over single-turn baselines, with the largest gains on intents that are expressed across multiple turns (e.g., "I need to change my booking" followed by clarification turns).

This work validates the core premise of TalkEx's context window design: that multi-turn context carries information that single-turn analysis misses. However, Zhang et al. use recurrent or transformer-based hierarchical encoders that are trained end-to-end, while TalkEx uses a simpler sliding window approach with mean pooling over frozen embeddings. The trade-off is expressiveness versus cost: TalkEx's approach requires no additional training and produces fixed-dimensional representations that are compatible with gradient-boosted classifiers, at the potential cost of losing fine-grained turn-level attention patterns. The context window construction mechanism is detailed in Chapter 4, Section 4.4.

### 3.4.5 Lyu et al. (2025): Attention Mechanisms for Text Classification

Lyu, Wang, Chen, and Zhang (2025) investigate the integration of neural attention mechanisms with large language models for text classification, evaluating on the AG News short-text benchmark. Their approach augments BERT-base with multi-head self-attention layers that learn to weight token representations before classification, achieving F1 = 0.89 versus 0.86 for BERT-base alone and 0.91 for a fine-tuned GPT-3.5 variant.

The contribution is incremental: attention-augmented classification has been studied extensively since Vaswani et al. (2017), and the 3-point improvement on AG News is modest. However, the paper provides a recent data point on the gap between frozen encoders and attention-augmented architectures, which is relevant to TalkEx's frozen-encoder design. The comparison suggests that task-specific attention could improve classification quality, but at the cost of requiring task-specific training — a cost that TalkEx avoids by design.

The critical limitation for TalkEx is the evaluation domain: AG News consists of short, single-sentence, formal English news articles, a register fundamentally different from multi-turn PT-BR customer service conversations. The authors do not evaluate on conversational data, multi-turn inputs, or non-English languages.

### 3.4.6 Gap: Multi-Turn PT-BR Intent Classification

The conversational intent classification literature is overwhelmingly English and overwhelmingly single-turn. The major benchmarks (CLINC150, BANKING77, ATIS, SNIPS) provide no conversational context, no Portuguese data, and no multi-turn evaluation. Zhang et al. (2021) address multi-turn modeling but on English dialogue benchmarks with end-to-end trained models. No prior work evaluates multi-turn intent classification on Brazilian Portuguese customer service data using a frozen encoder with heterogeneous features — the specific configuration evaluated in TalkEx. The 2,122-record post-audit corpus used in this thesis (Chapter 5) represents, to our knowledge, the first publicly available evaluation of this kind.

---

## 3.5 Cascaded and Cost-Aware Inference

### 3.5.1 The Cascade Principle

Cascaded inference is a computational efficiency strategy in which inputs are processed through a sequence of increasingly expensive classifiers, with confident predictions at early stages preventing unnecessary invocation of later, costlier stages. The underlying assumption is that a substantial fraction of inputs are "easy" and can be classified correctly by a cheap model, while only "hard" inputs require the full computational investment.

### 3.5.2 Viola and Jones (2001): The Original Cascade Classifier

Viola and Jones (2001) introduced the cascaded classifier in the context of face detection, constructing a sequence of increasingly complex Haar feature classifiers trained with AdaBoost. Each stage in the cascade is designed to achieve very high recall (to avoid missing faces) while progressively increasing precision (to reject non-face regions). The result is a system that processes the vast majority of image sub-windows in the first few stages, achieving real-time face detection on 2001-era hardware.

The Viola-Jones cascade established three design principles that remain relevant to TalkEx. First, early stages must be very cheap relative to later stages — the cost differential is the prerequisite for cascade benefit. Second, each stage must have high recall to avoid cascading misses. Third, the threshold at each stage controls the precision-recall trade-off and must be calibrated empirically. TalkEx's H4 experiment implements these principles in the conversational NLP domain, using a lexical-only classifier as Stage 1 and the full hybrid classifier as Stage 2, with confidence-based early exit thresholds. As reported in Chapter 6, the cascade fails because the cost differential prerequisite is violated: both stages share pre-computed embeddings, eliminating the cost advantage of the "cheap" stage.

### 3.5.3 Matveeva et al. (2006): Cascaded Retrieval in Web Search

Matveeva et al. (2006) applied the cascade principle to web search, using a sequence of increasingly sophisticated ranking models — from BM25 through linear combination features to neural rankers — with each stage re-ranking a progressively smaller candidate set. The approach reduces average query processing time by 40-60% while maintaining NDCG within 1-2% of the full-pipeline baseline.

This work demonstrates that cascaded inference is effective for retrieval tasks where the cost of the full model is dominated by the inference cost per candidate rather than by pre-computation. The lesson for TalkEx is that cascade efficiency depends on where computational cost is concentrated: if the dominant cost is a pre-computation step (embedding generation) shared across stages, cascading the downstream classifier provides no savings. This structural observation explains the H4 refutation and motivates the architectural redesign proposed in Chapter 7.

### 3.5.4 Schwartz et al. (2020): Efficiency in NLP

Schwartz, Dodge, Smith, and Etzioni (2020) provide a comprehensive survey of computational efficiency in NLP, covering model compression (distillation, pruning, quantization), early exit architectures (adaptive depth, confidence-based branching), and hardware-aware design. The survey argues that efficiency should be reported alongside accuracy in NLP evaluations, proposing the "efficiency-accuracy frontier" as the appropriate evaluation framework.

The survey identifies three conditions under which cascade and early-exit architectures are most effective: (1) high variance in input complexity, such that easy and hard inputs are clearly separable; (2) significant cost differential between the cheap and expensive models; and (3) reliable confidence estimation for the cheap model, such that early exits correspond to correct predictions. Condition (2) was not satisfied in TalkEx's H4 experiment, and condition (3) remains unevaluated due to the absence of calibration analysis (identified as future work in Chapter 7).

### 3.5.5 Gap: Cascade Applied to Conversational NLP with Hybrid Features

The cascade literature originates in computer vision (Viola and Jones, 2001) and has been applied to web search (Matveeva et al., 2006) and general NLP inference (Schwartz et al., 2020). Application to conversational NLP pipelines with hybrid features — where the "cheap" stage uses lexical features and the "expensive" stage adds embeddings and rules — is, to our knowledge, unexplored. TalkEx's H4 experiment (Chapter 6) provides the first empirical evidence on this configuration, yielding a negative result that identifies a specific architectural prerequisite (genuine cost separation between stages) for cascade effectiveness in this setting.

---

## 3.6 Portuguese NLP

### 3.6.1 BERTimbau: The Foundation of PT-BR Language Models

Souza, Nogueira, and Lotufo (2020) introduced BERTimbau, a BERT model pre-trained on a large Brazilian Portuguese corpus (brWaC, 2.7 billion tokens). BERTimbau achieves state-of-the-art results on Portuguese NLP benchmarks including named entity recognition (HAREM), sentence textual similarity (ASSIN/ASSIN2), and recognizing textual entailment, outperforming multilingual BERT by 1-5 percentage points across tasks.

BERTimbau's contribution is the demonstration that language-specific pre-training improves downstream performance over multilingual models for Portuguese. This finding is relevant to TalkEx's design trade-off: TalkEx uses a multilingual sentence encoder (paraphrase-multilingual-MiniLM-L12-v2) rather than a Portuguese-specific model, accepting potentially lower representation quality in exchange for zero language-specific training. Whether a PT-BR-specific encoder (e.g., a BERTimbau-based sentence transformer) would improve TalkEx's classification results is an open question identified as future work.

### 3.6.2 Portuguese NLP Datasets

The landscape of Portuguese NLP resources has improved substantially since 2020, but remains sparse relative to English, particularly for conversational data.

**MilkQA** (Bittar et al., 2020) provides 2,657 question-answer pairs in Brazilian Portuguese drawn from a dairy farming advisory service. The dataset is conversational in origin but consists of single-turn Q&A pairs without multi-turn context.

**FaQuAD** (Sayama et al., 2019) provides 900 question-answer pairs for Portuguese reading comprehension, following the SQuAD format. The data is drawn from Portuguese Wikipedia articles and is extractive in nature — a different task from intent classification.

**ASSIN and ASSIN2** (Fonseca et al., 2016; Real et al., 2020) provide sentence similarity and textual entailment evaluation for Portuguese, with approximately 10,000 sentence pairs. These datasets are critical for evaluating sentence-level semantics in Portuguese but do not address the conversational or classification dimensions of TalkEx.

**Portuguese NLI** datasets derived from XNLI (Conneau et al., 2018) provide cross-lingual natural language inference evaluation, enabling comparison of multilingual models on Portuguese. These benchmarks informed the choice of the paraphrase-multilingual-MiniLM-L12-v2 encoder in TalkEx, which was evaluated on XNLI as part of its multilingual training.

### 3.6.3 Challenges Specific to PT-BR Conversational Data

Brazilian Portuguese customer service data presents challenges that are absent or attenuated in the English-centric NLP literature.

**Diacritical inconsistency.** Brazilian Portuguese uses diacritical marks (accents, tildes, cedillas) that are systematically inconsistent in conversational text: "cancelamento" versus "cancelamênto," "não" versus "nao," "você" versus "voce." ASR systems exacerbate this inconsistency. TalkEx addresses this through Unicode NFKD normalization with accent stripping (Chapter 4, Section 4.3), a preprocessing step that is unnecessary for English but critical for Portuguese lexical matching.

**Informal register and abbreviations.** Customer service chat data in PT-BR exhibits extreme abbreviation: "vc" (voce), "td" (tudo), "blz" (beleza), "pq" (porque), "msg" (mensagem). These abbreviations are absent from the formal Portuguese text on which most language models were trained, creating a vocabulary gap that affects both lexical retrieval (BM25 treats "vc" and "voce" as different tokens) and embedding quality (the encoder may represent "vc" poorly).

**Code-switching with English.** Brazilian customer service conversations routinely include English loanwords and technical terms: "upgrade," "app," "feedback," "login," "premium." Multilingual encoders handle these terms better than Portuguese-specific models, which is one argument in favor of TalkEx's multilingual encoder choice.

**Limited public conversational datasets.** Prior to the dataset used in this thesis (RichardSakaguchiMS/brazilian-customer-service-conversations), no public PT-BR multi-turn customer service dataset with intent annotations was available. The absence of public evaluation data is the primary bottleneck for conversational NLP research in Portuguese.

### 3.6.4 Gap: PT-BR Conversational Intent Classification

Portuguese NLP has advanced substantially with BERTimbau and related models, but the conversational dimension remains underserved. No public PT-BR dataset provides multi-turn customer service conversations with intent annotations. No prior work evaluates hybrid retrieval, embedding-based classification, or rule integration on PT-BR conversational data. TalkEx's evaluation on the 2,122-record post-audit corpus (Chapter 5) represents the first systematic evaluation of these methods in this language and domain.

---

## 3.7 Positioning This Thesis

### 3.7.1 Synthesis of Gaps

The preceding sections identify six specific gaps in the literature, each addressed by a component of TalkEx:

1. **Hybrid retrieval on conversational PT-BR data.** Prior hybrid retrieval work evaluates on English web search, QA, or formal documents (Rayo et al., 2025; Harris, 2024). No evaluation exists for multi-turn PT-BR conversations.

2. **Frozen encoder + gradient boosting for PT-BR intent classification.** The embedding-based classification literature (AnthusAI, 2024; Tunstall et al., 2022) evaluates on English data, typically with fine-tuned encoders. The frozen-encoder + LightGBM combination on PT-BR is untested.

3. **Soft integration of DSL-compiled rules with ML classifiers.** Rule-based systems either operate independently (Chiticariu et al., 2013) or generate training labels (Ratner et al., 2017). The rules-as-features soft integration evaluated in TalkEx is novel.

4. **Multi-turn intent classification in PT-BR.** Benchmarks are single-turn and English (Larson et al., 2019; Casanueva et al., 2020). Multi-turn modeling (Zhang et al., 2021) has not been evaluated on Portuguese data.

5. **Cascaded inference for conversational NLP.** The cascade principle (Viola and Jones, 2001) has not been applied to conversational intent classification with hybrid features.

6. **Public PT-BR conversational evaluation.** No public dataset or benchmark exists for the specific task formulation.

### 3.7.2 Comparative Positioning Table

The following table compares TalkEx against the most relevant prior works across the dimensions that define the thesis contribution space. A checkmark indicates the feature is present; a dash indicates absence.

| Dimension | Rayo et al. (2025) | Harris (2024) | AnthusAI (2024) | Tunstall et al. (2022) | Huang & He (2025) | Larson et al. (2019) | Casanueva et al. (2020) | Zhang et al. (2021) | Chiticariu et al. (2013) | Ratner et al. (2017) | **TalkEx (this thesis)** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Language** | English | English | English | English | English | English | English | English | English | English | **PT-BR** |
| **Domain** | Regulatory | Medical | Customer svc | General | News/QA | Virtual asst | Banking | Dialogue | Enterprise | General | **Customer service** |
| **Hybrid retrieval** | Yes | Compared | -- | -- | -- | -- | -- | -- | -- | -- | **Yes** |
| **BM25 baseline** | Yes | Yes | -- | -- | -- | -- | -- | -- | -- | -- | **Yes** |
| **Dense retrieval** | Yes | Yes | -- | -- | -- | -- | -- | -- | -- | -- | **Yes** |
| **Score fusion** | Linear | Ensemble | -- | -- | -- | -- | -- | -- | -- | -- | **Linear + RRF** |
| **Embedding classification** | -- | -- | Yes | Yes | Implicit | -- | Fine-tuned | Fine-tuned | -- | -- | **Yes (frozen)** |
| **Frozen encoder** | -- | Yes | Partial | -- | -- | -- | -- | -- | -- | -- | **Yes** |
| **Heterogeneous features** | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Yes (emb+lex+struct+rule)** |
| **Gradient boosting** | -- | -- | Yes | -- | -- | -- | -- | -- | -- | -- | **Yes (LightGBM)** |
| **Rule engine** | -- | -- | -- | -- | -- | -- | -- | -- | Yes | Yes (labeling) | **Yes (DSL-AST)** |
| **Rules-as-features** | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Yes** |
| **Per-decision evidence** | -- | -- | -- | -- | -- | -- | -- | -- | Yes | -- | **Yes** |
| **Multi-turn context** | -- | -- | -- | -- | -- | -- | -- | Yes | -- | -- | **Yes (sliding window)** |
| **Cascaded inference** | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Yes (evaluated, refuted)** |
| **Statistical significance** | -- | -- | -- | Limited | -- | -- | Limited | Limited | -- | -- | **Yes (Wilcoxon, 5 seeds)** |
| **Ablation study** | Partial | -- | Partial | -- | -- | -- | -- | -- | -- | -- | **Yes (4 feature families)** |
| **Open-source artifact** | -- | -- | Yes | Yes | -- | Yes | Yes | -- | -- | Yes | **Yes (170 files, ~1900 tests)** |
| **Negative results reported** | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Yes (H3 inconclusive, H4 refuted)** |

### 3.7.3 The Unique Contribution Space

The positioning table reveals that TalkEx occupies a unique intersection in the literature. No prior work combines all of the following in a single evaluated system:

1. **Hybrid BM25 + dense retrieval** with parametric score fusion, evaluated on conversational data.
2. **Frozen multilingual encoder** used as a feature extractor (not fine-tuned), paired with gradient-boosted classification over heterogeneous features.
3. **DSL-compiled deterministic rules** integrated as soft features in the classifier's input space, with per-decision evidence production.
4. **Multi-turn context windows** as the primary unit of analysis, rather than individual sentences or documents.
5. **Cascaded inference** with confidence-based early exit, evaluated on classification cost-quality trade-offs.
6. **Brazilian Portuguese** conversational data with intent annotations.
7. **Transparent reporting** of both confirmed and refuted hypotheses with Wilcoxon signed-rank tests and systematic ablation.

The closest individual comparisons are Rayo et al. (2025) for hybrid retrieval (but on English regulatory text, without classification or rules), AnthusAI (2024) for embedding-based classification (but on English single-turn data, without hybrid retrieval or rules), and Chiticariu et al. (2013) for rule-based NLP (but without ML integration or soft feature injection). TalkEx's contribution is the specific combination of these approaches, their evaluation on PT-BR conversational data, and the honest reporting of both successes and failures.

The experimental results reported in Chapter 6 — hybrid retrieval MRR = 0.853 (p = 0.017 versus BM25), classification Macro-F1 = 0.722 with frozen embeddings rising to 0.740 with rule features, the cascade refutation, and the ablation quantifying embedding dominance at +33.0pp — provide the empirical grounding for this positioning. The negative results (H3 inconclusive, H4 refuted) are as informative as the positive results, delimiting the conditions under which the hybrid paradigm delivers measurable value and where it does not.

---

## References for This Chapter

The following references are cited in this chapter. Full bibliographic details are provided in the dissertation reference list.

AnthusAI. (2024). *Semantic Text Classification: Text Classification with Various Embedding Techniques*. GitHub repository. Retrieved from https://github.com/AnthusAI/semantic-text-classification.

Bittar, A., Patil, S., and Lu, W. (2020). MilkQA: A Dataset of Consumer Questions for the Task of Answer Selection. In *Proceedings of the 4th Workshop on e-Commerce and NLP* (pp. 42-47).

Casanueva, I., Temcinas, T., Gerber, D., Vandyke, D., and Mrksic, N. (2020). Efficient intent detection with dual sentence encoders. In *Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI* (pp. 38-45).

Chiticariu, L., Li, Y., and Reiss, F. R. (2013). Rule-based information extraction is dead! Long live rule-based information extraction systems! In *Proceedings of EMNLP 2013* (pp. 827-832).

Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S., Schwenk, H., and Stoyanov, V. (2018). XNLI: Evaluating cross-lingual sentence representations. In *Proceedings of EMNLP 2018* (pp. 2475-2485).

Fonseca, E. R., dos Santos, L. B., Criscuolo, M., and Aluisio, S. M. (2016). ASSIN: Avaliacao de Similaridade Semantica e INferencia textual. In *Proceedings of PROPOR 2016* (pp. 13-15).

Formal, T., Piwowarski, B., and Clinchant, S. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In *Proceedings of SIGIR 2021* (pp. 2288-2292).

Harris, L. (2024). Comparing lexical and semantic vector search methods when classifying medical documents. arXiv preprint arXiv:2505.11582v2.

Hemphill, C. T., Godfrey, J. J., and Doddington, G. R. (1990). The ATIS spoken language systems pilot corpus. In *Proceedings of the Workshop on Speech and Natural Language* (pp. 96-101).

Huang, C., and He, G. (2025). Text clustering as classification with LLMs. In *Proceedings of SIGIR-AP 2025*. arXiv preprint arXiv:2410.00927v3.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of EMNLP 2020* (pp. 6769-6781).

Khattab, O., and Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In *Proceedings of SIGIR 2020* (pp. 39-48).

Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., Kummerfeld, J. K., Leach, K., Laurenzano, M. A., Tang, L., and Mars, J. (2019). An evaluation dataset for intent detection and out-of-scope prediction. In *Proceedings of EMNLP 2019* (pp. 1311-1316).

Lyu, N., Wang, Y., Chen, F., and Zhang, Q. (2025). Advancing text classification with large language models and neural attention mechanisms. arXiv preprint arXiv:2512.09444v1.

Matveeva, I., Burges, C., Burkard, T., Lauber, A., and Wong, L. (2006). High accuracy retrieval with multiple nested ranker. In *Proceedings of SIGIR 2006* (pp. 437-444).

Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., and Re, C. (2017). Snorkel: Rapid training data creation with weak supervision. In *Proceedings of the VLDB Endowment*, 11(3), 269-282.

Rayo, J., de la Rosa, R., and Garrido, M. (2025). A hybrid approach to information retrieval and answer generation for regulatory texts. In *Proceedings of COLING 2025*. arXiv preprint arXiv:2502.16767v1.

Real, L., Fonseca, E., and Oliveira, H. G. (2020). The ASSIN 2 shared task: a survey. In *Proceedings of PROPOR 2020* (pp. 229-238).

Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019* (pp. 3982-3992).

Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., and Gatford, M. (1995). Okapi at TREC-3. In *Proceedings of the Third Text REtrieval Conference (TREC-3)* (pp. 109-126).

Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In *Proceedings of NAACL 2022* (pp. 3715-3734).

Sayama, H. F., Araujo, A. F., and Fernandes, E. R. (2019). FaQuAD: Reading comprehension dataset in the domain of Brazilian higher education. In *Proceedings of the IEEE 8th Brazilian Conference on Intelligent Systems (BRACIS)* (pp. 443-448).

Schwartz, R., Dodge, J., Smith, N. A., and Etzioni, O. (2020). Green AI. *Communications of the ACM*, 63(12), 54-63.

Souza, F., Nogueira, R., and Lotufo, R. (2020). BERTimbau: Pretrained BERT models for Brazilian Portuguese. In *Proceedings of BRACIS 2020* (pp. 403-417).

Thakur, N., Reimers, N., Rucktaschel, A., Srivastava, A., and Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In *Proceedings of NeurIPS 2021 Datasets and Benchmarks Track*.

Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M., and Pereg, O. (2022). Efficient few-shot learning without prompts. arXiv preprint arXiv:2209.11055.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In *Proceedings of NeurIPS 2017* (pp. 5998-6008).

Viola, P., and Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. In *Proceedings of CVPR 2001* (Vol. 1, pp. I-511-I-518).

Zhang, J., Hashimoto, K., Liu, W., Wu, C., Wan, Y., Yu, P., Socher, R., and Xiong, C. (2021). Discriminative nearest neighbor few-shot intent detection by transferring natural language inference. In *Proceedings of EMNLP 2021* (pp. 5064-5082).

---

*Chapter word count (approximate): 6,800 words.*
