# Chapter 2 — Background and Theoretical Foundations

## Abstract of Chapter

This chapter establishes the theoretical foundations required to understand the architecture, experiments, and results presented in this dissertation. We cover the core technical areas that underpin TalkEx: probabilistic information retrieval and the BM25 ranking function (Section 2.1), dense text representations from static word embeddings through sentence-level encoders (Section 2.2), hybrid retrieval architectures that combine lexical and semantic signals (Section 2.3), supervised text classification with heterogeneous feature families (Section 2.4), deterministic rule systems for auditable NLP (Section 2.5), conversational NLP and multi-turn context modeling (Section 2.6), and cascaded inference for cost-aware prediction (Section 2.7). Each section introduces the formal definitions, key algorithms, evaluation metrics, and open problems that motivate the specific design decisions described in Chapter 4 and the experimental protocol described in Chapter 5. The presentation assumes familiarity with linear algebra, probability theory, and basic machine learning; readers seeking a more introductory treatment are directed to Manning et al. (2008) for information retrieval and Jurafsky and Martin (2024) for natural language processing.

---

## 2.1 Information Retrieval Foundations

### 2.1.1 The Probabilistic Retrieval Framework

Information retrieval (IR) addresses the problem of identifying, from a large collection of documents $D = \{d_1, d_2, \ldots, d_N\}$, those documents relevant to a user's information need expressed as a query $q$. The probabilistic approach to IR, originating with Robertson and Jones (1976) and formalized in the Probability Ranking Principle (Robertson, 1977), holds that the optimal retrieval strategy ranks documents by their estimated probability of relevance given the query: $P(\text{rel} \mid q, d)$. Under the assumption that relevance judgments are binary and independent across documents, this ranking minimizes expected loss for a broad class of loss functions (Robertson, 1977).

The Binary Independence Model (BIM), developed by Robertson and Sparck Jones (1976), operationalizes this principle by modeling documents as binary term vectors and estimating relevance probabilities from term presence or absence. BIM assumes that terms occur independently in relevant and non-relevant documents — an assumption known to be false but empirically productive. The model derives an inverse document frequency (IDF) weighting that assigns higher scores to terms that discriminate between relevant and non-relevant documents:

$$\text{IDF}(t) = \log \frac{N - n_t + 0.5}{n_t + 0.5}$$

where $N$ is the total number of documents and $n_t$ is the number of documents containing term $t$. This formulation assigns high weight to terms that appear in few documents (discriminative) and low or negative weight to terms appearing in most documents (non-discriminative). The $+0.5$ smoothing prevents division by zero and moderates estimates for very rare terms.

### 2.1.2 BM25: Derivation and Parameters

BM25 (Best Matching 25) extends the binary independence model by incorporating within-document term frequency and document length normalization (Robertson et al., 1995; Robertson and Zaragoza, 2009). The scoring function for a query $q = \{t_1, t_2, \ldots, t_m\}$ against a document $d$ is:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where $f(t, d)$ is the frequency of term $t$ in document $d$, $|d|$ is the document length in tokens, $\text{avgdl}$ is the average document length across the collection, and $k_1$ and $b$ are free parameters.

The parameter $k_1$ controls term frequency saturation. When $k_1 = 0$, BM25 reduces to a binary model where only term presence matters, regardless of frequency. As $k_1 \to \infty$, the model approaches raw term frequency weighting without saturation. The standard setting $k_1 \in [1.2, 2.0]$ provides a sublinear response to term frequency: additional occurrences of a term increase the score but with diminishing returns. This sublinear saturation is critical for handling repetitive text — common in customer service conversations where customers may repeat a complaint keyword multiple times — without allowing frequency alone to dominate relevance.

The parameter $b \in [0, 1]$ controls the degree of document length normalization. When $b = 0$, no length normalization is applied; longer documents are systematically advantaged because they contain more term occurrences. When $b = 1$, full normalization penalizes longer documents proportionally to their excess length over the collection average. The standard setting $b = 0.75$ represents a moderate normalization that reduces length bias without excessively penalizing genuinely information-rich long documents. In the conversational domain, where "documents" are context windows of varying turn counts, the choice of $b$ directly affects whether longer conversations receive systematically higher or lower retrieval scores — a design consideration addressed in Chapter 4.

BM25 can be understood as a generalization of TF-IDF. When $k_1 \to \infty$ and $b = 0$, the BM25 term weighting reduces to $\text{IDF}(t) \cdot f(t, d)$, which is precisely the TF-IDF formulation (Salton and Buckley, 1988). The introduction of saturation ($k_1 < \infty$) and length normalization ($b > 0$) makes BM25 more robust across heterogeneous collections. The variant BM25+ (Lv and Zhai, 2011) introduces an additive lower bound $\delta$ to prevent the penalization of long documents that contain highly relevant terms:

$$\text{BM25+}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \left(\frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)} + \delta\right)$$

This variant is particularly relevant when document lengths vary widely, as in conversational corpora where interactions range from two-turn greetings to thirty-turn complaint escalations.

### 2.1.3 The Vector Space Model and the Vocabulary Mismatch Problem

The vector space model (VSM), introduced by Salton et al. (1975), represents both queries and documents as vectors in a $|V|$-dimensional space, where $|V|$ is the vocabulary size. Each dimension corresponds to a term, and the value along that dimension is typically a TF-IDF weight. Relevance is computed as the cosine similarity between the query vector and each document vector:

$$\cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \cdot |\vec{d}|}$$

The VSM provides a principled geometric framework for retrieval but shares with BM25 a fundamental limitation: both depend on exact lexical overlap between query and document terms. This creates the vocabulary mismatch problem (Furnas et al., 1987): when users and document authors employ different words to express the same concept, lexical retrieval methods fail. Furnas et al. (1987) estimated that the probability of two people using the same term for the same concept is less than 20% — a finding with direct consequences for customer service, where a customer saying "I don't want this anymore" and a labeled intent category "cancelamento" share zero lexical overlap.

Several approaches have been proposed to mitigate vocabulary mismatch within the lexical paradigm. Query expansion (Rocchio, 1971) augments the query with related terms drawn from pseudo-relevance feedback. Latent Semantic Indexing (Deerwester et al., 1990) projects the term-document matrix to a lower-dimensional space via singular value decomposition, capturing latent semantic associations. However, these methods introduce additional computational overhead and assumptions, and neither fully resolves the mismatch problem. The emergence of dense vector representations, discussed in Section 2.2, provides a more principled solution by mapping text to a continuous semantic space where similar meanings cluster regardless of surface form.

### 2.1.4 Evaluation Metrics for Information Retrieval

The evaluation of retrieval systems requires metrics that capture both the relevance and the ranking quality of returned results. We define the metrics used throughout this dissertation.

**Precision at $K$ ($P@K$)** measures the fraction of relevant documents among the top $K$ results:

$$P@K = \frac{|\{\text{relevant documents in top } K\}|}{K}$$

**Recall at $K$ ($R@K$)** measures the fraction of all relevant documents that appear in the top $K$ results:

$$R@K = \frac{|\{\text{relevant documents in top } K\}|}{|\{\text{all relevant documents}\}|}$$

**Mean Reciprocal Rank (MRR)** measures the average inverse rank of the first relevant result across a set of queries $Q$:

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the rank position of the first relevant document for query $i$. MRR is particularly appropriate when each query has a single correct answer or when only the first relevant result matters — a condition that frequently holds in intent classification, where the goal is to identify the single correct intent for a conversation. MRR is the primary retrieval metric used in the H1 experiments described in Chapter 6.

**Normalized Discounted Cumulative Gain (nDCG)** extends the evaluation to graded relevance judgments. For a ranked list of results, the discounted cumulative gain at rank $K$ is:

$$\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

where $\text{rel}_i$ is the relevance grade of the document at rank $i$. The normalization divides by the ideal DCG (IDCG), computed over the optimal ranking:

$$\text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

nDCG ranges from 0 to 1 and is the standard metric for retrieval benchmarks such as BEIR (Thakur et al., 2021) and MS-MARCO (Nguyen et al., 2016). While MRR is binary (relevant or not), nDCG accommodates graded relevance — relevant for future extensions of TalkEx to multi-label or hierarchical intent taxonomies.

---

## 2.2 Dense Representations and Sentence Embeddings

### 2.2.1 From Static Word Embeddings to Contextual Representations

The transition from sparse, high-dimensional lexical representations to dense, low-dimensional distributed representations fundamentally reshaped NLP. The distributional hypothesis — that words occurring in similar contexts have similar meanings (Harris, 1954; Firth, 1957) — provides the theoretical foundation. Word2Vec (Mikolov et al., 2013) operationalized this hypothesis by training shallow neural networks to predict a word from its context (CBOW) or a context from a word (Skip-gram), producing dense vectors of 100--300 dimensions that capture semantic and syntactic regularities. GloVe (Pennington et al., 2014) achieved similar representations through matrix factorization of global co-occurrence statistics, providing a connection between predictive and count-based methods.

Static word embeddings, however, assign a single vector per word type regardless of context. The word "bank" receives the same representation whether it denotes a financial institution or a river's edge. This polysemy limitation was addressed by contextual embedding models that produce token-level representations conditioned on the surrounding sentence.

ELMo (Peters et al., 2018) introduced context-dependent representations by concatenating the hidden states of a bidirectional LSTM language model. Each token receives a vector that reflects its specific usage in context, enabling disambiguation. However, the sequential nature of LSTMs limited scalability to long documents and parallelization on modern hardware.

The Transformer architecture (Vaswani et al., 2017) resolved the scalability limitation through self-attention, enabling each token to attend directly to all other tokens in the sequence. BERT (Devlin et al., 2019) applied the Transformer encoder in a masked language modeling pre-training objective, producing contextual representations that achieved state-of-the-art results across a wide range of NLP tasks. RoBERTa (Liu et al., 2019) demonstrated that careful optimization of pre-training hyperparameters — longer training, larger batches, dynamic masking — substantially improved upon BERT's results without architectural changes. XLM-RoBERTa (Conneau et al., 2020) extended this approach to 100 languages using Common Crawl data, establishing the multilingual pre-training paradigm that underlies the encoder used in TalkEx.

### 2.2.2 Sentence-Level Representations: The Pooling Problem

Transformer encoders produce token-level representations: for an input sequence of $n$ tokens, the encoder outputs $n$ vectors of dimension $d$. Many downstream tasks, however, require a single fixed-length representation per sentence or text segment — a sentence embedding. Converting token-level representations to sentence-level representations is the pooling problem.

The simplest approach, **mean pooling**, averages the token-level vectors:

$$\vec{s} = \frac{1}{n} \sum_{i=1}^{n} \vec{h}_i$$

where $\vec{h}_i$ is the hidden state of the $i$-th token at the final encoder layer. Mean pooling has the virtues of simplicity and stability but treats all tokens equally, regardless of their semantic importance. In the sentence "I absolutely need to cancel my subscription immediately," the tokens "cancel" and "subscription" carry far more intent-discriminative information than "I" or "my," yet mean pooling weights them identically.

**[CLS] token pooling** uses the representation of the special classification token as the sentence embedding. In BERT's pre-training, the [CLS] token is trained to aggregate sentence-level information for next-sentence prediction, but Reimers and Gurevych (2019) demonstrated that [CLS] representations without fine-tuning produce poor sentence embeddings — worse than GloVe averages on semantic textual similarity (STS) tasks.

**Attention-weighted pooling** learns a set of attention weights over the token representations:

$$\alpha_i = \frac{\exp(\vec{w}^T \vec{h}_i)}{\sum_{j=1}^{n} \exp(\vec{w}^T \vec{h}_j)}, \quad \vec{s} = \sum_{i=1}^{n} \alpha_i \vec{h}_i$$

where $\vec{w}$ is a learned parameter vector. This approach allows the model to weight semantically important tokens more heavily. Lyu et al. (2025) demonstrated that attention-based pooling improves classification F1 from 0.86 to 0.89 on the AG News dataset compared to standard BERT representations, confirming that learned token weighting provides a measurable advantage for discriminative tasks.

### 2.2.3 Sentence-BERT and the Siamese Training Paradigm

Reimers and Gurevych (2019) identified a critical limitation of using BERT directly for sentence similarity: computing the similarity between two sentences requires feeding both into the encoder simultaneously (cross-encoding), producing $O(n^2)$ complexity for pairwise comparisons over $n$ sentences. For a retrieval task over 10,000 candidates, this requires 10,000 forward passes per query — computationally prohibitive.

Sentence-BERT (SBERT) addresses this by fine-tuning BERT in a siamese or triplet network architecture. Two sentences are encoded independently by weight-shared encoders, and the resulting embeddings are compared using cosine similarity. The model is trained on Natural Language Inference (NLI) data — sentence pairs labeled as entailment, contradiction, or neutral — using a softmax classification objective, or on STS data using a regression objective with mean squared error loss. The key innovation is that inference requires only a single forward pass per sentence, enabling precomputation of all candidate embeddings and retrieval via approximate nearest-neighbor search at $O(\log n)$ cost.

The triplet training variant uses anchor-positive-negative triples with the objective:

$$\mathcal{L} = \max(0, \| \vec{a} - \vec{p} \| - \| \vec{a} - \vec{n} \| + \epsilon)$$

where $\vec{a}$, $\vec{p}$, $\vec{n}$ are the embeddings of the anchor, positive, and negative examples, and $\epsilon$ is a margin hyperparameter. This formulation directly optimizes the embedding space geometry for retrieval, pushing similar pairs closer and dissimilar pairs apart. Subsequent work introduced more efficient contrastive losses. The MultipleNegativesRankingLoss (Henderson et al., 2017) treats all other examples in the batch as negatives, dramatically increasing the effective number of negative samples without additional computation.

### 2.2.4 Multilingual Encoders and paraphrase-multilingual-MiniLM-L12-v2

Extending sentence embeddings to multilingual settings requires models trained on parallel or pseudo-parallel data across multiple languages. Knowledge distillation from a high-quality monolingual teacher to a multilingual student has proven effective: Reimers and Gurevych (2020) demonstrated that training a multilingual student to mimic the embeddings of a monolingual English SBERT teacher, using parallel sentence pairs, produces multilingual embeddings that approach the teacher's quality in English while extending coverage to over 50 languages.

The model used throughout TalkEx, **paraphrase-multilingual-MiniLM-L12-v2**, exemplifies this approach. It is a 12-layer Transformer encoder distilled from a larger paraphrase model, producing 384-dimensional sentence embeddings. The model supports over 50 languages, including Portuguese, and was trained on paraphrase data using the MultipleNegativesRankingLoss. Its 384-dimensional output represents a compression from the 768 dimensions of the BERT-base architecture, reducing storage requirements and nearest-neighbor search latency by a factor of two while retaining the majority of the representational capacity.

The choice of this specific model for TalkEx, described in detail in Chapter 4, reflects three constraints: (1) native support for Brazilian Portuguese without domain-specific fine-tuning, (2) manageable embedding dimensionality for large-scale indexing, and (3) availability as a frozen encoder — that is, a model used without gradient updates at deployment, ensuring that the embedding space remains stable across corpus versions and that longitudinal comparisons are valid.

### 2.2.5 Frozen versus Fine-Tuned Encoders

The decision to freeze or fine-tune a pre-trained encoder involves a trade-off between task-specific optimization and operational stability. Fine-tuning adapts the encoder's representations to the target domain and task, potentially improving performance significantly. However, fine-tuning introduces several complications. First, each fine-tuning run produces a different embedding space, invalidating previously computed vectors — a critical concern for systems that maintain persistent vector indices. Second, fine-tuning requires labeled training data in the target domain, which may be scarce or expensive to obtain. Third, fine-tuning risks catastrophic forgetting (McCloskey and Cohen, 1989; Kirkpatrick et al., 2017), where adaptation to the target domain degrades performance on the general capabilities the model acquired during pre-training.

The frozen-encoder approach treats the pre-trained model as a fixed feature extractor. All downstream learning occurs in lightweight classifiers (logistic regression, gradient boosted trees, small MLPs) that operate on the frozen embeddings. This approach sacrifices potential task-specific representational gains in exchange for embedding stability, reduced computational cost (no backward pass through the encoder), and the ability to precompute and cache all embeddings. Howard and Ruder (2018) and Peters et al. (2019) systematically studied the frozen-versus-fine-tuned trade-off, finding that fine-tuning provides substantial gains for tasks with significant domain shift but diminishing returns when the pre-training data is sufficiently broad.

In low-resource scenarios — defined not by absolute corpus size but by the ratio of labeled examples to class complexity — the frozen approach can outperform fine-tuning because the limited training data is insufficient to reliably update millions of encoder parameters without overfitting. This observation motivates the frozen-first architecture adopted in TalkEx and tested experimentally in Chapter 6.

---

## 2.3 Hybrid Retrieval

### 2.3.1 The Complementarity Thesis

The theoretical motivation for hybrid retrieval rests on a complementarity thesis: lexical and semantic retrieval methods have distinct and largely non-overlapping failure modes, and their combination should therefore improve recall beyond what either achieves alone. Lexical methods fail when relevant documents use different vocabulary from the query (vocabulary mismatch). Semantic methods fail when relevance depends on exact term matching — product codes, names, regulatory keywords, domain-specific abbreviations — where the pre-trained embedding space does not distinguish between surface forms with different operational meanings.

Lin et al. (2021) provided a systematic analysis of this complementarity on the MS-MARCO passage retrieval task. They computed the overlap between the top-1000 results of BM25 and a dense retriever (ANCE), finding that only approximately 60% of relevant passages appeared in both result sets. The remaining 40% were uniquely retrieved by one method or the other, confirming that the two approaches access genuinely different relevance signals. This finding has been replicated across multiple benchmarks and domains (Thakur et al., 2021; Formal et al., 2022).

In the conversational domain, complementarity is amplified by the heterogeneous nature of the text. Customer service conversations contain both structured signals (product names, plan identifiers, regulatory body references) that favor lexical matching and unstructured signals (implicit intent, paraphrased complaints, emotional indicators) that favor semantic matching. A customer saying "Procon" is using a regulatory keyword that BM25 retrieves with high precision; a customer saying "I'm going to take this to the consumer protection agency" expresses the same regulatory intent without the keyword, requiring semantic retrieval. A system that combines both modalities is therefore better matched to the heterogeneity of the data than either alone.

### 2.3.2 Score Fusion Methods

Given the candidate sets from lexical and semantic retrieval, the fusion step combines them into a single ranked list. Several fusion methods have been proposed, each with different assumptions and properties.

**Linear interpolation** computes a weighted sum of normalized scores:

$$s_{\text{hybrid}}(q, d) = \alpha \cdot \hat{s}_{\text{sem}}(q, d) + (1 - \alpha) \cdot \hat{s}_{\text{lex}}(q, d)$$

where $\hat{s}_{\text{sem}}$ and $\hat{s}_{\text{lex}}$ are the semantic and lexical scores normalized to $[0, 1]$ (typically via min-max normalization within each result set), and $\alpha \in [0, 1]$ is a fusion weight. Linear interpolation is simple and interpretable: $\alpha$ directly controls the relative importance of semantic versus lexical signals. However, the optimal $\alpha$ is domain-dependent and must be tuned empirically. In the TalkEx experiments (Chapter 6), $\alpha$ is varied systematically and the optimal value of $\alpha = 0.30$ (favoring the lexical component) is determined on the validation set.

**Reciprocal Rank Fusion (RRF)** (Cormack et al., 2009) fuses based on rank positions rather than scores:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $R$ is the set of rankers (lexical, semantic), $\text{rank}_r(d)$ is the rank of document $d$ in ranker $r$'s output, and $k$ is a constant (typically 60) that mitigates the influence of high-ranking outliers. RRF has the advantage of being score-agnostic — it does not require score normalization, which is problematic when lexical and semantic scores are on fundamentally different scales and with different distributions. Cormack et al. (2009) demonstrated that RRF is competitive with or superior to trained fusion methods on TREC datasets, despite requiring no training data.

**Learned fusion** trains a model (logistic regression, small neural network) to predict relevance from the individual retrieval scores and optionally additional features (query length, candidate document characteristics). This approach can capture non-linear interactions between the signal sources but requires relevance-labeled training data for the fusion model and risks overfitting to the training distribution. For the purposes of TalkEx, the interpretability advantage of linear interpolation and the score-agnostic property of RRF are preferred over the marginal gains of learned fusion — consistent with the design principles governing the system's architecture (Chapter 4).

### 2.3.3 Cross-Encoder Reranking

A limitation of bi-encoder retrieval — whether lexical, semantic, or hybrid — is that query and document are encoded independently. This prevents the model from capturing fine-grained token-level interactions between the query and candidate document. Cross-encoder reranking addresses this limitation by jointly encoding the query-document pair in a single Transformer forward pass:

$$s_{\text{cross}}(q, d) = \sigma(\text{MLP}(\text{BERT}([q; \text{SEP}; d])))$$

where $[q; \text{SEP}; d]$ denotes the concatenation of query and document tokens with a separator, and the model produces a scalar relevance score. Nogueira and Cho (2019) demonstrated that cross-encoder reranking over an initial BM25 candidate set improved MRR on MS-MARCO by 12 points absolute over BM25 alone, establishing the retrieve-then-rerank paradigm as the dominant architecture in modern IR.

The computational cost of cross-encoding is $O(K)$ forward passes per query, where $K$ is the candidate set size. This cost is acceptable as a second-stage refinement (reranking the top 100 candidates) but prohibitive as a first-stage retrieval method over millions of documents. In the TalkEx architecture, cross-encoder reranking is an optional second-stage component, applicable when the classification confidence from the first stage falls below a calibrated threshold — a design that integrates reranking into the cascaded inference strategy described in Section 2.7.

### 2.3.4 Dense-Sparse Fusion Architectures

Beyond the retrieve-and-fuse paradigm, recent work has explored architectures that integrate lexical and semantic signals at the representation level rather than at the score level.

**SPLADE** (Formal et al., 2021) learns sparse representations from a masked language model by computing term importance weights over the entire vocabulary. The resulting representation is sparse (most weights are zero) but learned, enabling the model to assign non-zero weights to semantically related terms not present in the input — effectively learning query expansion within the encoding process. SPLADE achieves competitive or superior performance to dense retrievers on BEIR benchmarks while maintaining the efficiency of inverted index retrieval.

**ColBERT** (Khattab and Zaharia, 2020) introduces late interaction: queries and documents are encoded into sequences of token-level embeddings, and relevance is computed as the sum of maximum similarities between each query token and all document tokens:

$$s_{\text{ColBERT}}(q, d) = \sum_{i=1}^{|q|} \max_{j \in \{1,\ldots,|d|\}} \vec{q}_i^T \vec{d}_j$$

This formulation preserves fine-grained token-level matching while still allowing document embeddings to be precomputed. ColBERTv2 (Santhanam et al., 2022) introduced residual compression to reduce the storage overhead of per-token representations. These architectures represent the frontier of hybrid retrieval research, though they require substantially more storage and computational resources than the bi-encoder approach used in TalkEx.

The design choice in TalkEx — bi-encoder retrieval with post-hoc score fusion — prioritizes simplicity, interpretability, and the ability to diagnose the contribution of each signal source independently. This choice is evaluated empirically in Chapter 6, where the linear fusion and RRF methods are compared directly.

---

## 2.4 Text Classification

### 2.4.1 Traditional Approaches: From Naive Bayes to Support Vector Machines

Text classification — the assignment of one or more predefined labels to a text document — is among the oldest and most thoroughly studied problems in NLP. The generative approach, exemplified by Multinomial Naive Bayes (McCallum and Nigam, 1998), models the joint probability $P(c, d)$ of class $c$ and document $d$ using the conditional independence assumption:

$$P(c \mid d) \propto P(c) \prod_{t \in d} P(t \mid c)$$

Despite the strong independence assumption, Naive Bayes performs surprisingly well on text classification — a phenomenon attributed to the fact that classification requires only correct ranking of posterior probabilities, not accurate probability estimates (Domingos and Pazzani, 1997). The model is fast to train, scales linearly with vocabulary size, and provides a natural probabilistic interpretation, making it a standard baseline.

Support Vector Machines (SVMs) (Joachims, 1998) approach classification discriminatively, finding the maximum-margin hyperplane in the feature space that separates classes. For text, the input features are typically TF-IDF vectors. SVMs are effective in high-dimensional spaces (large vocabularies) even with relatively few training examples, because the maximum-margin objective provides implicit regularization. Joachims (1998) demonstrated that SVMs consistently outperformed Naive Bayes, k-nearest neighbors, and decision trees on the Reuters-21578 text classification benchmark. The linear SVM, in particular, remains a strong baseline that is difficult to beat without substantially more data or more expressive representations.

**Logistic regression** occupies a middle ground: it is discriminative like SVMs but produces calibrated probability estimates (after appropriate regularization). For multi-class settings, the softmax extension (multinomial logistic regression) maps the linear predictions to a probability distribution over classes:

$$P(y = c \mid \vec{x}) = \frac{\exp(\vec{w}_c^T \vec{x} + b_c)}{\sum_{j=1}^{C} \exp(\vec{w}_j^T \vec{x} + b_j)}$$

Logistic regression over TF-IDF features is the standard lexical-only classification baseline in the TalkEx experiments (Chapter 6).

### 2.4.2 Gradient Boosted Trees for Heterogeneous Features

Traditional text classifiers operate on homogeneous feature representations — typically a single TF-IDF vector or a single embedding vector. In practice, conversational classification systems have access to multiple feature families: lexical features (TF-IDF, BM25 scores), semantic features (embedding vectors), structural features (speaker role, turn position, conversation length), and rule-derived features (binary indicators of rule activations). These feature families differ in dimensionality, scale, distribution, and information content.

Gradient boosted decision trees (GBDT), and specifically their modern implementations such as XGBoost (Chen and Guestrin, 2016) and LightGBM (Ke et al., 2017), are naturally suited to heterogeneous features. Decision trees partition the feature space through axis-aligned splits, inherently handling mixed feature types (continuous, categorical, binary) without the normalization and scaling that linear models and neural networks require. The ensemble is built sequentially, with each tree fitting the residual errors of the previous ensemble:

$$F_m(\vec{x}) = F_{m-1}(\vec{x}) + \eta \cdot h_m(\vec{x})$$

where $h_m$ is the $m$-th tree, $\eta$ is the learning rate, and the loss function (cross-entropy for classification) is optimized via functional gradient descent.

LightGBM introduces two key optimizations over XGBoost. **Gradient-based one-side sampling** (GOSS) retains all examples with large gradients (hard examples) while randomly sampling examples with small gradients, reducing training time without significant accuracy loss. **Exclusive feature bundling** (EFB) identifies features that rarely take non-zero values simultaneously and bundles them into a single feature, reducing the effective feature dimensionality. These optimizations are particularly relevant for conversational classification, where the feature vector may include a 384-dimensional embedding, a high-dimensional TF-IDF vector, and a handful of binary structural and rule features — a mixture that EFB handles efficiently.

In the TalkEx experiments, LightGBM with configuration $n_{\text{estimators}} = 100$ and $\text{num\_leaves} = 31$ is the primary classifier. This configuration provides a good balance between model complexity and generalization for the dataset size of 2,122 records, as validated on the held-out validation set (Chapter 6).

### 2.4.3 Neural Approaches

Neural text classifiers learn both representations and decision boundaries end-to-end. The **Multi-Layer Perceptron (MLP)** is the simplest neural classifier: one or more hidden layers with nonlinear activations followed by a softmax output layer. When the input is a pre-computed embedding vector, the MLP learns a nonlinear mapping from the embedding space to the label space. This approach has two advantages over linear classifiers: it can capture nonlinear interactions between embedding dimensions, and it can model class boundaries that are not linearly separable in the embedding space.

**Fine-tuned Transformers** represent the opposite end of the complexity spectrum. Rather than using frozen embeddings as input to a separate classifier, the entire encoder is updated jointly with a classification head during training. This allows the encoder to adapt its representations to the specific classification task, potentially learning domain-specific features that the pre-training objective did not optimize for. Devlin et al. (2019) demonstrated that fine-tuning BERT for text classification achieved state-of-the-art results across multiple benchmarks, often with only a few thousand labeled examples.

The cost of fine-tuning is substantial: millions of parameters are updated during training, requiring GPU computation and careful hyperparameter selection (learning rate, warmup steps, weight decay). For the 2,122-record TalkEx dataset, fine-tuning a 33M-parameter MiniLM encoder risks overfitting — a concern that motivates the frozen-encoder design and that the experimental comparison between frozen and fine-tuned encoders (discussed in Chapter 7) would quantify directly.

### 2.4.4 Multi-Class Evaluation Metrics

Classification evaluation in multi-class settings requires careful choice of aggregation strategy, as different metrics emphasize different aspects of performance.

**Macro-F1** computes the F1 score independently for each class and then averages:

$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

where $P_c$ and $R_c$ are the precision and recall for class $c$. Macro-F1 weights all classes equally regardless of their frequency, making it sensitive to performance on minority classes. This property is critical in the TalkEx dataset, where class frequencies range from 47 test examples (saudacao) to 121 test examples (cancelamento) — a ratio of nearly 3:1.

**Micro-F1** aggregates true positives, false positives, and false negatives across all classes before computing:

$$\text{Micro-F1} = \frac{2 \cdot \sum_c TP_c}{2 \cdot \sum_c TP_c + \sum_c FP_c + \sum_c FN_c}$$

Micro-F1 is dominated by the majority classes, and in balanced or near-balanced settings it approximates overall accuracy. In the TalkEx experiments, Macro-F1 is the primary metric because equal treatment of all intent classes reflects the operational requirement: a system that classifies "cancelamento" well but fails on "compra" is not operationally acceptable.

### 2.4.5 Calibration: Brier Score and Expected Calibration Error

Beyond discriminative accuracy, the reliability of predicted probabilities matters for systems that use confidence scores for downstream decisions — such as routing low-confidence predictions to human review or triggering cascaded inference stages.

The **Brier score** (Brier, 1950) measures the mean squared difference between predicted probabilities and actual outcomes:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} (p_{ic} - y_{ic})^2$$

where $p_{ic}$ is the predicted probability that example $i$ belongs to class $c$, and $y_{ic}$ is the indicator variable (1 if correct, 0 otherwise). The Brier score decomposes into reliability (calibration), resolution (discrimination), and uncertainty components (Murphy, 1973), providing a complete picture of probabilistic prediction quality.

**Expected Calibration Error (ECE)** (Naeini et al., 2015) partitions predictions into bins by confidence level and measures the average absolute difference between predicted confidence and actual accuracy within each bin:

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |acc_b - conf_b|$$

where $B$ is the number of bins, $n_b$ is the number of predictions in bin $b$, $acc_b$ is the empirical accuracy in the bin, and $conf_b$ is the average predicted confidence. A perfectly calibrated model has ECE = 0. Modern neural networks and gradient boosted trees are known to be overconfident (Guo et al., 2017), producing predicted probabilities that systematically exceed the true likelihood of correctness. Post-hoc calibration methods, such as Platt scaling (Platt, 1999) and temperature scaling (Guo et al., 2017), can reduce this miscalibration.

Calibration is directly relevant to TalkEx's cascaded inference strategy (Section 2.7), where confidence thresholds determine whether a prediction is accepted or escalated to a more expensive processing stage. A miscalibrated model that produces confidence scores of 0.95 for predictions with true accuracy of 0.70 would route too few cases to the expensive stage, degrading overall system quality.

---

## 2.5 Deterministic Rule Systems

### 2.5.1 Rule-Based NLP: Historical Context

Rule-based approaches to natural language processing predate statistical methods by decades. ELIZA (Weizenbaum, 1966) demonstrated that pattern-matching rules over surface text could produce remarkably convincing conversational behavior, despite encoding no linguistic knowledge. Early information extraction systems, such as those developed for the Message Understanding Conferences (MUC-3 through MUC-7, 1991--1998), relied extensively on hand-crafted extraction patterns, achieving high precision on narrow domains.

The statistical revolution of the 1990s and 2000s — initiated by Brown et al. (1990) for machine translation and Jelinek (1997) for speech recognition — displaced rule-based approaches in many NLP tasks. However, Chiticariu et al. (2013) argued persuasively that rule-based systems retain important advantages in production settings: they are transparent (a rule's behavior can be inspected and understood), deterministic (the same input always produces the same output), and modifiable (a business rule change is a text edit, not a model retraining cycle). In their survey of enterprise NLP deployments, Chiticariu et al. found that rule-based information extraction remained dominant in industry despite the academic preference for statistical methods.

The contemporary view, which informs the TalkEx design, treats rules and statistical models as complementary rather than competing. Rules provide guaranteed coverage for known patterns, produce auditable evidence for each decision, and impose zero inference cost per evaluation (no model forward pass). Statistical models provide coverage for patterns not explicitly encoded in rules and generalize across lexical variations. The integration strategy — how rules and models interact — determines the practical value of the combination.

### 2.5.2 AST-Based Rule Engines

A modern rule engine operates in two phases: compilation and evaluation. In the compilation phase, rules expressed in a human-readable domain-specific language (DSL) are parsed into an abstract syntax tree (AST). In the evaluation phase, the AST is traversed for each input, and predicates are evaluated against the input's features.

The AST representation provides several advantages over direct pattern matching. First, it enables static analysis: the compiler can detect conflicting rules, unreachable conditions, and redundant predicates before runtime. Second, it enables optimization: predicates can be reordered by cost (cheap lexical checks before expensive semantic computations), and short-circuit evaluation terminates early when a branch is determined to be false. Third, it provides a natural structure for evidence production: each evaluated predicate produces a trace record indicating what was checked, what value was observed, and whether the check passed or failed.

A rule in the TalkEx DSL has the following abstract structure:

```
RULE <name>
  WHEN <predicate_1> AND <predicate_2> AND ...
  THEN <action>
```

Predicates belong to four families: **lexical** (keyword presence, regex match, BM25 score threshold), **semantic** (embedding similarity to prototype vectors, intent score threshold), **structural** (speaker role, turn position, conversation length, channel), and **contextual** (pattern recurrence within a sliding window, sequential predicate satisfaction across turns). Actions include tagging, scoring, and priority assignment. Each predicate evaluation produces an evidence record that is attached to the final output, enabling complete auditability of every decision.

The TalkEx rule engine architecture is described in detail in Chapter 4.

### 2.5.3 Integration Strategies: Rules and Machine Learning

Three principal strategies exist for integrating deterministic rules with statistical classifiers:

**Rules-as-override.** The classifier produces a prediction; rules examine the prediction and the input features, and override the prediction when specific conditions are met. This is the dominant integration strategy in production systems (Chiticariu et al., 2013). Its advantage is simplicity: the classifier and rules operate independently, with rules acting as a post-processing safety net. Its limitation is that the classifier does not benefit from the rule system's knowledge during training — the two components are decoupled.

**Rules-as-features.** Each rule is evaluated on the training data, and its output (binary fire/no-fire or a continuous score) is appended to the feature vector used by the classifier. This allows the classifier to learn the conditional value of rule activations in combination with other features. The advantage is tighter integration: the classifier can learn, for example, that a "cancelamento" keyword rule firing in combination with a high semantic similarity to the "reclamacao" prototype is more indicative of complaint than of cancellation. The limitation is that adding rule features to the training data changes the feature distribution, requiring retraining whenever the ruleset is modified.

**Rules-as-postprocessing.** Rules operate on the classifier's output labels and scores, applying deterministic corrections. For example, a rule might reclassify any prediction with confidence below a threshold as "uncertain," or a rule might override a "saudacao" prediction when the conversation contains more than 10 turns (long conversations are unlikely to be mere greetings). This strategy requires no retraining and provides transparent, auditable corrections but cannot improve the classifier's internal representations.

In the TalkEx experiments (Chapter 6), the rules-as-features strategy is evaluated against rules-as-override and compared to the ML-only baseline. The hypothesis (H3) tests whether the integration of deterministic rules improves classification quality beyond what statistical features alone achieve.

### 2.5.4 Explainability and Auditability Requirements

In production NLP systems deployed in regulated industries — telecommunications, financial services, healthcare — the ability to explain and audit individual decisions is not a desirable feature but a regulatory requirement. The European Union's AI Act (2024) introduces obligations for "high-risk AI systems" including transparency, human oversight, and record-keeping. Brazil's Lei Geral de Protecao de Dados (LGPD, 2018) grants data subjects the right to request explanations of automated decisions that affect their interests.

Rule-based systems provide a natural mechanism for meeting these requirements. Every rule evaluation produces a trace: which rule was evaluated, what input features were examined, what values were observed, what thresholds were applied, and what outcome was reached. This trace constitutes a human-readable explanation of the decision that can be inspected, challenged, and corrected. Statistical models, by contrast, require post-hoc explanation methods (LIME, SHAP) that provide approximate and sometimes misleading explanations of model behavior (Rudin, 2019).

The TalkEx approach — combining statistical classifiers (which provide broad coverage and generalization) with deterministic rules (which provide auditable evidence for high-stakes decisions) — is designed to satisfy both the accuracy requirements of operational deployment and the transparency requirements of regulatory compliance. This dual objective is not standard in the literature, where classification accuracy and explainability are typically treated as competing objectives rather than co-requirements. The extent to which TalkEx achieves this dual objective is evaluated in Chapter 6.

---

## 2.6 Conversational NLP

### 2.6.1 Dialogue Act Classification and Intent Detection

Dialogue act classification assigns functional labels to utterances in a conversation, capturing the communicative intention of each turn. The foundational taxonomy is the DAMSL (Dialogue Act Markup in Several Layers) scheme (Core and Allen, 1997), which defines a hierarchy of communicative functions including statements, questions, directives, and commissives. The Switchboard-DAMSL corpus (Jurafsky et al., 1997) applied this scheme to telephone conversations, producing a 42-class taxonomy that remains a standard benchmark.

Intent detection, as studied in the task-oriented dialogue literature, is a specialization of dialogue act classification that assigns user utterances to a predefined set of intents relevant to a specific service domain. The ATIS (Airline Travel Information System) benchmark (Tur et al., 2010) contains 4,978 training utterances across 21 intent classes; the SNIPS dataset (Coucke et al., 2018) contains 13,784 training utterances across 7 intents. Both are single-turn benchmarks: each utterance is classified independently, without conversational context.

This single-turn assumption is a critical limitation for customer service applications. In a contact center conversation, the customer's intent frequently does not crystallize in a single utterance. A customer who says "I received my bill" in turn 1, "the amount is different from what I was told" in turn 3, and "I want to speak to a manager" in turn 7 is expressing a complaint-and-escalation intent that no single turn captures completely. The TalkEx approach to this problem — constructing sliding context windows over adjacent turns and generating embeddings at multiple granularities — is described in Section 2.6.2 and implemented in Chapter 4.

### 2.6.2 Multi-Turn Context Modeling

Modeling conversational context requires architectures that capture dependencies across turns. Several approaches have been proposed.

**Sliding window representations** concatenate or average the representations of $w$ adjacent turns into a single context vector. The window size $w$ controls the trade-off between local precision (small $w$ captures immediate context) and global coverage (large $w$ captures long-range dependencies). The stride $s$ determines the overlap between adjacent windows: when $s < w$, windows overlap, and each turn contributes to multiple context representations. This approach is computationally simple and interpretable — the contents of each window are directly inspectable — but assumes that the relevant context is localized within a fixed span.

**Hierarchical encoders** (Li et al., 2015; Serban et al., 2016) process conversations at two levels: a word-level encoder produces turn representations, and a turn-level encoder (typically a recurrent network or Transformer) processes the sequence of turn representations. The turn-level encoder learns to weight and combine turns based on their relevance to the classification task. The Hierarchical Attention Network (HAN) (Yang et al., 2016) extends this approach with attention at both levels, learning which words within a turn and which turns within a document are most relevant.

**Graph-based models** (Ghosal et al., 2019) represent conversations as directed graphs where nodes are turns and edges encode adjacency, same-speaker, and cross-speaker relationships. Graph neural networks then propagate information across the conversation structure, enabling each turn's representation to incorporate information from structurally related turns. The DialogueGCN model demonstrated that explicitly modeling speaker relationships improves emotion recognition in conversation, suggesting that speaker-aware representations may benefit intent classification as well.

TalkEx adopts the sliding window approach for its combination of simplicity, interpretability, and effectiveness, as described in Chapter 4. The window size and stride are configurable parameters whose values are determined empirically. The embedding is generated at the window level using the frozen multilingual encoder, and the concatenation of these window-level embeddings with lexical and structural features forms the input to the classifier.

### 2.6.3 Weak Supervision: Label Inheritance

In many operational settings, labels are available at the conversation level (from agent disposition codes, customer surveys, or topic taxonomies) but not at the turn or window level. This creates a weak supervision problem: the classifier operates at the sub-conversation level (windows), but the training signal is available only at the conversation level.

The simplest approach — **label inheritance** — assigns the conversation-level label to all windows within the conversation. This assumes that every window in a conversation labeled "cancelamento" exhibits the cancellation intent, which is clearly false: a 20-turn cancellation conversation will contain greeting turns, verification turns, and farewell turns that do not express cancellation intent. Label inheritance introduces label noise proportional to the conversation's topical heterogeneity.

More sophisticated approaches include **multiple instance learning** (MIL) (Dietterich et al., 1997), where the conversation is treated as a bag of windows and the learning algorithm must identify which instances within the bag are responsible for the bag-level label. Attention-based MIL (Ilse et al., 2018) learns to weight instances by their relevance to the bag label, effectively discovering the windows that carry the intent signal. Snorkel (Ratner et al., 2017) provides a programmatic weak supervision framework where multiple labeling functions vote on each instance, and a label model learns to denoise the votes.

The TalkEx experimental protocol uses label inheritance — conversation-level labels applied to all windows — and acknowledges this as a source of label noise. The impact of this noise on classification quality is a relevant factor in interpreting the experimental results in Chapter 6. The use of macro-averaged F1 as the primary metric partially mitigates the effect, as label noise affects classes differently depending on the proportion of semantically off-topic windows within conversations of each class.

### 2.6.4 PT-BR Specific Challenges

Brazilian Portuguese customer service text presents specific challenges that compound the general difficulties of conversational NLP.

**Diacritics and encoding inconsistency.** Portuguese uses diacritical marks (accents, cedillas, tildes) that are inconsistently represented in both ASR output and customer-typed text. "Cancelamento" may appear as "cancelamento," "cancelamênto," or "cancelamento" depending on the ASR system, the customer's keyboard, or the chat platform's encoding. Effective normalization — NFKD decomposition, accent stripping, case folding — is prerequisite to both lexical retrieval and classification. The TalkEx normalization pipeline, described in Chapter 4, applies these transformations uniformly.

**Informal register and abbreviations.** Customer service chat contains extensive informal language: "vc" (voce), "td" (tudo), "blz" (beleza), "pq" (porque), "mt" (muito), "qro" (quero). These abbreviations are not present in the formal Portuguese text on which multilingual encoders are primarily trained, creating a potential domain gap between pre-training and deployment distributions. The extent to which the frozen multilingual encoder handles this informal register — versus whether fine-tuning on in-domain data would improve performance — is an open empirical question related to the frozen-versus-fine-tuned comparison discussed in Section 2.2.5.

**Code-switching.** Brazilian customer service conversations frequently incorporate English terms, particularly for technology products: "upgrade," "app," "download," "feedback," "login." This code-switching is handled naturally by multilingual encoders that have seen both Portuguese and English during pre-training, but it creates challenges for lexical retrieval systems that may not index English stop words or apply Portuguese normalization rules to English terms.

**Regional variation.** Brazil's continental size produces significant regional linguistic variation. Expressions, idioms, and vocabulary differ across regions, and a customer from the northeast may use different words for the same intent as a customer from the south. The multilingual encoder's broad pre-training partially absorbs this variation, but the TalkEx dataset (drawn from a single synthetic corpus) does not systematically represent regional diversity — a limitation acknowledged in Chapter 7.

---

## 2.7 Cascaded Inference

### 2.7.1 Cost-Aware Machine Learning

The standard paradigm in machine learning research optimizes for accuracy, treating computational cost as an afterthought. In production systems processing millions of conversations per month, inference cost is a first-order constraint. The cost per prediction determines the system's operational viability: a model that achieves 2% higher accuracy but requires 10x the computational budget may be economically inferior to the cheaper alternative.

Cascaded inference addresses this tension by organizing models into a sequence of stages with increasing cost and accuracy. Simple, cheap models handle easy cases; only cases that the cheap model cannot resolve with sufficient confidence are escalated to expensive models. The cascade is governed by confidence thresholds: at each stage, if the predicted probability exceeds a threshold $\tau_k$, the prediction is accepted; otherwise, the input is passed to stage $k+1$.

The theoretical foundation for cascade classifiers was established by Viola and Jones (2001) in the context of face detection. Their cascade of increasingly complex Haar-feature classifiers achieved real-time face detection by rejecting approximately 50% of non-face image patches at each stage, so that the expensive final classifier processed only a small fraction of the input. The key insight is that the majority of inputs are "easy" — they can be correctly classified by a simple model — and only a minority of "hard" inputs require the full computational investment.

### 2.7.2 Cascaded Inference in NLP

The application of cascaded inference to NLP tasks has received less attention than in computer vision, though several relevant precedents exist.

In web search ranking, the retrieve-then-rerank paradigm (Matveeva et al., 2006; Nogueira and Cho, 2019) is a two-stage cascade: a fast first-stage retriever (BM25) produces a candidate set of hundreds of documents, and a slow second-stage ranker (neural cross-encoder) reranks only the candidates. The first stage has $O(N)$ cost over the full collection (with inverted index optimization), while the second stage has $O(K)$ cost over the top-$K$ candidates. The cascade ratio $K/N$ determines the cost savings.

Schwartz et al. (2020) proposed early exit strategies for Transformer models, where classification can be performed at intermediate layers rather than waiting for the full forward pass. If the prediction confidence at an early layer exceeds a threshold, the remaining layers are skipped. This reduces the average inference cost while preserving accuracy on easy examples. BERxiT (Xin et al., 2021) formalized this approach with learned exit classifiers at each Transformer layer, achieving 40% latency reduction with less than 1% accuracy degradation on GLUE benchmarks.

In multi-label classification, cascade approaches have been used to apply increasingly specialized classifiers in sequence: a first stage identifies the broad category, and subsequent stages discriminate within the category. This hierarchical cascade reduces the effective number of classes at each stage, simplifying the classification problem.

### 2.7.3 Calibration Requirements for Confidence-Based Routing

Cascade inference systems fundamentally depend on the reliability of confidence scores. The cascade threshold $\tau_k$ determines the trade-off between cost savings (higher $\tau_k$ accepts fewer predictions, escalating more to the expensive stage) and accuracy (lower $\tau_k$ accepts more predictions, including some incorrect ones). For this trade-off to be meaningful, the predicted confidence must accurately reflect the true probability of correctness.

A model that predicts confidence $p = 0.90$ should be correct approximately 90% of the time across all predictions with confidence near 0.90. If the model is overconfident — predicting $p = 0.90$ when the true accuracy is 0.70 — then a threshold $\tau = 0.85$ will accept predictions that are correct only 70% of the time, degrading the system's precision. Conversely, an underconfident model will escalate too many cases to the expensive stage, negating the cost savings.

The calibration metrics introduced in Section 2.4.5 — Brier score and ECE — directly quantify this reliability. In the TalkEx experiments, the cascade inference strategy (H4) relies on LightGBM's predicted probabilities to route predictions. The experimental results (Chapter 6) demonstrate that the cascade strategy, as implemented, does not reduce cost — a finding that may be partially attributable to the deterministic nature of the experimental setup (zero variance across seeds, as discussed in Chapter 6), which limits the variance in confidence scores that the cascade mechanism depends on.

### 2.7.4 Design Considerations for Production Cascades

Several practical considerations govern the design of cascade inference systems in production:

**Stage ordering.** Stages must be ordered by increasing computational cost. In TalkEx, the conceptual cascade is: (1) rule-based classification for known patterns (zero inference cost), (2) lightweight classifier over lexical features (minimal cost), (3) classifier over lexical plus embedding features (moderate cost, requires embedding computation), and (4) cross-encoder reranking or human review (high cost). The effectiveness of this ordering depends on the proportion of inputs resolved at each stage.

**Threshold selection.** Cascade thresholds can be set to optimize different objectives: minimize total cost subject to an accuracy constraint, maximize accuracy subject to a cost budget, or minimize a weighted combination. Bayesian optimization over the threshold space, evaluated on a validation set, provides a principled approach. In practice, thresholds are often set conservatively (high $\tau$) at deployment and relaxed incrementally as the system's behavior is observed.

**Fallback design.** When no stage in the cascade produces a sufficiently confident prediction, a fallback mechanism is required. Options include abstention (returning "uncertain" with no label), routing to human review, or returning the most confident prediction with a low-confidence flag. The choice of fallback strategy depends on the cost of errors versus the cost of abstention in the specific operational context.

**Monitoring and drift detection.** In production, the distribution of inputs evolves over time: new products, new complaint patterns, new vocabulary. A cascade system must monitor the proportion of inputs escalated at each stage. An increasing escalation rate signals distribution drift — the cheap stages are no longer resolving cases they previously handled — and triggers retraining or threshold recalibration.

The TalkEx cascade architecture, described in Chapter 4, implements the conceptual stages and evaluates their effectiveness in Chapter 6. The experimental finding that the cascade did not reduce cost (H4 refuted) motivates a discussion of the conditions under which cascaded inference is beneficial and the design modifications that might make it effective in future iterations (Chapter 7).

---

## 2.8 Chapter Summary

This chapter has established the theoretical foundations for the four pillars of the TalkEx architecture:

1. **Retrieval** (Sections 2.1 and 2.3): BM25 provides a strong, interpretable lexical baseline; dense sentence embeddings enable semantic generalization; hybrid fusion combines both signals; and cross-encoder reranking provides optional second-stage refinement.

2. **Representation** (Section 2.2): The progression from static word embeddings through contextual encoders to sentence-level representations via siamese training defines the embedding paradigm used in TalkEx. The frozen multilingual MiniLM encoder represents a deliberate trade-off between task-specific optimization and operational stability.

3. **Classification** (Section 2.4): Traditional classifiers over TF-IDF provide lexical baselines; gradient boosted trees (LightGBM) natively handle heterogeneous features from multiple signal families; neural approaches provide a ceiling reference; and calibration metrics (Brier score, ECE) quantify the reliability required for confidence-based routing.

4. **Auditability and cost control** (Sections 2.5 and 2.7): Deterministic rule engines provide transparent, auditable decisions for high-stakes cases; cascaded inference manages computational cost by resolving easy cases early; and the integration of rules and classifiers is formalized through three distinct strategies evaluated experimentally.

The conversational context (Section 2.6) adds a fifth dimension: multi-turn modeling, weak supervision, and the specific challenges of PT-BR customer service text. Together, these foundations define the design space within which TalkEx operates and the metrics against which it is evaluated in Chapter 6.

The related work that positions TalkEx relative to existing systems is discussed in Chapter 3. The concrete architecture that operationalizes these foundations is described in Chapter 4. The experimental protocol that tests the resulting hypotheses is detailed in Chapter 5.
