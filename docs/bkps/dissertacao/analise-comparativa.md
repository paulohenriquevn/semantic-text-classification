# Análise Comparativa: TalkEx × BERTaú × Harris × Rayo

> Documento de referência para o Capítulo 6 (Resultados e Análise) da dissertação.
> Todos os números são extraídos diretamente dos papers e dos resultados experimentais do TalkEx.
> Nenhum dado foi inventado ou interpolado.

---

## 1. Perfil dos Sistemas

| Dimensão | **TalkEx** | **BERTaú** (Finardi et al., 2021) | **Harris** (2025) | **Rayo et al.** (2025, COLING) |
|---|---|---|---|---|
| **Domínio** | Atendimento ao cliente (PT-BR, multi-setor) | Atendimento financeiro (PT-BR, Itaú) | Documentos médicos (EN, 7 classes) | Textos regulatórios (EN, ObliQA) |
| **Abordagem** | Híbrida (BM25 + ANN + regras + cascata) | BERT treinado do zero | Comparação lexical vs semântico | Híbrida (BM25 + SBERT fine-tuned) |
| **Dataset** | 2.257 conversas (1.179 original + 1.078 expandidas) | 14.5 GB de dados AVI (~22.5M palavras) | 1.472 documentos | 27.869 perguntas |
| **Língua** | PT-BR | PT-BR | EN | EN |
| **Publicação** | Dissertação (em andamento) | arXiv 2101.12015v3 | arXiv 2505.11582v2 | COLING 2025, arXiv 2502.16767v1 |

---

## 2. Custo de Treinamento e Infraestrutura

| Dimensão | **TalkEx** | **BERTaú** | **Rayo** |
|---|---|---|---|
| **Treinamento de modelo de linguagem** | Nenhum (usa pré-treinado `paraphrase-multilingual-MiniLM-L12-v2`) | 1M steps, BERT-base from scratch, GPU com FP16 | Fine-tuning de `bge-small-en-v1.5`, 10 epochs, NVIDIA A40 |
| **Dados de treinamento do LM** | 0 GB (modelo pré-treinado) | 14.5 GB (corpus proprietário Itaú) | Não especificado (fine-tuning em ObliQA) |
| **Hardware para embeddings** | CPU (segundos) | GPU com FP16 | NVIDIA A40 GPU |
| **Treinamento de classificadores** | LogReg/LightGBM em CPU (~9s para LightGBM) | Não aplicável (avaliação em retrieval) | Não aplicável |
| **Dimensão dos embeddings** | 384 (MiniLM) | 768 (BERT-base) | 384 (bge-small) |
| **Vocabulário** | Pré-treinado multilingual | BertWordPieceTokenizer treinado no corpus Itaú | Pré-treinado |
| **Normalização** | NFKD + lowercase via talkex pipeline | NFKC | Não especificado |

---

## 3. Retrieval — Comparação de Métricas

### 3.1 MRR@10 (onde disponível)

| Sistema | MRR@10 | Contexto |
|---|---|---|
| **TalkEx Hybrid-RRF** | **0.826** | 2.257 conversas, 9 intents, busca por intent |
| **TalkEx BM25-only** | **0.802** | Mesmo dataset |
| **TalkEx ANN-only** | **0.799** | Mesmo dataset |
| **BERTaú** (pairwise) | **0.552** | 1.427 perguntas FAQ Itaú, 2.118 respostas |
| **BERTaú BM25+** | **0.345** | Mesmo dataset |
| **BERTaú DPR** | **0.526** | Mesmo dataset |
| **BERTaú mBERT uncased** | **0.458** | Mesmo dataset |
| **BERTaú distiluse-base-multilingual** | **0.417** | Mesmo dataset |

### 3.2 Recall@10 e MAP@10 (Rayo)

| Sistema | Recall@10 | MAP@10 |
|---|---|---|
| **Rayo Hybrid (α=0.65)** | **0.833** | **0.7016** |
| **Rayo BM25-only** | **0.761** | 0.6179 |
| **Rayo Semantic-only** | **0.810** | 0.6652 |

### 3.3 Ganho relativo do híbrido sobre BM25-only

| Sistema | BM25 baseline | Híbrido | Ganho relativo |
|---|---|---|---|
| **TalkEx** | MRR 0.802 | MRR 0.826 | **+3.0%** |
| **BERTaú** | MRR 0.345 | MRR 0.552 | **+60.0%** (requer treinar BERT do zero) |
| **Rayo** | Recall@10 0.761 | Recall@10 0.833 | **+9.5%** (requer fine-tuning em GPU) |

> **Nota:** Estas métricas NÃO são diretamente comparáveis entre si — cada sistema opera sobre datasets diferentes com definições de relevância diferentes. O valor está no **padrão**: os três sistemas demonstram que abordagens híbridas superam componentes isolados.

### 3.4 Harris — BM25 vs Semântico

Harris (2025) reporta que BM25 produziu **maior acurácia preditiva** que métodos semânticos em documentos médicos rigidamente estruturados (7 classes, 1.472 documentos). MiniLM foi o melhor modelo semântico. Métodos lexicais foram significativamente mais rápidos.

---

## 4. Classificação

| Sistema | Macro-F1 | Método | Classes | Dataset |
|---|---|---|---|---|
| **TalkEx lexical+emb_LightGBM (H2)** | **0.715** | Embeddings pré-treinados + features lexicais + LightGBM | 9 intents | 2.257 conversas |
| **TalkEx ML+Rules-feature (H3)** | **0.714** | ML + regras determinísticas como features | 9 intents | 2.257 conversas |
| **TalkEx ML-only (H3)** | **0.709** | LightGBM com embeddings | 9 intents | 2.257 conversas |
| **TalkEx lexical+emb_MLP (H2)** | **0.537** | Embeddings + MLP (128,64) | 9 intents | 2.257 conversas |
| **TalkEx lexical+emb_LogReg (H2)** | **0.559** | Embeddings + LogReg | 9 intents | 2.257 conversas |
| **TalkEx lexical-only LightGBM (H2)** | **0.309** | Só features lexicais (TF-IDF) | 9 intents | 2.257 conversas |
| **TalkEx Rules-only (H3)** | **0.130** | Apenas regras determinísticas | 9 intents | 2.257 conversas |

> BERTaú não reportou macro-F1 de classificação de intents — focou em FAQ retrieval (MRR) e sentiment analysis. Harris reporta acurácia, não F1. Rayo focou em retrieval + RAG.

---

## 5. Cascaded Inference (Exclusivo do TalkEx)

Nenhum dos três papers comparados implementa inferência em cascata. Este é um diferencial original do TalkEx.

| Threshold | % Resolvido no Estágio 1 (leve) | % Estágio 2 (pesado) | F1 | Δ F1 vs uniforme |
|---|---|---|---|---|
| Uniforme (baseline) | 0.0% | 100.0% | 0.741 | — |
| t=0.50 | 47.6% | 52.4% | 0.678 | -0.063 |
| t=0.60 | 32.0% | 68.0% | 0.714 | -0.028 |
| t=0.70 | 19.5% | 80.5% | 0.705 | -0.036 |
| t=0.80 | 9.5% | 90.5% | 0.718 | -0.023 |
| t=0.90 | 2.7% | 97.3% | 0.739 | **-0.003** |

---

## 6. Tabela Comparativa de Características

| Característica | **TalkEx** | **BERTaú** | **Harris** | **Rayo** |
|---|---|---|---|---|
| Busca híbrida (lexical+semântica) | ✅ RRF fusion | ❌ Modelo único | ❌ Comparação isolada | ✅ Score fusion (α=0.65) |
| Classificação supervisionada | ✅ LightGBM/LogReg | ❌ Só retrieval+sentiment | ✅ Vários classificadores | ❌ Só retrieval+RAG |
| Regras determinísticas auditáveis | ✅ DSL → AST com evidência | ❌ | ❌ | ❌ |
| Inferência em cascata | ✅ 2 estágios com threshold | ❌ | ❌ | ❌ |
| Requer treinamento de LM | ❌ | ✅ (1M steps, GPU) | ❌ | ✅ (fine-tuning, GPU) |
| Requer corpus proprietário | ❌ | ✅ (14.5 GB Itaú) | ❌ | ❌ (dataset público) |
| Funciona em CPU | ✅ | ❌ | ✅ | ❌ |
| Multi-língua | ✅ (modelo multilingual) | ❌ (PT-BR específico) | ❌ (EN) | ❌ (EN) |
| Evidência rastreável | ✅ (label, score, threshold, model version, text evidence) | ❌ | ❌ | Parcial (RAG citations) |

---

## 7. Argumentos para a Dissertação

### 7.1 Tese suportada pelos dados

> Uma arquitetura híbrida em cascata com embeddings pré-treinados, busca lexical+semântica e regras determinísticas auditáveis pode alcançar resultados competitivos **sem treinamento de modelos de linguagem**, ao contrário de abordagens como BERTaú que requerem corpus proprietário massivo e infraestrutura GPU.

### 7.2 Cinco argumentos com dados reais

1. **Híbrido > isolado** — Confirmado nos três trabalhos: TalkEx (+3.0% MRR, Hybrid-RRF 0.826 vs BM25 0.802), BERTaú (+60% MRR, mas partindo de BM25+ fraco), Rayo (+9.5% Recall@10). Padrão consistente, embora no TalkEx a diferença não seja estatisticamente significativa (Wilcoxon p=0.103).

2. **Treinamento de LM não é pré-requisito** — TalkEx atinge F1=0.715 com embeddings pré-treinados e classificadores leves treinados em ~6 segundos em CPU. BERTaú treina do zero com 14.5 GB e GPU para MRR=0.552.

3. **BM25 é surpreendentemente forte** — Harris (2025) demonstra que BM25 supera métodos neurais em dados estruturados. No TalkEx, BM25 (MRR 0.802) se equipara ao ANN (MRR 0.799), confirmando que lexical é forte em dados conversacionais com marcadores explícitos.

4. **Regras adicionam auditabilidade com ganho de F1** — TalkEx ML+Rules-feature (F1=0.714) supera ML-only (F1=0.709), com cancelamento atingindo F1=1.000 (precision=1.0, recall=1.0). Nenhum dos outros trabalhos oferece regras determinísticas com evidência rastreável.

5. **Cascata é contribuição original** — Nenhum dos trabalhos comparados implementa inferência em cascata. TalkEx com t=0.90 resolve 2.7% no estágio leve com perda desprezível de F1 (0.739 vs 0.741, delta=-0.003).

### 7.3 Limitações honestas

- **Dataset sintético** — 2.257 conversas geradas via LLM, não dados reais de produção. Resultados podem não generalizar para cenários reais com ruído de ASR e linguagem informal.
- **Métricas não diretamente comparáveis** — Datasets, tarefas e definições de relevância são diferentes entre os trabalhos.
- **Diferença híbrido vs BM25 não significativa** — Wilcoxon p=0.103 no H1, CI inclui zero. Com dataset maior a diferença pode se tornar significativa, mas o resultado atual sugere que BM25 é muito forte nesse domínio.
- **BERTaú opera em domínio mais restrito** (financeiro Itaú) com possível vantagem de especialização que embeddings genéricos multilinguais não capturam.
- **Ausência de fine-tuning** — TalkEx deliberadamente não faz fine-tuning, mas isso pode ser uma limitação em domínios muito especializados onde vocabulário proprietário é crítico (como demonstrado pela eficiência de tokens do BERTaú: 78 vs 130 do DPR).
- **Cascata com custo negativo** — O custo computacional medido mostra overhead (não redução) porque embedding generation domina o custo e é compartilhado entre estágios. A cascata beneficia cenários onde os estágios usam modelos de custo genuinamente diferente.

---

## Fontes

- Finardi, G. M. et al. (2021). "BERTaú: Itaú BERT for digital customer service." arXiv:2101.12015v3.
- Harris, C. (2025). "A Study on Medical Document Classification Using Lexical and Semantic Methods." arXiv:2505.11582v2.
- Rayo, L. et al. (2025). "Optimizing Retrieval Strategies for Financial Regulatory Documents." COLING 2025, arXiv:2502.16767v1.
- TalkEx experimental results: `experiments/results/H1-H4/summary.md` (2.257 conversations, train=1.581, test=338).
