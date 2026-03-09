# KNOWLEDGE BASE

# Arquitetura de Classificação Semântica e Busca Vetorial em NLP

Versão: 1.0
Uso: Engenharia / Data Science / AI Platform
Escopo: Sistemas de NLP em larga escala (Search, Classification, Conversation Intelligence)

---

# 1. Objetivo do Documento

Este documento consolida **princípios arquiteturais, estratégias de modelagem e práticas recomendadas** para sistemas de:

* classificação de texto
* busca semântica
* clustering textual
* análise de conversas
* pipelines NLP escaláveis

A base foi construída a partir da análise de pesquisas recentes e experimentos documentados em:

*  Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents
*  Text Clustering as Classification with LLMs
*  Advancing Text Classification with LLMs and Neural Attention Mechanisms
*  Semantic Text Classification Repository

Este KB serve como **referência oficial para decisões arquiteturais em sistemas de NLP corporativos.**

---

# 2. Conceitos Fundamentais

## 2.1 Representação Vetorial de Texto

Textos são convertidos em vetores numéricos chamados **embeddings**.

Exemplo:

```
"cancelar assinatura"
→ [0.12, -0.33, 0.91, ...]
```

Esses vetores capturam:

* semântica
* contexto
* similaridade entre textos

Modelos comuns:

| Modelo     | Tipo                      |
| ---------- | ------------------------- |
| Word2Vec   | embeddings locais         |
| BERT       | embeddings contextuais    |
| E5         | embeddings para retrieval |
| BGE        | embeddings para busca     |
| Instructor | embeddings instruídos     |

Embeddings são usados para:

* classificação
* clustering
* busca semântica
* recomendação
* análise de conversas

---

# 3. Busca Lexical vs Busca Semântica

Um resultado crítico demonstrado em  é que:

> métodos lexicais podem superar embeddings semânticos em tarefas estruturadas.

## 3.1 Busca Lexical

Baseada em correspondência de termos.

Exemplo:

```
query: cancelar plano
document: quero cancelar meu plano
```

Algoritmos comuns:

* TF
* TF-IDF
* BM25

Características:

| propriedade             | valor    |
| ----------------------- | -------- |
| interpretabilidade      | alta     |
| custo computacional     | baixo    |
| latência                | baixa    |
| generalização semântica | limitada |

---

## 3.2 Busca Semântica

Baseada em embeddings.

Exemplo:

```
"encerrar contrato"
≈ "cancelar assinatura"
```

Características:

| propriedade         | valor |
| ------------------- | ----- |
| semântica profunda  | alta  |
| custo computacional | médio |
| latência            | média |
| interpretabilidade  | baixa |

---

## 3.3 Conclusão Experimental

Estudo em documentos médicos mostrou:

* **BM25 outperformou embeddings semânticos**
* execução mais rápida
* maior precisão



Motivo:

* documentos altamente estruturados
* vocabulário consistente

---

# 4. Estratégia Recomendada: Busca Híbrida

Arquitetura recomendada:

```
Query
 ↓
BM25 Retrieval
 ↓
Top-K Documents
 ↓
Embedding Reranking
 ↓
Final Results
```

Benefícios:

* alta precisão
* menor latência
* menor custo computacional

Esta abordagem é usada em sistemas como:

* ElasticSearch
* Vespa
* Bing Search
* Google Search

---

# 5. Classificação Baseada em Embeddings

Embeddings **não classificam diretamente**.

Eles apenas representam o texto.

Classificação requer um modelo adicional.

Exemplo:

```
embedding → classifier → label
```

Classificadores comuns:

| Modelo              | Uso               |
| ------------------- | ----------------- |
| Logistic Regression | baseline          |
| SVM                 | datasets pequenos |
| MLP                 | datasets médios   |
| Transformer         | datasets grandes  |

Demonstrado em:



---

# 6. Classificação via Similaridade Vetorial

Outra abordagem:

```
query_embedding
↓
nearest neighbor search
↓
class = nearest_class
```

Algoritmo:

```
kNN
```

Vantagens:

* simples
* sem treinamento

Desvantagens:

* difícil escalar
* dependente do dataset

---

# 7. Clustering Transformado em Classificação com LLM

Uma inovação proposta em:



é converter **clustering em classificação**.

## Pipeline

### Stage 1 — Geração de Labels

LLM analisa dataset e gera possíveis categorias.

Exemplo:

```
inputs:
- "cancelar serviço"
- "quero encerrar contrato"

labels:
- cancelamento
```

---

### Stage 2 — Merge de Labels

LLM combina labels semelhantes.

Exemplo:

```
cancelamento
encerramento
→ cancelamento
```

---

### Stage 3 — Classificação

LLM classifica textos nas categorias geradas.

---

## Benefícios

Resolve problemas clássicos:

* falta de labels
* clusters sem interpretação
* descoberta de intents

Aplicações:

* intent discovery
* topic modeling
* análise de conversas

---

# 8. Attention Mechanisms em Classificação

Estudo em  propõe melhorar representações usando:

```
attention pooling
```

---

## 8.1 Problema

Mean pooling ignora importância de palavras.

Exemplo:

```
"quero cancelar meu plano agora"
```

Palavra crítica:

```
cancelar
```

---

## 8.2 Attention Pooling

Modelo aprende pesos para cada palavra.

```
embedding = Σ attention_weight * token_embedding
```

Resultado:

* embeddings mais discriminativos
* melhor classificação

---

# 9. Pipeline NLP Escalável

Arquitetura recomendada para sistemas de produção.

```
Transcription
 ↓
Turn Segmentation
 ↓
Context Window Builder
 ↓
Embedding Generation
 ↓
Hybrid Retrieval
 ↓
Reranking
 ↓
Classification
 ↓
Semantic Rule Engine
```

---

# 10. Indexação Vetorial

Para datasets grandes, usar ANN (Approximate Nearest Neighbor).

Algoritmos comuns:

| Algoritmo | Complexidade     |
| --------- | ---------------- |
| HNSW      | alta eficiência  |
| IVF       | alta compressão  |
| PQ        | memória reduzida |

Bibliotecas:

* FAISS
* Qdrant
* Milvus
* Chroma

---

# 11. Métricas de Avaliação

Para classificação:

| Métrica   | Significado                 |
| --------- | --------------------------- |
| Accuracy  | acertos gerais              |
| Precision | qualidade das previsões     |
| Recall    | cobertura                   |
| F1        | equilíbrio precision/recall |
| AUC       | separabilidade              |

Usadas no estudo em:



---

# 12. Problemas Comuns em Sistemas Semânticos

## 12.1 Overuse de LLMs

LLMs são caros e lentos.

Uso recomendado:

```
offline labeling
```

Não inferência em massa.

---

## 12.2 Embedding errado para tarefa

Exemplo:

embedding genérico para domain específico.

Solução:

* fine tuning
* domain embeddings

---

## 12.3 Ignorar baseline lexical

Erro comum.

Sempre testar:

```
BM25 baseline
```

---

# 13. Experimentos Recomendados

Para qualquer sistema novo executar:

### Benchmark 1

```
BM25
vs
Embeddings
```

---

### Benchmark 2

```
kNN classifier
vs
Logistic regression
```

---

### Benchmark 3

```
mean pooling
vs
attention pooling
```

---

### Benchmark 4

```
embedding clustering
+
LLM labeling
```

---

# 14. Arquitetura Ideal para Sistemas de Conversa

Para sistemas como:

* call centers
* chat analytics
* intent detection

Arquitetura sugerida:

```
Speech
 ↓
ASR
 ↓
Turn Segmentation
 ↓
Context Window Builder
 ↓
Embedding Generation
 ↓
Hybrid Search
 ↓
Intent Classifier
 ↓
Rule Engine
 ↓
Analytics
```

---

# 15. Conclusões Principais

Principais lições dos estudos:

1️⃣ Lexical search continua extremamente forte
2️⃣ Embeddings precisam de classificadores
3️⃣ LLMs são melhores para **rotulação offline**
4️⃣ Attention pooling melhora embeddings
5️⃣ Arquiteturas híbridas são superiores

---

# 16. Referências

*  Harris, L. Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents
*  Huang, C. Text Clustering as Classification with LLMs
*  Lyu, N. Advancing Text Classification with Large Language Models and Neural Attention Mechanisms
*  AnthusAI Semantic Text Classification

---