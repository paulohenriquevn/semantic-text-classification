# PRD — Web Demo: Semantic Conversation Intelligence Engine

## 1. Objetivo do Produto

Criar uma **aplicação web demonstrativa de alto impacto** que permita explorar milhares de conversas reais de call center com:

* busca híbrida em **ms**
* visualização completa de conversas
* explicação de relevância
* classificação e regras
* analytics básicos

A demo deve mostrar claramente que o sistema resolve um problema real:

> Encontrar rapidamente padrões e problemas em conversas de call center.

---

# 2. Arquitetura do Sistema

## Arquitetura Geral

```
                  ┌────────────────────┐
                  │  HuggingFace Data  │
                  └─────────┬──────────┘
                            │
                   Offline / Precompute
                            │
       ┌──────────────────────────────────────────┐
       │ ingestion                                │
       │ segmentation                             │
       │ context windows                          │
       │ embeddings generation                    │
       │ index build (lexical + vector)           │
       └──────────────┬───────────────────────────┘
                      │
                      │
               FastAPI Backend
                      │
        ┌─────────────┴──────────────┐
        │ Hybrid Search API          │
        │ Conversation API           │
        │ Analytics API              │
        │ Filters API                │
        └─────────────┬──────────────┘
                      │
                React Frontend
                      │
     Tailwind + shadcn/ui interface
```

---

# 3. Princípios Arquiteturais

### 1️⃣ Offline computation

Tudo pesado acontece **offline**.

```
offline/precompute
→ embeddings
→ index build
```

Nada caro roda durante a busca.

### 2️⃣ Query path extremamente leve

Query online executa apenas:

```
query
→ BM25
→ vector ANN
→ fusion
→ response
```

### 3️⃣ Nada de LLM no caminho de busca

LLM só pode aparecer em páginas secundárias se necessário.

---

# 4. Stack Tecnológica

## Backend

```
FastAPI
uvicorn
pydantic
numpy
sentence-transformers
qdrant (vector index)
tantivy / bm25 index
```

## Frontend

```
React
Vite
Tailwind
shadcn/ui
TanStack Query
TypeScript
```

## Data

```
HuggingFace datasets
```

---

# 5. Dataset

Dataset principal:

```
AIxBlock/92k-real-world-call-center-scripts-english
```

Uso na demo:

```
subset: 5k – 20k conversations
```

Após segmentação:

```
~100k context windows
```

Isso permite demonstrar:

* escala
* busca realista

---

# 6. Pipeline de Dados (Offline)

## Stage 1 — Dataset ingestion

Script:

```
scripts/ingest_dataset.py
```

Output:

```
Conversation
Turns
metadata
```

---

## Stage 2 — Segmentation

Converter conversas em:

```
ContextWindow
```

ex:

```
customer: I want to cancel my subscription
agent: Let me check your billing
```

---

## Stage 3 — Embeddings

Modelo recomendado:

```
bge-small-en
```

ou

```
all-MiniLM-L6-v2
```

---

## Stage 4 — Index Build

### Lexical index

BM25

Campos:

```
window_text
speaker
conversation_id
```

---

### Vector index

Qdrant / HNSW

Campos:

```
embedding
metadata
```

---

# 7. APIs (IMPLEMENTAR PRIMEIRO)

A equipe **deve iniciar pelo backend**.

Frontend só começa quando as APIs estiverem estáveis.

---

# 7.1 Search API

```
POST /search
```

### Request

```json
{
  "query": "customer wants refund",
  "filters": {
    "speaker": "customer"
  },
  "top_k": 20
}
```

### Response

```
{
  "results": [
    {
      "window_id": "win_123",
      "conversation_id": "conv_45",
      "text": "...",
      "lexical_score": 0.8,
      "semantic_score": 0.91,
      "score": 0.88
    }
  ],
  "latency_ms": 14
}
```

---

# 7.2 Conversation API

```
GET /conversation/{id}
```

Retorna:

```
turns
metadata
predictions
rules
```

---

# 7.3 Filters API

```
GET /filters
```

Retorna:

```
speaker
channel
intent
domain
```

---

# 7.4 Analytics API

```
GET /analytics/summary
```

Retorna:

```
top intents
top issues
volume
```

---

# 8. Performance Targets

Essas metas são **obrigatórias**.

## Search latency

| corpus   | target  |
| -------- | ------- |
| 5k conv  | < 20 ms |
| 10k conv | < 30 ms |
| 50k conv | < 80 ms |

---

## API response

```
p95 < 100 ms
```

---

## Conversation fetch

```
p95 < 50 ms
```

---

# 9. Backend Performance Testing

A equipe deve implementar:

```
tests/perf/test_search_latency.py
```

Teste:

```
100 queries
p50
p95
p99
```

Output esperado:

```
p50: 12ms
p95: 34ms
```

---

# 10. Frontend Aplicação

Após APIs estáveis.

---

# 10.1 Home Page

Campo principal:

```
Search conversations...
```

Layout:

```
[ search bar ]

filters

results list
```

---

# 10.2 Results

Card:

```
conversation snippet
score
intent
channel
```

---

# 10.3 Conversation View

Layout:

```
customer | agent
```

Ex:

```
Customer:
I want to cancel my subscription

Agent:
Let me check your account
```

---

# 10.4 Explain Match

Mostrar:

```
lexical score
semantic score
fused score
```

---

# 10.5 Analytics Panel

Gráficos:

```
top intents
issue distribution
conversation volume
```

---

# 11. UI Stack

Componentes:

```
shadcn/ui
```

Elementos principais:

```
Input
Table
Card
Tabs
Badge
Dialog
Chart
```

---

# 12. UX Performance

Frontend metas:

| métrica         | target  |
| --------------- | ------- |
| search response | < 150ms |
| UI render       | < 50ms  |

---

# 13. Observabilidade

Backend deve retornar:

```
latency_ms
```

Frontend deve mostrar:

```
Search completed in 12 ms
```

Isso é importante para credibilidade.

---

# 14. Demo Credibility Panel

Na UI mostrar:

```
Indexed conversations: 10,000
Context windows: 120,000
Average search latency: 18 ms
```

---

# 15. Roadmap de Implementação

## Fase 1 — Backend

Semana 1

```
dataset ingestion
segmentation
embeddings
index build
```

---

## Fase 2 — APIs

Semana 2

```
search api
conversation api
filters
analytics
```

---

## Fase 3 — Performance

Semana 3

```
latency testing
profiling
optimization
```

---

## Fase 4 — Frontend

Semana 4

```
search page
conversation viewer
analytics panel
```

---

# 16. Métricas de Sucesso

Demo será considerada pronta se:

```
search latency < 50ms
dataset > 5000 conversations
UI response < 150ms
```

---

# 17. Resultado Esperado

Uma aplicação web onde o usuário pode:

1. Digitar um problema de cliente
2. Encontrar conversas reais relevantes
3. Explorar contexto completo
4. Ver analytics e classificações

Tudo em **milissegundos**.

---

