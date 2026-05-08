# CONTEÚDO PARA BANNER — PESQUISA APLICADA
# (Usar no template: 4. Template Banner - Relatório Técnico - Pesquisa Aplicada.pptx)

---

## TÍTULO

**TalkEx: Arquitetura Híbrida para Classificação de Intenções em Conversas de Atendimento ao Cliente em Português Brasileiro**

## AUTORES

Cauê Cavichioli Leão¹, Paulo Henrique Vieira Nascimento¹, Suele Susan Feitosa Sousa¹

¹Centro de Competência Embrapii em Tecnologias Imersivas (AKCIT), Universidade Federal de Goiás

---

## INTRODUÇÃO

A classificação de intenções em conversas de atendimento ao cliente é essencial para call centers em escala. Sistemas supervisionados tradicionais exigem retreinamento a cada nova classe, limitando a flexibilidade operacional.

**Objetivo:** Desenvolver uma arquitetura híbrida (BM25 + embeddings) para classificação de intenções em conversas PT-BR com **taxonomia dinâmica** (novas classes sem retreino).

**Hipóteses:**
- H1: Fusão léxico-semântica melhora sobre isolados
- H2: BM25 domina sobre embeddings em PT-BR
- H3: Encoder PT-BR > multilingual
- H4: Dados sintéticos são essenciais
- H5: Regras oferecem interpretabilidade sem custo

---

## MÉTODO

- **Dataset:** 2.120 conversas PT-BR (8 classes, balanceado), público no HuggingFace
- **Pipeline:** Turnos → Janelas de 5 turnos com [customer]/[agent] → Embeddings 384d → KNN por similaridade
- **6 métodos:** BM25 KNN, Embedding KNN, Hybrid, Rerank, Cascade, Routing
- **Motor de regras:** 4 famílias (lexical, estrutural, contextual, semântico)
- **Protocolo:** 5 splits, Wilcoxon + Holm-Bonferroni, 4 baselines, ablação de 5 componentes

*(Inserir Figura: diagrama da arquitetura)*

---

## RESULTADOS

### Ranking de Métodos (Macro-F1)

| # | Método | F1 |
|---|---|---|
| 1 | **BERTimbau+KNN** | **0,768** |
| 2 | Hybrid (BM25+Emb) | 0,748 |
| 3 | BM25-KNN | 0,748 |
| 4 | LogReg+MiniLM | 0,719 |
| 5 | MPNet+KNN | 0,716 |
| 6 | Embedding-KNN | 0,697 |

### Ablação

| Componente removido | Δ Macro-F1 |
|---|---|
| −Dados sintéticos | **−21,9pp** |
| −BM25 | −4,9pp |
| −Embeddings | −1,0pp |
| −Speaker markers | −0,6pp |

### Hipóteses

| H | Resultado |
|---|---|
| H1 | **Confirmada** (direcional, +1,1pp) |
| H2 | **Confirmada** (+4,8pp BM25>Emb) |
| H3 | **Confirmada** (+2,2pp BERTimbau) |
| H4 | **Confirmada** (−22pp sem sintéticos) |
| H5 | **Confirmada** (86,2% acc, −7,1% latência) |

*(Inserir Figuras: gráfico de barras dos métodos + gráfico de ablação)*

---

## CONCLUSÃO

- **BERTimbau+KNN** (encoder PT-BR) atinge melhor resultado (F1=0,77) sem retreino
- **BM25 domina** sobre embeddings em domínio com vocabulário restrito
- **Dados sintéticos** são críticos (+22pp) para corpus pequenos
- **Regras multi-sinal** oferecem interpretabilidade sem degradar accuracy
- **Taxonomia dinâmica:** novas classes adicionadas com exemplares, sem retreino
- **Dataset público:** `paulohenriquevn/talkex-augmented-pt-br`

---

## REFERÊNCIAS PRINCIPAIS

BRUCH, S. et al. Fusion Functions for Hybrid Retrieval. **ACM TOIS**, 2023.
SOUZA, F. et al. BERTimbau. **BRACIS**, 2020.
ZHANG, J. et al. DNNC: Few-Shot Intent Detection. **EMNLP**, 2020.
HARRIS, L. Lexical vs Semantic for Medical Docs. **arXiv**, 2025.
THAKUR, N. et al. BEIR Benchmark. **NeurIPS**, 2021.

---

## FINANCIAMENTO

Centro de Competências Embrapii em Tecnologias Imersivas (AKCIT)
Universidade Federal de Goiás
