# Protocolo de Execução — Step by Step

**Propósito:** Registro reprodutível de cada passo executado na pesquisa. Complementa o caderno de pesquisa (research-log.md), que registra o raciocínio. Este documento registra a **ação**.

---

## Sessão 1 — 2026-03-10: Definição da Pesquisa

### Step 1: Contextualização do projeto

**Ação:** Análise do estado atual do repositório TalkEx.

**Inputs:**
- Git status, últimos commits, arquivos modificados
- Documentação existente: CLAUDE.md, KB.md, KB_Complementar.md, PRD.md

**Achados:**
- Branch: main, com 8 arquivos modificados e 3 untracked (trabalho em progresso de expansão da DSL)
- Projeto maduro: 1822 testes, ~170 source files, 4 ADRs
- Trabalho em progresso: 8 novos operadores lexicais (word, stem, contains_all, not_contains, excludes_any, near, starts_with, ends_with), normalização de diacríticos, frontend atualizado
- KB e PRD documentam a arquitetura alvo com pipeline cascateado de 9 estágios

**Output:** Compreensão de que o TalkEx pode servir como artefato técnico de uma dissertação de mestrado.

---

### Step 2: Leitura e análise da base bibliográfica

**Ação:** Leitura de 5 trabalhos de referência, extraindo achados centrais e limitações.

**Papers lidos:**

#### Paper 1: Harris (2025)
- **Arquivo:** `deep_research/2505.11582v2.pdf` e `docs/pesquisas/2505.11582v2.pdf`
- **Título:** Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents
- **Leitura:** Páginas 1-5 (completo, 6 páginas)
- **Dados:** 1.472 docs médicos, 7 classes, 7 métodos de embedding, kNN classifier
- **Resultado-chave:** BM25 alcançou maior acurácia preditiva E foi mais rápido que todos os embeddings semânticos (word2vec, med2vec, MiniLM, mxbai)
- **Números relevantes:** Table II mostra BM25 com 1.020 dimensões e 8.288 bytes vs MiniLM com 384 dims e 3.200 bytes. Fig. 2 mostra F1 por classe.
- **Limitação identificada:** Apenas kNN como classificador. Domínio médico estruturado (vocabulário controlado).

#### Paper 2: Rayo et al. (2025)
- **Arquivo:** `deep_research/2502.16767v1.pdf`
- **Título:** A Hybrid Approach to Information Retrieval and Answer Generation for Regulatory Texts
- **Leitura:** 5 páginas (completo, COLING 2025)
- **Dados:** ObliQA dataset, 27.869 questões regulatórias, 40 documentos de Abu Dhabi Global Markets
- **Resultado-chave (Table 2):**
  - BM25 baseline: Recall@10 = 0.7611, MAP@10 = 0.6237
  - BM25 custom: Recall@10 = 0.7791, MAP@10 = 0.6415
  - Semântico puro: Recall@10 = 0.8103, MAP@10 = 0.6286
  - **Híbrido: Recall@10 = 0.8333, MAP@10 = 0.7016**
- **Fusão:** Score = α·Semântico + (1-α)·Lexical, com α = 0.65
- **Fine-tuning:** BGE-small-en-v1.5, 10 epochs, batch 64, lr 2e-4, MultipleNegativesRankingLoss
- **RAG:** GPT-3.5 Turbo alcançou RePASs = 0.57 (melhor entre 3 LLMs testados)
- **Limitação identificada:** Sem classificação multi-label. Sem regras. Textos regulatórios (formais, longos).

#### Paper 3: Huang & He (2025)
- **Arquivo:** `docs/pesquisas/2410.00927v3.pdf`
- **Título:** Text Clustering as Classification with LLMs
- **Leitura:** Páginas 1-5 + tabela de resultados (11 páginas no total)
- **Dados:** 5 datasets (ArxivS2S, GoEmo, Massive-I, Massive-D, MTOP-I), 18-102 clusters
- **Resultado-chave (Table 2):** Framework proposto supera K-means, DBSCAN, IDAS, PAS, Keyphrase Clustering, ClusterLLM em ACC, NMI e ARI. Performance próxima ao upper bound (LLM_known_labels).
- **Exemplo:** ArxivS2S ACC: 38.78 (proposto) vs 31.21 (K-means E5) vs 41.50 (upper bound)
- **Método:** 2 estágios — LLM gera labels em mini-batches → merge de labels similares → LLM classifica nos labels gerados
- **LLM usado:** GPT-3.5-turbo, batch size B = 15
- **Limitação identificada:** LLM online (custo). Sem retrieval. Sem rastreabilidade de evidência.

#### Paper 4: Lyu et al. (2025)
- **Arquivo:** `docs/pesquisas/2512.09444v1.pdf`
- **Título:** Advancing Text Classification with Large Language Models and Neural Attention Mechanisms
- **Leitura:** 5 páginas (completo)
- **Dados:** AG News dataset, 4 classes, ~120K textos de notícias
- **Resultado-chave (Table 1):**
  - BERT: P=0.87, R=0.85, F1=0.86, AUC=0.91
  - LSTM: P=0.82, R=0.80, F1=0.81, AUC=0.87
  - Transformer: P=0.85, R=0.83, F1=0.84, AUC=0.90
  - GAT: P=0.86, R=0.84, F1=0.85, AUC=0.89
  - **Proposto: P=0.90, R=0.88, F1=0.89, AUC=0.94**
- **Arquitetura:** Encoder → Attention (Q,K,V) → Combined pooling (mean + attention-weighted, com coeficiente α) → FC layer → Softmax
- **Sensibilidade:** Hidden dim peak em 512. Recall cai de 0.88 (balanced) para 0.80 (imbalance 1:6).
- **Limitação identificada:** Textos curtos (notícias). Sem multi-turn. Sem retrieval.

#### Paper 5: AnthusAI — Semantic Text Classification
- **Arquivo:** `docs/pesquisas/semantic-text-classification.md`
- **Fonte:** GitHub (repositório com Jupyter notebook)
- **Dados:** Dataset rotulado por GPT-3.5, classificação binária (questões sobre imigração espanhola)
- **Métodos comparados:** Word2Vec, BERT, OpenAI Ada-2, todos com Logistic Regression
- **Resultado-chave:** BERT e Ada-2 alcançaram desempenho equivalente e superior a Word2Vec
- **Conclusão deles:** "the most powerful and expensive models may not always be necessary"
- **Limitação identificada:** Classificação binária simples. Sem features heterogêneas. Sem domínio conversacional.

**Output:** Lacuna identificada — nenhum trabalho combina retrieval híbrido + classificação supervisionada com features heterogêneas + regras determinísticas auditáveis em dados conversacionais multi-turn.

---

### Step 3: Formulação da tese e hipóteses

**Ação:** Síntese dos achados da literatura em tese central + 4 hipóteses falsificáveis.

**Raciocínio:**
- Harris mostrou que lexical pode vencer semântico → precisa de baseline BM25 (H1)
- Rayo mostrou que híbrido vence isolados → nossa H1 testa isso em conversas
- Lyu mostrou que attention melhora classificação → H2 testa multi-nível + pooling
- Ninguém testou regras determinísticas como complemento a ML → H3
- Ninguém mediu custo-benefício de cascata em NLP conversacional → H4

**Output:**
- Tese central formulada (ver research-log.md)
- 4 hipóteses com critérios de confirmação/refutação numéricos
- Título em PT e EN

---

### Step 4: Estrutura de capítulos

**Ação:** Elaboração da estrutura de 7 capítulos da dissertação.

**Output:** Arquivo `docs/dissertacao/estrutura-capitulos.md`
- 7 capítulos detalhados com subseções
- Estimativa de extensão por capítulo (total: 128-179 páginas)
- Lista de referências-chave e referências a buscar

---

### Step 5: Desenho experimental

**Ação:** Elaboração do protocolo experimental para H1-H4.

**Output:** Arquivo `docs/dissertacao/desenho-experimental.md`
- Protocolo para cada hipótese: sistemas comparados, parâmetros, métricas, critérios de confirmação/refutação
- Seções sobre dataset, análise estatística, ablation studies, ameaças à validade
- Cronograma estimado (17-23 semanas)
- Checklist de reprodutibilidade

---

### Step 6: Seleção e análise exploratória do dataset

**Ação:** Identificação do dataset `RichardSakaguchiMS/brazilian-customer-service-conversations`, verificação do cache local, e análise exploratória completa.

#### Step 6.1: Busca de informações do dataset
- **Método:** WebFetch na página do HuggingFace
- **Achado:** 944 conversas, PT-BR, sintético (NVIDIA LLM), 9 intents, 3 sentimentos, 8 setores, Apache 2.0

#### Step 6.2: Verificação do cache local
- **Método:** `find` no diretório de cache do HuggingFace
- **Resultado:** Dataset encontrado em `~/.cache/huggingface/datasets/RichardSakaguchiMS___brazilian-customer-service-conversations`

#### Step 6.3: Carga e inspeção de schema
```python
from datasets import load_dataset
ds = load_dataset('RichardSakaguchiMS/brazilian-customer-service-conversations')
```
- **Resultado:** 3 splits (train: 755, val: 94, test: 95), columns: ['messages', 'id', 'metadata']
- **Schema:** messages = [{role, content}], metadata = {intent, sentiment, sector, turns, generated, generator}

#### Step 6.4: Distribuição de intents
- **Método:** Counter sobre metadata.intent por split
- **Resultado:** Distribuição quase uniforme (~108 por classe, ~11% cada)
- **Exceções:** elogio (98), outros (93) ligeiramente menores
- **Test set:** De 6 (elogio) a 17 (cancelamento) por classe — muito pequeno

#### Step 6.5: Distribuição de sentimentos e setores
- **Resultado:** Sentimentos: 33.7% / 33.3% / 33.1% (quase perfeito terço)
- **Setores:** 8 setores de 9.4% a 14.3%
- **Cruzamentos:** Intent×Sentimento ~35-36 por combinação, Intent×Setor ~15 por combinação (ambos uniformes)

#### Step 6.6: Análise de turnos e comprimento
- **Resultado:**
  - Turnos: min=6, max=8, mediana=8, 90.4% = exatamente 8 turnos
  - Palavras/mensagem: média 23.6, customer 16.6, agent 30.7
  - Palavras/conversa: média 184.6, min 79, max 321

#### Step 6.7: Análise lexical por intent
- **Método:** Tokenização, remoção de stopwords, contagem por intent
- **Resultado:** Palavras discriminativas identificadas (cancelar→cancelamento, erro→suporte_técnico, etc.)
- **Sobreposição:** "obrigado", "bom", "fazer" em 9/9 intents (poder discriminativo zero)

#### Step 6.8: Análise de diacríticos e padrões linguísticos
- **Método:** Unicode NFD decomposition para encontrar palavras acentuadas; contagem de gírias
- **Resultado:** 1.019 palavras únicas com diacríticos. Gírias frequentes: "vc" (1957x), "td" (1708x), "show" (733x)

#### Step 6.9: Amostra de conversas por intent
- **Método:** Seleção e exibição de conversas de cancelamento, reclamação, suporte_técnico
- **Resultado:** Confirmação de linguagem informal PT-BR com sinais lexicais claros para algumas classes

**Output completo:** Análise integrada no research-log.md com estatísticas, interpretações e identificação de limitações.

---

### Step 7: Documentação do processo

**Ação:** Criação do caderno de pesquisa (research-log.md) com mentalidade acadêmica + este protocolo de execução (steps-log.md).

**Output:**
- `docs/dissertacao/research-log.md` — Raciocínio, reflexões, questões abertas
- `docs/dissertacao/steps-log.md` — Este documento (passos reprodutíveis)

---

## Inventário de Artefatos — Sessão 1

| # | Artefato | Caminho | Natureza |
|---|----------|---------|----------|
| 1 | Estrutura de capítulos | `docs/dissertacao/estrutura-capitulos.md` | Planejamento |
| 2 | Desenho experimental | `docs/dissertacao/desenho-experimental.md` | Metodologia |
| 3 | Caderno de pesquisa | `docs/dissertacao/research-log.md` | Reflexão |
| 4 | Protocolo de execução | `docs/dissertacao/steps-log.md` | Reprodutibilidade |

## Sessão 2 — 2026-03-10: Expansão do Dataset

### Step 8: Decisão sobre estratégia de expansão do dataset

**Ação:** Mapeamento do cenário de datasets PT-BR disponíveis + decisão metodológica sobre estratégia de expansão.

#### Step 8.1: Busca de datasets PT-BR de atendimento

- **Método:** WebSearch em HuggingFace, Kaggle, GitHub, dados.gov.br
- **Busca por:** "customer service Portuguese dataset", "Reclame Aqui dataset", "conversas atendimento dataset", "call center Portuguese NLP"

**Datasets encontrados:**

| Dataset | Fonte | Tamanho | Formato | Labels | Licença |
|---------|-------|---------|---------|--------|---------|
| Consumidor.gov.br | dados.gov.br | 325k+/trimestre | Texto único (CSV) | segmento, assunto, problema | Gov. aberto |
| Reclame Aqui (rdemarqui) | GitHub | 7k telecom | Texto único | 14 categorias | Sem licença formal |
| B2W-Reviews01 | HuggingFace | 130k reviews | Texto único | 5-star rating | CC BY-NC-SA 4.0 |
| Olist E-Commerce | Kaggle | 100k orders | Reviews | Star rating | CC BY-NC-SA 4.0 |
| Bitext Customer Support | HuggingFace | 27k | Single-turn | 27 intents | Apache 2.0 |
| AxonData Call Center | HuggingFace | 10k horas | Multi-turn (áudio) | domain, intent | Comercial |
| PT-BR Sentiment (Kaggle) | Kaggle | Compilação | Texto único | Sentimento | — |

**Achado-chave:** Não existe dataset público de conversas multi-turn de call center em PT-BR além do RichardSakaguchiMS.

#### Step 8.2: Análise do pipeline de ingestão existente

- **Método:** Leitura de `demo/scripts/ingest_br_dataset.py` (128 linhas) e `demo/scripts/ingest_dataset.py` (239 linhas)
- **Formato de saída:** JSONL com schema: `{conversation_id, text, domain, topic, asr_confidence, audio_duration_seconds, word_count, source_file, sentiment}`
- **Pipeline downstream:** JSONL → TurnSegmenter → SlidingWindowBuilder → Embeddings → BM25 + Qdrant
- **Estado atual:** 944 conversas → 18.309 turnos → 6.421 janelas de contexto

#### Step 8.3: Avaliação de 4 estratégias de expansão

| Estratégia | Controle | Validade | Viabilidade | Limitação principal |
|-----------|----------|----------|-------------|---------------------|
| 1. Expansão puramente sintética | Alto | Baixa | Alta | Não melhora artificialidade |
| 2. Corpus semi-real (Consumidor.gov.br) | Baixo | Média | Média | Sem turnos multi-turn |
| 3. Conversacionalização de reclamações | Médio | Média | Média | Viés duplo (texto real + estrutura sintética) |
| **4. Sintética controlada + validação real** | **Alto** | **Média-Alta** | **Alta** | Complexidade de 2 corpora |

#### Step 8.4: Decisão

**Escolha: Estratégia 4 — Expansão sintética controlada + validação em dados reais.**

**Justificativa:**
1. Corpus primário sintético permite controle de variáveis para ablation studies
2. Corpus secundário real (Consumidor.gov.br) fornece verificação de transferência
3. Posicionamento acadêmico: "contribuição metodológica com verificação de transferência"

**Especificação do corpus primário expandido:**
- Alvo: ~3.500 conversas (de 944 para ~3.500)
- Turnos: variáveis 4-20 (distribuição log-normal, média 8, stdev 4)
- Distribuição de intents: realista e desbalanceada (dúvida ~35%, reclamação ~20%, cancelamento ~8%, elogio ~4%)
- Personas: 5 perfis (formal, informal, irritado, idoso, jovem/gírias)
- Splits: Train 70% / Val 15% / Test 15%, stratified, seed 42

**Especificação do corpus secundário (validação):**
- Fonte: Consumidor.gov.br (1 trimestre, CSV)
- Amostra: ~2.000 reclamações estratificadas
- Mapeamento: categorias oficiais → 9 intents
- Uso: avaliar classificadores treinados no corpus primário (modo Conv-only)

**Output:** Decisão documentada em `research-log.md` com raciocínio metodológico completo.

---

## Inventário de Artefatos — Sessão 2

| # | Artefato | Caminho | Natureza |
|---|----------|---------|----------|
| 5 | Decisão de expansão | `docs/dissertacao/research-log.md` (Sessão 2) | Decisão metodológica |

## Próximos Steps Previstos

- Step 9: Implementação do script de expansão sintética (geração de conversas com LLM)
- Step 10: Geração do corpus primário expandido (~3.500 conversas)
- Step 11: Análise exploratória do corpus expandido
- Step 12: Download e preparação do corpus Consumidor.gov.br
- Step 13: Mapeamento de categorias Consumidor.gov.br → 9 intents
- Step 14: Piloto do Exp. H1 com corpus expandido (viabilidade)
