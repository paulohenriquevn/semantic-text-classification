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

### Step 9: Implementação do script de expansão sintética

**Ação:** Criação de dois scripts para expansão e split do dataset.

#### Step 9.1: Análise do formato existente

- **Método:** Leitura de `demo/scripts/ingest_br_dataset.py` e `demo/data/conversations.jsonl`
- **Schema JSONL:** `{conversation_id, text, domain, topic, asr_confidence, audio_duration_seconds, word_count, source_file, sentiment}`
- **Formato de texto:** `[customer] ... \n[agent] ...` (turnos concatenados com `\n`)
- **Metadados do dataset original:** 9 intents, 8 setores, 3 sentimentos, metadata.generated=true, metadata.generator="nvidia"

#### Step 9.2: Extração de valores únicos do dataset

```python
from datasets import load_dataset
ds = load_dataset('RichardSakaguchiMS/brazilian-customer-service-conversations')
# Intents: cancelamento, compra, duvida_produto, duvida_servico, elogio, outros, reclamacao, saudacao, suporte_tecnico
# Sentiments: negative, neutral, positive
# Sectors: ecommerce, educacao, financeiro, imobiliario, restaurante, saude, tecnologia, telecom
```

#### Step 9.3: Implementação do script de expansão

- **Script:** `experiments/scripts/expand_dataset.py`
- **Linguagem:** Python 3.11+ com click, anthropic SDK
- **Funcionalidades:**
  - Plano de geração determinístico (seed=42) com distribuição controlada
  - 5 personas com descrições detalhadas para guiar o LLM
  - 9 intents com descrições contextuais
  - 8 setores com contexto de negócio
  - Turnos via distribuição log-normal (4-20, média ~8)
  - Few-shot examples do dataset original (2 por conversa)
  - Checkpoint/resume para interrupções
  - Rate limiting configurável
  - Modo `--dry-run` para visualizar o plano sem gerar
- **Modelo padrão:** `claude-sonnet-4-20250514`
- **Temperatura:** 0.9 (alta diversidade)

#### Step 9.4: Implementação do script de splits

- **Script:** `experiments/scripts/build_splits.py`
- **Funcionalidades:**
  - Combina dataset original + expandido
  - Split estratificado preservando distribuição de intents
  - Gera train.jsonl, val.jsonl, test.jsonl, all.jsonl
  - Manifest com metadados de contagem e distribuição
  - Seed fixo: 42

#### Step 9.5: Validação com dry-run

```bash
python3 experiments/scripts/expand_dataset.py --dry-run
```

**Resultado do plano de geração:**
- Total a gerar: 2.556 conversas (944 existentes → 3.500 alvo)
- Distribuição de intents: reclamacao 511 (20.0%), duvida_produto 460 (18.0%), duvida_servico 435 (17.0%), suporte_tecnico 383 (15.0%), compra 256 (10.0%), cancelamento 204 (8.0%), saudacao 128 (5.0%), elogio 102 (4.0%), outros 77 (3.0%)
- Distribuição de turnos: 4t=281, 6t=647, 8t=670, 10t=457, 12t=245, 14t=129, 16t=64, 18t=32, 20t=31
- Distribuição de personas: ~20% cada (uniforme, como planejado)
- Distribuição de setores: ~12-14% cada (uniforme, como planejado)

**Output:**
- `experiments/scripts/expand_dataset.py` — Script de geração sintética (350+ linhas)
- `experiments/scripts/build_splits.py` — Script de combinação e split (150+ linhas)

---

## Inventário de Artefatos — Sessão 2

| # | Artefato | Caminho | Natureza |
|---|----------|---------|----------|
| 5 | Decisão de expansão | `docs/dissertacao/research-log.md` (Sessão 2) | Decisão metodológica |
| 6 | Script de expansão | `experiments/scripts/expand_dataset.py` | Implementação |
| 7 | Script de splits | `experiments/scripts/build_splits.py` | Implementação |

### Step 10: Geração do corpus primário expandido

**Ação:** Execução do script de expansão com a API Anthropic (Claude Sonnet).

#### Step 10.1: Teste de qualidade (5 conversas)

- **Comando:** `python3 experiments/scripts/expand_dataset.py --output experiments/data/test_sample.jsonl --target-total 949`
- **Resultado:** 5/5 conversas geradas com sucesso, 0 falhas, tempo: 1.7 min
- **Validação de qualidade:**
  - Persona "idoso": linguagem formal, "minha filha", "por gentileza", "o senhor" ✓
  - Persona "jovem": "mano", "pfv", "msm", "pq", gírias ✓
  - Persona "irritado": MAIÚSCULAS, cobranças diretas, tom agressivo ✓
  - Turnos alvo = turnos reais em todas as 5 conversas ✓
  - Detalhes concretos de setor (R$ 89,90, 200 Mbps, nomes de produtos) ✓

#### Step 10.2: Geração completa

- **Comando:** `python3 experiments/scripts/expand_dataset.py --output experiments/data/expanded.jsonl --target-total 3500 --batch-delay 0.5`
- **Modelo:** claude-sonnet-4-20250514
- **Temperatura:** 0.9
- **Alvo:** 2.556 novas conversas
- **Estimativa de tempo:** ~2-3 horas
- **Estimativa de custo:** ~$15-25 USD
- **Status:** EM EXECUÇÃO (background)

**Output esperado:** `experiments/data/expanded.jsonl`

---

### Step 10.3: Validação de dificuldade do dataset original (Phase 0.5)

**Ação:** Medir a dificuldade intrínseca da task de classificação ANTES da expansão, para ter baseline.

- **Script:** `experiments/scripts/validate_dataset.py`
- **Comando:** `python3 experiments/scripts/validate_dataset.py --input demo/data/conversations.jsonl`

**Resultados:**

| Check | Resultado | Interpretação |
|-------|-----------|---------------|
| Majority baseline | 11.4% accuracy (9 classes, ~11% cada) | Classes quase uniformes — imbalance ratio 1.2 |
| Random baseline | 11.1% accuracy | Muito próximo de majority → classes equilibradas |
| Lexical exclusivity | Mean 1.68 | GOOD: baixa exclusividade → task genuinamente difícil |
| Cross-intent overlap | 100% no top-20 | GOOD: palavras mais comuns (ajudar, beleza, posso) aparecem em TODAS as 9 classes |
| Risco identificado | Classes too balanced | Uniforme artificial — call centers reais são desbalanceados |

**Achados-chave por intent:**
- **cancelamento** (exclusividade 2.87): mais discriminativo — "cancelar" é quase exclusivo
- **elogio** (2.50): "obrigado", "bom" discriminam parcialmente
- **suporte_técnico** (2.16): "erro", "tentar", "consigo" são sinais
- **duvida_servico** (0.99): MAIS DIFÍCIL — sem palavras discriminativas exclusivas

**Significância para a pesquisa:** O dataset original tem dificuldade genuína (overlap lexical alto), exceto por cancelamento/elogio que têm sinais lexicais fortes. A expansão deve preservar essa característica — se o corpus expandido tiver exclusividade > 3.0, o LLM gerador criou sinais artificiais.

**Output:** `experiments/data/validation_report.json`

---

### Step 10.4: Ajustes metodológicos no script de expansão

**Ação:** Adição de rastreamento de few-shot IDs para auditoria de leakage.

- **Campo adicionado:** `metadata.few_shot_ids` — lista de `conversation_id` das conversas originais usadas como exemplo
- **Propósito:** Permitir que o split estratificado evite colocar uma conversa original no test set quando suas derivadas estão no train set

**Ação:** Atualização do `build_splits.py` para produzir output complementar ao pipeline TalkEx.

- **Output 1:** `demo/data/conversations.jsonl` — corpus unificado para `build_index.py`
- **Output 2:** `demo/data/splits/{train,val,test}.jsonl` — splits para classificação
- **Input original preservado:** `demo/data/conversations_original.jsonl`

---

## Inventário de Artefatos — Sessão 2

| # | Artefato | Caminho | Natureza |
|---|----------|---------|----------|
| 5 | Decisão de expansão | `docs/dissertacao/research-log.md` (Sessão 2) | Decisão metodológica |
| 6 | Script de expansão | `experiments/scripts/expand_dataset.py` | Implementação |
| 7 | Script de splits | `experiments/scripts/build_splits.py` | Implementação |
| 8 | Script de validação | `experiments/scripts/validate_dataset.py` | Implementação |
| 9 | Relatório de validação (original) | `experiments/data/validation_report.json` | Dados |

### Step 10.5: Investigação do Consumidor.gov.br (descartado)

**Ação:** Examinar dicionário de dados e viabilidade do Consumidor.gov.br como corpus de validação externa.

- **Método:** WebFetch do dicionário de dados (PDF), análise das colunas do CSV
- **Achado:** O CSV contém 30 colunas, **nenhuma com texto livre**. Os campos mais próximos são categóricos: Área (L1), Assunto (L2), Grupo Problema (L3), Problema (L4).
- **Problema adicional:** Plataforma exclusivamente de reclamações — impossível mapear elogio, saudação, dúvida.
- **Decisão:** Consumidor.gov.br **descartado** como corpus de validação.
- **Alternativas avaliadas:** B2W-Reviews01 (só ratings), Reclame Aqui (legalmente cinza, só reclamações). Nenhuma viável.
- **Impacto:** Estratégia revisada — robustez via Phase 6.1 (ablation no original) em vez de validação externa.
- **Documentação:** Decisão e raciocínio completo em `research-log.md` (Sessão 2, seção "Decisão: Consumidor.gov.br descartado").

---

---

## Sessão 3 — 2026-03-11: Revisão de literatura e preparação de dados

### Step 11: Revisão de literatura — Cascaded inference e paradigmas híbridos

**Ação:** Busca sistemática de trabalhos acadêmicos sobre inferência cascateada, early exit, hybrid retrieval e regras + ML.

**Método:** Web search com queries: "cascaded inference NLP", "multi-stage classification cost-quality tradeoff", "early exit transformer", "hybrid retrieval BM25 dense", "rule systems machine learning NLP".

**Resultados:**
- **22 trabalhos encontrados** em 7 categorias (ver `research-log.md`, Sessão 3 para lista completa)
- **Trabalhos-chave para H4:** Varshney & Baral (2022) — 88.93% redução de custo; FrugalML (2020) — 90% redução; DeeBERT (2020) — ~40% redução
- **Para H1:** DPR (Karpukhin et al., 2020), RRF (Ma et al., 2021), Lin et al. (2023) survey
- **Para H3:** SystemT (Chiticariu et al., 2010), Snorkel (Ratner et al., 2017), Safranchik et al. (2020)

**Achado central — Lacuna confirmada:**
Nenhum dos 22 trabalhos combina os 3 paradigmas (retrieval híbrido + classificação multi-nível + regras determinísticas) sobre dados conversacionais. A contribuição do TalkEx é genuína.

**Implicação para H4:** Target de ≥40% redução de custo é conservador (literatura mostra 40-90%). Precaução: nosso cenário é mais simples (9 classes, domínio único) — ser honesto na discussão.

**Output:** Documentação completa em `research-log.md` (Sessão 3).

---

### Step 11.1: Correção da geração — intent-sentiment coherence

**Issue descoberta:** Análise dos 144 registros gerados revelou 9% de combinações incoerentes (elogio+negative, reclamação+positive, saudação+negative). O LLM prioriza o sentimento sobre o intent, produzindo texto de reclamação rotulado como "elogio".

**Causa raiz:** Distribuição de sentimento era global (40% negative, 30% neutral, 30% positive), aplicada uniformemente a todos os intents. Sem restrição de que "elogio" deve ser positive/neutral e "reclamação" deve ser negative/neutral.

**Correção aplicada:**
1. Adicionado `INTENT_SENTIMENT_CONSTRAINTS` — distribuição de sentimento por intent:
   - `elogio`: 75% positive, 25% neutral (nunca negative)
   - `reclamacao`: 65% negative, 35% neutral (nunca positive)
   - `saudacao`: 40% positive, 60% neutral (nunca negative)
   - Demais intents: distribuições realistas com todos os sentimentos permitidos
2. Adicionado `few_shot_ids` no metadata (tracking para auditoria de leakage)
3. Processo anterior (PID 219622, 144/2.556 registros) terminado e dados apagados
4. Nova geração iniciada com script corrigido — seed=42, mesma sequência determinística mas com sentimentos coerentes

**Verificação:** Primeiro registro confirmado: `conv_synth_03327`, intent=elogio, sentiment=positive, few_shot_ids presente.

**Artefato auxiliar:** `experiments/scripts/patch_few_shot_ids.py` — script de replay determinístico preparado (não mais necessário, mas mantido como fallback).

**Custo da correção:** ~$0.50-1.00 em API (144 registros descartados). Decisão correta: 229 registros mislabeled (~9%) comprometeriam a classificação.

---

### Step 11.2: Revisão de literatura — NLP em contact centers

**Ação:** Busca por "call center conversation classification NLP", "contact center intent detection", "Portuguese customer service NLP".

**Resultados:**
- **13 papers encontrados** em 3 tiers de relevância
- **Paper-chave:** Shah et al. (2023) — revisão sistemática de 125 papers sobre NLP em contact centers (Springer). Confirma fragmentação do campo.
- **Único paper PT-BR:** BERTau (Finardi et al., 2021) — BERT treinado em conversas do Itau Unibanco. Limitado a FAQ retrieval, sentimento e NER.
- **Multi-turn:** MINT-CL (2024, CIKM 2025) — contrastive learning para classificação multi-turn de intents.
- **Call center imbalanced:** MDPI (2025) — KoBERT + EDA para classificação com desbalanceamento em call center coreano.
- **Gap mantido:** Nenhum paper combina os 3 paradigmas em dados conversacionais.

**Output:** Documentação completa em `research-log.md` (Busca 2).

---

### Step 11.3: Revisão de literatura — DSL e regras para NLP

**Ação:** Busca por "domain-specific language NLP rules", "declarative rule systems text classification", "rule-based + ML hybrid NLP", "weak supervision rules NLP".

**Resultados:**
- **15 trabalhos/sistemas encontrados** em 4 categorias (regras puras, weak supervision, híbridos, neuro-simbólico)
- **Sistemas de regras DSL:** SystemT/AQL (IBM, 2010), UIMA Ruta (2016), GATE/JAPE (2000+), spaCy Matcher
- **Paper seminal:** Chiticariu et al. (2013) — "Rule-Based IE is Dead! Long Live Rule-Based IE Systems!" (EMNLP) — regras dominam indústria
- **Weak supervision:** Snorkel (2017), Snorkel DryBell (Google, 2019), skweak (2021), WRENCH benchmark (2021)
- **Híbrido ML+regras:** Villena-Roman (2011) — kNN + regras como pós-processamento (mais próximo precedente)
- **Sistema mais próximo do TalkEx:** UIMA Ruta — DSL + introspecção, mas sem predicados semânticos

**Achado crucial:** AST-based rule evaluation para NLP é pouco representado. Weak supervision (Snorkel) usa regras para gerar labels, não para inferência auditável em tempo real. O TalkEx combina ambos os usos.

**Output:** Documentação completa em `research-log.md` (Busca 3).

---

### Step 11.4: Síntese da revisão de literatura

**Gap confirmado em 3 buscas independentes (Buscas 1, 2, 3):**

Tabela expandida com 14 trabalhos × 5 dimensões: nenhum cobre todas as 5 dimensões que o TalkEx integra (hybrid retrieval + multi-level classification + deterministic rules + conversational data + PT-BR).

**10 papers prioritários identificados para Cap. 3**, mapeados a seções específicas. Ver `research-log.md` (tabela "Papers prioritários para Cap. 3").

**Formulação da novidade:**
> "Nenhum trabalho existente combina retrieval lexical (BM25), embeddings semânticos e regras determinísticas auditáveis numa arquitetura integrada operando sobre representações multi-nível de conversas multi-turn."

---

## Inventário de Artefatos — Sessão 3

| # | Artefato | Caminho | Natureza |
|---|----------|---------|----------|
| 10 | Script de patch (fallback) | `experiments/scripts/patch_few_shot_ids.py` | Implementação |
| 11 | Revisão de literatura (22+13+15 papers) | `docs/dissertacao/research-log.md` (Sessão 3) | Pesquisa |

---

## Próximos Steps Previstos

- Step 12: Aguardar geração completa (~3-4h) → verificar falhas e qualidade
- Step 13: Validação de dificuldade do corpus expandido (comparar com original)
- Step 14: Combinação e split do corpus final (build_splits.py)
- Step 15: Rebuild indexes via TalkEx pipeline (build_index.py)
- Step 16: Piloto do Exp. H1 com corpus expandido (viabilidade)
