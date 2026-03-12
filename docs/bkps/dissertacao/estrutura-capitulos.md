# Estrutura de Capítulos — Dissertação de Mestrado

## Título

**"TalkEx: Uma Arquitetura Híbrida em Cascata para Classificação e Retrieval de Conversas de Atendimento com Regras Semânticas Auditáveis"**

**Título em inglês:** "TalkEx: A Cascaded Hybrid Architecture for Conversation Classification and Retrieval with Auditable Semantic Rules"

---

## Tese Central

Uma arquitetura híbrida em cascata que combina retrieval lexical (BM25), retrieval semântico (embeddings densos) e um motor de regras determinísticas baseado em AST, operando sobre representações conversacionais em múltiplos níveis de granularidade (turno, janela de contexto, conversa), alcança qualidade de classificação e retrieval superior às abordagens que dependem de qualquer paradigma isolado — ao mesmo tempo em que preserva explicabilidade operacional e eficiência computacional — no domínio de análise de conversas de atendimento.

---

## Estrutura Detalhada

### Capítulo 1 — Introdução

**Objetivo:** Contextualizar o problema, motivar a pesquisa, apresentar a tese e antecipar as contribuições.

**Conteúdo:**

#### 1.1 Contexto e Motivação
- Volume massivo de conversas em call centers e canais digitais (dados: volume médio por mês em operações brasileiras de grande porte)
- Perda de valor informacional por falta de ferramentas adequadas de classificação e busca
- Limitações das abordagens atuais: classificação superficial de intents, busca puramente lexical, ausência de auditabilidade
- Relevância para compliance, retenção, qualidade operacional e CX

#### 1.2 Problema de Pesquisa
- **Pergunta central:** Como combinar retrieval lexical, retrieval semântico e regras determinísticas em uma arquitetura unificada que maximize qualidade, explicabilidade e eficiência na análise de conversas de atendimento?
- Desafios específicos do domínio conversacional:
  - Texto ruidoso (transcrição ASR, coloquialismo, gírias)
  - Dependências multi-turn (intenção emerge do contexto, não de frase isolada)
  - Variação linguística (paráfrases, diacríticos em PT-BR)
  - Exigência simultânea de qualidade, explicabilidade e custo controlado

#### 1.3 Hipóteses
- H1: Superioridade do retrieval híbrido sobre abordagens isoladas
- H2: Ganho da representação multi-nível sobre nível único
- H3: Complementaridade das regras determinísticas ao pipeline estatístico
- H4: Eficiência da inferência em cascata vs pipeline uniforme

#### 1.4 Objetivos
- **Objetivo geral:** Projetar, implementar e avaliar uma arquitetura híbrida em cascata para classificação e retrieval de conversas de atendimento
- **Objetivos específicos:**
  1. Implementar retrieval híbrido (BM25 + ANN) com fusão de scores e avaliar contra baselines isolados
  2. Projetar representações multi-nível (turno, janela, conversa) e avaliar impacto na classificação
  3. Projetar e implementar um motor de regras semânticas baseado em DSL/AST com evidência rastreável
  4. Avaliar a eficiência de custo da inferência cascateada

#### 1.5 Contribuições
- Framework arquitetural com 3 paradigmas complementares
- DSL auditável compilada para AST com evidência por decisão
- Estudo empírico comparativo no domínio conversacional
- Análise quantitativa do trade-off custo-qualidade

#### 1.6 Organização da Dissertação
- Mapa dos capítulos seguintes

**Extensão estimada:** 8-10 páginas

---

### Capítulo 2 — Fundamentação Teórica

**Objetivo:** Estabelecer os conceitos fundamentais que sustentam a dissertação. Este capítulo é didático — ensina o leitor os fundamentos necessários para compreender o trabalho.

**Conteúdo:**

#### 2.1 Representação Vetorial de Texto
- Embeddings: definição, intuição geométrica e propriedades
- Evolução: bag-of-words → TF-IDF → Word2Vec → BERT → Sentence Transformers
- Modelos de embedding para retrieval: E5, BGE, Instructor
- Dimensionalidade, pooling strategies (mean, max, attention) e impacto na tarefa

#### 2.2 Busca Lexical
- Term Frequency e Inverse Document Frequency
- BM25: formulação matemática, hiperparâmetros (k₁, b), propriedades
- Normalização textual: stemming, stopwords, remoção de diacríticos, tokenização
- Pontos fortes: velocidade, interpretabilidade, eficácia em vocabulário consistente
- Pontos fracos: incapacidade de capturar paráfrases e intenção implícita

#### 2.3 Busca Semântica
- Dense Passage Retrieval (DPR) e busca por vizinhos mais próximos
- Approximate Nearest Neighbor (ANN): HNSW, IVF, PQ
- Métricas de similaridade: cosseno, distância euclidiana
- Sentence-BERT e bi-encoders para matching semântico
- Pontos fortes: generalização semântica, paráfrases
- Pontos fracos: custo computacional, interpretabilidade baixa, sensibilidade a domínio

#### 2.4 Busca Híbrida e Fusão de Scores
- Motivação: complementaridade entre lexical e semântico
- Estratégias de fusão: combinação linear ponderada, Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking como estágio de refinamento
- Configuração: α (peso semântico vs lexical), top-K por estágio

#### 2.5 Classificação de Texto
- Classificação supervisionada: logistic regression, gradient boosting, MLP
- Multi-label e multi-class: diferenças e estratégias
- Features heterogêneas: embeddings + lexicais + estruturais + contextuais
- Princípio: "embeddings representam, classificadores decidem"
- Métricas: F1 (macro/micro), precision, recall, calibration error, AUC

#### 2.6 Análise de Conversas
- Estrutura conversacional: turnos, falantes, transições
- Context windows: janelas deslizantes sobre turnos
- Representação multi-nível: turno, janela, conversa, por papel (cliente/agente)
- Desafios específicos: ruído de ASR, coloquialismo, alternância de tópicos

#### 2.7 Motores de Regras e DSLs
- Sistemas baseados em regras: histórico e papel em sistemas de decisão
- DSL (Domain-Specific Language): definição, design e compilação
- AST (Abstract Syntax Tree): representação, traversal e avaliação
- Short-circuit evaluation e otimização por custo de predicado
- Explainability: rastreabilidade de evidência por nó da árvore

#### 2.8 Inferência em Cascata
- Princípio: aplicar processamento progressivamente mais caro
- Cascaded inference em sistemas de NLP: filtros baratos → modelos caros
- Trade-off custo-qualidade: onde economizar sem sacrificar recall
- Exemplos na indústria: Bing, Google, ElasticSearch

**Extensão estimada:** 25-35 páginas

---

### Capítulo 3 — Trabalhos Relacionados

**Objetivo:** Posicionar a pesquisa no estado da arte, demonstrando a lacuna que a dissertação preenche. Diferente do Cap. 2 (conceitos), aqui se discutem **trabalhos específicos** e seus limites.

**Conteúdo:**

#### 3.1 Retrieval Híbrido em Domínios Especializados
- **Rayo et al. (2025)** — Hybrid approach for regulatory texts
  - Contribuição: BM25 + fine-tuned BGE com α=0.65 supera abordagens isoladas
  - Limitação: domínio regulatório (textos longos, formais); sem classificação multi-label; sem regras
  - Relação com esta dissertação: validamos a hipótese híbrida em domínio conversacional (ruidoso, informal, multi-turn)

- **Gokhan et al. (2024)** — RegNLP baseline com BM25
  - Contribuição: baseline lexical robusto para textos regulatórios
  - Limitação: sem componente semântico
  - Relação: nosso BM25 com normalização accent-aware estende a abordagem para PT-BR

#### 3.2 Busca Lexical vs Semântica em Classificação
- **Harris (2025)** — Lexical vs semantic vector search for medical documents
  - Contribuição: BM25 supera embeddings semânticos off-the-shelf em docs estruturados; dados bespoke superam modelos genéricos
  - Limitação: classificação por kNN apenas (sem classificador supervisionado); documentos médicos estruturados (não conversacionais)
  - Relação: reforça nossa decisão de benchmark obrigatório contra BM25 e de usar classificadores supervisionados, não apenas similaridade

#### 3.3 Clustering como Classificação com LLMs
- **Huang & He (2025)** — Text Clustering as Classification with LLMs
  - Contribuição: LLMs geram e consolidam labels sem fine-tuning ou embeddings; desempenho próximo ao upper bound
  - Limitação: uso de LLM online (custo proibitivo em escala); sem integração com retrieval; sem rastreabilidade de evidência
  - Relação: adotamos LLMs para discovery offline (labeling, taxonomy), não para inferência online; nossas regras fornecem a auditabilidade que o LLM não oferece

#### 3.4 Classificação com Attention e LLMs
- **Lyu et al. (2025)** — Advancing text classification with LLMs and attention mechanisms
  - Contribuição: attention pooling + LLM encoder melhora F1/AUC vs LSTM, Transformer, GAT
  - Limitação: avaliado em textos curtos de notícias (AG News); sem contexto multi-turn; sem retrieval
  - Relação: fundamenta nossa escolha de attention pooling em janelas de contexto longas; estendemos para domínio conversacional

#### 3.5 Embeddings para Classificação
- **AnthusAI — Semantic Text Classification**
  - Contribuição: demonstra que embeddings (BERT, Ada-2) + logistic regression classifica com alta acurácia; BERT equipara modelos maiores em tarefas específicas
  - Limitação: classificação binária simples; sem features heterogêneas; sem domínio conversacional
  - Relação: adotamos o princípio "embeddings → classifier" como axioma; estendemos com features heterogêneas e multi-nível

#### 3.6 Conversation Intelligence em Call Centers
- Panorama de ferramentas comerciais (Observe.AI, CallMiner, Verint, NICE)
- Limitações: modelos black-box, sem auditabilidade, dependência de LLM online, custo proibitivo para volumes brasileiros
- Trabalhos acadêmicos em intent detection conversacional
- Gap: nenhum combina retrieval híbrido + classificação supervisionada + regras auditáveis

#### 3.7 Síntese e Posicionamento
- Tabela comparativa: dimensões x trabalhos (retrieval, classificação, regras, multi-turn, explainability, eficiência)
- Identificação explícita da lacuna
- Como esta dissertação preenche a lacuna

**Extensão estimada:** 15-20 páginas

---

### Capítulo 4 — Arquitetura Proposta: TalkEx

**Objetivo:** Descrever a arquitetura do sistema proposto em detalhe técnico suficiente para reprodução.

**Conteúdo:**

#### 4.1 Visão Geral do Pipeline
- Diagrama arquitetural completo (ingestion → segmentation → context → embeddings → indexing → retrieval → classification → rules → analytics)
- Princípios de design: modularidade, cascata, separação online/offline
- Fluxo de dados: como uma conversa bruta se transforma em insights classificados e auditáveis

#### 4.2 Modelo de Dados Conversacional
- Entidades: Conversation, Turn, ContextWindow, EmbeddingRecord, Prediction, RuleExecution
- Relações e granularidade
- Representação em Pydantic: frozen, strict, validadores
- Decisão: `list[float]` em modelos, `ndarray` em computação (ADR-003)

#### 4.3 Normalização e Pré-processamento
- Normalização textual: lowercase, remoção de diacríticos (Unicode NFD), pontuação
- Módulo `text_normalization`: `strip_accents()`, `normalize_for_matching()`
- Justificativa: essencial para PT-BR onde "não" e "nao", "cancelamento" e "cancelámento" devem ser equivalentes
- Integração: aplicada em BM25, predicados lexicais, predicados contextuais, regex

#### 4.4 Representações Multi-Nível
- **Turno:** embedding do turno individual, captura intenção local
- **Janela de contexto:** janela deslizante de N turnos adjacentes, captura dependências multi-turn
  - Parâmetros: window_size, stride, speaker alignment, recency weighting
- **Conversa:** embedding global, captura objetivo dominante e desfecho
- **Por papel:** visões separadas cliente/agente para compliance e detecção de intenção real
- Estratégia de pooling: mean pooling (baseline) vs attention pooling (experimental)

#### 4.5 Retrieval Híbrido
- **Componente lexical:** BM25 com normalização accent-aware
  - Tokenização: lowercase → strip_accents → remove_punctuation
  - Hiperparâmetros: k₁ = 1.5, b = 0.75
- **Componente semântico:** ANN sobre embeddings (FAISS/HNSW)
  - Modelos candidatos: E5, BGE, Instructor
- **Fusão de scores:** combinação linear ponderada (Score = α · semantic + (1-α) · lexical)
  - Variação: Reciprocal Rank Fusion (RRF)
- **Reranking opcional:** cross-encoder sobre shortlist
- Pipeline: BM25 top-K + ANN top-K → union → fusão → rerank → filtros de negócio

#### 4.6 Classificação Supervisionada
- Features heterogêneas: embeddings (turno + janela + conversa) + scores BM25 contra protótipos + distância a centroides + entidades + flags de regras + metadados
- Modelos candidatos:
  - Logistic Regression (baseline)
  - Gradient Boosting (LightGBM/XGBoost) para features heterogêneas
  - MLP para features densas
- Multi-label e multi-class em múltiplas granularidades
- Output: label, score, confidence, threshold, evidence, model_version

#### 4.7 Motor de Regras Semânticas (DSL → AST)
- **DSL:** sintaxe legível (`RULE ... WHEN ... THEN ...`)
- **Parser:** transforma DSL em AST
- **Predicados suportados:**
  - Lexicais: contains, contains_any, contains_all, word, stem, not_contains, excludes_any, near, starts_with, ends_with, regex, bm25_score
  - Semânticos: intent_score, embedding_similarity, topic_score
  - Estruturais: speaker, turn_index, duration, channel
  - Contextuais: repeated_in_window, occurs_after, count_in_window
- **Executor:** avaliação com short-circuit ordenada por custo
- **Evidência:** cada nó da AST produz metadata rastreável (matched_words, scores, thresholds, actual_distance)
- **Versionamento:** versão da regra, data, autor, compatibilidade com modelo

#### 4.8 Inferência em Cascata
- Estágio 1: filtros baratos (idioma, canal, fila, regras lexicais simples)
- Estágio 2: retrieval híbrido (BM25 + ANN)
- Estágio 3: classificação supervisionada + regras semânticas
- Estágio 4: revisão excepcional (LLM offline para casos ambíguos)
- Lógica de decisão: quando promover para estágio seguinte vs terminar cedo

#### 4.9 Decisões Arquiteturais (ADRs)
- Referência aos ADRs existentes (ADR-001 a ADR-004)
- Justificativas para escolhas tecnológicas

**Extensão estimada:** 25-35 páginas

---

### Capítulo 5 — Desenho Experimental e Metodologia

**Objetivo:** Descrever com rigor metodológico como cada hipótese será testada. Detalhe suficiente para **reprodutibilidade**.

**Conteúdo:** Ver documento `desenho-experimental.md` para detalhamento completo.

#### 5.1 Dataset
- Descrição da(s) fonte(s) de dados
- Estatísticas descritivas: volume, distribuição de classes, comprimento médio de turnos/conversas
- Pré-processamento e splits (train/val/test — holdout temporal quando aplicável)
- Critérios de anotação e qualidade dos labels

#### 5.2 Métricas de Avaliação
- Retrieval: Recall@K, MRR, nDCG, Precision@K
- Classificação: Macro-F1, Micro-F1, Precision/Recall por classe, AUC, calibration error
- Regras: Precision/Recall por regra crítica, false positive burden, cobertura
- Eficiência: latência (p50/p95/p99), custo por conversa, throughput

#### 5.3 Protocolo Experimental para H1 (Retrieval Híbrido)
#### 5.4 Protocolo Experimental para H2 (Representação Multi-Nível)
#### 5.5 Protocolo Experimental para H3 (Regras Determinísticas)
#### 5.6 Protocolo Experimental para H4 (Inferência em Cascata)
#### 5.7 Análise Estatística
#### 5.8 Ameaças à Validade

**Extensão estimada:** 15-20 páginas

---

### Capítulo 6 — Resultados e Análise

**Objetivo:** Apresentar os resultados experimentais de forma objetiva e analisá-los criticamente.

**Conteúdo:**

#### 6.1 Resultados do Retrieval Híbrido (H1)
- Tabelas comparativas: BM25 vs ANN vs Híbrido
- Gráficos: Recall@K para diferentes valores de K
- Análise: em quais tipos de query o híbrido ganha e em quais perde
- Impacto do parâmetro α (peso semântico vs lexical)
- Impacto da normalização de diacríticos no BM25

#### 6.2 Resultados da Representação Multi-Nível (H2)
- Tabelas: F1 por nível (turno-only, janela-only, conversa-only, multi-nível)
- Análise por classe: quais intents se beneficiam mais de contexto
- Impacto do tamanho da janela (3, 5, 7, 10 turnos)
- Mean pooling vs attention pooling

#### 6.3 Resultados das Regras Determinísticas (H3)
- Precision/Recall com e sem regras em classes críticas
- Exemplos qualitativos: decisões que as regras acertam e os modelos erram (e vice-versa)
- Análise de evidência: qualidade e utilidade da rastreabilidade
- Custo adicional de latência por regra

#### 6.4 Resultados da Inferência em Cascata (H4)
- Custo computacional: pipeline uniforme vs cascata
- Degradação de F1 por estágio de cascata
- Curva de Pareto: custo vs qualidade para diferentes configurações
- Análise: quais conversas são resolvidas em estágios baratos vs caros

#### 6.5 Análise de Erro
- Categorias de erro mais frequentes
- Erros por tipo de conversa (canal, duração, complexidade)
- Impacto do ruído de ASR na qualidade
- Classes raras e desbalanceamento

#### 6.6 Discussão dos Resultados
- Confirmação ou refutação de cada hipótese
- Resultados inesperados e possíveis explicações
- Comparação com resultados da literatura (Harris, Rayo, Huang, Lyu)

**Extensão estimada:** 20-30 páginas

---

### Capítulo 7 — Conclusão

**Objetivo:** Sintetizar as contribuições, limitações e direções futuras.

**Conteúdo:**

#### 7.1 Síntese das Contribuições
- Recapitulação da tese e dos resultados que a sustentam
- Contribuição 1: framework arquitetural cascateado
- Contribuição 2: DSL auditável com evidência
- Contribuição 3: evidência empírica da complementaridade
- Contribuição 4: análise custo-qualidade

#### 7.2 Limitações
- Dependência do(s) dataset(s) utilizados — generalização para outros domínios
- Ausência de ASR real (se usar apenas texto transcrito)
- Escala: experimentos em volume limitado vs produção com milhões
- Regras: cobertura limitada ao que é codificado manualmente
- Embedding models: avaliados off-the-shelf, sem fine-tuning no domínio

#### 7.3 Trabalhos Futuros
- Fine-tuning de embedding models para o domínio de call center
- Active learning para redução de custo de anotação
- Discovery offline com LLMs (clustering como classificação — Huang & He)
- Avaliação em produção com shadow deployment
- Extensão para outros idiomas e canais
- Integração com RAG para respostas geradas (inspirado em Rayo et al.)

#### 7.4 Considerações Finais
- Reflexão sobre o impacto prático da pesquisa
- Conexão com o cenário brasileiro de atendimento

**Extensão estimada:** 6-8 páginas

---

### Referências Bibliográficas

**Formato:** ACM ou IEEE (verificar norma do programa)

**Referências-chave já identificadas:**
1. Harris, L. (2025). Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents. arXiv:2505.11582v2
2. Rayo, J., de la Rosa, R., Garrido, M. (2025). A Hybrid Approach to Information Retrieval and Answer Generation for Regulatory Texts. COLING 2025. arXiv:2502.16767v1
3. Huang, C., He, G. (2025). Text Clustering as Classification with LLMs. SIGIR-AP 2025. arXiv:2410.00927v3
4. Lyu, N., Wang, Y., Chen, F., Zhang, Q. (2025). Advancing Text Classification with Large Language Models and Neural Attention Mechanisms. arXiv:2512.09444v1
5. Robertson, S.E. et al. (1996). Okapi at TREC-4. (BM25)
6. Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. arXiv:1810.04805
7. Reimers, N., Gurevych, I. (2019). Sentence-BERT. EMNLP 2019
8. Karpukhin, V. et al. (2020). Dense Passage Retrieval. arXiv:2004.04906
9. Lewis, P. et al. (2021). Retrieval-Augmented Generation. arXiv:2005.11401
10. Xiao, S. et al. (2023). C-Pack: BGE embeddings. arXiv:2309.07597

**Referências adicionais necessárias:**
- Sentence Transformers (E5, Instructor)
- FAISS (Johnson et al.)
- scikit-learn, XGBoost/LightGBM
- Trabalhos em conversation intelligence e call center analytics
- DSL design e AST evaluation
- Cascaded inference em NLP

### Apêndices

- **Apêndice A:** Especificação completa da DSL (gramática, operadores, exemplos)
- **Apêndice B:** Exemplos de regras completas com evidência
- **Apêndice C:** Detalhes de configuração experimental (hiperparâmetros, hardware)
- **Apêndice D:** Análise qualitativa de conversas (exemplos anonimizados)

---

## Estimativa de Extensão

| Capítulo | Páginas |
|----------|---------|
| 1. Introdução | 8-10 |
| 2. Fundamentação Teórica | 25-35 |
| 3. Trabalhos Relacionados | 15-20 |
| 4. Arquitetura Proposta | 25-35 |
| 5. Desenho Experimental | 15-20 |
| 6. Resultados e Análise | 20-30 |
| 7. Conclusão | 6-8 |
| Referências | 4-6 |
| Apêndices | 10-15 |
| **Total** | **128-179** |

Faixa típica para dissertação de mestrado em Ciência da Computação: **80-150 páginas**. Se necessário, comprimir Cap. 2 (Fundamentação) priorizando conceitos diretamente usados na dissertação.
