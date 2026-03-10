# PRD — TalkEx — Conversation Intelligence Engine (Call Center)

**Versão:** 1.0
**Status:** Draft
**Owner:** Product + ML Platform + NLP Engineering
**Data:** 2026-03-09

---

## 1. Visão Geral

O **TalkEx — Conversation Intelligence Engine** é uma plataforma de processamento semântico para ambientes de atendimento (voz, chat e canais digitais) desenhada para analisar, classificar, buscar, auditar e extrair inteligência operacional de milhões de conversas. O sistema combina **transcrição**, **segmentação de turnos**, **janelas de contexto conversacional**, **embeddings semânticos**, **busca híbrida (lexical + vetorial)**, **classificação multi-label**, **reranking** e um **motor de regras semânticas auditável baseado em AST**.

O objetivo central é transformar conversas brutas em sinais acionáveis para operação, qualidade, compliance, produto, retenção, cobrança, fraude, atendimento e analytics.

A arquitetura deve equilibrar quatro exigências simultâneas:

1. **Qualidade semântica** suficiente para lidar com paráfrases, contexto multi-turn e linguagem natural ruidosa.
2. **Escala operacional** para processar milhões de conversas por mês com latência previsível.
3. **Auditabilidade e governança** para cenários críticos como compliance, fraude e reclamações regulatórias.
4. **Eficiência de custo** via cascatas de inferência, uso seletivo de LLMs e arquitetura híbrida.

---

## 2. Problema de Negócio

Operações de call center e atendimento digital acumulam grandes volumes de interações não estruturadas. Hoje, grande parte do valor contido nessas conversas é perdido por limitações como:

* classificação superficial de intents e motivos de contato;
* dificuldade de descobrir novos padrões semânticos emergentes;
* baixa capacidade de auditoria sobre regras e decisões;
* busca ineficiente no histórico de conversas;
* dependência de taxonomias manuais lentas de manter;
* alto custo para analisar volumes massivos de texto com qualidade;
* pouca integração entre sinais semânticos, regras de negócio e metadados operacionais.

Essas limitações afetam diretamente:

* taxa de resolução no primeiro contato;
* retenção e churn prevention;
* monitoria de qualidade;
* compliance e risco;
* identificação de falhas de produto/processo;
* produtividade analítica;
* governança sobre decisões automatizadas.

---

## 3. Objetivos do Produto

### 3.1 Objetivos primários

* Classificar automaticamente conversas, turnos e janelas de contexto em múltiplas taxonomias operacionais.
* Permitir busca semântica e lexical sobre grandes volumes históricos.
* Detectar padrões críticos de compliance, risco, retenção, objeção, fraude e experiência.
* Descobrir novos intents e tópicos emergentes offline.
* Fornecer explicabilidade operacional por evidência textual, scores e regras acionadas.
* Reduzir custo analítico manual e aumentar velocidade de investigação.

### 3.2 Objetivos secundários

* Criar base reutilizável para analytics, copilots internos e retrieval inteligente.
* Suportar evolução contínua de taxonomias e regras com versionamento.
* Permitir avaliação padronizada de embeddings, classificadores e estratégias de retrieval.

### 3.3 Não-objetivos

* Não substituir integralmente agentes humanos.
* Não operar como decisão autônoma de alto impacto sem camada de governança.
* Não depender de LLM online para todo o tráfego produtivo.
* Não assumir que embeddings semânticos bastam sem baseline lexical e validação supervisionada.

---

## 4. Usuários e Stakeholders

### 4.1 Usuários principais

**Operações / Qualidade**
Precisam localizar conversas por intenção, falha, comportamento do agente e desvio de script.

**Compliance / Risco / Jurídico**
Precisam detectar falas, padrões e sequências específicas com rastreabilidade e evidência.

**Retention / CX / Produto**
Precisam entender churn drivers, objeções, atritos e causas-raiz.

**Data Science / NLP / ML Platform**
Precisam de uma plataforma mensurável e extensível para experimentação controlada.

**BI / Analytics**
Precisam consultar grandes volumes com filtros híbridos e produzir dashboards operacionais.

### 4.2 Stakeholders

* VP/Head de Operações
* Head de CX
* Compliance Officer
* Product Analytics
* Engenharia de Dados
* MLOps / Platform
* Segurança / Privacidade / Governança

---

## 5. Escopo Funcional

### 5.1 Entradas suportadas

* áudio de chamadas;
* transcrições de voz;
* chats de atendimento;
* e-mails e tickets textuais;
* metadados de CRM;
* dados operacionais de fila/canal/produto;
* labels humanos e revisões.

### 5.2 Saídas do sistema

* intents e sub-intents;
* motivo de contato;
* tópicos e clusters emergentes;
* flags de compliance e risco;
* scores semânticos por classe;
* evidências e explicações;
* embeddings por nível (turno, janela, conversa);
* resultados de busca híbrida;
* eventos acionáveis para downstreams.

---

## 6. Casos de Uso Prioritários

1. **Intent detection multi-label** por turno, janela e conversa.
2. **Busca histórica** por semântica, termos, entidades e filtros estruturais.
3. **Detecção de risco de cancelamento** e oportunidades de retenção.
4. **Monitoria de compliance** com regras semânticas auditáveis.
5. **Discovery offline** de novos motivos de contato.
6. **Classificação de causa-raiz** de falhas operacionais e de produto.
7. **Análise de comportamento do agente** e aderência a script.
8. **Criação de datasets rotulados** via revisão humana e LLM assistivo offline.

---

## 7. Requisitos Funcionais

### 7.1 Ingestão e pré-processamento

O sistema deve:

* ingerir dados em batch e streaming;
* aceitar áudio e texto;
* executar ASR quando necessário;
* suportar diarização de falantes;
* gerar segmentação por turnos;
* normalizar texto de forma configurável;
* preservar offsets, timestamps e vinculação ao áudio original.

### 7.2 Construção de contexto

O sistema deve:

* construir janelas deslizantes de contexto configuráveis;
* manter representações em múltiplos níveis: turno, janela, conversa e conversa+metadados;
* separar falas de cliente e agente;
* permitir enriquecimento com entidades e metadados operacionais.

### 7.3 Embeddings

O sistema deve:

* gerar embeddings por turno, janela e conversa;
* suportar múltiplos modelos de embedding configuráveis;
* armazenar versão de modelo e parâmetros de pooling;
* permitir comparação offline entre modelos.

### 7.4 Indexação e retrieval

O sistema deve:

* manter índice lexical BM25/similar;
* manter índice vetorial ANN;
* executar busca híbrida por combinação de scores;
* permitir reranking opcional;
* aplicar filtros por canal, fila, produto, data, região e outros metadados.

### 7.5 Classificação

O sistema deve:

* suportar classificação multi-classe e multi-label;
* operar em múltiplos níveis de granularidade;
* combinar features semânticas, lexicais, estruturais e contextuais;
* expor score, confiança, threshold e evidências.

### 7.6 Rule Engine Semântico

O sistema deve:

* oferecer DSL versionável de regras;
* compilar regras para AST;
* suportar predicados lexicais, semânticos, estruturais e contextuais;
* explicar por que uma regra acionou ou falhou;
* permitir short-circuit e otimização por custo.

### 7.7 Discovery e taxonomia

O sistema deve:

* suportar clustering offline;
* permitir geração e merge de labels com LLM assistivo offline;
* propor novos intents/tópicos para revisão humana;
* versionar taxonomias e mapear mudanças.

### 7.8 Observabilidade e governança

O sistema deve:

* registrar inferências, versão de modelos e versão de regras;
* expor métricas de qualidade, latência, throughput e drift;
* suportar auditoria e reprocessamento;
* garantir lineage do dado e reprodutibilidade.

---

## 8. Requisitos Não Funcionais

### 8.1 Escala

* Suportar **milhões de conversas por mês**.
* Suportar crescimento incremental sem redesign fundamental.
* Operar com sharding horizontal dos índices e pipelines.

### 8.2 Latência

Metas iniciais:

* inferência leve online por turno/janela: **p95 < 400 ms** sem LLM;
* busca híbrida com top-K curto: **p95 < 700 ms**;
* avaliação de regra semântica com features pré-computadas: **p95 < 150 ms**;
* pipelines offline podem operar em SLA batch.

### 8.3 Disponibilidade

* componentes online com alvo de **99.9%**;
* degradação graciosa em falha de componentes caros (por exemplo, sem reranker).

### 8.4 Segurança e privacidade

* criptografia em trânsito e repouso;
* segregação por tenant/unidade quando aplicável;
* mascaramento/redação de PII;
* trilha de auditoria;
* aderência a políticas regulatórias internas.

### 8.5 Explainability

Toda decisão operacional crítica deve produzir:

* score;
* evidência textual;
* classe/threshold;
* regra acionada, quando houver;
* versão de modelo e versão de regra.

---

## 9. Hipóteses de Produto e Princípios Técnicos

1. **Busca lexical continua sendo baseline obrigatório** em domínios estruturados.
2. **Embeddings não substituem classificadores**, apenas representam o texto.
3. **Arquitetura híbrida é superior** a abordagens puramente lexicais ou puramente semânticas na maioria dos cenários enterprise.
4. **LLMs devem ser usados prioritariamente offline** para labeling, taxonomia e casos ambíguos.
5. **Conversas exigem representação multi-nível**, não apenas embedding de transcrição completa.
6. **Regras auditáveis são necessárias** para cenários críticos.
7. **Cascata de decisão** é essencial para custo e latência em escala.

---

## 10. Arquitetura de Alto Nível

```text
Ingestão → ASR/Diarização → Segmentação de Turnos → Context Window Builder
→ Embeddings Multi-Nível → Índice Vetorial + Índice Lexical
→ Retrieval Híbrido / Reranking → Classificação Supervisionada
→ Rule Engine Semântico (AST) → Analytics / APIs / Dashboards / Feedback Loop
```

### 10.1 Camadas

**Camada de ingestão**
Recebe áudio, texto e metadados; enfileira e roteia para processamento.

**Camada de preparação**
Realiza ASR, diarização, limpeza leve, segmentação e normalização.

**Camada de contexto**
Cria turnos, janelas, agregados de conversa e features estruturadas.

**Camada semântica**
Gera embeddings e persiste representações vetoriais.

**Camada de retrieval**
Fornece BM25, ANN e busca híbrida.

**Camada de decisão**
Executa classificadores e regras semânticas.

**Camada de analytics/governança**
Expõe métricas, auditoria, dashboards e feedback para melhoria contínua.

---

## 11. Modelo de Dados Conceitual

### 11.1 Entidades principais

**Conversation**

* conversation_id
* channel
* start_time
* end_time
* customer_id (quando permitido)
* product
* queue
* region
* metadata

**Turn**

* turn_id
* conversation_id
* speaker
* start_offset
* end_offset
* raw_text
* normalized_text
* entities
* lexical_features
* semantic_features

**ContextWindow**

* window_id
* conversation_id
* turn_ids
* window_text
* role-aware views
* embedding_id

**EmbeddingRecord**

* embedding_id
* object_type (turn/window/conversation)
* model_name
* model_version
* pooling_strategy
* vector
* created_at

**Prediction**

* object_id
* task_name
* label
* score
* threshold
* evidence
* model_version

**RuleExecution**

* rule_id
* rule_version
* object_id
* matched
* score
* evidence
* execution_time_ms

---

## 12. Design do Pipeline de NLP

### 12.1 Segmentação

A unidade mínima é o **turno**, mas decisões não devem depender apenas de um turno isolado. O sistema deve produzir:

* embeddings de turno;
* embeddings de janelas deslizantes;
* embedding global da conversa;
* visões separadas por papel (cliente/agente).

### 12.2 Janelas de contexto

Parâmetros configuráveis:

* tamanho da janela em turnos;
* stride;
* alinhamento por speaker;
* peso relativo de turnos recentes;
* agregação de features estruturais no contexto.

### 12.3 Normalização

Deve ser **leve e conservadora**:

* lowercase opcional;
* normalização Unicode;
* padronização de espaços;
* remoção seletiva de ruído;
* preservação de termos críticos, números relevantes e entidades quando o caso de uso exigir.

---

## 13. Estratégia de Embeddings

### 13.1 Princípio

Não usar um único embedding para tudo.

### 13.2 Tipos de embedding por função

**Retrieval embeddings**
Voltados a recall alto em query-document matching.

**Classification embeddings**
Voltados a separabilidade de classes.

**Discovery embeddings**
Voltados a clustering, exploração e taxonomia.

**Rule-support embeddings**
Voltados a matching de protótipos e regras semânticas.

### 13.3 Recomendações iniciais

* **E5**: retrieval e matching query/passage.
* **BGE**: retrieval geral e pipelines híbridos.
* **Instructor**: cenários orientados por instrução e experimentação controlada.

### 13.4 Pooling

Comparar ao menos:

* mean pooling;
* attention/weighted pooling para janelas e conversa.

### 13.5 Estratégia operacional

* pré-computar embeddings sempre que possível;
* recalcular apenas componentes afetados por novas falas;
* versionar embeddings por modelo e pooling.

---

## 14. Busca Híbrida

### 14.1 Motivação

A busca híbrida combina o melhor de:

* **lexical retrieval** para precisão em termos obrigatórios, nomes de produto, códigos, scripts e compliance;
* **semantic retrieval** para paráfrases, intenção implícita e linguagem variável.

### 14.2 Estratégia recomendada

```text
BM25 top-N
+ ANN top-N
→ union
→ score fusion / reciprocal rank fusion
→ cross-encoder rerank opcional
→ filtros de negócio
```

### 14.3 Requisitos

* filtros estruturados antes e depois do retrieval;
* configuração por caso de uso;
* ability to degrade gracefully sem reranker;
* logging de score lexical e score semântico.

---

## 15. Classificação Supervisionada

### 15.1 Estratégia

A decisão final deve combinar múltiplas famílias de features:

* embeddings de turno, janela e conversa;
* scores BM25 contra protótipos de intent;
* distância a centroides por classe;
* entidades e padrões lexicais;
* metadados de negócio;
* flags de regras;
* sinais contextuais (repetição, posição, transição).

### 15.2 Modelos candidatos

* logistic regression para baseline;
* gradient boosting para mistura heterogênea de features;
* MLP para features densas;
* cross-encoder para reranking e casos críticos;
* arquiteturas transformer apenas quando o ganho justificar custo.

### 15.3 Granularidade

Suportar tasks em:

* turn-level;
* window-level;
* conversation-level;
* customer-history-level, quando permitido.

---

## 16. Rule Engine Semântico Baseado em AST

### 16.1 Objetivo

Garantir controle determinístico, auditabilidade e rapidez de adaptação para cenários críticos.

### 16.2 DSL

A plataforma deve expor uma DSL legível que seja compilada para AST.

Exemplo conceitual:

```text
RULE risco_cancelamento_alto
WHEN
    speaker == "customer"
    AND semantic.intent("cancelamento") > 0.82
    AND (
        lexical.contains_any(["cancelar", "encerrar", "desistir"])
        OR semantic.similarity("quero cancelar meu serviço") > 0.86
    )
    AND context.turn_window(5).count(intent="insatisfacao") >= 2
THEN
    tag("cancelamento_risco_alto")
    score(0.95)
    priority("high")
```

### 16.3 Predicados suportados

**Lexicais**

* contains
* contains_any
* contains_all
* phrase_match
* regex
* bm25_score

**Semânticos**

* intent_score
* embedding_similarity
* topic_score
* reranker_score
* nearest_prototype
* distance_to_centroid

**Estruturais**

* speaker
* turn_index
* duration
* channel
* queue
* product

**Contextuais**

* repeated_in_window
* occurs_after
* count_in_window
* transition_between_states
* first_occurrence_position

### 16.4 Requisitos do engine

* parser;
* validator de sintaxe e tipos;
* planner/optimizer;
* executor com short-circuit;
* explainability por nó da AST;
* versionamento;
* test suite por regra.

### 16.5 Integração com modelos

* regras podem usar scores de modelos como predicados;
* regras podem gerar features para classificadores;
* regras podem atuar como override em casos críticos;
* regras podem consolidar ou suprimir labels no pós-processamento.

---

## 17. Descoberta de Intents e Taxonomia

### 17.1 Objetivo

Identificar novos motivos de contato, intents emergentes e áreas de ambiguidade.

### 17.2 Abordagem

* clustering offline sobre embeddings de janelas/conversas;
* geração de labels com LLM assistivo;
* merge de labels semanticamente próximos;
* revisão humana;
* promoção controlada de novos labels para taxonomia oficial.

### 17.3 Requisitos

* suportar NMI/ARI/qualidade qualitativa;
* manter lineage entre cluster provisório e taxonomia oficial;
* permitir amostragem humana por cluster;
* registrar sugestões rejeitadas/aprovadas.

---

## 18. APIs e Interfaces

### 18.1 APIs principais

**/ingest**
Recebe lote ou evento de conversa/turno.

**/classify**
Classifica objeto textual em tasks configuradas.

**/search**
Busca híbrida com filtros estruturais.

**/rules/evaluate**
Executa regras sobre um objeto.

**/taxonomy/suggest**
Expõe sugestões offline de novos labels.

**/feedback**
Recebe correções humanas.

### 18.2 Interface interna

O sistema deve disponibilizar:

* visualização de conversa com turnos;
* scores por task;
* evidências;
* regras acionadas;
* busca híbrida;
* comparação de versões de modelo/regra;
* fila de revisão humana.

---

## 19. Métricas de Sucesso

### 19.1 Métricas de modelo

**Classificação**

* macro F1
* micro F1
* precision/recall por classe
* calibration error
* cobertura de classes raras

**Retrieval**

* Recall@K
* MRR
* nDCG
* precision@K

**Clustering/discovery**

* NMI
* ARI
* pureza qualitativa
* taxa de clusters aproveitados

**Rule engine**

* precision/recall por regra crítica
* false positive burden
* coverage
* tempo de execução

### 19.2 Métricas de produto

* redução de esforço manual de análise;
* tempo para localizar conversas relevantes;
* melhoria em retenção/compliance/QA conforme use case;
* tempo para criar/ajustar taxonomias;
* adoção por times internos.

### 19.3 Métricas operacionais

* throughput;
* latência p50/p95/p99;
* disponibilidade;
* custo por 1k conversas;
* taxa de fallback/degradação.

---

## 20. Estratégia de Avaliação

### 20.1 Baselines obrigatórios

Antes de promover qualquer arquitetura, comparar contra:

* BM25 lexical;
* classificador leve com TF-IDF;
* kNN sobre embeddings;
* modelo supervisionado com embeddings;
* híbrido lexical + embeddings.

### 20.2 Protocolos

* holdout temporal quando aplicável;
* avaliação por canal/região/fila/produto;
* análise de erro por classe rara;
* avaliação de robustez a ruído de ASR;
* comparação offline e shadow mode online.

### 20.3 Gate de promoção

Nenhum modelo ou regra nova sobe para produção sem:

* superar baseline definido;
* passar em teste de regressão;
* atender latência/custo-alvo;
* ter explicação e governança adequadas.

---

## 21. Estratégia de Rollout

### Fase 1 — Fundação

* ingestão, turn segmentation, embeddings básicos;
* índice lexical + vetorial;
* baseline de classificação;
* observabilidade mínima.

### Fase 2 — Híbrido produtivo

* score fusion;
* features mistas;
* API de busca;
* primeiras regras AST críticas.

### Fase 3 — Escala e governança

* sharding;
* versionamento robusto;
* fila de revisão humana;
* painéis operacionais.

### Fase 4 — Discovery e melhoria contínua

* clustering offline;
* labeling assistido por LLM;
* active learning;
* recalibração periódica.

---

## 22. Dependências

* ASR/Diarização
* Lakehouse / Data Warehouse
* Feature Store / Metadata Store
* Vector DB / ANN index
* Search engine lexical
* Orquestração de pipelines batch/stream
* MLOps / model registry
* Controle de acesso e mascaramento de dados

---

## 23. Riscos e Mitigações

### Risco: superdependência de embeddings genéricos

**Mitigação:** benchmark obrigatório contra lexical e híbrido.

### Risco: custo excessivo de inferência

**Mitigação:** cascata de decisão, pré-cálculo e uso seletivo de LLM.

### Risco: baixa explainability

**Mitigação:** rule engine AST, logging de evidência, modelos calibrados.

### Risco: drift de linguagem e taxonomia

**Mitigação:** discovery offline, active learning e revisão periódica.

### Risco: ruído de ASR degradando precisão

**Mitigação:** features robustas, janelas contextuais, filtros por qualidade de transcrição.

### Risco: explosão de regras difíceis de manter

**Mitigação:** versionamento, owners, testes, linting e limites de complexidade.

---

## 24. Governança

Cada task, modelo e regra deve ter:

* owner;
* descrição de objetivo;
* dataset de avaliação;
* métricas-alvo;
* política de rollout/rollback;
* versionamento;
* changelog;
* monitoramento contínuo.

---

## 25. Roadmap Inicial

### Quarter 1

* ingestão unificada;
* turn segmentation;
* índice BM25;
* embeddings de turno/janela;
* baseline classifier;
* painel de busca básica.

### Quarter 2

* hybrid retrieval;
* multi-label classifier;
* evidência e explainability;
* primeiras regras AST;
* shadow deployment.

### Quarter 3

* conversation-level aggregation;
* discovery offline;
* labeling assistido por LLM;
* feedback loop humano.

### Quarter 4

* otimização de custo/latência;
* expansão de taxonomias;
* governança avançada;
* autosserviço controlado para regras e consultas.

---

## 26. Critérios de Aceite da V1

A V1 será considerada pronta quando:

1. O pipeline processar dados reais de ao menos um canal produtivo.
2. O sistema suportar turn-level e conversation-level classification.
3. A busca híbrida estiver funcional com filtros estruturais.
4. Houver ao menos uma taxonomia operacional validada.
5. O rule engine AST suportar regras críticas com explainability.
6. Métricas e logs de inferência estiverem disponíveis.
7. Houver processo de revisão humana e feedback.

---

## 27. Decisões em Aberto

* escolha inicial do stack vetorial e lexical;
* política de retenção de dados e masking por tenant;
* thresholding por classe e por negócio;
* mecanismo de score fusion default;
* critério oficial de promoção de novos intents;
* estratégia de cross-encoder e seu orçamento de latência;
* uso de fine-tuning vs modelos frozen na fase 1.

---

## 28. Apêndice A — Princípios de Implementação

* preferir componentes modulares e substituíveis;
* separar claramente online vs offline;
* preservar reprodutibilidade de features;
* não acoplar taxonomia a código estático;
* registrar versão de tudo: dado, modelo, regra e prompt offline;
* tratar busca, classificação e regras como camadas complementares.

---

## 29. Apêndice B — Benchmarks obrigatórios da fase inicial

1. BM25 vs embeddings puros para retrieval.
2. TF-IDF + logistic regression vs embeddings + classifier.
3. kNN vs classifier supervisionado.
4. mean pooling vs attention pooling em janelas longas.
5. lexical only vs semantic only vs hybrid.
6. regras puras vs regras + semantic predicates.
7. impacto de ruído de ASR por canal.

---

## 30. Resumo Executivo

O produto proposto é uma plataforma híbrida e auditável de conversation intelligence, desenhada para maximizar qualidade semântica sem sacrificar escala, custo e governança. O sistema parte do princípio de que conversas reais exigem a combinação de retrieval lexical, embeddings semânticos, classificação supervisionada, contexto multi-turn e regras determinísticas. A entrega incremental prioriza valor operacional rápido, sem bloquear a evolução para discovery, taxonomias dinâmicas e analytics avançado.
