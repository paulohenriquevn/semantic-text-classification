# KB Complementar

# Arquitetura Escalável de Conversation Intelligence, Embeddings e Regras Semânticas

Versão: 1.1
Escopo: NLP em larga escala / Call Center / Search / Classification / Semantic Rules
Baseado em lições extraídas de:    

---

# 17. Arquitetura completa para processar milhões de conversas

## 17.1 Objetivo arquitetural

A arquitetura deve suportar simultaneamente:

* ingestão massiva
* processamento incremental
* classificação multiestágio
* busca semântica e lexical
* regras auditáveis
* baixa latência operacional
* reprocessamento histórico
* governança e explainability

Para cenários de call center, o pipeline precisa lidar com:

* áudio
* ASR/transcrição
* turnos de fala
* contexto multi-turn
* intents
* motivos de contato
* compliance
* sentimentos/sinais de risco
* busca e analytics sobre o corpus histórico

---

## 17.2 Princípio central

A principal lição dos estudos é que **não existe um único mecanismo ótimo para tudo**.

Os documentos sugerem a seguinte divisão funcional:

* **lexical retrieval** para alta velocidade e forte baseline em dados estruturados 
* **embeddings** para generalização semântica
* **classificadores supervisionados** para decisão robusta 
* **LLMs** para rotulação, descoberta de categorias e taxonomia, preferencialmente offline 
* **attention/weighted aggregation** para representar janelas longas e conversas multi-turn 
* **rule engine auditável** para política, compliance e lógica determinística

---

## 17.3 Arquitetura de referência

```text
[1] Ingestão
    └── áudio / texto / metadados / CRM / eventos

[2] Pré-processamento
    └── ASR
    └── diarização
    └── segmentação de turnos
    └── limpeza leve
    └── normalização

[3] Construção de contexto
    └── janela por turno
    └── janela deslizante
    └── agregados por conversa
    └── features estruturadas

[4] Camada semântica
    └── embeddings por turno
    └── embeddings por janela
    └── embedding global da conversa
    └── indexação vetorial

[5] Camada lexical
    └── BM25 / sparse index
    └── filtros estruturados
    └── dicionários / aliases / termos críticos

[6] Classificação
    └── intent classifier
    └── topic classifier
    └── risk/compliance classifier
    └── multi-label classifier

[7] Rule Engine Semântico
    └── AST parser
    └── predicates
    └── operadores lógicos
    └── scoring / threshold / precedence

[8] Busca e Analytics
    └── hybrid retrieval
    └── dashboards
    └── auditoria
    └── explainability

[9] Feedback loop
    └── hard negatives
    └── active learning
    └── revisão humana
    └── recalibração
```

---

## 17.4 Separação entre pipeline online e offline

Essa separação é essencial.

### Pipeline online

Responsável por baixa latência.

Inclui:

* classificação operacional
* busca em tempo real
* regras imediatas
* alertas
* priorização de atendimento

### Pipeline offline

Responsável por qualidade e evolução do sistema.

Inclui:

* relabeling
* clustering exploratório
* descoberta de intents
* retraining
* avaliação
* recalibração de thresholds
* enriquecimento com LLM

A formulação de clustering como classificação com LLM é muito útil aqui: o LLM pode gerar e consolidar labels offline, e depois esses labels alimentam classificadores mais baratos em produção .

---

## 17.5 Unidades de processamento recomendadas

A conversa deve ser materializada em múltiplos níveis:

### Nível 1 — Turno

Exemplo:

* agente: “em que posso ajudar?”
* cliente: “quero cancelar meu plano”

Uso:

* intent local
* compliance local
* busca granular

### Nível 2 — Janela de contexto

Exemplo: 3 a 8 turnos adjacentes

Uso:

* desambiguação
* contexto de objeção
* mudança de intenção
* sinais semânticos distribuídos

### Nível 3 — Conversa completa

Uso:

* classificação final
* motivo de contato dominante
* satisfação
* resolução
* roteamento analítico

### Nível 4 — Conversa + metadados

Exemplo:

* canal
* fila
* produto
* duração
* cliente
* tipo de contrato
* status CRM

Uso:

* classificação híbrida semântico-estrutural
* regras com contexto de negócio

---

## 17.6 Padrão recomendado de armazenamento

### Camada quente

Para busca e inferência de baixa latência:

* índice vetorial ANN
* índice lexical BM25
* cache de embeddings
* store de features

### Camada morna

Para consultas analíticas frequentes:

* warehouse
* parquet/lakehouse
* tabelas agregadas por conversa

### Camada fria

Para reprocessamento completo:

* áudio bruto
* transcrição integral
* versão de modelos
* snapshots de regras
* rótulos humanos

---

## 17.7 Estratégia de escala

Para milhões de conversas, o erro mais comum é tentar processar tudo com o mesmo nível de custo semântico.

A estratégia correta é usar **cascata de decisão**.

### Estágio 1 — filtros baratos

* idioma
* canal
* fila
* expressão lexical crítica
* regras simples
* classificador leve

### Estágio 2 — recuperação híbrida

* BM25 top-K
* ANN top-K
* merge/rerank

### Estágio 3 — classificação mais cara

* cross-encoder
* classificador multi-label
* regra semântica contextual

### Estágio 4 — revisão excepcional

* LLM apenas para casos ambíguos, novos ou críticos

Esse desenho segue diretamente a lição de que embeddings off-the-shelf nem sempre vencem lexical baselines, especialmente em conteúdo estruturado .

---

## 17.8 Arquitetura de classificação recomendada

Em produção, evitar depender apenas de nearest neighbor. O repositório de referência deixa claro que embedding é representação; quem decide é o classificador .

Arquitetura recomendada:

```text
Embedding features
+ lexical features
+ structured features
+ conversation features
→ classifier
```

### Features recomendadas

* embedding do turno atual
* embedding da janela anterior
* embedding da conversa
* score BM25 contra intents protótipo
* distância ao centroide por classe
* presença de entidades
* flags de regras
* metadados de negócio

### Modelos candidatos

* logistic regression para baseline forte
* XGBoost / LightGBM para mistura de features heterogêneas
* MLP para embeddings + features densas
* transformer/cross-encoder quando precisão justificar custo

---

## 17.9 Retrieval para analytics e search

A arquitetura de busca deve ser híbrida.

### Componente lexical

Usar para:

* termos obrigatórios
* nomes de produto
* códigos
* scripts
* palavras de compliance
* frases com pouca ambiguidade

### Componente semântico

Usar para:

* paráfrases
* intenções implícitas
* reformulações
* linguagem natural variada

### Merge strategy

Estratégias comuns:

* weighted sum de scores
* reciprocal rank fusion
* cross-encoder rerank

### Recomendação prática

Para call center, quase sempre o melhor ponto de partida é:

```text
BM25 top-100
+
vector top-100
+
union
+
cross-encoder top-50
+
business filters
```

---

## 17.10 Latência e throughput

### Objetivo

Minimizar custo por conversa sem perder recall.

### Regras gerais

* embeddings devem ser pré-computados sempre que possível
* indexação deve ser incremental
* janelas de contexto devem ser montadas uma vez e reutilizadas
* inferência pesada deve ser reservada aos casos ambíguos
* classificadores simples devem absorver o tráfego comum

### Gargalos típicos

* re-embedding frequente da conversa inteira
* uso excessivo de LLM online
* cross-encoder em top-K muito grande
* ausência de cache por turno/janela
* regras sem short-circuit
* busca vetorial sem filtros estruturais

---

# 18. Design de embeddings recomendado

## 18.1 Princípio de escolha

O paper de comparação lexical vs semântico mostra que **não se deve presumir superioridade automática de embeddings semânticos** .

A escolha do embedding depende de:

* tipo de texto
* tamanho médio do texto
* variabilidade linguística
* domínio
* tarefa
* orçamento de latência
* estratégia de indexação

---

## 18.2 Regra prática

### E5

Mais indicado para:

* retrieval
* busca semântica
* matching query-document
* tarefas em que a formulação query/passages é importante

### BGE

Mais indicado para:

* retrieval geral
* reranking combinado com ecossistema BGE
* pipelines que exigem bom equilíbrio entre qualidade e custo

### Instructor

Mais indicado para:

* tarefas em que instrução explícita melhora a representação
* cenários multi-tarefa
* uso experimental/controlado em que a task prompt é parte do design

---

## 18.3 Padrão por função, não por modelo único

Não usar o mesmo embedding para todo o pipeline.

### Embedding de retrieval

Objetivo: recall alto para busca
Modelos candidatos: E5 / BGE

### Embedding de classificação

Objetivo: separabilidade entre classes
Pode ser:

* o mesmo embedding de retrieval, se funcionar bem
* um encoder ajustado para classificação
* features combinadas com lexical/estrutural

### Embedding de clustering/discovery

Objetivo: agrupar e explorar taxonomias
Pode ser:

* BGE / E5
* embeddings especializados por domínio
* representação enriquecida por contexto

### Embedding de regras semânticas

Objetivo: matching entre regra e evidência textual
Preferir:

* embeddings estáveis
* baixa latência
* alta consistência intra-domínio

---

## 18.4 Estratégia de avaliação de embeddings

Avaliar embeddings em quatro tarefas separadas:

### A. Retrieval

Métricas:

* Recall@K
* MRR
* nDCG
* latency

### B. Classification

Métricas:

* macro F1
* micro F1
* recall por classe
* calibration error

### C. Clustering/discovery

Métricas:

* NMI
* ARI
* purity
* utilidade qualitativa dos clusters

O paper sobre clustering como classificação usa ACC, NMI e ARI como métricas principais .

### D. Robustez operacional

* estabilidade entre versões
* memória
* throughput
* sensibilidade a ASR noise
* variação por canal/região

---

## 18.5 Como desenhar embeddings para conversa

O erro comum é gerar apenas um embedding da transcrição completa.

A recomendação é trabalhar com múltiplas visões:

### Embedding do turno

Captura intenção local.

### Embedding da janela

Captura dependência contextual.

### Embedding global da conversa

Captura objetivo dominante e desfecho.

### Embedding por papel

Separar:

* cliente
* agente

Isso melhora:

* detecção de intenção real
* distinção entre script do agente e demanda do cliente
* compliance

---

## 18.6 Pooling recomendado

O estudo sobre classificação com mecanismos de atenção reforça que agregação simples pode desperdiçar sinal importante; attention-based enhancement e pooling ponderado tendem a produzir representações mais discriminativas .

### Poolings candidatos

#### Mean pooling

Vantagens:

* simples
* estável
* barato

Desvantagem:

* dilui tokens críticos

#### Max pooling

Vantagens:

* destaca sinais fortes

Desvantagem:

* pode amplificar ruído

#### Attention pooling

Vantagens:

* pondera tokens e trechos mais relevantes
* melhor para conversas longas
* melhor para intents contextuais

Desvantagem:

* mais custo
* requer treinamento ou design adicional

### Recomendação

* começar com mean pooling como baseline
* comparar com attention pooling nas classes críticas
* usar attention pooling principalmente em janela/conversa, não necessariamente em todos os turnos

---

## 18.7 Recomendações concretas por cenário

### Cenário 1 — Busca semântica em grande volume

* embedding: E5 ou BGE
* index: HNSW
* retrieval: top-K vetorial + BM25
* rerank: cross-encoder apenas no shortlist

### Cenário 2 — Intent classification em call center

* embedding de turno + embedding de janela
* features lexicais e estruturais adicionais
* classificador supervisionado
* regras AST para intents críticas

### Cenário 3 — Discovery de novos motivos de contato

* embeddings de janela/conversa
* clustering exploratório
* LLM para gerar e consolidar labels offline, como sugerido pelo paper de clustering como classificação 

### Cenário 4 — Compliance e risco

* lexical gates primeiro
* embeddings para contexto e reformulações
* regras AST com evidência textual e thresholds

---

# 19. Pipeline de regras semânticas baseado em AST

## 19.1 Objetivo

O papel da rule engine semântica é complementar classificadores e retrieval com:

* precisão determinística
* auditabilidade
* explainability
* controle de política
* resposta imediata a requisitos de negócio

Ela é crítica quando a organização precisa responder:

* por que a conversa foi classificada assim?
* qual evidência textual acionou a regra?
* qual threshold foi aplicado?
* qual versão da política estava vigente?

---

## 19.2 Princípio central

A regra não deve ser texto livre executado diretamente.

Ela deve ser compilada em **AST (Abstract Syntax Tree)**.

Isso traz:

* validação sintática
* validação semântica
* otimização
* versionamento
* execução segura
* explainability por nó

---

## 19.3 Estrutura lógica da regra

Uma regra semântica robusta deve combinar quatro famílias de sinal:

### 1. Sinal lexical

Exemplo:

* contém “cancelar”
* contém “ouvidoria”
* regex de protocolo

### 2. Sinal semântico

Exemplo:

* similaridade com protótipo “ameaça de cancelamento”
* score de intenção “fraude”
* score de compliance

### 3. Sinal estrutural

Exemplo:

* falante = cliente
* turno ocorreu antes da retenção
* duração > 60s
* canal = voz

### 4. Sinal contextual

Exemplo:

* evento ocorreu após negativa do agente
* mesma intenção repetida em múltiplos turnos
* coocorrência em janela de 5 turnos

---

## 19.4 DSL recomendada

Exemplo de DSL legível para negócios:

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

Essa DSL deve ser convertida em AST.

---

## 19.5 AST de referência

Exemplo simplificado:

```text
RuleNode
 ├── AndNode
 │    ├── EqNode(speaker, "customer")
 │    ├── GtNode(semantic.intent("cancelamento"), 0.82)
 │    ├── OrNode
 │    │    ├── ContainsAnyNode(["cancelar","encerrar","desistir"])
 │    │    └── GtNode(semantic.similarity(prototype), 0.86)
 │    └── GteNode(context.turn_window(5).count(intent="insatisfacao"), 2)
 └── ActionNode(tag="cancelamento_risco_alto", score=0.95, priority="high")
```

---

## 19.6 Componentes do engine

### Parser

Transforma DSL em AST.

### Validator

Verifica:

* nomes de campos
* tipos
* operadores permitidos
* thresholds válidos
* referências a modelos existentes

### Planner/Otimizer

Reordena predicados para custo mínimo.

Exemplo:
executar primeiro:

* speaker == customer
* canal == voz

e deixar por último:

* semantic.similarity()

### Executor

Avalia a AST sobre a conversa/turno/janela.

### Explainer

Produz:

* nós avaliados
* valores intermediários
* evidências
* motivo do match/fail

### Versionador

Mantém:

* versão da regra
* data de ativação
* autor
* motivo da mudança
* compatibilidade com versão do modelo

---

## 19.7 Ordem de execução recomendada

Para custo eficiente, usar short-circuit.

### Etapa 1 — Predicados baratos

* igualdade
* metadata
* speaker
* canal
* produto

### Etapa 2 — Lexical

* contains
* phrase match
* regex
* BM25 against prototypes

### Etapa 3 — Semânticos baratos

* score já pré-computado
* classe prevista
* distância a centroide

### Etapa 4 — Semânticos caros

* similaridade on-demand
* cross-encoder
* inferência adicional

---

## 19.8 Tipos de predicado recomendados

### Lexical

* contains
* contains_any
* contains_all
* regex
* phrase_match
* bm25_score

### Semântico

* embedding_similarity
* intent_score
* topic_score
* reranker_score
* nearest_prototype
* distance_to_centroid

### Estrutural

* speaker
* turn_index
* duration
* silence_ratio
* overlap
* interruption

### Contextual

* in_previous_turn
* repeated_in_window
* occurs_after
* count_in_window
* transition_between_states

### Agregados de conversa

* dominant_intent
* max_risk_score
* count_matches
* proportion_of_negative_sentiment
* first_occurrence_position

---

## 19.9 Granularidade de execução

A mesma regra pode ser avaliada em diferentes scopes:

### Turn-level

Para eventos imediatos.

### Window-level

Para dependências locais.

### Conversation-level

Para decisão final agregada.

### Customer-history-level

Quando permitido por política, para recorrência histórica.

---

## 19.10 Estratégia de evidência

Toda regra semântica deve devolver evidências.

Formato sugerido:

```json
{
  "rule_id": "risco_cancelamento_alto",
  "matched": true,
  "score": 0.95,
  "evidence": [
    {
      "type": "lexical",
      "text": "quero cancelar meu plano",
      "turn_id": 18
    },
    {
      "type": "semantic",
      "signal": "intent(cancelamento)",
      "value": 0.91,
      "threshold": 0.82
    },
    {
      "type": "contextual",
      "signal": "count(intent=insatisfacao, window=5)",
      "value": 3,
      "threshold": 2
    }
  ],
  "rule_version": "12"
}
```

Isso é essencial para auditoria.

---

## 19.11 Integração entre rule engine e modelos

A rule engine não substitui os modelos. Ela orquestra e governa.

### Padrões úteis

#### Modelo como feature da regra

Exemplo:

* `intent_score("cancelamento") > 0.8`

#### Regra como feature do classificador

Exemplo:

* `flag_ouvidoria = 1`
* `flag_risco_fraude = 1`

#### Regra como override

Usar só em cenários críticos:

* compliance
* fraude
* emergência
* termos legalmente mandatórios

#### Regra como pós-processamento

Exemplo:

* reduzir falso positivo
* consolidar classes
* suprimir conflito entre labels

---

## 19.12 Governança de regras

Cada regra precisa de:

* owner
* objetivo de negócio
* descrição
* escopo
* data de vigência
* métricas-alvo
* conjunto de testes
* exemplos positivos e negativos
* dependências de modelo
* política de rollback

### Testes obrigatórios

* unit test por predicado
* test set dourado
* regressão entre versões
* teste de latência
* teste de explainability

---

# 20. Padrão recomendado final

A arquitetura recomendada para o projeto é:

```text
1. Transcrição e diarização
2. Segmentação de turnos
3. Construção de janelas contextuais
4. Embeddings multi-nível (turno, janela, conversa)
5. Indexação híbrida (BM25 + vetorial)
6. Classificação supervisionada com features mistas
7. Rule engine semântico baseado em AST
8. Reranking e pós-processamento
9. Analytics + feedback loop offline
10. LLM apenas para discovery, labeling e curadoria
```

Essa formulação é a mais coerente com os estudos analisados:

* o paper médico reforça a importância de lexical baselines e do híbrido 
* o paper de clustering mostra o valor de usar LLMs para gerar e consolidar labels, não necessariamente para inferência massiva online 
* o paper de attention reforça que agregação inteligente melhora representação e classificação, especialmente em textos longos/contextuais 
* o repositório demonstra a separação correta entre embedding e classificador 

---

# 21. Diretrizes de decisão rápida para a equipe

## Quando usar BM25 primeiro

* domínio estruturado
* termos específicos
* necessidade de explainability
* baixa latência
* base com muito vocabulário repetido

## Quando usar embeddings fortemente

* muita paráfrase
* linguagem aberta
* intenção implícita
* discovery
* busca semântica transversal

## Quando usar LLM

* gerar taxonomia
* consolidar labels
* rotular dataset
* análise exploratória de casos ambíguos

## Quando usar regras AST

* compliance
* fraude
* negócio crítico
* override controlado
* necessidade de auditoria

---

