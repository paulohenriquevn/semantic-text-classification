# Capitulo 4 — Arquitetura Proposta: TalkEx

Este capitulo descreve a arquitetura do TalkEx, o artefato tecnico central desta dissertacao. Apresentamos o pipeline completo de processamento de conversas, o modelo de dados conversacional, os mecanismos de normalizacao de texto, as representacoes multi-nivel, o sistema de recuperacao hibrida, a classificacao supervisionada, o motor de regras semanticas e a estrategia de inferencia em cascata. O nivel de detalhe tecnico visa permitir a reproducao integral do sistema.

A descricao segue o fluxo natural dos dados: desde a ingestao de uma conversa bruta ate a producao de insights classificados e auditaveis. Ao longo do capitulo, referenciamos decisoes arquiteturais formalizadas em Architecture Decision Records (ADRs), garantindo rastreabilidade entre principios de design e implementacao concreta.

---

## 4.1 Visao Geral do Pipeline

### 4.1.1 Diagrama Arquitetural

O TalkEx implementa um pipeline de NLP conversacional multi-estagio, projetado para transformar conversas brutas em insights classificados e auditaveis:

```
 +--------------+     +------------------+     +--------------------+
 |  Ingestao    |---->|  Segmentacao     |---->|  Normalizacao &    |
 |              |     |  de Turnos       |     |  Pre-processamento |
 +--------------+     +------------------+     +--------------------+
                                                        |
                                                        v
                                               +--------------------+
                                               |  Construtor de     |
                                               |  Janela de Contexto|
                                               +--------------------+
                                                        |
                                                        v
                                               +--------------------+
                                               |  Geracao de        |
                                               |  Embeddings        |
                                               |  (multi-nivel)     |
                                               +--------------------+
                                                        |
                                            +-----------+-----------+
                                            |                       |
                                            v                       v
                                   +----------------+     +------------------+
                                   |  Indice Lexico |     |  Indice Vetorial |
                                   |  (BM25)        |     |  (ANN/FAISS)     |
                                   +----------------+     +------------------+
                                            |                       |
                                            +----------+------------+
                                                       |
                                                       v
                                              +-------------------+
                                              | Recuperacao       |
                                              | Hibrida           |
                                              | (fusao + rerank)  |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Classificacao    |
                                              |  Supervisionada   |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Motor de Regras  |
                                              |  Semanticas       |
                                              |  (DSL → AST)     |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Analytics /      |
                                              |  APIs / Feedback  |
                                              +-------------------+
```

### 4.1.2 Principios de Design

A arquitetura se sustenta em tres principios interdependentes:

**Modularidade.** Cada estagio do pipeline e um modulo independente com interfaces bem definidas. Os modulos se comunicam exclusivamente por meio de tipos de dados compartilhados (modelos Pydantic imutaveis), sem acoplamento a implementacoes concretas. Isso permite substituir qualquer componente — por exemplo, trocar o indice vetorial de FAISS para Qdrant — sem modificar modulos a montante ou a jusante. A estrutura de pacotes reflete essa modularidade:

```
src/talkex/
  __init__.py           # Raiz do pacote, exporta __version__
  exceptions.py         # Hierarquia de excecoes de dominio (base EngineError)
  text_normalization.py # Utilitarios compartilhados de normalizacao de texto
  models/               # Tipos de dados Pydantic (frozen, strict)
  ingestion/            # Ingestao de dados de multiplas fontes
  segmentation/         # Segmentacao de turnos e normalizacao
  context/              # Construtor de janela de contexto deslizante
  embeddings/           # Geracao de embeddings multi-nivel
  retrieval/            # Busca hibrida: BM25 + ANN + fusao de scores
  classification/       # Classificacao supervisionada multi-classe
  rules/                # Motor de regras: DSL → AST → executor com evidencia
  analytics/            # APIs e endpoints analiticos
```

Essa organizacao segue a convencao de layout `src/` (ADR-001), onde todas as importacoes utilizam o namespace `talkex`:

```python
from talkex.models import Conversation, Turn, ContextWindow
from talkex.retrieval import InMemoryBM25Index
from talkex.rules import SimpleRuleEvaluator
```

**Inferencia em cascata.** O pipeline aplica processamento progressivamente mais custoso. Filtros lexicos baratos (custo O(1)) precedem a recuperacao hibrida (O(log n) para ANN), que precede a classificacao supervisionada (O(d) para dimensao de features d), que precede regras semanticas com predicados de embeddings. A Secao 4.8 detalha a logica de decisao entre estagios.

**Separacao online/offline.** O TalkEx distingue dois modos operacionais. O pipeline online prioriza baixa latencia: classificacao, busca e regras imediatas. O pipeline offline prioriza qualidade e cobertura: re-rotulagem, clusterizacao, descoberta de intencoes com LLMs, re-treinamento de modelos e recalibracao de limiares. LLMs sao utilizados exclusivamente no pipeline offline, nunca para inferencia online, garantindo custo previsivel e latencia controlada em producao.

### 4.1.3 Fluxo de Dados

A transformacao de uma conversa bruta se processa em sete estagios:

1. **Ingestao**: a conversa e recebida de uma fonte (API, arquivo, fila de mensagens) e validada contra o schema `Conversation`.
2. **Segmentacao**: o texto bruto e segmentado em turnos (`Turn`), cada um atribuido a um falante (cliente, agente, sistema).
3. **Normalizacao**: cada turno recebe normalizacao de texto (lowercasing, remocao de diacriticos, pontuacao) para consumo por componentes lexicos.
4. **Janelas de contexto**: turnos adjacentes sao agrupados em janelas deslizantes (`ContextWindow`) de tamanho N com passo S.
5. **Embeddings**: vetores densos sao gerados em multiplos niveis de granularidade (turno, janela, conversa, por papel do falante).
6. **Indexacao e recuperacao**: os textos sao simultaneamente indexados no indice lexico (BM25) e no indice vetorial (ANN). Consultas hibridas combinam ambos via fusao de scores.
7. **Classificacao e regras**: features heterogeneas alimentam classificadores supervisionados. O motor de regras semanticas aplica regras deterministicas compiladas em ASTs, produzindo decisoes com evidencia rastreavel.

Ao final do pipeline, cada conversa esta associada a predicoes de classificacao (`Prediction`) e resultados de avaliacao de regras (`RuleExecution`), ambos portando metadados de evidencia, versao do modelo e tempo de execucao.

---

## 4.2 Modelo de Dados Conversacional

O modelo de dados do TalkEx define seis entidades de dominio que fluem por todo o pipeline. Todas utilizam Pydantic v2 com `ConfigDict(frozen=True, strict=True)` (ADR-002), garantindo imutabilidade e tipagem estrita.

### 4.2.1 Entidades e Relacoes

A Tabela 4.1 apresenta as seis entidades centrais e seus papeis no pipeline.

| Entidade         | Descricao                                                  | Granularidade       |
|------------------|------------------------------------------------------------|---------------------|
| `Conversation`   | Interacao completa entre cliente e agente                   | Conversa inteira    |
| `Turn`           | Enunciado individual atribuido a um falante                 | Turno               |
| `ContextWindow`  | Janela deslizante de N turnos adjacentes                    | Janela de contexto  |
| `EmbeddingRecord`| Representacao vetorial versionada de um objeto textual      | Multi-nivel         |
| `Prediction`     | Resultado de classificacao com score, confianca e limiar    | Multi-nivel         |
| `RuleExecution`  | Resultado de avaliacao de regra com evidencia rastreavel     | Multi-nivel         |

As relacoes seguem uma hierarquia de composicao:

```
Conversation (1)
  |
  +--< Turn (N)                  # Uma conversa contem N turnos
  |
  +--< ContextWindow (M)         # Uma conversa gera M janelas
         |
         +--< turn_ids[]         # Cada janela referencia turnos
         |
         +--< EmbeddingRecord    # Cada janela pode ter embeddings
         |
         +--< Prediction         # Cada janela pode ter predicoes
         |
         +--< RuleExecution      # Cada janela pode ter avaliacoes de regras
```

`EmbeddingRecord`, `Prediction` e `RuleExecution` sao polimorficos em granularidade: cada um porta um campo `source_type` indicando se o objeto de origem e um turno, janela de contexto ou conversa. Isso permite que embeddings, classificacoes e regras operem em qualquer nivel de granularidade sem multiplicar tipos de dados.

### 4.2.2 Decisoes de Design

**Por que modelos imutaveis (ADR-002).** Modelos imutaveis previnem mutacao acidental de dados compartilhados conforme fluem pelos estagios do pipeline. Quando um estagio precisa enriquecer um objeto (por exemplo, adicionar texto normalizado a um turno), ele cria um novo objeto em vez de mutar o original. Isso garante que estagios a montante sempre vejam dados consistentes e simplifica a depuracao em pipelines concorrentes.

**Por que `list[float]` para vetores (ADR-003).** Vetores de embeddings sao armazenados como `list[float]` nos modelos Pydantic para compatibilidade de serializacao, e convertidos para `numpy.ndarray` nas fronteiras de computacao (calculo de similaridade, indexacao FAISS). Isso evita problemas de serializacao do Pydantic com tipos numpy, preservando eficiencia computacional onde ela importa.

**Por que modo estrito com ampliacao sem perda.** O modo estrito do Pydantic v2 rejeita coercao de tipos nas fronteiras (API, desserializacao de arquivos), mas permite ampliacao sem perda (int → float) em operacoes em memoria. Isso captura problemas de qualidade de dados na ingestao sem ser excessivamente restritivo durante a computacao.

---

## 4.3 Normalizacao de Texto

A normalizacao de texto e uma etapa critica de pre-processamento para o dominio conversacional em PT-BR. A funcao `normalize_for_matching()` fornece normalizacao com consciencia de acentos, essencial para componentes lexicos:

```python
def normalize_for_matching(text: str) -> str:
    """Lowercase + Unicode NFD diacritics removal for PT-BR matching."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text
```

**Por que isso importa para PT-BR.** O portugues brasileiro apresenta extensa variacao diacritica: "nao"/"nao", "numero"/"numero", "cancelamento"/"cancelamento". Em transcricoes de atendimento ao cliente, diacriticos sao aplicados de forma inconsistente — particularmente em saidas de ASR e chat informal. Sem normalizacao, BM25 e predicados de regras lexicas tratariam "cancelamento" e "cancelamento" (com acento agudo) como termos diferentes, reduzindo o recall.

**Decisao de design: normalizar para correspondencia, preservar para exibicao.** O texto original (`raw_text`) e sempre preservado para auditabilidade. A normalizacao produz um campo paralelo (`normalized_text`) utilizado exclusivamente por componentes lexicos (indexacao BM25, predicados de regras, extracao de features). Componentes semanticos (geracao de embeddings) operam sobre o texto original, pois modelos transformer lidam com diacriticos internamente.

---

## 4.4 Segmentacao de Turnos

O modulo `TurnSegmenter` transforma texto bruto de conversa em uma sequencia de objetos `Turn` atribuidos. A estrategia de segmentacao depende da fonte de dados:

- **Fontes estruturadas** (plataformas de chat, sistemas de tickets): turnos sao pre-segmentados pelo sistema de origem, requerendo apenas validacao e atribuicao de falante.
- **Fontes nao estruturadas** (transcricoes de voz): o segmentador aplica uma heuristica de alternancia de falantes, dividindo o texto nas fronteiras de mudanca de falante identificadas por marcadores de diarizacao do ASR.

Cada turno recebe um `turn_id`, papel de `speaker` (cliente, agente, sistema, desconhecido), offsets posicionais e texto normalizado opcional.

---

## 4.5 Construtor de Janela de Contexto

### 4.5.1 Motivacao

Turnos individuais capturam intencao local ("Quero cancelar"), mas perdem dependencias multi-turno. Uma reclamacao que escala ao longo de turnos — comecando como uma pergunta, progredindo pela insatisfacao e culminando em um pedido de cancelamento — so pode ser detectada examinando turnos adjacentes conjuntamente. O `SlidingWindowBuilder` constroi janelas sobrepostas que capturam essas dependencias.

### 4.5.2 Configuracao

O construtor de janelas e parametrizado por:

| Parametro | Padrao | Descricao |
|-----------|--------|-----------|
| `window_size` | 5 | Numero de turnos por janela |
| `stride` | 2 | Tamanho do passo entre janelas consecutivas |
| `min_turns` | 3 | Minimo de turnos necessarios para formar uma janela |

**Por que janelas de 5 turnos com passo 2.** Cinco turnos tipicamente abrangem 2-3 trocas conversacionais (pares cliente-agente), capturando contexto suficiente para desambiguacao de intencao sem diluir o sinal. Um passo de 2 fornece sobreposicao entre janelas consecutivas, garantindo que nenhuma transicao conversacional caia entre janelas. Esses parametros foram selecionados com base na analise da distribuicao de turnos do dataset (mediana de 8 turnos por conversa) e validados empiricamente durante o desenvolvimento — uma janela de 5 turnos gera 2-3 janelas por conversa, fornecendo multiplas oportunidades de classificacao por interacao.

### 4.5.3 Features Estruturais

Cada janela de contexto porta features estruturais extraidas durante a construcao:

- **Contagem de turnos** e **distribuicao de falantes** (% turnos do cliente, % turnos do agente)
- **Posicao da janela** (inicio, meio, fim da conversa)
- **Estatisticas textuais** (contagem total de palavras, media de palavras por turno)
- **Transicoes de falante** (numero de mudancas de falante dentro da janela)

Essas features estruturais servem como sinais adicionais de classificacao (testadas no estudo de ablacao, Secao 6.6) e como predicados para o motor de regras.

### 4.5.4 Supervisao Fraca

Janelas de contexto herdam o rotulo de intencao de sua conversa pai. Essa suposicao de supervisao fraca introduz ruido: janelas intermediarias (por exemplo, uma janela de saudacao em uma conversa de cancelamento) podem carecer de sinais explicitos da intencao no nivel da conversa. Reconhecemos isso como uma limitacao (Secao 7.3) e observamos que isso afeta tanto o treinamento quanto a avaliacao de forma consistente, enviesando o F1 por classe para baixo em classes com padroes de inicio gradual (por exemplo, "compra", "saudacao").

---

## 4.6 Geracao de Embeddings Multi-Nivel

### 4.6.1 Arquitetura

O TalkEx gera representacoes vetoriais densas em quatro niveis de granularidade:

| Nivel | Texto de Origem | Proposito |
|-------|----------------|-----------|
| **Turno** | Enunciado individual | Intencao local, deteccao de palavras-chave |
| **Janela** | Turnos concatenados na janela de contexto | Dependencias multi-turno, padroes de escalacao |
| **Conversa** | Todos os turnos concatenados | Resolucao global, tom dominante |
| **Por papel** | Turnos somente do cliente ou somente do agente | Analise de comportamento especifico por falante |

O modelo de embeddings primario e `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensoes), um Sentence Transformer treinado para similaridade semantica multilingual. Este modelo foi selecionado por tres razoes:

1. **Suporte multilingual**: capacidade nativa para PT-BR sem fine-tuning especifico por idioma.
2. **Eficiencia**: 384 dimensoes (vs. 768 para BERT-base ou 1024 para E5-large) reduz custo de armazenamento e computacao em 2-4x.
3. **Implantacao com encoder congelado**: o encoder e utilizado tal como esta, sem fine-tuning — uma decisao arquitetural deliberada que troca potencial acuracia especifica do dominio por acessibilidade e reprodutibilidade.

**Por que um encoder congelado.** Fine-tuning requer infraestrutura substancial de GPU, dados de treinamento especificos do dominio e gestao continua de modelos. Ao utilizar um encoder congelado pre-treinado, o TalkEx demonstra que classificacao competitiva e alcancavel sem infraestrutura dedicada de treinamento de ML — todos os experimentos foram executados na GPU Tesla T4 do tier gratuito do Google Colab, que acelera a inferencia de embeddings mas nao requer orcamento de fine-tuning. Os resultados empiricos (Capitulo 6) mostram que embeddings congelados combinados com LightGBM alcancam Macro-F1 = 0,722, validando essa escolha pragmatica. Uma comparacao com fine-tuning permanece como trabalho futuro (Secao 7.4).

### 4.6.2 Versionamento de Embeddings

Cada registro de embedding porta metadados de versionamento:

```python
@dataclass(frozen=True)
class EmbeddingRecord:
    source_id: str          # ID of the source object (turn, window, conversation)
    source_type: str        # "turn" | "window" | "conversation"
    model_name: str         # e.g., "paraphrase-multilingual-MiniLM-L12-v2"
    model_version: str      # Semantic version of the model
    pooling_strategy: str   # "mean" | "max" | "cls"
    vector: list[float]     # Dense vector (384 dimensions)
    generated_at: datetime  # Timestamp for provenance
```

Esse versionamento garante reprodutibilidade: se o modelo de embeddings mudar, consumidores a jusante (classificadores, indices) podem detectar a incompatibilidade em vez de degradar silenciosamente.

---

## 4.7 Recuperacao Hibrida

### 4.7.1 Indexacao Dual

O TalkEx mantem dois indices paralelos sobre o mesmo corpus:

**Indice lexico (BM25).** Um indice BM25 em memoria construido com a biblioteca `rank-bm25`, operando sobre texto normalizado. Scores BM25 sao computados com parametros padrao (k₁ = 1,5, b = 0,75). O indice suporta correspondencia exata de termos, codigos de produtos e palavras-chave de conformidade — dominios onde precisao lexica e essencial.

**Indice vetorial (ANN).** Um indice FAISS sobre vetores de embeddings, utilizando IVF (Inverted File Index) com quantizacao de produto para escalabilidade. O indice suporta busca aproximada de vizinhos mais proximos com trade-offs configuraveis entre recall e velocidade.

### 4.7.2 Fusao de Scores

Dada uma consulta q, o pipeline de recuperacao hibrida:

1. Recupera os top-K₁ resultados do BM25 (lexico)
2. Recupera os top-K₂ resultados do ANN (semantico)
3. Computa a uniao dos conjuntos de resultados
4. Aplica fusao de scores para produzir um ranking unificado

Duas estrategias de fusao sao suportadas:

**Fusao linear ponderada:**

```
Score(d) = α · sim_semantic(q, d) + (1 - α) · score_BM25(q, d)
```

onde α ∈ [0, 1] controla o peso semantico. Ambos os scores sao normalizados por min-max para [0, 1] antes da fusao. O α otimo e selecionado no conjunto de validacao.

**Reciprocal Rank Fusion (RRF):**

```
RRF_score(d) = Σᵢ 1 / (k + rankᵢ(d))
```

onde k = 60 (constante padrao) e a soma e sobre todos os sistemas de ranking que recuperaram o documento d. RRF e baseado em ranking em vez de score, tornando-o robusto a diferencas de escala de score entre BM25 e similaridade cosseno.

### 4.7.3 Re-ranking Opcional com Cross-Encoder

Para consultas de alto risco, os top-N resultados da fusao de scores podem ser re-ranqueados por um modelo cross-encoder que codifica conjuntamente o par consulta-documento. Isso melhora a precisao ao custo de O(N) passadas diretas por um modelo transformer. Na avaliacao experimental (Capitulo 6), o re-ranking nao foi aplicado, pois o foco estava na eficacia da fusao de scores.

---

## 4.8 Classificacao Supervisionada

### 4.8.1 Construcao de Features

O pipeline de classificacao constroi vetores de features heterogeneos combinando quatro familias de sinais:

| Familia | Features | Dimensao | Origem |
|---------|----------|----------|--------|
| **Embedding** | Vetor denso do embedding no nivel de janela | 384 | Sentence Transformer |
| **Lexico** | Scores TF-IDF, contagens de termos, padroes de n-gramas | ~11 | Analise textual |
| **Estrutural** | Contagem de turnos, proporcao de falantes, contagem de palavras, posicao | ~4 | Metadados da janela de contexto |
| **Regras** | Flags binarias da avaliacao do motor de regras | ~2 | Motor de regras |

O vetor de features total tem aproximadamente 397 dimensoes (384 embedding + 11 lexico + 4 estrutural + 2 regras). Features sao construidas como dicionarios Python (`list[dict[str, float]]`), convertidos para arrays numpy na fronteira do classificador.

**Principio de design: embeddings representam, classificadores decidem.** Seguindo o principio da AnthusAI, vetores de embeddings nunca sao utilizados diretamente para classificacao via similaridade cosseno. Em vez disso, servem como features de entrada para classificadores supervisionados que aprendem fronteiras de decisao a partir de dados rotulados. Essa separacao garante que o classificador possa aprender fronteiras de decisao nao lineares e incorporar sinais heterogeneos que a similaridade bruta nao consegue capturar.

### 4.8.2 Modelos de Classificador

Tres arquiteturas de classificador sao avaliadas:

**Regressao Logistica (baseline).** Um modelo linear simples que serve como baseline obrigatorio. Sua fronteira de decisao linear revela quanto da classificacao pode ser alcancado com combinacoes simples de features.

**LightGBM (primario).** Um framework de gradient boosting (100 estimadores, 31 folhas) que lida com features heterogeneas nativamente — uma vantagem critica ao combinar embeddings densos com indicadores lexicos esparsos. LightGBM e deterministico com divisoes fixas e estado aleatorio, permitindo experimentos reprodutiveis.

**MLP (Multi-Layer Perceptron).** Uma rede neural de duas camadas (tamanhos ocultos: 256, 128) que captura interacoes nao lineares de features. MLP exibe sensibilidade a semente devido a inicializacao estocastica de pesos, sendo o unico classificador que apresenta desvio padrao nao nulo entre sementes.

### 4.8.3 Saida de Predicao

Cada classificacao produz um objeto `Prediction` contendo:

- **label**: classe de intencao predita
- **score**: probabilidade para a classe predita
- **confidence**: confianca calibrada (quando disponivel)
- **threshold**: limiar de decisao aplicado
- **model_version**: versao do classificador para proveniencia
- **evidence**: importancia de features ou principais features contribuintes

Esse formato de saida garante que toda predicao seja rastreavel e auditavel, atendendo aos requisitos de explicabilidade de dominios regulados.

---

## 4.9 Motor de Regras Semanticas

### 4.9.1 Motivacao

Classificadores supervisionados aprendem padroes a partir de dados, mas nao fornecem garantias formais e oferecem rastreabilidade limitada de evidencias. Em dominios regulados — telecomunicacoes (Anatel), servicos financeiros (Bacen), saude suplementar (ANS) — decisoes devem ser auditaveis, contestaveis e explicaveis. O motor de regras aborda essa lacuna fornecendo logica de decisao deterministica e rastreavel que complementa a classificacao estatistica.

### 4.9.2 Linguagem de Dominio Especifico (DSL)

Regras sao expressas em uma DSL legivel por humanos que e compilada para Abstract Syntax Trees (ASTs) para avaliacao eficiente:

```
RULE detect_cancellation
  WHEN speaker == "customer"
   AND contains_any(["cancelar", "encerrar", "desistir"])
  THEN tag("cancelamento", confidence=1.0)
```

A DSL suporta quatro familias de predicados:

| Familia | Predicados | Custo | Exemplos |
|---------|-----------|-------|---------|
| **Lexico** | `contains`, `contains_any`, `regex_match`, `bm25_score` | O(n) | Deteccao exata de palavras-chave |
| **Semantico** | `intent_score`, `embedding_similarity` | O(d) | Correspondencia semantica suave |
| **Estrutural** | `speaker`, `turn_index`, `channel`, `duration` | O(1) | Condicoes sobre metadados |
| **Contextual** | `repeated_in_window`, `occurs_after`, `count_in_window` | O(w) | Padroes multi-turno |

### 4.9.3 Compilacao e Execucao de AST

O parser da DSL compila regras em nos de AST. O executor percorre a AST com duas otimizacoes:

1. **Avaliacao em curto-circuito**: se o primeiro predicado em uma conjuncao AND falha, os predicados restantes nao sao avaliados.
2. **Execucao ordenada por custo**: predicados dentro de uma conjuncao sao ordenados por custo (estrutural O(1) → lexico O(n) → semantico O(d)), garantindo que filtros baratos eliminem candidatos antes da execucao de operacoes custosas.

### 4.9.4 Trilha de Evidencia

Cada no de AST produz evidencia rastreavel ao ser avaliado:

```python
@dataclass(frozen=True)
class PredicateResult:
    predicate_type: str     # "lexical" | "semantic" | "structural" | "contextual"
    matched: bool           # Whether the predicate was satisfied
    score: float            # Numeric score (0.0-1.0)
    evidence: dict          # {"matched_words": [...], "positions": [...], ...}
    execution_time_ms: float
```

A avaliacao completa de uma regra produz um objeto `RuleExecution` com a cadeia de evidencias completa: quais predicados dispararam, que texto correspondeu, quais scores foram computados e quais limiares foram aplicados. Essa trilha de evidencia e o principal diferencial em relacao a predicoes de ML do tipo caixa-preta.

### 4.9.5 Integracao com Classificacao

Regras podem ser integradas com classificacao ML em tres modos:

1. **Regras como sobrescrita**: decisoes de regras sobrescrevem predicoes do classificador (pos-processamento).
2. **Regras como feature**: flags de correspondencia de regras sao adicionadas como features binarias a entrada do classificador.
3. **Regras como pos-processamento**: regras ajustam a confianca do classificador ou adicionam rotulos secundarios.

A avaliacao experimental (Capitulo 6) compara esses modos de integracao, constatando que regras como feature e o unico modo que evita degradacao catastrofica.

---

## 4.10 Inferencia em Cascata

### 4.10.1 Design

O pipeline de inferencia em cascata aplica processamento progressivamente mais custoso, permitindo resolucao antecipada quando a confianca e suficiente:

| Estagio | Processamento | Custo Esperado | Criterio de Resolucao |
|---------|--------------|----------------|----------------------|
| 1 | Regras lexicas baratas + filtros de metadados | ~1 ms | Confianca >= limiar |
| 2 | Recuperacao hibrida + classificador leve (LogReg) | ~10-50 ms | Confianca >= limiar |
| 3 | Classificacao completa (LightGBM) + regras semanticas | ~50-200 ms | Sempre resolve |
| 4 | Revisao excepcional (LLM, somente offline) | ~500 ms-2s | Revisao manual |

O limiar de confianca em cada estagio determina o trade-off custo-qualidade. Limiares mais baixos resolvem mais conversas em estagios mais baratos, mas arriscam classificacao incorreta; limiares mais altos encaminham mais conversas para estagios custosos, mas preservam a qualidade.

### 4.10.2 Restricao de Implementacao

Na configuracao experimental atual, tanto o classificador leve (Estagio 2) quanto o classificador completo (Estagio 3) operam sobre embeddings e janelas de contexto pre-computados. Isso limita o diferencial de custo entre estagios ao tempo de inferencia do classificador isoladamente (LogReg ~20ms vs LightGBM ~30ms, uma razao de ~1,1x). Conforme os resultados experimentais demonstram (Secao 6.5), esse diferencial de custo insuficiente impede a cascata de alcancar reducao significativa de custo. Uma cascata em nivel de producao exigiria um Estagio 1 genuinamente barato (por exemplo, features puramente lexicas sem computacao de embeddings) para concretizar os beneficios teoricos.

---

## 4.11 Engenharia de Software

### 4.11.1 Escala e Qualidade

O TalkEx compreende aproximadamente 170 arquivos-fonte e 15.773 linhas de codigo de producao, organizados em 11 modulos de pipeline. A suite de testes inclui 100 arquivos de teste com 1.883 testes cobrindo validacao unitaria, de integracao e de pipeline. Todo o codigo passa por portas de qualidade continuas: `ruff format`, `ruff check`, verificacao de tipos com `mypy` e execucao de `pytest`.

### 4.11.2 Architecture Decision Records

Quatro ADRs documentam decisoes arquiteturais irreversiveis:

| ADR | Decisao | Justificativa |
|-----|---------|---------------|
| ADR-001 | Layout `src/` com API publica via re-exportacoes em `__init__.py` | Empacotamento Python padrao, fronteira publica/privada clara |
| ADR-002 | Modelos Pydantic frozen + strict; desserializacao em fronteira com `strict=False` | Garantia de imutabilidade + serializacao pratica |
| ADR-003 | Vetores como `list[float]` nos modelos, `ndarray` na computacao | Compatibilidade de serializacao + eficiencia computacional |
| ADR-004 | Design de campos estruturais da janela de contexto | Extracao explicita de features no momento de construcao da janela |

---

## 4.12 Resumo

A arquitetura do TalkEx implementa os tres paradigmas investigados nesta dissertacao — recuperacao lexica, recuperacao semantica e regras deterministicas — dentro de um unico pipeline modular. O design prioriza reprodutibilidade (encoder congelado, modelos versionados, classificadores deterministicos), auditabilidade (trilhas de evidencia de regras, proveniencia de predicoes) e acessibilidade (classificadores treinaveis em CPU, inferencia de embeddings acelerada por GPU via tier gratuito do Google Colab). O Capitulo 5 descreve como essa arquitetura e avaliada experimentalmente, e o Capitulo 6 apresenta os resultados.
