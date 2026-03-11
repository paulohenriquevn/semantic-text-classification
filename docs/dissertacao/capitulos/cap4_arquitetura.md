# Capítulo 4 — Arquitetura Proposta: TalkEx

Este capítulo descreve a arquitetura do TalkEx, o artefato técnico central desta dissertação. Apresentamos o pipeline completo de processamento de conversas, o modelo de dados conversacional, os mecanismos de normalização e pré-processamento, as representações multi-nível, o sistema de retrieval híbrido, a classificação supervisionada, o motor de regras semânticas e a estratégia de inferência em cascata. O nível de detalhe técnico visa permitir a reprodução integral do sistema.

A descrição segue a ordem natural do fluxo de dados: da ingestão de uma conversa bruta até a produção de insights classificados e auditáveis. Ao longo do capítulo, referenciamos as decisões arquiteturais formalizadas em ADRs (Architecture Decision Records), garantindo rastreabilidade entre princípios de design e implementação concreta.

---

## 4.1 Visão Geral do Pipeline

### 4.1.1 Diagrama Arquitetural

O TalkEx implementa um pipeline multi-estágio de NLP conversacional, projetado para transformar conversas brutas em insights classificados e auditáveis. O diagrama a seguir apresenta o fluxo completo:

```
 +--------------+     +------------------+     +--------------------+
 |  Ingestao    |---->|  Segmentacao     |---->|  Normalizacao e    |
 |  (ingestion) |     |  em Turnos       |     |  Pre-processamento |
 +--------------+     +------------------+     +--------------------+
                                                        |
                                                        v
                                               +--------------------+
                                               |  Construtor de     |
                                               |  Janelas de        |
                                               |  Contexto          |
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
                                   |  Indice        |     |  Indice          |
                                   |  Lexico (BM25) |     |  Vetorial (ANN)  |
                                   +----------------+     +------------------+
                                            |                       |
                                            +----------+------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Retrieval        |
                                              |  Hibrido          |
                                              |  (fusao + rerank) |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Classificacao    |
                                              |  Supervisionada   |
                                              |  (multi-label)    |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Motor de Regras  |
                                              |  Semanticas       |
                                              |  (DSL -> AST)     |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              |  Analytics /      |
                                              |  APIs / Feedback  |
                                              +-------------------+
```

### 4.1.2 Princípios de Design

A arquitetura do TalkEx fundamenta-se em três princípios interdependentes:

**Modularidade.** Cada estágio do pipeline é um módulo independente com interfaces bem definidas. Os módulos comunicam-se exclusivamente por meio de tipos de dados compartilhados (modelos Pydantic imutáveis), sem acoplamento a implementações concretas. Isso permite substituir qualquer componente -- por exemplo, trocar o índice vetorial de FAISS para Qdrant -- sem alterar os módulos a montante ou a jusante. A estrutura de pacotes reflete essa modularidade:

```
src/talkex/
  __init__.py           # Raiz do pacote, exporta __version__
  exceptions.py         # Hierarquia de exceções de domínio
  text_normalization.py # Normalização textual compartilhada
  models/               # Tipos de dados Pydantic (frozen, strict)
  ingestion/            # Ingestão de múltiplas fontes
  segmentation/         # Segmentação em turnos e normalização
  context/              # Construtor de janelas de contexto
  embeddings/           # Geração de embeddings multi-nível
  retrieval/            # Busca híbrida: BM25 + ANN + fusão
  classification/       # Classificação multi-label supervisionada
  rules/                # Motor de regras: DSL -> AST -> executor
  analytics/            # APIs e endpoints analíticos
```

Esta organização segue a convenção `src/` layout (ADR-001), onde o diretório `src/` é invisível para consumidores e todas as importações utilizam o namespace `talkex`:

```python
from talkex.models import Conversation, Turn, ContextWindow
from talkex.retrieval import InMemoryBM25Index
from talkex.rules import SimpleRuleEvaluator
```

**Inferência em cascata.** O pipeline aplica processamento progressivamente mais caro. Filtros lexicais simples (custo O(1)) precedem retrieval híbrido (custo O(log n) para ANN), que precede classificação supervisionada (custo O(d) para dimensão d do vetor de features), que precede regras semânticas com predicados de embedding. A Seção 4.8 detalha a lógica de decisão entre estágios.

**Separação online/offline.** O TalkEx distingue dois modos operacionais. O pipeline online prioriza baixa latência: classificação, busca e regras imediatas. O pipeline offline prioriza qualidade e cobertura: relabeling, clustering, intent discovery com LLMs, retreinamento de modelos e recalibração de thresholds. LLMs são utilizados exclusivamente no pipeline offline, nunca para inferência online, garantindo custo previsível e latência controlada em produção.

### 4.1.3 Fluxo de Dados

O fluxo de transformação de uma conversa bruta pode ser descrito em sete etapas:

1. **Ingestão**: a conversa é recebida de uma fonte (API, arquivo, fila de mensagens) e validada contra o esquema `Conversation`.
2. **Segmentação**: o texto bruto é segmentado em turnos (`Turn`), cada um atribuído a um falante (customer, agent, system).
3. **Normalização**: cada turno recebe normalização textual (lowercase, remoção de diacríticos, pontuação) para consumo por componentes lexicais.
4. **Janelas de contexto**: turnos adjacentes são agrupados em janelas deslizantes (`ContextWindow`) de tamanho N com stride S.
5. **Embeddings**: vetores densos são gerados em múltiplos níveis de granularidade (turno, janela, conversa, por papel).
6. **Indexação e retrieval**: textos são indexados simultaneamente no índice léxico (BM25) e no índice vetorial (ANN). Consultas híbridas combinam ambos via fusão de scores.
7. **Classificação e regras**: features heterogêneas alimentam classificadores supervisionados. O motor de regras semânticas aplica regras determinísticas compiladas para AST, produzindo decisões com evidência rastreável.

Ao final do pipeline, cada conversa está associada a predições de classificação (`Prediction`) e resultados de avaliação de regras (`RuleExecution`), ambos com metadados de evidência, versão de modelo e tempo de execução.

---

## 4.2 Modelo de Dados Conversacional

O modelo de dados do TalkEx define seis entidades de domínio que fluem pelo pipeline inteiro. Todas utilizam Pydantic v2 com `ConfigDict(frozen=True, strict=True)` (ADR-002), garantindo imutabilidade e tipagem estrita.

### 4.2.1 Entidades e Relações

A Tabela 4.1 apresenta as seis entidades centrais e seus papéis no pipeline.

| Entidade         | Descrição                                                    | Granularidade      |
|------------------|--------------------------------------------------------------|--------------------|
| `Conversation`   | Interação completa entre cliente e agente                    | Conversa inteira   |
| `Turn`           | Enunciado individual atribuído a um falante                  | Turno              |
| `ContextWindow`  | Janela deslizante de N turnos adjacentes                     | Janela de contexto |
| `EmbeddingRecord`| Representação vetorial versionada de um objeto de texto      | Multi-nível        |
| `Prediction`     | Resultado de classificação com score, confiança e threshold  | Multi-nível        |
| `RuleExecution`  | Resultado de avaliação de regra com evidência rastreável      | Multi-nível        |

As relações entre entidades seguem uma hierarquia de composição:

```
Conversation (1)
  |
  +--< Turn (N)                  # Uma conversa contém N turnos
  |
  +--< ContextWindow (M)         # Uma conversa gera M janelas
         |
         +--< turn_ids[]         # Cada janela referencia turnos
         |
         +--< EmbeddingRecord    # Cada janela pode ter embeddings
         |
         +--< Prediction         # Cada janela pode ter predições
         |
         +--< RuleExecution      # Cada janela pode ter regras avaliadas
```

`EmbeddingRecord`, `Prediction` e `RuleExecution` são polimórficos em granularidade: cada um carrega um `source_type` que indica se o objeto fonte é um turno, janela de contexto ou conversa. Isso permite que embeddings, classificações e regras operem em qualquer nível de granularidade sem multiplicar os tipos de dados.

### 4.2.2 Conversation

A entidade `Conversation` representa uma interação completa. Carrega metadados operacionais (canal, fila, produto, região) utilizados como filtros de negócio no retrieval e como features estruturais na classificação.

```python
class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    conversation_id: ConversationId    # Formato: conv_<uuid4>
    channel: Channel                   # voice | chat | email | ticket
    start_time: datetime
    end_time: datetime | None = None
    customer_id: str | None = None
    product: str | None = None
    queue: str | None = None
    region: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Validadores de domínio garantem invariantes em tempo de construção: `conversation_id` não pode ser vazio e `end_time` não pode preceder `start_time`. A validação ocorre na fronteira do sistema (momento da ingestão); após construção, o objeto é confiável e imutável.

### 4.2.3 Turn

O `Turn` é a unidade mais fina de processamento. Cada turno é atribuído a um falante (`SpeakerRole`) e carrega tanto o texto bruto quanto uma versão normalizada opcional:

```python
class Turn(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    turn_id: TurnId                    # Formato: turn_<uuid4>
    conversation_id: ConversationId    # Referência à conversa pai
    speaker: SpeakerRole               # customer | agent | system | unknown
    raw_text: str                      # Texto original (ASR ou digitado)
    start_offset: int                  # Posição inicial (caracteres ou ms)
    end_offset: int                    # Posição final
    normalized_text: str | None = None # Texto normalizado (opcional)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

O campo `raw_text` preserva o texto original sem alteração -- essencial para auditoria e para reproduzir o input exato visto pelo pipeline. O campo `normalized_text` é preenchido pelo estágio de normalização (Seção 4.3) e utilizado por componentes lexicais.

### 4.2.4 ContextWindow

A `ContextWindow` captura dependências multi-turno agrupando turnos adjacentes. É a unidade primária para classificação contextual e avaliação de regras semânticas:

```python
class ContextWindow(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    window_id: WindowId
    conversation_id: ConversationId
    turn_ids: list[TurnId]        # IDs dos turnos na janela (ordenados)
    window_text: str              # Texto concatenado dos turnos
    start_index: int              # Índice do primeiro turno (0-based)
    end_index: int                # Índice do último turno (0-based)
    window_size: int              # Número de turnos (auditável)
    stride: int                   # Stride usado na geração (reprodutível)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

A decisão de incluir `start_index`, `end_index`, `window_size` e `stride` como campos explícitos (ADR-004) garante que cada janela é um registro auto-descritivo. Um validador cruzado impõe `window_size == len(turn_ids)`, detectando inconsistências do construtor em tempo de construção (fail-fast). A redundância é intencional: `window_size` é um campo de auditoria, não um derivado computado dinamicamente.

### 4.2.5 EmbeddingRecord

O `EmbeddingRecord` armazena uma representação vetorial densa versionada:

```python
class EmbeddingRecord(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    embedding_id: EmbeddingId
    source_id: str                  # ID do objeto representado
    source_type: ObjectType         # turn | context_window | conversation
    model_name: str                 # Ex: 'e5-large-v2', 'bge-base-en'
    model_version: str
    pooling_strategy: PoolingStrategy  # mean | max | attention
    dimensions: int
    vector: list[float]             # Vetor denso (ver ADR-003)
```

A decisão de armazenar vetores como `list[float]` em vez de `numpy.ndarray` (ADR-003) prioriza serialização limpa e compatibilidade com Pydantic strict mode. A conversão para `ndarray` ocorre nas fronteiras dos módulos de computação:

```python
import numpy as np
array = np.array(record.vector, dtype=np.float32)
```

O versionamento por `model_name`, `model_version` e `pooling_strategy` garante que vetores produzidos por modelos diferentes nunca sejam inadvertidamente comparados. Um validador cruzado impõe `dimensions == len(vector)`.

### 4.2.6 Prediction

A entidade `Prediction` encapsula o resultado de um classificador com toda a informação necessária para auditoria:

```python
class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    prediction_id: PredictionId
    source_id: str
    source_type: ObjectType
    label: str                    # Label predito
    score: float                  # Output do classificador [0.0, 1.0]
    confidence: float             # Confiança calibrada [0.0, 1.0]
    threshold: float              # Limiar de decisão [0.0, 1.0]
    model_name: str
    model_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)
```

A propriedade derivada `is_above_threshold` indica se a predição atinge o limiar de decisão. Nenhuma predição existe sem `model_name` e `model_version` -- rastreabilidade não é opcional.

### 4.2.7 RuleExecution

O `RuleExecution` registra a avaliação de uma regra semântica com evidência estruturada:

```python
class RuleExecution(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    rule_id: RuleId
    rule_name: str
    source_id: str
    source_type: ObjectType
    matched: bool                 # Regra ativou?
    score: float                  # Score agregado [0.0, 1.0]
    execution_time_ms: float      # Latência da avaliação
    evidence: list[EvidenceItem]  # Evidência por predicado
    metadata: dict[str, Any] = Field(default_factory=dict)
```

O `EvidenceItem` é um `TypedDict` com campos opcionais que variam por tipo de predicado:

```python
class EvidenceItem(TypedDict, total=False):
    predicate_type: str      # 'lexical', 'semantic', etc.
    matched_text: str        # Fragmento que ativou o match
    score: float             # Score numérico
    threshold: float         # Threshold utilizado
    model_name: str          # Modelo (predicados semânticos)
    model_version: str
    metadata: dict[str, Any]
```

Um validador cruzado impõe que `matched=True` requer ao menos um `EvidenceItem` -- uma asserção de match sem evidência é um bug no executor, não um estado válido. Esta invariante é central para a auditabilidade do sistema.

### 4.2.8 Tipos Auxiliares e Enumerações

O modelo de dados utiliza `NewType` para criar tipos nominais que previnem confusão entre identificadores:

```python
ConversationId = NewType("ConversationId", str)
TurnId = NewType("TurnId", str)
WindowId = NewType("WindowId", str)
EmbeddingId = NewType("EmbeddingId", str)
PredictionId = NewType("PredictionId", str)
RuleId = NewType("RuleId", str)
```

Quatro enumerações (`StrEnum`) definem conjuntos finitos de valores válidos:

| Enum              | Valores                                  | Uso                                   |
|-------------------|------------------------------------------|---------------------------------------|
| `SpeakerRole`     | customer, agent, system, unknown         | Atribuição de turnos                  |
| `Channel`         | voice, chat, email, ticket               | Canal da conversa                     |
| `ObjectType`      | turn, context_window, conversation       | Granularidade de embeddings/predições |
| `PoolingStrategy`  | mean, max, attention                     | Estratégia de pooling de embeddings   |

---

## 4.3 Normalização e Pré-processamento

### 4.3.1 Motivação

O domínio de conversas de atendimento em português brasileiro apresenta desafios lexicais específicos. Textos provenientes de transcrição ASR frequentemente perdem diacríticos, coloquialismos introduzem variações ortográficas, e a digitação em chat omite acentos por conveniência. Considere as seguintes variantes que devem ser tratadas como equivalentes:

- "nao" e "não" (com acento)
- "cancelamento" e "cancelamento" (com acento)
- "QUERO CANCELAR" e "quero cancelar"
- "fatura," e "fatura" (com pontuação)

Sem normalização, o índice BM25 trataria "nao" (sem acento) e "não" (com acento cedilha/til) como termos distintos, prejudicando recall. Predicados lexicais do motor de regras apresentariam o mesmo problema: uma regra buscando "cancelar" não ativaria para "cancélar" (com acento indevido de ASR).

### 4.3.2 Módulo text_normalization

O TalkEx implementa normalização textual em um módulo compartilhado (`talkex.text_normalization`) utilizado tanto pelo índice BM25 quanto pelo avaliador de regras lexicais. O módulo depende exclusivamente da biblioteca padrão Python (`unicodedata`), sem dependências externas:

```python
import unicodedata

def strip_accents(text: str) -> str:
    """Remove marcas diacríticas (acentos) do texto.

    Utiliza decomposição NFD para separar caracteres base de marcas
    combinantes, e então remove todos os caracteres da categoria
    Unicode 'Mn' (Nonspacing Mark).
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

def normalize_for_matching(text: str) -> str:
    """Normaliza texto para matching léxico: lowercase + strip accents."""
    return strip_accents(text.lower())
```

A função `strip_accents` aplica decomposição NFD (Normal Form Decomposition), que separa caracteres compostos em caracteres base seguidos de marcas combinantes. Por exemplo, "ã" (U+00E3) decompõe-se em "a" (U+0061) + "~" (U+0303, combining tilde). A filtragem subsequente remove todos os caracteres da categoria Unicode `Mn` (Nonspacing Mark), que inclui todas as marcas diacríticas combinantes.

A função `normalize_for_matching` compõe lowercase e strip accents em uma única operação. Ambos os lados de qualquer comparação lexical -- tanto o texto de busca quanto o texto alvo -- passam por esta função antes da comparação, garantindo simetria.

### 4.3.3 Integração no Pipeline

A normalização é aplicada em quatro pontos do pipeline:

1. **Tokenização BM25**: o tokenizador do índice léxico aplica `normalize_for_matching` antes de dividir em tokens por whitespace e remover pontuação:

```python
def _tokenize(text: str) -> list[str]:
    normalized = normalize_for_matching(text)
    cleaned = _PUNCTUATION_RE.sub("", normalized)
    return [tok for tok in cleaned.split() if tok]
```

2. **Predicados lexicais do motor de regras**: o avaliador aplica `normalize_for_matching` tanto ao texto do input quanto ao valor do predicado antes de executar a operação (contains, word, stem, etc.).

3. **Predicados contextuais**: operadores como `repeated_in_window` e `occurs_after` normalizam o texto da janela antes de buscar ocorrências.

4. **Predicados regex**: a expressão regular é aplicada sobre texto normalizado, permitindo que padrões como `cancel|terminate` capturem variantes com acentos.

Esta centralização evita duplicação de lógica de normalização (DRY) e garante consistência entre todos os componentes lexicais do sistema.

---

## 4.4 Representações Multi-Nível

Um dos axiomas de design do TalkEx é que diferentes níveis de granularidade capturam diferentes aspectos semânticos de uma conversa. A intenção imediata de um cliente pode ser capturada em um único turno, mas o padrão de insatisfação crescente só emerge quando observamos uma janela de turnos consecutivos, e o desfecho global da interação requer a visão da conversa inteira.

### 4.4.1 Embedding de Turno

O embedding de turno captura a intenção local de um enunciado individual. É gerado diretamente a partir do `raw_text` (ou `normalized_text`) de um `Turn`, utilizando um modelo de sentence embedding (ex: E5, BGE, Instructor).

**Aplicações**: retrieval por similaridade com queries curtas (ex: "quero cancelar meu plano"), classificação de intenção pontual, predicados semânticos do motor de regras.

**Limitação**: captura apenas a intenção local. Um turno isolado como "sim, pode ser" não carrega semântica suficiente sem o contexto precedente.

### 4.4.2 Embedding de Janela de Contexto

O embedding de janela é gerado a partir do `window_text` de um `ContextWindow` -- texto concatenado de N turnos adjacentes. Parâmetros configuráveis controlam a construção:

- **window_size**: número de turnos por janela (valores típicos: 3, 5, 7, 10)
- **stride**: passo entre janelas consecutivas (1 para máximo overlap, window_size para zero overlap)
- **speaker alignment**: opção de filtrar turnos por papel (apenas cliente, apenas agente)
- **recency weighting**: peso maior para turnos mais recentes dentro da janela

**Aplicações**: classificação contextual (intenção que emerge de múltiplos turnos), detecção de padrões multi-turno (escalamento, insatisfação progressiva), retrieval que captura contexto completo de uma interação.

**Justificativa**: a hipótese H2 desta dissertação postula que representações multi-nível (incluindo janelas de contexto) superam representações de nível único para classificação conversacional.

### 4.4.3 Embedding de Conversa

O embedding de conversa é gerado a partir da concatenação de todos os turnos de uma `Conversation`, capturando o objetivo dominante e o desfecho global da interação.

**Aplicações**: clustering de conversas por similaridade temática, classificação macro (retenção vs cancelamento vs informação), analytics de tendências.

### 4.4.4 Embeddings por Papel (Role-Aware)

Visões separadas para cliente e agente permitem análise diferenciada:

- **Embedding do cliente**: concatenação apenas dos turnos com `speaker == "customer"`. Captura a intenção real do cliente, sem diluição pelas respostas do agente.
- **Embedding do agente**: concatenação apenas dos turnos com `speaker == "agent"`. Captura o comportamento do agente, útil para avaliação de qualidade e compliance.

**Aplicações**: detecção de intenção real do cliente (separando ruído do agente), avaliação de qualidade do atendimento, detecção de scripts de compliance no discurso do agente.

### 4.4.5 Estratégia de Pooling

Todos os embeddings são gerados por modelos de sentence embedding que produzem representações de comprimento fixo a partir de sequências de tokens de comprimento variável. A estratégia de pooling determina como os embeddings de tokens individuais são agregados:

- **Mean pooling** (baseline): média aritmética dos embeddings de tokens. Simples, robusto e eficaz na maioria dos cenários.
- **Max pooling**: máximo elemento-a-elemento. Captura as features mais salientes, mas perde informação distribucional.
- **Attention pooling** (experimental): ponderação aprendida sobre tokens. Recomendado para janelas longas onde nem todos os tokens contribuem igualmente para a semântica global.

O `PoolingStrategy` é registrado no `EmbeddingRecord`, garantindo que embeddings produzidos com diferentes estratégias nunca sejam inadvertidamente comparados.

### 4.4.6 Versionamento de Embeddings

Cada `EmbeddingRecord` é completamente versionado por `(model_name, model_version, pooling_strategy)`. Esta tripla identifica univocamente o espaço vetorial em que o embedding reside. Operações de similaridade só são válidas entre embeddings do mesmo espaço vetorial -- embeddings produzidos por modelos diferentes não são comparáveis.

O versionamento é essencial para:
- Retrocompatibilidade com vetores armazenados
- Reprodução de experimentos com configurações anteriores
- Migrações graduais entre modelos de embedding

---

## 4.5 Retrieval Híbrido

O módulo de retrieval do TalkEx implementa busca híbrida combinando retrieval léxico (BM25) e retrieval semântico (ANN sobre embeddings densos). A fusão dos resultados explora a complementaridade entre os dois paradigmas: BM25 captura correspondências lexicais exatas (nomes de produtos, códigos, termos técnicos), enquanto embeddings capturam paráfrase e intenção implícita.

### 4.5.1 Componente Léxico: BM25

A implementação do BM25 segue a fórmula Okapi (Robertson et al., 1996):

```
score(q, d) = SUM_t IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))
```

onde:
- `tf(t,d)` = frequência do termo t no documento d
- `IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)`
- `N` = total de documentos indexados
- `n(t)` = documentos contendo o termo t
- `|d|` = comprimento do documento em tokens
- `avgdl` = comprimento médio dos documentos

Os hiperparâmetros default são `k1 = 1.5` (saturação de frequência de termos) e `b = 0.75` (normalização por comprimento do documento), valores estabelecidos pela literatura como robustos para a maioria dos domínios.

A tokenização aplica o pipeline de normalização descrito na Seção 4.3: `normalize_for_matching` (lowercase + accent stripping) seguido de remoção de pontuação e split por whitespace. Essa normalização accent-aware é particularmente relevante para PT-BR:

```python
def _tokenize(text: str) -> list[str]:
    normalized = normalize_for_matching(text)
    cleaned = _PUNCTUATION_RE.sub("", normalized)
    return [tok for tok in cleaned.split() if tok]
```

A implementação atual é em Python puro com numpy para eficiência computacional, adequada para benchmarking e pipelines batch. Para produção em escala (milhões de documentos), o protocolo `LexicalIndex` permite substituição por uma implementação baseada em Elasticsearch ou Tantivy sem alterar o restante do pipeline.

### 4.5.2 Componente Semântico: ANN

O componente semântico utiliza busca por vizinhos mais próximos aproximados (Approximate Nearest Neighbors) sobre embeddings densos. A configuração suporta três tipos de índice:

| Tipo        | Descrição                                        | Uso                              |
|-------------|--------------------------------------------------|----------------------------------|
| `FLAT`      | Busca exata (força bruta)                        | Benchmarking, datasets pequenos  |
| `HNSW`      | Hierarchical Navigable Small World (aproximado)  | Produção                         |
| `IVF_FLAT`  | Índice particionado com busca exata por partição | Datasets médios                  |

A métrica de distância default é similaridade cosseno, alinhada com os modelos de sentence embedding utilizados (E5, BGE, Instructor), que são otimizados para cosseno.

O `VectorIndexConfig` encapsula a configuração:

```python
class VectorIndexConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.FLAT
    dimensions: int
    train_required: bool = False
    top_k_default: int = 25
```

### 4.5.3 Fusão de Scores

O `SimpleHybridRetriever` orquestra a execução de ambos os componentes e funde os resultados. O roteamento segue o tipo de query:

```
QueryType.LEXICAL   -> BM25 apenas
QueryType.SEMANTIC  -> ANN apenas
QueryType.HYBRID    -> BM25 + ANN -> fusão
```

Quando ambos os componentes retornam resultados, duas estratégias de fusão são suportadas:

**Reciprocal Rank Fusion (RRF)**. Combina rankings sem depender das escalas dos scores:

```
score_rrf(d) = SUM_i 1 / (k + rank_i(d))
```

onde `k` é uma constante de suavização (default 60) e `rank_i(d)` é o ranking do documento d na lista i. RRF é robusto quando as distribuições de scores léxico e semântico são incompatíveis (o que é típico: scores BM25 são não-normalizados, scores de similaridade cosseno estão em [-1, 1]).

**Combinação Linear Ponderada**. Combina scores normalizados:

```
score_linear(d) = alpha * sem_norm(d) + (1 - alpha) * lex_norm(d)
```

onde `alpha` é o peso do componente semântico e `sem_norm`/`lex_norm` são scores min-max normalizados para [0, 1]. O parâmetro `alpha` (denominado `fusion_weight` na configuração) controla o balanço entre os paradigmas. Valores próximos de 1.0 privilegiam semântica; próximos de 0.0, privilegiam léxico.

A deduplicação por `object_id` ocorre antes da fusão, e os scores componentes (lexical_score, semantic_score) são preservados no resultado final para observabilidade.

### 4.5.4 Reranking Opcional

O pipeline de retrieval reserva um slot para reranking por cross-encoder após a fusão. Cross-encoders processam o par (query, documento) conjuntamente, produzindo scores de relevância mais precisos que bi-encoders, porém a um custo computacional significativamente maior. O reranking é aplicado sobre a shortlist pós-fusão (tipicamente 10-25 documentos), não sobre o corpus inteiro.

Na implementação atual, o slot de reranking existe como protocolo (`_Reranker`) mas não é invocado, seguindo o princípio YAGNI -- será ativado quando experimentos justificarem o custo adicional.

### 4.5.5 Pipeline Completo de Retrieval

O fluxo completo de uma query híbrida:

```
Query (texto + tipo)
    |
    +---> BM25: tokenize -> score -> top-K lexicais
    |
    +---> ANN: embed query -> busca vetorial -> top-K semânticos
    |
    +---> União dos candidatos (dedup por object_id)
    |
    +---> Fusão (RRF ou LINEAR)
    |
    +---> [Reranking opcional]
    |
    +---> Top-K final
    |
    v
RetrievalResult(hits, total_candidates, mode, stats)
```

O `RetrievalResult` carrega metadados operacionais (`stats`) que incluem latência por componente, contagem de candidatos em cada estágio e estratégia de fusão utilizada, permitindo análise pós-hoc do desempenho do retrieval.

A degradação graceful é um princípio de design: se o índice vetorial não está disponível, o retriever degrada para BM25 puro; se o índice léxico não está disponível, degrada para ANN puro. O modo real de operação (`RetrievalMode`) é reportado no resultado.

---

## 4.6 Classificação Supervisionada

O módulo de classificação do TalkEx implementa classificação supervisionada multi-label e multi-class operando em múltiplos níveis de granularidade (turno, janela de contexto, conversa). Um princípio central da arquitetura é: **embeddings representam, classificadores decidem**. Nunca tratamos similaridade cosseno como classificação -- embeddings alimentam classificadores como features, não como critério de decisão.

### 4.6.1 Features Heterogêneas

Os classificadores do TalkEx operam sobre vetores de features heterogêneas, não apenas embeddings. O `FeatureSet` é a unidade de composição:

```python
@dataclass(frozen=True)
class FeatureSet:
    features: dict[str, float]
    feature_names: list[str]

    def to_vector(self) -> list[float]:
        return [self.features.get(name, 0.0) for name in self.feature_names]
```

Quatro famílias de features são extraídas e combinadas:

**Features lexicais.** Extraídas diretamente do texto por funções puras:
- `char_count`, `word_count`, `avg_word_length`
- `question_count`, `exclamation_count`
- `uppercase_ratio`, `digit_ratio`

**Features estruturais.** Derivadas de metadados da conversa:
- `is_customer`, `is_agent` (papel do falante)
- `turn_count`, `speaker_count` (estrutura da janela)

**Features de embedding.** Vetores densos pré-computados pelo módulo de embeddings. Representam a semântica do texto em dimensionalidade fixa (tipicamente 384 a 1024 dimensões).

**Features de regras.** Flags e scores produzidos pelo motor de regras podem ser incorporados como features adicionais, criando um ciclo de retroalimentação entre regras e classificação.

A função `merge_feature_sets` combina múltiplos `FeatureSet` em um único vetor, preservando a ordem e deduplicando nomes:

```python
def merge_feature_sets(*feature_sets: FeatureSet) -> FeatureSet:
    merged_features: dict[str, float] = {}
    merged_names: list[str] = []
    # ... merge logic preservando ordem
    return FeatureSet(features=merged_features, feature_names=merged_names)
```

### 4.6.2 Modelos de Classificação

Três arquiteturas de classificadores são implementadas, cada uma adequada a um perfil de features e requisitos:

**Logistic Regression (baseline).** Modelo linear adequado para features de alta dimensionalidade com separação linear. Baseline obrigatório contra o qual todos os demais modelos são comparados. Rápido de treinar, interpretável via coeficientes, eficaz quando o poder preditivo reside principalmente nos embeddings.

**LightGBM (Gradient Boosting).** Modelo baseado em árvores, particularmente eficaz para features heterogêneas (numéricas em escalas diferentes, categóricas codificadas). Captura interações não-lineares entre features lexicais, estruturais e embeddings sem exigir normalização de features. Recomendado como modelo primário de produção.

**MLP (Multi-Layer Perceptron).** Rede neural densa para features exclusivamente numéricas. Captura relações não-lineares nos embeddings, mas requer mais dados de treinamento e cuidado com regularização. Indicado quando os embeddings são a feature dominante.

A Tabela 4.2 sumariza as características dos três classificadores.

| Classificador      | Tipo de Feature        | Complexidade    | Interpretabilidade | Uso Primário        |
|--------------------|------------------------|-----------------|--------------------|--------------------|
| LogisticRegression | Embeddings + numéricas | Baixa           | Alta               | Baseline            |
| LightGBM          | Heterogêneas           | Média           | Média (SHAP)       | Produção            |
| MLP                | Embeddings + numéricas | Alta            | Baixa              | Dominância embeddings|

### 4.6.3 Modos de Classificação

O TalkEx suporta dois modos de operação:

- **Single-label** (`ClassificationMode.SINGLE_LABEL`): exatamente um label por input, selecionado por argmax sobre os scores.
- **Multi-label** (`ClassificationMode.MULTI_LABEL`): zero ou mais labels por input, cada um avaliado independentemente contra seu threshold.

A classificação opera em três níveis de granularidade:

- **Turno** (`ClassificationLevel.TURN`): classificação de intenção pontual.
- **Janela** (`ClassificationLevel.WINDOW`): classificação contextual (caso de uso primário).
- **Conversa** (`ClassificationLevel.CONVERSATION`): classificação macro (desfecho, tema dominante).

### 4.6.4 Output de Classificação

Cada predição é encapsulada em um `ClassificationResult` que carrega todos os scores (não apenas os positivos) para observabilidade:

```python
@dataclass(frozen=True)
class ClassificationResult:
    source_id: str
    source_type: str
    label_scores: list[LabelScore]    # Todos os labels, ordenados por score
    model_name: str
    model_version: str
    stats: dict[str, Any]             # Latência, contagem de features, etc.
```

Cada `LabelScore` carrega `label`, `score`, `confidence` e `threshold`, com a propriedade `is_positive` indicando se o score atinge o threshold. A distinção entre `score` (output bruto do classificador) e `confidence` (confiança calibrada) permite recalibração sem alterar o classificador.

O `ClassificationOrchestrator` coordena a execução de múltiplos classificadores sobre um batch de inputs, produzindo `ClassificationBatchResult` com metadados de execução.

### 4.6.5 Fronteira de Domínio: ClassificationResult -> Prediction

Na fronteira entre o subsistema de classificação e o modelo de domínio, cada `ClassificationResult` positivo é mapeado para uma entidade `Prediction`, que é a representação persistente e auditável:

```
ClassificationResult (interno ao subsistema)
    |
    +---> para cada label com is_positive == True:
    |       Prediction(
    |           prediction_id = pred_<uuid>,
    |           source_id = ...,
    |           label = ...,
    |           score = ...,
    |           confidence = ...,
    |           threshold = ...,
    |           model_name = ...,
    |           model_version = ...
    |       )
    v
list[Prediction]  (entidades de domínio)
```

Esta separação entre tipos internos e tipos de domínio segue o princípio de inversão de dependência: o subsistema de classificação define seus próprios tipos de dados; a conversão para entidades de domínio ocorre na fronteira.

---

## 4.7 Motor de Regras Semânticas (DSL -> AST)

O motor de regras semânticas do TalkEx é o componente mais diferenciado da arquitetura. Permite definir regras determinísticas auditáveis que combinam sinais lexicais, semânticos, estruturais e contextuais em uma única expressão lógica. A hipótese H3 desta dissertação postula que regras determinísticas complementam o pipeline estatístico, melhorando precisão em classes críticas.

### 4.7.1 Visão Geral da Arquitetura

O motor de regras opera em três fases:

```
  Texto DSL                   AST (imutável)              Resultado + Evidência
+-----------+    Parser    +-------------+   Evaluator   +------------------+
| RULE ...  | ----------> | AndNode     | ------------> | RuleResult       |
| WHEN ...  |             |   Pred(lex) |               |   matched: true  |
| THEN ...  |             |   Pred(sem) |               |   score: 0.92    |
+-----------+             +-------------+               |   evidence: [..] |
                                                        +------------------+
```

1. **DSL (Domain-Specific Language)**: linguagem legível para definição de regras.
2. **Parser**: compilador recursive-descent que transforma DSL em AST tipada.
3. **Executor (Evaluator)**: caminhador de AST com short-circuit e coleta de evidência.

### 4.7.2 Sintaxe da DSL

A DSL suporta duas formas sintáticas:

**Sintaxe inline** (expressões diretas):

```
keyword("billing") AND speaker("customer")

intent("cancel_subscription") AND NOT keyword("upgrade")
```

**Sintaxe de bloco** (regras nomeadas com ações):

```
RULE risco_cancelamento
WHEN
    speaker == "customer"
    AND semantic.intent("cancelamento") > 0.82
    AND lexical.contains_any(["cancelar", "encerrar", "desistir"])
THEN
    tag("cancelamento_risco") score(0.95) priority("high")
```

A gramática formal em EBNF:

```
program      = rule_block | expression
rule_block   = "RULE" IDENTIFIER "WHEN" expression "THEN" action+
expression   = or_expr
or_expr      = and_expr ( "OR" and_expr )*
and_expr     = not_expr ( "AND" not_expr )*
not_expr     = "NOT" not_expr | atom
atom         = dotted_expr | infix_comparison | predicate | "(" expression ")"
dotted_expr  = IDENTIFIER "." method_chain [comparison_op value]
predicate    = predicate_name "(" arguments ")"
action       = IDENTIFIER "(" arguments ")"
```

A sintaxe de bloco utiliza namespaces pontuados (dotted) que se mapeiam para os mesmos predicados da sintaxe inline:

| Dotted Syntax                                    | Inline Equivalente         |
|--------------------------------------------------|---------------------------|
| `lexical.contains("billing")`                    | `keyword("billing")`      |
| `lexical.word("cancelar")`                       | `word("cancelar")`        |
| `lexical.contains_any(["cancelar", "encerrar"])` | `contains_any([...])`     |
| `semantic.intent("cancelamento") > 0.82`         | `intent("cancelamento")`  |
| `speaker == "customer"`                          | `speaker("customer")`     |

### 4.7.3 As Quatro Famílias de Predicados

Os predicados da DSL são organizados em quatro famílias, cada uma com um custo computacional relativo que governa a ordem de avaliação em short-circuit:

#### Família Lexical (custo = 1)

Predicados de correspondência textual. Todos aplicam `normalize_for_matching` antes da comparação:

| Predicado        | Descrição                                         | Exemplo                                     |
|------------------|----------------------------------------------------|----------------------------------------------|
| `contains`       | Substring match                                    | `lexical.contains("fatura")`                |
| `word`           | Match por palavra inteira (word boundary)          | `lexical.word("cancelar")`                  |
| `stem`           | Match por prefixo de palavra (stemming simples)    | `lexical.stem("cancel")`                    |
| `contains_any`   | Qualquer palavra da lista presente                 | `lexical.contains_any(["cancelar", "encerrar"])` |
| `contains_all`   | Todas as palavras da lista presentes               | `lexical.contains_all(["cancelar", "conta"])` |
| `not_contains`   | Texto NÃO contém o valor                           | `lexical.not_contains("teste")`             |
| `excludes_any`   | Texto NÃO contém nenhuma das palavras              | `lexical.excludes_any(["teste", "debug"])`  |
| `near`           | Duas palavras dentro de N palavras de distância    | `lexical.near("cancelar", "conta", 3)`      |
| `starts_with`    | Texto inicia com prefixo                           | `lexical.starts_with("FAT-")`              |
| `ends_with`      | Texto termina com sufixo                           | `lexical.ends_with(".pdf")`                |
| `regex`          | Match por expressão regular (accent-normalized)    | `lexical.regex("cancel\|terminate")`         |

#### Família Estrutural (custo = 2)

Predicados sobre metadados e campos estruturais:

| Predicado    | Descrição                        | Exemplo                    |
|--------------|----------------------------------|----------------------------|
| `speaker`    | Igualdade de papel do falante    | `speaker == "customer"`    |
| `channel`    | Igualdade de canal               | `channel == "voice"`       |
| `field_eq`   | Igualdade genérica de campo      | `field_eq("queue", "sac")` |
| `field_gte`  | Comparação numérica >=           | `field_gte("duration", 300)` |
| `field_lte`  | Comparação numérica <=           | `field_lte("duration", 60)` |

#### Família Contextual (custo = 3)

Predicados que analisam padrões ao longo de múltiplos turnos:

| Predicado      | Descrição                                            | Exemplo                              |
|----------------|------------------------------------------------------|--------------------------------------|
| `repeated`     | Menção repetida N vezes na janela                    | `repeated("text", "cancelar", 3)`    |
| `occurs_after` | Ocorrência sequencial (A antes de B)                 | `occurs_after("text", "problema", "cancelar")` |

#### Família Semântica (custo = 4)

Predicados que consultam scores pré-computados de modelos de embedding:

| Predicado    | Descrição                                  | Exemplo                                        |
|--------------|--------------------------------------------|-------------------------------------------------|
| `intent`     | Score de intenção >= threshold             | `semantic.intent("cancelamento") > 0.82`       |
| `similarity` | Similaridade de embedding >= threshold     | `semantic.similarity("quero cancelar") > 0.86` |

Predicados semânticos não computam embeddings em tempo de avaliação -- eles consultam scores pré-computados e armazenados no `features` do `RuleEvaluationInput`. A computação de embedding pertence ao pipeline de embeddings; o motor de regras apenas lê resultados.

### 4.7.4 Parser Recursive-Descent

O parser implementa um compilador em duas fases:

**Fase 1: Tokenização.** O texto DSL é transformado em uma sequência de tokens tipados:

```
AND, OR, NOT        -- operadores lógicos (case-insensitive)
RULE, WHEN, THEN    -- keywords de bloco
LPAREN, RPAREN      -- parênteses de agrupamento
LBRACKET, RBRACKET  -- colchetes para listas
STRING, NUMBER       -- literais
IDENTIFIER          -- nomes de função/campo
EQ_OP, GT, GTE, ...  -- operadores de comparação
DOT                 -- separador de namespace
```

**Fase 2: Parsing.** Um parser recursive-descent consome a sequência de tokens e produz uma árvore AST. O parser consulta o `PREDICATE_REGISTRY` para resolver nomes de função em atributos de `PredicateNode`:

```python
PREDICATE_REGISTRY = {
    # (predicate_type, default_field, operator, cost_hint)
    "keyword":      ("lexical",    "text",               "contains", 1),
    "intent":       ("semantic",   "intent_score",       "gte",      4),
    "speaker":      ("structural", "speaker_role",       "eq",       2),
    "repeated":     ("contextual", "field",              "repeated_in_window", 3),
    # ... 18 predicados registrados
}
```

Para a sintaxe de bloco com namespaces pontuados, o `NAMESPACE_PREDICATE_MAP` resolve `(namespace, método)` para o nome do predicado no registry:

```python
NAMESPACE_PREDICATE_MAP = {
    ("lexical", "contains"):  "keyword",
    ("semantic", "intent"):   "intent",
    ("lexical", "word"):      "word",
    # ...
}
```

### 4.7.5 Abstract Syntax Tree (AST)

A AST é composta por quatro tipos de nós imutáveis (frozen dataclasses):

```python
@dataclass(frozen=True)
class PredicateNode:     # Folha: verificação concreta
    predicate_type: PredicateType
    field_name: str
    operator: str
    value: Any
    threshold: float | None = None
    cost_hint: int = 1
    metadata: dict[str, Any]

@dataclass(frozen=True)
class AndNode:           # Todos os filhos devem casar
    children: list[ASTNode]

@dataclass(frozen=True)
class OrNode:            # Ao menos um filho deve casar
    children: list[ASTNode]

@dataclass(frozen=True)
class NotNode:           # Filho NÃO deve casar
    child: ASTNode
```

A regra DSL do exemplo anterior compila para:

```
AndNode
  +-- PredicateNode(structural, speaker_role, eq, "customer", cost=2)
  +-- PredicateNode(semantic, intent_score, gte, "cancelamento", threshold=0.82, cost=4)
  +-- PredicateNode(lexical, text, contains_any, ["cancelar","encerrar","desistir"], cost=1)
```

### 4.7.6 Executor com Short-Circuit e Evidência

O `SimpleRuleEvaluator` caminha a AST recursivamente, avaliando predicados contra um `RuleEvaluationInput`:

```python
class SimpleRuleEvaluator:
    def evaluate(
        self,
        rules: list[RuleDefinition],
        evaluation_input: RuleEvaluationInput,
        config: RuleEngineConfig,
    ) -> list[RuleResult]:
```

**Short-circuit.** A avaliação implementa curto-circuito em dois níveis:

1. **Intra-regra**: em nós `AND`, a avaliação para no primeiro filho que falha; em nós `OR`, para no primeiro que sucede.
2. **Inter-regra**: no modo `SHORT_CIRCUIT`, a avaliação de regras para após a primeira regra que ativa.

**Ordenação por custo.** A política `COST_ASCENDING` reordena filhos de nós AND/OR por `cost_hint` antes da avaliação:

```
AndNode (COST_ASCENDING)
  1. PredicateNode(lexical, cost=1)     <-- avaliado primeiro
  2. PredicateNode(structural, cost=2)
  3. PredicateNode(semantic, cost=4)    <-- avaliado por último (ou nem avaliado)
```

Se o predicado lexical (custo 1) falha, os predicados mais caros (custo 2 e 4) nunca são executados. Esta otimização é particularmente eficaz quando predicados semânticos exigem consulta a embeddings.

**Coleta de evidência.** Cada predicado avaliado produz um `PredicateResult` com:

```python
@dataclass
class PredicateResult:
    predicate_type: PredicateType
    field_name: str
    operator: str
    matched: bool
    score: float
    threshold: float | None = None
    matched_text: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any]
```

A política de evidência (`EvidencePolicy`) controla quanta evidência é coletada:
- `ALWAYS`: coleta para predicados matched e unmatched (auditoria completa).
- `MATCH_ONLY`: coleta apenas para predicados matched (menor custo, uso em produção).

### 4.7.7 Exemplo Completo de Avaliação

Considere a regra:

```
RULE risco_cancelamento
WHEN
    speaker == "customer"
    AND semantic.intent("cancelamento") > 0.82
    AND lexical.contains_any(["cancelar", "encerrar", "desistir"])
THEN
    tag("cancelamento_risco") score(0.95) priority("high")
```

Avaliada contra o input:

```python
RuleEvaluationInput(
    text="eu quero cancelar minha conta, nao quero mais esse servico",
    speaker_role="customer",
    features={"intent_score": 0.91},
    source_type="context_window",
    source_id="win_abc123",
)
```

Com política `COST_ASCENDING`, a execução procede:

1. **lexical.contains_any** (custo 1): normaliza texto e busca ["cancelar", "encerrar", "desistir"]. Encontra "cancelar". `matched=True, matched_text="cancelar"`.
2. **speaker == "customer"** (custo 2): compara `speaker_role` com "customer". `matched=True`.
3. **semantic.intent("cancelamento") > 0.82** (custo 4): lê `features["intent_score"] = 0.91`, compara com threshold 0.82. `matched=True, score=0.91, threshold=0.82`.

Resultado:

```python
RuleResult(
    rule_id="rule_001",
    rule_name="risco_cancelamento",
    matched=True,
    score=0.91,     # min dos scores (AND)
    predicate_results=[...],  # 3 PredicateResults com evidência
    short_circuited=False,    # todos os predicados avaliados
    execution_time_ms=0.12,
)
```

### 4.7.8 Fronteira de Domínio: RuleResult -> RuleExecution

Na fronteira entre o motor de regras e o modelo de domínio, cada `RuleResult` é mapeado para uma entidade `RuleExecution`:

```python
def map_to_rule_execution(result: RuleResult) -> RuleExecution:
    evidence = [pr.to_evidence_item()
                for pr in result.predicate_results if pr.matched]

    return RuleExecution(
        rule_id=RuleId(result.rule_id),
        rule_name=result.rule_name,
        source_id=result.source_id,
        source_type=object_type,
        matched=result.matched,
        score=result.score,
        execution_time_ms=result.execution_time_ms,
        evidence=evidence,
        metadata={
            "rule_version": result.rule_version,
            "short_circuited": result.short_circuited,
            "predicate_count": len(result.predicate_results),
        },
    )
```

O `metadata` preserva informações operacionais (versão da regra, se houve short-circuit, contagem de predicados) que são essenciais para debugging e otimização.

---

## 4.8 Inferência em Cascata

A estratégia de inferência em cascata é o mecanismo de otimização de custo central do TalkEx. O princípio é simples: não processar toda conversa com o mesmo nível de investimento computacional. Conversas simples podem ser classificadas por filtros baratos; apenas conversas ambíguas ou críticas justificam modelos caros.

### 4.8.1 Os Quatro Estágios

O pipeline em cascata implementa quatro estágios de custo crescente:

```
+------------------------------------------------------------------+
|  Estágio 1: Filtros Baratos                          Custo: $    |
|  - Idioma, canal, fila, data                                     |
|  - Regras lexicais simples (contains, regex)                     |
|  - Resolução: ~60% das conversas                                 |
+------------------------------------------------------------------+
              |  conversas não resolvidas
              v
+------------------------------------------------------------------+
|  Estágio 2: Retrieval Híbrido                        Custo: $$   |
|  - BM25 + ANN + score fusion                                     |
|  - Classificação por similaridade com protótipos                  |
|  - Resolução: ~25% das conversas                                 |
+------------------------------------------------------------------+
              |  conversas ambíguas
              v
+------------------------------------------------------------------+
|  Estágio 3: Classificação + Regras Semânticas        Custo: $$$  |
|  - Classificadores supervisionados (LogReg, LightGBM, MLP)       |
|  - Regras com predicados semânticos (intent, similarity)         |
|  - Resolução: ~12% das conversas                                 |
+------------------------------------------------------------------+
              |  conversas excepcionais
              v
+------------------------------------------------------------------+
|  Estágio 4: Revisão Excepcional (offline)            Custo: $$$$ |
|  - LLM para casos ambíguos/novos/críticos                        |
|  - Revisão humana assistida                                      |
|  - Resolução: ~3% das conversas                                  |
+------------------------------------------------------------------+
```

### 4.8.2 Estágio 1: Filtros Baratos

Filtros de custo computacional O(1) que resolvem conversas triviais:

- **Filtros estruturais**: idioma, canal, fila de atendimento, produto, região. Conversas de canais ou filas específicas podem ser classificadas diretamente (ex: "fila de cancelamento" -> label "cancelamento" com alta confiança).
- **Regras lexicais simples**: predicados `contains`, `regex`, `contains_any` que não requerem embedding. Palavras-chave inequívocas (ex: "quero cancelar meu plano" com `contains("cancelar")` + `speaker("customer")`) resolvem uma fração significativa das conversas.
- **Thresholds de confiança**: se um filtro barato atinge confiança acima de um limiar configurável (ex: 0.95), a conversa é resolvida sem prosseguir.

A estimativa de 60% de resolução neste estágio é conservadora para operações de call center com vocabulário previsível e filas especializadas.

### 4.8.3 Estágio 2: Retrieval Híbrido

Conversas não resolvidas no Estágio 1 passam pelo retrieval híbrido:

- **BM25 contra protótipos**: a conversa é comparada com exemplos representativos de cada classe via BM25. Scores altos indicam alta similaridade lexical com padrões conhecidos.
- **ANN contra protótipos**: a conversa é comparada via similaridade de embedding com os mesmos protótipos. Captura paráfrase e variação linguística.
- **Fusão e decisão**: a fusão RRF ou linear produz um ranking. Se o score fusionado do top-1 excede um limiar, a classificação é aceita.

Este estágio é significativamente mais barato que classificação supervisionada completa porque opera sobre um conjunto pequeno de protótipos (dezenas a centenas), não sobre o corpus inteiro.

### 4.8.4 Estágio 3: Classificação Completa + Regras Semânticas

Conversas que permanecem ambíguas recebem o tratamento completo:

- **Extração de features heterogêneas**: lexicais + estruturais + embeddings multi-nível.
- **Classificação supervisionada**: LightGBM (primário) ou LogReg/MLP, operando sobre o vetor de features completo.
- **Regras semânticas**: o motor de regras avalia regras com predicados das quatro famílias, incluindo predicados semânticos (`intent`, `similarity`) que consultam scores de embedding.
- **Decisão combinada**: o resultado final combina predições dos classificadores com decisões das regras, priorizando regras para classes críticas (compliance, fraude).

### 4.8.5 Estágio 4: Revisão Excepcional

Conversas que nenhum estágio anterior resolveu com confiança suficiente são encaminhadas para revisão excepcional:

- **LLM offline**: modelos de linguagem (GPT-4, Claude, Llama) geram labels candidatos com justificativa textual. Operam exclusivamente em modo offline (batch), nunca em tempo real.
- **Revisão humana assistida**: em cenários de alta criticidade (compliance, fraude), um revisor humano recebe a conversa com as predições parciais dos estágios anteriores como contexto.
- **Feedback loop**: labels produzidos neste estágio alimentam o conjunto de treinamento para retreinar classificadores do Estágio 3, melhorando progressivamente a cobertura dos estágios anteriores.

### 4.8.6 Lógica de Decisão

A decisão de promover uma conversa para o estágio seguinte versus aceitar a classificação atual baseia-se em dois critérios:

1. **Confiança**: o score ou confiança da predição excede o threshold configurado para o estágio?
2. **Criticidade**: a classe predita é uma classe crítica (compliance, fraude) que exige maior confiança?

```
Se confiança >= threshold_estágio:
    ACEITAR predição no estágio atual
Senão se classe_crítica AND confiança >= threshold_crítico:
    ACEITAR com flag de revisão
Senão:
    PROMOVER para estágio seguinte
```

Os thresholds são configuráveis por classe e por estágio, permitindo calibração fina do trade-off custo-qualidade. A Seção 6.4 apresentará os resultados experimentais da curva de Pareto entre custo e qualidade para diferentes configurações de thresholds.

### 4.8.7 Impacto no Custo

A cascata reduz custo computacional ao concentrar processamento caro em uma fração do volume. Considere um volume hipotético de 10.000 conversas/hora:

| Estágio | Conversas | Custo/Conversa | Custo Total |
|---------|-----------|----------------|-------------|
| 1       | 10.000    | $0.001         | $10         |
| 2       | 4.000     | $0.01          | $40         |
| 3       | 1.500     | $0.10          | $150        |
| 4       | 300       | $1.00          | $300        |
| **Total** | --      | --             | **$500**    |

Sem cascata, todas as 10.000 conversas passariam pelo Estágio 3, custando $1.000, e 10.000 pelo Estágio 4 (se aplicável), custando $10.000. A cascata reduz o custo em aproximadamente 50% (comparando com Estágio 3 para todas) a 95% (comparando com Estágio 4 para todas).

---

## 4.9 Decisões Arquiteturais (ADRs)

As decisões arquiteturais do TalkEx são formalizadas em Architecture Decision Records (ADRs), seguindo a prática de documentar não apenas o que foi decidido, mas também o que foi rejeitado e por quê. Esta seção sumariza os quatro ADRs existentes.

### ADR-001: Package Layout e API Pública

**Status**: Aceito.

**Contexto**: o projeto necessitava de uma estrutura de pacote Python que suportasse instalação editável, importações limpas, tooling (ruff, mypy, pytest) e extensibilidade.

**Decisão**: adotar o layout `src/` com `talkex` como pacote real. O diretório `src/` é invisível para consumidores; todas as importações usam o namespace `talkex`.

**Alternativa rejeitada**: usar `src` como namespace package (`import src.models`). Rejeitada porque mistura infraestrutura de build com identidade do pacote.

**Consequências positivas**:
- `src` nunca faz parte de nenhum caminho de importação -- prática padrão de packaging Python.
- `pip install -e .` funciona corretamente com `[tool.setuptools.packages.find] where = ["src"]`.
- Todas as ferramentas (ruff, mypy, pytest) resolvem o pacote via `pythonpath = ["src"]`.

**Regras derivadas**:
- Todas as re-exportações públicas usam `__all__` explícito.
- Símbolos privados recebem prefixo `_`.
- Nenhum subpacote deve ser importado por consumidores sem ser re-exportado pelo `__init__.py` pai.

### ADR-002: Modelos Pydantic Frozen e Strict

**Status**: Aceito.

**Contexto**: as seis entidades de domínio (Conversation, Turn, ContextWindow, EmbeddingRecord, Prediction, RuleExecution) são contratos entre todos os estágios do pipeline. Qualquer mutação após criação introduziria bugs sutis em um pipeline multi-estágio concorrente.

**Decisão**: todos os seis modelos utilizam `model_config = ConfigDict(frozen=True, strict=True)`.

**Consequências positivas**:
- **Imutabilidade**: estágios do pipeline não podem mutar estado compartilhado acidentalmente.
- **Tipagem estrita**: sem coerção silenciosa -- `"0.95"` não se torna `0.95` implicitamente.
- **Debugabilidade**: os valores de uma instância são exatamente os que foram passados na construção.

**Consequências negativas**:
- Código que passa strings onde ints são esperados quebrará. Intencional -- detecta bugs cedo.
- Deserialização de JSON/dict requer tipos exatos. Fronteiras de deserialização podem usar `strict=False` explicitamente.

### ADR-003: Vetores de Embedding como list[float]

**Status**: Aceito.

**Contexto**: `EmbeddingRecord` armazena vetores densos consumidos por FAISS, classificadores e motor de regras. O tipo natural de computação é `numpy.ndarray`, mas o tipo natural de serialização é `list[float]`.

**Decisão**: `vector: list[float]` no modelo Pydantic. Conversão para `ndarray` ocorre nas fronteiras dos módulos de computação.

**Alternativas rejeitadas**:
- `ndarray` no modelo: requer serializers customizados, complica JSON round-trips, acopla numpy na camada de dados.
- Ambos os campos (`vector` + `array`): viola single source of truth, dobra memória, adiciona complexidade de sincronização.

**Consequências positivas**:
- Serialização limpa: `model_dump()` e `model_validate()` funcionam sem serializers customizados.
- Zero acoplamento com numpy na camada de dados.
- Compatível com strict mode do Pydantic v2.

**Regras derivadas**:
- Conversão para `ndarray` é feita uma vez na fronteira do módulo consumidor.
- Nenhum import de numpy em `src/talkex/models/`.

### ADR-004: Campos Estruturais em ContextWindow

**Status**: Aceito.

**Contexto**: o contrato inicial de `ContextWindow` incluía apenas campos de identidade e conteúdo. Durante a implementação, tornou-se evidente que sem informação posicional e paramétrica, uma janela de contexto não é reproduzível.

**Decisão**: expandir `ContextWindow` com quatro campos estruturais:
- `start_index` / `end_index`: posição do primeiro/último turno na conversa pai (0-based).
- `window_size`: redundante com `len(turn_ids)` por design, com validador cruzado.
- `stride`: parâmetro de passo utilizado na geração.

**Consequências**:
- `ContextWindow` é agora um registro auto-descritivo e auditável.
- O invariante `window_size == len(turn_ids)` detecta bugs do construtor em tempo de construção.
- Campos deferidos (`role_aware_views`, `embedding_id`) serão adicionados quando seus pipelines forem implementados.

---

## 4.10 Considerações do Capítulo

A arquitetura do TalkEx foi projetada para testar as quatro hipóteses desta dissertação de forma modular e mensurável. Cada componente pode ser avaliado isoladamente (BM25 vs ANN vs híbrido) e em composição (cascata completa vs pipeline uniforme).

Os três princípios de design -- modularidade, inferência em cascata e separação online/offline -- não são apenas princípios teóricos: estão codificados na estrutura de pacotes, nos protocolos entre módulos e nas configurações de runtime. As quatro ADRs formalizam decisões que afetam a integridade dos dados (frozen/strict), a interoperabilidade entre componentes (list[float] vs ndarray) e a reprodutibilidade dos experimentos (campos estruturais em ContextWindow).
