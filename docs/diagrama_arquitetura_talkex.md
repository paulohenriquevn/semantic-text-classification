# TalkEx — Diagrama de Arquitetura

## Visão Geral do Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            TalkEx: Pipeline Híbrido Cascateado                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  CONVERSAS   │  Voz, Chat, E-mail (PT-BR)
    │   BRUTAS     │  "[customer] Quero cancelar... [agent] Posso ajudar..."
    └──────┬───────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ESTÁGIO 1: INGESTÃO E SEGMENTAÇÃO                                                  │
│                                                                                      │
│  ┌────────────────┐    ┌─────────────────────┐    ┌────────────────────────┐        │
│  │ TranscriptInput│───▶│  TurnSegmenter      │───▶│  Lista de Turnos       │        │
│  │ (validação)    │    │  • Parser [customer] │    │  [{speaker, text}, ...]│        │
│  └────────────────┘    │  • Normalização NFKC │    └────────────────────────┘        │
│                        │  • Features lexicais │                                      │
│                        └─────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ESTÁGIO 2: JANELAS DE CONTEXTO                                                     │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  SlidingWindowBuilder (window_size=5, stride=2)                            │     │
│  │                                                                            │     │
│  │  Turno1  Turno2  Turno3  Turno4  Turno5  Turno6  Turno7  Turno8          │     │
│  │  ├──────────────────────────────────┤                                      │     │
│  │           Janela 1 (turnos 1-5)                                            │     │
│  │                   ├──────────────────────────────────┤                     │     │
│  │                            Janela 2 (turnos 3-7)                           │     │
│  │                                    ├──────────────────────────────────┤    │     │
│  │                                             Janela 3 (turnos 5-8+pad) │    │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  Saída: ContextWindow {window_id, window_text, turn_ids, métricas}                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ESTÁGIO 3: NORMALIZAÇÃO DE TEXTO                                                    │
│                                                                                      │
│  ┌───────────────────────────┐      ┌───────────────────────────────────┐           │
│  │  NFKC (segmentação)       │      │  NFD (matching lexical)           │           │
│  │  • Canonicalização Unicode │      │  • Remoção de acentos             │           │
│  │  • ﬁ → fi, ² → 2         │      │  • "não" → "nao", "café" → "cafe"│           │
│  └───────────────────────────┘      └───────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ESTÁGIO 4: GERAÇÃO DE EMBEDDINGS                                                    │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐                 │
│  │  paraphrase-multilingual-MiniLM-L12-v2 (CONGELADO)            │                 │
│  │                                                                │                 │
│  │  "Quero cancelar meu plano" ──▶ [0.12, -0.34, 0.56, ..., 0.08]│                │
│  │                                     384 dimensões               │                 │
│  │  • L2 normalizado                                              │                 │
│  │  • Batch processing (64 textos por vez)                        │                 │
│  │  • GPU (Tesla T4) para aceleração                              │                 │
│  └────────────────────────────────────────────────────────────────┘                 │
│                                                                                      │
│  Saída: EmbeddingRecord {embedding_id, vector[384], model_name, version}            │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │
           ├──────────────────────────────────────────────┐
           ▼                                              ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────────────┐
│  ESTÁGIO 5A: ÍNDICE BM25        │    │  ESTÁGIO 5B: ÍNDICE VETORIAL             │
│                                  │    │                                          │
│  ┌──────────────────────────┐   │    │  ┌──────────────────────────────────┐   │
│  │  InMemoryBM25Index       │   │    │  │  InMemoryVectorIndex             │   │
│  │  • Tokenização + NFD     │   │    │  │  • Similaridade cosseno          │   │
│  │  • IDF caching           │   │    │  │  • Busca exata (numpy)           │   │
│  │  • BM25 Okapi (k1=2.0,  │   │    │  │  • Equivalente a FAISS flat      │   │
│  │    b=0.75)               │   │    │  │  • 384 dimensões                 │   │
│  └──────────────────────────┘   │    │  └──────────────────────────────────┘   │
│                                  │    │                                          │
│  Sinal: correspondência exata    │    │  Sinal: similaridade semântica           │
│  "cancelar" encontra "cancelar"  │    │  "desistir" encontra "cancelar"          │
└──────────────────┬───────────────┘    └────────────────────┬─────────────────────┘
                   │                                          │
                   └──────────────────┬───────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ESTÁGIO 5C: RECUPERAÇÃO HÍBRIDA (FUSÃO)                                             │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │                                                                            │     │
│  │   Scores BM25 ──┐                                                         │     │
│  │   (normalizados) ├──▶ FUSÃO LINEAR: score = α×ANN + (1-α)×BM25           │     │
│  │   Scores ANN  ──┘     (α = 0.5 — peso semântico)                         │     │
│  │                                                                            │     │
│  │   Alternativa: RRF = Σ 1/(k + rank_i), k=60                              │     │
│  │                                                                            │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  Saída: RetrievalResult {hits[top-K], scores, modo_usado}                           │
│  Métricas: MRR = 0.8516 (RRF), 0.8482 (LINEAR α=0.5)                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ESTÁGIO 6: CLASSIFICAÇÃO SUPERVISIONADA                                             │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  EXTRAÇÃO DE FEATURES (397 total)                                          │     │
│  │                                                                            │     │
│  │  ┌──────────────────┐ ┌───────────────────┐ ┌────────────┐ ┌───────────┐ │     │
│  │  │ Embeddings (384) │ │ Lexicais (7)      │ │Estrut. (4) │ │Regras (2) │ │     │
│  │  │ • vetor MiniLM   │ │ • word_count      │ │• is_customer│ │• cancel   │ │     │
│  │  │                  │ │ • char_count      │ │• is_agent   │ │• complaint│ │     │
│  │  │                  │ │ • avg_word_length │ │• turn_count │ │           │ │     │
│  │  │                  │ │ • question_count  │ │• speaker_cnt│ │           │ │     │
│  │  │                  │ │ • exclamation_cnt │ └────────────┘ └───────────┘ │     │
│  │  │                  │ │ • uppercase_ratio │                              │     │
│  │  │                  │ │ • digit_ratio     │                              │     │
│  │  └──────────────────┘ └───────────────────┘                              │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  LightGBM Classifier                                                      │     │
│  │  • 100 estimadores, 31 folhas                                             │     │
│  │  • Treinado em ~6 segundos (CPU)                                          │     │
│  │  • Saída: probabilidades para cada uma das 8 classes                      │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  Saída: Prediction {label, score, confidence, threshold, model_version}             │
│  Métrica: Macro-F1 = 0.7216                                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ESTÁGIO 7: MOTOR DE REGRAS SEMÂNTICAS (DSL)                                        │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  DSL → Parser → AST → Evaluator                                           │     │
│  │                                                                            │     │
│  │  RULE risco_cancelamento                                                   │     │
│  │    WHEN speaker == "customer"                                              │     │
│  │     AND semantic.intent("cancelamento") > 0.82                            │     │
│  │     AND lexical.contains_any(["cancelar", "encerrar", "desistir"])        │     │
│  │    THEN tag("cancelamento_risco") score(0.95) priority("high")            │     │
│  │                                                                            │     │
│  │  ┌──────────────────────────────────────────────────────────┐             │     │
│  │  │  4 Famílias de Predicados (avaliação por custo crescente)│             │     │
│  │  │                                                          │             │     │
│  │  │  ① Léxicos (custo 1)     → palavras-chave, regex        │             │     │
│  │  │  ② Estruturais (custo 2) → speaker, channel, campos     │             │     │
│  │  │  ③ Contextuais (custo 3) → repetição, sequência         │             │     │
│  │  │  ④ Semânticos (custo 4)  → scores de embeddings         │             │     │
│  │  │                                                          │             │     │
│  │  │  Short-circuit: AND para no 1o falso, OR para no 1o true │             │     │
│  │  └──────────────────────────────────────────────────────────┘             │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  10 regras cobrindo as 8 classes de intenção                                        │
│  Saída: RuleExecution {matched, score, evidence[], execution_time_ms}               │
│  Uso: como features suaves para o classificador (Macro-F1 = 0.7400)                │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ESTÁGIO 8: AGREGAÇÃO JANELA → CONVERSA                                             │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │                                                                            │     │
│  │  Conversa "conv_123" tem 3 janelas:                                       │     │
│  │                                                                            │     │
│  │  Janela 1: P(cancel)=0.9, P(reclam)=0.05, P(suporte)=0.02, ...          │     │
│  │  Janela 2: P(cancel)=0.7, P(reclam)=0.15, P(suporte)=0.08, ...          │     │
│  │  Janela 3: P(cancel)=0.8, P(reclam)=0.10, P(suporte)=0.05, ...          │     │
│  │                                                                            │     │
│  │  ──────────────────────────────────────────────────────────────            │     │
│  │  Média:   P(cancel)=0.8, P(reclam)=0.10, P(suporte)=0.05, ...           │     │
│  │                                                                            │     │
│  │  argmax → CANCELAMENTO (confiança: 0.80)                                  │     │
│  │                                                                            │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  Estratégia: média das probabilidades por classe → argmax                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  SAÍDA FINAL                                                                         │
│                                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐       │
│  │  Classificação  │  │  Evidência       │  │  Analytics                   │       │
│  │  • intenção     │  │  • regras que    │  │  • métricas por classe       │       │
│  │  • confiança    │  │    dispararam    │  │  • tendências temporais      │       │
│  │  • modelo usado │  │  • texto matched │  │  • distribuição de intenções │       │
│  └─────────────────┘  │  • scores        │  └──────────────────────────────┘       │
│                        └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Fluxo de Dados Simplificado

```
Conversa bruta (texto)
    │
    ├──① Segmentação ──▶ Turnos individuais
    │
    ├──② Windowing ────▶ Janelas de 5 turnos (stride 2)
    │
    ├──③ Embeddings ───▶ Vetores 384d (MiniLM-L12-v2)
    │        │
    │        ├──────────▶ Índice Vetorial (busca semântica)
    │        │
    │        └──────────▶ Features para classificação
    │
    ├──④ BM25 Index ───▶ Índice lexical (busca por palavras)
    │        │
    │        └──────────┐
    │                   ▼
    ├──⑤ Fusão ────────▶ Recuperação híbrida (MRR=0.85)
    │
    ├──⑥ Features ─────▶ 384 emb + 7 lex + 4 struct + 2 rules = 397
    │        │
    │        └──────────▶ LightGBM ──▶ Probabilidades (8 classes)
    │
    ├──⑦ Regras DSL ───▶ 10 regras → features binárias + evidência
    │
    └──⑧ Agregação ────▶ Média por conversa → Intenção final (F1=0.74)
```

---

## Módulos do Código-Fonte

```
src/talkex/
│
├── ingestion/          ← Estágio 1: Validação de entrada
│   ├── inputs.py           TranscriptInput (boundary object)
│   └── enums.py            SourceFormat, Channel
│
├── segmentation/       ← Estágio 1: Segmentação de turnos
│   ├── segmenter.py        TurnSegmenter (orquestrador)
│   ├── parsing.py          Parsers (labeled, multiline, plain)
│   ├── normalization.py    Unicode NFKC
│   └── features.py         Features lexicais por turno
│
├── context/            ← Estágio 2: Janelas de contexto
│   ├── builder.py          SlidingWindowBuilder
│   ├── windowing.py        Geração de slices (size=5, stride=2)
│   └── rendering.py        Texto da janela + role views
│
├── embeddings/         ← Estágio 4: Geração de embeddings
│   ├── generator.py        NullEmbeddingGenerator + SentenceTransformerGenerator
│   ├── preprocessing.py    Truncation, language detection
│   ├── pooling.py          Mean/Max/CLS pooling
│   └── cache.py            Cache de embeddings (dedup)
│
├── retrieval/          ← Estágio 5: Recuperação híbrida
│   ├── bm25.py             InMemoryBM25Index (numpy)
│   ├── vector_index.py     InMemoryVectorIndex (cosseno exato)
│   ├── hybrid.py           SimpleHybridRetriever
│   └── fusion.py           RRF + LINEAR fusion
│
├── classification/     ← Estágio 6: Classificação
│   ├── features.py         Extração de features (7 lex + 4 struct)
│   ├── lightgbm_classifier.py  LightGBMClassifier
│   ├── logistic.py         LogisticClassifier (stage 1 cascade)
│   ├── orchestrator.py     ClassificationOrchestrator
│   └── labels.py           LabelSpace (8 classes)
│
├── rules/              ← Estágio 7: Motor de regras
│   ├── dsl.py              Tokenizer + PREDICATE_REGISTRY
│   ├── parser.py           Recursive-descent parser
│   ├── ast.py              PredicateNode, AndNode, OrNode, NotNode
│   ├── compiler.py         SimpleRuleCompiler (DSL → RuleDefinition)
│   └── evaluator.py        SimpleRuleEvaluator (AST walker)
│
├── analytics/          ← Observabilidade
│   ├── aggregators.py      Agregação por dimensão/tempo
│   ├── metrics.py          match_rate, avg_score, p95_latency
│   └── query_runner.py     Motor de consultas analíticas
│
├── pipeline/           ← Orquestração
│   ├── pipeline.py         TextProcessingPipeline (seg + ctx)
│   ├── system_pipeline.py  SystemPipeline (todos os estágios)
│   ├── runner.py           PipelineRunner (CLI/batch)
│   └── cli.py              Click CLI (run, benchmark, config)
│
├── models/             ← Entidades de domínio (frozen, strict)
│   ├── turn.py             Turn
│   ├── context_window.py   ContextWindow
│   ├── conversation.py     Conversation
│   ├── embedding_record.py EmbeddingRecord
│   ├── prediction.py       Prediction
│   └── rule_execution.py   RuleExecution + EvidenceItem
│
├── text_normalization.py  ← NFD accent stripping (shared)
└── exceptions.py          ← Hierarquia de exceções de domínio
```

---

## Cascata de Custos (H4)

```
┌─────────────────────────────────────────────────────────────────┐
│  INFERÊNCIA CASCATEADA                                           │
│                                                                  │
│  ┌──────────────┐     confiança ≥ threshold?                    │
│  │  ESTÁGIO 1   │────────────────────────────▶ SIM → Aceita     │
│  │  LogReg      │         │                                      │
│  │  (leve)      │         │ NÃO                                  │
│  └──────────────┘         ▼                                      │
│                   ┌──────────────┐                               │
│                   │  ESTÁGIO 2   │──────────────▶ Aceita         │
│                   │  LightGBM    │                               │
│                   │  (pesado)    │                               │
│                   └──────────────┘                               │
│                                                                  │
│  Thresholds testados: 0.50, 0.60, 0.70, 0.80, 0.90             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integração de Regras (H3)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ESTRATÉGIA 1: ML-only (baseline)                               │
│  Features [397] ──▶ LightGBM ──▶ Predição                      │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  ESTRATÉGIA 2: Rules-as-override (PREJUDICA)                    │
│  Features [397] ──▶ LightGBM ──▶ Predição ML                   │
│  Texto ──▶ Regras ──▶ Match? ──SIM──▶ SOBRESCREVE (❌ -5.8%)   │
│                         │                                        │
│                        NÃO ──▶ Usa predição ML                  │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  ESTRATÉGIA 3: Rules-as-features (AJUDA)                        │
│  Features [397] + Rule matches [10] ──▶ LightGBM ──▶ Predição  │
│                     ▲                        (✅ +2.5%)          │
│                     │                                            │
│  Texto ──▶ Regras ──┘ (binárias: 0 ou 1)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
