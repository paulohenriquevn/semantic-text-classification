# Desenho Experimental — Dissertação de Mestrado

## Visão Geral

Este documento detalha o protocolo experimental para validar as 4 hipóteses da dissertação. Cada hipótese é tratada como um experimento independente, mas todos compartilham o mesmo dataset e infraestrutura.

```
H1: Retrieval Híbrido > Isolados
H2: Multi-Nível > Nível Único
H3: Pipeline + Regras > Pipeline sem Regras (em classes críticas)
H4: Cascata reduz custo ≥40% com degradação F1 < 2%
```

---

## 1. Dataset

### 1.1 Fontes de Dados

O experimento utiliza um corpus sintético expandido com controle de variabilidade, e valida robustez via ablation no dataset original.

#### Corpus — Conversas sintéticas expandidas (PT-BR)
- **Base:** `RichardSakaguchiMS/brazilian-customer-service-conversations` (944 conversas, Apache 2.0)
- **Expansão:** Geração sintética controlada via LLM (Claude Sonnet, batch mode, offline)
- **Alvo:** ~3.500 conversas multi-turn
- **Correções sobre o dataset original:**
  - Turnos variáveis: 4-20 (distribuição log-normal, média 8, stdev 4) — original: 90% = 8 turnos
  - Distribuição de classes realista: desbalanceada — original: ~11% por classe (uniforme)
  - Variabilidade lexical: 5 personas (formal, informal, irritado, idoso, jovem/gírias) — original: estilo uniforme
  - Sentimentos por intent: distribuição condicionada ao intent (reclamação→65% neg/35% neu, elogio→75% pos/25% neu, etc.) — original: 33%/33%/33% sem restrições
- **Vantagens:** Controle de variáveis para ablation studies, reprodutibilidade, acesso aberto
- **Limitação:** Dados sintéticos — afirmações limitadas ao escopo metodológico
- **Uso:** Experimentos H1-H4 (todos)

#### Validação de robustez — Dataset original (944 conversas)
- Todos os experimentos H1-H4 são repetidos no dataset original de 944 conversas (não expandido)
- Se as conclusões se mantêm, a expansão não introduziu artefatos
- Se divergem, a dissertação reporta ambos os resultados e discute causas

#### Sobre validação externa com dados reais
- **Consumidor.gov.br foi investigado e descartado:** CSV não contém texto livre (apenas campos categóricos: Área, Assunto, Grupo Problema, Problema). Plataforma exclusivamente de reclamações — impossível mapear intents como elogio, saudação, dúvida.
- **Nenhum dataset público PT-BR** combina conversas multi-turn + intents de atendimento + texto livre.
- **Trabalho futuro:** Validação com dados reais requer parceria com operador de contact center.

**Justificativa:** A inexistência de datasets conversacionais abertos em PT-BR é uma lacuna do campo, não da dissertação. O rigor vem do controle experimental (expansão controlada) + teste de robustez (ablation no original) + transparência sobre limitações. Ver `research-log.md` (Sessão 2) para raciocínio completo.

### 1.2 Especificação do Corpus Primário

| Dimensão | Valor |
|----------|-------|
| Total de conversas | ~3.500 |
| Turnos por conversa | 4-20 (log-normal, média 8, stdev 4) |
| Classes/intents | 9 (mantidas do dataset original) |
| Idioma | PT-BR (informal, com diacríticos e gírias) |
| Anotação | Intent + sentimento + setor por conversa |
| Splits | Train 70% (~2.450) / Val 15% (~525) / Test 15% (~525) |
| Seed | 42 |

**Distribuição-alvo de intents:**

| Intent | % | Conversas |
|--------|---|-----------|
| reclamacao | 20% | ~700 |
| duvida_produto | 18% | ~630 |
| duvida_servico | 17% | ~595 |
| suporte_tecnico | 15% | ~525 |
| compra | 10% | ~350 |
| cancelamento | 8% | ~280 |
| saudacao | 5% | ~175 |
| elogio | 4% | ~140 |
| outros | 3% | ~105 |

### 1.3 Validação de Dificuldade do Dataset (Phase 0.5)

Antes dos experimentos, validar que o dataset tem dificuldade genuína:

| Verificação | Métrica | Critério de aprovação |
|-------------|---------|----------------------|
| **Baseline de maioria** | Acurácia da classe mais frequente | < 25% (desbalanceado mas não trivial) |
| **Exclusividade lexical** | Score médio de exclusividade por intent | 1.0 < score < 3.0 (nem impossível nem trivial) |
| **Overlap lexical** | % de intents com cross-intent word overlap | > 50% (vocabulário compartilhado) |
| **Separação de embeddings** | Razão inter/intra-classe (cosine) | < 2.0 (não trivialmente separável) |
| **Leakage few-shot** | Contaminação entre train/test via few_shot_ids | 0% no test set |
| **Original vs expandido** | Divergência de distribuição turn count / word count | Sem desvio estatisticamente significativo |

**Resultados baseline (original, 944 convs):** Exclusividade 1.68, overlap 100%, imbalance ratio 1.2.
**Script:** `experiments/scripts/validate_dataset.py`

### 1.5 Pré-processamento

1. **Normalização textual**: `normalize_for_matching()` — lowercase + strip_accents
2. **Segmentação de turnos**: já estruturado no dataset ou via heurística de alternância de falantes
3. **Construção de janelas**: sliding window com parâmetros configuráveis
4. **Geração de embeddings**: múltiplos modelos, cache persistido
5. **Indexação**: BM25 (rank-bm25) + ANN (FAISS)

### 1.6 Splits e Reprodutibilidade

- **Seed fixo** para todos os splits e shuffles (42)
- **Holdout temporal** quando timestamps disponíveis (treino = mais antigos, teste = mais recentes)
- Se não houver timestamps: stratified random split preservando distribuição de classes
- **Cross-validation 5-fold** para estimar variância dos resultados

---

## 2. Métricas de Avaliação

### 2.1 Métricas de Retrieval (H1)

| Métrica | Definição | Por que usar |
|---------|-----------|-------------|
| **Recall@K** | Fração de documentos relevantes recuperados no top-K | Mede cobertura — essencial para pipelines downstream |
| **Precision@K** | Fração de documentos no top-K que são relevantes | Mede qualidade do ranking |
| **MRR** | Mean Reciprocal Rank — 1/posição do primeiro resultado relevante | Mede quão cedo o relevante aparece |
| **nDCG@K** | Normalized Discounted Cumulative Gain | Mede qualidade do ranking considerando posições |
| **MAP@K** | Mean Average Precision | Mede precisão média ao longo do ranking |

**K padrão:** 5, 10, 20

### 2.2 Métricas de Classificação (H2, H3)

| Métrica | Definição | Por que usar |
|---------|-----------|-------------|
| **Macro-F1** | Média de F1 por classe (ponderação igualitária) | Sensível a classes raras |
| **Micro-F1** | F1 global (ponderação por volume) | Visão geral do desempenho |
| **Precision por classe** | Especialmente para classes críticas | Mede false positive burden |
| **Recall por classe** | Especialmente para classes raras | Mede cobertura |
| **AUC-ROC** | Separabilidade entre classes | Robustez ao threshold |
| **Calibration Error (ECE)** | Erro esperado de calibração | Confiabilidade dos scores |

### 2.3 Métricas de Regras (H3)

| Métrica | Definição |
|---------|-----------|
| **Precision da regra** | Dos casos que a regra acionou, quantos são verdadeiros |
| **Recall da regra** | Dos casos reais, quantos a regra capturou |
| **False Positive Burden** | Quantos falsos positivos cada regra gera por 1000 conversas |
| **Cobertura** | % das conversas em que pelo menos uma regra produziu evidência |
| **Latência por regra** | Tempo de avaliação da regra (ms) |

### 2.4 Métricas de Eficiência (H4)

| Métrica | Definição |
|---------|-----------|
| **Custo por conversa** | Tempo total de processamento (ms CPU/GPU) |
| **Throughput** | Conversas processadas por segundo |
| **Latência p50/p95/p99** | Distribuição de latência por conversa |
| **% resolvido por estágio** | Quantas conversas terminam em cada estágio da cascata |
| **Δ F1** | Diferença de F1 entre pipeline uniforme e cascata |

---

## 3. Experimento H1 — Superioridade do Retrieval Híbrido

### 3.1 Hipótese

> O retrieval híbrido (BM25 + ANN com fusão de scores) supera tanto BM25 isolado quanto busca semântica isolada em Recall@K, MRR e nDCG quando aplicado a conversas de call center.

### 3.2 Configuração

#### Sistemas comparados

| Sistema | Descrição |
|---------|-----------|
| **BM25-base** | BM25 vanilla (lowercase + stopwords) |
| **BM25-norm** | BM25 com normalização accent-aware (strip_accents + punctuation removal) |
| **ANN-E5** | Busca semântica com E5-base embeddings |
| **ANN-BGE** | Busca semântica com BGE-small embeddings |
| **Hybrid-linear** | BM25-norm + melhor ANN, fusão linear (Score = α·sem + (1-α)·lex) |
| **Hybrid-RRF** | BM25-norm + melhor ANN, Reciprocal Rank Fusion |
| **Hybrid-rerank** | Hybrid-linear + cross-encoder reranking no top-50 |

#### Parâmetros a variar

| Parâmetro | Valores |
|-----------|---------|
| α (peso semântico) | 0.3, 0.5, 0.65, 0.7, 0.8 |
| K (top-K por estágio) | 10, 20, 50, 100 |
| BM25 k₁ | 1.2, 1.5, 2.0 |
| BM25 b | 0.5, 0.75, 1.0 |

### 3.3 Construção do Ground Truth

Para avaliar retrieval, precisamos de queries com documentos relevantes anotados:

1. **Queries:** intents/motivos de contato da taxonomia (ex: "cancelamento", "reclamação de cobrança")
2. **Documentos:** turnos ou janelas de contexto do corpus
3. **Relevância:** anotada por classe — turno/janela pertence à classe do intent da query
4. Alternativa: usar conversas anotadas, tratar cada conversa como "documento" e o intent como "query"

### 3.4 Protocolo

```
Para cada sistema S ∈ {BM25-base, BM25-norm, ANN-E5, ANN-BGE, Hybrid-linear, Hybrid-RRF, Hybrid-rerank}:
  Para cada query Q do conjunto de avaliação:
    1. Executar retrieval com S, obter top-K resultados
    2. Calcular Recall@K, Precision@K, MRR, nDCG@K, MAP@K
  3. Agregar métricas (média ± desvio padrão)
  4. Repetir com 5 seeds diferentes para estimar variância

Para Hybrid-linear:
  Variar α ∈ {0.3, 0.5, 0.65, 0.7, 0.8}
  Plotar curva α vs Recall@10 para determinar melhor ponto
```

### 3.5 Análise Esperada

- Tabela principal: sistema × métricas (como Table 2 de Rayo et al.)
- Gráfico: Recall@K para K = 5, 10, 20 por sistema
- Gráfico: α vs Recall@10 para o sistema híbrido
- Análise qualitativa: exemplos de queries onde híbrido acerta e isolados erram
- Teste estatístico: paired t-test ou Wilcoxon signed-rank entre melhor híbrido e melhor isolado

### 3.6 Critério de Confirmação de H1

H1 é **confirmada** se:
- O melhor sistema híbrido supera **todos** os sistemas isolados em Recall@10 e MAP@10
- A diferença é estatisticamente significativa (p < 0.05)

H1 é **parcialmente confirmada** se:
- Híbrido supera em algumas métricas mas não em todas

H1 é **refutada** se:
- Um sistema isolado (BM25 ou ANN) supera ou empata com o melhor híbrido

---

## 4. Experimento H2 — Ganho da Representação Multi-Nível

### 4.1 Hipótese

> Classificadores que utilizam features em múltiplos níveis de granularidade (turno + janela de contexto + conversa) alcançam F1 superior a classificadores que operam em um único nível.

### 4.2 Configuração

#### Representações comparadas

| Config | Features de entrada |
|--------|-------------------|
| **Turn-only** | Embedding do turno individual |
| **Window-only** | Embedding da janela de contexto (5 turnos) |
| **Conv-only** | Embedding da conversa completa |
| **Turn+Window** | Concatenação de embedding do turno + janela |
| **Turn+Conv** | Concatenação de embedding do turno + conversa |
| **Multi-level** | Concatenação de embedding turno + janela + conversa |
| **Multi-level+lex** | Multi-level + features lexicais (TF-IDF, BM25 scores contra protótipos) |
| **Multi-level+struct** | Multi-level + features estruturais (speaker, position, duration) |
| **Full** | Multi-level + lexical + structural + contextual |

#### Classificadores

Para cada representação, treinar e avaliar:

| Classificador | Justificativa |
|--------------|---------------|
| **Logistic Regression** | Baseline linear, interpretável |
| **LightGBM** | Melhor para features heterogêneas |
| **MLP (2 camadas)** | Baseline neural para features densas |

#### Parâmetros de janela

| Parâmetro | Valores |
|-----------|---------|
| Window size (turnos) | 3, 5, 7, 10 |
| Stride | 1, 2, 3 |
| Pooling | mean, attention-weighted |

### 4.3 Protocolo

```
Para cada representação R ∈ {Turn-only, Window-only, ..., Full}:
  Para cada classificador C ∈ {LogReg, LightGBM, MLP}:
    1. Gerar features para train/val/test
    2. Treinar C com features R no train set
    3. Selecionar hiperparâmetros via val set (grid search ou Optuna)
    4. Avaliar no test set: Macro-F1, Micro-F1, Precision/Recall por classe, AUC
    5. Repetir com 5-fold cross-validation para estimar variância

Para window size:
  Fixar melhor classificador e melhor representação multi-nível
  Variar window_size ∈ {3, 5, 7, 10}
  Plotar window_size vs Macro-F1

Para pooling:
  Fixar melhor configuração
  Comparar mean pooling vs attention pooling
```

### 4.4 Análise Esperada

- Tabela principal: representação × classificador × Macro-F1 (como Table 2 de Huang & He)
- Heatmap: F1 por classe × representação (quais classes se beneficiam de contexto)
- Gráfico: window_size vs Macro-F1
- Gráfico: mean vs attention pooling por classe
- Análise por classe: intents que requerem contexto multi-turn (ex: "objeção após oferta de retenção") vs intents detectáveis por turno isolado (ex: "solicitar segunda via")

### 4.5 Critério de Confirmação de H2

H2 é **confirmada** se:
- A melhor configuração multi-nível supera **todas** as configurações single-level em Macro-F1
- Diferença estatisticamente significativa (p < 0.05)
- O ganho é observável em pelo menos 60% das classes

H2 é **parcialmente confirmada** se:
- Multi-nível melhora apenas algumas classes específicas

H2 é **refutada** se:
- Turn-only ou Conv-only supera multi-nível

---

## 5. Experimento H3 — Complementaridade das Regras Determinísticas

### 5.1 Hipótese

> A adição de um motor de regras semânticas (DSL → AST) ao pipeline híbrido melhora a precision em classes críticas sem degradar o recall global, além de fornecer rastreabilidade de evidência por decisão.

### 5.2 Configuração

#### Definição de "classes críticas"

Selecionar 3-5 classes que representem cenários críticos de negócio:
- **Cancelamento/churn** — intenção de cancelar serviço
- **Compliance** — menção a órgãos reguladores, ouvidoria, processo judicial
- **Fraude** — padrões suspeitos de engenharia social
- **Insatisfação grave** — escalação, ameaça, reclamação formal

#### Configurações comparadas

| Config | Descrição |
|--------|-----------|
| **ML-only** | Melhor classificador do Exp. H2 (sem regras) |
| **Rules-lexical** | Apenas regras com predicados lexicais (contains, regex) |
| **Rules-full** | Regras com predicados lexicais + semânticos + contextuais |
| **ML+Rules-override** | ML-only + regras como override (regra prevalece quando aciona) |
| **ML+Rules-feature** | ML-only + flags de regras como features adicionais do classificador |
| **ML+Rules-postproc** | ML-only + regras como pós-processamento (ajuste de scores) |

#### Conjunto de regras

Para cada classe crítica, definir 3-5 regras no DSL. Exemplo para cancelamento:

```dsl
RULE cancelamento_explicito
WHEN
    speaker == "customer"
    AND lexical.contains_any(["cancelar", "encerrar", "desistir", "rescindir"])
    AND NOT lexical.excludes_any(["teste", "simulação"])
THEN
    tag("cancelamento")
    score(0.90)

RULE cancelamento_implicito
WHEN
    speaker == "customer"
    AND semantic.intent_score("cancelamento") > 0.80
    AND context.repeated_in_window("insatisfação", 3) >= 2
THEN
    tag("cancelamento")
    score(0.85)

RULE compliance_ouvidoria
WHEN
    lexical.contains_any(["ouvidoria", "procon", "anatel", "bacen", "reclame aqui"])
    OR lexical.regex("processo\\s+\\d+")
THEN
    tag("compliance_risco")
    score(0.95)
    priority("high")
```

### 5.3 Protocolo

```
1. Definir regras para cada classe crítica (3-5 regras por classe)
2. Para cada configuração C:
   a. Executar classificação no test set
   b. Para classes críticas: calcular Precision, Recall, F1
   c. Para todas as classes: calcular Macro-F1, Micro-F1
   d. Registrar % de decisões com evidência rastreável
   e. Registrar latência adicional por regra

3. Análise qualitativa:
   a. Selecionar 50 casos onde ML e regras discordam
   b. Avaliar manualmente: quem acerta?
   c. Categorizar tipos de erro corrigidos pelas regras
   d. Categorizar tipos de erro introduzidos pelas regras
```

### 5.4 Análise Esperada

- Tabela principal: configuração × Precision/Recall/F1 nas classes críticas
- Tabela secundária: Macro-F1 global com e sem regras (verificar que regras não degradam)
- Gráfico: Precision vs Recall trade-off por classe crítica, comparando ML-only vs ML+Rules
- Exemplos qualitativos (3-5 por classe):
  - Caso que ML errou e regra acertou (ex: paráfrase que o classificador não capturou, mas contains_any capturou)
  - Caso que regra errou e ML acertou (ex: "cancelar o download" não é intenção de cancelamento)
  - Caso que ambos acertaram com evidências complementares
- Análise de evidência: exemplo de output com metadata rastreável

### 5.5 Critério de Confirmação de H3

H3 é **confirmada** se:
- Precision nas classes críticas melhora com regras (qualquer configuração ML+Rules > ML-only)
- Recall global não degrada mais de 1 ponto percentual
- ≥80% das decisões em classes críticas produzem evidência rastreável

H3 é **parcialmente confirmada** se:
- Precision melhora em algumas classes mas degrada em outras
- Ou: melhora em precision mas com custo de latência inaceitável

H3 é **refutada** se:
- Regras não melhoram precision ou degradam recall significativamente

---

## 6. Experimento H4 — Eficiência da Inferência em Cascata

### 6.1 Hipótese

> Um pipeline com inferência cascateada reduz o custo computacional médio por conversa em pelo menos 40% comparado ao pipeline uniforme, com degradação de qualidade inferior a 2% em F1.

### 6.2 Configuração

#### Pipeline uniforme (baseline)

Todas as conversas passam por todos os estágios:
```
Conversa → Normalização → Embeddings → BM25 + ANN → Fusão → Classificação → Regras → Output
```

#### Pipeline cascateado

```
Estágio 1 — Filtros baratos (custo: ~1ms)
  └── Metadata: canal, fila, idioma, duração
  └── Regras lexicais simples (contains, regex)
  └── Se confiança ≥ threshold_1 → OUTPUT (não avança)

Estágio 2 — Retrieval híbrido (custo: ~10-50ms)
  └── BM25 + ANN → fusão de scores
  └── Classificador leve (Logistic Regression) sobre scores de retrieval
  └── Se confiança ≥ threshold_2 → OUTPUT

Estágio 3 — Classificação completa (custo: ~50-200ms)
  └── Embeddings multi-nível + features heterogêneas
  └── Classificador completo (LightGBM/MLP)
  └── Regras semânticas completas
  └── Se confiança ≥ threshold_3 → OUTPUT

Estágio 4 — Revisão excepcional (custo: ~500ms-2s)
  └── Cross-encoder reranking
  └── Classificação com features adicionais
  └── Flag para revisão humana se confiança < threshold_4
```

#### Parâmetros da cascata

| Parâmetro | Valores a testar |
|-----------|-----------------|
| threshold_1 (confiança estágio 1) | 0.90, 0.95, 0.98 |
| threshold_2 (confiança estágio 2) | 0.85, 0.90, 0.95 |
| threshold_3 (confiança estágio 3) | 0.80, 0.85, 0.90 |

### 6.3 Protocolo

```
1. Instrumentar cada estágio com medição de tempo (ms)
2. Executar pipeline UNIFORME no test set completo:
   a. Registrar: classificação final, tempo total, tempo por estágio
   b. Calcular: Macro-F1, custo médio por conversa

3. Para cada configuração de thresholds T:
   a. Executar pipeline CASCATEADO no test set completo:
      - Registrar: em qual estágio cada conversa foi resolvida
      - Registrar: classificação final, tempo total
   b. Calcular: Macro-F1, custo médio, % resolvido por estágio

4. Comparar:
   a. Δ custo = (custo_uniforme - custo_cascata) / custo_uniforme × 100
   b. Δ F1 = F1_uniforme - F1_cascata
   c. Plotar curva de Pareto: Δ custo vs Δ F1

5. Análise por estágio:
   a. Que tipos de conversa são resolvidos no estágio 1? (fáceis, óbvias)
   b. Que tipos precisam do estágio 3? (ambíguas, complexas)
   c. Que tipos são flagados para revisão? (novos intents, edge cases)
```

### 6.4 Análise Esperada

- Tabela: configuração × custo médio × Macro-F1 × % resolvido por estágio
- Gráfico: curva de Pareto (redução de custo vs degradação de F1)
- Gráfico de barras empilhadas: % de conversas resolvidas em cada estágio
- Gráfico: distribuição de confiança por estágio (histograma)
- Análise: perfil das conversas "fáceis" (estágio 1) vs "difíceis" (estágio 3-4)
  - Comprimento médio, número de turnos, variabilidade de intents

### 6.5 Critério de Confirmação de H4

H4 é **confirmada** se:
- Existe pelo menos uma configuração de thresholds onde:
  - Redução de custo ≥ 40%
  - Degradação de F1 < 2 pontos percentuais

H4 é **parcialmente confirmada** se:
- Redução de custo ≥ 40% mas degradação > 2%, ou
- Degradação < 2% mas redução < 40%

H4 é **refutada** se:
- Nenhuma configuração atinge ambos os critérios simultaneamente

---

## 7. Análise Estatística

### 7.1 Testes de Significância

Para todas as comparações entre sistemas:

| Comparação | Teste Recomendado |
|-----------|-------------------|
| 2 sistemas, métricas pareadas | Wilcoxon signed-rank test |
| Múltiplos sistemas | Friedman test + Nemenyi post-hoc |
| Confiança da diferença | Bootstrap confidence intervals (95%) |

**Nível de significância:** α = 0.05

### 7.2 Variância e Reprodutibilidade

- **5-fold cross-validation** para todos os experimentos de classificação
- **5 seeds diferentes** para experimentos de retrieval
- Reportar **média ± desvio padrão** em todas as tabelas
- Todos os hiperparâmetros, seeds e configurações documentados para reprodução

### 7.3 Ablation Studies

Para o sistema completo (Full pipeline), remover um componente de cada vez e medir impacto:

| Ablação | O que se remove | O que se espera |
|---------|----------------|-----------------|
| -BM25 | Remove componente lexical do retrieval | Queda em queries com termos exatos |
| -ANN | Remove componente semântico do retrieval | Queda em queries com paráfrases |
| -Rules | Remove motor de regras | Queda em precision de classes críticas |
| -Window | Remove features de janela de contexto | Queda em intents contextuais |
| -Struct | Remove features estruturais | Impacto menor, validar |
| -Accent-norm | Remove normalização de diacríticos | Queda em recall para termos PT-BR |

---

## 8. Ameaças à Validade

### 8.1 Validade Interna

| Ameaça | Mitigação |
|--------|-----------|
| Overfitting nos hiperparâmetros | Separação rigorosa train/val/test; cross-validation |
| Viés na construção de regras | Regras definidas antes de ver resultados no test set |
| Escolha de métricas favorável | Reportar múltiplas métricas; incluir métricas onde o sistema perde |
| Bugs no pipeline | Testes unitários extensivos (1822+ testes no TalkEx) |

### 8.2 Validade Externa

| Ameaça | Mitigação |
|--------|-----------|
| Dataset não representativo | Disclosure explícito das limitações do dataset |
| Generalização para outros idiomas | Testar com e sem normalização de diacríticos |
| Generalização para outros domínios | Discussão explícita; trabalhos futuros |
| Escala (volumes de produção) | Análise de complexidade computacional teórica + medições empíricas |

### 8.3 Validade de Construto

| Ameaça | Mitigação |
|--------|-----------|
| Métricas não capturam utilidade real | Incluir análise qualitativa com exemplos |
| Explicabilidade não avaliada formalmente | Definir critérios de qualidade da evidência |
| "Custo" simplificado (apenas tempo) | Discutir custo de GPU/memória quando relevante |

---

## 9. Infraestrutura Experimental

### 9.1 Hardware

Documentar hardware utilizado para reprodutibilidade:
- CPU: modelo, cores
- GPU: modelo, VRAM (se usado para embeddings)
- RAM: quantidade
- Storage: SSD/HDD

### 9.2 Software

| Componente | Tecnologia |
|-----------|-----------|
| Linguagem | Python 3.11+ |
| Embeddings | sentence-transformers (E5, BGE) |
| BM25 | rank-bm25 ou implementação própria (bm25.py) |
| ANN | FAISS (faiss-cpu) |
| Classificação | scikit-learn, LightGBM |
| Rule Engine | TalkEx DSL/AST (implementação própria) |
| Avaliação | scikit-learn metrics, custom evaluation scripts |
| Visualização | matplotlib, seaborn |
| Reprodutibilidade | seeds fixos, requirements.txt versionado |

### 9.3 Código e Dados

- Código: repositório TalkEx (GitHub, quando publicado)
- Dados: referência ao dataset público ou procedimento de construção
- Modelos treinados: armazenados e versionados
- Notebooks experimentais: `experiments/` com scripts reprodutíveis

---

## 10. Cronograma Experimental

| Fase | Atividade | Duração Estimada |
|------|-----------|-----------------|
| 1 | Preparação do dataset (coleta, anotação, splits) | 3-4 semanas |
| 2 | Exp. H1: Retrieval híbrido | 2-3 semanas |
| 3 | Exp. H2: Representação multi-nível | 3-4 semanas |
| 4 | Exp. H3: Regras determinísticas | 2-3 semanas |
| 5 | Exp. H4: Inferência em cascata | 2-3 semanas |
| 6 | Ablation studies e análise estatística | 2 semanas |
| 7 | Análise qualitativa e redação de resultados | 3-4 semanas |
| **Total** | | **17-23 semanas** |

**Nota:** Fases 2-5 podem ter sobreposição se o pipeline for implementado incrementalmente (cada experimento adiciona ao anterior).

---

## 11. Checklist de Reprodutibilidade

Antes de publicar resultados, verificar:

- [ ] Seeds fixos e documentados para todos os splits e treinamentos
- [ ] Hiperparâmetros documentados para todos os modelos
- [ ] Hardware e versões de software documentados
- [ ] Dataset acessível (público) ou procedimento de construção documentado
- [ ] Scripts de avaliação versionados no repositório
- [ ] Resultados reportados com média ± desvio padrão
- [ ] Testes estatísticos aplicados para todas as comparações centrais
- [ ] Análise qualitativa com exemplos concretos
- [ ] Ameaças à validade discutidas explicitamente
- [ ] Código do TalkEx com testes passando (quality gates verdes)
