# Etapa 0 — Research Assessment

**Autor:** Dr. Rafael A. Monteiro (PhD, NLP & Information Retrieval)
**Data:** 2026-03-12
**Objetivo:** Diagnóstico rigoroso do estado atual do trabalho antes da escrita da dissertação.

> Este documento avalia objetivamente o que existe, o que falta, o que é defensável
> perante uma banca do MIT Media Lab, e o que precisa ser corrigido antes da defesa.

---

## 1. Diagnóstico geral

O projeto TalkEx apresenta uma **implementação técnica substancial** (97 módulos, 15.773 LOC, 1.884 testes) e um **pipeline experimental funcional** (5 hipóteses testadas com testes estatísticos). No entanto, a **evidência empírica é insuficiente para sustentar as claims originais** da dissertação, por razões detalhadas a seguir.

**Estado resumido:**

| Dimensão | Estado | Avaliação |
|---|---|---|
| Arquitetura de software | Completa | Defensável |
| Pipeline experimental | Funcional | Defensável com ressalvas |
| Dataset | Problemático | **Não defensável como está** |
| Protocolo de avaliação | Incompleto | **Requer extensão** |
| Resultados empíricos | Preliminares | Defensável se posicionados corretamente |
| Tese central | Overclaiming | **Requer reformulação** |

---

## 2. O que é sólido e defensável

### 2.1 Arquitetura de software

O TalkEx é um artefato técnico maduro:

- **97 módulos de produção** com 15.773 LOC
- **1.884 testes unitários** com ratio teste/código de 1.6:1
- **Pipeline completo:** ingestion → segmentation → context windows → embeddings → retrieval (BM25 + ANN + fusão) → classification (3 modelos) → rule engine (DSL → AST → avaliação com evidência) → analytics
- **Decisões arquiteturais documentadas** em 4 ADRs
- **Módelos Pydantic rigorosos** (frozen, strict, validados)
- **CLI funcional** com orquestração de pipeline

**Avaliação:** A arquitetura é a contribuição mais sólida do trabalho. O design em cascata com separação clara de estágios, a escolha de encoder congelado, e o motor de regras com evidência rastreável são contribuições genuínas de engenharia.

### 2.2 Protocolo de auditoria de dados

O processo de auditoria do dataset é exemplar:

- Deduplicação em dois níveis (0.92 flag / 0.97 hard)
- Detecção de leakage few-shot com splitting contamination-aware
- Categorização A/B/C da classe "outros" com revisão humana
- Traceabilidade completa (cada decisão documentada em JSON)
- Validação humana com taxa de confirmação ≥96.7%

**Avaliação:** Este protocolo é publicável independentemente. Demonstra rigor metodológico raro em trabalhos de ML aplicado.

### 2.3 Resultados parcialmente defensáveis

Os resultados pós-auditoria sobre o dataset limpo (2.122 registros, 8 intents):

| Hipótese | Resultado | Defensável? |
|---|---|---|
| H1 — Retrieval híbrido | MRR=0.853 vs BM25 0.835, p=0.017 | Sim, com escopo limitado |
| H2 — Embeddings + léxico | Macro-F1=0.722 vs 0.334 lexical-only | Sim, resultado forte |
| H3 — Regras + ML | +1.8pp, p=0.131 | Parcial — direção positiva mas não significativa |
| H4 — Cascata | Cascata custa mais | Sim — resultado negativo honesto |
| Ablação — Embeddings | +33.0pp | Sim, contribuição dominante |

---

## 3. Problemas críticos (Threats to Validity)

### 3.1 CRÍTICO — Dataset sintético sem validação externa

**Fato:** 60.1% do dataset (1.275 de 2.122 registros) é gerado sinteticamente por LLM.

**Implicações científicas:**
- Os modelos podem estar aprendendo padrões da distribuição do LLM gerador, não padrões reais de conversação
- A performance reportada (Macro-F1=0.722) pode não transferir para dados reais
- Não há como saber se a melhoria do retrieval híbrido (H1) se mantém em dados naturais
- O few-shot prompting usado para gerar os dados sintéticos introduz dependências estruturais que o splitting contamination-aware mitiga mas não elimina completamente

**O que uma banca do MIT perguntaria:**
> "How do you know your classifier isn't just learning the LLM's generation patterns rather than real conversational intent signals?"

**Status:** Não temos resposta para essa pergunta. Não existe validação em dados reais independentes.

### 3.2 CRÍTICO — Domínio único sem avaliação de generalização

**Fato:** Todo o dataset vem de um único cenário (atendimento ao cliente brasileiro). Não existe avaliação cross-domain.

**Implicações científicas:**
- A tese original afirma que a arquitetura é "portável entre cenários" — mas não há evidência empírica disso
- O posicionamento "frozen encoder vs fine-tuning" requer comparação em múltiplos domínios para ser defensável
- Sem LODO (Leave-One-Domain-Out), qualquer claim sobre generalização é especulativa

**O que uma banca do MIT perguntaria:**
> "You claim portability across domains, but you tested on exactly one domain. How is this different from any other single-domain classifier?"

**Status:** Não temos resposta para essa pergunta. LODO não está implementado.

### 3.3 IMPORTANTE — Ausência de baseline fine-tuned

**Fato:** A dissertação propõe "encoder congelado" como estratégia, mas nunca compara contra encoder fine-tuned no mesmo dataset.

**Implicações científicas:**
- Não podemos afirmar que frozen é "competitivo" sem saber o que fine-tuning alcançaria
- Um reviewer perguntará: "What's the ceiling? How much are you leaving on the table?"
- A ablação mostra que embeddings são o componente dominante (+33.0pp) — o que um encoder fine-tuned faria?

**O que uma banca do MIT perguntaria:**
> "You chose frozen encoders, but you never measured what you lose compared to fine-tuning. How can you justify this choice without a comparison?"

**Status:** Não temos esse baseline.

### 3.4 IMPORTANTE — Motor de regras sub-explorado

**Fato:** O motor de regras tem 11 módulos e 3.599 LOC de implementação, mas os experimentos usam apenas **2 regras lexicais simples** (cancelamento e reclamação por keywords).

**Implicações científicas:**
- A arquitetura de regras é sofisticada (DSL → AST, 4 tipos de predicado, evidência rastreável), mas os experimentos não exercitam essa sofisticação
- Os predicados SEMANTIC, STRUCTURAL e CONTEXTUAL existem no código mas não são usados nos experimentos
- H3 testa "regras + ML" com apenas 2 regras keyword — a conclusão "regras são inconclusivas" pode ser artefato da pobreza do ruleset, não da abordagem
- O efeito das regras (+1.8pp, p=0.131) pode ser substancialmente diferente com um ruleset adequado

**O que uma banca do MIT perguntaria:**
> "You have a sophisticated rule engine with semantic and contextual predicates, but you only tested two keyword rules. How is this a fair test of the architecture?"

**Status:** O ruleset experimental é insuficiente para avaliar a contribuição do motor de regras.

### 3.5 MODERADO — Determinismo artificial nos multi-seed

**Fato:** LightGBM produz resultados idênticos (std=0.000) em todos os 5 seeds porque os splits são fixos e os embeddings são determinísticos.

**Implicações científicas:**
- A variação reportada (std=0.000) não captura incerteza real
- Multi-seed só varia o random_state do classificador, que não afeta LightGBM com esses dados
- A robustez estatística aparente é artificial — um único run produz o mesmo resultado

**Mitigação possível:** Cross-validation (k-fold) em vez de splits fixos. Ou bootstrap sobre o test set para estimar intervalos de confiança.

### 3.6 MODERADO — F1 por classe altamente variável

**Fato (melhor modelo, lexical+emb LightGBM):**

| Classe | F1 |
|---|---|
| cancelamento | 0.909 |
| elogio | 0.873 |
| suporte_tecnico | 0.844 |
| reclamacao | 0.800 |
| duvida_servico | 0.750 |
| duvida_produto | 0.677 |
| saudacao | 0.478 |
| compra | 0.442 |

**Implicações:**
- Macro-F1=0.722 mascara que 2 classes (saudacao, compra) têm desempenho abaixo de 0.50
- Em produção, ~25% das classes seriam classificadas com qualidade insuficiente
- Isso pode ser artefato da natureza sintética dos dados nessas classes

### 3.7 MODERADO — Ausência de métricas de calibração

**Fato:** Não são reportados: Brier score, Expected Calibration Error (ECE), reliability diagrams, curvas de calibração.

**Implicações:**
- Macro-F1 mede acurácia de decisão mas não mede confiabilidade das probabilidades
- Para o mecanismo de abstenção proposto, calibração é fundamental
- Sem calibração, não é possível definir thresholds de confiança de forma principled

---

## 4. Reformulação necessária da tese

### 4.1 Tese original (não defensável como está)

> "Uma arquitetura híbrida em cascata [...] alcança qualidade de classificação e retrieval
> **superior** às abordagens que dependem de qualquer paradigma isolado — ao mesmo tempo em
> que preserva explicabilidade operacional e eficiência computacional."

**Problemas:**
- "Superior" é overclaiming — H3 é inconclusiva, H4 é refutada
- "Eficiência computacional" é contradito pelos dados — cascata custa mais
- Não há evidência de generalização para além do dataset testado
- Dataset sintético limita a força de qualquer claim

### 4.2 Tese reformulada (defensável)

> "Este trabalho investiga se uma arquitetura híbrida que combina retrieval lexical,
> representações semânticas de encoder congelado e regras determinísticas auditáveis
> pode oferecer classificação competitiva de intenções conversacionais com
> transparência operacional. Os resultados experimentais em um corpus de atendimento
> ao cliente em português brasileiro indicam que representações semânticas são o
> componente dominante (+33pp Macro-F1), que a fusão léxico-semântica em retrieval
> oferece ganhos modestos mas estatisticamente significativos (+1.8pp MRR, p=0.017),
> e que a integração de regras determinísticas apresenta direção positiva mas
> evidência inconclusiva. A inferência em cascata, conforme implementada, não
> reduziu custo computacional. Os resultados são limitados a um único domínio com
> dados parcialmente sintéticos e requerem validação em cenários adicionais."

**Diferenças fundamentais:**
- "Investiga se" em vez de "alcança"
- "Indicam que" em vez de "demonstra que"
- Resultados negativos explícitos (H4 refutada)
- Limitações na própria tese, não escondidas em seção secundária
- Sem claim de generalização sem evidência

---

## 5. Opções estratégicas para a dissertação

### Opção A — Dissertação exploratória (menor risco, defensável agora)

**Posicionamento:** Estudo exploratório de uma arquitetura híbrida com resultados preliminares em um domínio.

**Estrutura:**
1. Introdução — problema e motivação
2. Trabalhos relacionados — posicionamento na literatura
3. Arquitetura TalkEx — contribuição técnica principal
4. Protocolo experimental — metodologia com transparência total
5. Resultados e análise — dados reais sem overclaiming
6. Discussão — limitações explícitas, threats to validity, trabalho futuro
7. Conclusão — contribuições modestas e honestas

**Vantagem:** Pode ser defendida com os dados atuais.
**Risco:** Banca pode considerar escopo insuficiente para SM thesis do MIT.

### Opção B — Dissertação com extensão experimental (maior impacto, requer trabalho adicional)

**Trabalho adicional necessário:**
1. **Adicionar 1-2 datasets reais** (mesmo que pequenos) para validação externa
2. **Implementar baseline fine-tuned** (mesmo modelo MiniLM com fine-tuning no dataset)
3. **Expandir ruleset** para cobrir mais classes e usar predicados semânticos/contextuais
4. **Adicionar cross-validation** (5-fold stratified) para intervalos de confiança reais
5. **Implementar métricas de calibração** (Brier, ECE, reliability diagram)

**Vantagem:** Evidência muito mais forte. Publicável em workshop ACL/EMNLP.
**Risco:** Requer tempo adicional significativo.

### Opção C — Reposicionar como contribuição arquitetural (pragmática)

**Posicionamento:** A contribuição é a arquitetura e o framework de avaliação, não os números. Os experimentos são demonstrativos, não conclusivos.

**Estrutura:**
1. Introdução — o problema de construir sistemas de NLP conversacional auditáveis
2. Trabalhos relacionados
3. Arquitetura TalkEx — contribuição principal (detalhada)
4. Protocolo de auditoria de dados — contribuição secundária
5. Avaliação demonstrativa — resultados como evidência direcional
6. Discussão e trabalho futuro
7. Conclusão

**Vantagem:** Não depende de claims empíricas fortes.
**Risco:** Pode parecer trabalho de engenharia sem contribuição científica suficiente.

---

## 6. Recomendação

**Para uma SM thesis do MIT Media Lab, recomendo a Opção A com elementos da Opção B.**

Justificativa:

1. O Media Lab valoriza **artefatos técnicos funcionais** — o TalkEx é sólido nessa dimensão
2. Uma dissertação que reporta **resultados negativos com honestidade** (H4 refutada, H3 inconclusiva) é mais respeitada do que uma que força narrativas positivas
3. O protocolo de auditoria de dados é uma **contribuição metodológica genuína** que pode ser destacada
4. A ausência de fine-tuning comparison e cross-domain evaluation são **gaps identificados mas não necessariamente fatais** se a dissertação os reconhece explicitamente como limitações e trabalho futuro

**Itens mínimos recomendados antes da escrita:**
- [ ] Implementar 5-fold cross-validation para intervalos de confiança reais
- [ ] Adicionar Brier score e ECE ao pipeline de métricas
- [ ] Expandir para pelo menos 5-8 regras cobrindo mais classes
- [ ] Redigir seção explícita de Threats to Validity

**Itens ideais (se o prazo permitir):**
- [ ] Obter pelo menos 1 dataset adicional (mesmo que pequeno) para validação externa
- [ ] Implementar baseline fine-tuned para comparação direta
- [ ] Implementar mecanismo de abstenção com threshold calibrado

---

## 7. Próxima etapa

Após sua decisão sobre a estratégia (A, B ou C), prossigo para a **Etapa 1 — Thesis Statement & Scope**, onde formalizamos:

- Tese central definitiva
- Hipóteses operacionais (reformuladas se necessário)
- Delimitação explícita do escopo
- Contribuições esperadas

---

*Este documento reflete exclusivamente dados reais do projeto. Nenhum dado foi inventado ou omitido. Resultados negativos (H3, H4) e limitações (dataset sintético, domínio único) foram apresentados com a mesma proeminência que resultados positivos (H1, H2).*
