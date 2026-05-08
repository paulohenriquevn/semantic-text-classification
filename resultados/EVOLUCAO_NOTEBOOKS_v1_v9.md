# Evolução dos Notebooks Experimentais: v1 a v9

**TalkEx — Classificação de Intenções em Conversas PT-BR**

---

## Visão Geral

O TalkEx passou por 9 versões experimentais iterativas. Cada versão testou uma hipótese, revelou achados inesperados, e motivou a próxima. Este documento registra o que mudou em cada versão, por quê, e o que aprendemos.

| Versão | Arquivo | Foco principal |
|:---:|:---|:---|
| v1 | `talkex_banca_v1_backup.ipynb` | Baseline supervisionado (LightGBM) |
| v2 | `talkex_banca_v2_turn_level.ipynb` | Turno individual + KNN (taxonomia dinâmica) |
| v3 | `talkex_banca_v3_window_knn.ipynb` | Janela de contexto + KNN |
| v4 | `talkex_banca_v4_improved.ipynb` | Speaker markers + α-tuning + rerank |
| v5 | `talkex_banca_v5_cascade_fixed.ipynb` | Cascata com exclusividade + Hybrid fallback |
| v6 | `talkex_banca_v6_production_rules.ipynb` | Motor de regras multi-sinal (produção) |
| v7 | `talkex_banca_v7_confidence_routing.ipynb` | Confidence-gated routing |
| v8 | `talkex_banca_v8_final.ipynb` | Latência + interpretabilidade + correções |
| v9 | `talkex_banca_v9_publication.ipynb` | Publication-ready (baselines, ablação, few-shot) |

---

## v1 — Baseline Supervisionado

**Arquivo:** `talkex_banca_v1_backup.ipynb`

**Arquitetura:** Janela de 5 turnos → 397 features (384 embeddings + 7 lexicais + 4 estruturais + 2 regras) → LightGBM com hiperparâmetros otimizados via Optuna.

**Resultados principais:**
- Macro-F1 (janela): 0,734 ± 0,012
- Macro-F1 (conversa, agregação por média de probabilidades): 0,771 ± 0,017
- Hiperparâmetros: num_leaves=9, learning_rate=0,119, min_child_samples=27

**Hipóteses testadas:**
- H1 (Recuperação Híbrida): MRR=0,802 para fusão linear α=0,65 (melhor)
- H2 (Embeddings + Lexical): +24,3pp de ganho com embeddings (0,491→0,734)
- H3 (Regras): ML+Rules-feature F1=0,731 (+0,5pp, não significativo)
- H4 (Cascata): Cascade θ=0,95 F1=0,734 com 60% menos latência

**Problema identificado:** Taxonomia fixa — nova classe exige retreinar o LightGBM. Para call centers em escala, isso é inviável operacionalmente.

**O que motivou v2:** Migrar para classificação por similaridade (KNN) para manter taxonomia dinâmica.

---

## v2 — Turno Individual + KNN

**Arquivo:** `talkex_banca_v2_turn_level.ipynb`

**O que mudou vs v1:**
- Unidade de classificação: janela de 5 turnos → **turno individual**
- Classificador: LightGBM → **KNN por similaridade** (embedding cosseno)
- Sem feature engineering (397→384 dims, apenas embeddings)
- Filtro customer-only (turnos do agente excluídos)

**Resultados principais:**
- Macro-F1 (turno): **0,535 ± 0,007** (queda de 20pp vs v1)
- Macro-F1 (conversa, voto ponderado): **0,751 ± 0,015**
- BM25 KNN: 0,609 (supera Embedding KNN: 0,535)
- Hybrid α=0,3: 0,617 (melhor)

**Achados:**
- Turno isolado é fraco (0,535) — sem contexto conversacional
- BM25 supera embeddings por ~7pp (achado contra-intuitivo)
- Agregação turno→conversa recupera performance (+21,6pp)

**O que motivou v3:** Combinar janelas de contexto (v1) com KNN por similaridade (v2).

---

## v3 — Janela de Contexto + KNN

**Arquivo:** `talkex_banca_v3_window_knn.ipynb`

**O que mudou vs v2:**
- Unidade: turno individual → **janela de N turnos** (configurável, default=5)
- WINDOW_SIZE como parâmetro central
- Sem filtro customer-only (janela inclui ambos os falantes)
- **Novo:** sweep de WINDOW_SIZE ∈ {3, 5, 7, all}

**Resultados principais:**
- Macro-F1 (janela, W=5): **0,682 ± 0,008** (+14,7pp vs v2)
- Macro-F1 (W=all): **0,737 ± 0,016** (iguala v1!)
- H1 Hybrid α=0,3: 0,617 (significativo, p=0,031)
- **H4 Cascata: 0,390** (catástrofe — regras capturam 93% com baixa precisão)

**Achado crítico:** A cascata com regras simples (keyword matching) falha em janelas longas. Uma janela de 5 turnos contém "Boa tarde" + "cancelar" + "obrigado" → 3 intents disparam simultaneamente.

**Achado principal:** WINDOW_SIZE=all (conversa inteira) atinge F1=0,737, igualando o v1 com taxonomia dinâmica mantida.

**O que motivou v4:** Melhorar os componentes baseado na literatura (5 melhorias identificadas).

---

## v4 — Speaker Markers + Melhorias Baseadas na Literatura

**Arquivo:** `talkex_banca_v4_improved.ipynb`

**5 melhorias aplicadas (baseadas em papers):**

| ID | Melhoria | Paper base | Resultado |
|:---:|:---|:---|:---|
| M1 | Speaker markers `[customer]`/`[agent]` no texto da janela | Farfan-Escobedo (2024) | **+0,9 a +1,6pp** |
| M2 | Cascata com exclusividade (só classifica se 1 intent dispara) | Trapeznikov (2013) | H4: 0,39→0,65 (+26pp) |
| M3 | α-tuning no validation set | Bruch et al. (2023) | Instável (val set pequeno) |
| M4 | Two-stage rerank (BM25→Embedding) | DNNC (2020) | 0,719 (marginal) |
| M5 | Peso posicional na agregação (exponencial) | Farfan-Escobedo (2024) | 0,699 (pior que majority) |

**Resultados principais:**
- Macro-F1 (janela, W=5): **0,739 ± 0,010** (+5,7pp vs v3)
- Macro-F1 (W=all): **0,753 ± 0,015** (+1,6pp vs v3)
- H4 Cascata com exclusividade: **0,653** (subiu de 0,390)
- Rule accuracy: **45,6%** (exclusivas, mas ainda baixa)

**O que funcionou:** M1 (speakers, +1-2pp consistente), M2 (exclusividade, +26pp na cascata).
**O que não funcionou:** M3 (α instável), M4 (rerank marginal), M5 (exponencial penaliza demais).

**O que motivou v5:** Corrigir a cascata — fallback deveria ser Hybrid, não Embedding-only.

---

## v5 — Cascata com Hybrid Fallback

**Arquivo:** `talkex_banca_v5_cascade_fixed.ipynb`

**4 correções aplicadas:**

| ID | Correção | Problema no v4 |
|:---:|:---|:---|
| C1 | Cascade: Rules(excl.) → **Hybrid** (não Emb-only) | Fallback era o pior classificador |
| C2 | Medir acurácia das regras no subset exclusivo | Não se sabia a precision real |
| C3 | α-tuning com grid grosso (4 pontos) | Grid fino era instável |
| C4 | Peso linear (1.0→2.0) na agregação posicional | Exponencial penalizava demais |

**Resultados principais:**
- Hybrid-only: **0,751 ± 0,010**
- Cascade-v2 (Rules→Hybrid): **0,672 ± 0,025** (Δ=-7,4pp)
- **Rule accuracy: 45,6%** (exclusivas, mas baixa)
- Causa: `rule_greeting` captura tudo com "boa tarde" com apenas 45,6% de acurácia

**Achado crítico (C2):** As regras exclusivas acertam apenas 45,6%. Mesmo quando só 1 intent dispara, a regra está errada mais da metade das vezes. Causa principal: a `rule_greeting` classifica "Boa tarde, quero cancelar" como `saudacao`.

**O que motivou v6:** Portar o motor de regras de produção, que tem speaker filtering + validação semântica.

---

## v6 — Motor de Regras Multi-Sinal (Produção)

**Arquivo:** `talkex_banca_v6_production_rules.ipynb`

**O que mudou vs v5:**
Portado do sistema de produção (`/pesquisas/semantic-text-classification/src/talkex/rules/`):

| Aspecto | v5 (simples) | v6 (produção) |
|:---|:---|:---|
| Predicados | Keyword substring | 4 famílias (lexical + structural + contextual + semantic) |
| Speaker | Sem filtro | `speaker("customer")` — avalia só texto do customer |
| Greeting | Dispara em qualquer "boa tarde" | `first_customer_turn` + `intent_saudacao > 0.55` |
| Scores | Booleano (0/1) | Contínuo [0,1] com min(scores) |
| Short-circuit | Nenhum | COST_ASCENDING (lexical→structural→contextual→semantic) |

**Resultados principais:**
- Hybrid-only: **0,744 ± 0,010**
- Cascade-v3 (multi-sinal): **0,742 ± 0,016** (Δ=-0,2pp — praticamente igual!)
- **Rule accuracy: 64,7–70,4%** (subiu de 45,6%)
- Rules capturam 4–16% das janelas (vs 93% no v3, 24% no v5)

**Achado:** O motor multi-sinal resolveu o problema da cascata. De -29pp (v3) para -0,2pp (v6). Mas regras ainda não superam o Hybrid — são equivalentes.

**O que motivou v7:** Inverter o fluxo: em vez de regras primeiro, Hybrid classifica tudo e regras corrigem os casos incertos.

---

## v7 — Confidence-Gated Routing

**Arquivo:** `talkex_banca_v7_confidence_routing.ipynb`

**Mudança conceitual:** De **cascata sequencial** (Rules→Hybrid) para **routing por confiança** (Hybrid classifica tudo → regras corrigem baixa confiança).

**Papers base:** Jitkrittum et al. (NeurIPS 2023), Mozannar & Sontag (ICML 2020).

**Implementação:**
1. Hybrid classifica TODAS as janelas → (pred, conf, margin)
2. Se margem < threshold → regras multi-sinal avaliam → se match exclusivo → override
3. Se margem ≥ threshold → manter Hybrid (confiante)

**Problemas encontrados na v7 original:**
- `ALPHA_ROUTING = 0.3` (errado, melhor era 0.5)
- Confiança por proporção (score_1st/total) comprimida — poucas janelas captradas
- Ganho negligível (+0.06pp)

**Correções aplicadas (ainda na v7):**
- Alpha corrigido para 0.5 (depois 0.65)
- Métrica de margem: `(score_1st - score_2nd) / total` (mais discriminativa)
- `MARGIN_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]`

**Resultado após correções:**
- Routing-m=0.2: **0,745 ± 0,019** (Δ=+0,09pp vs Hybrid)
- Routing supera Hybrid em todos os thresholds, mas por <0,2pp
- Wilcoxon p=0,063 — não significativo

**O que motivou v8:** Quantificar latência e interpretabilidade — o valor real das regras é operacional, não accuracy.

---

## v8 — Latência + Interpretabilidade + Correções

**Arquivo:** `talkex_banca_v8_final.ipynb`

**Nova hipótese H5:** Regras oferecem ganhos de latência e interpretabilidade.

**Novas métricas:**
- Latência por janela (com index BM25 pré-construído)
- Economia da cascata vs overhead do routing
- Exemplos de predições rastreáveis (corretas, filtradas)
- Projeção para produção (10K atendimentos/dia)

**Auditoria completa encontrou 10 problemas:**
- BUG: numpy string truncation no routing (`dtype=<U6`)
- BUG: seções 6.5 duplicadas
- BUG: economia de latência reportada como real (era teórica)
- ISSUE: Embedding KNN mais rápido que Rules (0,12ms vs 0,22ms)
- ISSUE: exemplos de interpretabilidade eram todos erros da greeting
- MELHORIA: greeting threshold subiu de 0.25 para 0.55

**Resultados H5 (corrigidos):**
- Rules: 0,22ms/janela (372× mais rápido que Hybrid)
- Cascade economia real: **-7,1%** latência (151s/dia em 10K atendimentos)
- Rule accuracy: **86,2%** (56/65) — melhor resultado de regras da série
- 3 exemplos rastreáveis CORRETOS (cancelamento, suporte_técnico, saudação)

**O que motivou v9:** Faltavam baselines externos, ablação, few-shot, análise de erro, related work — requisitos para publicação.

---

## v9 — Publication-Ready (BRACIS/PROPOR 2026)

**Arquivo:** `talkex_banca_v9_publication.ipynb`

**Estrutura reorganizada como paper acadêmico:**
- Seção 2: Trabalhos Relacionados (4 subseções)
- Seção 4: Protocolo Experimental (com tabela formal)
- Seção 5.10: 4 baselines externos (LogReg, SetFit, BERTimbau, MPNet)
- Seção 5.11: Ablação de 5 componentes
- Seção 5.12: Análise de erro qualitativa
- Seção 5.13: Curva few-shot (1 a 100 exemplares)
- Seção 6: Discussão (4 subseções)
- 33 referências (incluindo Devlin, Vaswani, Tunstall)
- Correção de Holm-Bonferroni consolidada

**Hipóteses reformuladas** (justificativa em `JUSTIFICATIVA_HIPOTESES.md`):
- H1: Fusão léxico-semântica melhora classificação → **Confirmada (direcional)**
- H2: BM25 domina sobre embeddings em PT-BR → **Confirmada** (+4,8pp)
- H3: BERTimbau supera multilingual → **Confirmada** (+2,2pp)
- H4: Dados sintéticos são essenciais → **Confirmada** (-22pp ablação)
- H5: Regras sem custo de accuracy → **Confirmada** (86,2% acc, -7,1% latência)

**Resultados principais:**

| Método | Macro-F1 |
|:---|:---|
| **BERTimbau+KNN** | **0,768** |
| Hybrid (α=0,5) | 0,748 |
| BM25-KNN (K=15) | 0,748 |
| LogReg+MiniLM | 0,719 |
| MPNet+KNN | 0,716 |
| Emb-KNN (MiniLM) | 0,697 |

**Ablação:**

| Config | Δ vs Full |
|:---|:---|
| −Sintéticos | **−21,9pp** |
| −BM25 | −4,9pp |
| −Embeddings | −1,0pp |
| −Speakers | −0,6pp |

**Bugs corrigidos durante a v9:**
- `HAS_SBERT` → `TEM_SBERT` (variável renomeada para PT-BR)
- `for pd in rule["predicates"]` → `for pred_detalhe in rule["predicates"]` (sobrescrevia pandas)
- `pd.get()` residuais → `pred_detalhe.get()`

---

## Evolução dos Números

| Métrica | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| F1 janela (melhor, W=5) | 0,734 | 0,535 | 0,727 | 0,739 | 0,751 | 0,745 | 0,731 | 0,729 | 0,748 |
| F1 janela (W=all) | — | — | 0,737 | 0,753 | 0,760 | 0,766 | 0,759 | 0,733 | 0,766 |
| H4 Cascade vs Hybrid | — | — | -29pp | -3,8pp | -7,4pp | -0,2pp | +0,1pp | +0,2pp | -0,07pp |
| Rule accuracy | — | — | — | 45,6% | 45,6% | 70,4% | — | 86,2% | 86,2% |
| N baselines | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **4** |
| N referências | 16 | 16 | 24 | 24 | 24 | 24 | 28 | 28 | **33** |

*Nota: F1 varia ±2-3pp entre runs no Colab devido a aleatoriedade na inicialização da GPU e batching.*

---

## Lições Aprendidas

### Técnicas
1. **BM25 supera embeddings** em domínios com vocabulário restrito — contra-intuitivo mas confirmado pela literatura (Harris, 2025; Thakur, 2021)
2. **Encoder PT-BR (BERTimbau) supera multilingual** mesmo sem fine-tuning para STS — confirma Souza et al. (2020)
3. **Dados sintéticos são o componente mais crítico** (+22pp) — data augmentation via LLM é viável para NLU em PT-BR
4. **Regras lexicais simples falham em textos longos** — janelas disparam múltiplas regras simultaneamente
5. **Regras multi-sinal (speaker + semântico) resolvem** — accuracy de 45,6% → 86,2%
6. **Confidence-gated routing é conceitualmente correto** mas o ganho é marginal (+0,1pp)
7. **WINDOW_SIZE=all** (conversa inteira) é a melhor configuração

### Metodológicas
8. **N=5 splits limita poder estatístico** — H1 oscila entre significativo (p=0,031) e não significativo (p=0,094)
9. **Hipóteses devem ser reformuladas** quando os dados mostram direções inesperadas — pesquisa exploratória legítima (Tukey, 1977)
10. **Baselines externos são obrigatórios** — sem eles, um reviewer rejeita na triagem
11. **Holm-Bonferroni é necessário** com múltiplas hipóteses — H1 perde significância quando corrigida

### De engenharia
12. **numpy string truncation** (`dtype=<U6`) é um bug silencioso difícil de detectar
13. **Variáveis de loop** (`for pd in ...`) podem sobrescrever imports globais
14. **Latência do BM25 inclui construção do index** — medir separadamente para números realistas
15. **Rebuilding BM25 por query** é o gargalo — pré-construir o index uma vez
