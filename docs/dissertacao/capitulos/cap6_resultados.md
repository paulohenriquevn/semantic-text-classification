# Capítulo 6 — Resultados e Análise

Este capítulo apresenta os resultados experimentais obtidos para cada uma das quatro hipóteses formuladas no Capítulo 1, seguidos dos estudos de ablação e da análise comparativa com trabalhos correlatos. Para cada hipótese, reportamos as métricas definidas no protocolo experimental (Capítulo 5), aplicamos os testes estatísticos previstos e interpretamos os resultados à luz da tese central. Encerramos com a discussão integrada dos achados e suas implicações.

---

## 6.1 Configuração Experimental

Os experimentos foram conduzidos sobre o corpus consolidado de 2.257 conversas de atendimento em português brasileiro (1.179 do corpus base HuggingFace + 1.078 conversas expandidas sinteticamente), dividido em splits estratificados: treinamento (1.581 conversas, 70%), validação (338, 15%) e teste (338, 15%), com seed fixo (42) para reprodutibilidade. A distribuição das 9 classes de intent no split de teste reflete a proporção do corpus completo.

Todos os experimentos utilizam a biblioteca TalkEx para geração de embeddings, garantindo que o pipeline de pré-processamento (normalização NFKD, tokenização), encoding (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensões), pooling e normalização L2 seja idêntico em todas as condições experimentais.

**Tabela 6.1** — Distribuição de classes no split de teste.

| Classe | N (teste) | % |
|---|---|---|
| reclamacao | 55 | 16,3% |
| duvida_produto | 51 | 15,1% |
| duvida_servico | 50 | 14,8% |
| suporte_tecnico | 46 | 13,6% |
| compra | 36 | 10,7% |
| cancelamento | 32 | 9,5% |
| saudacao | 26 | 7,7% |
| elogio | 22 | 6,5% |
| outros | 20 | 5,9% |

---

## 6.2 Resultados do Retrieval Híbrido (H1)

**Hipótese H1:** *O retrieval híbrido (BM25 + ANN com fusão de scores) supera os componentes isolados em métricas de recuperação.*

### 6.2.1 Resultados Gerais

A Tabela 6.2 apresenta os resultados de retrieval para os 8 sistemas avaliados. O ground truth define como relevantes todas as conversas do conjunto de treinamento que compartilham o mesmo intent da query de teste.

**Tabela 6.2** — Resultados de retrieval (H1). Melhor valor em cada métrica em negrito.

| Sistema | MRR | nDCG@10 | Recall@10 | Recall@20 | Precision@5 |
|---|---|---|---|---|---|
| BM25-base | 0,802 | 0,583 | 0,029 | 0,052 | 0,603 |
| BM25-norm | 0,802 | 0,583 | 0,029 | 0,052 | 0,603 |
| ANN-MiniLM | 0,799 | 0,591 | 0,030 | 0,053 | 0,617 |
| **Hybrid-RRF** | **0,826** | **0,613** | **0,031** | **0,055** | **0,635** |
| Hybrid-LINEAR α=0,30 | 0,813 | 0,601 | 0,030 | 0,055 | 0,629 |
| Hybrid-LINEAR α=0,50 | 0,825 | 0,613 | 0,031 | 0,055 | 0,623 |
| Hybrid-LINEAR α=0,65 | 0,823 | 0,610 | 0,031 | 0,055 | 0,631 |
| Hybrid-LINEAR α=0,80 | 0,810 | 0,598 | 0,030 | 0,055 | 0,626 |

### 6.2.2 Análise Estatística

Aplicamos o teste de Wilcoxon signed-rank pareado sobre os scores de reciprocal rank por query (338 pares) para avaliar a significância das diferenças.

**Tabela 6.3** — Testes estatísticos para H1 (Hybrid-RRF vs baselines).

| Comparação | Diff. média | p-valor | Significativo (α=0,05) | Tamanho do efeito |
|---|---|---|---|---|
| Hybrid-RRF vs BM25-base | +0,024 | 0,103 | Não | 0,185 (pequeno) |
| Hybrid-RRF vs ANN-MiniLM | +0,027 | **0,032** | **Sim** | 0,245 (pequeno) |
| Hybrid-RRF vs LINEAR-α0,50 | +0,001 | 0,969 | Não | 0,006 |

O intervalo de confiança bootstrap (95%) para a diferença Hybrid-RRF vs BM25 é [-0,003; 0,052], incluindo zero, o que corrobora a não-significância estatística.

### 6.2.3 Discussão

O Hybrid-RRF atinge o maior MRR (0,826), porém a diferença em relação ao BM25 isolado (0,802) não é estatisticamente significativa (p=0,103). Este resultado é cientificamente relevante por dois motivos:

1. **BM25 é surpreendentemente forte neste domínio.** Conversas de atendimento contêm marcadores lexicais explícitos — termos como "cancelar", "problema", "elogio" — que o BM25 captura eficientemente. Este achado é consistente com Harris (2025), que reportou superioridade do BM25 sobre métodos neurais em documentos médicos rigidamente estruturados.

2. **O híbrido supera significativamente o ANN isolado** (p=0,032), demonstrando que a combinação com BM25 estabiliza o retrieval semântico. O componente lexical atua como "rede de segurança" para queries com termos discriminativos explícitos.

A normalização de diacríticos (BM25-norm) não produziu nenhuma diferença em relação ao BM25-base (MRR idêntico: 0,802). Isto se explica pelo fato de que as conversas do corpus — tanto originais quanto expandidas — já utilizam linguagem com acentuação consistente, eliminando a necessidade de normalização.

**Veredicto H1:** *Parcialmente confirmada.* O retrieval híbrido é consistentemente superior, mas a diferença não atinge significância estatística contra o BM25 neste corpus. A significância contra o ANN isolado confirma a complementaridade dos paradigmas.

---

## 6.3 Resultados da Classificação Multi-Nível (H2)

**Hipótese H2:** *A combinação de features lexicais com embeddings densos pré-treinados supera representações de nível único na classificação de intents.*

### 6.3.1 Resultados Gerais

A Tabela 6.4 apresenta o Macro-F1 e Micro-F1 para as 6 configurações avaliadas (2 representações × 3 classificadores).

**Tabela 6.4** — Resultados de classificação (H2).

| Representação | Classificador | Macro-F1 | Micro-F1 | Duração (ms) |
|---|---|---|---|---|
| Lexical-only | LogReg | 0,198 | 0,233 | 1.048 |
| Lexical-only | LightGBM | 0,309 | 0,330 | 645 |
| Lexical-only | MLP | 0,042 | 0,164 | 67 |
| Lexical+Emb | LogReg | 0,559 | 0,604 | 103.061 |
| **Lexical+Emb** | **LightGBM** | **0,715** | **0,757** | **6.084** |
| Lexical+Emb | MLP | 0,537 | 0,626 | 788 |

### 6.3.2 Análise por Classe

A Figura 6.1 (heatmap H2) revela padrões distintos de dificuldade por classe:

**Tabela 6.5** — F1 por classe para a melhor configuração (lexical+emb LightGBM).

| Classe | F1 | Precision | Recall | Interpretação |
|---|---|---|---|---|
| cancelamento | 0,951 | 1,000 | 0,906 | Classe mais fácil — vocabulário distintivo |
| duvida_servico | 0,855 | 0,807 | 0,920 | Alto recall, boa discriminação semântica |
| suporte_tecnico | 0,796 | 0,769 | 0,870 | Termos técnicos são discriminativos |
| reclamacao | 0,774 | 0,667 | 0,946 | Alto recall, precision limitada por sobreposição com "outros" |
| duvida_produto | 0,743 | 0,740 | 0,725 | Sobreposição moderada com "duvida_servico" |
| elogio | 0,722 | 0,929 | 0,591 | Alta precision, recall limitado |
| saudacao | 0,667 | 0,882 | 0,577 | Padrão similar ao elogio |
| compra | 0,649 | 0,600 | 0,583 | Maior confusão com duvida_produto |
| outros | 0,276 | 0,500 | 0,150 | Classe residual — sem padrão coeso |

### 6.3.3 Discussão

O ganho da adição de embeddings ao LightGBM é de +131% em Macro-F1 (0,309 → 0,715), confirmando que representações semânticas densas capturam padrões que features lexicais (TF-IDF, contagem de tokens) não conseguem representar. Os embeddings pré-treinados (paraphrase-multilingual-MiniLM-L12-v2) fornecem 384 dimensões de informação semântica sem nenhum treinamento no domínio.

O LightGBM supera consistentemente LogReg (+28%) e MLP (+33%) na configuração lexical+emb. A robustez do LightGBM a features de escalas diferentes (TF-IDF em [0,1] vs embeddings L2-normalizados em [-1,1]) e sua capacidade de selecionar features informativas via gradient boosting explicam essa superioridade.

A classe "outros" (F1=0,276) representa a maior dificuldade — isto é esperado e desejável, pois trata-se de uma classe residual que agrupa conversas sem intent claro. A validação do dataset (Seção 5.1.4) confirmou que o par "outros↔saudacao" apresenta a maior similaridade inter-classe (0,972), explicando a confusão observada.

**Veredicto H2:** *Confirmada.* A combinação lexical+embedding supera representações de nível único por margem substancial (+131% vs lexical-only). O LightGBM com embeddings pré-treinados atinge Macro-F1=0,715 em 9 classes sem nenhum treinamento de modelo de linguagem.

---

## 6.4 Resultados das Regras Determinísticas (H3)

**Hipótese H3:** *Regras determinísticas complementam o pipeline de ML, melhorando precision em classes críticas sem degradar o desempenho global.*

### 6.4.1 Resultados Gerais

A Tabela 6.6 apresenta os resultados de quatro configurações: ML isolado, regras isoladas, regras como override pós-classificação, e regras como features adicionais do classificador.

**Tabela 6.6** — Resultados de complementaridade ML+Regras (H3).

| Configuração | Macro-F1 | F1 cancel. | F1 reclam. | Duração (ms) |
|---|---|---|---|---|
| ML-only | 0,709 | 0,951 | 0,782 | 40 |
| Rules-only | 0,130 | 0,674 | 0,353 | 103 |
| ML+Rules-override | 0,624 | 0,674 | 0,677 | 92 |
| **ML+Rules-feature** | **0,714** | **1,000** | **0,806** | 7.239 |

### 6.4.2 Análise de Classes Críticas

A Tabela 6.7 detalha precision e recall nas duas classes críticas para as quais regras foram definidas.

**Tabela 6.7** — Precision e Recall em classes críticas (H3).

| Config | cancel. P | cancel. R | cancel. F1 | reclam. P | reclam. R | reclam. F1 |
|---|---|---|---|---|---|---|
| ML-only | 1,000 | 0,906 | 0,951 | 0,667 | 0,946 | 0,782 |
| Rules-only | 0,508 | 1,000 | 0,674 | 0,500 | 0,273 | 0,353 |
| ML+Rules-override | 0,508 | 1,000 | 0,674 | 0,587 | 0,800 | 0,677 |
| **ML+Rules-feature** | **1,000** | **1,000** | **1,000** | **0,703** | **0,946** | **0,806** |

### 6.4.3 Discussão

O resultado mais notável deste experimento é que a configuração ML+Rules-feature atinge **F1=1,000 na classe cancelamento** (precision e recall perfeitos), superando o ML isolado (F1=0,951, com recall de 0,906). A adição da feature binária `rule_cancel` permite que o LightGBM identifique os 3 casos de cancelamento que o ML sozinho errava (falsos negativos), sem introduzir falsos positivos.

A estratégia de **override** — onde a regra substitui a predição do ML quando dispara — revelou-se prejudicial (Macro-F1 0,624 vs 0,709 do ML-only). Isso ocorre porque as regras lexicais, embora sensíveis (recall alto), são imprecisas: o termo "cancelar" aparece em conversas de outras classes (e.g., "gostaria de cancelar minha reclamação sobre..."), gerando falsos positivos quando a regra sobrescreve a predição do ML.

Em contraposição, a estratégia **rules-as-feature** delega a decisão final ao classificador, que aprende a ponderar o sinal da regra em contexto. O LightGBM efetivamente aprende que "regra de cancelamento disparou E embeddings indicam cancelamento → cancelamento com alta confiança". Essa sinergia é a contribuição central da H3.

**Veredicto H3:** *Confirmada.* Regras como features complementam o ML com ganho de +0,5pp no Macro-F1 global e ganhos dramáticos em classes críticas (cancelamento: +4,9pp, atingindo F1 perfeito). A estratégia de override, entretanto, é prejudicial e não recomendada.

---

## 6.5 Resultados da Inferência em Cascata (H4)

**Hipótese H4:** *A inferência em cascata permite reduzir o custo computacional com degradação controlada da qualidade.*

### 6.5.1 Resultados Gerais

A Tabela 6.8 apresenta os resultados da cascata com dois estágios: Estágio 1 (LogReg, classificador linear rápido) e Estágio 2 (LightGBM com 200 árvores, classificador ensemble completo). Ambos utilizam features lexicais + embeddings.

**Tabela 6.8** — Resultados da inferência em cascata (H4).

| Configuração | Macro-F1 | % Estágio 1 | % Estágio 2 | Δ F1 vs uniforme |
|---|---|---|---|---|
| **Uniforme (baseline)** | **0,741** | 0% | 100% | — |
| Cascade t=0,50 | 0,678 | 47,6% | 52,4% | -0,063 |
| Cascade t=0,60 | 0,714 | 32,0% | 68,0% | -0,028 |
| Cascade t=0,70 | 0,705 | 19,5% | 80,5% | -0,036 |
| Cascade t=0,80 | 0,718 | 9,5% | 90,5% | -0,023 |
| Cascade t=0,90 | 0,739 | 2,7% | 97,3% | **-0,003** |

### 6.5.2 Análise de Custo

O custo por amostra medido (com warmup e 10 repetições) revela:
- LogReg (estágio leve): 0,043 ms/amostra
- LightGBM (estágio completo): 0,071 ms/amostra

A razão de custo é de 1,6×, o que é modesto — ambos os classificadores são leves em CPU. Em cenário de produção, a diferença seria amplificada quando o estágio leve usa regras lexicais simples (μs) e o estágio completo requer geração de embeddings via modelo neural (~100 ms).

### 6.5.3 Discussão

O threshold t=0,90 representa o ponto ótimo operacional: resolve 2,7% das amostras no estágio leve com degradação de apenas 0,3pp em Macro-F1 (0,739 vs 0,741). Essas são as conversas mais "fáceis" — tipicamente saudações curtas ou cancelamentos explícitos — onde o LogReg atinge confiança >90%.

A curva de Pareto (Figura 6.4) mostra que não existe nenhum threshold que alcance simultaneamente redução de custo substancial e degradação inferior a 2% de F1 no cenário experimental atual. Isso é esperado dada a semelhança de custo entre os dois estágios. Em cenário de produção, onde o estágio leve utilizaria regras determinísticas (sem embeddings) e o estágio completo exigiria inferência neural, a cascata ofereceria benefícios de custo proporcionalmente maiores.

**Veredicto H4:** *Parcialmente confirmada.* A cascata demonstra viabilidade conceitual com degradação controlável (t=0,90: delta=-0,003), mas os benefícios de custo são limitados quando ambos os estágios compartilham o mesmo custo de geração de embeddings. A contribuição é mais arquitetural que numérica neste experimento.

---

## 6.6 Estudos de Ablação

Para quantificar a contribuição individual de cada família de features ao pipeline completo, conduzimos um estudo de ablação sistemático. O baseline é o LightGBM com todas as features (lexicais + estruturais + embeddings + regras, Macro-F1=0,714).

### 6.6.1 Resultados

**Tabela 6.9** — Estudo de ablação: impacto da remoção de cada componente.

| Configuração | Macro-F1 | Δ vs Full | N features | Interpretação |
|---|---|---|---|---|
| **Full pipeline** | **0,714** | baseline | 397 | Todas as features |
| -Structural | 0,722 | **+0,008** | 393 | Structural features são ruído |
| -Rules | 0,709 | -0,005 | 395 | Regras contribuem +0,5pp |
| Emb-only | 0,708 | -0,006 | 384 | Embeddings sozinhos quase igualam full |
| -Lexical | 0,699 | -0,015 | 390 | Lexical contribui +1,5pp |
| -Embeddings | 0,364 | **-0,350** | 13 | Colapso sem embeddings |
| Lexical-only | 0,295 | -0,420 | 11 | Insuficiente isoladamente |

### 6.6.2 Discussão

A ablação revela uma hierarquia clara de importância dos componentes:

1. **Embeddings são indispensáveis** (contribuição: +35,0pp). Sua remoção causa colapso do Macro-F1 de 0,714 para 0,364 — uma degradação de 49%. Os embeddings pré-treinados fornecem a representação semântica que permite ao classificador distinguir intents com vocabulário sobreposto.

2. **Features lexicais contribuem moderadamente** (+1,5pp). TF-IDF e contadores lexicais adicionam informação complementar sobre frequência e estrutura do texto que os embeddings não capturam diretamente.

3. **Features de regras contribuem cirurgicamente** (+0,5pp global, mas +4,9pp em cancelamento). A contribuição global é modesta, porém o impacto em classes críticas é desproporcional — conforme demonstrado na Seção 6.4.

4. **Features estruturais são prejudiciais** (-0,8pp quando presentes). As features `turn_count`, `speaker_count` e flags booleanos de role são constantes ou quase-constantes no corpus (todas as conversas têm 2 speakers), introduzindo dimensões sem informação que o LightGBM não consegue descartar completamente.

O achado de que **Emb-only (F1=0,708)** praticamente iguala o **full pipeline (F1=0,714)** sugere que, em cenários onde simplicidade é prioritária, embeddings pré-treinados com LightGBM constituem uma solução altamente competitiva com apenas 384 features.

---

## 6.7 Comparação com Trabalhos Correlatos

A Tabela 6.10 situa os resultados do TalkEx no contexto da literatura revisada no Capítulo 3. Ressaltamos que as métricas não são diretamente comparáveis entre trabalhos devido a diferenças de datasets, tarefas e definições de relevância.

**Tabela 6.10** — Comparação contextualizada com trabalhos correlatos.

| Dimensão | TalkEx | BERTaú (Finardi, 2021) | Harris (2025) | Rayo (COLING, 2025) |
|---|---|---|---|---|
| **Retrieval (melhor)** | MRR 0,826 (Hybrid-RRF) | MRR 0,552 (pairwise) | — | Recall@10 0,833 |
| **Classificação** | F1 0,715 (9 classes) | — | — | — |
| **Treinamento de LM** | Nenhum (pré-treinado) | 1M steps, 14,5 GB, GPU | Nenhum | Fine-tuning, GPU A40 |
| **Hardware mínimo** | CPU | GPU com FP16 | CPU | GPU A40 |
| **Regras auditáveis** | Sim (DSL→AST) | Não | Não | Não |
| **Inferência cascata** | Sim (2 estágios) | Não | Não | Não |
| **Língua** | PT-BR | PT-BR | EN | EN |

O padrão "híbrido > isolado" é confirmado nos três trabalhos que avaliam busca híbrida: TalkEx (+3,0% MRR vs BM25), BERTaú (+60% vs BM25+) e Rayo (+9,5% Recall@10 vs BM25). Entretanto, o investimento necessário para alcançar esses ganhos difere dramaticamente: BERTaú requer treinamento from scratch com 14,5 GB de dados proprietários e GPU, enquanto TalkEx utiliza embeddings pré-treinados multilinguais em CPU.

Harris (2025) demonstra que BM25 supera métodos neurais em documentos médicos rigidamente estruturados — achado corroborado pelo forte desempenho do BM25 no TalkEx (MRR 0,802, não significativamente inferior ao híbrido).

---

## 6.8 Discussão Integrada

### 6.8.1 Síntese das Hipóteses

**Tabela 6.11** — Veredicto consolidado das hipóteses.

| Hipótese | Veredicto | Evidência principal |
|---|---|---|
| H1: Retrieval híbrido | Parcialmente confirmada | MRR +3% (p=0,103 vs BM25); significativo vs ANN (p=0,032) |
| H2: Representação multi-nível | Confirmada | F1 +131% com embeddings (+0,406 absolutos) |
| H3: Regras + ML | Confirmada | F1 global +0,5pp; cancelamento F1=1,000 |
| H4: Inferência cascata | Parcialmente confirmada | t=0,90: delta=-0,003; benefício limitado por custo compartilhado |

### 6.8.2 Resultados Inesperados

Três resultados merecem destaque por divergirem das expectativas iniciais:

1. **A força do BM25.** Esperávamos que o retrieval semântico superasse o lexical por margem significativa. O BM25 atingiu MRR 0,802 — apenas 0,024 abaixo do híbrido — sugerindo que conversas de atendimento possuem sinais lexicais mais fortes do que o antecipado. Isso tem implicação prática direta: para organizações que não podem investir em infraestrutura de embeddings, BM25 é uma alternativa viável.

2. **O efeito prejudicial do override.** Esperávamos que regras como pós-processamento melhorassem classes críticas. O Macro-F1 caiu de 0,709 para 0,624 com override, demonstrando que regras lexicais simples não devem substituir predições de ML, mas informá-las como features adicionais.

3. **Features estruturais como ruído.** A remoção de features estruturais (turn count, speaker flags) melhorou o F1 em 0,8pp, indicando que features com baixa variância podem prejudicar classificadores tree-based quando o número de features informativas é grande (384 embeddings vs 4 features estruturais constantes).

### 6.8.3 Implicações Práticas

Os resultados sugerem um pipeline prático para classificação de conversas de atendimento:

1. **Embeddings pré-treinados multilinguais** como representação base — sem necessidade de fine-tuning ou GPU.
2. **LightGBM** como classificador — treinamento em ~6 segundos em CPU, inferência em <0,1ms por amostra.
3. **Regras DSL como features** para classes críticas — ganho cirúrgico em precision/recall com auditabilidade total.
4. **BM25 para retrieval** quando infraestrutura é limitada — competitivo com híbrido neste domínio.
5. **Cascata** reservada para cenários onde os estágios têm custo genuinamente diferente (e.g., regras simples vs inferência neural).

---

## 6.9 Ameaças à Validade

### 6.9.1 Validade Interna

- **Corpus sintético.** Os dados foram gerados via LLM, o que pode introduzir vieses de geração (linguagem mais regular, menor ruído que dados reais). Mitigamos com o protocolo de validação Phase 0.5 (Seção 5.1.4).
- **Normalização L2 nos embeddings.** O `SentenceTransformerGenerator` aplica normalização L2 por default, o que comprime a escala das features de embedding para [-1,1]. Isso beneficia classificadores tree-based (invariantes a transformações monotônicas) mas pode prejudicar modelos lineares e redes neurais quando combinadas com features lexicais em escala diferente.

### 6.9.2 Validade Externa

- **Domínio único.** Os resultados são específicos para conversas de atendimento em PT-BR. A generalização para outros domínios (jurídico, saúde, financeiro) requer validação adicional.
- **Tamanho do corpus.** Com 2.257 conversas, o corpus é menor que os utilizados por BERTaú (14,5 GB) e Rayo (27.869 perguntas). Resultados com dataset maior poderiam alterar a significância estatística de H1.

### 6.9.3 Validade de Constructo

- **Regras manuais.** As regras foram definidas pelo pesquisador com base em análise qualitativa do corpus, não por especialistas do domínio. Em cenário operacional, regras seriam definidas por analistas de qualidade com conhecimento do negócio.
- **Métricas de custo em H4.** O custo foi medido em tempo de inferência Python em CPU, não em custo monetário real de infraestrutura. Em produção, o custo inclui geração de embeddings, latência de rede e utilização de GPU.
