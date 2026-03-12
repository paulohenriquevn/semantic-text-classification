# Capítulo 6 — Resultados e Análise

Este capítulo apresenta os resultados experimentais obtidos para cada uma das quatro hipóteses formuladas no Capítulo 1, seguidos dos estudos de ablação e da análise comparativa com trabalhos correlatos. Para cada hipótese, reportamos as métricas definidas no protocolo experimental (Capítulo 5), aplicamos os testes estatísticos previstos e interpretamos os resultados à luz da tese central. Encerramos com a discussão integrada dos achados e suas implicações.

---

## 6.1 Configuração Experimental

Os experimentos foram conduzidos sobre o corpus consolidado de 2.257 conversas de atendimento em português brasileiro (1.179 do corpus base HuggingFace + 1.078 conversas expandidas sinteticamente), dividido em splits estratificados: treinamento (1.581 conversas, 70%), validação (338, 15%) e teste (338, 15%), com seed fixo (42) para reprodutibilidade. A distribuição das 9 classes de intent no split de teste reflete a proporção do corpus completo.

Para os experimentos de classificação (H2–H4 e ablação), a unidade de análise é a **janela de contexto** (5 turnos, stride 2), gerada pelo módulo `SlidingWindowBuilder` do TalkEx. Cada conversa produz em média ~4 janelas, totalizando 6.583 janelas de treino, 1.429 de validação e 1.368 de teste. O treinamento opera no nível da janela; a avaliação agrega predições ao nível da conversa via média de probabilidades de classe (ver Seção 5.1.6). Todos os experimentos utilizam o pipeline completo do TalkEx — segmentação de turnos (`TurnSegmenter`), construção de janelas (`SlidingWindowBuilder`), geração de embeddings (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensões) e extração de features — garantindo que a arquitetura avaliada é idêntica à arquitetura descrita.

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

**Veredicto H1: Refutada no critério primário; confirmada no critério secundário.** O retrieval híbrido (Hybrid-RRF) apresentou MRR numericamente superior ao BM25 (0,826 vs 0,802), porém a diferença não atingiu significância estatística (p=0,103 > 0,05), não satisfazendo o critério primário definido a priori. Em relação ao ANN isolado, a superioridade do Hybrid-RRF foi confirmada com significância estatística (p<0,05). Os resultados indicam que, neste corpus de 2.257 conversas, o ganho do componente semântico sobre o lexical é consistente mas insuficiente para rejeitar a hipótese nula com o nível de confiança estabelecido.

---

## 6.3 Resultados da Classificação Multi-Nível (H2)

**Hipótese H2:** *A combinação de features lexicais com embeddings densos pré-treinados supera representações puramente lexicais na classificação de intents.*

### 6.3.1 Resultados Gerais

A Tabela 6.4 apresenta o Macro-F1 e Micro-F1 para as 6 configurações avaliadas (2 representações × 3 classificadores).

**Tabela 6.4** — Resultados de classificação (H2). Avaliação multi-seed (5 seeds) com agregação janela→conversa via média de probabilidades.

| Representação | Classificador | Macro-F1 | Accuracy | std (Macro-F1) |
|---|---|---|---|---|
| Lexical-only | LogReg | 0,183 | 0,240 | 0,000 |
| Lexical-only | LightGBM | 0,334 | 0,364 | 0,000 |
| Lexical-only | MLP | 0,092 | 0,185 | 0,048 |
| Lexical+Emb | LogReg | 0,548 | 0,630 | 0,000 |
| **Lexical+Emb** | **LightGBM** | **0,659** | **0,722** | **0,000** |
| Lexical+Emb | MLP | 0,586 | 0,653 | 0,042 |

*Nota: LightGBM configurado com 100 árvores e 31 folhas. std=0,000 indica determinismo total do modelo (variância apenas no MLP devido à inicialização aleatória).*

O resultado anômalo do MLP com features exclusivamente lexicais (Macro-F1=0,092±0,048, inferior ao classificador aleatório com 9 classes ≈ 0,111) sugere falha de convergência da rede neural sobre o espaço esparso de features lexicais. Este resultado reforça que a escolha do classificador deve considerar a natureza das features: modelos baseados em árvores (LightGBM) são mais robustos a features heterogêneas.

### 6.3.2 Análise por Classe

A Figura 6.1 (heatmap H2) revela padrões distintos de dificuldade por classe:

**Tabela 6.5** — F1 por classe para a melhor configuração (lexical+emb LightGBM, avaliação por conversa).

| Classe | F1 | Precision | Recall | Interpretação |
|---|---|---|---|---|
| cancelamento | 0,968 | 1,000 | 0,938 | Classe mais fácil — vocabulário distintivo |
| suporte_tecnico | 0,833 | 0,800 | 0,870 | Termos técnicos são discriminativos |
| duvida_servico | 0,796 | 0,774 | 0,820 | Boa discriminação semântica |
| elogio | 0,778 | 1,000 | 0,636 | Alta precision, recall limitado |
| reclamacao | 0,722 | 0,584 | 0,945 | Alto recall, precision limitada por sobreposição |
| duvida_produto | 0,694 | 0,600 | 0,824 | Sobreposição com "duvida_servico" |
| saudacao | 0,541 | 0,909 | 0,385 | Alta precision, baixo recall |
| compra | 0,500 | 0,700 | 0,389 | Maior confusão com duvida_produto |
| outros | 0,095 | 1,000 | 0,050 | Classe residual — sem padrão coeso |

### 6.3.3 Discussão

Comparando o mesmo classificador (LightGBM), a adição de embeddings elevou o Macro-F1 de 0,334 para 0,659 — um ganho de 97%. Para LogReg, o ganho correspondente foi de 0,183 para 0,548 (+199%). Em ambos os casos, embeddings densos constituem o fator dominante de desempenho, confirmando que representações semânticas densas capturam padrões que features lexicais (contagem de tokens, razões de caracteres) não conseguem representar. Os embeddings pré-treinados (paraphrase-multilingual-MiniLM-L12-v2) fornecem 384 dimensões de informação semântica sem nenhum treinamento no domínio.

O LightGBM supera consistentemente LogReg (+20%) e MLP (+12%) na configuração lexical+emb. A robustez do LightGBM a features de escalas diferentes e sua capacidade de selecionar features informativas via gradient boosting explicam essa superioridade.

A classe "outros" (F1=0,095) representa a maior dificuldade — isto é esperado e desejável, pois trata-se de uma classe residual que agrupa conversas sem intent claro. Com a avaliação por janelas de contexto agregadas ao nível da conversa, a classe "outros" sofre particularmente porque suas janelas intermediárias não contêm sinais discriminativos, diluindo a predição durante a agregação.

Os testes Wilcoxon signed-rank sobre acurácia por query confirmam que a superioridade do lexical+emb LightGBM é significativa contra todos os baselines (p < 0,05 em todas as comparações). A diferença de accuracy entre lexical+emb LightGBM e lexical-only LightGBM é de 0,358 (IC 95%: [0,299; 0,417]).

**Veredicto H2:** *Confirmada.* A combinação lexical+embedding supera representações de nível único por margem substancial (+97% vs lexical-only). O LightGBM com embeddings pré-treinados atinge Macro-F1=0,659 (Accuracy=0,722) em 9 classes sem nenhum treinamento de modelo de linguagem. Os resultados são avaliados com o pipeline real do TalkEx, incluindo segmentação de turnos, janelas deslizantes e agregação de predições ao nível da conversa.

---

## 6.4 Resultados das Regras Determinísticas (H3)

**Hipótese H3:** *Regras determinísticas complementam o pipeline de ML, melhorando precision em classes críticas sem degradar o desempenho global.*

### 6.4.1 Resultados Gerais

A Tabela 6.6 apresenta os resultados de quatro configurações: ML isolado, regras isoladas, regras como override pós-classificação, e regras como features adicionais do classificador.

**Tabela 6.6** — Resultados de complementaridade ML+Regras (H3). Multi-seed (5 seeds), avaliação por conversa.

| Configuração | Macro-F1 | Accuracy | F1 cancel. | F1 reclam. |
|---|---|---|---|---|
| **ML-only** | **0,659** | **0,722** | 0,968 | 0,722 |
| Rules-only | 0,130 | 0,195 | 0,681 | 0,349 |
| ML+Rules-override | 0,574 | 0,618 | 0,719 | 0,629 |
| ML+Rules-feature | 0,654 | 0,713 | **0,984** | **0,723** |

*Nota: O baseline ML-only utiliza o mesmo LightGBM de H2 (100 árvores, 31 folhas). As regras são avaliadas por janela de contexto; a agregação ao nível da conversa segue a mesma estratégia de média de probabilidades.*

### 6.4.2 Análise de Classes Críticas

A Tabela 6.7 detalha precision e recall nas duas classes críticas para as quais regras foram definidas.

**Tabela 6.7** — Precision e Recall em classes críticas (H3).

| Config | cancel. P | cancel. R | cancel. F1 | reclam. P | reclam. R | reclam. F1 |
|---|---|---|---|---|---|---|
| ML-only | 1,000 | 0,938 | 0,968 | 0,584 | 0,945 | 0,722 |
| Rules-only | 0,516 | 1,000 | 0,681 | 0,484 | 0,273 | 0,349 |
| ML+Rules-override | 0,561 | 1,000 | 0,719 | 0,518 | 0,800 | 0,629 |
| **ML+Rules-feature** | **1,000** | **0,969** | **0,984** | **0,593** | **0,927** | **0,723** |

### 6.4.3 Discussão

A configuração ML+Rules-feature atinge **F1=0,984 na classe cancelamento** (precision=1,000, recall=0,969), ligeiramente acima do ML isolado (F1=0,968, recall=0,938). A adição da feature binária `rule_cancel` permite que o LightGBM recupere 1 caso de cancelamento que o ML sozinho errava, sem introduzir falsos positivos.

Cabe notar que a classe cancelamento contém apenas 32 amostras no conjunto de teste, o que limita a generalização deste resultado. O Macro-F1 global do ML+Rules-feature (0,654) é ligeiramente inferior ao ML-only (0,659), uma diferença de -0,5pp. Com context windows, o efeito das regras é diluído: as regras lexicais disparam em janelas intermediárias onde o contexto não sustenta plenamente o intent, e a agregação por média de probabilidades absorve esse ruído.

A estratégia de **override** — onde a regra substitui a predição do ML quando dispara — revelou-se significativamente prejudicial (Macro-F1 0,574 vs 0,659 do ML-only, p ≈ 10⁻⁷). Com context windows, o efeito é mais pronunciado que com conversas completas: regras que avaliam janelas parciais geram mais falsos positivos, pois o termo "cancelar" aparece em janelas de contexto de outras classes (e.g., turnos de esclarecimento onde o agente menciona cancelamento).

A estratégia **rules-as-feature** delega a decisão final ao classificador, que aprende a ponderar o sinal da regra em contexto. No entanto, com a granularidade de janelas, o benefício é modesto: o classificador já dispõe de embeddings ricos que capturam o mesmo sinal semântico que as regras tentam codificar lexicalmente.

**Teste estatístico:** O teste de Wilcoxon signed-rank entre ML-only e ML+Rules-feature obteve p=0,4669 (não significativo a α=0,05). O intervalo de confiança bootstrap de 95% para a diferença de acurácia é [−0,015; +0,033], contendo zero. O tamanho de efeito (r=0,18) é pequeno.

**Veredicto H3:** *Refutada.* Nenhuma estratégia de integração de regras supera o ML isolado com significância estatística. A configuração ML+Rules-feature apresenta Macro-F1=0,654, ligeiramente inferior ao ML-only (0,659), com p=0,4669. A estratégia de override é significativamente prejudicial (Macro-F1=0,574, p < 10⁻⁶). Com context windows, as regras lexicais perdem eficácia porque operam sobre janelas parciais onde keywords podem ocorrer sem contexto suficiente. A contribuição qualitativa permanece: a arquitetura rules-as-feature não degrada o desempenho, e em cenários com regras mais sofisticadas (semânticas, contextuais), o benefício pode ser maior. A estratégia de override é definitivamente contraindicada.

---

## 6.5 Resultados da Inferência em Cascata (H4)

**Hipótese H4:** *A inferência em cascata permite reduzir o custo computacional com degradação controlada da qualidade.*

### 6.5.1 Resultados Gerais

A Tabela 6.8 apresenta os resultados da cascata com dois estágios: Estágio 1 (LogReg, classificador linear rápido) e Estágio 2 (LightGBM, classificador ensemble completo). Ambos utilizam features lexicais + embeddings.

**Tabela 6.8** — Resultados da inferência em cascata (H4). Multi-seed (5 seeds), avaliação por conversa com janelas de contexto.

| Configuração | Macro-F1 | % Estágio 1 | Custo (ms) | Δ F1 vs uniform |
|---|---|---|---|---|
| **Uniforme (baseline)** | **0,659** | 0% | 110 | — |
| Cascade t=0,50 | 0,609 | 46,1% | 178 | −0,050 |
| Cascade t=0,60 | 0,640 | 31,9% | 194 | −0,019 |
| Cascade t=0,70 | 0,653 | 19,9% | 207 | −0,006 |
| Cascade t=0,80 | 0,657 | 10,2% | 218 | −0,002 |
| Cascade t=0,90 | 0,659 | 4,4% | 224 | 0,000 |

*Nota: O baseline uniforme utiliza o mesmo LightGBM de H2 (100 árvores, 31 folhas). A cascata opera por janela: cada janela é avaliada pelo Estágio 1 (LogReg) e, se abaixo do threshold de confiança, encaminhada ao Estágio 2 (LightGBM). O threshold foi selecionado por validação (melhor: t=0,70).*

### 6.5.2 Análise de Custo

O custo por janela medido revela:
- LogReg (estágio leve): 0,087 ms/janela
- LightGBM (estágio completo): 0,081 ms/janela

A razão de custo é de ~1,1×, essencialmente idêntica. Com janelas de contexto, ambos os classificadores recebem os mesmos embeddings pré-computados; a inferência é dominada pela passagem pelo modelo, não pela complexidade do modelo em si. Em cenário de produção, a diferença seria amplificada quando o estágio leve usasse regras lexicais simples (μs) sem necessidade de embeddings, enquanto o estágio completo requer geração de embeddings via modelo neural (~100 ms por janela).

### 6.5.3 Discussão

O threshold t=0,90 resolve 4,4% das janelas no estágio leve com degradação nula em Macro-F1 (0,659 vs 0,659). O threshold t=0,70, selecionado por validação, resolve 19,9% das janelas com degradação de apenas 0,6pp. Essas são janelas com padrões lexicais altamente discriminativos — tipicamente janelas iniciais de conversas de cancelamento ou saudação — onde o LogReg atinge confiança elevada.

Nenhuma configuração de cascata alcança redução de custo no cenário experimental: todas apresentam custo total **superior** ao baseline uniforme, porque o estágio leve tem custo essencialmente idêntico ao estágio completo (~0,08 ms/janela para ambos). O overhead de executar o estágio leve em janelas que são subsequentemente encaminhadas ao estágio completo supera qualquer economia. Em cenário de produção, onde o estágio leve utilizaria regras determinísticas (sem embeddings, custo ~μs) e o estágio completo exigiria inferência neural (~100 ms/janela), a cascata ofereceria benefícios de custo proporcionalmente maiores.

**Veredicto H4: Refutada.** Nenhuma configuração de cascata atingiu a redução de custo de 40% estabelecida como critério. A razão de custo entre os estágios (~1,1×) foi insuficiente para compensar o overhead de execução dupla. A premissa motivacional de H4 — estágios com custos radicalmente distintos — não se materializou na implementação experimental, onde ambos os estágios utilizam os mesmos embeddings pré-computados e operam sobre janelas de contexto idênticas. O resultado positivo é a degradação mínima de F1 (Δ ≤ 0,006 para t ≥ 0,70), demonstrando que o roteamento por confiança preserva qualidade. Em arquiteturas de produção com estágio leve genuinamente barato (filtros lexicais sem embeddings), a hipótese permanece plausível mas não verificada neste trabalho.

---

## 6.6 Estudos de Ablação

Para quantificar a contribuição individual de cada família de features ao pipeline completo, conduzimos um estudo de ablação sistemático. O baseline é o LightGBM com todas as features (lexicais + estruturais + embeddings + regras, Macro-F1=0,654), avaliado com 5 seeds e agregação janela→conversa.

### 6.6.1 Resultados

**Tabela 6.9** — Estudo de ablação: impacto da remoção de cada componente.

| Configuração | Macro-F1 | Accuracy | Δ F1 vs Full | Interpretação |
|---|---|---|---|---|
| -Rules | **0,659** | **0,722** | **+0,5pp** | Regras prejudicam levemente |
| **Full pipeline** | **0,654** | **0,713** | baseline | Todas as features |
| -Structural | 0,641 | 0,701 | -1,3pp | Structural contribui modestamente |
| -Lexical | 0,634 | 0,701 | -2,0pp | Lexical contribui +2,0pp |
| Emb-only | 0,631 | 0,710 | -2,3pp | Embeddings quase igualam full |
| -Embeddings | 0,396 | 0,426 | **-25,8pp** | Colapso sem embeddings |
| Lexical-only | 0,334 | 0,364 | -32,0pp | Insuficiente isoladamente |

### 6.6.2 Discussão

A ablação revela uma hierarquia clara de importância dos componentes:

1. **Embeddings são indispensáveis** (contribuição: +25,8pp). Sua remoção causa colapso do Macro-F1 de 0,654 para 0,396 — uma degradação de 39%. Os embeddings pré-treinados fornecem a representação semântica que permite ao classificador distinguir intents com vocabulário sobreposto, mesmo quando avaliados por janela de contexto.

2. **Features lexicais contribuem moderadamente** (+2,0pp). Contadores lexicais (word count, question marks, uppercase ratio) adicionam informação complementar sobre estrutura do texto que os embeddings não capturam diretamente.

3. **Features estruturais contribuem modestamente** (+1,3pp). Com janelas de contexto, as features `turn_count`, `speaker_count` e flags de role agora apresentam variância real (janelas podem ter diferentes composições de falantes), o que as torna marginalmente informativas — em contraste com a avaliação anterior onde eram quase-constantes.

4. **Features de regras são levemente prejudiciais** (-0,5pp quando presentes). Com context windows, regras lexicais disparam em janelas parciais onde o contexto não sustenta o intent, introduzindo ruído. A configuração `-Rules` (Macro-F1=0,659) é a melhor da ablação — resultado consistente com os achados da H3.

O achado de que **Emb-only (F1=0,631)** aproxima-se do **full pipeline (F1=0,654)** sugere que, em cenários onde simplicidade é prioritária, embeddings pré-treinados com LightGBM constituem uma solução competitiva com apenas 384 features.

---

## 6.7 Comparação com Trabalhos Correlatos

A Tabela 6.10 situa os resultados do TalkEx no contexto da literatura revisada no Capítulo 3.

**Nota metodológica:** Os valores apresentados na Tabela 6.10 não são diretamente comparáveis entre sistemas. BERTaú avalia MRR em tarefa de FAQ retrieval com pares pergunta-resposta manuais, enquanto TalkEx avalia MRR em classificação por similaridade com ground truth definido por label compartilhado. Harris (2025) e Rayo (2025) operam em domínios e idiomas distintos. A tabela tem propósito de contextualização dimensional, não de ranking de desempenho.

**Tabela 6.10** — Comparação contextualizada com trabalhos correlatos.

| Dimensão | TalkEx | BERTaú (Finardi, 2021) | Harris (2025) | Rayo (COLING, 2025) |
|---|---|---|---|---|
| **Retrieval (melhor)** | MRR 0,826 (Hybrid-RRF) | MRR 0,552 (pairwise) | — | Recall@10 0,833 |
| **Classificação** | F1 0,659 (9 classes, janelas) | — | — | — |
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
| H1: Retrieval híbrido | Refutada no critério primário; confirmada no secundário | MRR +3% (p=0,103 vs BM25); significativo vs ANN (p=0,032) |
| H2: Representação multi-nível | Confirmada | F1 +97% com embeddings (+0,325 absolutos); 5/5 comparações significativas (Wilcoxon) |
| H3: Regras + ML | Refutada | ML-only (0,659) ≥ ML+Rules-feature (0,654), p=0,467; override prejudicial (0,574) |
| H4: Inferência cascata | Refutada | Custo ratio ~1,1×; nenhuma redução de custo; Δ F1 ≤ 0,006 para t ≥ 0,70 |

### 6.8.2 Resultados Inesperados

Quatro resultados merecem destaque por divergirem das expectativas iniciais:

1. **A força do BM25.** Esperávamos que o retrieval semântico superasse o lexical por margem significativa. O BM25 atingiu MRR 0,802 — apenas 0,024 abaixo do híbrido — sugerindo que conversas de atendimento possuem sinais lexicais mais fortes do que o antecipado. Isso tem implicação prática direta: para organizações que não podem investir em infraestrutura de embeddings, BM25 é uma alternativa viável.

2. **O efeito prejudicial do override com context windows.** A regra-como-override reduziu o Macro-F1 de 0,659 para 0,574 (−8,5pp). Com janelas de contexto, o efeito é mais pronunciado do que com conversas completas: regras lexicais avaliam janelas parciais onde keywords podem ocorrer sem que o intent esteja presente, gerando falsos positivos amplificados pela fragmentação do texto.

3. **Regras como features são neutras, não benéficas.** Esperávamos ganhos em classes críticas. Com janelas de contexto, o classificador com embeddings já captura os mesmos sinais que as regras tentam codificar. A ablação confirma: remover regras **melhora** o F1 em 0,5pp. Este resultado demonstra que, quando a representação semântica é suficientemente rica, regras lexicais simples são redundantes.

4. **Features estruturais passam de prejudiciais a contributivas.** Na avaliação anterior (conversas completas), features estruturais tinham variância quase zero. Com context windows, `turn_count`, `speaker_count` e flags de role variam entre janelas, contribuindo +1,3pp — uma reversão do achado anterior que demonstra como a granularidade de análise afeta a utilidade das features.

### 6.8.3 Implicações Práticas

Os resultados sugerem um pipeline prático para classificação de conversas de atendimento:

1. **Embeddings pré-treinados multilinguais** como representação base — sem necessidade de fine-tuning ou GPU. A ablação demonstrou que embeddings contribuem +25,8pp, sendo o componente mais crítico.
2. **LightGBM** como classificador — treinamento em ~6 segundos em CPU, inferência em <0,1ms por janela. Supera LogReg (+20%) e MLP (+12%) consistentemente.
3. **Features lexicais como complemento** — contadores lexicais (word count, question marks, uppercase ratio) contribuem +2,0pp adicionais, com custo de extração negligível.
4. **Regras determinísticas com cautela.** A estratégia rules-as-feature não degradou significativamente o desempenho, mas também não o melhorou (Macro-F1 0,654 vs 0,659 sem regras). Regras lexicais simples são redundantes quando embeddings ricos estão disponíveis. Regras devem ser reservadas para requisitos de compliance e auditabilidade — não como mecanismo de melhoria de classificação. A estratégia de override é definitivamente contraindicada.
5. **BM25 para retrieval** quando infraestrutura é limitada — competitivo com híbrido neste domínio (MRR 0,802 vs 0,826, diferença não significativa).
6. **Cascata** reservada para cenários onde os estágios têm custo genuinamente diferente (e.g., filtros lexicais sem embeddings vs inferência neural com embeddings).

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
