# Capítulo 5 — Desenho Experimental e Metodologia

Este capítulo descreve com rigor metodológico o protocolo experimental adotado para validar as quatro hipóteses da dissertação. Detalhamos o dataset utilizado, as métricas de avaliação, os protocolos específicos para cada hipótese, a análise estatística empregada e as ameaças à validade identificadas. O nível de detalhe visa garantir a reprodutibilidade integral dos experimentos por pesquisadores independentes.

---

## 5.1 Dataset

### 5.1.1 Fontes de Dados

A avaliação experimental requer um corpus de conversas de atendimento em português brasileiro com anotações de intent, sentimento e metadados estruturais. Diante da escassez de datasets conversacionais abertos em PT-BR — uma lacuna reconhecida no campo de processamento de linguagem natural para o português —, adotamos uma estratégia de expansão sintética controlada a partir de um dataset público existente.

**Corpus base.** Utilizamos como ponto de partida o dataset `RichardSakaguchiMS/brazilian-customer-service-conversations`, disponível na plataforma HuggingFace sob licença Apache 2.0. Este corpus contém 944 conversas de atendimento ao cliente em português brasileiro, anotadas com intent e sentimento. O dataset original, embora valioso como ponto de partida, apresenta limitações que comprometem sua adequação direta para os experimentos propostos: (i) distribuição de classes aproximadamente uniforme (~11% por classe), distante da distribuição naturalmente desbalanceada observada em operações reais de atendimento; (ii) baixa variabilidade no número de turnos por conversa, com aproximadamente 90% das conversas contendo exatamente 8 turnos; (iii) estilo lexical homogêneo, sem variação de personas ou registros linguísticos; e (iv) distribuição de sentimentos desvinculada do intent, com proporções uniformes de 33% para cada polaridade independentemente da classe.

**Expansão sintética controlada.** Para superar essas limitações, realizamos uma expansão sintética do corpus utilizando geração via modelo de linguagem de grande porte (Claude Sonnet, Anthropic) em modo batch offline. A expansão foi conduzida com controles rigorosos sobre as variáveis de geração, conforme descrevemos a seguir:

- *Variabilidade de turnos:* as conversas geradas seguem uma distribuição log-normal com média de 8 turnos e desvio padrão de 4, produzindo conversas com 4 a 20 turnos — refletindo a variabilidade natural observada em operações de atendimento.
- *Distribuição de classes:* adotamos uma distribuição desbalanceada intencional que reproduz padrões típicos de contact centers, onde reclamações e dúvidas predominam sobre elogios e saudações.
- *Variabilidade lexical:* cada conversa gerada é condicionada a uma entre cinco personas linguísticas — formal, informal, irritado, idoso e jovem/gírias — introduzindo diversidade de registro e vocabulário.
- *Sentimento condicionado ao intent:* a distribuição de sentimentos é condicionada à classe do intent. Por exemplo, conversas de reclamação apresentam 65% de sentimento negativo e 35% neutro, enquanto elogios apresentam 75% de sentimento positivo e 25% neutro. Esse condicionamento reflete a correlação natural entre intenção e polaridade emocional.

É fundamental registrar que o uso de dados sintéticos impõe uma limitação epistemológica ao estudo: as conclusões derivadas deste corpus são válidas dentro do escopo metodológico definido, não podendo ser generalizadas irrestritamente para dados conversacionais reais. Mitigamos essa limitação por meio do protocolo de validação de robustez descrito na Seção 5.1.4.

**Investigação de fontes alternativas.** Investigamos a possibilidade de utilizar dados do portal Consumidor.gov.br como fonte de validação externa. A análise revelou que os dados disponíveis em formato CSV contêm apenas campos categóricos (Área, Assunto, Grupo Problema, Problema), sem texto livre conversacional. Ademais, a plataforma é exclusivamente voltada a reclamações, impossibilitando o mapeamento de intents como elogio, saudação ou dúvida. Não identificamos nenhum outro dataset público em PT-BR que combine conversas multi-turn com anotação de intents de atendimento e texto livre. Registramos que a validação com dados operacionais reais constitui trabalho futuro e requer parceria com operadores de contact center.

### 5.1.2 Especificação do Corpus Primário

A Tabela 5.1 sumariza as características do corpus expandido utilizado nos experimentos.

**Tabela 5.1** — Especificação do corpus primário expandido.

| Dimensão | Valor |
|---|---|
| Total de conversas | ~3.500 |
| Turnos por conversa | 4–20 (distribuição log-normal, $\mu=8$, $\sigma=4$) |
| Classes (intents) | 9 |
| Idioma | PT-BR (informal, com diacríticos e gírias) |
| Anotação por conversa | Intent + sentimento + setor |
| Split de treino | 70% (~2.450 conversas) |
| Split de validação | 15% (~525 conversas) |
| Split de teste | 15% (~525 conversas) |
| Seed para splits | 42 |

O corpus compreende 9 classes de intent, mantidas do dataset original. A Tabela 5.2 apresenta a distribuição-alvo de intents no corpus expandido, projetada para refletir o desbalanceamento natural de operações de atendimento.

**Tabela 5.2** — Distribuição-alvo de intents no corpus expandido.

| Intent | Proporção | Conversas (aprox.) |
|---|---|---|
| reclamacao | 20% | 700 |
| duvida_produto | 18% | 630 |
| duvida_servico | 17% | 595 |
| suporte_tecnico | 15% | 525 |
| compra | 10% | 350 |
| cancelamento | 8% | 280 |
| saudacao | 5% | 175 |
| elogio | 4% | 140 |
| outros | 3% | 105 |

Essa distribuição desbalanceada é metodologicamente relevante para dois objetivos: (i) avaliar o comportamento dos classificadores diante de classes minoritárias, cenário frequente em aplicações reais; e (ii) testar a eficácia das regras determinísticas (H3) em classes críticas de baixa frequência como `cancelamento` e `elogio`.

### 5.1.3 Validação de Dificuldade do Dataset (Fase 0.5)

Antes da execução dos experimentos propriamente ditos, conduzimos uma fase preliminar (denominada Fase 0.5) para verificar que o dataset apresenta dificuldade genuína para as tarefas propostas. Um dataset trivialmente separável invalidaria os experimentos comparativos, uma vez que qualquer abordagem — isolada ou híbrida — alcançaria desempenho próximo ao teto. A Tabela 5.3 apresenta as verificações realizadas e seus critérios de aprovação.

**Tabela 5.3** — Verificações de dificuldade do dataset (Fase 0.5).

| Verificação | Métrica | Critério de aprovação |
|---|---|---|
| Baseline de maioria | Acurácia da classe mais frequente | < 25% |
| Exclusividade lexical | Score médio de exclusividade por intent | $1{,}0 < s < 3{,}0$ |
| Overlap lexical | % de intents com cross-intent word overlap | > 50% |
| Separação de embeddings | Razão inter/intra-classe (cosseno) | < 2,0 |
| Leakage few-shot | Contaminação entre train/test via few-shot IDs | 0% no test set |
| Original vs expandido | Divergência de distribuição (turnos, palavras) | Sem desvio estatisticamente significativo |

Os resultados preliminares sobre o corpus original (944 conversas) indicam: exclusividade lexical de 1,68 (dentro da faixa desejada), overlap entre intents de 100% (vocabulário fortemente compartilhado entre classes) e razão de desbalanceamento de 1,2. Esses valores sugerem que o dataset possui dificuldade adequada para a avaliação comparativa, uma vez que a sobreposição lexical impede a resolução trivial do problema por métodos puramente baseados em palavras-chave.

A verificação de divergência entre o corpus original e o expandido é particularmente relevante: caso as distribuições de comprimento de turnos e de palavras por conversa apresentem desvios estatisticamente significativos, seria indicativo de que a expansão sintética introduziu artefatos. Aplicamos testes de Kolmogorov-Smirnov para verificar essa condição.

### 5.1.4 Validação de Robustez

Para garantir que as conclusões experimentais não são artefatos da expansão sintética, todos os experimentos H1–H4 são replicados integralmente no dataset original de 944 conversas (sem expansão). Esse protocolo de validação cruzada permite duas inferências:

- Se as conclusões se mantêm em ambos os corpora, a expansão não introduziu viés significativo e os resultados são robustos.
- Se houver divergência, reportamos ambos os resultados e discutimos as causas potenciais, distinguindo efeitos do tamanho amostral de artefatos de geração.

### 5.1.5 Pré-processamento

O pré-processamento do corpus segue cinco etapas sequenciais:

1. **Normalização textual.** Aplicamos a função `normalize_for_matching()` do módulo `talkex.text_normalization`, que realiza conversão para minúsculas e remoção de diacríticos via decomposição Unicode NFD. Essa normalização é essencial para o português brasileiro, onde variações como "não"/"nao" e "cancelamento"/"cancelámento" devem ser tratadas como equivalentes para fins de matching lexical.

2. **Segmentação de turnos.** O dataset já contém turnos estruturados com atribuição de falante. Quando necessário, aplicamos heurística de alternância de falantes para reconstruir a segmentação.

3. **Construção de janelas de contexto.** Aplicamos janelas deslizantes (sliding windows) com parâmetros configuráveis de tamanho, passo (stride) e alinhamento por falante, conforme descrito na Seção 5.4.

4. **Geração de embeddings.** Geramos representações vetoriais em múltiplos níveis de granularidade (turno, janela, conversa) utilizando os modelos candidatos. Os embeddings são persistidos em cache para garantir reprodutibilidade e eficiência computacional.

5. **Indexação.** Construímos índices paralelos para retrieval lexical (BM25, via biblioteca rank-bm25) e retrieval semântico (ANN, via FAISS com índice IVF).

### 5.1.6 Splits e Reprodutibilidade

Adotamos seed fixo (42) para todas as operações que envolvem aleatoriedade: particionamento dos dados, embaralhamento durante treinamento e inicialização de modelos. A partição é estratificada, preservando a distribuição de classes em cada split (treino 70%, validação 15%, teste 15%).

Quando disponíveis, utilizamos timestamps para holdout temporal — o conjunto de treino contém conversas mais antigas e o conjunto de teste conversas mais recentes —, simulando o cenário real de implantação onde o modelo é treinado com dados históricos e avaliado em dados futuros. Na ausência de timestamps, recorremos à partição estratificada aleatória.

Complementarmente, empregamos validação cruzada estratificada de 5 folds para estimar a variância dos resultados, conforme detalhado na Seção 5.7.

---

## 5.2 Métricas de Avaliação

A avaliação do sistema proposto requer métricas distintas para cada componente da arquitetura. Organizamos as métricas em quatro famílias, correspondentes às dimensões avaliadas pelos experimentos H1–H4.

### 5.2.1 Métricas de Retrieval (H1)

Para avaliar a qualidade do retrieval híbrido, utilizamos cinco métricas consagradas na literatura de recuperação de informação (Manning et al., 2008; Robertson & Zaragoza, 2009). A Tabela 5.4 sumariza as métricas adotadas.

**Tabela 5.4** — Métricas de avaliação para retrieval.

| Métrica | Definição | Justificativa |
|---|---|---|
| Recall@K | Fração dos documentos relevantes recuperados no top-K | Mede cobertura, essencial para pipelines downstream que dependem da qualidade do retrieval |
| Precision@K | Fração dos documentos no top-K que são relevantes | Mede qualidade do conjunto recuperado |
| MRR | Mean Reciprocal Rank — média de $1/r_i$, onde $r_i$ é a posição do primeiro resultado relevante para a query $i$ | Mede quão rapidamente o sistema retorna o primeiro resultado útil |
| nDCG@K | Normalized Discounted Cumulative Gain | Mede a qualidade do ranking considerando a posição relativa dos documentos relevantes |
| MAP@K | Mean Average Precision | Mede a precisão média ao longo de todo o ranking |

Avaliamos todas as métricas para $K \in \{5, 10, 20\}$. A escolha desses valores de $K$ reflete cenários práticos: $K=5$ representa um cenário restritivo onde apenas os resultados mais relevantes são apresentados; $K=10$ corresponde ao cenário padrão em interfaces de busca; e $K=20$ captura cenários de recall ampliado para pipelines downstream.

### 5.2.2 Métricas de Classificação (H2, H3)

Para avaliar os classificadores, adotamos um conjunto de métricas que captura tanto o desempenho global quanto o comportamento por classe, com atenção especial à calibração dos scores de confiança. A Tabela 5.5 apresenta as métricas utilizadas.

**Tabela 5.5** — Métricas de avaliação para classificação.

| Métrica | Definição | Justificativa |
|---|---|---|
| Macro-F1 | Média aritmética do F1 por classe, com ponderação igualitária | Sensível ao desempenho em classes raras; métrica primária |
| Micro-F1 | F1 calculado sobre todos os exemplos, ponderando por volume | Reflete o desempenho global considerando o desbalanceamento |
| Precision por classe | Proporção de predições positivas corretas por classe | Mede o custo de falsos positivos, especialmente relevante para classes críticas |
| Recall por classe | Proporção de exemplos positivos corretamente identificados | Mede cobertura, relevante para detecção de intents raros |
| AUC-ROC | Área sob a curva ROC, medindo separabilidade | Robusta à escolha de threshold de decisão |
| ECE | Expected Calibration Error (Guo et al., 2017) | Mede a confiabilidade dos scores de confiança — um score de 0,9 deve corresponder a 90% de acerto |

A escolha do Macro-F1 como métrica primária é deliberada: em um corpus desbalanceado, o Micro-F1 pode mascarar baixo desempenho em classes minoritárias. O ECE é particularmente relevante para o experimento H4 (cascata), onde decisões de roteamento entre estágios dependem da calibração dos scores de confiança.

### 5.2.3 Métricas de Regras (H3)

A avaliação do motor de regras demanda métricas específicas que capturem não apenas a acurácia das regras, mas também seu impacto operacional. A Tabela 5.6 apresenta essas métricas.

**Tabela 5.6** — Métricas de avaliação para o motor de regras.

| Métrica | Definição |
|---|---|
| Precision da regra | Proporção de acionamentos corretos sobre o total de acionamentos |
| Recall da regra | Proporção de casos reais capturados pela regra |
| False Positive Burden | Número de falsos positivos por 1.000 conversas processadas |
| Cobertura | Percentual de conversas em que ao menos uma regra produziu evidência |
| Latência por regra | Tempo médio de avaliação de cada regra, em milissegundos |

O *False Positive Burden* é uma métrica operacional: em ambientes de atendimento, cada falso positivo em classes como `compliance` ou `fraude` pode desencadear processos de investigação custosos. A latência por regra é relevante para validar que a adição de regras ao pipeline não compromete os requisitos de tempo de resposta.

### 5.2.4 Métricas de Eficiência (H4)

Para avaliar a inferência em cascata, definimos métricas que capturam o trade-off entre custo computacional e qualidade de classificação. A Tabela 5.7 apresenta essas métricas.

**Tabela 5.7** — Métricas de avaliação para eficiência.

| Métrica | Definição |
|---|---|
| Custo por conversa | Tempo total de processamento em milissegundos (CPU/GPU) |
| Throughput | Conversas processadas por segundo |
| Latência p50/p95/p99 | Percentis da distribuição de latência por conversa |
| % resolvido por estágio | Proporção de conversas que terminam o processamento em cada estágio da cascata |
| $\Delta$F1 | Diferença de Macro-F1 entre o pipeline uniforme e o pipeline cascateado |

A combinação de custo por conversa e $\Delta$F1 permite construir a fronteira de Pareto que é central para a avaliação de H4: identificar configurações que reduzem custo com degradação aceitável de qualidade.

---

## 5.3 Protocolo Experimental para H1 — Retrieval Híbrido

### 5.3.1 Hipótese

> *O retrieval híbrido (BM25 + ANN com fusão de scores) supera tanto BM25 isolado quanto busca semântica isolada em Recall@K, MRR e nDCG quando aplicado a conversas de call center em português brasileiro.*

Esta hipótese fundamenta-se na complementaridade teórica entre busca lexical e semântica, amplamente documentada na literatura (Rayo et al., 2025; Lin et al., 2021). A busca lexical (BM25) é eficaz para termos exatos, nomes de produtos, códigos e palavras-chave de compliance, enquanto a busca semântica excele na captura de paráfrases, intenção implícita e variação linguística. A hipótese postula que a combinação supera ambas as abordagens isoladas no domínio conversacional.

### 5.3.2 Sistemas Comparados

Definimos sete sistemas de retrieval para comparação, organizados em três categorias: lexicais, semânticos e híbridos. A Tabela 5.8 descreve cada sistema.

**Tabela 5.8** — Sistemas de retrieval comparados no experimento H1.

| Sistema | Categoria | Descrição |
|---|---|---|
| BM25-base | Lexical | BM25 vanilla com lowercase e remoção de stopwords |
| BM25-norm | Lexical | BM25 com normalização accent-aware (strip\_accents + remoção de pontuação) |
| ANN-E5 | Semântico | Busca por vizinhos mais próximos com embeddings E5-base (Wang et al., 2022) |
| ANN-BGE | Semântico | Busca por vizinhos mais próximos com embeddings BGE-small (Xiao et al., 2023) |
| Hybrid-linear | Híbrido | BM25-norm + melhor ANN, fusão linear: $S = \alpha \cdot s_{sem} + (1-\alpha) \cdot s_{lex}$ |
| Hybrid-RRF | Híbrido | BM25-norm + melhor ANN, Reciprocal Rank Fusion (Cormack et al., 2009) |
| Hybrid-rerank | Híbrido | Hybrid-linear + cross-encoder reranking sobre os top-50 candidatos |

A inclusão de BM25-base e BM25-norm como sistemas separados permite isolar o efeito da normalização de diacríticos — uma decisão de design específica para PT-BR implementada no módulo `talkex.text_normalization`. A seleção dos modelos E5 e BGE segue a recomendação da literatura recente: E5 foi projetado especificamente para tarefas de retrieval (Wang et al., 2022), enquanto BGE oferece um equilíbrio entre desempenho e eficiência computacional (Xiao et al., 2023).

### 5.3.3 Espaço de Parâmetros

Para os sistemas híbridos, variamos os parâmetros apresentados na Tabela 5.9.

**Tabela 5.9** — Parâmetros variados no experimento H1.

| Parâmetro | Valores | Justificativa |
|---|---|---|
| $\alpha$ (peso semântico) | 0,3; 0,5; 0,65; 0,7; 0,8 | Faixa centrada em 0,65, valor ótimo reportado por Rayo et al. (2025) para textos regulatórios |
| $K$ (top-K por estágio) | 10, 20, 50, 100 | Avaliação do efeito do tamanho do conjunto candidato |
| BM25 $k_1$ | 1,2; 1,5; 2,0 | Controle da saturação de frequência de termos (Robertson et al., 1996) |
| BM25 $b$ | 0,5; 0,75; 1,0 | Controle da normalização por comprimento do documento |

A escolha de $\alpha = 0{,}65$ como valor central é motivada pelos resultados de Rayo et al. (2025), que reportaram esse valor como ótimo para textos regulatórios em espanhol. Incluímos valores menores (0,3; 0,5) e maiores (0,7; 0,8) para explorar se o domínio conversacional em PT-BR apresenta ponto ótimo diferente.

### 5.3.4 Construção do Ground Truth

A avaliação de retrieval requer um conjunto de queries com documentos relevantes anotados. Adotamos a seguinte estratégia de construção:

1. **Queries:** utilizamos os intents da taxonomia (ex.: "cancelamento", "reclamação de cobrança") como queries de busca.
2. **Documentos:** turnos individuais ou janelas de contexto do corpus constituem o espaço de documentos.
3. **Relevância:** definida pela correspondência entre a classe do intent da query e a anotação da conversa de origem do turno/janela. Essa abordagem permite construção automática do ground truth sem anotação manual adicional.

### 5.3.5 Protocolo de Execução

O protocolo segue os seguintes passos:

Para cada sistema $S \in \{$BM25-base, BM25-norm, ANN-E5, ANN-BGE, Hybrid-linear, Hybrid-RRF, Hybrid-rerank$\}$:

1. Para cada query $Q$ do conjunto de avaliação, executar o retrieval com $S$ e obter os top-$K$ resultados.
2. Calcular Recall@$K$, Precision@$K$, MRR, nDCG@$K$ e MAP@$K$.
3. Agregar métricas (média $\pm$ desvio padrão) sobre todas as queries.
4. Repetir com 5 seeds diferentes para estimar a variância devida à aleatoriedade da indexação ANN.

Para o sistema Hybrid-linear especificamente, variamos $\alpha \in \{0{,}3; 0{,}5; 0{,}65; 0{,}7; 0{,}8\}$ e plotamos a curva $\alpha$ versus Recall@10 para determinar o ponto ótimo.

### 5.3.6 Análise Prevista

A análise dos resultados de H1 compreende:

- **Tabela comparativa principal:** sistema $\times$ métricas, no formato estabelecido por Rayo et al. (2025, Table 2).
- **Gráfico de Recall@K:** curvas de Recall@$K$ para $K = 5, 10, 20$ por sistema, permitindo visualizar a superioridade relativa em diferentes pontos de corte.
- **Curva de $\alpha$:** $\alpha$ versus Recall@10 para o sistema híbrido com fusão linear, identificando o ponto ótimo e a sensibilidade do sistema a esse parâmetro.
- **Análise qualitativa:** seleção de exemplos de queries onde o sistema híbrido acerta e os sistemas isolados erram, e vice-versa, para compreender os mecanismos de complementaridade.
- **Teste estatístico:** Wilcoxon signed-rank test entre o melhor sistema híbrido e o melhor sistema isolado, com nível de significância $\alpha_{stat} = 0{,}05$.

### 5.3.7 Critérios de Confirmação

H1 é **confirmada** se o melhor sistema híbrido supera todos os sistemas isolados em Recall@10 e MAP@10, com diferença estatisticamente significativa ($p < 0{,}05$).

H1 é **parcialmente confirmada** se o sistema híbrido supera em algumas métricas mas não em todas, ou se a diferença não atinge significância estatística.

H1 é **refutada** se um sistema isolado (BM25 ou ANN) supera ou empata com o melhor sistema híbrido nas métricas primárias.

---

## 5.4 Protocolo Experimental para H2 — Representação Multi-Nível

### 5.4.1 Hipótese

> *Classificadores que utilizam features em múltiplos níveis de granularidade (turno + janela de contexto + conversa) alcançam F1 superior a classificadores que operam em um único nível de representação.*

Esta hipótese fundamenta-se na observação de que intents conversacionais operam em granularidades distintas: alguns intents são identificáveis a partir de um único turno (ex.: "quero a segunda via da fatura"), enquanto outros emergem apenas do contexto multi-turn (ex.: objeção após oferta de retenção) ou do arco narrativo da conversa completa. Trabalhos como Lyu et al. (2025) demonstraram que mecanismos de atenção sobre representações contextualizadas melhoram a classificação de textos, motivando a exploração de representações multi-nível no domínio conversacional.

### 5.4.2 Representações Comparadas

Definimos nove configurações de representação, variando os níveis de granularidade e os tipos de features. A Tabela 5.10 descreve cada configuração.

**Tabela 5.10** — Configurações de representação comparadas no experimento H2.

| Configuração | Features de entrada |
|---|---|
| Turn-only | Embedding do turno individual |
| Window-only | Embedding da janela de contexto (5 turnos) |
| Conv-only | Embedding da conversa completa |
| Turn+Window | Concatenação dos embeddings de turno e janela |
| Turn+Conv | Concatenação dos embeddings de turno e conversa |
| Multi-level | Concatenação dos embeddings de turno, janela e conversa |
| Multi-level+lex | Multi-level + features lexicais (TF-IDF, scores BM25 contra protótipos de classe) |
| Multi-level+struct | Multi-level + features estruturais (speaker, posição do turno, duração) |
| Full | Multi-level + features lexicais + estruturais + contextuais |

As configurações Turn-only, Window-only e Conv-only servem como baselines de nível único. As combinações intermediárias (Turn+Window, Turn+Conv) permitem avaliar o ganho marginal de cada nível adicional. A configuração Full representa o sistema completo com todas as famílias de features.

### 5.4.3 Classificadores

Para cada representação, treinamos e avaliamos três classificadores, selecionados por suas características complementares. A Tabela 5.11 apresenta os classificadores e suas justificativas.

**Tabela 5.11** — Classificadores utilizados no experimento H2.

| Classificador | Justificativa |
|---|---|
| Logistic Regression | Baseline linear, interpretável, adequado para avaliar a separabilidade das features (scikit-learn) |
| LightGBM | Gradient boosting otimizado para features heterogêneas (numéricas + categóricas), robusto a escalas distintas (Ke et al., 2017) |
| MLP (2 camadas) | Baseline neural para features densas, capaz de capturar interações não-lineares (scikit-learn) |

A Logistic Regression serve como baseline linear: se uma representação não melhora o desempenho com regressão logística, a melhoria provavelmente não é robusta. O LightGBM é particularmente adequado para o cenário Full, onde features de naturezas distintas (embeddings contínuos, scores BM25, flags binários, metadados categóricos) coexistem. O MLP complementa a análise ao avaliar se interações não-lineares entre features oferecem ganho adicional.

### 5.4.4 Parâmetros de Janela de Contexto

A janela de contexto é um componente central da representação multi-nível. Variamos seus parâmetros conforme a Tabela 5.12.

**Tabela 5.12** — Parâmetros de janela de contexto variados no experimento H2.

| Parâmetro | Valores | Justificativa |
|---|---|---|
| Tamanho da janela (turnos) | 3, 5, 7, 10 | Explorar o trade-off entre contexto e ruído |
| Stride (passo) | 1, 2, 3 | Avaliar sobreposição vs cobertura |
| Pooling | mean, attention-weighted | Comparar agregação uniforme vs ponderada por relevância |

A escolha de $w=5$ como valor central segue a observação empírica de que a maioria dos intents conversacionais se manifesta em janelas de 3 a 7 turnos. Valores maiores ($w=10$) testam se conversas longas se beneficiam de contexto estendido ou se o ruído degrada o desempenho. A comparação entre mean pooling e attention-weighted pooling é motivada por Lyu et al. (2025), que demonstraram ganhos com mecanismos de atenção em classificação de texto.

### 5.4.5 Protocolo de Execução

O protocolo segue os seguintes passos:

Para cada representação $R \in \{$Turn-only, Window-only, ..., Full$\}$:

1. Gerar features para os conjuntos de treino, validação e teste.
2. Para cada classificador $C \in \{$LogReg, LightGBM, MLP$\}$:
   - Treinar $C$ com features $R$ no conjunto de treino.
   - Selecionar hiperparâmetros via conjunto de validação (grid search para LogReg e MLP; Optuna para LightGBM).
   - Avaliar no conjunto de teste: Macro-F1, Micro-F1, Precision e Recall por classe, AUC-ROC.
   - Repetir com validação cruzada estratificada de 5 folds para estimar variância.

Para a análise do tamanho da janela:

1. Fixar o melhor classificador e a melhor representação multi-nível.
2. Variar $w \in \{3, 5, 7, 10\}$.
3. Plotar $w$ versus Macro-F1 para identificar o tamanho ótimo.

Para a análise de pooling:

1. Fixar a melhor configuração geral.
2. Comparar mean pooling versus attention-weighted pooling.
3. Analisar por classe: quais intents se beneficiam do pooling por atenção.

### 5.4.6 Análise Prevista

A análise dos resultados de H2 compreende:

- **Tabela principal:** representação $\times$ classificador $\times$ Macro-F1, no formato empregado por Huang e He (2025).
- **Heatmap:** F1 por classe $\times$ representação, identificando quais classes se beneficiam de contexto multi-nível.
- **Curva de tamanho de janela:** $w$ versus Macro-F1, com intervalos de confiança.
- **Comparação de pooling:** mean versus attention pooling por classe.
- **Análise por classe:** identificação de intents que requerem contexto multi-turn (ex.: objeção após oferta de retenção, escalação progressiva) versus intents detectáveis por turno isolado (ex.: "quero cancelar", "solicitar segunda via").

### 5.4.7 Critérios de Confirmação

H2 é **confirmada** se a melhor configuração multi-nível supera todas as configurações de nível único em Macro-F1, com diferença estatisticamente significativa ($p < 0{,}05$), e o ganho é observável em pelo menos 60% das classes.

H2 é **parcialmente confirmada** se a representação multi-nível melhora apenas classes específicas (tipicamente aquelas que dependem de contexto multi-turn).

H2 é **refutada** se Turn-only ou Conv-only supera a representação multi-nível.

---

## 5.5 Protocolo Experimental para H3 — Regras Determinísticas

### 5.5.1 Hipótese

> *A adição de um motor de regras semânticas (DSL $\to$ AST) ao pipeline híbrido melhora a precision em classes críticas sem degradar o recall global, além de fornecer rastreabilidade de evidência por decisão.*

Esta hipótese parte da observação de que modelos estatísticos de classificação, embora eficazes no caso geral, apresentam limitações em cenários onde (i) o custo de falsos positivos é assimétrico (ex.: classificação errônea de fraude), (ii) requisitos de compliance exigem decisões auditáveis, e (iii) padrões linguísticos específicos do domínio são conhecidos a priori. Sistemas baseados em regras, apesar de limitados em cobertura, oferecem precisão controlável e rastreabilidade total — propriedades complementares aos modelos estatísticos.

### 5.5.2 Definição de Classes Críticas

Selecionamos quatro classes que representam cenários críticos de negócio em operações de atendimento:

- **Cancelamento/churn:** intenção de cancelar serviço — impacto direto em receita e estratégias de retenção.
- **Compliance:** menção a órgãos reguladores (Procon, Anatel, Bacen, ouvidoria) ou processos judiciais — requer escalonamento imediato.
- **Fraude:** padrões suspeitos de engenharia social — requer investigação com rastreabilidade.
- **Insatisfação grave:** escalação, ameaça formal, reclamação com potencial viral — requer intervenção prioritária.

A classificação de uma conversa como pertencente a essas classes tem consequências operacionais diretas, justificando a exigência de alta precision e evidência auditável.

### 5.5.3 Configurações Comparadas

Definimos seis configurações que permitem isolar o efeito das regras no pipeline. A Tabela 5.13 descreve cada configuração.

**Tabela 5.13** — Configurações comparadas no experimento H3.

| Configuração | Descrição |
|---|---|
| ML-only | Melhor classificador do experimento H2, sem regras |
| Rules-lexical | Apenas regras com predicados lexicais (contains, regex) |
| Rules-full | Regras com predicados lexicais + semânticos + contextuais |
| ML+Rules-override | ML-only + regras como override — quando a regra aciona, sua decisão prevalece |
| ML+Rules-feature | ML-only + flags de regras como features adicionais do classificador |
| ML+Rules-postproc | ML-only + regras como pós-processamento, ajustando scores de confiança |

Essa organização permite distinguir entre o valor das regras como sistema independente (Rules-lexical, Rules-full), como mecanismo de override do modelo (ML+Rules-override), como features enriquecedoras (ML+Rules-feature) e como camada de pós-processamento (ML+Rules-postproc).

### 5.5.4 Conjunto de Regras

Para cada classe crítica, definimos entre 3 e 5 regras na DSL do TalkEx. As regras são projetadas antes de qualquer avaliação no conjunto de teste — essa restrição temporal é essencial para evitar viés de construção. A seguir, apresentamos exemplos ilustrativos para a classe de cancelamento:

```
RULE cancelamento_explicito
WHEN
    speaker == "customer"
    AND lexical.contains_any(["cancelar", "encerrar", "desistir", "rescindir"])
    AND NOT lexical.excludes_any(["teste", "simulação"])
THEN
    tag("cancelamento")
    score(0.90)
```

```
RULE cancelamento_implicito
WHEN
    speaker == "customer"
    AND semantic.intent_score("cancelamento") > 0.80
    AND context.repeated_in_window("insatisfação", 3) >= 2
THEN
    tag("cancelamento")
    score(0.85)
```

```
RULE compliance_ouvidoria
WHEN
    lexical.contains_any(["ouvidoria", "procon", "anatel", "bacen", "reclame aqui"])
    OR lexical.regex("processo\\s+\\d+")
THEN
    tag("compliance_risco")
    score(0.95)
    priority("high")
```

Essas regras ilustram as três famílias de predicados: lexicais (contains\_any, regex), semânticos (intent\_score) e contextuais (repeated\_in\_window). A regra `cancelamento_explicito` captura menções diretas; `cancelamento_implicito` captura intenções implícitas via score semântico combinado com padrão contextual; `compliance_ouvidoria` identifica menções a órgãos reguladores com alta confiança.

### 5.5.5 Protocolo de Execução

O protocolo segue os seguintes passos:

1. Definir regras para cada classe crítica (3–5 regras por classe), com regras finalizadas antes de qualquer avaliação no conjunto de teste.
2. Para cada configuração $C$:
   - Executar classificação no conjunto de teste.
   - Para classes críticas: calcular Precision, Recall e F1.
   - Para todas as classes: calcular Macro-F1 e Micro-F1.
   - Registrar o percentual de decisões com evidência rastreável.
   - Registrar a latência adicional por regra.
3. Conduzir análise qualitativa:
   - Selecionar 50 casos onde o modelo e as regras discordam.
   - Avaliar manualmente: quem acerta em cada caso.
   - Categorizar tipos de erro corrigidos pelas regras (ex.: paráfrase não capturada pelo classificador).
   - Categorizar tipos de erro introduzidos pelas regras (ex.: "cancelar o download" classificado erroneamente como intenção de cancelamento de serviço).

### 5.5.6 Análise Prevista

A análise dos resultados de H3 compreende:

- **Tabela principal:** configuração $\times$ Precision/Recall/F1 nas classes críticas.
- **Tabela de impacto global:** Macro-F1 global com e sem regras, verificando que as regras não degradam o desempenho geral.
- **Gráfico de trade-off:** curvas Precision versus Recall por classe crítica, comparando ML-only com as configurações ML+Rules.
- **Exemplos qualitativos** (3–5 por classe): (i) caso que o modelo errou e a regra acertou; (ii) caso que a regra errou e o modelo acertou; (iii) caso com evidências complementares.
- **Análise de evidência:** exemplo completo de output com metadados rastreáveis (matched\_words, scores, thresholds, versão do modelo, versão da regra).

### 5.5.7 Critérios de Confirmação

H3 é **confirmada** se a precision nas classes críticas melhora com qualquer configuração ML+Rules em relação a ML-only, o recall global não degrada mais de 1 ponto percentual, e pelo menos 80% das decisões em classes críticas produzem evidência rastreável.

H3 é **parcialmente confirmada** se a precision melhora em algumas classes mas degrada em outras, ou se a melhoria de precision é acompanhada de custo de latência operacionalmente inaceitável.

H3 é **refutada** se as regras não melhoram a precision em nenhuma classe crítica ou degradam significativamente o recall global.

---

## 5.6 Protocolo Experimental para H4 — Inferência em Cascata

### 5.6.1 Hipótese

> *Um pipeline com inferência cascateada reduz o custo computacional médio por conversa em pelo menos 40% comparado ao pipeline uniforme, com degradação de qualidade inferior a 2 pontos percentuais em F1.*

Esta hipótese fundamenta-se no princípio de inferência cascateada (cascaded inference), amplamente utilizado em sistemas de recuperação de informação de larga escala (Matveeva et al., 2006; Wang et al., 2011). O princípio postula que a aplicação progressiva de modelos cada vez mais custosos — resolvendo cedo os casos fáceis e reservando modelos caros para casos ambíguos — reduz o custo médio sem degradação significativa de qualidade.

### 5.6.2 Pipeline Uniforme (Baseline)

No pipeline uniforme, todas as conversas passam por todos os estágios de processamento, independentemente de sua complexidade:

$$\text{Conversa} \to \text{Normalização} \to \text{Embeddings} \to \text{BM25 + ANN} \to \text{Fusão} \to \text{Classificação} \to \text{Regras} \to \text{Output}$$

Este pipeline representa o cenário onde não há otimização de custo: cada conversa consome o mesmo tempo computacional, seja uma saudação simples ou uma reclamação complexa com múltiplas mudanças de intent.

### 5.6.3 Pipeline Cascateado

O pipeline cascateado organiza o processamento em quatro estágios de custo crescente, com critérios de saída antecipada:

**Estágio 1 — Filtros baratos** (custo estimado: ~1ms). Análise de metadados (canal, fila, idioma, duração) e regras lexicais simples (contains, regex). Se a confiança da classificação atinge $\theta_1$, a conversa é resolvida sem avançar para estágios posteriores.

**Estágio 2 — Retrieval híbrido** (custo estimado: ~10–50ms). BM25 + ANN com fusão de scores, seguido de classificador leve (Logistic Regression) sobre os scores de retrieval. Se a confiança atinge $\theta_2$, a conversa é resolvida.

**Estágio 3 — Classificação completa** (custo estimado: ~50–200ms). Embeddings multi-nível com features heterogêneas, classificador completo (LightGBM/MLP) e regras semânticas completas. Se a confiança atinge $\theta_3$, a conversa é resolvida.

**Estágio 4 — Revisão excepcional** (custo estimado: ~500ms–2s). Cross-encoder reranking, classificação com features adicionais, e flag para revisão humana quando a confiança permanece abaixo de $\theta_4$.

A eficiência da cascata depende de dois fatores: (i) a proporção de conversas resolvidas nos estágios baratos e (ii) a calibração dos scores de confiança (ECE). Conversas triviais — saudações, despedidas, dúvidas com palavras-chave claras — devem ser resolvidas no estágio 1, liberando recursos computacionais para conversas ambíguas.

### 5.6.4 Espaço de Parâmetros

Os thresholds de confiança para cada estágio constituem os parâmetros centrais da cascata. A Tabela 5.14 apresenta os valores testados.

**Tabela 5.14** — Thresholds de confiança variados no experimento H4.

| Parâmetro | Valores testados | Justificativa |
|---|---|---|
| $\theta_1$ (estágio 1) | 0,90; 0,95; 0,98 | Estágio barato — threshold alto para evitar erros em decisões rápidas |
| $\theta_2$ (estágio 2) | 0,85; 0,90; 0,95 | Estágio intermediário — equilíbrio entre resolução e encaminhamento |
| $\theta_3$ (estágio 3) | 0,80; 0,85; 0,90 | Estágio completo — threshold menor aceitável pois o modelo é mais capaz |

O total de configurações a avaliar é $3 \times 3 \times 3 = 27$, uma grade exaustiva que permite identificar a configuração ótima de Pareto.

### 5.6.5 Protocolo de Execução

O protocolo segue os seguintes passos:

1. Instrumentar cada estágio com medição de tempo (milissegundos) e registro de metadados.
2. Executar o pipeline uniforme no conjunto de teste completo:
   - Registrar: classificação final, tempo total, tempo por estágio.
   - Calcular: Macro-F1, custo médio por conversa.
3. Para cada configuração de thresholds $T$:
   - Executar o pipeline cascateado no conjunto de teste completo.
   - Registrar: em qual estágio cada conversa foi resolvida, classificação final, tempo total.
   - Calcular: Macro-F1, custo médio, percentual resolvido por estágio.
4. Comparar pipeline uniforme versus cascateado:
   - $\Delta\text{custo} = (\text{custo}_{uniforme} - \text{custo}_{cascata}) / \text{custo}_{uniforme} \times 100$
   - $\Delta\text{F1} = \text{F1}_{uniforme} - \text{F1}_{cascata}$
   - Plotar a fronteira de Pareto: $\Delta\text{custo}$ versus $\Delta\text{F1}$.
5. Analisar por estágio:
   - Que tipos de conversa são resolvidos no estágio 1? (esperamos: saudações, despedidas, dúvidas com termos exatos)
   - Que tipos requerem o estágio 3? (esperamos: intents ambíguos, mudanças de intent mid-conversation)
   - Que tipos são flagados para revisão? (esperamos: intents novos, edge cases, multilíngue)

### 5.6.6 Análise Prevista

A análise dos resultados de H4 compreende:

- **Tabela principal:** configuração $\times$ custo médio $\times$ Macro-F1 $\times$ percentual resolvido por estágio.
- **Fronteira de Pareto:** gráfico de redução de custo versus degradação de F1, identificando as configurações Pareto-ótimas.
- **Gráfico de barras empilhadas:** percentual de conversas resolvidas em cada estágio, por configuração.
- **Histograma de confiança:** distribuição dos scores de confiança por estágio, validando a calibração.
- **Perfil de conversas por estágio:** comprimento médio, número de turnos, variabilidade de intents para conversas "fáceis" (estágio 1) versus "difíceis" (estágios 3–4).

### 5.6.7 Critérios de Confirmação

H4 é **confirmada** se existe pelo menos uma configuração de thresholds onde a redução de custo é $\geq 40\%$ e a degradação de F1 é $< 2$ pontos percentuais.

H4 é **parcialmente confirmada** se a redução de custo atinge $\geq 40\%$ com degradação $> 2\%$, ou se a degradação é $< 2\%$ mas a redução de custo não atinge $40\%$.

H4 é **refutada** se nenhuma configuração atinge ambos os critérios simultaneamente.

---

## 5.7 Análise Estatística

A análise estatística é transversal a todos os experimentos e visa garantir que as diferenças observadas entre sistemas não são atribuíveis ao acaso. Adotamos um protocolo rigoroso que combina testes de significância, estimativa de variância e estudos de ablação.

### 5.7.1 Testes de Significância

Para todas as comparações entre sistemas, utilizamos os testes apresentados na Tabela 5.15, selecionados com base na natureza da comparação e no número de sistemas envolvidos.

**Tabela 5.15** — Testes estatísticos aplicados por tipo de comparação.

| Tipo de comparação | Teste | Justificativa |
|---|---|---|
| 2 sistemas, métricas pareadas | Wilcoxon signed-rank test | Teste não-paramétrico adequado para amostras pareadas sem pressuposto de normalidade (Wilcoxon, 1945) |
| Múltiplos sistemas | Friedman test + Nemenyi post-hoc | Equivalente não-paramétrico da ANOVA para medidas repetidas, com correção para comparações múltiplas (Demšar, 2006) |
| Intervalo de confiança | Bootstrap com 10.000 reamostras | Estimativa empírica do intervalo de confiança de 95%, sem pressupostos distribucionais (Efron & Tibshirani, 1993) |

O nível de significância adotado é $\alpha = 0{,}05$ para todos os testes. A escolha de testes não-paramétricos é motivada pela impossibilidade de garantir distribuição normal das métricas de avaliação, especialmente em amostras de tamanho moderado. O Wilcoxon signed-rank test é aplicado nas comparações diretas entre pares de sistemas (ex.: melhor híbrido versus melhor isolado em H1), enquanto o teste de Friedman com post-hoc de Nemenyi é aplicado nas comparações envolvendo múltiplos sistemas simultaneamente (ex.: as nove representações de H2).

A adoção do método de Demšar (2006) para comparação de múltiplos classificadores sobre múltiplos datasets é particularmente relevante: esse protocolo é o padrão de facto na comunidade de aprendizado de máquina para evitar a inflação do erro tipo I decorrente de comparações múltiplas.

### 5.7.2 Estimativa de Variância e Reprodutibilidade

Adotamos duas estratégias complementares para estimar a variância dos resultados:

- **Validação cruzada estratificada de 5 folds** para todos os experimentos de classificação (H2, H3, H4). A estratificação preserva a distribuição de classes em cada fold, garantindo que classes minoritárias estejam representadas em todos os subconjuntos.
- **5 seeds diferentes** para os experimentos de retrieval (H1), variando a aleatoriedade da construção do índice ANN e do embaralhamento dos dados.

Todas as tabelas de resultados reportam média $\pm$ desvio padrão. Para métricas particularmente relevantes (Macro-F1, Recall@10), reportamos adicionalmente intervalos de confiança de 95% via bootstrap.

Todos os hiperparâmetros, seeds, configurações de hardware e versões de software são documentados para reprodução integral. O checklist de reprodutibilidade completo inclui: (i) seed fixo (42) para splits e treinamentos; (ii) versões de todas as bibliotecas registradas em `requirements.txt`; (iii) especificação de hardware; (iv) scripts de avaliação versionados no repositório; e (v) testes automatizados passando (quality gates verdes) como pré-condição.

### 5.7.3 Estudos de Ablação

Para o sistema completo (configuração Full), realizamos estudos de ablação sistemáticos removendo um componente por vez e medindo o impacto no desempenho. A Tabela 5.16 descreve as ablações planejadas.

**Tabela 5.16** — Estudos de ablação sobre o sistema completo.

| Ablação | Componente removido | Impacto esperado |
|---|---|---|
| $-$BM25 | Componente lexical do retrieval | Queda em queries com termos exatos, nomes de produtos, códigos |
| $-$ANN | Componente semântico do retrieval | Queda em queries com paráfrases e intenção implícita |
| $-$Rules | Motor de regras | Queda em precision de classes críticas |
| $-$Window | Features de janela de contexto | Queda em intents que dependem de contexto multi-turn |
| $-$Struct | Features estruturais | Impacto potencialmente menor; a ablação valida essa suposição |
| $-$Accent-norm | Normalização de diacríticos | Queda em recall para termos PT-BR com variação acentual |

Cada ablação é avaliada com o mesmo protocolo do sistema completo (mesmos splits, métricas e testes estatísticos), permitindo quantificar a contribuição de cada componente. A ablação de normalização de diacríticos ($-$Accent-norm) é especialmente relevante para o domínio PT-BR, onde a decisão de normalizar "não" e "nao" como equivalentes é uma escolha de design com impacto mensurável.

---

## 5.8 Ameaças à Validade

Reconhecemos e documentamos explicitamente as ameaças à validade dos experimentos, organizadas nas três categorias clássicas de Cook e Campbell (1979): validade interna, externa e de construto. Para cada ameaça, descrevemos as mitigações adotadas.

### 5.8.1 Validade Interna

A validade interna diz respeito à capacidade de atribuir as diferenças observadas às variáveis independentes manipuladas, e não a fatores confundidores.

**Tabela 5.17** — Ameaças à validade interna e mitigações.

| Ameaça | Descrição | Mitigação |
|---|---|---|
| Overfitting nos hiperparâmetros | Seleção de hiperparâmetros que maximizam desempenho no conjunto de teste por acaso | Separação rigorosa train/val/test; hiperparâmetros selecionados exclusivamente no conjunto de validação; confirmação via validação cruzada de 5 folds |
| Viés na construção de regras | Regras definidas com conhecimento do conjunto de teste | Regras definidas e finalizadas antes de qualquer avaliação no conjunto de teste; registro temporal da data de criação das regras |
| Escolha de métricas favorável | Seleção post-hoc de métricas que favorecem o sistema proposto | Métricas definidas a priori no desenho experimental; reporte de múltiplas métricas, incluindo aquelas onde o sistema perde |
| Bugs no pipeline | Erros de implementação que invalidam resultados | Suite extensiva de testes automatizados (1.822+ testes unitários no TalkEx); verificação de quality gates antes de cada experimento |

### 5.8.2 Validade Externa

A validade externa refere-se à generalização dos resultados para contextos além do estudo.

**Tabela 5.18** — Ameaças à validade externa e mitigações.

| Ameaça | Descrição | Mitigação |
|---|---|---|
| Dataset não representativo | Corpus sintético pode não capturar padrões de conversas reais | Disclosure explícito das limitações; validação de robustez no dataset original (944 conversas); discussão detalhada das diferenças potenciais |
| Generalização para outros idiomas | Resultados específicos para PT-BR podem não se transferir | Análise da contribuição da normalização de diacríticos ($-$Accent-norm ablation); discussão de componentes language-agnostic versus language-specific |
| Generalização para outros domínios | Conversas de atendimento têm características específicas (turnos curtos, vocabulário restrito) | Discussão explícita dos pressupostos de domínio; identificação de componentes portáveis versus domain-specific |
| Escala de produção | Experimentos conduzidos em escala de pesquisa (~3.500 conversas), não de produção (milhões) | Análise de complexidade computacional teórica; medições empíricas de throughput e latência; discussão de estratégias de escalabilidade |

### 5.8.3 Validade de Construto

A validade de construto refere-se à adequação entre os conceitos teóricos investigados e as medições efetivamente realizadas.

**Tabela 5.19** — Ameaças à validade de construto e mitigações.

| Ameaça | Descrição | Mitigação |
|---|---|---|
| Métricas não capturam utilidade real | F1 e Recall@K podem não refletir a utilidade percebida pelo operador humano | Inclusão de análise qualitativa com exemplos concretos; discussão da relação entre métricas automáticas e utilidade operacional |
| Explicabilidade não avaliada formalmente | A rastreabilidade de evidência é um diferencial alegado, mas não avaliado com métricas de explicabilidade formais | Definição de critérios qualitativos de qualidade da evidência; exemplos de outputs completos com metadados |
| Custo simplificado | A métrica de custo (tempo em ms) não captura custos de GPU, memória e infraestrutura | Discussão complementar de custo de GPU e memória quando relevante; medição de pico de memória nos estágios mais custosos |

A documentação transparente dessas ameaças é parte integrante do rigor metodológico: reconhecemos os limites das conclusões deriváveis deste estudo e indicamos, quando pertinente, como trabalhos futuros podem endereçá-los.

---

## 5.9 Infraestrutura Experimental

### 5.9.1 Hardware

Documentamos integralmente a configuração de hardware utilizada nos experimentos para garantir a reprodutibilidade das medições de desempenho. As especificações incluem: modelo e número de cores do processador (CPU); modelo e quantidade de VRAM da GPU (quando utilizada para geração de embeddings); quantidade de memória RAM; e tipo de armazenamento (SSD/HDD). Esses dados são reportados no Apêndice C.

### 5.9.2 Software

A Tabela 5.20 sumariza o stack de software utilizado.

**Tabela 5.20** — Stack de software experimental.

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Embeddings | sentence-transformers (E5, BGE) (Reimers & Gurevych, 2019) |
| BM25 | rank-bm25 + implementação própria (`talkex.retrieval.bm25`) |
| ANN | FAISS (faiss-cpu) (Johnson et al., 2019) |
| Classificação | scikit-learn (Pedregosa et al., 2011), LightGBM (Ke et al., 2017) |
| Motor de regras | TalkEx DSL/AST (implementação própria) |
| Avaliação | scikit-learn metrics, scripts customizados |
| Visualização | matplotlib, seaborn |
| Otimização de hiperparâmetros | Optuna (Akiba et al., 2019) |
| Reprodutibilidade | seeds fixos, `requirements.txt` versionado, testes automatizados |

Todas as versões de bibliotecas são fixadas no arquivo `requirements.txt` do repositório TalkEx, garantindo reprodutibilidade exata do ambiente experimental.

### 5.9.3 Código e Dados

O código-fonte do TalkEx está disponível como repositório Git, incluindo o pipeline completo de NLP, o motor de regras (DSL, parser, AST, executor) e os scripts experimentais. Os dados são referenciados pelo identificador do dataset público no HuggingFace, e o procedimento de expansão sintética é documentado para reprodução. Os modelos treinados são armazenados e versionados com registro de hiperparâmetros. Os scripts experimentais reprodutíveis são mantidos no diretório `experiments/` do repositório.

---

## Síntese

Este capítulo apresentou o desenho experimental completo para a avaliação das quatro hipóteses da dissertação. O protocolo foi projetado para garantir: (i) comparabilidade justa entre sistemas, com baselines adequados e controle de variáveis; (ii) rigor estatístico, com testes de significância e estimativa de variância; (iii) reprodutibilidade integral, com seeds fixos, versões documentadas e scripts versionados; e (iv) transparência sobre limitações, com documentação explícita das ameaças à validade.

Os quatro experimentos — retrieval híbrido (H1), representação multi-nível (H2), regras determinísticas (H3) e inferência em cascata (H4) — compartilham o mesmo dataset e infraestrutura, mas são avaliados com métricas e protocolos específicos para suas respectivas hipóteses. Os critérios de confirmação, confirmação parcial e refutação foram definidos a priori, antes da execução dos experimentos, prevenindo viés de interpretação post-hoc.

O capítulo seguinte (Capítulo 6) apresentará os resultados obtidos e sua análise crítica.
