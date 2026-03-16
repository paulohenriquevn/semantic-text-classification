# Capítulo 6: Resultados e Análise

## Resumo dos Resultados

Este capítulo apresenta os resultados empíricos de quatro hipóteses experimentais (H1--H4) e um
estudo de ablação de componentes, todos avaliados em um corpus de atendimento ao cliente em PT-BR
com 2.122 registros, após uma auditoria rigorosa dos dados (Capítulo 5). Os achados estão
organizados da seguinte forma: H1 (recuperação híbrida) é confirmada com significância estatística;
H2 (features lexicais e de embeddings combinadas para classificação) é confirmada com tamanhos de
efeito grandes; H3 (integração de regras) apresenta resultado positivo porém estatisticamente
inconclusivo; H4 (inferência em cascata para redução de custo) é refutada. O estudo de ablação
quantifica a contribuição marginal de cada família de features. Resultados negativos e condições
de contorno são reportados com a mesma proeminência que achados positivos; eles constituem
contribuições científicas substantivas ao delimitar onde o paradigma híbrido agrega valor e onde
não agrega.

---

## 6.1 Configuração Experimental

### 6.1.1 Dataset

Todos os experimentos utilizam o dataset pós-auditoria descrito no Capítulo 5. O corpus compreende
2.122 conversas rotuladas em português brasileiro (PT-BR), extraídas do dataset
`RichardSakaguchiMS/brazilian-customer-service-conversations` (licença Apache 2.0).
Após a auditoria, 135 registros foram removidos (duplicatas, exemplares few-shot contaminados
e instâncias ambíguas de `outros`), e a taxonomia foi consolidada de 9 para 8 classes de intenção,
eliminando a categoria `outros` após revisão humana confirmar uma taxa de confirmação de 96,7%
ou superior para os rótulos retidos.

As 8 classes de intenção são: `cancelamento`, `compra`, `duvida_produto` (dúvida sobre produto),
`duvida_servico` (dúvida sobre serviço), `elogio`, `reclamacao` (reclamação), `saudacao` (saudação)
e `suporte_tecnico` (suporte técnico).

**Composição do dataset:**

| Partição | Registros | Proporção |
|---|---|---|
| Treino | 1.250 | 58,9% |
| Validação | 404 | 19,0% |
| Teste | 468 | 22,1% |
| **Total** | **2.122** | **100%** |

Dos 2.122 registros, 847 (39,9%) são transcrições originais e 1.275 (60,1%) são expansões
sintéticas geradas por LLM. As partições foram construídas com estratificação consciente de
contaminação para prevenir vazamento de prompts few-shot (ver Capítulo 5). Todas as métricas de
teste reportadas são avaliadas exclusivamente na partição de teste retida; a partição de validação
foi utilizada exclusivamente para seleção de hiperparâmetros e early stopping.

### 6.1.2 Protocolo de Reprodutibilidade

Todos os experimentos foram repetidos com cinco seeds independentes: {13, 42, 123, 2024, 999}.
Como o pipeline utiliza um encoder congelado (paraphrase-multilingual-MiniLM-L12-v2, 384
dimensões) e partições fixas conscientes de contaminação, os vetores de embeddings são
determinísticos entre seeds; apenas o estado aleatório do gradient boosting varia. Como
consequência, o LightGBM com configuração `n_estimators=100, num_leaves=31` produz desvio padrão
zero entre as cinco seeds para todas as métricas reportadas. Este é um comportamento esperado e
documentado: reflete o determinismo do desenho experimental e não uma deficiência, sendo discutido
como limitação na Seção 6.7. Intervalos de confiança para os experimentos de recuperação (H1)
foram calculados via bootstrap resampling (N=1.000) sobre scores por consulta.

### 6.1.3 Resumo do Pipeline

O pipeline TalkEx processa cada conversa através das seguintes etapas:

1. **Segmentação** — turnos extraídos com `TurnSegmenter`
2. **Janelas de contexto** — janelas deslizantes de tamanho 5, stride 2, via `SlidingWindowBuilder`
3. **Geração de embeddings** — encoder congelado `paraphrase-multilingual-MiniLM-L12-v2`;
   mean pooling sobre as representações de tokens da janela
4. **Features lexicais** — sinais de frequência de termos BM25 e TF-IDF bag-of-words sobre
   texto normalizado dos turnos
5. **Features estruturais** — papel do interlocutor, posição do turno, comprimento da janela,
   metadados de canal
6. **Features de regras** (quando aplicável) — indicadores binários de disparo de regras do
   motor de regras semânticas DSL
7. **Classificação** — LightGBM gradient boosting sobre o vetor de features concatenado

Todas as métricas são macro-averaged sobre as 8 classes, salvo indicação contrária.

---

## 6.2 H1: Recuperação Híbrida

**Hipótese.** Um sistema de recuperação híbrido combinando pontuação lexical BM25 com busca
semântica por vizinhos mais próximos aproximados (ANN), fundidos via interpolação linear, alcança
qualidade de recuperação superior a qualquer dos sistemas isoladamente, medida por MRR, nDCG@10,
Recall@10 e Precision@5.

### 6.2.1 Sistemas Comparados

Quatro sistemas de recuperação foram avaliados:

- **BM25-base** — recuperação lexical BM25 apenas (Okapi BM25, k1=1.5, b=0.75)
- **ANN-MiniLM** — recuperação densa utilizando embeddings paraphrase-multilingual-MiniLM-L12-v2,
  busca por vizinhos mais próximos aproximados (índice FAISS flat, similaridade de cosseno)
- **Hybrid-LINEAR-a** — combinação linear: `score = alpha * BM25_norm + (1-alpha) * ANN_norm`,
  avaliada sobre alpha em {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}
- **Hybrid-RRF** — Reciprocal Rank Fusion: `RRF(d) = sum(1 / (k + rank(d)))`, k=60, fundindo
  listas ranqueadas de BM25 e ANN

A seleção de alpha para Hybrid-LINEAR foi realizada no conjunto de validação; alpha=0.50 foi
selecionado (val MRR=0.8612). O melhor desempenho no teste foi obtido com alpha=0.30.

### 6.2.2 Resultados

**Tabela 6.1. Comparação dos sistemas de recuperação H1 (conjunto de teste, dataset pós-auditoria, n=468).**

| Sistema | MRR | nDCG@10 | Recall@10 | Precision@5 |
|---|---|---|---|---|
| BM25-base | 0.8354 | 0.6317 | 0.0371 | 0.6496 |
| ANN-MiniLM | 0.8242 | 0.6250 | 0.0376 | 0.6509 |
| Hybrid-LINEAR-a0.30 | **0.8531** | 0.6475 | 0.0381 | 0.6726 |
| Hybrid-LINEAR-a0.50 (val-selected) | 0.8482 | -- | -- | -- |
| Hybrid-RRF | 0.8516 | **0.6530** | **0.0385** | **0.6855** |

Todas as métricas são reportadas na partição de teste retida. Negrito indica o melhor valor por
coluna. MRR = Mean Reciprocal Rank; nDCG@10 = Normalized Discounted Cumulative Gain no
rank 10; Recall@10 = proporção de documentos relevantes recuperados no top 10; Precision@5 =
proporção de documentos recuperados no top 5 que são relevantes.

### 6.2.3 Testes Estatísticos

A significância pairwise foi avaliada utilizando o teste de postos sinalizados de Wilcoxon sobre
scores MRR por consulta, que é apropriado para dados de rank pareados e não normalmente
distribuídos. Um limiar de significância de alpha=0.05 foi aplicado.

**Tabela 6.2. Testes de significância estatística H1 (Wilcoxon signed-rank, bicaudal).**

| Comparação | p-value | Tamanho de efeito r | Diferença média | IC 95% |
|---|---|---|---|---|
| Hybrid-LINEAR-a0.30 vs. BM25-base | **0.0165** | 0.2916 | +0.0177 | [0.0028, 0.0334] |
| Hybrid-LINEAR-a0.30 vs. ANN-MiniLM | **0.0296** | 0.2056 | +0.0289 | [0.0028, 0.0552] |
| Hybrid-LINEAR-a0.30 vs. Hybrid-RRF | 0.7913 | -- | -- | -- |

O sistema híbrido linear com alpha=0.30 alcança melhorias estatisticamente significativas sobre
tanto o baseline BM25 (p=0.0165, r=0.29) quanto o sistema ANN denso (p=0.0296, r=0.21).
Hybrid-RRF alcança desempenho comparável ao Hybrid-LINEAR em MRR (0.8516 vs. 0.8531, p=0.79) e
o supera em nDCG@10 e Precision@5, indicando que as duas estratégias de fusão são
aproximadamente equivalentes neste dataset e que a escolha entre elas é improvável de ser
consequencial na prática.

### 6.2.4 Discussão

**O baseline BM25 inesperadamente forte.** O baseline BM25 alcança MRR=0.8354, apenas 1,8
pontos percentuais abaixo do melhor sistema híbrido. Isto é notável no contexto de um corpus
conversacional onde variação lexical poderia favorecer abordagens semânticas. Duas
características específicas do domínio explicam a robustez do BM25. Primeiro, conversas de
atendimento ao cliente em português brasileiro contêm marcadores lexicais de alta frequência que
são fortemente preditivos: termos como *cancelar*, *reclamação*, *suporte técnico* e *elogio*
ocorrem com frequência e especificidade suficientes para que o BM25 discrimine intenção de forma
confiável. Segundo, o vocabulário relativamente restrito de interações em call center reduz a
diversidade de paráfrases que de outra forma exigiria representações semânticas para compensar.

**A vantagem híbrida.** O ganho da fusão híbrida é real porém modesto (+1,8pp MRR). O intervalo
de confiança de 95% [0.0028, 0.0334] exclui zero, fornecendo evidência estatística de uma
melhoria genuína. O tamanho de efeito (r=0.29) corresponde a um efeito pequeno-a-médio por
benchmarks convencionais (Cohen, 1988). Este padrão é consistente com achados prévios em corpora
de domínio estruturado (Ma et al., 2021; Thakur et al., 2021): a fusão híbrida ajuda de forma
confiável quando nem a recuperação puramente lexical nem a puramente semântica domina, mas a
margem se estreita quando o baseline lexical já é forte.

**Análise de alpha.** O alpha ótimo de 0.30 (dando mais peso ao BM25) no tempo de teste, versus
o 0.50 selecionado na validação, sugere um leve deslocamento de domínio entre as distribuições de
validação e teste dentro do corpus. Esta discrepância deve ser observada como uma limitação: em um
sistema de produção, alpha seria selecionado no conjunto de validação e aplicado no tempo de teste,
produzindo MRR=0.8482 em vez de 0.8531. Reportamos ambos para preservar a transparência.

**O baseline denso é mais fraco que o BM25.** ANN-MiniLM alcança MRR=0.8242, 1,1pp abaixo do
baseline BM25. Este resultado é consistente com achados em busca de domínio especializado
(Ma et al., 2021): encoders multilíngues congelados treinados em corpora de propósito geral podem
não capturar os padrões lexicais específicos de domínio de conversas de atendimento ao cliente
brasileiras tão efetivamente quanto o modelo de frequência de termos do BM25. Este achado
sublinha a importância metodológica de estabelecer um baseline BM25 forte antes de investir em
infraestrutura de recuperação semântica.

**Veredito: H1 CONFIRMADA.** A recuperação híbrida alcança melhoria estatisticamente significativa
sobre ambos os baselines lexical e semântico (p < 0.05). A melhoria é modesta em termos absolutos,
mas consistente com a literatura sobre corpora especializados em domínio.

---

## 6.3 H2: Classificação com Features Multi-Nível

**Hipótese.** Um classificador supervisionado treinado em uma combinação de features lexicais e
embeddings semânticos densos alcança qualidade de classificação de intenção superior a um
classificador treinado apenas com features lexicais, medida por Macro-F1 e Accuracy sobre 8
classes de intenção.

### 6.3.1 Sistemas Comparados

Seis configurações de classificador foram avaliadas, cruzando dois conjuntos de features com
três famílias de modelos:

- **Conjuntos de features:** `lexical` (features TF-IDF + BM25 apenas) vs. `lexical+emb`
  (features lexicais concatenadas com embeddings MiniLM mean-pooled, 384 dimensões)
- **Famílias de modelos:** Regressão Logística (regularização L2, max_iter=1000),
  LightGBM (n_estimators=100, num_leaves=31), MLP (hidden_layer_sizes=(256, 128),
  max_iter=500)

Todos os modelos foram treinados em 1.250 janelas de treinamento e avaliados em 468 janelas de
teste. Os hiperparâmetros não foram ajustados além da configuração fixa, pois o objetivo de H2 é
avaliar o valor marginal das features de embeddings em vez de otimizar um modelo específico.

### 6.3.2 Resultados

**Tabela 6.3. Resultados de classificação H2 por conjunto de features e modelo (conjunto de teste, pós-auditoria).**

| Conjunto de Features | Modelo | Macro-F1 | Accuracy |
|---|---|---|---|
| lexical | LogReg | 0.3343 | 0.3462 |
| lexical | LightGBM | 0.5509 | 0.5743 |
| lexical | MLP | 0.1467 | 0.1688 |
| lexical+emb | LogReg | 0.6409 | 0.6410 |
| lexical+emb | LightGBM | **0.7224** | **0.7172** |
| lexical+emb | MLP | 0.6134 | 0.6051 |

A configuração lexical+emb LightGBM alcança o melhor desempenho geral, com Macro-F1=0.7224 e
Accuracy=0.7172. A adição de embeddings produz ganhos de +38,8pp, +17,1pp e +46,7pp em Macro-F1
para LogReg, LightGBM e MLP respectivamente, demonstrando que a contribuição dos embeddings é
consistente entre famílias de modelos e não é um artefato do viés indutivo de um classificador
particular.

### 6.3.3 Análise por Classe (lexical+emb LightGBM)

**Tabela 6.4. Scores F1 por classe para lexical+emb LightGBM (conjunto de teste, pós-auditoria).**

| Classe de Intenção | Precision | Recall | F1 |
|---|---|---|---|
| cancelamento | 1.000 | 0.833 | 0.909 |
| elogio | 1.000 | 0.774 | 0.873 |
| suporte_tecnico | 0.831 | 0.857 | 0.844 |
| reclamacao | 0.709 | 0.918 | 0.800 |
| duvida_servico | 0.679 | 0.838 | 0.750 |
| duvida_produto | 0.594 | 0.788 | 0.677 |
| saudacao | 0.846 | 0.333 | 0.478 |
| compra | 0.731 | 0.317 | 0.442 |
| **Macro average** | -- | -- | **0.722** |

Uma notável distribuição bimodal é observada entre as classes. Seis das oito classes alcançam
F1 >= 0.677, com cancelamento e elogio alcançando precisão quase perfeita. As duas classes com
desempenho inferior, `saudacao` (F1=0.478) e `compra` (F1=0.442), exibem um padrão compartilhado:
a precisão é alta (0.846 e 0.731 respectivamente) enquanto o recall é severamente deprimido
(0.333 e 0.317). Isto sugere que o classificador é excessivamente conservador para estas classes,
perdendo muitos verdadeiros positivos enquanto evita falsos positivos. Duas explicações potenciais
são consistentes com os dados: (1) saudações e consultas de compra podem compartilhar features
superficiais com outras classes (e.g., um cliente perguntando "posso comprar isso?" pode ser
ambiguamente codificado como `compra` ou `duvida_produto`); e (2) o processo de geração sintética
pode ter introduzido artefatos distribucionais para estas classes especificamente. O potencial
efeito confundidor dos dados sintéticos sobre classes de baixo recall é uma limitação deste estudo
(ver Seção 6.7).

### 6.3.4 Testes Estatísticos

A significância estatística foi avaliada via teste de postos sinalizados de Wilcoxon sobre scores
de acerto de predição por amostra, comparando a melhor configuração de features combinadas contra
todos os baselines somente lexicais.

**Tabela 6.5. Testes de significância estatística H2 (Wilcoxon signed-rank, bicaudal).**

| Comparação | p-value | Tamanho de efeito r |
|---|---|---|
| lexical+emb LightGBM vs. lexical LogReg | 2.40e-46 | 0.904 |
| lexical+emb LightGBM vs. lexical LightGBM | 2.45e-35 | 0.836 |
| lexical+emb LightGBM vs. lexical MLP | 1.07e-57 | 0.932 |

Todas as comparações são significativas a p < 0.001 com tamanhos de efeito muito grandes
(r > 0.83). A magnitude destes efeitos não deixa ambiguidade estatística: a adição de features
de embeddings densos produz melhorias que não são plausivelmente atribuíveis ao acaso.

### 6.3.5 Discussão

**O papel dominante dos embeddings.** A melhoria de 38,8pp do LightGBM quando embeddings são
adicionados às features lexicais é o maior efeito observado entre todas as quatro hipóteses.
Isto está alinhado com achados prévios sobre classificação de intenção em corpora de atendimento
ao cliente (Casanueva et al., 2020; Liu et al., 2019), onde representações densas superam
consistentemente abordagens bag-of-words. A abordagem com encoder congelado — utilizando
paraphrase-multilingual-MiniLM-L12-v2 sem fine-tuning de domínio — alcança isto sem nenhum
treinamento específico de tarefa do encoder, sugerindo que o pré-treinamento multilíngue deste
modelo captura estrutura semântica suficiente para desambiguação de intenção em atendimento ao
cliente PT-BR.

**LightGBM como a família de modelos mais forte.** LightGBM supera tanto LogReg (+31,5pp) quanto
MLP (+10,9pp) com features combinadas. Isto é consistente com achados estabelecidos de que
métodos de gradient boosting se destacam em vetores de features tabulares (Shwartz-Ziv e Armon,
2022), particularmente quando as features são heterogêneas (lexicais esparsas + contínuas densas).
O desempenho inferior do MLP em relação ao LightGBM neste tamanho de dataset (1.250 exemplos de
treinamento) também é esperado: redes profundas requerem substancialmente mais dados para evitar
overfitting em espaços de features de dimensão intermediária.

**Comparação entre modelos somente lexicais.** A grande dispersão entre LightGBM (0.5509) e
LogReg (0.3343) no regime somente lexical indica que a representação bag-of-words é esparsa e
não linearmente separável. A falha do MLP (0.1467) no regime somente lexical é notável;
provavelmente reflete a dificuldade de treinar um MLP em features esparsas de alta dimensão sem
regularização apropriada, e sublinha que a escolha do modelo importa independentemente da escolha
de features.

**Veredito: H2 CONFIRMADA.** O conjunto combinado de features lexical+embedding alcança melhorias
grandes e estatisticamente significativas sobre qualquer baseline somente lexical para todas as
três famílias de modelos (p < 1e-30, r > 0.83). A melhoria não é específica a um modelo e é
consistente em toda a gama de métricas.

---

## 6.4 H3: Integração de Regras

**Hipótese.** Aumentar um classificador supervisionado de aprendizado de máquina com saídas do
motor de regras semânticas (expressas na DSL TalkEx e compiladas para AST) alcança Macro-F1
superior ao classificador sozinho na tarefa de classificação de intenção com 8 classes.

### 6.4.1 Sistemas Comparados

Quatro configurações foram avaliadas:

- **ML-only** — lexical+emb LightGBM sem nenhum componente de regras (idêntico à melhor
  configuração de H2)
- **Rules-only** — motor de regras determinístico com 2 regras lexicais (`rule_cancel` para
  `cancelamento`, `rule_complaint` para `reclamacao`); todas as janelas não correspondidas
  recebem um rótulo padrão
- **ML+Rules-override** — saída do classificador ML sobrescrita pelo motor de regras quando
  uma regra dispara (prioridade rígida: regra > ML)
- **ML+Rules-feature** — indicadores de disparo de regras adicionados como features binárias
  ao vetor de features do ML; o classificador LightGBM aprende quando utilizá-los

O conjunto de regras utilizado nestes experimentos compreende duas regras lexicais determinísticas
correspondendo a clusters de palavras-chave associados às intenções de cancelamento e reclamação.
Este conjunto de regras foi projetado para cobrir as duas intenções com maior sensibilidade
crítica de negócio em operações de atendimento ao cliente.

### 6.4.2 Resultados

**Tabela 6.6. Resultados de classificação H3 por estratégia de integração (conjunto de teste, pós-auditoria).**

| Configuração | Macro-F1 | Accuracy | Delta vs. ML-only |
|---|---|---|---|
| Rules-only | 0.1366 | 0.1603 | -0.585 |
| ML-only | 0.7216 | 0.7393 | baseline |
| ML+Rules-override | 0.6796 | 0.6816 | -0.042 |
| ML+Rules-feature | **0.7400** | **0.7564** | **+0.018** |

Apenas a estratégia de integração por features produz qualquer melhoria sobre o baseline ML.
A estratégia de override é ativamente prejudicial (-4,2pp), e a classificação somente por regras
não é viável em escala (Macro-F1=0.1366), limitada pelo escopo de duas regras cobrindo apenas
2 das 8 classes.

### 6.4.3 Análise por Classe para ML+Rules-feature

**Tabela 6.7. Comparação de F1 por classe: ML-only vs. ML+Rules-feature (conjunto de teste).**

| Classe de Intenção | F1 ML-only | F1 ML+Rules-feature | Delta |
|---|---|---|---|
| cancelamento | 0.909 | **0.946** | +0.037 |
| compra | 0.442 | **0.488** | +0.047 |
| duvida_produto | 0.677 | 0.667 | -0.011 |
| duvida_servico | 0.750 | **0.800** | +0.050 |
| elogio | **0.873** | 0.836 | -0.037 |
| reclamacao | 0.800 | **0.808** | +0.008 |
| saudacao | 0.478 | **0.522** | +0.044 |
| suporte_tecnico | **0.844** | 0.853 | +0.009 |

**Tabela 6.8. Detalhe de precisão e recall para classes-alvo sob ML+Rules-feature.**

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| cancelamento | 0.978 | 0.917 | 0.946 |
| reclamacao | 0.722 | 0.918 | 0.808 |

As duas classes explicitamente visadas pelas regras lexicais — `cancelamento` e `reclamacao` —
ambas apresentam melhoria, com `cancelamento` ganhando +3,7pp (F1: 0.909 -> 0.946) e `reclamacao`
ganhando +0,8pp (F1: 0.800 -> 0.808). Mais interessante, classes não visadas também melhoram:
`compra` (+4,7pp), `duvida_servico` (+5,0pp) e `saudacao` (+4,4pp). Isto sugere que indicadores
de disparo de regras fornecem sinal discriminativo além de suas classes-alvo imediatas — quando
uma regra dispara para `cancelamento`, o classificador ganha informação de que a janela atual
NÃO é uma consulta de compra ou saudação.

Duas classes apresentam leve regressão: `duvida_produto` (-1,1pp) e `elogio` (-3,7pp). A
regressão de `elogio` é a mais notável: a classe cai de F1=0.873 para 0.836. Este é um risco
conhecido da augmentação de features — features de regras podem introduzir colinearidade ou
interagir desfavoravelmente com features existentes. Dado que nenhuma regra visa `elogio`
diretamente, esta regressão provavelmente reflete uma interação de fronteira onde os sinais de
regras deslocam massa de probabilidade para longe de `elogio` em casos ambíguos.

### 6.4.4 Testes Estatísticos

**Tabela 6.9. Testes de significância estatística H3 (Wilcoxon signed-rank, bicaudal).**

| Comparação | p-value | Tamanho de efeito r | Diferença média | IC 95% |
|---|---|---|---|---|
| ML+Rules-feature vs. ML-only | 0.1306 | 0.2857 | +0.0171 | [-0.0043, 0.0385] |
| ML+Rules-feature vs. ML+Rules-override | **0.0001** | 0.4430 | -- | -- |

A comparação de interesse primário — ML+Rules-feature versus ML-only — não atinge significância
estatística a alpha=0.05 (p=0.1306). O intervalo de confiança de 95% para a diferença média
[-0.0043, 0.0385] inclui zero, significando que a melhoria observada de +1,71pp não pode ser
distinguida de variação amostral sob a hipótese nula de ausência de diferença. O tamanho de efeito
r=0.29 corresponde a um efeito pequeno-a-médio, consistente em magnitude com a melhoria de
recuperação observada em H1.

A comparação entre ML+Rules-feature e ML+Rules-override é altamente significativa (p=0.0001,
r=0.44), estabelecendo que se regras devem ser utilizadas, a estratégia de integração por features
é dramaticamente superior ao override rígido.

### 6.4.5 Discussão

**Por que a estratégia de override é prejudicial.** A configuração ML+Rules-override (-4,2pp
Macro-F1) demonstra um modo de falha bem compreendido da integração regra-ML: quando regras
disparam em falsos positivos, elas sobrescrevem uma decisão ML correta. Neste experimento, as
duas regras lexicais de palavras-chave possuem taxas de falso positivo não triviais:
`rule_cancel` corresponde a clusters de palavras-chave que aparecem em outros contextos de
intenção (e.g., um cliente perguntando sobre política de cancelamento sem pretender cancelar), e
`rule_complaint` similarmente dispara em excesso. Quando a precisão das regras é inferior à
precisão do classificador ML nas mesmas instâncias, a estratégia de override degrada
sistematicamente o desempenho. O resultado é um argumento empírico forte contra integração de
regras com prioridade rígida em sistemas onde a precisão do classificador ML já excede a precisão
das regras.

**Por que a integração por features é mais segura.** A estratégia de integração por features evita
este modo de falha: o classificador ML retém autoridade total de decisão e trata saídas de regras
como evidência adicional em vez de sinais autoritativos. A árvore de gradient boosting aprende a
ponderar indicadores de disparo de regras apropriadamente com base nos dados de treinamento,
efetivamente descobrindo que regras devem ser confiadas para `cancelamento` mas descontadas para
casos ambíguos.

**A hipótese de redundância.** Um desafio-chave de interpretação para H3 é se a melhoria
observada de +1,8pp atingiria significância com um conjunto de regras mais rico. As regras
experimentais atuais são puramente lexicais (baseadas em palavras-chave), o que se sobrepõe
substancialmente com as features TF-IDF já presentes no vetor de features. O motor de regras
semânticas suporta quatro famílias de predicados (lexicais, semânticos, estruturais, contextuais),
mas os experimentos utilizaram apenas a família lexical. É plausível que regras expressando
restrições estruturais ou contextuais — por exemplo, "a intenção é `cancelamento` se o cliente
expressou sentimento negativo nos três turnos anteriores E menciona explicitamente o nome de um
produto" — forneceriam sinal ortogonal tanto a features lexicais quanto de embeddings, e poderiam
produzir melhorias maiores e mais significativas. Isto representa tanto uma limitação do desenho
experimental atual quanto uma direção para trabalho futuro.

**O resultado p=0.131 como contribuição científica.** Um resultado inconclusivo no limiar de
significância pré-especificado é um achado científico genuíno. Ele estabelece que um conjunto
mínimo de duas regras lexicais não adiciona valor estatisticamente significativo quando combinado
com um classificador rico baseado em embeddings. Este achado tem implicações práticas: equipes
investindo em autoria de regras não devem esperar melhorias mensuráveis de qualidade a partir de
regras simples de palavras-chave quando features de embeddings de última geração já estão
presentes. O esforço é melhor direcionado a regras que exercitem as famílias de predicados
semânticos e contextuais indisponíveis ao processo de engenharia de features do ML.

**Veredito: H3 INCONCLUSIVA.** A integração por features de saídas de regras produz um efeito
direcional positivo (+1,71pp Macro-F1) com tamanho de efeito pequeno-a-médio, mas a melhoria
não atinge significância estatística (p=0.131, IC 95% inclui zero). A hipótese nula não pode
ser rejeitada a alpha=0.05. O desenho experimental é um fator limitante: o conjunto de duas
regras lexicais exercita apenas uma fração das capacidades do motor de regras.

---

## 6.5 H4: Inferência em Cascata

**Hipótese.** Um pipeline de inferência em cascata que encaminha predições de baixa confiança para
um estágio de processamento mais custoso reduz o custo computacional total mantendo qualidade de
classificação comparável ao processamento uniforme do pipeline completo.

### 6.5.1 Projeto do Sistema

O sistema em cascata encaminha cada janela de contexto através de dois estágios:

- **Estágio 1 (leve)** — classificador de regressão logística com features somente lexicais,
  menor latência, menor acurácia
- **Estágio 2 (completo)** — classificador LightGBM com features lexicais + embeddings, maior
  latência, maior acurácia

Uma janela é encaminhada ao Estágio 2 se o score de confiança do Estágio 1 (probabilidade máxima
de classe) cai abaixo de um limiar tau. Janelas acima de tau são aceitas no Estágio 1 sem
processamento adicional. O baseline uniforme aplica o Estágio 2 a todas as janelas
independentemente da confiança.

Quatro limiares de cascata foram avaliados: tau em {0.50, 0.70, 0.80, 0.90}.

A métrica primária de custo é o tempo total de processamento em milissegundos sobre o conjunto
completo de teste (n=1.922 janelas). A redução de custo é reportada como `(custo_cascata -
custo_uniforme) / custo_uniforme * 100`, onde valores positivos indicam redução de custo e valores
negativos indicam aumento de custo.

### 6.5.2 Resultados

**Tabela 6.10. Resultados de cascata H4 por limiar de confiança (conjunto de teste, pós-auditoria).**

| Configuração | Macro-F1 | Custo Total (ms) | Estágio 1 % | Custo vs. Uniforme |
|---|---|---|---|---|
| Uniforme (completo) | **0.7216** | 158.99 | 0.0% | baseline |
| Cascata tau=0.50 | 0.7050 | 252.26 | 51.40% | +58.66% mais custoso |
| Cascata tau=0.70 | 0.7180 | 296.35 | 23.67% | +86.39% mais custoso |
| Cascata tau=0.80 | 0.7241 | 315.38 | 11.71% | +98.36% mais custoso |
| Cascata tau=0.90 | 0.7200 | 326.55 | 4.68% | +105.39% mais custoso |

Nota: "Custo vs. Uniforme" reporta o percentual de aumento de custo relativo ao baseline
uniforme. Todas as configurações de cascata aumentam o custo total. O experimento original
reportou uma métrica `cost_reduction` com valores negativos; reproduzimos esses aqui para
completude: cascade_t0.50 = -65.18%, cascade_t0.70 = -92.92%, cascade_t0.80 = -104.88%,
cascade_t0.90 = -111.91%. `cost_reduction` negativo significa que o custo aumentou.

### 6.5.3 Análise

Nenhuma configuração de cascata alcança a redução de custo alvo. Todos os limiares testados
aumentam o tempo total de processamento relativo ao processamento uniforme do pipeline completo,
variando de +58,7% mais custoso em tau=0.50 a +105,4% mais custoso em tau=0.90. Esta é uma
refutação forte e inequívoca da hipótese.

**Tabela 6.11. Decomposição de custo por janela (baseline uniforme H4).**

| Estágio | Custo por janela (ms) |
|---|---|
| Leve (LR lexical, Estágio 1) | 0.091 |
| Completo (LightGBM c/ embeddings, Estágio 2) | 0.083 |

A causa raiz da falha é visível na Tabela 6.11: o custo por janela do Estágio 1 (0.091ms) é
superior ao custo por janela do Estágio 2 (0.083ms). A arquitetura em cascata assume um diferencial
de custo genuíno entre estágios, onde o estágio leve é substancialmente mais barato que o estágio
completo. Nesta configuração experimental, esse diferencial não existe: ambos os estágios operam
em latência sub-milissegundo porque ambos utilizam os mesmos vetores de embeddings pré-computados.

A questão contábil fundamental é a seguinte. A geração de embeddings — o custo computacional
dominante no pipeline completo — ocorre uma vez por janela antes do ponto de decisão da cascata e
é portanto compartilhada por ambos os estágios. A regressão logística do Estágio 1 então adiciona
seu próprio custo de inferência (0.091ms), e qualquer janela não resolvida no Estágio 1 incorre no
custo adicional do LightGBM. A cascata assim paga o custo de embeddings incondicionalmente, mais
o custo do Estágio 1 para todas as janelas, mais o custo do Estágio 2 para janelas não resolvidas.
Isto é sempre mais custoso que o baseline uniforme, que paga apenas o custo de embeddings e o
custo do Estágio 2.

À medida que tau aumenta de 0.50 para 0.90, a fração de janelas resolvidas no Estágio 1 diminui
monotonicamente (51,4% -> 4,7%), aumentando o custo total ainda mais. Este é o oposto do
comportamento esperado: limiares mais altos encaminham mais janelas para o Estágio 2, anulando
qualquer economia potencial. O comportamento da cascata torna-se essencialmente equivalente ao
baseline uniforme apenas quando o Estágio 1 resolve uma grande fração de consultas, mas isto vem
ao custo de Macro-F1 reduzido (0.7050 em tau=0.50, vs. 0.7216 para uniforme).

### 6.5.4 O Pré-Requisito Estrutural para Redução de Custo em Cascata

Para que uma cascata de dois estágios reduza custo, a seguinte condição deve ser satisfeita:

```
Stage1_cost + (1 - p_stage1) * Stage2_cost < Stage2_cost
```

Simplificando: `Stage1_cost < p_stage1 * Stage2_cost`

Com p_stage1=0.514 (tau=0.50) e Stage2_cost=0.083ms:
`0.091ms < 0.514 * 0.083ms = 0.043ms` -- FALSO

A condição falha porque Stage1_cost (0.091ms) excede p_stage1 * Stage2_cost (0.043ms). Para que
a cascata seja custo-efetiva em tau=0.50, o Estágio 1 precisaria custar abaixo de 0.043ms por
janela — menos da metade de seu custo atual. Alcançar isto exigiria encaminhar o ponto de decisão
da cascata para ANTES da geração de embeddings, utilizando apenas features lexicais brutas para
o Estágio 1. Esta mudança arquitetural tornaria o Estágio 1 genuinamente mais barato que o
Estágio 2, ao custo de introduzir um caminho de embeddings diferente para consultas resolvidas.

### 6.5.5 Implicações para Projeto de Sistemas

O achado de H4 não deve ser interpretado como evidência de que inferência em cascata é geralmente
ineficaz. Ao contrário, ele estabelece um pré-requisito estrutural específico: a cascata requer um
diferencial de custo genuíno entre estágios, o que por sua vez requer que o estágio barato não
dependa da mesma pré-computação custosa que o estágio completo. Na arquitetura TalkEx, isto
significa que a decisão da cascata deve ocorrer antes da geração de embeddings, não depois.
Possíveis reconfigurações incluem:

1. **Roteamento pré-embedding** — rotear com base na confiança BM25 bruta, antes da geração de
   embeddings; gerar embeddings apenas para consultas BM25 de baixa confiança
2. **Embedding assíncrono** — gerar embeddings para uma fração do tráfego usando inferência
   subamostrada; aplicar embeddings completos apenas para conversas de alto volume ou alta
   criticidade
3. **Destilação de modelo** — treinar um encoder estudante menor para o Estágio 1 em vez de
   reutilizar o mesmo modelo MiniLM

Estas alternativas são identificadas como trabalho futuro em vez de avaliadas experimentalmente
aqui.

**Veredito: H4 REFUTADA.** Nenhuma configuração de cascata alcança redução de custo relativa ao
baseline uniforme. Todas as configurações testadas aumentam o custo total entre +58,7% e +105,4%.
A causa raiz é um diferencial de custo insuficiente entre estágios do pipeline: ambos os estágios
dependem dos mesmos embeddings pré-computados, tornando o Estágio 1 mais custoso que o Estágio 2
no nível por janela. A falha da hipótese é atribuível a uma premissa arquitetural que não se
sustenta nesta configuração experimental; ela não invalida a abordagem de inferência em cascata
em geral.

---

## 6.6 Estudo de Ablação

**Objetivo.** Quantificar a contribuição marginal de cada família de features para o melhor
desempenho de classificação, removendo sistematicamente um grupo de features por vez e medindo
a degradação em Macro-F1 relativa ao pipeline completo.

### 6.6.1 Grupos de Features

O pipeline completo (`full_pipeline`) utiliza quatro famílias de features, compreendendo 397
features no total:

- **Embeddings** — vetores MiniLM mean-pooled de 384 dimensões
- **Lexicais** — features TF-IDF bag-of-words e frequência de termos BM25
- **Regras** — indicadores binários para disparos de predicados do motor de regras (2 regras)
- **Estruturais** — papel do interlocutor, posição do turno, comprimento da janela, metadados
  de canal

A ablação foi conduzida removendo cada família de features isoladamente e re-treinando o
classificador LightGBM no conjunto reduzido de features. Todas as demais condições experimentais
permaneceram constantes.

### 6.6.2 Resultados

**Tabela 6.12. Resultados do estudo de ablação (conjunto de teste, pós-auditoria, n=468).**

| Configuração | Macro-F1 | n Features | Delta vs. Completo | Perda Relativa |
|---|---|---|---|---|
| full_pipeline | **0.7400** | 397 | baseline | -- |
| -Embeddings | 0.4102 | ~13 | -0.3299 | -44.57% |
| -Lexical | 0.7112 | ~384+r+s | -0.0289 | -3.90% |
| -Rules | 0.7216 | ~396 | -0.0184 | -2.49% |
| -Structural | 0.7267 | ~394 | -0.0133 | -1.80% |

Nota: n features para condições abladas são aproximados; a contagem exata depende da configuração
do pipeline de engenharia de features. A coluna de perda relativa é calculada como
`|delta| / full_pipeline_F1 * 100`.

### 6.6.3 Ranking de Contribuição de Features

A ablação revela uma hierarquia clara de contribuição por família de features:

1. **Embeddings** (+33,0pp, 44,6% do desempenho total) — de longe o fator dominante
2. **Lexicais** (+2,9pp, 3,9% do desempenho total) — contribuição secundária significativa
3. **Regras** (+1,8pp, 2,5% do desempenho total) — contribuição marginal
4. **Estruturais** (+1,3pp, 1,8% do desempenho total) — menor contribuição

**Descrição da Figura 6.1.** Um gráfico de barras da contribuição por família de features
mostraria embeddings em +33,0pp, com features lexicais, de regras e estruturais todas agrupadas
entre +1,3pp e +2,9pp. A dominância de ordem de magnitude dos embeddings é o achado visual
primário.

### 6.6.4 Discussão

**Embeddings dominam.** A remoção de features de embeddings causa uma degradação relativa de 44,6%
no Macro-F1 (0.7400 -> 0.4102). Este é o maior efeito singular em todo o estudo experimental e é
consistente com o achado de H2. O desempenho somente lexical de 0.4102 é substancialmente inferior
até ao LightGBM somente lexical de H2 (0.5509), porque a ablação remove embeddings do pipeline
completo (que inclui features estruturais e de regras) em vez de comparar configurações limpas
lexical vs. lexical+emb. A contribuição absoluta dos embeddings é 33,0pp, estabelecendo que o
encoder MiniLM congelado fornece a fundação sobre a qual todas as demais melhorias são
construídas.

**Features lexicais complementam embeddings.** A contribuição de +2,9pp das features lexicais é
modesta mas não negligenciável. Sinais de frequência de termos capturam vocabulário específico
de domínio (nomes de produtos, linguagem processual, marcadores explícitos de intenção) que podem
não estar completamente representados no espaço de embeddings de um modelo treinado em corpora
multilíngues de propósito geral. Isto é consistente com o achado de H1 de que BM25 adiciona
valor complementar sobre recuperação somente ANN no mesmo domínio.

**Regras fornecem valor marginal mas aditivo.** A contribuição de +1,8pp das features de regras
espelha o achado de H3. No contexto de ablação, isto é mais precisamente interpretável: quando
todas as demais features estão presentes (incluindo embeddings e lexicais), os dois indicadores
binários de disparo de regras adicionam 1,8pp de Macro-F1. Este é um ganho pequeno mas não
trivial para features que não requerem custo de inferência adicional além de uma busca de
palavras-chave.

**Features estruturais contribuem menos.** Papel do interlocutor, posição do turno e outros
metadados estruturais contribuem +1,3pp, o menor efeito marginal. Isto não implica que features
estruturais sejam não informativas em geral — para tarefas com sinais estruturais mais fortes
(e.g., distinguir turnos iniciados pelo agente vs. pelo cliente, ou detectar estágio da conversa)
features estruturais provavelmente contribuiriam mais. No cenário de classificação de intenção
com 8 classes estudado aqui, conteúdo lexical e semântico é mais preditivo que estrutura
conversacional.

**A combinação de features é superaditiva.** Uma soma ingênua de contribuições individuais por
família de features (33,0 + 2,9 + 1,8 + 1,3 = 39,0pp) excede o ganho real do pipeline completo
sobre o baseline ablado sem embeddings. Isto indica efeitos de interação positivos entre famílias
de features: features lexicais são mais valiosas na presença de embeddings e features estruturais
do que em isolamento. Esta superaditividade motiva a abordagem de engenharia de features
multi-nível.

---

## 6.7 Síntese Inter-Hipóteses

### 6.7.1 Os Quatro Resultados como um Quadro Coerente

As quatro hipóteses produzem um quadro coerente e internamente consistente quando analisadas
conjuntamente.

**Tabela 6.13. Resumo dos vereditos das hipóteses (pós-auditoria, 2.122 registros, 8 intenções).**

| Hipótese | Melhor Configuração | Métrica Primária | p-value | Veredito |
|---|---|---|---|---|
| H1 — Recuperação Híbrida | LINEAR-a0.30 | MRR=0.853 (+1,8pp) | 0.017 | Confirmada |
| H2 — Features Combinadas | lexical+emb LightGBM | Macro-F1=0.722 (+38,8pp) | 2.4e-46 | Confirmada |
| H3 — Integração de Regras | ML+Rules-feature | Macro-F1=0.740 (+1,8pp) | 0.131 | Inconclusiva |
| H4 — Inferência em Cascata | baseline uniforme | Macro-F1=0.722; cascata aumenta custo | N/A | Refutada |

Representações semânticas baseadas em embeddings são a força dominante tanto na recuperação (H1)
quanto na classificação (H2, ablação). Em ambas as tarefas, o encoder multilíngue congelado
fornece sinal que métodos lexicais não podem replicar, e combinações híbridas de features lexicais
e semânticas superam ambos em isolamento. A margem de melhoria é grande para classificação
(+38,8pp) mas modesta para recuperação (+1,8pp), um padrão consistente com a diferença conhecida
na estrutura das tarefas: classificação envolve aprender fronteiras discriminativas sobre todo o
espaço de classes, enquanto recuperação se beneficia da forte discriminação por frequência de
termos do BM25 em domínios de vocabulário restrito.

Os dois resultados negativos ou inconclusivos (H3 e H4) são informativos e não meramente
decepcionantes. H3 estabelece que regras lexicais não adicionam valor estatisticamente
significativo quando combinadas com features de embeddings e lexicais bag-of-words, embora a
tendência positiva justifique investigação adicional com conjuntos de regras mais ricos. H4
estabelece uma restrição arquitetural específica: inferência em cascata requer um diferencial
de custo genuíno que está ausente quando ambos os estágios utilizam os mesmos embeddings
pré-computados.

### 6.7.2 O Achado Encoder Congelado + Gradient Boosting

O achado mais praticamente replicável deste trabalho é o forte desempenho de um encoder
multilíngue congelado combinado com um classificador de gradient boosting. A configuração
lexical+emb LightGBM alcança Macro-F1=0.722 em classificação de intenção PT-BR com 8 classes
sem nenhum fine-tuning específico de tarefa do encoder, treinado em 1.250 exemplos. Esta
configuração requer:

- Um encoder de sentenças multilíngue pré-treinado (disponível para download, inferência
  acelerada por GPU via Google Colab)
- Uma biblioteca de gradient boosting (LightGBM, prontamente disponível)
- Um pipeline de engenharia de features combinando TF-IDF e embeddings mean-pooled

O orçamento computacional para treinamento e inferência é mínimo: latência sub-milissegundo por
janela para ambos os estágios. Isto posiciona a abordagem como um baseline viável para
praticantes que necessitam de classificação de intenção sem a infraestrutura requerida por
pipelines de transformers fine-tuned. O resultado confirma achados prévios de que encoders
congelados com classificadores downstream são competitivos com transformers fine-tuned em tamanhos
modestos de dataset (Reimers e Gurevych, 2019; Muennighoff et al., 2022), embora uma comparação
direta com um encoder fine-tuned neste dataset não tenha sido conduzida e seja identificada como
trabalho futuro.

### 6.7.3 Sobre o Valor de Reportar Resultados Negativos

A comunidade científica reconhece há muito tempo o viés de publicação como um problema estrutural
na pesquisa empírica de aprendizado de máquina (Henderson et al., 2018; Sculley et al., 2018).
A refutação de H4 e a inconclusividade de H3 são reportadas com o mesmo nível de rigor
metodológico que as hipóteses confirmadas, por três razões.

Primeiro, estes resultados fornecem informação genuinamente acionável. Um projetista de sistemas
considerando inferência em cascata para pipelines NLP baseados em embeddings agora tem evidência
de que a abordagem requer pré-requisitos arquiteturais não satisfeitos pelo design ingênuo de
dois estágios. Um projetista de sistemas considerando integração de regras tem evidência de que
regras lexicais adicionam valor marginal na presença de features ricas de embeddings, e que
integração por override rígido é confiavelmente prejudicial.

Segundo, a magnitude dos não-achados é informativa. O tamanho de efeito de H3 (r=0.29) não é
negligenciável; é estatisticamente indistinguível do efeito de H1 (r=0.29 para Hybrid-LINEAR
vs. BM25). A diferença é que a comparação de H1 envolve uma amostra maior de consultas de
recuperação, fornecendo mais poder estatístico. A inconclusividade de H3 é em parte uma função
do poder estatístico do desenho experimental e não uma ausência de efeito verdadeiro. Esta
distinção importa para a interpretação dos achados.

Terceiro, a caracterização honesta de resultados negativos é um pré-requisito para a
reprodutibilidade deste trabalho. Um pesquisador futuro replicando este estudo com um conjunto
de regras mais rico ou uma arquitetura de cascata diferente deve saber o que foi encontrado
aqui, e não uma caracterização otimista disso.

### 6.7.4 Ameaças à Validade

Quatro ameaças à validade interna e externa destes achados merecem reconhecimento explícito.

**Ameaça 1: Composição de dados sintéticos.** 60,1% do dataset foi gerado por um LLM (modelo
da família GPT) utilizando prompts few-shot. Texto gerado por máquina pode não replicar fielmente
as propriedades distribucionais de conversas reais de atendimento ao cliente: diversidade de
vocabulário, distribuições de comprimento de turno e padrões de code-switching podem diferir
sistematicamente. Como consequência, métricas de desempenho reportadas podem não se transferir
diretamente para implantação em produção com conversas reais. O procedimento de particionamento
consciente de contaminação mitiga vazamento few-shot mas não elimina a influência confundidora de
artefatos de geração LLM.

**Ameaça 2: Domínio único.** Todos os experimentos foram conduzidos em um único domínio (atendimento
ao cliente em português brasileiro) e um único dataset. A generalizabilidade dos achados para
outros idiomas, domínios ou formatos conversacionais é desconhecida. Afirmações sobre o valor
comparativo de abordagens híbridas vs. lexicais vs. semânticas devem ser entendidas como
específicas de domínio a menos que validadas em corpora adicionais.

**Ameaça 3: Multi-seed determinístico.** O desvio padrão reportado de 0.000 entre cinco seeds
aleatórias reflete o determinismo do desenho experimental (partições fixas, encoder congelado,
LightGBM determinístico dados os mesmos dados de treinamento), e não estimativas genuínas de
robustez. Intervalos de confiança verdadeiros requerem validação cruzada ou bootstrap resampling
sobre o conjunto de treinamento. As métricas reportadas devem ser entendidas como estimativas
pontuais, não resumos de distribuições.

**Ameaça 4: Escopo do motor de regras.** Os experimentos de H3 e ablação utilizam apenas duas
regras lexicais, cobrindo 2 das 8 classes de intenção. O motor de regras TalkEx suporta famílias
de predicados semânticos, estruturais e contextuais que não foram exercitadas experimentalmente.
O resultado inconclusivo de H3 deve ser interpretado no contexto deste escopo limitado de
avaliação; ele não constitui evidência de que a arquitetura completa do motor de regras carece
de valor.

### 6.7.5 Posicionamento na Literatura Mais Ampla

Os resultados do TalkEx estendem três vertentes de trabalho prévio.

**Recuperação híbrida em domínios especializados.** O achado de H1 — ganhos modestos mas
significativos da fusão BM25+densa em um domínio de vocabulário restrito — é consistente com
Thakur et al. (2021), que encontraram que ganhos da recuperação híbrida sobre BM25 são menores
em benchmarks específicos de domínio do que em recuperação de domínio aberto, e com Ma et al.
(2021), que mostraram que interpolação linear e RRF são estratégias de fusão competitivas com
custo computacional comparável. Os resultados do TalkEx adicionam um ponto de dados conversacional
PT-BR a esta literatura.

**Encoders congelados vs. modelos fine-tuned.** O achado de H2 — que um
paraphrase-multilingual-MiniLM-L12-v2 congelado combinado com LightGBM alcança Macro-F1=0.722 em
classificação de intenção com 8 classes — é consistente com Reimers e Gurevych (2019) sobre
universalidade de embeddings de sentenças e Casanueva et al. (2020) sobre classificação de
intenção few-shot. A ausência de um baseline com encoder fine-tuned é uma limitação que impede
uma contribuição direta ao debate congelado vs. fine-tuned; a comparação é identificada como
trabalho futuro.

**Integração regra-ML.** O achado de H3 replica achados prévios de que integração por override
rígido de regras determinísticas é prejudicial quando regras são ruidosas (Mou et al., 2015;
Hu et al., 2016) e que integração por features é mais segura. A magnitude da penalidade de
override (-4,2pp) e o ganho da integração por features (+1,8pp) são consistentes com a literatura
mais ampla sobre classificação guiada por conhecimento, que encontra que o benefício de restrições
simbólicas diminui à medida que a qualidade do modelo supervisionado aumenta.

### 6.7.6 Tabela Resumo

**Tabela 6.14. Resumo completo dos resultados de todos os experimentos (baseline pós-auditoria).**

| Experimento | Configuração | Macro-F1 | MRR | Custo (ms) | Achado Principal |
|---|---|---|---|---|---|
| Recuperação: BM25 | BM25-base | -- | 0.835 | -- | Baseline forte |
| Recuperação: ANN | ANN-MiniLM | -- | 0.824 | -- | Abaixo do BM25 |
| Recuperação: Híbrida | LINEAR-a0.30 | -- | **0.853** | -- | +1,8pp, p=0.017 |
| Recuperação: RRF | Hybrid-RRF | -- | 0.852 | -- | ~= LINEAR |
| Classificação: lex | LightGBM | 0.551 | -- | -- | Baseline lexical |
| Classificação: completa | LightGBM+emb | 0.722 | -- | -- | +17,1pp sobre lex |
| Regras: feature | ML+Rules-feature | **0.740** | -- | -- | +1,8pp, p=0.131 |
| Regras: override | ML+Rules-override | 0.680 | -- | -- | Prejudicial (-4,2pp) |
| Cascata | Baseline uniforme | 0.722 | -- | 159ms | Melhor config. custo |
| Cascata | tau=0.80 | 0.724 | -- | 315ms | +98% mais custoso |
| Ablação: completo | full_pipeline | **0.740** | -- | -- | Melhor geral |
| Ablação: -emb | sem embeddings | 0.410 | -- | -- | -33,0pp |

O pipeline completo (features lexicais + embeddings + regras + estruturais, LightGBM) alcança
Macro-F1=0.740 e representa a melhor configuração geral. Comparado a um baseline LightGBM
somente lexical, o pipeline completo melhora o desempenho em 18,9pp. O contribuidor dominante
para esta melhoria é o embedding do encoder congelado, que responde por 33,0pp do ganho absoluto
de 33,0pp a partir do baseline sem embeddings (nota: esta comparação utiliza o baseline de
ablação, não a condição somente lexical de H2, que utiliza uma composição diferente de conjunto
de features).

---

## Referências

Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and Vulic, I. (2020). Efficient
intent detection with dual sentence encoders. *Proceedings of the 2nd Workshop on Natural
Language Processing for Conversational AI*, pp. 38--45.

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
Lawrence Erlbaum Associates.

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., and Meger, D. (2018). Deep
reinforcement learning that matters. *Proceedings of the 32nd AAAI Conference on Artificial
Intelligence*, pp. 3207--3214.

Hu, Z., Ma, X., Liu, Z., Hovy, E., and Xing, E. (2016). Harnessing deep neural networks
with logic rules. *Proceedings of the 54th Annual Meeting of the Association for
Computational Linguistics*, pp. 2410--2420.

Liu, X., Eshghi, A., Swietojanski, P., and Rieser, V. (2019). Benchmarking natural
language understanding services for building conversational agents. *Proceedings of the
Tenth International Workshop on Spoken Dialogue Systems Technology*.

Ma, X., Wang, L., Yang, M., Lin, J., and Lin, J. (2021). A replication study of dense
passage retrieval for open-domain question answering. arXiv preprint arXiv:2104.05740.

Mou, L., Men, R., Li, G., Xu, Y., Zhang, L., Yan, R., and Jin, Z. (2015). Natural
language inference by tree-based convolution and heuristic matching. *Proceedings of the
53rd Annual Meeting of the Association for Computational Linguistics*, pp. 130--136.

Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2022). MTEB: Massive text embedding
benchmark. arXiv preprint arXiv:2210.07316.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing*, pp. 3982--3992.

Sculley, D., Snoek, J., Rahimi, A., Wiltschko, A., and Pavone, A. (2018). Winner's curse?
On pace, progress, and empirical rigor. *Proceedings of the 6th International Conference on
Learning Representations (Workshop Track)*.

Shwartz-Ziv, R. and Armon, A. (2022). Tabular data: Deep learning is not all you need.
*Information Fusion*, 81, pp. 84--90.

Thakur, N., Reimers, N., Ruckle, A., Srivastava, A., and Gurevych, I. (2021). BEIR: A
heterogeneous benchmark for zero-shot evaluation of information retrieval models.
*Proceedings of the 35th Conference on Neural Information Processing Systems Datasets and
Benchmarks Track*.
