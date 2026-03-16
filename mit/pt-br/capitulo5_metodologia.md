# Capítulo 5: Desenho Experimental e Metodologia

## Resumo do Capítulo

Este capítulo especifica o protocolo experimental utilizado para avaliar as quatro hipóteses da
tese. Descrevemos o dataset (pós-auditoria), as métricas de avaliação para cada hipótese, as
configurações experimentais comparadas, o framework de análise estatística, o desenho do estudo
de ablação e as ameaças à validade. O nível de detalhe visa possibilitar a reprodução completa
por pesquisadores independentes. Todas as decisões experimentais — métricas, baselines,
critérios de confirmação, testes estatísticos — foram definidas *a priori*, antes da observação
de quaisquer resultados, para prevenir viés de interpretação post-hoc.

---

## 5.1 Dataset

### 5.1.1 Fontes de Dados

A avaliação empírica requer um corpus de conversas de atendimento ao cliente em português
brasileiro anotadas com intenções e metadados estruturais. Datasets conversacionais públicos
em PT-BR são escassos — uma lacuna reconhecida no processamento de linguagem natural em
português (Souza et al., 2020). Endereçamos essa limitação por meio da expansão sintética
controlada de um corpus público existente.

**Corpus base.** Utilizamos o dataset
`RichardSakaguchiMS/brazilian-customer-service-conversations`, hospedado no HuggingFace sob
licença Apache 2.0. O corpus original contém aproximadamente 944 conversas anotadas com
rótulos de intenção e polaridades de sentimento. Embora valioso como ponto de partida, o
dataset original apresenta quatro limitações que comprometem sua adequação para experimentação
rigorosa: (i) uma distribuição de classes aproximadamente uniforme (~11% por classe), distante
das distribuições naturalmente desbalanceadas observadas em centrais de atendimento em
produção; (ii) baixa variabilidade no comprimento das conversas, com aproximadamente 90% das
conversas contendo exatamente 8 turnos; (iii) estilo lexical homogêneo sem variação de persona
ou registro; e (iv) distribuições de sentimento desacopladas da intenção, com proporções
uniformes de 33% para cada polaridade, independentemente da classe.

**Expansão sintética controlada.** Para superar essas limitações, realizamos uma expansão
sintética utilizando um modelo de linguagem de grande escala (Claude Sonnet, Anthropic) em
modo batch offline. A expansão foi conduzida com controles rigorosos sobre as seguintes
variáveis de geração:

- *Variabilidade de turnos:* as conversas geradas seguem uma distribuição log-normal com
  média de 8 turnos e desvio padrão de 4, produzindo conversas com 4 a 20 turnos — refletindo
  a variabilidade natural observada em operações de centrais de atendimento.
- *Distribuição de classes:* adotamos uma distribuição intencionalmente desbalanceada que
  reproduz padrões típicos de centrais de atendimento, onde reclamações e dúvidas predominam
  sobre elogios e saudações.
- *Variabilidade lexical:* cada conversa gerada é condicionada a uma de cinco personas
  linguísticas — formal, informal, irritado, idoso e jovem/gírias — introduzindo diversidade
  de registro e vocabulário.
- *Sentimento condicionado à intenção:* as distribuições de sentimento são condicionadas à
  classe de intenção. Por exemplo, conversas de reclamação exibem 65% de sentimento negativo
  e 35% neutro, enquanto conversas de elogio exibem 75% de sentimento positivo e 25% neutro.

É essencial reconhecer que o uso de dados sintéticos impõe uma limitação epistemológica:
conclusões derivadas deste corpus são válidas dentro do escopo metodológico definido e não
podem ser generalizadas incondicionalmente para dados conversacionais reais. Mitigamos essa
limitação por meio do protocolo de validação de robustez descrito na Seção 5.1.4 e da
auditoria de dados descrita na Seção 5.1.3.

### 5.1.2 Especificação do Corpus Pós-Auditoria

O corpus expandido bruto continha 2.257 conversas distribuídas em 9 classes de intenção.
Antes de qualquer experimentação, conduzimos uma auditoria de dados abrangente (Fase 1) para
identificar e remover registros problemáticos. O processo de auditoria e seus resultados são
descritos na Seção 5.1.3. Após a auditoria, o corpus foi reduzido para **2.122 conversas**
distribuídas em **8 classes de intenção**. A Tabela 5.1 resume a especificação do corpus
pós-auditoria.

**Tabela 5.1** — Especificação do corpus pós-auditoria.

| Dimensão | Valor |
|---|---|
| Total de conversas | 2.122 |
| Conversas originais | 847 |
| Conversas sintéticas | 1.275 |
| Turnos por conversa | 4–20 (log-normal, $\mu=8$, $\sigma=4$) |
| Classes de intenção | 8 |
| Idioma | PT-BR (informal, com diacríticos e gírias) |
| Anotação por conversa | Intenção + sentimento + setor |
| Sementes aleatórias | 5 sementes: [13, 42, 123, 2024, 999] |

As 8 classes de intenção retidas após a auditoria são: `cancelamento`, `compra`,
`duvida_produto` (dúvida sobre produto), `duvida_servico` (dúvida sobre serviço), `elogio`,
`reclamacao` (reclamação), `saudacao` (saudação) e `suporte_tecnico` (suporte técnico). A
Tabela 5.2 apresenta a distribuição de classes.

**Tabela 5.2** — Distribuição de classes de intenção no corpus pós-auditoria.

| Intenção | Contagem | Proporção |
|---|---|---|
| reclamacao | ~424 | ~20,0% |
| duvida_produto | ~382 | ~18,0% |
| duvida_servico | ~361 | ~17,0% |
| suporte_tecnico | ~318 | ~15,0% |
| compra | ~212 | ~10,0% |
| cancelamento | ~170 | ~8,0% |
| saudacao | ~149 | ~7,0% |
| elogio | ~106 | ~5,0% |

A distribuição desbalanceada é metodologicamente relevante para dois objetivos: (i) avaliar o
comportamento do classificador em classes minoritárias, um cenário frequente em aplicações de
produção; e (ii) testar a eficácia de regras determinísticas (H3) em classes críticas de baixa
frequência, como `cancelamento` e `elogio`.

### 5.1.3 Auditoria de Dados (Fase 1)

Antes de executar qualquer experimento, conduzimos uma auditoria de dados rigorosa que removeu
135 registros do corpus original de 2.257 conversas. A auditoria abordou três categorias de
problemas de qualidade de dados:

1. **Remoção de duplicatas.** Conversas duplicadas exatas e quase-duplicatas foram
   identificadas e removidas utilizando impressão digital de texto (MinHash) e análise de
   identificadores de conversa. Duplicatas entre os corpora original e sintético foram
   priorizadas para remoção, a fim de preservar a diversidade do corpus.

2. **Remoção de contaminação.** A expansão sintética utilizou exemplares few-shot extraídos
   do corpus original. Qualquer registro identificado como exemplar few-shot que vazou para
   as partições de avaliação foi removido para prevenir contaminação entre treino e teste. A
   contaminação foi detectada por correspondência de prefixos de `conversation_id`
   (`conv_synth_*`) e metadados de `source_file` contra os logs de geração.

3. **Consolidação da taxonomia.** A classe `outros` — presente na taxonomia original de 9
   classes — foi eliminada após revisão humana confirmar que suas instâncias eram ambíguas,
   rotuladas incorretamente ou representáveis pelas 8 classes restantes. A auditoria humana
   alcançou uma taxa de confirmação de $\geq 96,7\%$ para rótulos retidos, estabelecendo o
   corpus pós-auditoria como a verdade de referência autoritativa para todos os experimentos
   subsequentes.

A auditoria foi concluída em 12/03/2026 com aprovação de revisão humana. Todos os experimentos
relatados nesta dissertação utilizam exclusivamente o corpus pós-auditoria de 2.122 registros.

### 5.1.4 Pré-processamento

O pipeline de pré-processamento segue cinco estágios sequenciais, implementados no framework
TalkEx (Capítulo 4):

1. **Normalização de texto.** Aplicamos a função `normalize_for_matching()` do módulo
   `talkex.text_normalization`, que realiza conversão para minúsculas e remoção de diacríticos
   via decomposição Unicode NFD. Essa normalização é essencial para o português brasileiro,
   onde variações como "não"/"nao" e "cancelamento"/"cancelámento" devem ser tratadas como
   equivalentes para correspondência lexical.

2. **Segmentação de turnos.** O dataset contém turnos pré-estruturados com atribuição de
   falante. Quando necessário, heurísticas de alternância de falante reconstroem a segmentação
   utilizando o `TurnSegmenter` do módulo `talkex.segmentation`.

3. **Construção de janelas de contexto.** Janelas deslizantes com tamanho, passo e alinhamento
   de falante configuráveis são geradas utilizando o `SlidingWindowBuilder` do módulo
   `talkex.context`. A configuração padrão utiliza janelas de 5 turnos com passo 2.

4. **Geração de embeddings.** Representações vetoriais densas são geradas em múltiplos
   níveis de granularidade (turno, janela, conversa) utilizando o encoder
   `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões). Os embeddings são armazenados em
   cache para reprodutibilidade e eficiência computacional.

5. **Indexação.** Índices paralelos são construídos para recuperação lexical (BM25, via
   `rank-bm25`) e recuperação semântica (ANN, via FAISS com índice flat).

### 5.1.5 Partições, Sementes e Reprodutibilidade

Adotamos **cinco sementes aleatórias** [13, 42, 123, 2024, 999] para todos os experimentos.
Para cada semente, uma partição estratificada treino/validação/teste (70%/15%/15%) é gerada,
preservando a distribuição de classes em cada partição. Todos os resultados são reportados como
média $\pm$ desvio padrão entre as cinco sementes, com significância estatística avaliada via
testes de postos sinalizados de Wilcoxon sobre as observações pareadas por semente.

A decisão de utilizar cinco sementes em vez de uma única semente fixa (como no desenho
experimental original) foi motivada pela necessidade de estimar a variância devida ao
particionamento aleatório e de possibilitar testes estatísticos pareados não paramétricos. O
protocolo de cinco sementes fornece cinco observações pareadas por comparação, que é o mínimo
requerido para um teste de Wilcoxon bilateral com $\alpha = 0,05$.

**Integridade da partição no nível da janela.** Nos experimentos H2–H4 e no estudo de ablação,
a unidade de classificação é a **janela de contexto** (5 turnos, passo 2), não a conversa
completa. Janelas sobrepostas da mesma conversa poderiam introduzir vazamento de dados se
janelas da mesma conversa aparecessem tanto no conjunto de treino quanto no de teste. Para
prevenir isso, o particionamento é realizado **no nível da conversa** antes da geração de
janelas. Cada conversa pertence inteiramente a uma única partição; janelas são geradas apenas
após o particionamento. Nenhuma janela de teste compartilha turnos com qualquer janela de
treino.

**Herança de rótulos e supervisão fraca.** Os rótulos de intenção são atribuídos no nível da
conversa. Ao gerar janelas deslizantes, cada janela **herda** o rótulo de sua conversa de
origem. Isso constitui uma forma de supervisão fraca: janelas intermediárias (por exemplo,
turnos de resolução ou esclarecimento) podem não conter evidência lexical ou semântica direta
da intenção anotada. Esta é uma limitação reconhecida, compartilhada com trabalhos anteriores
que transferem rótulos do nível de documento para segmentos (Rayo et al., 2024).

**Agregação de janela para conversa.** O treinamento do classificador opera no **nível da
janela**: cada janela é um exemplo de treinamento com suas próprias features. A avaliação,
entretanto, opera no **nível da conversa**: as probabilidades de classe preditas para cada
janela são agregadas por média (média das probabilidades de classe), e a classe final é
determinada por argmax sobre a distribuição média. Essa estratégia impede que qualquer janela
individual com uma predição extrema domine o resultado — uma escolha conservadora alinhada com
a natureza ruidosa da supervisão fraca via herança de rótulos.

---

## 5.2 Métricas de Avaliação

A avaliação requer métricas distintas para cada componente do pipeline. Organizamos as métricas
em quatro famílias correspondentes às dimensões avaliadas pelas hipóteses H1–H4.

### 5.2.1 Métricas de Recuperação (H1)

Para a avaliação de recuperação híbrida, adotamos métricas padrão de recuperação de informação
(Manning et al., 2008):

**Tabela 5.3** — Métricas de avaliação de recuperação.

| Métrica | Definição | Justificativa |
|---|---|---|
| Recall@K | Fração de documentos relevantes recuperados no top-K | Mede a cobertura para estágios subsequentes do pipeline |
| Precision@K | Fração dos documentos no top-K que são relevantes | Mede a qualidade do conjunto de resultados |
| MRR | Mean Reciprocal Rank — média de $1/r_i$ onde $r_i$ é a posição do primeiro resultado relevante | Mede a rapidez com que o sistema retorna o primeiro resultado útil |
| nDCG@K | Normalized Discounted Cumulative Gain | Mede a qualidade do ranqueamento considerando posições relativas |

Todas as métricas são avaliadas para $K \in \{5, 10, 20\}$. MRR é a métrica primária para H1,
pois captura a qualidade percebida pelo usuário do sistema de recuperação em cenários
operacionais onde o primeiro resultado relevante é o mais importante.

### 5.2.2 Métricas de Classificação (H2, H3)

**Tabela 5.4** — Métricas de avaliação de classificação.

| Métrica | Definição | Justificativa |
|---|---|---|
| Macro-F1 | Média aritmética dos scores F1 por classe, com ponderação igual | Sensível ao desempenho em classes minoritárias; métrica primária |
| Micro-F1 | F1 computado sobre todas as instâncias, ponderado por volume | Reflete o desempenho geral considerando o desbalanceamento de classes |
| Precisão por classe | Proporção de predições positivas corretas por classe | Mede o custo de falsos positivos por classe |
| Recall por classe | Proporção de instâncias positivas corretamente identificadas por classe | Mede a cobertura por classe |

A escolha de Macro-F1 como métrica primária é deliberada: em um corpus desbalanceado, Micro-F1
pode mascarar desempenho insatisfatório em classes minoritárias. Macro-F1 atribui peso igual a
cada classe, garantindo que os classificadores devem apresentar bom desempenho em toda a
taxonomia — incluindo intenções raras como `elogio` (5%) e `cancelamento` (8%) — para
alcançar scores elevados.

### 5.2.3 Métricas de Regras (H3)

**Tabela 5.5** — Métricas de avaliação de regras.

| Métrica | Definição |
|---|---|
| Precisão da regra | Proporção de ativações corretas sobre o total de ativações |
| Recall da regra | Proporção de verdadeiros positivos capturados pela regra |
| F1 da regra | Média harmônica da precisão e recall da regra |
| Cobertura | Percentual de conversas em que pelo menos uma regra produziu evidência |

### 5.2.4 Métricas de Eficiência (H4)

**Tabela 5.6** — Métricas de avaliação de eficiência.

| Métrica | Definição |
|---|---|
| Custo por janela | Tempo de processamento em milissegundos por unidade de classificação |
| $\Delta$F1 | Diferença de Macro-F1 entre os pipelines uniforme e em cascata |
| % resolvidos por estágio | Proporção de janelas resolvidas em cada estágio da cascata |

A combinação de custo por janela e $\Delta$F1 possibilita a construção da fronteira de Pareto
que é central para a avaliação de H4: identificar configurações que reduzem o custo com
degradação de qualidade aceitável.

---

## 5.3 Protocolo Experimental para H1 — Recuperação Híbrida

### 5.3.1 Hipótese

> *A recuperação híbrida (BM25 + ANN com fusão de scores) supera tanto o BM25 isolado quanto
> a busca semântica isolada em MRR quando aplicada a conversas de atendimento ao cliente em
> PT-BR.*

Esta hipótese se fundamenta na complementaridade teórica entre busca lexical e semântica,
amplamente documentada na literatura (Lin et al., 2021; Formal et al., 2021; Rayo et al.,
2025). BM25 se destaca na correspondência exata de termos — nomes de produtos, códigos,
palavras-chave regulatórias — enquanto a recuperação densa captura paráfrases, intenção
implícita e variação linguística. A hipótese postula que a combinação supera ambas as
abordagens isoladas no domínio conversacional.

### 5.3.2 Sistemas Comparados

Definimos sistemas de recuperação abrangendo três categorias:

**Tabela 5.7** — Sistemas de recuperação comparados em H1.

| Sistema | Categoria | Descrição |
|---|---|---|
| BM25-base | Lexical | BM25 padrão com conversão para minúsculas e remoção de stopwords |
| ANN | Semântico | Busca por vizinhos mais próximos aproximados com embeddings paraphrase-multilingual-MiniLM-L12-v2 (384 dims) |
| Hybrid-LINEAR | Híbrido | BM25 + ANN, fusão linear de scores: $S = \alpha \cdot s_{bm25} + (1-\alpha) \cdot s_{ann}$ |
| Hybrid-RRF | Híbrido | BM25 + ANN, Reciprocal Rank Fusion (Cormack et al., 2009) |

### 5.3.3 Espaço de Parâmetros

Para os sistemas híbridos, variamos o peso de fusão $\alpha$ (peso do BM25) no intervalo
[0,05, 0,95] em incrementos de 0,05. Essa varredura abrangente identifica o trade-off ótimo
entre contribuições lexical e semântica. Com base em trabalhos anteriores (Rayo et al., 2025),
esperamos que o $\alpha$ ótimo esteja no intervalo [0,20, 0,40], ponderando a contribuição
lexical moderadamente enquanto permite que a busca semântica predomine.

### 5.3.4 Construção do Ground Truth

A avaliação de recuperação requer consultas com documentos relevantes anotados. Construímos o
ground truth da seguinte forma: consultas são derivadas da taxonomia de intenções (por exemplo,
"cancelamento", "reclamação"), e a relevância é definida pela correspondência entre a intenção
da consulta e a anotação de intenção no nível da conversa de cada janela indexada. Isso
possibilita a construção automática do ground truth sem anotação manual adicional.

### 5.3.5 Critérios de Confirmação

H1 é **confirmada** se o melhor sistema híbrido superar todos os sistemas isolados em MRR com
diferença estatisticamente significativa ($p < 0,05$ no teste de postos sinalizados de Wilcoxon
entre 5 sementes).

H1 é **parcialmente confirmada** se o sistema híbrido superar em algumas métricas mas não em
outras, ou se a diferença não alcançar significância estatística.

H1 é **refutada** se um sistema isolado (BM25 ou ANN) igualar ou superar o melhor sistema
híbrido na métrica primária (MRR).

---

## 5.4 Protocolo Experimental para H2 — Representações Combinadas de Features

### 5.4.1 Hipótese

> *Classificadores que combinam features lexicais com embeddings densos pré-treinados alcançam
> Macro-F1 superior em comparação com classificadores que utilizam apenas features lexicais,
> de forma consistente entre múltiplas famílias de classificadores.*

Esta hipótese se fundamenta na complementaridade entre representações lexicais esparsas
(TF-IDF, scores BM25) e representações densas pré-treinadas (embeddings de sentenças).
Features lexicais capturam correspondências exatas de termos e padrões de frequência;
embeddings densos capturam relações semânticas, paráfrases e variação linguística. A hipótese
postula que a combinação supera features exclusivamente lexicais independentemente da
arquitetura do classificador.

### 5.4.2 Configurações de Features

**Tabela 5.8** — Configurações de features comparadas em H2.

| Configuração | Features |
|---|---|
| lexical-only | Vetores TF-IDF + scores BM25 contra protótipos de classe |
| embedding-only | Embeddings congelados de 384 dimensões (paraphrase-multilingual-MiniLM-L12-v2) |
| lexical+embedding | Concatenação de features lexicais e de embedding |
| lexical+embedding+structural | Conjunto completo de features: lexical + embedding + features estruturais da conversa (proporção de falantes, contagem de turnos, posição) |

### 5.4.3 Classificadores

Para cada configuração de features, treinamos e avaliamos três classificadores:

**Tabela 5.9** — Classificadores utilizados em H2.

| Classificador | Configuração | Justificativa |
|---|---|---|
| Regressão Logística | Parâmetros padrão do scikit-learn | Baseline linear; avalia a separabilidade das features |
| LightGBM | 100 estimadores, 31 folhas | Gradient boosting otimizado para features heterogêneas |
| MLP | 2 camadas ocultas (scikit-learn) | Baseline neural para interações de features densas |

A configuração do LightGBM (`n_estimators=100, num_leaves=31`) foi fixada a priori com base
em padrões default e é utilizada uniformemente em todos os experimentos H2, H3 e ablação.
Nenhum ajuste de hiperparâmetros por experimento foi realizado no conjunto de teste.

### 5.4.4 Parâmetros da Janela de Contexto

A janela de contexto é um componente central da representação multinível. A configuração
padrão utiliza janelas de 5 turnos com passo 2, baseada na observação empírica de que a
maioria das intenções conversacionais se manifesta dentro de 3 a 7 turnos. Embeddings no nível
da janela são gerados por mean pooling dos embeddings no nível do turno dentro de cada janela.

### 5.4.5 Critérios de Confirmação

H2 é **confirmada** se configurações lexical+embedding superarem consistentemente configurações
lexical-only em Macro-F1, com diferença estatisticamente significativa ($p < 0,05$), em pelo
menos duas das três famílias de classificadores.

H2 é **refutada** se configurações lexical-only igualarem ou superarem configurações
lexical+embedding na maioria dos classificadores.

---

## 5.5 Protocolo Experimental para H3 — Regras Determinísticas

### 5.5.1 Hipótese

> *A adição de um motor de regras semânticas (DSL → AST) ao pipeline híbrido de ML melhora o
> Macro-F1 ao mesmo tempo em que fornece rastreabilidade de evidência por decisão.*

Esta hipótese é motivada pela observação de que classificadores estatísticos, embora eficazes
no caso geral, apresentam limitações em cenários onde (i) o custo de falsos positivos é
assimétrico, (ii) requisitos de conformidade exigem decisões auditáveis e (iii) padrões
linguísticos específicos do domínio são conhecidos a priori. Sistemas baseados em regras,
apesar da cobertura limitada, oferecem precisão controlável e rastreabilidade completa —
propriedades complementares aos modelos estatísticos.

### 5.5.2 Conjunto de Regras

O experimento H3 utiliza duas regras lexicais implementadas no motor de regras do TalkEx:

1. **rule_cancel** — detecta intenção explícita de cancelamento via padrões lexicais
   (palavras-chave: "cancelar", "encerrar", "desistir", "rescindir" e variantes). Direciona
   a classe `cancelamento`.

2. **rule_complaint** — detecta padrões de reclamação via padrões lexicais (palavras-chave:
   "reclamação", "absurdo", "desrespeito", "procon" e variantes). Direciona a classe
   `reclamacao`.

Essas regras foram definidas antes de qualquer avaliação no conjunto de teste para prevenir
viés de construção. O conjunto de regras é deliberadamente mínimo (2 regras lexicais) para
estabelecer um limite inferior da contribuição das regras. O motor de regras do TalkEx suporta
famílias adicionais de predicados — semânticos (`intent_score`, `embedding_similarity`),
estruturais e contextuais (`repeated_in_window`, `occurs_after`) — que não foram exercitadas
em H3. A expansão do conjunto de regras com predicados semânticos é identificada como uma
direção primária para trabalhos futuros.

### 5.5.3 Estratégias de Integração Comparadas

**Tabela 5.10** — Estratégias de integração de regras comparadas em H3.

| Configuração | Descrição |
|---|---|
| ML-only | Melhor classificador LightGBM de H2, sem regras |
| Rules-only | Apenas ativações de regras, sem classificador ML |
| ML+Rules-feature | Classificador ML com flags binários de ativação de regras como features adicionais |
| ML+Rules-override | Classificador ML com decisões de regras sobrepondo predições ML quando regras são ativadas |

A estratégia regras-como-features trata as ativações de regras como features binárias
adicionais no espaço de features do LightGBM, permitindo que o classificador aprenda a
informatividade de cada regra. A estratégia regras-como-sobreposição aplica as decisões das
regras diretamente quando uma regra é disparada, contornando a predição do ML. Essa comparação
isola se as regras são melhor utilizadas como sinais de entrada ou como sobreposições de
decisão.

### 5.5.4 Critérios de Confirmação

H3 é **confirmada** se qualquer configuração ML+Rules superar ML-only em Macro-F1 com
significância estatística ($p < 0,05$).

H3 é **inconclusiva** se ML+Rules apresentar uma tendência positiva mas não alcançar
significância estatística em $\alpha = 0,05$.

H3 é **refutada** se as regras degradarem o Macro-F1 em todas as estratégias de integração.

---

## 5.6 Protocolo Experimental para H4 — Inferência em Cascata

### 5.6.1 Hipótese

> *Um pipeline de inferência em cascata reduz o custo computacional médio por janela em
> comparação com o pipeline uniforme, com degradação de qualidade aceitável ($\Delta$F1 < 2
> pp).*

Esta hipótese se fundamenta no princípio de inferência em cascata (Viola & Jones, 2001;
Matveeva et al., 2006): aplicar estágios de processamento progressivamente mais custosos,
resolvendo casos fáceis precocemente com modelos baratos e reservando modelos custosos para
casos ambíguos.

### 5.6.2 Configurações do Pipeline

**Pipeline uniforme (baseline).** Todas as janelas passam por todos os estágios de
processamento independentemente da complexidade:

$$\text{Window} \to \text{Normalization} \to \text{Embeddings} \to \text{BM25 + ANN} \to \text{Fusion} \to \text{Classification} \to \text{Rules} \to \text{Output}$$

**Pipeline em cascata.** Uma cascata de dois estágios onde janelas classificadas por um modelo
leve no Estágio 1 (LightGBM lexical-only) com confiança acima do limiar $\theta$ não são
escaladas para o Estágio 2 (pipeline completo com features de embedding + lexicais):

- **Estágio 1** — Classificador LightGBM lexical-only. Se $\max(P(y|x)) \geq \theta$, a
  janela é resolvida. Custo: apenas extração de features lexicais.
- **Estágio 2** — Classificador LightGBM completo com features lexicais + embedding. Aplicado
  apenas às janelas não resolvidas no Estágio 1.

### 5.6.3 Espaço de Limiares

Os limiares de confiança são variados em: $\theta \in \{0,50, 0,55, 0,60, ..., 0,90\}$, uma
grade de 9 valores que explora todo o trade-off entre taxa de resolução precoce e qualidade de
classificação.

### 5.6.4 Critérios de Confirmação

H4 é **confirmada** se pelo menos uma configuração de limiar alcançar redução de custo
$\geq 20\%$ com $\Delta$F1 $< 2$ pontos percentuais.

H4 é **refutada** se nenhuma configuração alcançar redução de custo significativa sem
degradação de qualidade inaceitável, ou se o modelo de custo revelar impedimentos estruturais
ao benefício da cascata.

---

## 5.7 Desenho do Estudo de Ablação

O estudo de ablação quantifica a contribuição marginal de cada família de features para o
desempenho de classificação do pipeline completo. Partindo da configuração completa (features
lexicais + embedding + estruturais + ativação de regras com LightGBM), removemos
sistematicamente uma família de features por vez e medimos a mudança resultante no Macro-F1.

**Tabela 5.11** — Configurações de ablação.

| Configuração | Features Removidas | Features Retidas |
|---|---|---|
| full_pipeline | Nenhuma | Lexical + Embedding + Estrutural + Regras |
| no_embeddings | Features de embedding | Lexical + Estrutural + Regras |
| no_lexical | Features lexicais | Embedding + Estrutural + Regras |
| no_rules | Features de ativação de regras | Lexical + Embedding + Estrutural |
| no_structural | Features estruturais | Lexical + Embedding + Regras |

Para cada configuração, o classificador LightGBM (100 estimadores, 31 folhas) é retreinado e
avaliado utilizando o mesmo protocolo de 5 sementes dos experimentos principais. A ablação é
aditiva: cada configuração remove exatamente uma família, possibilitando a atribuição direta
da contribuição marginal. O ranking esperado de contribuições, baseado em considerações
teóricas, é: embeddings >> lexical > regras ≈ estrutural.

---

## 5.8 Framework de Análise Estatística

### 5.8.1 Testes de Significância

Todas as comparações pareadas utilizam o **teste de postos sinalizados de Wilcoxon**, um teste
não paramétrico para amostras pareadas que não assume normalidade (Wilcoxon, 1945). Com 5
observações pareadas (uma por semente), o p-value mínimo alcançável é $2/2^5 = 0,0625$ para
um teste bilateral, o que estabelece uma limitação inerente de poder estatístico. Quando todas
as 5 diferenças pareadas têm o mesmo sinal, $p = 0,0625$; significância estatística em
$\alpha = 0,05$ requer que as diferenças observadas sejam consistentemente na mesma direção
entre sementes E suficientemente grandes em magnitude para que a estatística de postos
sinalizados exceda o valor crítico.

**Nota de correção.** Reconhecemos que o teste de Wilcoxon com cinco sementes possui poder
estatístico limitado. Quando o teste resulta em $p > 0,05$ com uma tendência direcional
consistente, interpretamos isso como inconclusivo em vez de como evidência contra a hipótese
alternativa — seguindo a recomendação de Demsar (2006) para experimentos de ML com tamanhos
amostrais pequenos.

### 5.8.2 Intervalos de Confiança

Para a métrica primária de cada hipótese, reportamos **intervalos de confiança bootstrap de
95%** computados com 10.000 reamostras sobre as diferenças pareadas por semente. O intervalo
de confiança bootstrap fornece uma estimativa não paramétrica do intervalo plausível do
verdadeiro tamanho do efeito. Quando o intervalo de confiança exclui zero, ele fornece
evidência adicional além do p-value isolado.

### 5.8.3 Tamanho do Efeito

Para comparações estatisticamente significativas, reportamos o tamanho do efeito $r$ computado
como $r = Z / \sqrt{N}$, onde $Z$ é a estatística padronizada de Wilcoxon e $N$ é o número de
observações pareadas. Os tamanhos de efeito são classificados seguindo as convenções de Cohen:
pequeno ($r \approx 0,1$), médio ($r \approx 0,3$), grande ($r \approx 0,5$).

---

## 5.9 Ameaças à Validade

Documentamos explicitamente as ameaças à validade organizadas nas três categorias clássicas de
Cook e Campbell (1979): validade interna, externa e de construto.

### 5.9.1 Validade Interna

**Tabela 5.12** — Ameaças à validade interna e mitigações.

| Ameaça | Descrição | Mitigação |
|---|---|---|
| Overfitting de hiperparâmetros | Seleção de hiperparâmetros que maximizam o desempenho no teste por acaso | Configuração fixa do LightGBM (100t/31l) utilizada uniformemente; nenhum ajuste por experimento no conjunto de teste |
| Viés de construção de regras | Regras definidas com conhecimento do conjunto de teste | Regras definidas e finalizadas antes de qualquer avaliação no teste; data de criação registrada |
| Viés de seleção de métricas | Seleção post-hoc de métricas favoráveis ao sistema proposto | Todas as métricas definidas a priori no desenho experimental; métricas onde o sistema perde são reportadas com igual destaque |
| Bugs de implementação | Erros de código que invalidam resultados | Suíte automatizada abrangente de testes (1.883+ testes unitários no TalkEx); quality gates (ruff, mypy, pytest) aplicados antes de cada experimento |
| Seleção de sementes aleatórias | Sementes escolhidas para favorecer certos resultados | Sementes [13, 42, 123, 2024, 999] selecionadas antes de qualquer experimento; resultados reportados para TODAS as sementes |
| Vazamento de dados | Contaminação treino-teste por janelas sobrepostas | Particionamento no nível da conversa aplicado antes da geração de janelas; auditoria de contaminação na Fase 1 |
| Contaminação few-shot | Exemplares de geração sintética vazando para partições de teste | Detecção de contaminação via prefixos de conversation_id e metadados de source_file; registros contaminados removidos na auditoria |

### 5.9.2 Validade Externa

**Tabela 5.13** — Ameaças à validade externa e mitigações.

| Ameaça | Descrição | Mitigação |
|---|---|---|
| Dados sintéticos | O corpus é parcialmente sintético; conclusões podem não generalizar para conversas reais | Divulgação explícita; análise de robustez comparando subconjuntos originais vs sintéticos; discussão de diferenças potenciais |
| Idioma único | Resultados são específicos para PT-BR | Análise de componentes específicos do idioma (ablação de normalização de diacríticos); identificação de módulos agnósticos vs específicos do idioma |
| Domínio único | Conversas de atendimento ao cliente possuem características específicas | Discussão explícita de premissas do domínio; identificação de componentes portáveis vs específicos do domínio |
| Escala | Experimentos em escala de pesquisa (2.122 conversas), não em escala de produção (milhões) | Análise teórica de complexidade computacional; medições empíricas de throughput; discussão de escalabilidade |
| Dataset único | Todos os experimentos utilizam uma única fonte de dataset | Reconhecido como limitação primária; experimentos LODO e validação cruzada k-fold planejados como extensões |

### 5.9.3 Validade de Construto

**Tabela 5.14** — Ameaças à validade de construto e mitigações.

| Ameaça | Descrição | Mitigação |
|---|---|---|
| Lacuna métrica-utilidade | F1 e MRR podem não refletir a utilidade percebida por operadores humanos | Inclusão de análise qualitativa com exemplos concretos; discussão da correspondência métrica-utilidade |
| Explicabilidade não avaliada formalmente | Rastreabilidade de evidência é reivindicada mas não medida com métricas formais de explicabilidade | Critérios qualitativos para qualidade de evidência; exemplos completos de saída com metadados |
| Modelo de custo simplificado | Custo baseado em tempo (ms) não captura custos de GPU, memória e infraestrutura | Discussão complementar de custos de memória e infraestrutura onde relevante |
| Ruído de supervisão fraca | Herança de rótulos da conversa para a janela introduz ruído nos rótulos | Limitação reconhecida; agregação janela-para-conversa mitiga o efeito; análise por classe identifica intenções afetadas |
| Cobertura de regras | Apenas 2 regras lexicais testadas; não representa a capacidade completa do motor de regras | Reconhecido como a limitação mais significativa de H3; conjunto expandido de regras com predicados semânticos identificado como trabalho futuro |

---

## 5.10 Infraestrutura Experimental

### 5.10.1 Stack de Software

**Tabela 5.15** — Stack de software.

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Encoder de sentenças | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| BM25 | rank-bm25 + implementação TalkEx (`talkex.retrieval`) |
| ANN | FAISS (faiss-cpu), índice flat |
| Classificação | scikit-learn (LogReg, MLP), LightGBM |
| Motor de regras | TalkEx DSL → AST → executor (`talkex.rules`) |
| Avaliação | métricas scikit-learn, scripts de avaliação customizados |
| Testes estatísticos | scipy.stats (Wilcoxon), bootstrap (customizado) |
| Visualização | matplotlib, seaborn |
| Reprodutibilidade | Sementes fixas, pinagem de dependências via `pyproject.toml`, 1.883+ testes automatizados |

### 5.10.2 Ambiente de Execução

Todos os experimentos foram executados no Google Colab com runtime GPU Tesla T4 (15 GB VRAM).
A geração de embeddings utiliza aceleração GPU via PyTorch/CUDA por meio da biblioteca
sentence-transformers, enquanto a classificação LightGBM treina em CPU (~6 segundos). A suíte
completa de experimentos é concluída em menos de 1 hora. O uso do runtime GPU gratuito do
Google Colab valida a reivindicação de acessibilidade da tese: o pipeline experimental completo
é reprodutível sem infraestrutura dedicada de ML, requerendo apenas um navegador web e uma
conta Google.

### 5.10.3 Disponibilidade de Código e Dados

O código-fonte do TalkEx está disponível como repositório Git, incluindo o pipeline completo
de NLP, motor de regras (DSL, parser, AST, executor) e scripts experimentais no diretório
`experiments/`. O dataset é referenciado por seu identificador HuggingFace
(`RichardSakaguchiMS/brazilian-customer-service-conversations`), e o procedimento de expansão
sintética é documentado para reprodução. O executor unificado de experimentos
(`experiments/scripts/run_experiment.py`) executa todos os experimentos H1–H4 e o estudo de
ablação em uma única invocação.

---

## 5.11 Resumo

Este capítulo especificou o protocolo experimental completo para avaliação das quatro hipóteses
da tese. O desenho assegura: (i) comparação justa entre sistemas, com baselines apropriados e
variáveis controladas; (ii) rigor estatístico, com testes de significância não paramétricos e
intervalos de confiança bootstrap entre cinco sementes aleatórias; (iii) reprodutibilidade
completa, com sementes fixas, versões de software documentadas e scripts versionados; e (iv)
transparência sobre limitações, com documentação explícita de ameaças à validade.

Os quatro experimentos — recuperação híbrida (H1), representações combinadas de features (H2),
regras determinísticas (H3) e inferência em cascata (H4) — compartilham o mesmo dataset
pós-auditoria e infraestrutura, mas são avaliados com métricas e protocolos específicos de
cada hipótese. Os critérios de confirmação foram definidos a priori, antes da execução, para
prevenir viés de interpretação post-hoc. O estudo de ablação complementa o teste de hipóteses
quantificando a contribuição marginal de cada família de features para o pipeline completo.

O Capítulo 6 apresenta os resultados obtidos e sua análise crítica.
