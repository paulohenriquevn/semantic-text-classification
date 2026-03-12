# Capítulo 2 — Fundamentação Teórica

Este capítulo estabelece os conceitos fundamentais que sustentam a arquitetura proposta nesta dissertação. Apresentamos de forma progressiva os pilares teóricos necessários para compreender como o TalkEx combina técnicas lexicais, semânticas e determinísticas em um pipeline unificado de análise de conversas. Cada seção introduz um conceito e, ao final, conecta-o ao papel que desempenha na arquitetura.

---

## 2.1 Representação Vetorial de Texto

### 2.1.1 Intuição Geométrica

O processamento computacional de linguagem natural exige que textos sejam convertidos em representações numéricas sobre as quais operações matemáticas possam ser realizadas. A ideia central é mapear unidades linguísticas --- palavras, sentenças ou documentos --- para vetores em um espaço de alta dimensionalidade, de modo que relações semânticas entre textos se traduzam em relações geométricas entre vetores. Se duas sentenças expressam significados próximos, seus vetores devem ocupar regiões vizinhas nesse espaço; se expressam significados distintos, devem estar afastados.

Essas representações numéricas são chamadas **embeddings**. Formalmente, um embedding é uma função $f: \mathcal{T} \rightarrow \mathbb{R}^d$ que mapeia um texto $t \in \mathcal{T}$ para um vetor denso de dimensão $d$. A qualidade de um embedding é medida pela capacidade de preservar relações semânticas: textos similares devem produzir vetores próximos segundo alguma métrica de distância ou similaridade.

### 2.1.2 Evolução Histórica das Representações

A história das representações vetoriais de texto pode ser organizada em cinco gerações, cada uma superando limitações fundamentais da anterior.

**Bag-of-Words (BoW).** A representação mais elementar trata cada documento como um multiconjunto de palavras, ignorando ordem e estrutura. Dado um vocabulário $V = \{w_1, w_2, \ldots, w_{|V|}\}$, um documento $d$ é representado por um vetor $\mathbf{x} \in \mathbb{N}^{|V|}$, onde cada componente $x_i$ indica a frequência do termo $w_i$ em $d$. Apesar de sua simplicidade, o BoW apresenta limitações severas: ignora a ordem das palavras (``o gato comeu o rato'' e ``o rato comeu o gato'' produzem a mesma representação), gera vetores esparsos de alta dimensionalidade e não captura relações semânticas --- ``cancelar'' e ``encerrar'' são tratados como dimensões completamente independentes.

**TF-IDF (Term Frequency -- Inverse Document Frequency).** Proposta como refinamento do BoW, a ponderação TF-IDF [Salton e Buckley, 1988] atribui pesos que refletem a importância relativa de cada termo. A intuição é que termos frequentes em um documento específico, mas raros no corpus geral, são mais informativos. O peso de um termo $t$ em um documento $d$ pertencente a um corpus $D$ é dado por:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

onde:

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

é a frequência relativa do termo no documento, e:

$$\text{IDF}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$$

penaliza termos que ocorrem em muitos documentos (como artigos e preposições). TF-IDF melhora significativamente a capacidade discriminativa em relação ao BoW simples, mas herda suas limitações fundamentais: continua ignorando ordem, sinonímia e contexto.

**Word2Vec e Embeddings Estáticos.** A primeira revolução em representação vetorial veio com o Word2Vec [Mikolov et al., 2013], que introduziu a ideia de aprender vetores densos de baixa dimensionalidade ($d \approx 100$--$300$) a partir da co-ocorrência de palavras em grandes corpora. A hipótese distribucional --- ``uma palavra é caracterizada pelas companhias que mantém'' [Firth, 1957] --- é formalizada em duas arquiteturas: Skip-gram, que prediz palavras de contexto dado um termo central, e CBOW (Continuous Bag of Words), que prediz o termo central dado o contexto.

A contribuição fundamental do Word2Vec foi demonstrar que relações semânticas emergem como regularidades algébricas no espaço vetorial. O exemplo canônico é a analogia $\vec{rei} - \vec{homem} + \vec{mulher} \approx \vec{rainha}$, que sugere que o modelo captura relações semânticas abstratas. Modelos subsequentes como GloVe [Pennington et al., 2014] e FastText [Bojanowski et al., 2017] refinariam essa abordagem, com o FastText sendo particularmente relevante para línguas morfologicamente ricas como o português, ao operar sobre subpalavras.

A limitação central dos embeddings estáticos é que cada palavra recebe um único vetor independentemente do contexto. A palavra ``banco'' em ``sentei no banco'' e ``abri uma conta no banco'' recebe a mesma representação, apesar de significados completamente distintos.

**BERT e Embeddings Contextuais.** O modelo BERT (Bidirectional Encoder Representations from Transformers) [Devlin et al., 2018] resolveu a limitação dos embeddings estáticos ao introduzir representações contextuais: cada token recebe um vetor diferente dependendo de seu contexto na sentença. BERT utiliza a arquitetura Transformer [Vaswani et al., 2017], composta por camadas de self-attention que permitem a cada token ``atender'' a todos os demais tokens da sequência, capturando dependências de longa distância de forma bidirecional.

O BERT é pré-treinado em duas tarefas: Masked Language Modeling (MLM), onde o modelo prediz tokens mascarados a partir do contexto, e Next Sentence Prediction (NSP), onde classifica se duas sentenças são consecutivas. Esse pré-treinamento sobre vastos corpora produz representações ricas que podem ser ajustadas (fine-tuned) para tarefas específicas. O impacto do BERT foi transformador: estabeleceu novos estados da arte em praticamente todas as tarefas de NLP e inaugurou a era dos modelos de linguagem pré-treinados de larga escala.

Entretanto, o BERT por si só não produz embeddings de sentença otimizados. A abordagem ingênua de usar o token [CLS] ou fazer mean pooling sobre os tokens de saída resulta em representações que frequentemente são inferiores a médias de GloVe para tarefas de similaridade semântica [Reimers e Gurevych, 2019].

**Sentence Transformers e Embeddings de Sentença.** O Sentence-BERT (SBERT) [Reimers e Gurevych, 2019] resolveu essa limitação ao propor uma arquitetura siamesa/triplet sobre o BERT, treinada especificamente para produzir embeddings de sentença semanticamente significativos. A ideia é que duas sentenças são codificadas independentemente por redes BERT compartilhadas, e a função de perda (contrastive loss ou triplet loss) força sentenças semanticamente similares a produzirem vetores próximos.

A contribuição prática do SBERT foi tornar viável a busca semântica em larga escala. Enquanto comparar duas sentenças com BERT via cross-encoding (concatenar ambas e classificar) exige $O(n^2)$ inferências para $n$ sentenças, com SBERT cada sentença é codificada uma única vez ($O(n)$ inferências), e comparações subsequentes são simples operações de similaridade de cosseno entre vetores pré-computados.

### 2.1.3 Modelos de Embedding para Retrieval

A partir do framework de Sentence Transformers, diversos modelos foram desenvolvidos especificamente para tarefas de retrieval, cada um com estratégias distintas de treinamento e arquitetura.

**E5 (EmbEddings from bidirEctional Encoder rEpresentations).** A família de modelos E5 [Wang et al., 2022] foi treinada com o paradigma de prefixos textuais: queries são prefixadas com ``query:'' e documentos com ``passage:'', permitindo ao modelo aprender representações assimétricas otimizadas para o matching query-documento. Essa assimetria é particularmente útil em cenários de retrieval onde a query é curta e o documento é longo.

**BGE (BAAI General Embedding).** Os modelos BGE [Xiao et al., 2023] fazem parte do pacote C-Pack e foram treinados com uma combinação de dados de retrieval, classificação e clustering. O BGE se destaca por seu equilíbrio entre qualidade e custo computacional, sendo frequentemente a escolha padrão em pipelines de produção. Além dos modelos de bi-encoding, o ecossistema BGE inclui um reranker (cross-encoder) que pode ser usado como estágio de refinamento.

**Instructor.** O modelo Instructor [Su et al., 2022] introduziu a ideia de condicionar o embedding a uma instrução textual que descreve a tarefa. Por exemplo, ``Represent the customer service query for classification:'' produz embeddings otimizados para classificação, enquanto ``Represent the document for retrieval:'' produz embeddings otimizados para busca. Essa flexibilidade torna o Instructor particularmente interessante para cenários multi-tarefa.

### 2.1.4 Estratégias de Pooling

Modelos baseados em Transformer produzem uma sequência de vetores --- um para cada token da entrada. Para obter uma única representação vetorial da sentença inteira, é necessário aplicar uma estratégia de **pooling** que agregue esses vetores individuais.

**Mean Pooling.** Calcula a média aritmética dos vetores de todos os tokens:

$$\mathbf{e}_{\text{mean}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i$$

onde $\mathbf{h}_i$ é o vetor de saída do token $i$ e $n$ é o número de tokens. Mean pooling é simples, estável e computacionalmente barato. Sua principal desvantagem é tratar todos os tokens com igual importância, diluindo o sinal de tokens críticos em sentenças longas.

**Max Pooling.** Seleciona, para cada dimensão, o valor máximo entre todos os tokens:

$$\mathbf{e}_{\text{max}}[j] = \max_{i=1}^{n} \mathbf{h}_i[j]$$

Max pooling destaca os sinais mais fortes em cada dimensão, mas pode amplificar ruído, especialmente em textos ruidosos como transcrições de fala.

**Attention Pooling.** Aprende pesos de importância para cada token, produzindo uma média ponderada:

$$\mathbf{e}_{\text{attn}} = \sum_{i=1}^{n} \alpha_i \mathbf{h}_i, \quad \alpha_i = \frac{\exp(s(\mathbf{h}_i))}{\sum_{j=1}^{n} \exp(s(\mathbf{h}_j))}$$

onde $s(\cdot)$ é uma função de score (tipicamente uma camada linear ou MLP). Lyu et al. [2025] demonstraram que attention pooling produz embeddings mais discriminativos para classificação, especialmente em textos longos onde a importância dos tokens varia significativamente. O custo adicional de treinamento e computação é compensado pelo ganho em tarefas onde o contexto é heterogêneo.

**Impacto na arquitetura TalkEx.** No TalkEx, adotamos mean pooling como baseline por sua estabilidade e simplicidade, reservando attention pooling para representações de janelas de contexto e conversas completas, onde a variação de importância entre turnos é mais pronunciada. Essa decisão segue a recomendação da literatura de iniciar com a abordagem mais simples e justificar complexidade adicional com evidência empírica.

---

## 2.2 Busca Lexical

### 2.2.1 Fundamentos: Term Frequency e Inverse Document Frequency

A busca lexical opera por correspondência de termos entre uma query e um corpus de documentos. A premissa é direta: um documento é relevante para uma query se compartilha termos com ela, e a relevância é proporcional à quantidade e importância dos termos compartilhados.

A forma mais simples de busca lexical conta ocorrências de termos. A **Term Frequency (TF)** mede quantas vezes um termo aparece em um documento. Documentos com mais ocorrências de um termo de busca são considerados mais relevantes. Entretanto, a frequência bruta favorece documentos longos e não diferencia termos informativos de termos triviais.

A **Inverse Document Frequency (IDF)** corrige esse viés ao penalizar termos que ocorrem em muitos documentos. A intuição é que um termo como ``atendimento'', que aparece em quase todas as conversas de um call center, carrega pouca informação discriminativa, enquanto um termo como ``ouvidoria'' ou ``fraude'', que aparece em poucas conversas, é altamente informativo.

A combinação TF-IDF, conforme formalizada na Seção 2.1.2, é a base sobre a qual modelos mais sofisticados de retrieval lexical foram construídos.

### 2.2.2 BM25: Formulação Matemática

O BM25 (Best Matching 25) [Robertson et al., 1996] é a função de ranking lexical mais utilizada em sistemas de recuperação de informação. Desenvolvido no contexto dos experimentos TREC, o BM25 refina a ponderação TF-IDF com duas contribuições fundamentais: saturação da frequência de termos e normalização por comprimento do documento.

Dado um documento $d$ e uma query $q = \{q_1, q_2, \ldots, q_m\}$ composta por $m$ termos, o score BM25 é calculado como:

$$\text{BM25}(d, q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

onde:

- $f(q_i, d)$ é a frequência do termo $q_i$ no documento $d$;
- $|d|$ é o comprimento do documento (em tokens);
- $\text{avgdl}$ é o comprimento médio dos documentos no corpus;
- $k_1$ é o parâmetro de saturação de frequência;
- $b$ é o parâmetro de normalização por comprimento.

A formulação com saturação de frequência de termos e normalização por comprimento de documento, parametrizada por $k_1$ e $b$, é detalhada em Robertson e Zaragoza (2009), que consolidam o desenvolvimento do BM25 desde os experimentos TREC originais.

O componente IDF é tipicamente calculado como:

$$\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}$$

onde $N$ é o número total de documentos no corpus e $n(q_i)$ é o número de documentos que contém o termo $q_i$.

### 2.2.3 Hiperparâmetros e Seu Efeito

Os dois hiperparâmetros do BM25 controlam aspectos complementares do ranking:

**Parâmetro $k_1$ (saturação de frequência).** Controla a rapidez com que o efeito da frequência de um termo satura. Com $k_1 = 0$, a frequência é completamente ignorada (modelo binário: o termo está presente ou não). Com $k_1 \to \infty$, a frequência nunca satura e o modelo se aproxima do TF puro. Valores típicos situam-se entre 1.2 e 2.0. O valor padrão $k_1 = 1.5$ oferece um equilíbrio razoável para a maioria dos domínios: a terceira ocorrência de um termo contribui menos que a segunda, que contribui menos que a primeira, refletindo a intuição de que repetições adicionais trazem retornos decrescentes de relevância.

**Parâmetro $b$ (normalização por comprimento).** Controla o grau de normalização pelo comprimento do documento. Com $b = 0$, não há normalização --- documentos longos são naturalmente favorecidos por terem mais chances de conter os termos da query. Com $b = 1$, a normalização é completa --- um termo em um documento duas vezes mais longo que a média recebe metade do peso. O valor padrão $b = 0.75$ representa um meio-termo: documentos longos são levemente penalizados, mas não ao ponto de ignorar informação adicional genuína.

No contexto de conversas de atendimento, onde o comprimento varia significativamente (desde interações de dois turnos até conversas com dezenas de turnos), a calibração de $b$ é particularmente relevante. Conversas mais longas tendem a cobrir mais tópicos, e a normalização excessiva pode penalizar injustamente conversas complexas que são legitimamente relevantes para queries multifacetadas.

### 2.2.4 Normalização Textual para BM25

A eficácia do BM25 depende criticamente do pré-processamento textual. Como o matching é literal --- baseado em correspondência exata de tokens --- qualquer variação superficial que não reflita diferença semântica deve ser normalizada antes da indexação e da busca.

As etapas de normalização relevantes incluem:

- **Conversão para minúsculas (lowercasing):** ``Cancelar'' e ``cancelar'' devem ser tratados como o mesmo token.
- **Remoção de diacríticos:** Essencial para o português brasileiro, onde ``nao'' e ``não'', ``cancelamento'' e ``cancelámento'' devem ser equivalentes. A normalização Unicode via decomposição NFD seguida de remoção de marcas combinantes é a abordagem padrão.
- **Remoção de pontuação:** Pontuação tipicamente não carrega significado discriminativo para retrieval.
- **Stemming ou lematização:** Reduzem variantes morfológicas a uma forma comum (``cancelar'', ``cancelamento'', ``cancelando'' $\to$ ``cancel''). Stemming é mais agressivo e rápido; lematização preserva formas linguisticamente válidas.
- **Remoção de stopwords:** Palavras funcionais de alta frequência (``de'', ``o'', ``que'') contribuem pouco para o ranking e podem ser removidas.

### 2.2.5 Forças e Fraquezas

A busca lexical, e o BM25 em particular, apresenta forças que explicam sua longevidade como baseline em sistemas de retrieval:

**Forças:**
- **Velocidade:** Índices invertidos permitem busca em tempo sub-linear, mesmo sobre corpora de milhões de documentos.
- **Interpretabilidade:** O score pode ser decomposto em contribuições por termo, permitindo explicar por que um documento foi ranqueado acima de outro.
- **Eficácia em vocabulário consistente:** Quando o domínio utiliza terminologia estável (códigos de produto, termos técnicos, nomes de procedimentos), o BM25 é altamente eficaz [Harris, 2025].
- **Sem treinamento:** Não requer dados rotulados, GPUs ou processos de fine-tuning.

**Fraquezas:**
- **Incapacidade de capturar sinonímia:** ``cancelar'' e ``encerrar'' são termos completamente independentes para o BM25.
- **Incapacidade de capturar intenção implícita:** Uma conversa sobre insatisfação com o serviço pode ser altamente relevante para a query ``risco de cancelamento'' sem conter a palavra ``cancelar''.
- **Sensibilidade à formulação:** A mesma necessidade de informação expressa com palavras diferentes pode produzir resultados dramaticamente distintos.

Harris [2025] demonstrou empiricamente que, em domínios com vocabulário estruturado e consistente (documentos médicos), o BM25 pode superar embeddings semânticos genéricos. Esse resultado reforça a importância de nunca descartar baselines lexicais sem evidência empírica --- um princípio central na arquitetura do TalkEx.

---

## 2.3 Busca Semântica

### 2.3.1 Dense Passage Retrieval

A busca semântica opera sobre representações densas (embeddings) em vez de correspondência de termos. A ideia é codificar queries e documentos no mesmo espaço vetorial e recuperar documentos cujos vetores são próximos ao vetor da query.

O framework de **Dense Passage Retrieval (DPR)** [Karpukhin et al., 2020] formalizou essa abordagem com uma arquitetura de bi-encoder: dois modelos BERT independentes --- um para queries ($E_Q$) e outro para documentos ($E_D$) --- são treinados conjuntamente para maximizar a similaridade entre queries e seus documentos relevantes. Dado um par (query $q$, documento $d^+$ relevante), a função de perda contrastiva é:

$$\mathcal{L} = -\log \frac{e^{\text{sim}(E_Q(q), E_D(d^+))}}{e^{\text{sim}(E_Q(q), E_D(d^+))} + \sum_{j=1}^{n} e^{\text{sim}(E_Q(q), E_D(d_j^-))}}$$

onde $d_j^-$ são documentos negativos (irrelevantes) e $\text{sim}(\cdot, \cdot)$ é uma função de similaridade, tipicamente o produto escalar ou a similaridade de cosseno.

A **similaridade de cosseno** entre dois vetores $\mathbf{u}$ e $\mathbf{v}$ é definida como:

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \cdot \sqrt{\sum_{i=1}^{d} v_i^2}}$$

O valor varia de $-1$ (vetores opostos) a $1$ (vetores idênticos), com $0$ indicando ortogonalidade (ausência de similaridade). Na prática, embeddings treinados com funções contrastivas tendem a produzir valores positivos para textos semanticamente relacionados.

### 2.3.2 Approximate Nearest Neighbor (ANN)

A busca exata por vizinhos mais próximos em espaços de alta dimensionalidade ($d \geq 100$) tem complexidade $O(n \cdot d)$, onde $n$ é o número de vetores no índice. Para corpora com milhões de documentos, isso se torna proibitivo. Algoritmos de **Approximate Nearest Neighbor (ANN)** trocam precisão por velocidade, retornando vizinhos ``aproximadamente'' próximos com garantias probabilísticas.

**HNSW (Hierarchical Navigable Small World).** Proposto por Malkov e Yashunin [2018], o HNSW constrói um grafo hierárquico de múltiplas camadas. Cada camada é um grafo navegável de ``mundo pequeno'' (small world), onde vértices são conectados tanto a vizinhos locais quanto a vértices distantes. A busca começa nas camadas superiores (mais esparsas) para rapidamente localizar a região relevante do espaço, e desce para camadas inferiores (mais densas) para refinar os resultados. HNSW oferece complexidade de busca $O(\log n)$ com alto recall, sendo a escolha predominante em sistemas de produção.

**IVF (Inverted File Index).** Particiona o espaço vetorial em $k$ clusters via k-means. Na busca, apenas os $n_{\text{probe}}$ clusters mais próximos da query são examinados, reduzindo drasticamente o número de comparações. IVF é particularmente eficaz quando combinado com quantização.

**PQ (Product Quantization).** Comprime vetores dividindo cada vetor em sub-vetores e quantizando cada sub-vetor independentemente com um codebook aprendido. PQ reduz o uso de memória por ordens de magnitude, permitindo manter índices de bilhões de vetores em memória. A desvantagem é uma perda de precisão proporcional ao grau de compressão.

Bibliotecas como FAISS [Johnson et al., 2019] implementam essas técnicas e suas combinações (IVF-PQ, HNSW+PQ), permitindo busca vetorial eficiente sobre dezenas de milhões de documentos com latência na ordem de milissegundos.

### 2.3.3 Sentence-BERT e Bi-Encoders

Conforme introduzido na Seção 2.1.2, o Sentence-BERT [Reimers e Gurevych, 2019] adaptou o BERT para produzir embeddings de sentença semanticamente significativos. A arquitetura de bi-encoder permite pré-computar embeddings de todos os documentos do corpus uma única vez. Na busca, apenas a query precisa ser codificada em tempo real, e a recuperação se reduz a uma busca ANN no espaço vetorial.

Essa propriedade é fundamental para sistemas de produção: o custo computacional da codificação (uma inferência BERT por query) é constante, independente do tamanho do corpus. O trade-off é que bi-encoders são menos expressivos que cross-encoders (que codificam query e documento conjuntamente), sacrificando alguma qualidade de matching pela viabilidade de busca em larga escala.

### 2.3.4 Forças e Fraquezas

**Forças:**
- **Generalização semântica:** Captura sinonímia, paráfrases e intenção implícita. ``Quero cancelar'' e ``vou desistir do serviço'' são reconhecidos como semanticamente próximos.
- **Robustez à formulação:** Diferentes formas de expressar a mesma necessidade produzem representações similares.
- **Capacidade de transferência:** Modelos pré-treinados em grandes corpora generalizam razoavelmente para domínios novos, mesmo sem fine-tuning.

**Fraquezas:**
- **Custo computacional:** A geração de embeddings requer inferência em modelos de centenas de milhões de parâmetros.
- **Interpretabilidade baixa:** A similaridade de cosseno entre vetores de 768 dimensões é opaca --- não é possível explicar *por que* dois textos são considerados similares.
- **Sensibilidade ao domínio:** Embeddings genéricos podem falhar em domínios especializados com vocabulário técnico não visto no pré-treinamento [Harris, 2025].
- **Overhead de indexação:** Índices ANN consomem memória significativa e requerem reconstrução quando novos documentos são adicionados (embora índices como HNSW suportem inserção incremental).

---

## 2.4 Busca Híbrida e Fusão de Scores

### 2.4.1 Motivação: Complementaridade

A análise das forças e fraquezas da busca lexical (Seção 2.2) e semântica (Seção 2.3) revela uma complementaridade fundamental: onde uma falha, a outra frequentemente acerta. O BM25 é preciso para termos exatos mas cego para paráfrases; embeddings capturam paráfrases mas podem falhar em termos específicos. Essa observação motiva a construção de sistemas de **busca híbrida** que combinam ambas as abordagens.

Rayo et al. [2025] demonstraram empiricamente essa complementaridade no domínio de textos regulatórios: a combinação de BM25 com BGE fine-tuned superou cada abordagem isolada, com o parâmetro de peso $\alpha = 0.65$ favorecendo levemente o componente semântico. A contribuição do componente lexical foi particularmente evidente em queries contendo termos técnicos e códigos de referência --- precisamente os casos onde a correspondência exata é essencial.

No domínio de conversas de atendimento, a complementaridade é ainda mais pronunciada. Conversas de call center frequentemente misturam terminologia técnica (nomes de planos, códigos de protocolo, termos de compliance) com linguagem coloquial, paráfrases e intenções implícitas. Nenhuma abordagem isolada cobre adequadamente esse espectro.

### 2.4.2 Combinação Linear Ponderada

A estratégia mais direta de fusão atribui pesos aos scores de cada sistema e os combina linearmente. Dado um score lexical $s_{\text{lex}}(d, q)$ e um score semântico $s_{\text{sem}}(d, q)$ para um documento $d$ e uma query $q$, o score híbrido é:

$$s_{\text{hybrid}}(d, q) = \alpha \cdot s_{\text{sem}}(d, q) + (1 - \alpha) \cdot s_{\text{lex}}(d, q)$$

onde $\alpha \in [0, 1]$ controla o peso relativo dos componentes. Com $\alpha = 0$, o sistema é puramente lexical; com $\alpha = 1$, puramente semântico.

Para que a combinação seja válida, os scores devem estar em escalas comparáveis. Como scores BM25 são não-limitados (podem variar de zero a centenas) enquanto similaridade de cosseno varia em $[-1, 1]$, uma etapa de normalização é necessária. Abordagens comuns incluem min-max normalization por query e normalização por z-score.

O parâmetro $\alpha$ pode ser fixo (determinado empiricamente em um conjunto de validação) ou adaptativo (ajustado dinamicamente com base em características da query, como o número de termos técnicos versus termos genéricos).

### 2.4.3 Reciprocal Rank Fusion (RRF)

Uma alternativa que evita o problema de normalização de scores é a **Reciprocal Rank Fusion (RRF)** [Cormack et al., 2009]. Em vez de combinar scores, o RRF combina rankings. Para cada documento $d$ ranqueado na posição $r_i(d)$ pelo sistema $i$, o score RRF é:

$$\text{RRF}(d) = \sum_{i=1}^{k} \frac{1}{c + r_i(d)}$$

onde $c$ é uma constante (tipicamente $c = 60$) que controla a importância relativa de posições altas versus baixas no ranking. Documentos não retornados por um sistema recebem posição infinita (contribuição zero).

O RRF tem a vantagem de ser invariante à escala --- não importa se um sistema retorna scores de 0 a 1 e outro de 0 a 1000, pois opera sobre posições. Além disso, é surpreendentemente eficaz na prática, frequentemente competitivo com fusões baseadas em scores que requerem normalização cuidadosa.

### 2.4.4 Cross-Encoder Reranking

As abordagens de fusão operam sobre resultados de bi-encoders, que codificam queries e documentos independentemente. Para refinar os resultados do topo, pode-se aplicar um **cross-encoder** como estágio de reranking.

Um cross-encoder recebe a concatenação da query e do documento como entrada única e produz um score de relevância:

$$s_{\text{cross}}(q, d) = \text{CrossEncoder}(\text{[CLS]} \; q \; \text{[SEP]} \; d \; \text{[SEP]})$$

Como query e documento são processados conjuntamente, o cross-encoder pode capturar interações finas entre termos de ambos --- algo impossível para bi-encoders. O custo é que a inferência deve ser realizada para cada par (query, documento), tornando o cross-encoder proibitivo como mecanismo de busca primário, mas viável como estágio de refinamento sobre um shortlist de dezenas a centenas de candidatos.

A arquitetura de reranking em dois estágios --- retrieval rápido seguido de reranking preciso --- é um padrão consolidado na indústria, utilizado em sistemas como Bing, Google e ElasticSearch.

### 2.4.5 Aplicação no TalkEx

No TalkEx, o retrieval híbrido segue o pipeline:

1. BM25 top-K sobre o índice lexical (com normalização accent-aware para PT-BR);
2. ANN top-K sobre o índice vetorial;
3. União dos conjuntos de resultados;
4. Fusão de scores (combinação linear ponderada ou RRF);
5. Reranking opcional com cross-encoder sobre o shortlist;
6. Aplicação de filtros de negócio (canal, fila, produto, data).

A decisão entre combinação linear e RRF, assim como o valor de $\alpha$ e o tamanho do top-K, são tratados como hiperparâmetros do pipeline, calibrados empiricamente para o domínio conversacional.

---

## 2.5 Classificação de Texto

### 2.5.1 Classificação Supervisionada

A classificação de texto é a tarefa de atribuir uma ou mais categorias predefinidas a um documento, com base em um modelo treinado a partir de exemplos rotulados. Diferentemente da busca (que recupera documentos relevantes para uma query), a classificação **decide** --- atribui um label com um grau de confiança.

Formalmente, dado um espaço de features $\mathcal{X}$ e um conjunto de classes $\mathcal{Y} = \{y_1, y_2, \ldots, y_c\}$, um classificador é uma função $h: \mathcal{X} \rightarrow \mathcal{Y}$ (ou $h: \mathcal{X} \rightarrow [0,1]^c$ no caso probabilístico) aprendida a partir de um conjunto de treinamento $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$.

### 2.5.2 Modelos de Classificação

**Regressão Logística.** O modelo mais simples e robusto para classificação de texto. Aprende um hiperplano de separação no espaço de features:

$$P(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

onde $\sigma$ é a função sigmoide, $\mathbf{w}$ é o vetor de pesos e $b$ é o bias. Para multiclasse, a generalização softmax é utilizada. A regressão logística é o baseline natural para classificação com embeddings: é rápida, interpretável (pesos indicam a importância de cada feature) e surpreendentemente competitiva com modelos mais complexos em muitas tarefas [AnthusAI, 2024].

**Gradient Boosting (XGBoost, LightGBM).** Ensembles de árvores de decisão treinadas sequencialmente, onde cada árvore corrige os erros da anterior. O diferencial do gradient boosting para classificação de texto é sua capacidade de operar nativamente sobre **features heterogêneas**: embeddings densos, scores BM25, features categóricas (canal, fila), features numéricas (duração, número de turnos) e flags binários (presença de entidades, ativação de regras). Essa versatilidade o torna particularmente adequado para pipelines que combinam sinais de naturezas distintas.

**MLP (Multi-Layer Perceptron).** Redes neurais feedforward com uma ou mais camadas ocultas. Dada uma entrada $\mathbf{x}$, a saída de uma MLP de duas camadas é:

$$\hat{y} = \sigma_2(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x} + b_1) + b_2)$$

MLPs são mais expressivas que regressão logística, capazes de capturar interações não-lineares entre features. São particularmente adequadas quando as features são predominantemente densas (embeddings), onde árvores de decisão perdem parte de sua vantagem.

### 2.5.3 Multi-Class vs Multi-Label

Duas configurações de classificação são relevantes:

**Multi-class:** Cada documento recebe exatamente um label. É o caso quando as classes são mutuamente exclusivas --- por exemplo, o motivo de contato principal de uma conversa.

**Multi-label:** Cada documento pode receber zero ou mais labels. É o caso quando as categorias não são mutuamente exclusivas --- por exemplo, uma conversa pode envolver simultaneamente ``reclamação'', ``solicitação de cancelamento'' e ``insatisfação com atendimento''.

Estratégias para multi-label incluem:
- **Binary relevance:** Treinar um classificador binário independente para cada classe.
- **Classifier chains:** Treinar classificadores sequenciais onde cada um recebe como feature adicional as predições dos anteriores.
- **Threshold tuning:** Usar um classificador multiclasse com saída probabilística e definir thresholds por classe para converter probabilidades em labels binários.

### 2.5.4 Features Heterogêneas

Um princípio central da arquitetura proposta nesta dissertação é que classificação robusta em domínios conversacionais requer **features heterogêneas** que combinam sinais de naturezas distintas:

- **Features de embedding:** Vetores densos representando turno, janela de contexto e conversa completa.
- **Features lexicais:** Scores BM25 contra protótipos de cada classe, presença de termos-chave.
- **Features estruturais:** Metadados da conversa (canal, fila, duração, número de turnos, falante).
- **Features contextuais:** Padrões de sequência (intent repetido, mudança de tópico, escalação).
- **Features de regras:** Flags de ativação de regras do motor determinístico.

Essa combinação supera abordagens que dependem exclusivamente de embeddings, pois captura sinais que nenhuma representação vetorial individual consegue codificar completamente.

### 2.5.5 O Princípio ``Embeddings Representam, Classificadores Decidem''

Um erro recorrente em sistemas de NLP é tratar a similaridade de cosseno entre embeddings como classificação. Observar que o embedding de ``quero cancelar meu plano'' tem alta similaridade com o protótipo de ``cancelamento'' não é classificação --- é uma observação sobre representação. A decisão de classificar requer:

- Um modelo treinado sobre dados rotulados;
- Um threshold calibrado;
- Evidência rastreável (score, confiança, versão do modelo, features utilizadas).

Harris [2025] e AnthusAI [2024] demonstraram que embeddings combinados com classificadores supervisionados (mesmo simples como regressão logística) superam consistentemente abordagens baseadas exclusivamente em similaridade (como kNN). O embedding é uma feature --- possivelmente a mais poderosa --- mas é o classificador que decide.

Esse princípio é um axioma de design do TalkEx: embeddings são gerados no estágio de representação e consumidos como features no estágio de classificação, nunca tratados como decisões de classificação em si.

### 2.5.6 Métricas de Avaliação

A avaliação de classificadores de texto requer métricas que capturem diferentes aspectos de desempenho:

**Precision** mede a proporção de previsões positivas que são corretas:

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** mede a proporção de positivos reais que foram corretamente identificados:

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score** é a média harmônica de precision e recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Para multiclasse, o F1 pode ser calculado como:
- **Macro-F1:** Média aritmética do F1 por classe, tratando todas as classes igualmente. Favorece desempenho uniforme, sendo sensível a classes raras.
- **Micro-F1:** Calcula precision e recall globais (somando TP, FP, FN de todas as classes) e depois computa o F1. Favorece classes frequentes.

**AUC (Area Under the ROC Curve)** mede a capacidade do classificador de separar classes positivas e negativas em diferentes thresholds. Um AUC de 1.0 indica separação perfeita; 0.5 indica desempenho aleatório.

**Calibration Error** mede o quanto as probabilidades preditas correspondem às frequências reais. Um classificador que prediz 80% de confiança deveria acertar em aproximadamente 80% dos casos. Calibração é especialmente importante quando as probabilidades são usadas para decisões downstream (como thresholds de regras).

---

## 2.6 Análise de Conversas

### 2.6.1 Estrutura Conversacional

Conversas são fundamentalmente diferentes de documentos textuais convencionais. Enquanto um documento é tipicamente escrito por um único autor com estrutura deliberada, uma conversa é co-construída por dois ou mais participantes em tempo real, com estrutura emergente e frequentemente imprevista.

Uma conversa de atendimento é composta por:

- **Turnos:** A unidade atômica da conversa. Cada turno é uma emissão contígua de um falante. Turnos variam drasticamente em comprimento --- desde confirmações monossilábicas (``sim'', ``ok'') até explicações detalhadas de problemas complexos.
- **Falantes (speakers):** Em conversas de atendimento, tipicamente dois: cliente e agente. A distinção de papel é semanticamente significativa: a mesma frase dita pelo cliente (``vou cancelar'') e pelo agente (``posso cancelar para você'') carrega intenções distintas.
- **Transições:** A sequência de turnos e a alternância entre falantes criam padrões de interação (pergunta-resposta, objeção-contraproposta, escalação) que carregam informação sobre o fluxo da conversa.
- **Metadados:** Canal (voz, chat, email), fila de atendimento, produto, duração, horário, status do cliente no CRM.

### 2.6.2 Context Windows: Janelas Deslizantes sobre Turnos

Um conceito central para a análise de conversas é a **janela de contexto** (context window): um grupo contíguo de $n$ turnos adjacentes que forma a unidade de análise contextual.

A motivação para janelas de contexto é que a intenção de um turno frequentemente só pode ser compreendida em contexto. O turno isolado ``sim'' é completamente ambíguo; ``sim'' após ``você gostaria de cancelar?'' revela uma intenção de cancelamento. Similarmente, a expressão de insatisfação de um cliente pode só se tornar evidente quando observamos uma sequência de turnos onde o agente oferece soluções e o cliente as rejeita repetidamente.

Os parâmetros de uma janela de contexto incluem:

- **Tamanho (window size):** Número de turnos na janela. Janelas menores (3--5 turnos) capturam interações locais imediatas; janelas maiores (7--10 turnos) capturam padrões de negociação e escalação mais longos.
- **Stride:** Deslocamento entre janelas consecutivas. Um stride de 1 produz janelas maximamente sobrepostas (cada turno aparece em múltiplas janelas); um stride igual ao tamanho produz janelas disjuntas.
- **Alinhamento por falante (speaker alignment):** Opção de ancorar janelas em turnos de um falante específico (por exemplo, janelas centradas em turnos do cliente para detectar intenção do cliente).
- **Peso por recência (recency weighting):** Turnos mais recentes na janela podem receber peso maior, refletindo a intuição de que o último turno frequentemente resume ou redireciona a interação.

### 2.6.3 Representação Multi-Nível

Uma contribuição central desta dissertação é a proposta de representações em múltiplos níveis de granularidade, cada um capturando aspectos distintos da conversa:

**Nível de turno.** O embedding de cada turno individual captura a intenção local e o conteúdo imediato. É a representação mais granular e permite busca e classificação ao nível de enunciados individuais.

**Nível de janela de contexto.** O embedding da janela de contexto captura dependências multi-turn --- desambiguação, mudanças de intenção, padrões de objeção e contraproposta. É a representação primária para classificação contextual, onde a intenção não pode ser inferida de um turno isolado.

**Nível de conversa.** O embedding da conversa completa captura o objetivo dominante e o desfecho geral. É a representação adequada para classificação de alto nível (motivo de contato principal, resultado da interação).

**Nível de papel (role-aware).** Embeddings separados para turnos do cliente e turnos do agente. A separação por papel permite:
- Detectar a intenção real do cliente sem contaminação pelo script do agente;
- Avaliar a conformidade do agente com procedimentos operacionais;
- Analisar alinhamento ou desalinhamento entre as agendas de cliente e agente.

Essa estratégia multi-nível permite que diferentes tarefas consumam a representação mais adequada: busca granular opera sobre turnos, classificação de intenção opera sobre janelas, e classificação de motivo de contato opera sobre conversas.

### 2.6.4 Desafios Específicos

**Ruído de ASR (Automatic Speech Recognition).** Em conversas telefônicas, o texto disponível é frequentemente uma transcrição automática de áudio. Transcrições de ASR contêm erros de reconhecimento, especialmente em nomes próprios, números, termos técnicos e sotaques regionais. Esses erros propagam-se para todas as etapas downstream: embeddings de sentenças com erros de transcrição podem gerar representações distorcidas, e regras lexicais que buscam termos exatos podem falhar.

**Coloquialismo e gírias.** O português brasileiro falado em contextos de atendimento telefônico difere significativamente do português escrito padrão. Expressões como ``to querendo'', ``me ajuda ai'', ``que nem'' e abreviações como ``vc'', ``pq'', ``tb'' são frequentes e podem não estar adequadamente representadas nos dados de pré-treinamento de modelos de linguagem.

**Alternância de tópicos.** Conversas de atendimento frequentemente cobrem múltiplos tópicos em uma única interação: o cliente pode iniciar perguntando sobre uma fatura, migrar para uma reclamação sobre qualidade do serviço e terminar solicitando cancelamento. Essa dinâmica multi-tópico desafia tanto a classificação (qual é o ``motivo de contato principal''?) quanto o retrieval (a conversa é relevante para múltiplas queries).

**Estrutura implícita.** Ao contrário de documentos com seções e parágrafos explícitos, a estrutura de uma conversa é implícita e emerge do fluxo de turnos. Identificar onde termina a fase de identificação e começa a resolução, ou onde o cliente muda de um tópico para outro, requer análise contextual não-trivial.

---

## 2.7 Motores de Regras e DSLs

### 2.7.1 Sistemas Baseados em Regras: Histórico e Papel

Sistemas baseados em regras são a forma mais antiga de sistemas de decisão automatizada em inteligência artificial, remontando aos sistemas especialistas (expert systems) das décadas de 1970 e 1980 [Buchanan e Shortliffe, 1984]. Nesses sistemas, o conhecimento de domínio é codificado como regras da forma ``SE condição ENTÃO ação'', e um motor de inferência aplica as regras sobre fatos conhecidos para derivar conclusões.

Embora a era do aprendizado de máquina e dos modelos neurais tenha deslocado o foco da comunidade acadêmica para abordagens baseadas em dados, sistemas baseados em regras continuam indispensáveis em contextos onde:

- **Auditabilidade é obrigatória:** Regulações de compliance, fraude e privacidade frequentemente exigem que decisões automatizadas sejam explicáveis e rastreáveis até a regra que as originou.
- **Determinismo é necessário:** Em cenários críticos (bloqueio de fraude, detecção de violação de compliance), a organização precisa de garantias de que determinadas condições sempre produzem determinados resultados, sem a variabilidade inerente a modelos probabilísticos.
- **Velocidade de adaptação:** Regras podem ser atualizadas em minutos por analistas de negócio, enquanto retreinar modelos de machine learning requer dados, validação e deployment.
- **Complementaridade com modelos:** Regras podem atuar como pós-processamento sobre saídas de modelos --- reduzindo falsos positivos, aplicando overrides para casos críticos, consolidando labels conflitantes.

### 2.7.2 DSL: Definição e Design

Uma **Domain-Specific Language (DSL)** é uma linguagem de programação de escopo restrito, projetada para expressar soluções em um domínio específico com máxima clareza e mínima cerimônia [Fowler, 2010]. Diferentemente de linguagens de propósito geral (Python, Java), uma DSL sacrifica generalidade por expressividade no domínio-alvo.

No contexto de motores de regras, a DSL define como regras são escritas. Uma DSL bem projetada deve:

- **Ser legível por não-programadores:** Analistas de negócio, auditores e gestores devem ser capazes de ler e entender uma regra sem conhecimento de programação.
- **Ser precisa o suficiente para compilação:** Apesar de legível, a DSL deve ter sintaxe e semântica formais suficientes para ser compilada em uma representação executável.
- **Restringir o espaço de expressão:** Limitar deliberadamente o que pode ser expresso, evitando que usuários criem regras excessivamente complexas ou perigosas.
- **Suportar composição:** Predicados simples devem ser combináveis com operadores lógicos (AND, OR, NOT) para formar condições complexas.

Um exemplo concreto no domínio de análise de conversas:

```
RULE risco_cancelamento_alto
WHEN
    speaker == "customer"
    AND intent_score("cancelamento") > 0.82
    AND (
        contains_any(["cancelar", "encerrar", "desistir"])
        OR similarity("quero cancelar meu servico") > 0.86
    )
    AND count_in_window("insatisfacao", 5) >= 2
THEN
    tag("cancelamento_risco_alto")
    score(0.95)
    priority("high")
```

Essa regra combina quatro famílias de sinais --- estrutural (speaker), semântico (intent_score), lexical (contains_any), e contextual (count_in_window) --- em uma expressão legível que pode ser auditada, versionada e explicada.

### 2.7.3 AST: Representação, Traversal e Avaliação

A DSL textual não é executada diretamente. Ela é compilada em uma **Abstract Syntax Tree (AST)** --- uma representação em árvore que captura a estrutura lógica da regra sem as particularidades sintáticas da DSL.

A compilação segue o processo padrão de linguagens de programação:

1. **Análise léxica (tokenização):** O texto da regra é dividido em tokens (palavras-chave, operadores, literais, identificadores).
2. **Análise sintática (parsing):** Os tokens são organizados em uma árvore conforme a gramática da DSL.
3. **Validação semântica:** Verifica-se que os predicados referenciam campos existentes, que os tipos são compatíveis e que os thresholds são válidos.
4. **Otimização (opcional):** A árvore pode ser reordenada para eficiência de execução.

A AST resultante é uma estrutura de dados composta por nós tipados:

```
RuleNode("risco_cancelamento_alto")
 |-- ConditionNode(AND)
 |    |-- ComparisonNode(speaker == "customer")
 |    |-- ComparisonNode(intent_score("cancelamento") > 0.82)
 |    |-- ConditionNode(OR)
 |    |    |-- PredicateNode(contains_any(["cancelar", "encerrar", "desistir"]))
 |    |    |-- ComparisonNode(similarity("quero cancelar...") > 0.86)
 |    |-- ComparisonNode(count_in_window("insatisfacao", 5) >= 2)
 |-- ActionNode(tag, score, priority)
```

A **avaliação** da AST consiste em percorrer a árvore (traversal), avaliando cada nó de baixo para cima (bottom-up) ou de cima para baixo (top-down) e acumulando resultados. Cada nó folha é avaliado contra os dados da conversa/turno/janela, e os nós internos combinam os resultados dos filhos segundo os operadores lógicos.

### 2.7.4 Short-Circuit Evaluation e Otimização por Custo

Em expressões lógicas com operador AND, se o primeiro predicado avalia como falso, o resultado da conjunção inteira é falso --- independentemente dos demais predicados. Essa propriedade, chamada **short-circuit evaluation**, permite economizar computação significativa ao ordenar os predicados por custo crescente.

A intuição é que predicados de custo diferente devem ser avaliados em sequência:

1. **Predicados estruturais** (custo negligível): igualdade de campos de metadados (speaker, canal, fila). Se a regra especifica ``speaker == customer'' e o turno é do agente, não há necessidade de computar predicados semânticos caros.
2. **Predicados lexicais** (custo baixo): contains, regex, BM25 contra protótipos. Operações sobre strings são ordens de magnitude mais rápidas que inferência neural.
3. **Predicados semânticos pré-computados** (custo médio): scores de intent e similaridade já calculados no estágio de embedding do pipeline.
4. **Predicados semânticos on-demand** (custo alto): similaridade de embedding calculada em tempo real, cross-encoder, inferência adicional.

No TalkEx, o executor da AST reordena os filhos de nós AND por custo estimado de cada predicado, garantindo que predicados baratos filtrem antes que predicados caros sejam avaliados. Essa otimização é especialmente importante quando milhares de regras são avaliadas sobre milhões de conversas.

### 2.7.5 Rastreabilidade de Evidência

Uma característica que distingue um motor de regras de produção de uma simples coleção de `if-else` é a capacidade de produzir **evidência rastreável** para cada decisão. Cada nó da AST, ao ser avaliado, registra:

- O valor observado (por exemplo, o score de intent calculado);
- O threshold esperado (por exemplo, > 0.82);
- O resultado (verdadeiro/falso);
- O trecho de texto que acionou o predicado (para predicados lexicais);
- A versão do modelo utilizado (para predicados semânticos).

Essa evidência permite:
- **Auditoria regulatória:** Explicar por que uma conversa foi classificada como ``risco de fraude'', com referência ao trecho de texto, ao score do modelo e à versão da regra vigente.
- **Depuração:** Identificar por que uma regra não acionou quando deveria (qual predicado falhou? com qual valor?).
- **Evolução de regras:** Comparar o comportamento de versões diferentes da mesma regra sobre o mesmo corpus.
- **Feedback para modelos:** Se uma regra determinística consistentemente corrige erros de um modelo, isso sinaliza uma deficiência no modelo que pode ser abordada com retreino.

---

## 2.8 Inferência em Cascata

### 2.8.1 Princípio: Processamento Progressivamente Mais Caro

Em sistemas de NLP que operam sobre grandes volumes de dados, o custo computacional por unidade de processamento varia dramaticamente entre técnicas. Verificar se uma conversa contém o termo ``cancelar'' custa microsegundos; computar o embedding de uma janela de contexto com um modelo Transformer custa milissegundos; avaliar um par query-documento com um cross-encoder custa dezenas de milissegundos; consultar um LLM via API custa centenas de milissegundos e centavos por chamada.

A **inferência em cascata** (cascaded inference) é a estratégia de organizar técnicas de processamento em estágios de custo crescente, onde cada estágio filtra o volume de dados que prossegue para o estágio seguinte. A premissa é que a maioria dos casos pode ser resolvida por técnicas baratas, e apenas uma fração requer o investimento computacional de técnicas caras.

Formalmente, dado um conjunto de conversas $C = \{c_1, c_2, \ldots, c_N\}$ e uma sequência de estágios $S_1, S_2, \ldots, S_k$ com custos crescentes $\text{cost}(S_1) < \text{cost}(S_2) < \ldots < \text{cost}(S_k)$, a cascata funciona como:

$$C_0 = C, \quad C_i = S_i(C_{i-1}) \subseteq C_{i-1}, \quad |C_i| \ll |C_{i-1}|$$

O custo total é:

$$\text{Custo}_{\text{cascata}} = \sum_{i=1}^{k} \text{cost}(S_i) \cdot |C_{i-1}|$$

que é significativamente menor que o custo de aplicar o estágio mais caro a todas as conversas:

$$\text{Custo}_{\text{uniforme}} = \text{cost}(S_k) \cdot |C|$$

### 2.8.2 Cascaded Inference em Sistemas de NLP

A aplicação de cascata em sistemas de NLP segue uma organização natural dos tipos de processamento:

**Estágio 1 --- Filtros baratos.** Regras baseadas em metadados (idioma, canal, fila, produto) e expressões lexicais simples (presença de termos críticos). Custo: microsegundos por conversa. Filtra conversas que claramente não são relevantes para o processamento downstream. Uma conversa em inglês pode ser imediatamente excluída de um pipeline de classificação para português; uma conversa da fila de vendas pode ser excluída de regras de compliance.

**Estágio 2 --- Retrieval híbrido.** BM25 + ANN sobre os índices lexical e vetorial. Custo: milissegundos por query. Recupera as conversas potencialmente relevantes de um corpus de milhões para um shortlist de centenas ou milhares. O retrieval híbrido funciona como um funil que reduz o volume em duas ou três ordens de magnitude.

**Estágio 3 --- Classificação e regras.** Classificadores supervisionados com features heterogêneas e regras semânticas baseadas em AST. Custo: dezenas de milissegundos por conversa. Opera sobre o shortlist do Estágio 2, produzindo labels, scores e evidências. Regras com predicados semânticos (intent_score, embedding_similarity) são avaliadas aqui, beneficiando-se do short-circuit para evitar computações desnecessárias.

**Estágio 4 --- Revisão excepcional.** LLMs ou modelos de alta capacidade, reservados para casos ambíguos, novos ou críticos. Custo: centenas de milissegundos a segundos, com custo monetário por chamada de API. Apenas uma fração percentual das conversas chega a este estágio. No TalkEx, este estágio é exclusivamente offline --- conversas são enfileiradas para revisão assíncrona, sem impacto na latência do pipeline online.

### 2.8.3 Trade-off Custo-Qualidade

A cascata introduz um trade-off fundamental: ao filtrar conversas em estágios baratos, há o risco de descartar prematuramente conversas que seriam corretamente classificadas por estágios mais caros. Esse risco manifesta-se como perda de recall nos estágios iniciais.

A chave para uma cascata eficaz é garantir **alto recall nos estágios baratos**, mesmo que a precision seja moderada. O Estágio 1 deve ser conservador --- é aceitável que ele deixe passar falsos positivos (conversas irrelevantes que prosseguem desnecessariamente), mas não que descarte verdadeiros positivos (conversas relevantes que não serão processadas). Cada estágio subsequente refina a precision.

Essa configuração produz uma **curva de Pareto** entre custo e qualidade: para diferentes pontos de corte em cada estágio, obtém-se diferentes combinações de custo total e F1 final. A calibração dos thresholds de cada estágio é um problema de otimização que deve ser resolvido empiricamente para cada domínio.

### 2.8.4 Exemplos na Indústria

O princípio de cascata é ubíquo em sistemas de informação em larga escala:

- **Motores de busca (Google, Bing):** Utilizam índices invertidos para retrieval rápido de milhares de candidatos, seguido por modelos neurais de reranking cada vez mais caros sobre shortlists progressivamente menores.
- **ElasticSearch:** O pipeline de query pode combinar filtros booleanos baratos, BM25, rescoring com scripts e, mais recentemente, reranking semântico.
- **Sistemas de detecção de fraude:** Regras simples baseadas em thresholds filtram a grande maioria das transações legítimas; modelos de machine learning analisam transações suspeitas; analistas humanos revisam apenas os casos de maior risco.

No contexto do TalkEx, a cascata é particularmente relevante porque conversas de call center variam enormemente em complexidade. Uma conversa trivial de atualização cadastral pode ser resolvida no Estágio 1; uma conversa ambígua envolvendo ameaça velada de cancelamento com linguagem indireta pode exigir o Estágio 3 ou 4. A cascata garante que o investimento computacional seja proporcional à complexidade do caso.

---

## 2.9 Síntese e Conexão com a Arquitetura

Os conceitos apresentados neste capítulo formam os pilares sobre os quais a arquitetura do TalkEx é construída. A Tabela 2.1 sintetiza como cada conceito fundamenta um componente específico do sistema.

| Conceito | Seção | Componente no TalkEx |
|---|---|---|
| Embeddings e Sentence Transformers | 2.1 | Geração de representações multi-nível (turno, janela, conversa) |
| BM25 e busca lexical | 2.2 | Componente lexical do retrieval híbrido |
| Busca semântica e ANN | 2.3 | Componente semântico do retrieval híbrido |
| Fusão de scores e reranking | 2.4 | Pipeline de retrieval híbrido |
| Classificação supervisionada | 2.5 | Classificadores com features heterogêneas |
| Análise de conversas | 2.6 | Modelo de dados conversacional e context windows |
| Motores de regras e DSL/AST | 2.7 | Motor de regras semânticas com evidência rastreável |
| Inferência em cascata | 2.8 | Organização do pipeline em estágios de custo crescente |

A contribuição central desta dissertação reside não na invenção de técnicas novas, mas na **composição arquitetural** que integra esses conceitos em um pipeline coeso, onde cada componente complementa os demais: o retrieval híbrido combina as forças da busca lexical e semântica; a classificação consome features de ambos os paradigmas; as regras proveem auditabilidade e determinismo onde modelos probabilísticos são insuficientes; e a cascata garante que o custo computacional seja proporcional à complexidade do caso.

No próximo capítulo, posicionamos este trabalho em relação à literatura existente, identificando a lacuna específica que a integração desses paradigmas preenche.

---

## Referências do Capítulo 2

- [AnthusAI, 2024] AnthusAI. Semantic Text Classification. GitHub Repository, 2024.
- [Bojanowski et al., 2017] Bojanowski, P., Grave, E., Joulin, A., Mikolov, T. Enriching Word Vectors with Subword Information. *Transactions of the ACL*, 5:135--146, 2017.
- [Buchanan e Shortliffe, 1984] Buchanan, B. G., Shortliffe, E. H. *Rule-Based Expert Systems*. Addison-Wesley, 1984.
- [Cormack et al., 2009] Cormack, G. V., Clarke, C. L. A., Buettcher, S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR*, 2009.
- [Devlin et al., 2018] Devlin, J., Chang, M., Lee, K., Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805, 2018.
- [Firth, 1957] Firth, J. R. A Synopsis of Linguistic Theory. In: *Studies in Linguistic Analysis*, pp. 1--32. Blackwell, 1957.
- [Fowler, 2010] Fowler, M. *Domain-Specific Languages*. Addison-Wesley, 2010.
- [Harris, 2025] Harris, L. Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents. arXiv:2505.11582v2, 2025.
- [Johnson et al., 2019] Johnson, J., Douze, M., Jegou, H. Billion-Scale Similarity Search with GPUs. *IEEE Transactions on Big Data*, 2019.
- [Karpukhin et al., 2020] Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W. Dense Passage Retrieval for Open-Domain Question Answering. arXiv:2004.04906, 2020.
- [Lyu et al., 2025] Lyu, N., Wang, Y., Chen, F., Zhang, Q. Advancing Text Classification with Large Language Models and Neural Attention Mechanisms. arXiv:2512.09444v1, 2025.
- [Malkov e Yashunin, 2018] Malkov, Y. A., Yashunin, D. A. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. *IEEE Transactions on PAMI*, 42(4):824--836, 2018.
- [Mikolov et al., 2013] Mikolov, T., Chen, K., Corrado, G., Dean, J. Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781, 2013.
- [Pennington et al., 2014] Pennington, J., Socher, R., Manning, C. D. GloVe: Global Vectors for Word Representation. *EMNLP*, 2014.
- [Rayo et al., 2025] Rayo, J., de la Rosa, R., Garrido, M. A Hybrid Approach to Information Retrieval and Answer Generation for Regulatory Texts. *COLING 2025*. arXiv:2502.16767v1, 2025.
- [Reimers e Gurevych, 2019] Reimers, N., Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*, 2019.
- [Robertson et al., 1996] Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., Gatford, M. Okapi at TREC-3. In: *Overview of the Third Text REtrieval Conference*, 1996.
- [Robertson e Zaragoza, 2009] Robertson, S., Zaragoza, H. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4):333--389, 2009.
- [Salton e Buckley, 1988] Salton, G., Buckley, C. Term-Weighting Approaches in Automatic Text Retrieval. *Information Processing & Management*, 24(5):513--523, 1988.
- [Su et al., 2022] Su, H., Shi, W., Kasber, J., Wang, Y., Hu, Y., Ostendorf, M., Yih, W., Smith, N. A., Zettlemoyer, L., Yu, T. One Embedder, Any Task: Instruction-Finetuned Text Embeddings. arXiv:2212.09741, 2022.
- [Vaswani et al., 2017] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. Attention Is All You Need. *NeurIPS*, 2017.
- [Wang et al., 2022] Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., Wei, F. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533, 2022.
- [Xiao et al., 2023] Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. C-Pack: Packaged Resources to Advance General Chinese Embedding. arXiv:2309.07597, 2023.
