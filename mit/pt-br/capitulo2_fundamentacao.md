# Capitulo 2 — Fundamentacao Teorica

## Resumo do Capitulo

Este capitulo estabelece os fundamentos teoricos necessarios para compreender a arquitetura, os experimentos e os resultados apresentados nesta dissertacao. Cobrimos as areas tecnicas centrais que sustentam o TalkEx: recuperacao de informacao probabilistica e a funcao de ranqueamento BM25 (Secao 2.1), representacoes densas de texto desde word embeddings estaticos ate encoders em nivel de sentenca (Secao 2.2), arquiteturas de recuperacao hibrida que combinam sinais lexicais e semanticos (Secao 2.3), classificacao supervisionada de textos com familias heterogeneas de features (Secao 2.4), sistemas de regras deterministicos para NLP auditavel (Secao 2.5), NLP conversacional e modelagem de contexto multi-turno (Secao 2.6), e inferencia em cascata para predicao com consciencia de custo (Secao 2.7). Cada secao introduz as definicoes formais, os algoritmos-chave, as metricas de avaliacao e os problemas em aberto que motivam as decisoes de projeto especificas descritas no Capitulo 4 e o protocolo experimental descrito no Capitulo 5. A apresentacao assume familiaridade com algebra linear, teoria de probabilidade e aprendizado de maquina basico; leitores que buscam um tratamento mais introdutorio sao direcionados a Manning et al. (2008) para recuperacao de informacao e Jurafsky e Martin (2024) para processamento de linguagem natural.

---

## 2.1 Fundamentos de Recuperacao de Informacao

### 2.1.1 O Framework Probabilistico de Recuperacao

A Recuperacao de Informacao (RI) aborda o problema de identificar, em uma grande colecao de documentos $D = \{d_1, d_2, \ldots, d_N\}$, aqueles documentos relevantes para a necessidade de informacao de um usuario expressa como uma consulta $q$. A abordagem probabilistica da RI, originada com Robertson e Jones (1976) e formalizada no Principio de Ranqueamento por Probabilidade (Robertson, 1977), sustenta que a estrategia otima de recuperacao ranqueia documentos pela probabilidade estimada de relevancia dada a consulta: $P(\text{rel} \mid q, d)$. Sob a suposicao de que os julgamentos de relevancia sao binarios e independentes entre documentos, esse ranqueamento minimiza a perda esperada para uma classe ampla de funcoes de perda (Robertson, 1977).

O Modelo de Independencia Binaria (BIM), desenvolvido por Robertson e Sparck Jones (1976), operacionaliza este principio modelando documentos como vetores binarios de termos e estimando probabilidades de relevancia a partir da presenca ou ausencia de termos. O BIM assume que os termos ocorrem independentemente em documentos relevantes e nao relevantes — uma suposicao reconhecidamente falsa, mas empiricamente produtiva. O modelo deriva uma ponderacao por frequencia inversa de documento (IDF) que atribui scores mais altos a termos que discriminam entre documentos relevantes e nao relevantes:

$$\text{IDF}(t) = \log \frac{N - n_t + 0.5}{n_t + 0.5}$$

onde $N$ e o numero total de documentos e $n_t$ e o numero de documentos que contem o termo $t$. Esta formulacao atribui peso alto a termos que aparecem em poucos documentos (discriminativos) e peso baixo ou negativo a termos que aparecem na maioria dos documentos (nao discriminativos). A suavizacao $+0.5$ previne a divisao por zero e modera as estimativas para termos muito raros.

### 2.1.2 BM25: Derivacao e Parametros

O BM25 (Best Matching 25) estende o modelo de independencia binaria incorporando frequencia de termo intra-documento e normalizacao pelo comprimento do documento (Robertson et al., 1995; Robertson e Zaragoza, 2009). A funcao de score para uma consulta $q = \{t_1, t_2, \ldots, t_m\}$ contra um documento $d$ e:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

onde $f(t, d)$ e a frequencia do termo $t$ no documento $d$, $|d|$ e o comprimento do documento em tokens, $\text{avgdl}$ e o comprimento medio dos documentos na colecao, e $k_1$ e $b$ sao parametros livres.

O parametro $k_1$ controla a saturacao da frequencia de termo. Quando $k_1 = 0$, o BM25 reduz-se a um modelo binario onde apenas a presenca do termo importa, independentemente da frequencia. A medida que $k_1 \to \infty$, o modelo aproxima-se da ponderacao por frequencia bruta de termo sem saturacao. A configuracao padrao $k_1 \in [1.2, 2.0]$ fornece uma resposta sublinear a frequencia de termo: ocorrencias adicionais de um termo aumentam o score, mas com retornos decrescentes. Essa saturacao sublinear e critica para lidar com texto repetitivo — comum em conversas de atendimento ao cliente onde clientes podem repetir uma palavra-chave de reclamacao multiplas vezes — sem permitir que a frequencia sozinha domine a relevancia.

O parametro $b \in [0, 1]$ controla o grau de normalizacao pelo comprimento do documento. Quando $b = 0$, nenhuma normalizacao de comprimento e aplicada; documentos mais longos sao sistematicamente favorecidos porque contem mais ocorrencias de termos. Quando $b = 1$, a normalizacao completa penaliza documentos mais longos proporcionalmente ao seu excesso de comprimento sobre a media da colecao. A configuracao padrao $b = 0.75$ representa uma normalizacao moderada que reduz o vies de comprimento sem penalizar excessivamente documentos longos genuinamente ricos em informacao. No dominio conversacional, onde "documentos" sao janelas de contexto com quantidades variadas de turnos, a escolha de $b$ afeta diretamente se conversas mais longas recebem scores de recuperacao sistematicamente mais altos ou mais baixos — uma consideracao de projeto abordada no Capitulo 4.

O BM25 pode ser entendido como uma generalizacao do TF-IDF. Quando $k_1 \to \infty$ e $b = 0$, a ponderacao de termos do BM25 reduz-se a $\text{IDF}(t) \cdot f(t, d)$, que e precisamente a formulacao TF-IDF (Salton e Buckley, 1988). A introducao da saturacao ($k_1 < \infty$) e da normalizacao de comprimento ($b > 0$) torna o BM25 mais robusto em colecoes heterogeneas. A variante BM25+ (Lv e Zhai, 2011) introduz um limite inferior aditivo $\delta$ para prevenir a penalizacao de documentos longos que contem termos altamente relevantes:

$$\text{BM25+}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \left(\frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)} + \delta\right)$$

Esta variante e particularmente relevante quando os comprimentos dos documentos variam amplamente, como em corpora conversacionais onde as interacoes vao desde cumprimentos de dois turnos ate escalacoes de reclamacao de trinta turnos.

### 2.1.3 O Modelo de Espaco Vetorial e o Problema de Descasamento de Vocabulario

O modelo de espaco vetorial (VSM), introduzido por Salton et al. (1975), representa tanto consultas quanto documentos como vetores em um espaco $|V|$-dimensional, onde $|V|$ e o tamanho do vocabulario. Cada dimensao corresponde a um termo, e o valor ao longo dessa dimensao e tipicamente um peso TF-IDF. A relevancia e calculada como a similaridade cosseno entre o vetor da consulta e cada vetor de documento:

$$\cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \cdot |\vec{d}|}$$

O VSM fornece um framework geometrico fundamentado para recuperacao, mas compartilha com o BM25 uma limitacao fundamental: ambos dependem de sobreposicao lexical exata entre os termos da consulta e do documento. Isso cria o problema de descasamento de vocabulario (Furnas et al., 1987): quando usuarios e autores de documentos empregam palavras diferentes para expressar o mesmo conceito, os metodos de recuperacao lexical falham. Furnas et al. (1987) estimaram que a probabilidade de duas pessoas usarem o mesmo termo para o mesmo conceito e inferior a 20% — uma constatacao com consequencias diretas para o atendimento ao cliente, onde um cliente dizendo "eu nao quero mais isso" e uma categoria de intencao rotulada como "cancelamento" compartilham zero sobreposicao lexical.

Diversas abordagens foram propostas para mitigar o descasamento de vocabulario dentro do paradigma lexical. A expansao de consulta (Rocchio, 1971) aumenta a consulta com termos relacionados extraidos de feedback de pseudo-relevancia. A Indexacao Semantica Latente (Deerwester et al., 1990) projeta a matriz termo-documento em um espaco de menor dimensionalidade via decomposicao em valores singulares, capturando associacoes semanticas latentes. No entanto, esses metodos introduzem sobrecarga computacional adicional e suposicoes, e nenhum deles resolve completamente o problema de descasamento. O surgimento de representacoes vetoriais densas, discutido na Secao 2.2, fornece uma solucao mais fundamentada ao mapear texto para um espaco semantico continuo onde significados similares se agrupam independentemente da forma superficial.

### 2.1.4 Metricas de Avaliacao para Recuperacao de Informacao

A avaliacao de sistemas de recuperacao requer metricas que capturem tanto a relevancia quanto a qualidade do ranqueamento dos resultados retornados. Definimos as metricas utilizadas ao longo desta dissertacao.

**Precision at $K$ ($P@K$)** mede a fracao de documentos relevantes entre os $K$ primeiros resultados:

$$P@K = \frac{|\{\text{documentos relevantes nos top } K\}|}{K}$$

**Recall at $K$ ($R@K$)** mede a fracao de todos os documentos relevantes que aparecem nos $K$ primeiros resultados:

$$R@K = \frac{|\{\text{documentos relevantes nos top } K\}|}{|\{\text{todos os documentos relevantes}\}|}$$

**Mean Reciprocal Rank (MRR)** mede o inverso medio da posicao do primeiro resultado relevante ao longo de um conjunto de consultas $Q$:

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

onde $\text{rank}_i$ e a posicao no ranqueamento do primeiro documento relevante para a consulta $i$. O MRR e particularmente apropriado quando cada consulta possui uma unica resposta correta ou quando apenas o primeiro resultado relevante importa — uma condicao que frequentemente se verifica na classificacao de intencoes, onde o objetivo e identificar a unica intencao correta para uma conversa. O MRR e a metrica primaria de recuperacao utilizada nos experimentos H1 descritos no Capitulo 6.

**Normalized Discounted Cumulative Gain (nDCG)** estende a avaliacao para julgamentos de relevancia gradual. Para uma lista ranqueada de resultados, o ganho cumulativo descontado na posicao $K$ e:

$$\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

onde $\text{rel}_i$ e o grau de relevancia do documento na posicao $i$. A normalizacao divide pelo DCG ideal (IDCG), calculado sobre o ranqueamento otimo:

$$\text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

O nDCG varia de 0 a 1 e e a metrica padrao para benchmarks de recuperacao como BEIR (Thakur et al., 2021) e MS-MARCO (Nguyen et al., 2016). Enquanto o MRR e binario (relevante ou nao), o nDCG acomoda relevancia gradual — relevante para futuras extensoes do TalkEx a taxonomias de intencoes multi-rotulo ou hierarquicas.

---

## 2.2 Representacoes Densas e Sentence Embeddings

### 2.2.1 De Word Embeddings Estaticos a Representacoes Contextuais

A transicao de representacoes lexicais esparsas e de alta dimensionalidade para representacoes distribuidas densas e de baixa dimensionalidade reformulou fundamentalmente o NLP. A hipotese distribucional — de que palavras que ocorrem em contextos similares tem significados similares (Harris, 1954; Firth, 1957) — fornece a fundamentacao teorica. O Word2Vec (Mikolov et al., 2013) operacionalizou essa hipotese treinando redes neurais rasas para prever uma palavra a partir de seu contexto (CBOW) ou um contexto a partir de uma palavra (Skip-gram), produzindo vetores densos de 100--300 dimensoes que capturam regularidades semanticas e sintaticas. O GloVe (Pennington et al., 2014) alcancou representacoes similares por meio de fatoracao matricial de estatisticas de coocorrencia globais, fornecendo uma conexao entre metodos preditivos e baseados em contagem.

Word embeddings estaticos, no entanto, atribuem um unico vetor por tipo de palavra independentemente do contexto. A palavra "banco" recebe a mesma representacao seja denotando uma instituicao financeira ou a margem de um rio. Essa limitacao de polissemia foi abordada por modelos de embeddings contextuais que produzem representacoes em nivel de token condicionadas na sentenca circundante.

O ELMo (Peters et al., 2018) introduziu representacoes dependentes de contexto concatenando os estados ocultos de um modelo de linguagem LSTM bidirecional. Cada token recebe um vetor que reflete seu uso especifico em contexto, permitindo desambiguacao. No entanto, a natureza sequencial dos LSTMs limitou a escalabilidade para documentos longos e a paralelizacao em hardware moderno.

A arquitetura Transformer (Vaswani et al., 2017) resolveu a limitacao de escalabilidade por meio do mecanismo de self-attention, permitindo que cada token atenda diretamente a todos os outros tokens na sequencia. O BERT (Devlin et al., 2019) aplicou o encoder Transformer em um objetivo de pre-training com modelagem de linguagem mascarada, produzindo representacoes contextuais que alcancaram resultados estado-da-arte em uma ampla gama de tarefas de NLP. O RoBERTa (Liu et al., 2019) demonstrou que a otimizacao cuidadosa de hiperparametros de pre-training — treinamento mais longo, batches maiores, mascaramento dinamico — melhorou substancialmente os resultados do BERT sem mudancas arquiteturais. O XLM-RoBERTa (Conneau et al., 2020) estendeu essa abordagem para 100 idiomas usando dados do Common Crawl, estabelecendo o paradigma de pre-training multilingue que fundamenta o encoder utilizado no TalkEx.

### 2.2.2 Representacoes em Nivel de Sentenca: O Problema de Pooling

Encoders Transformer produzem representacoes em nivel de token: para uma sequencia de entrada de $n$ tokens, o encoder produz $n$ vetores de dimensao $d$. Muitas tarefas downstream, no entanto, requerem uma unica representacao de comprimento fixo por sentenca ou segmento de texto — um sentence embedding. Converter representacoes em nivel de token para representacoes em nivel de sentenca e o problema de pooling.

A abordagem mais simples, **mean pooling**, calcula a media dos vetores em nivel de token:

$$\vec{s} = \frac{1}{n} \sum_{i=1}^{n} \vec{h}_i$$

onde $\vec{h}_i$ e o estado oculto do $i$-esimo token na camada final do encoder. O mean pooling tem as virtudes da simplicidade e estabilidade, mas trata todos os tokens igualmente, independentemente de sua importancia semantica. Na sentenca "Eu preciso absolutamente cancelar minha assinatura imediatamente", os tokens "cancelar" e "assinatura" carregam muito mais informacao discriminativa de intencao do que "Eu" ou "minha", mas o mean pooling os pondera de forma identica.

**Pooling pelo token [CLS]** utiliza a representacao do token especial de classificacao como o sentence embedding. No pre-training do BERT, o token [CLS] e treinado para agregar informacao em nivel de sentenca para a predicao da proxima sentenca, mas Reimers e Gurevych (2019) demonstraram que representacoes [CLS] sem fine-tuning produzem sentence embeddings pobres — piores que medias de GloVe em tarefas de similaridade textual semantica (STS).

**Pooling ponderado por attention** aprende um conjunto de pesos de attention sobre as representacoes de tokens:

$$\alpha_i = \frac{\exp(\vec{w}^T \vec{h}_i)}{\sum_{j=1}^{n} \exp(\vec{w}^T \vec{h}_j)}, \quad \vec{s} = \sum_{i=1}^{n} \alpha_i \vec{h}_i$$

onde $\vec{w}$ e um vetor de parametros aprendidos. Essa abordagem permite que o modelo pondere tokens semanticamente importantes com maior peso. Lyu et al. (2025) demonstraram que pooling baseado em attention melhora o F1 de classificacao de 0.86 para 0.89 no dataset AG News comparado a representacoes BERT padrao, confirmando que a ponderacao aprendida de tokens fornece uma vantagem mensuravel para tarefas discriminativas.

### 2.2.3 Sentence-BERT e o Paradigma de Treinamento Siames

Reimers e Gurevych (2019) identificaram uma limitacao critica no uso direto do BERT para similaridade de sentencas: calcular a similaridade entre duas sentencas requer alimentar ambas no encoder simultaneamente (cross-encoding), produzindo complexidade $O(n^2)$ para comparacoes par-a-par sobre $n$ sentencas. Para uma tarefa de recuperacao sobre 10.000 candidatos, isso requer 10.000 passagens diretas por consulta — computacionalmente proibitivo.

O Sentence-BERT (SBERT) aborda isso realizando fine-tuning do BERT em uma arquitetura de rede siamesa ou triplet. Duas sentencas sao codificadas independentemente por encoders com pesos compartilhados, e os embeddings resultantes sao comparados usando similaridade cosseno. O modelo e treinado em dados de Inferencia de Linguagem Natural (NLI) — pares de sentencas rotulados como implicacao, contradicao ou neutro — usando um objetivo de classificacao softmax, ou em dados STS usando um objetivo de regressao com erro quadratico medio. A inovacao-chave e que a inferencia requer apenas uma unica passagem direta por sentenca, permitindo a pre-computacao de todos os embeddings candidatos e a recuperacao via busca de vizinhos mais proximos aproximada com custo $O(\log n)$.

A variante de treinamento triplet utiliza triplas ancora-positivo-negativo com o objetivo:

$$\mathcal{L} = \max(0, \| \vec{a} - \vec{p} \| - \| \vec{a} - \vec{n} \| + \epsilon)$$

onde $\vec{a}$, $\vec{p}$, $\vec{n}$ sao os embeddings dos exemplos ancora, positivo e negativo, e $\epsilon$ e um hiperparametro de margem. Essa formulacao otimiza diretamente a geometria do espaco de embeddings para recuperacao, aproximando pares similares e afastando pares dissimilares. Trabalhos subsequentes introduziram funcoes de perda contrastiva mais eficientes. A MultipleNegativesRankingLoss (Henderson et al., 2017) trata todos os outros exemplos no batch como negativos, aumentando dramaticamente o numero efetivo de amostras negativas sem computacao adicional.

### 2.2.4 Encoders Multilingues e paraphrase-multilingual-MiniLM-L12-v2

Estender sentence embeddings para cenarios multilingues requer modelos treinados em dados paralelos ou pseudo-paralelos entre multiplos idiomas. A destilacao de conhecimento de um professor monolingue de alta qualidade para um aluno multilingue provou ser eficaz: Reimers e Gurevych (2020) demonstraram que treinar um aluno multilingue para imitar os embeddings de um professor SBERT monolingue em ingles, usando pares de sentencas paralelas, produz embeddings multilingues que se aproximam da qualidade do professor em ingles enquanto estendem a cobertura para mais de 50 idiomas.

O modelo utilizado ao longo do TalkEx, **paraphrase-multilingual-MiniLM-L12-v2**, exemplifica essa abordagem. Trata-se de um encoder Transformer de 12 camadas destilado de um modelo de parafrase maior, produzindo sentence embeddings de 384 dimensoes. O modelo suporta mais de 50 idiomas, incluindo portugues, e foi treinado em dados de parafrase usando a MultipleNegativesRankingLoss. Sua saida de 384 dimensoes representa uma compressao das 768 dimensoes da arquitetura BERT-base, reduzindo os requisitos de armazenamento e a latencia de busca de vizinhos mais proximos por um fator de dois, mantendo a maioria da capacidade representacional.

A escolha deste modelo especifico para o TalkEx, descrita em detalhe no Capitulo 4, reflete tres restricoes: (1) suporte nativo ao portugues brasileiro sem fine-tuning especifico de dominio, (2) dimensionalidade de embeddings gerenciavel para indexacao em larga escala, e (3) disponibilidade como encoder congelado — isto e, um modelo utilizado sem atualizacoes de gradiente em producao, garantindo que o espaco de embeddings permaneca estavel entre versoes do corpus e que comparacoes longitudinais sejam validas.

### 2.2.5 Encoders Congelados versus Fine-Tuned

A decisao de congelar ou realizar fine-tuning de um encoder pre-treinado envolve um trade-off entre otimizacao especifica para a tarefa e estabilidade operacional. O fine-tuning adapta as representacoes do encoder ao dominio e tarefa alvo, potencialmente melhorando o desempenho significativamente. No entanto, o fine-tuning introduz diversas complicacoes. Primeiro, cada execucao de fine-tuning produz um espaco de embeddings diferente, invalidando vetores previamente calculados — uma preocupacao critica para sistemas que mantem indices vetoriais persistentes. Segundo, o fine-tuning requer dados rotulados de treinamento no dominio alvo, que podem ser escassos ou caros de obter. Terceiro, o fine-tuning arrisca o esquecimento catastrofico (McCloskey e Cohen, 1989; Kirkpatrick et al., 2017), onde a adaptacao ao dominio alvo degrada o desempenho nas capacidades gerais que o modelo adquiriu durante o pre-training.

A abordagem de encoder congelado trata o modelo pre-treinado como um extrator de features fixo. Todo o aprendizado downstream ocorre em classificadores leves (regressao logistica, gradient boosted trees, MLPs pequenos) que operam sobre os embeddings congelados. Essa abordagem sacrifica ganhos representacionais potenciais especificos da tarefa em troca de estabilidade dos embeddings, custo computacional reduzido (sem passagem reversa pelo encoder) e a capacidade de pre-computar e armazenar em cache todos os embeddings. Howard e Ruder (2018) e Peters et al. (2019) estudaram sistematicamente o trade-off congelado-versus-fine-tuned, constatando que o fine-tuning fornece ganhos substanciais para tarefas com mudanca significativa de dominio, mas retornos decrescentes quando os dados de pre-training sao suficientemente amplos.

Em cenarios de baixo recurso — definidos nao pelo tamanho absoluto do corpus, mas pela razao entre exemplos rotulados e complexidade de classes — a abordagem congelada pode superar o fine-tuning porque os dados de treinamento limitados sao insuficientes para atualizar confiavelmente milhoes de parametros do encoder sem overfitting. Essa observacao motiva a arquitetura frozen-first adotada no TalkEx e testada experimentalmente no Capitulo 6.

---

## 2.3 Recuperacao Hibrida

### 2.3.1 A Tese da Complementaridade

A motivacao teorica para a recuperacao hibrida repousa em uma tese de complementaridade: metodos de recuperacao lexical e semantica possuem modos de falha distintos e amplamente nao sobrepostos, e sua combinacao deveria, portanto, melhorar o recall alem do que qualquer um deles alcanca isoladamente. Metodos lexicais falham quando documentos relevantes utilizam vocabulario diferente da consulta (descasamento de vocabulario). Metodos semanticos falham quando a relevancia depende de correspondencia exata de termos — codigos de produtos, nomes, palavras-chave regulatorias, abreviacoes especificas de dominio — onde o espaco de embeddings pre-treinado nao distingue entre formas superficiais com diferentes significados operacionais.

Lin et al. (2021) forneceram uma analise sistematica dessa complementaridade na tarefa de recuperacao de passagens do MS-MARCO. Eles calcularam a sobreposicao entre os top-1000 resultados do BM25 e um recuperador denso (ANCE), constatando que apenas aproximadamente 60% das passagens relevantes apareciam em ambos os conjuntos de resultados. Os 40% restantes eram recuperados exclusivamente por um metodo ou pelo outro, confirmando que as duas abordagens acessam sinais de relevancia genuinamente diferentes. Essa constatacao foi replicada em multiplos benchmarks e dominios (Thakur et al., 2021; Formal et al., 2022).

No dominio conversacional, a complementaridade e amplificada pela natureza heterogenea do texto. Conversas de atendimento ao cliente contem tanto sinais estruturados (nomes de produtos, identificadores de planos, referencias a orgaos reguladores) que favorecem a correspondencia lexical quanto sinais nao estruturados (intencao implicita, reclamacoes parafraseadas, indicadores emocionais) que favorecem a recuperacao semantica. Um cliente dizendo "Procon" esta usando uma palavra-chave regulatoria que o BM25 recupera com alta precisao; um cliente dizendo "vou levar isso para o orgao de defesa do consumidor" expressa a mesma intencao regulatoria sem a palavra-chave, exigindo recuperacao semantica. Um sistema que combina ambas as modalidades e, portanto, mais adequado a heterogeneidade dos dados do que qualquer uma delas isoladamente.

### 2.3.2 Metodos de Fusao de Scores

Dados os conjuntos de candidatos da recuperacao lexical e semantica, a etapa de fusao os combina em uma unica lista ranqueada. Varios metodos de fusao foram propostos, cada um com diferentes suposicoes e propriedades.

**Interpolacao linear** calcula uma soma ponderada dos scores normalizados:

$$s_{\text{hybrid}}(q, d) = \alpha \cdot \hat{s}_{\text{sem}}(q, d) + (1 - \alpha) \cdot \hat{s}_{\text{lex}}(q, d)$$

onde $\hat{s}_{\text{sem}}$ e $\hat{s}_{\text{lex}}$ sao os scores semanticos e lexicais normalizados para $[0, 1]$ (tipicamente via normalizacao min-max dentro de cada conjunto de resultados), e $\alpha \in [0, 1]$ e um peso de fusao. A interpolacao linear e simples e interpretavel: $\alpha$ controla diretamente a importancia relativa dos sinais semanticos versus lexicais. No entanto, o $\alpha$ otimo e dependente do dominio e deve ser ajustado empiricamente. Nos experimentos do TalkEx (Capitulo 6), $\alpha$ e variado sistematicamente e o valor otimo de $\alpha = 0.30$ (favorecendo o componente lexical) e determinado no conjunto de validacao.

**Reciprocal Rank Fusion (RRF)** (Cormack et al., 2009) realiza a fusao baseada em posicoes de ranqueamento em vez de scores:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

onde $R$ e o conjunto de ranqueadores (lexical, semantico), $\text{rank}_r(d)$ e a posicao do documento $d$ na saida do ranqueador $r$, e $k$ e uma constante (tipicamente 60) que mitiga a influencia de outliers de alta posicao. O RRF tem a vantagem de ser agnostico a scores — nao requer normalizacao de scores, que e problematica quando scores lexicais e semanticos estao em escalas fundamentalmente diferentes e com distribuicoes distintas. Cormack et al. (2009) demonstraram que o RRF e competitivo com ou superior a metodos de fusao treinados em datasets do TREC, apesar de nao requerer dados de treinamento.

**Fusao aprendida** treina um modelo (regressao logistica, rede neural pequena) para prever relevancia a partir dos scores individuais de recuperacao e opcionalmente features adicionais (comprimento da consulta, caracteristicas do documento candidato). Essa abordagem pode capturar interacoes nao lineares entre as fontes de sinal, mas requer dados de treinamento rotulados por relevancia para o modelo de fusao e arrisca overfitting a distribuicao de treinamento. Para os propositos do TalkEx, a vantagem de interpretabilidade da interpolacao linear e a propriedade de agnosticismo a scores do RRF sao preferidos aos ganhos marginais da fusao aprendida — consistente com os principios de projeto que governam a arquitetura do sistema (Capitulo 4).

### 2.3.3 Reranqueamento com Cross-Encoder

Uma limitacao da recuperacao com bi-encoder — seja lexical, semantica ou hibrida — e que a consulta e o documento sao codificados independentemente. Isso impede que o modelo capture interacoes finas em nivel de token entre a consulta e o documento candidato. O reranqueamento com cross-encoder aborda essa limitacao codificando conjuntamente o par consulta-documento em uma unica passagem direta do Transformer:

$$s_{\text{cross}}(q, d) = \sigma(\text{MLP}(\text{BERT}([q; \text{SEP}; d])))$$

onde $[q; \text{SEP}; d]$ denota a concatenacao dos tokens da consulta e do documento com um separador, e o modelo produz um score escalar de relevancia. Nogueira e Cho (2019) demonstraram que o reranqueamento com cross-encoder sobre um conjunto inicial de candidatos do BM25 melhorou o MRR no MS-MARCO em 12 pontos absolutos sobre o BM25 sozinho, estabelecendo o paradigma retrieve-then-rerank como a arquitetura dominante na RI moderna.

O custo computacional do cross-encoding e $O(K)$ passagens diretas por consulta, onde $K$ e o tamanho do conjunto de candidatos. Esse custo e aceitavel como um refinamento de segundo estagio (reranqueando os top 100 candidatos), mas proibitivo como metodo de recuperacao de primeiro estagio sobre milhoes de documentos. Na arquitetura do TalkEx, o reranqueamento com cross-encoder e um componente opcional de segundo estagio, aplicavel quando a confianca da classificacao do primeiro estagio cai abaixo de um threshold calibrado — um design que integra o reranqueamento na estrategia de inferencia em cascata descrita na Secao 2.7.

### 2.3.4 Arquiteturas de Fusao Denso-Esparso

Alem do paradigma de recuperar-e-fundir, trabalhos recentes exploraram arquiteturas que integram sinais lexicais e semanticos no nivel da representacao, em vez do nivel do score.

**SPLADE** (Formal et al., 2021) aprende representacoes esparsas a partir de um modelo de linguagem mascarado, computando pesos de importancia de termos sobre todo o vocabulario. A representacao resultante e esparsa (a maioria dos pesos e zero), mas aprendida, permitindo que o modelo atribua pesos nao nulos a termos semanticamente relacionados que nao estao presentes na entrada — efetivamente aprendendo expansao de consulta dentro do processo de codificacao. O SPLADE alcanca desempenho competitivo ou superior aos recuperadores densos em benchmarks BEIR, mantendo a eficiencia da recuperacao por indice invertido.

**ColBERT** (Khattab e Zaharia, 2020) introduz a interacao tardia (late interaction): consultas e documentos sao codificados em sequencias de embeddings em nivel de token, e a relevancia e calculada como a soma das similaridades maximas entre cada token da consulta e todos os tokens do documento:

$$s_{\text{ColBERT}}(q, d) = \sum_{i=1}^{|q|} \max_{j \in \{1,\ldots,|d|\}} \vec{q}_i^T \vec{d}_j$$

Essa formulacao preserva a correspondencia fina em nivel de token enquanto ainda permite que os embeddings dos documentos sejam pre-computados. O ColBERTv2 (Santhanam et al., 2022) introduziu compressao residual para reduzir a sobrecarga de armazenamento das representacoes por token. Essas arquiteturas representam a fronteira da pesquisa em recuperacao hibrida, embora requeiram substancialmente mais armazenamento e recursos computacionais do que a abordagem bi-encoder utilizada no TalkEx.

A escolha de projeto no TalkEx — recuperacao com bi-encoder com fusao de scores pos-hoc — prioriza simplicidade, interpretabilidade e a capacidade de diagnosticar a contribuicao de cada fonte de sinal independentemente. Essa escolha e avaliada empiricamente no Capitulo 6, onde os metodos de fusao linear e RRF sao comparados diretamente.

---

## 2.4 Classificacao de Textos

### 2.4.1 Abordagens Tradicionais: De Naive Bayes a Support Vector Machines

A classificacao de textos — a atribuicao de um ou mais rotulos predefinidos a um documento de texto — esta entre os problemas mais antigos e mais extensivamente estudados em NLP. A abordagem generativa, exemplificada pelo Multinomial Naive Bayes (McCallum e Nigam, 1998), modela a probabilidade conjunta $P(c, d)$ da classe $c$ e do documento $d$ usando a suposicao de independencia condicional:

$$P(c \mid d) \propto P(c) \prod_{t \in d} P(t \mid c)$$

Apesar da forte suposicao de independencia, o Naive Bayes apresenta desempenho surpreendentemente bom em classificacao de textos — um fenomeno atribuido ao fato de que a classificacao requer apenas o ranqueamento correto das probabilidades a posteriori, nao estimativas precisas de probabilidade (Domingos e Pazzani, 1997). O modelo e rapido de treinar, escala linearmente com o tamanho do vocabulario e fornece uma interpretacao probabilistica natural, tornando-o uma baseline padrao.

As Support Vector Machines (SVMs) (Joachims, 1998) abordam a classificacao de forma discriminativa, encontrando o hiperplano de margem maxima no espaco de features que separa as classes. Para texto, as features de entrada sao tipicamente vetores TF-IDF. As SVMs sao eficazes em espacos de alta dimensionalidade (vocabularios grandes) mesmo com relativamente poucos exemplos de treinamento, porque o objetivo de margem maxima fornece regularizacao implicita. Joachims (1998) demonstrou que as SVMs consistentemente superaram Naive Bayes, k-vizinhos mais proximos e arvores de decisao no benchmark de classificacao de textos Reuters-21578. A SVM linear, em particular, permanece como uma baseline forte que e dificil de superar sem substancialmente mais dados ou representacoes mais expressivas.

**Regressao logistica** ocupa uma posicao intermediaria: e discriminativa como as SVMs, mas produz estimativas de probabilidade calibradas (apos regularizacao apropriada). Para configuracoes multi-classe, a extensao softmax (regressao logistica multinomial) mapeia as predicoes lineares para uma distribuicao de probabilidade sobre as classes:

$$P(y = c \mid \vec{x}) = \frac{\exp(\vec{w}_c^T \vec{x} + b_c)}{\sum_{j=1}^{C} \exp(\vec{w}_j^T \vec{x} + b_j)}$$

A regressao logistica sobre features TF-IDF e a baseline de classificacao apenas lexical padrao nos experimentos do TalkEx (Capitulo 6).

### 2.4.2 Gradient Boosted Trees para Features Heterogeneas

Classificadores de texto tradicionais operam sobre representacoes de features homogeneas — tipicamente um unico vetor TF-IDF ou um unico vetor de embedding. Na pratica, sistemas de classificacao conversacional tem acesso a multiplas familias de features: features lexicais (TF-IDF, scores BM25), features semanticas (vetores de embeddings), features estruturais (papel do falante, posicao do turno, comprimento da conversa) e features derivadas de regras (indicadores binarios de ativacoes de regras). Essas familias de features diferem em dimensionalidade, escala, distribuicao e conteudo informacional.

Gradient boosted decision trees (GBDT), e especificamente suas implementacoes modernas como XGBoost (Chen e Guestrin, 2016) e LightGBM (Ke et al., 2017), sao naturalmente adequados para features heterogeneas. Arvores de decisao particionam o espaco de features por meio de divisoes alinhadas aos eixos, lidando inerentemente com tipos mistos de features (continuas, categoricas, binarias) sem a normalizacao e escalonamento que modelos lineares e redes neurais requerem. O ensemble e construido sequencialmente, com cada arvore ajustando os erros residuais do ensemble anterior:

$$F_m(\vec{x}) = F_{m-1}(\vec{x}) + \eta \cdot h_m(\vec{x})$$

onde $h_m$ e a $m$-esima arvore, $\eta$ e a taxa de aprendizado, e a funcao de perda (entropia cruzada para classificacao) e otimizada via descida de gradiente funcional.

O LightGBM introduz duas otimizacoes-chave sobre o XGBoost. **Gradient-based one-side sampling** (GOSS) retem todos os exemplos com gradientes grandes (exemplos dificeis) enquanto amostra aleatoriamente exemplos com gradientes pequenos, reduzindo o tempo de treinamento sem perda significativa de acuracia. **Exclusive feature bundling** (EFB) identifica features que raramente assumem valores nao nulos simultaneamente e as agrupa em uma unica feature, reduzindo a dimensionalidade efetiva das features. Essas otimizacoes sao particularmente relevantes para classificacao conversacional, onde o vetor de features pode incluir um embedding de 384 dimensoes, um vetor TF-IDF de alta dimensionalidade e um punhado de features binarias estruturais e de regras — uma mistura que o EFB lida eficientemente.

Nos experimentos do TalkEx, o LightGBM com configuracao $n_{\text{estimators}} = 100$ e $\text{num\_leaves} = 31$ e o classificador primario. Essa configuracao fornece um bom equilibrio entre complexidade do modelo e generalizacao para o tamanho do dataset de 2.122 registros, conforme validado no conjunto de validacao retido (Capitulo 6).

### 2.4.3 Abordagens Neurais

Classificadores neurais de texto aprendem tanto representacoes quanto fronteiras de decisao de ponta a ponta. O **Multi-Layer Perceptron (MLP)** e o classificador neural mais simples: uma ou mais camadas ocultas com ativacoes nao lineares seguidas de uma camada de saida softmax. Quando a entrada e um vetor de embedding pre-computado, o MLP aprende um mapeamento nao linear do espaco de embeddings para o espaco de rotulos. Essa abordagem tem duas vantagens sobre classificadores lineares: pode capturar interacoes nao lineares entre dimensoes do embedding, e pode modelar fronteiras de classe que nao sao linearmente separaveis no espaco de embeddings.

**Transformers com fine-tuning** representam o extremo oposto do espectro de complexidade. Em vez de usar embeddings congelados como entrada para um classificador separado, o encoder inteiro e atualizado conjuntamente com uma cabeca de classificacao durante o treinamento. Isso permite que o encoder adapte suas representacoes a tarefa de classificacao especifica, potencialmente aprendendo features especificas do dominio que o objetivo de pre-training nao otimizou. Devlin et al. (2019) demonstraram que realizar fine-tuning do BERT para classificacao de textos alcancou resultados estado-da-arte em multiplos benchmarks, frequentemente com apenas alguns milhares de exemplos rotulados.

O custo do fine-tuning e substancial: milhoes de parametros sao atualizados durante o treinamento, exigindo computacao em GPU e selecao cuidadosa de hiperparametros (taxa de aprendizado, passos de warmup, decaimento de pesos). Para o dataset do TalkEx de 2.122 registros, realizar fine-tuning de um encoder MiniLM de 33M parametros arrisca overfitting — uma preocupacao que motiva o design com encoder congelado e que a comparacao experimental entre encoders congelados e com fine-tuning (discutida no Capitulo 7) quantificaria diretamente.

### 2.4.4 Metricas de Avaliacao Multi-Classe

A avaliacao de classificacao em configuracoes multi-classe requer escolha cuidadosa da estrategia de agregacao, pois diferentes metricas enfatizam diferentes aspectos do desempenho.

**Macro-F1** calcula o F1 score independentemente para cada classe e entao faz a media:

$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

onde $P_c$ e $R_c$ sao a precision e o recall para a classe $c$. O Macro-F1 pondera todas as classes igualmente independentemente de sua frequencia, tornando-o sensivel ao desempenho em classes minoritarias. Essa propriedade e critica no dataset do TalkEx, onde as frequencias de classe variam de 47 exemplos de teste (saudacao) a 121 exemplos de teste (cancelamento) — uma razao de quase 3:1.

**Micro-F1** agrega verdadeiros positivos, falsos positivos e falsos negativos em todas as classes antes de calcular:

$$\text{Micro-F1} = \frac{2 \cdot \sum_c TP_c}{2 \cdot \sum_c TP_c + \sum_c FP_c + \sum_c FN_c}$$

O Micro-F1 e dominado pelas classes majoritarias, e em configuracoes balanceadas ou quase balanceadas ele aproxima a acuracia geral. Nos experimentos do TalkEx, o Macro-F1 e a metrica primaria porque o tratamento igualitario de todas as classes de intencao reflete o requisito operacional: um sistema que classifica bem "cancelamento" mas falha em "compra" nao e operacionalmente aceitavel.

### 2.4.5 Calibracao: Brier Score e Expected Calibration Error

Alem da acuracia discriminativa, a confiabilidade das probabilidades preditas importa para sistemas que usam scores de confianca para decisoes downstream — como encaminhar predicoes de baixa confianca para revisao humana ou acionar estagios de inferencia em cascata.

O **Brier score** (Brier, 1950) mede a diferenca quadratica media entre as probabilidades preditas e os resultados reais:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} (p_{ic} - y_{ic})^2$$

onde $p_{ic}$ e a probabilidade predita de que o exemplo $i$ pertence a classe $c$, e $y_{ic}$ e a variavel indicadora (1 se correto, 0 caso contrario). O Brier score se decompoe em componentes de confiabilidade (calibracao), resolucao (discriminacao) e incerteza (Murphy, 1973), fornecendo um quadro completo da qualidade de predicao probabilistica.

**Expected Calibration Error (ECE)** (Naeini et al., 2015) particiona as predicoes em bins por nivel de confianca e mede a diferenca absoluta media entre a confianca predita e a acuracia real dentro de cada bin:

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |acc_b - conf_b|$$

onde $B$ e o numero de bins, $n_b$ e o numero de predicoes no bin $b$, $acc_b$ e a acuracia empirica no bin, e $conf_b$ e a confianca media predita. Um modelo perfeitamente calibrado tem ECE = 0. Redes neurais modernas e gradient boosted trees sao conhecidos por serem excessivamente confiantes (Guo et al., 2017), produzindo probabilidades preditas que sistematicamente excedem a verdadeira probabilidade de correcao. Metodos de calibracao pos-hoc, como Platt scaling (Platt, 1999) e temperature scaling (Guo et al., 2017), podem reduzir essa descalibracao.

A calibracao e diretamente relevante para a estrategia de inferencia em cascata do TalkEx (Secao 2.7), onde thresholds de confianca determinam se uma predicao e aceita ou escalada para um estagio de processamento mais caro. Um modelo descalibrado que produz scores de confianca de 0.95 para predicoes com acuracia real de 0.70 encaminharia poucos casos ao estagio caro, degradando a qualidade geral do sistema.

---

## 2.5 Sistemas de Regras Deterministicos

### 2.5.1 NLP Baseado em Regras: Contexto Historico

Abordagens baseadas em regras para processamento de linguagem natural antecedem metodos estatisticos em decadas. O ELIZA (Weizenbaum, 1966) demonstrou que regras de correspondencia de padroes sobre texto superficial podiam produzir comportamento conversacional notavelmente convincente, apesar de nao codificar nenhum conhecimento linguistico. Sistemas iniciais de extracao de informacao, como aqueles desenvolvidos para as Conferencias de Compreensao de Mensagens (MUC-3 a MUC-7, 1991--1998), dependiam extensivamente de padroes de extracao construidos manualmente, alcancando alta precisao em dominios restritos.

A revolucao estatistica dos anos 1990 e 2000 — iniciada por Brown et al. (1990) para traducao automatica e Jelinek (1997) para reconhecimento de fala — deslocou as abordagens baseadas em regras em muitas tarefas de NLP. No entanto, Chiticariu et al. (2013) argumentaram persuasivamente que sistemas baseados em regras retem vantagens importantes em ambientes de producao: sao transparentes (o comportamento de uma regra pode ser inspecionado e compreendido), deterministicos (a mesma entrada sempre produz a mesma saida) e modificaveis (uma mudanca de regra de negocio e uma edicao de texto, nao um ciclo de retreinamento de modelo). Em sua pesquisa sobre implantacoes de NLP empresarial, Chiticariu et al. constataram que a extracao de informacao baseada em regras permanecia dominante na industria apesar da preferencia academica por metodos estatisticos.

A visao contemporanea, que informa o design do TalkEx, trata regras e modelos estatisticos como complementares em vez de concorrentes. Regras fornecem cobertura garantida para padroes conhecidos, produzem evidencia auditavel para cada decisao e impoem custo zero de inferencia por avaliacao (sem passagem direta de modelo). Modelos estatisticos fornecem cobertura para padroes nao explicitamente codificados em regras e generalizam entre variacoes lexicais. A estrategia de integracao — como regras e modelos interagem — determina o valor pratico da combinacao.

### 2.5.2 Motor de Regras Baseado em AST

Um motor de regras moderno opera em duas fases: compilacao e avaliacao. Na fase de compilacao, regras expressas em uma linguagem especifica de dominio (DSL) legivel por humanos sao parseadas em uma arvore sintatica abstrata (AST). Na fase de avaliacao, a AST e percorrida para cada entrada, e os predicados sao avaliados contra as features da entrada.

A representacao em AST fornece diversas vantagens sobre a correspondencia direta de padroes. Primeiro, permite analise estatica: o compilador pode detectar regras conflitantes, condicoes inalcancaveis e predicados redundantes antes da execucao. Segundo, permite otimizacao: predicados podem ser reordenados por custo (verificacoes lexicais baratas antes de computacoes semanticas caras), e a avaliacao em curto-circuito termina antecipadamente quando um ramo e determinado como falso. Terceiro, fornece uma estrutura natural para producao de evidencia: cada predicado avaliado produz um registro de rastreamento indicando o que foi verificado, qual valor foi observado e se a verificacao passou ou falhou.

Uma regra na DSL do TalkEx tem a seguinte estrutura abstrata:

```
RULE <nome>
  WHEN <predicado_1> AND <predicado_2> AND ...
  THEN <acao>
```

Os predicados pertencem a quatro familias: **lexicais** (presenca de palavra-chave, correspondencia de regex, threshold de score BM25), **semanticos** (similaridade de embedding com vetores prototipo, threshold de score de intencao), **estruturais** (papel do falante, posicao do turno, comprimento da conversa, canal) e **contextuais** (recorrencia de padrao dentro de uma janela deslizante, satisfacao sequencial de predicados entre turnos). As acoes incluem rotulacao, pontuacao e atribuicao de prioridade. Cada avaliacao de predicado produz um registro de evidencia que e anexado a saida final, permitindo auditabilidade completa de cada decisao.

A arquitetura do motor de regras do TalkEx e descrita em detalhe no Capitulo 4.

### 2.5.3 Estrategias de Integracao: Regras e Aprendizado de Maquina

Tres estrategias principais existem para integrar regras deterministicas com classificadores estatisticos:

**Regras-como-override.** O classificador produz uma predicao; as regras examinam a predicao e as features de entrada, e sobrescrevem a predicao quando condicoes especificas sao atendidas. Esta e a estrategia de integracao dominante em sistemas de producao (Chiticariu et al., 2013). Sua vantagem e a simplicidade: o classificador e as regras operam independentemente, com as regras atuando como uma rede de seguranca de pos-processamento. Sua limitacao e que o classificador nao se beneficia do conhecimento do sistema de regras durante o treinamento — os dois componentes sao desacoplados.

**Regras-como-features.** Cada regra e avaliada nos dados de treinamento, e sua saida (binaria disparo/nao-disparo ou um score continuo) e adicionada ao vetor de features utilizado pelo classificador. Isso permite que o classificador aprenda o valor condicional das ativacoes de regras em combinacao com outras features. A vantagem e uma integracao mais forte: o classificador pode aprender, por exemplo, que uma regra de palavra-chave "cancelamento" disparando em combinacao com uma alta similaridade semantica com o prototipo de "reclamacao" e mais indicativa de reclamacao do que de cancelamento. A limitacao e que adicionar features de regras aos dados de treinamento altera a distribuicao de features, exigindo retreinamento sempre que o conjunto de regras e modificado.

**Regras-como-pos-processamento.** As regras operam sobre os rotulos e scores de saida do classificador, aplicando correcoes deterministicas. Por exemplo, uma regra pode reclassificar qualquer predicao com confianca abaixo de um threshold como "incerto", ou uma regra pode sobrescrever uma predicao "saudacao" quando a conversa contem mais de 10 turnos (conversas longas sao improvaveis de serem meros cumprimentos). Essa estrategia nao requer retreinamento e fornece correcoes transparentes e auditaveis, mas nao pode melhorar as representacoes internas do classificador.

Nos experimentos do TalkEx (Capitulo 6), a estrategia de regras-como-features e avaliada contra regras-como-override e comparada com a baseline apenas-ML. A hipotese (H3) testa se a integracao de regras deterministicas melhora a qualidade da classificacao alem do que features estatisticas sozinhas alcancam.

### 2.5.4 Requisitos de Explicabilidade e Auditabilidade

Em sistemas de NLP em producao implantados em industrias reguladas — telecomunicacoes, servicos financeiros, saude — a capacidade de explicar e auditar decisoes individuais nao e uma funcionalidade desejavel, mas um requisito regulatorio. O AI Act da Uniao Europeia (2024) introduz obrigacoes para "sistemas de IA de alto risco", incluindo transparencia, supervisao humana e manutencao de registros. A Lei Geral de Protecao de Dados do Brasil (LGPD, 2018) garante aos titulares de dados o direito de solicitar explicacoes sobre decisoes automatizadas que afetem seus interesses.

Sistemas baseados em regras fornecem um mecanismo natural para atender a esses requisitos. Cada avaliacao de regra produz um rastro: qual regra foi avaliada, quais features de entrada foram examinadas, quais valores foram observados, quais thresholds foram aplicados e qual resultado foi alcancado. Esse rastro constitui uma explicacao legivel por humanos da decisao que pode ser inspecionada, contestada e corrigida. Modelos estatisticos, por outro lado, requerem metodos de explicacao pos-hoc (LIME, SHAP) que fornecem explicacoes aproximadas e por vezes enganosas do comportamento do modelo (Rudin, 2019).

A abordagem do TalkEx — combinando classificadores estatisticos (que fornecem ampla cobertura e generalizacao) com regras deterministicas (que fornecem evidencia auditavel para decisoes de alto risco) — e projetada para satisfazer tanto os requisitos de acuracia da implantacao operacional quanto os requisitos de transparencia da conformidade regulatoria. Esse objetivo duplo nao e padrao na literatura, onde acuracia de classificacao e explicabilidade sao tipicamente tratadas como objetivos concorrentes em vez de co-requisitos. A medida em que o TalkEx alcanca esse objetivo duplo e avaliada no Capitulo 6.

---

## 2.6 NLP Conversacional

### 2.6.1 Classificacao de Atos de Dialogo e Deteccao de Intencao

A classificacao de atos de dialogo atribui rotulos funcionais a elocucoes em uma conversa, capturando a intencao comunicativa de cada turno. A taxonomia fundacional e o esquema DAMSL (Dialogue Act Markup in Several Layers) (Core e Allen, 1997), que define uma hierarquia de funcoes comunicativas incluindo declaracoes, perguntas, diretivas e comissivas. O corpus Switchboard-DAMSL (Jurafsky et al., 1997) aplicou esse esquema a conversas telefonicas, produzindo uma taxonomia de 42 classes que permanece como um benchmark padrao.

A deteccao de intencao, como estudada na literatura de dialogo orientado a tarefas, e uma especializacao da classificacao de atos de dialogo que atribui elocucoes do usuario a um conjunto predefinido de intencoes relevantes para um dominio de servico especifico. O benchmark ATIS (Airline Travel Information System) (Tur et al., 2010) contem 4.978 elocucoes de treinamento em 21 classes de intencao; o dataset SNIPS (Coucke et al., 2018) contem 13.784 elocucoes de treinamento em 7 intencoes. Ambos sao benchmarks de turno unico: cada elocucao e classificada independentemente, sem contexto conversacional.

Essa suposicao de turno unico e uma limitacao critica para aplicacoes de atendimento ao cliente. Em uma conversa de central de atendimento, a intencao do cliente frequentemente nao se cristaliza em uma unica elocucao. Um cliente que diz "eu recebi minha fatura" no turno 1, "o valor esta diferente do que me informaram" no turno 3 e "quero falar com um gerente" no turno 7 esta expressando uma intencao de reclamacao-e-escalacao que nenhum turno individual captura completamente. A abordagem do TalkEx para esse problema — construir janelas de contexto deslizantes sobre turnos adjacentes e gerar embeddings em multiplas granularidades — e descrita na Secao 2.6.2 e implementada no Capitulo 4.

### 2.6.2 Modelagem de Contexto Multi-Turno

Modelar o contexto conversacional requer arquiteturas que capturam dependencias entre turnos. Diversas abordagens foram propostas.

**Representacoes por janela deslizante** concatenam ou fazem a media das representacoes de $w$ turnos adjacentes em um unico vetor de contexto. O tamanho da janela $w$ controla o trade-off entre precisao local (janela pequena $w$ captura contexto imediato) e cobertura global (janela grande $w$ captura dependencias de longo alcance). O passo $s$ determina a sobreposicao entre janelas adjacentes: quando $s < w$, as janelas se sobrepoem e cada turno contribui para multiplas representacoes de contexto. Essa abordagem e computacionalmente simples e interpretavel — o conteudo de cada janela e diretamente inspecionavel — mas assume que o contexto relevante esta localizado dentro de um intervalo fixo.

**Encoders hierarquicos** (Li et al., 2015; Serban et al., 2016) processam conversas em dois niveis: um encoder em nivel de palavra produz representacoes de turnos, e um encoder em nivel de turno (tipicamente uma rede recorrente ou Transformer) processa a sequencia de representacoes de turnos. O encoder em nivel de turno aprende a ponderar e combinar turnos com base em sua relevancia para a tarefa de classificacao. A Hierarchical Attention Network (HAN) (Yang et al., 2016) estende essa abordagem com attention em ambos os niveis, aprendendo quais palavras dentro de um turno e quais turnos dentro de um documento sao mais relevantes.

**Modelos baseados em grafos** (Ghosal et al., 2019) representam conversas como grafos direcionados onde nos sao turnos e arestas codificam relacoes de adjacencia, mesmo-falante e falante-cruzado. Redes neurais em grafos entao propagam informacao pela estrutura da conversa, permitindo que a representacao de cada turno incorpore informacao de turnos estruturalmente relacionados. O modelo DialogueGCN demonstrou que modelar explicitamente as relacoes entre falantes melhora o reconhecimento de emocoes em conversas, sugerindo que representacoes conscientes do falante podem beneficiar a classificacao de intencoes tambem.

O TalkEx adota a abordagem de janela deslizante por sua combinacao de simplicidade, interpretabilidade e eficacia, conforme descrito no Capitulo 4. O tamanho da janela e o passo sao parametros configuraveis cujos valores sao determinados empiricamente. O embedding e gerado no nivel da janela usando o encoder multilingue congelado, e a concatenacao desses embeddings em nivel de janela com features lexicais e estruturais forma a entrada do classificador.

### 2.6.3 Supervisao Fraca: Heranca de Rotulos

Em muitos ambientes operacionais, rotulos estao disponiveis no nivel da conversa (a partir de codigos de disposicao do agente, pesquisas com clientes ou taxonomias de topicos), mas nao no nivel do turno ou da janela. Isso cria um problema de supervisao fraca: o classificador opera no nivel sub-conversa (janelas), mas o sinal de treinamento esta disponivel apenas no nivel da conversa.

A abordagem mais simples — **heranca de rotulos** — atribui o rotulo em nivel de conversa a todas as janelas dentro da conversa. Isso assume que toda janela em uma conversa rotulada como "cancelamento" exibe a intencao de cancelamento, o que e claramente falso: uma conversa de cancelamento de 20 turnos contera turnos de cumprimento, turnos de verificacao e turnos de despedida que nao expressam intencao de cancelamento. A heranca de rotulos introduz ruido de rotulo proporcional a heterogeneidade tematica da conversa.

Abordagens mais sofisticadas incluem **aprendizado de multiplas instancias** (MIL) (Dietterich et al., 1997), onde a conversa e tratada como um saco de janelas e o algoritmo de aprendizado deve identificar quais instancias dentro do saco sao responsaveis pelo rotulo em nivel de saco. O MIL baseado em attention (Ilse et al., 2018) aprende a ponderar instancias por sua relevancia para o rotulo do saco, efetivamente descobrindo as janelas que carregam o sinal de intencao. O Snorkel (Ratner et al., 2017) fornece um framework de supervisao fraca programatica onde multiplas funcoes de rotulacao votam em cada instancia, e um modelo de rotulacao aprende a remover o ruido dos votos.

O protocolo experimental do TalkEx utiliza heranca de rotulos — rotulos em nivel de conversa aplicados a todas as janelas — e reconhece isso como uma fonte de ruido de rotulo. O impacto desse ruido na qualidade da classificacao e um fator relevante na interpretacao dos resultados experimentais no Capitulo 6. O uso de Macro-F1 como metrica primaria mitiga parcialmente o efeito, pois o ruido de rotulo afeta classes de forma diferente dependendo da proporcao de janelas semanticamente fora do topico dentro das conversas de cada classe.

### 2.6.4 Desafios Especificos do PT-BR

Texto de atendimento ao cliente em portugues brasileiro apresenta desafios especificos que compoe as dificuldades gerais do NLP conversacional.

**Diacriticos e inconsistencia de codificacao.** O portugues utiliza sinais diacriticos (acentos, cedilhas, tils) que sao representados de forma inconsistente tanto na saida de ASR quanto no texto digitado pelo cliente. "Cancelamento" pode aparecer como "cancelamento", "cancelamento" ou "cancelamênto" dependendo do sistema de ASR, do teclado do cliente ou da codificacao da plataforma de chat. A normalizacao eficaz — decomposicao NFKD, remocao de acentos, uniformizacao de caixa — e pre-requisito tanto para a recuperacao lexical quanto para a classificacao. O pipeline de normalizacao do TalkEx, descrito no Capitulo 4, aplica essas transformacoes uniformemente.

**Registro informal e abreviacoes.** O chat de atendimento ao cliente contem linguagem informal extensiva: "vc" (voce), "td" (tudo), "blz" (beleza), "pq" (porque), "mt" (muito), "qro" (quero). Essas abreviacoes nao estao presentes no texto formal em portugues sobre o qual os encoders multilingues sao primariamente treinados, criando uma potencial lacuna de dominio entre as distribuicoes de pre-training e implantacao. A medida em que o encoder multilingue congelado lida com esse registro informal — versus se o fine-tuning com dados in-domain melhoraria o desempenho — e uma questao empirica em aberto relacionada a comparacao congelado-versus-fine-tuned discutida na Secao 2.2.5.

**Alternancia de codigo (code-switching).** Conversas de atendimento ao cliente brasileiras frequentemente incorporam termos em ingles, particularmente para produtos de tecnologia: "upgrade", "app", "download", "feedback", "login". Essa alternancia de codigo e tratada naturalmente por encoders multilingues que viram tanto portugues quanto ingles durante o pre-training, mas cria desafios para sistemas de recuperacao lexical que podem nao indexar stop words em ingles ou aplicar regras de normalizacao do portugues a termos em ingles.

**Variacao regional.** O tamanho continental do Brasil produz variacao linguistica regional significativa. Expressoes, idiomas e vocabulario diferem entre regioes, e um cliente do nordeste pode usar palavras diferentes para a mesma intencao que um cliente do sul. O pre-training amplo do encoder multilingue absorve parcialmente essa variacao, mas o dataset do TalkEx (extraido de um unico corpus sintetico) nao representa sistematicamente a diversidade regional — uma limitacao reconhecida no Capitulo 7.

---

## 2.7 Inferencia em Cascata

### 2.7.1 Aprendizado de Maquina com Consciencia de Custo

O paradigma padrao na pesquisa em aprendizado de maquina otimiza para acuracia, tratando o custo computacional como uma preocupacao secundaria. Em sistemas de producao que processam milhoes de conversas por mes, o custo de inferencia e uma restricao de primeira ordem. O custo por predicao determina a viabilidade operacional do sistema: um modelo que alcanca 2% mais acuracia mas requer 10x o orcamento computacional pode ser economicamente inferior a alternativa mais barata.

A inferencia em cascata aborda essa tensao organizando modelos em uma sequencia de estagios com custo e acuracia crescentes. Modelos simples e baratos lidam com casos faceis; apenas casos que o modelo barato nao consegue resolver com confianca suficiente sao escalados para modelos caros. A cascata e governada por thresholds de confianca: em cada estagio, se a probabilidade predita excede um threshold $\tau_k$, a predicao e aceita; caso contrario, a entrada e passada para o estagio $k+1$.

A fundamentacao teorica para classificadores em cascata foi estabelecida por Viola e Jones (2001) no contexto de deteccao facial. Sua cascata de classificadores de features Haar com complexidade crescente alcancou deteccao facial em tempo real rejeitando aproximadamente 50% dos patches de imagem nao-faciais em cada estagio, de modo que o classificador final caro processava apenas uma pequena fracao da entrada. O insight-chave e que a maioria das entradas e "facil" — pode ser corretamente classificada por um modelo simples — e apenas uma minoria de entradas "dificeis" requer o investimento computacional completo.

### 2.7.2 Inferencia em Cascata em NLP

A aplicacao de inferencia em cascata a tarefas de NLP recebeu menos atencao do que em visao computacional, embora diversos precedentes relevantes existam.

No ranqueamento de busca web, o paradigma retrieve-then-rerank (Matveeva et al., 2006; Nogueira e Cho, 2019) e uma cascata de dois estagios: um recuperador rapido de primeiro estagio (BM25) produz um conjunto de candidatos de centenas de documentos, e um ranqueador lento de segundo estagio (cross-encoder neural) reranqueia apenas os candidatos. O primeiro estagio tem custo $O(N)$ sobre a colecao completa (com otimizacao por indice invertido), enquanto o segundo estagio tem custo $O(K)$ sobre os top-$K$ candidatos. A razao de cascata $K/N$ determina a economia de custo.

Schwartz et al. (2020) propuseram estrategias de saida antecipada para modelos Transformer, onde a classificacao pode ser realizada em camadas intermediarias em vez de esperar pela passagem direta completa. Se a confianca da predicao em uma camada inicial excede um threshold, as camadas restantes sao puladas. Isso reduz o custo medio de inferencia enquanto preserva a acuracia em exemplos faceis. O BERxiT (Xin et al., 2021) formalizou essa abordagem com classificadores de saida aprendidos em cada camada do Transformer, alcancando 40% de reducao de latencia com menos de 1% de degradacao de acuracia nos benchmarks GLUE.

Na classificacao multi-rotulo, abordagens em cascata foram usadas para aplicar classificadores cada vez mais especializados em sequencia: um primeiro estagio identifica a categoria ampla, e estagios subsequentes discriminam dentro da categoria. Essa cascata hierarquica reduz o numero efetivo de classes em cada estagio, simplificando o problema de classificacao.

### 2.7.3 Requisitos de Calibracao para Roteamento Baseado em Confianca

Sistemas de inferencia em cascata dependem fundamentalmente da confiabilidade dos scores de confianca. O threshold de cascata $\tau_k$ determina o trade-off entre economia de custo (threshold $\tau_k$ mais alto aceita menos predicoes, escalando mais para o estagio caro) e acuracia (threshold $\tau_k$ mais baixo aceita mais predicoes, incluindo algumas incorretas). Para que esse trade-off seja significativo, a confianca predita deve refletir com precisao a verdadeira probabilidade de correcao.

Um modelo que prediz confianca $p = 0.90$ deveria estar correto aproximadamente 90% do tempo em todas as predicoes com confianca proxima a 0.90. Se o modelo e excessivamente confiante — predizendo $p = 0.90$ quando a acuracia real e 0.70 — entao um threshold $\tau = 0.85$ aceitara predicoes que estao corretas apenas 70% do tempo, degradando a precisao do sistema. Por outro lado, um modelo sub-confiante escalara muitos casos para o estagio caro, anulando as economias de custo.

As metricas de calibracao introduzidas na Secao 2.4.5 — Brier score e ECE — quantificam diretamente essa confiabilidade. Nos experimentos do TalkEx, a estrategia de inferencia em cascata (H4) depende das probabilidades preditas pelo LightGBM para rotear predicoes. Os resultados experimentais (Capitulo 6) demonstram que a estrategia de cascata, como implementada, nao reduz o custo — uma constatacao que pode ser parcialmente atribuivel a natureza deterministica da configuracao experimental (variancia zero entre seeds, conforme discutido no Capitulo 6), que limita a variancia nos scores de confianca dos quais o mecanismo de cascata depende.

### 2.7.4 Consideracoes de Projeto para Cascatas em Producao

Diversas consideracoes praticas governam o projeto de sistemas de inferencia em cascata em producao:

**Ordenacao de estagios.** Os estagios devem ser ordenados por custo computacional crescente. No TalkEx, a cascata conceitual e: (1) classificacao baseada em regras para padroes conhecidos (custo de inferencia zero), (2) classificador leve sobre features lexicais (custo minimo), (3) classificador sobre features lexicais mais embeddings (custo moderado, requer computacao de embeddings) e (4) reranqueamento com cross-encoder ou revisao humana (custo alto). A eficacia dessa ordenacao depende da proporcao de entradas resolvidas em cada estagio.

**Selecao de threshold.** Os thresholds de cascata podem ser definidos para otimizar diferentes objetivos: minimizar o custo total sujeito a uma restricao de acuracia, maximizar a acuracia sujeita a um orcamento de custo, ou minimizar uma combinacao ponderada. A otimizacao bayesiana sobre o espaco de thresholds, avaliada em um conjunto de validacao, fornece uma abordagem fundamentada. Na pratica, os thresholds sao frequentemente definidos conservadoramente (threshold $\tau$ alto) na implantacao e relaxados incrementalmente a medida que o comportamento do sistema e observado.

**Design de fallback.** Quando nenhum estagio na cascata produz uma predicao com confianca suficiente, um mecanismo de fallback e necessario. As opcoes incluem abstencao (retornar "incerto" sem rotulo), encaminhamento para revisao humana ou retorno da predicao mais confiante com uma flag de baixa confianca. A escolha da estrategia de fallback depende do custo dos erros versus o custo da abstencao no contexto operacional especifico.

**Monitoramento e deteccao de drift.** Em producao, a distribuicao das entradas evolui ao longo do tempo: novos produtos, novos padroes de reclamacao, novo vocabulario. Um sistema de cascata deve monitorar a proporcao de entradas escaladas em cada estagio. Uma taxa de escalacao crescente sinaliza drift de distribuicao — os estagios baratos nao estao mais resolvendo casos que anteriormente lidavam — e aciona retreinamento ou recalibracao de thresholds.

A arquitetura de cascata do TalkEx, descrita no Capitulo 4, implementa os estagios conceituais e avalia sua eficacia no Capitulo 6. A constatacao experimental de que a cascata nao reduziu o custo (H4 refutada) motiva uma discussao sobre as condicoes sob as quais a inferencia em cascata e benefica e as modificacoes de projeto que poderiam torna-la eficaz em iteracoes futuras (Capitulo 7).

---

## 2.8 Resumo do Capitulo

Este capitulo estabeleceu os fundamentos teoricos para os quatro pilares da arquitetura do TalkEx:

1. **Recuperacao** (Secoes 2.1 e 2.3): O BM25 fornece uma baseline lexical forte e interpretavel; sentence embeddings densos permitem generalizacao semantica; a fusao hibrida combina ambos os sinais; e o reranqueamento com cross-encoder fornece refinamento opcional de segundo estagio.

2. **Representacao** (Secao 2.2): A progressao de word embeddings estaticos atraves de encoders contextuais ate representacoes em nivel de sentenca via treinamento siames define o paradigma de embeddings utilizado no TalkEx. O encoder multilingue MiniLM congelado representa um trade-off deliberado entre otimizacao especifica para a tarefa e estabilidade operacional.

3. **Classificacao** (Secao 2.4): Classificadores tradicionais sobre TF-IDF fornecem baselines lexicais; gradient boosted trees (LightGBM) lidam nativamente com features heterogeneas de multiplas familias de sinais; abordagens neurais fornecem uma referencia de teto; e metricas de calibracao (Brier score, ECE) quantificam a confiabilidade necessaria para roteamento baseado em confianca.

4. **Auditabilidade e controle de custo** (Secoes 2.5 e 2.7): Motores de regras deterministicos fornecem decisoes transparentes e auditaveis para casos de alto risco; a inferencia em cascata gerencia o custo computacional resolvendo casos faceis antecipadamente; e a integracao de regras e classificadores e formalizada por meio de tres estrategias distintas avaliadas experimentalmente.

O contexto conversacional (Secao 2.6) adiciona uma quinta dimensao: modelagem multi-turno, supervisao fraca e os desafios especificos do texto de atendimento ao cliente em PT-BR. Juntos, esses fundamentos definem o espaco de projeto dentro do qual o TalkEx opera e as metricas contra as quais e avaliado no Capitulo 6.

Os trabalhos relacionados que posicionam o TalkEx em relacao a sistemas existentes sao discutidos no Capitulo 3. A arquitetura concreta que operacionaliza esses fundamentos e descrita no Capitulo 4. O protocolo experimental que testa as hipoteses resultantes e detalhado no Capitulo 5.
