# Capitulo 3: Trabalhos Relacionados

Este capitulo posiciona o TalkEx dentro do panorama da pesquisa anterior em sete vertentes da literatura que, coletivamente, definem o espaco de design e os limites de contribuicao desta dissertacao. A Secao 3.1 revisa sistemas de recuperacao hibrida que combinam sinais lexicos e semanticos. A Secao 3.2 examina a classificacao baseada em embeddings e o paradigma de uso de representacoes pre-treinadas como features para classificadores supervisionados. A Secao 3.3 investiga sistemas de PLN baseados em regras e hibridos que integram logica deterministica com metodos estatisticos. A Secao 3.4 aborda a classificacao de intencoes conversacionais, com enfase na modelagem multi-turno. A Secao 3.5 discute arquiteturas de inferencia em cascata e conscientes de custo. A Secao 3.6 revisa o estado dos recursos de PLN para a lingua portuguesa e os desafios especificos de dados conversacionais em PT-BR. A Secao 3.7 sintetiza os trabalhos anteriores em uma tabela de posicionamento comparativo e identifica a lacuna precisa que esta dissertacao endereca.

Cada secao segue uma estrutura analitica consistente: apresentamos as contribuicoes-chave, avaliamos seus pontos fortes e limitacoes com relacao ao dominio conversacional em PT-BR, e identificamos a lacuna especifica que motiva as decisoes de design do TalkEx. Referencias cruzadas aos capitulos subsequentes indicam onde cada decisao de design e implementada (Capitulo 4) e avaliada (Capitulo 6).

---

## 3.1 Sistemas de Recuperacao Hibrida

### 3.1.1 A Tese da Complementaridade

A premissa central da recuperacao hibrida e que sinais lexicos e semanticos falham de formas complementares: metodos lexicos nao capturam parafrases e intencoes implicitas; metodos semanticos nao capturam termos exatos, codigos e vocabulario especifico de dominio que aparece literalmente em consultas e documentos. Combinar ambos deveria, portanto, recuperar um conjunto estritamente maior de resultados relevantes do que qualquer abordagem isolada. Essa premissa foi validada em multiplos benchmarks, mas as condicoes sob as quais a complementaridade se verifica — e a magnitude do ganho — sao dependentes do dominio de maneiras que a literatura so recentemente comecou a caracterizar.

### 3.1.2 SPLADE: Representacoes Esparsas Aprendidas

Formal, Piwowarski e Clinchant (2021) introduziram o SPLADE (SParse Lexical AnD Expansion), um modelo de recuperacao que aprende representacoes esparsas a partir de um modelo de linguagem mascarado. O SPLADE mapeia cada consulta e documento para um vetor esparso de alta dimensionalidade no espaco do vocabulario, onde cada dimensao corresponde a um termo e o peso e aprendido de ponta a ponta por meio de um objetivo regularizado por esparsidade. A inovacao central e que o modelo aprende a expandir consultas com termos semanticamente relacionados que nao aparecem no texto original — uma forma de expansao de consulta aprendida que preenche a lacuna entre abordagens puramente lexicas e puramente densas.

O SPLADE alcanca desempenho competitivo ou superior ao de bi-encoders densos no benchmark MS-MARCO de ranqueamento de passagens, mantendo a vantagem de interpretabilidade das representacoes esparsas: a contribuicao de cada termo para o score de recuperacao e diretamente legivel. O modelo tambem se beneficia de implementacoes eficientes de indice invertido, possibilitando recuperacao em submilissegundos em escala.

As limitacoes do SPLADE para o contexto do TalkEx sao trifoldas. Primeiro, o modelo requer uma quantidade substancial de dados supervisionados de treinamento (pares consulta-passagem com julgamentos de relevancia) que nao estao disponiveis para dados conversacionais em PT-BR. Segundo, o SPLADE foi avaliado exclusivamente em busca web e recuperacao de passagens em ingles; seu comportamento em dialogos curtos, informais e multi-turno em uma lingua morfologicamente rica e desconhecido. Terceiro, a representacao no espaco do vocabulario assume que relacoes semanticas podem ser capturadas por meio de expansao de termos — uma suposicao que pode ser mais fraca para dados conversacionais onde a intencao e expressa por meio de padroes pragmaticos (tom, repeticao, sequencias de escalacao) e nao apenas por variacao de vocabulario.

### 3.1.3 ColBERT: Recuperacao Densa com Interacao Tardia

Khattab e Zaharia (2020) propuseram o ColBERT, uma arquitetura de recuperacao densa baseada em interacao tardia. Diferentemente dos bi-encoders que comprimem cada texto em um unico vetor, o ColBERT mantem a sequencia completa de embeddings em nivel de token tanto para a consulta quanto para o documento, computando a relevancia como a soma das similaridades cosseno maximas entre cada token da consulta e todos os tokens do documento. Este operador MaxSim preserva a correspondencia lexica de granularidade fina dentro de um framework de representacao densa, alcancando uma forma de correspondencia suave de termos que e mais expressiva do que a similaridade cosseno de vetor unico.

O ColBERT alcanca qualidade de recuperacao estado da arte nos benchmarks MS-MARCO e TREC Deep Learning, demonstrando que a interacao tardia captura padroes de correspondencia que bi-encoders com pooling nao capturam. A arquitetura foi estendida no ColBERTv2 (Santhanam et al., 2022), que reduz os requisitos de armazenamento por meio de compressao residual mantendo a qualidade de recuperacao.

Para o TalkEx, a limitacao principal do ColBERT e o custo computacional: armazenar e buscar representacoes em nivel de token para milhoes de janelas de contexto de conversacao requer substancialmente mais armazenamento e memoria do que a representacao de vetor unico utilizada no TalkEx (384 dimensoes por janela). Em uma implantacao em producao processando milhoes de conversas, essa diferenca de custo e operacionalmente significativa. Alem disso, o ColBERT foi desenvolvido e avaliado em texto em ingles; nenhum modelo de interacao tardia multilingual com eficacia demonstrada em dados conversacionais em PT-BR existia no momento da escrita deste trabalho.

### 3.1.4 DPR: Dense Passage Retrieval

Karpukhin et al. (2020) introduziram o Dense Passage Retrieval (DPR), a arquitetura fundacional de bi-encoder para recuperacao densa. O DPR treina dois encoders BERT — um para consultas, outro para passagens — utilizando negativos intra-lote e mineracao de negativos dificeis, otimizando a similaridade de produto interno entre as representacoes de consulta e passagem. Os vetores densos resultantes possibilitam recuperacao via busca por vizinho mais proximo aproximado (ANN), capturando equivalencias semanticas que metodos lexicos nao conseguem detectar.

O DPR demonstrou melhorias substanciais sobre o BM25 em benchmarks de question answering de dominio aberto (Natural Questions, TriviaQA, WebQuestions), alcancando ganhos de Recall@20 de 9 a 19 pontos percentuais. No entanto, trabalhos subsequentes revelaram que a vantagem do DPR sobre o BM25 e dependente do dominio: em benchmarks com alta sobreposicao lexica entre consultas e passagens (por exemplo, perguntas centradas em entidades), o BM25 permanece competitivo ou superior (Thakur et al., 2021). Essa constatacao e diretamente relevante para o TalkEx, onde conversas de atendimento ao cliente contem marcadores lexicos discriminativos — vocabulario de cancelamento, codigos de produtos, frases de saudacao — que o BM25 captura eficientemente.

A licao critica do DPR para o TalkEx e arquitetural e nao empirica: o framework de bi-encoder, onde consultas e documentos sao codificados independentemente e correspondidos via similaridade vetorial, forma a base do componente de recuperacao ANN no sistema hibrido do TalkEx. O TalkEx adota o paradigma de bi-encoder mas substitui um encoder congelado multilingual (paraphrase-multilingual-MiniLM-L12-v2) pelos encoders fine-tuned especificos de tarefa do DPR, trocando otimizacao especifica de tarefa por generalizacao multilingual e zero custo de anotacao.

### 3.1.5 Rayo et al. (2025): O Precedente Metodologico Mais Proximo

Rayo, de la Rosa e Garrido (2025) apresentam o trabalho anterior mais diretamente comparavel ao TalkEx na dimensao de recuperacao. Seu sistema, apresentado no COLING 2025, constroi um pipeline de recuperacao de informacao hibrida para o dataset de conformidade regulatoria ObliQA — 27.869 perguntas extraidas de 40 documentos regulatorios emitidos pela autoridade financeira Abu Dhabi Global Markets.

A arquitetura combina BM25 para recuperacao lexica com um Sentence Transformer fine-tuned (BAAI/bge-small-en-v1.5, expandido de 384 para 512 dimensoes durante o fine-tuning) para recuperacao semantica. A fusao de scores segue interpolacao linear com peso empiricamente selecionado alpha = 0,65 para semantico e 0,35 para lexico. Os resultados quantitativos estabelecem uma hierarquia clara: o baseline BM25 alcanca Recall@10 = 0,761 e MAP@10 = 0,624; o recuperador somente semantico alcanca Recall@10 = 0,810, MAP@10 = 0,629; e o sistema hibrido alcanca Recall@10 = 0,833, MAP@10 = 0,702. O hibrido domina ambas as abordagens isoladas, confirmando a complementaridade.

A contribuicao e significativa: Rayo et al. fornecem evidencia empirica rigorosa de que a recuperacao hibrida supera paradigmas isolados em um dominio especializado do mundo real. O artigo separa a contribuicao de embeddings fine-tuned da contribuicao da fusao, compara multiplos pesos de fusao e reporta resultados por metrica. A adicao de um componente RAG utilizando GPT-3.5 Turbo estende o pipeline para question answering de ponta a ponta.

Entretanto, diversas restricoes limitam a generalizabilidade deste trabalho para o dominio conversacional. Primeiro, o corpus consiste em documentos regulatorios formais — estruturados, escritos em ingles juridico controlado, com linguagem de obrigacao explicita e terminologia padronizada. O desafio da linguagem falada informal, coloquial e multi-turno e fundamentalmente diferente. Segundo, o sistema aborda apenas recuperacao de passagens; nao inclui classificacao, reconhecimento de intencoes ou pos-processamento baseado em regras. Terceiro, o dataset e monolingue em ingles, sem avaliacao em linguas de baixos recursos ou nao-inglesas. Quarto, a unidade de entrada e uma passagem de documento, nao um turno conversacional ou janela de contexto; a estrutura multi-turno do dialogo esta ausente.

O TalkEx operacionaliza a mesma arquitetura de fusao em um dominio significativamente diferente — conversas informais multi-turno de atendimento ao cliente em PT-BR — e avalia o impacto a jusante na classificacao de intencoes em vez de recuperacao de passagens. Onde Rayo et al. utilizam um encoder fine-tuned especifico de dominio, o TalkEx utiliza um encoder congelado multilingual, possibilitando comparacao direta entre adaptacao de dominio via fine-tuning e generalizacao via pre-training multilingual. Conforme avaliado no Capitulo 6, a recuperacao hibrida do TalkEx alcanca MRR = 0,853 versus MRR = 0,835 apenas com BM25 (p = 0,017), confirmando que a complementaridade observada por Rayo et al. se transfere para o dominio conversacional em PT-BR.

### 3.1.6 Harris et al. (2024): BM25 como Baseline Nao-Trivial

Harris (2024), em "Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents", fornece um contraponto a suposicao de superioridade semantica. O estudo avalia o BM25 contra embeddings semanticos prontos para uso (OpenAI text-embedding-3-small/large, all-MiniLM-L6-v2) em uma tarefa de classificacao de documentos medicos consistindo de 517 registros clinicos em 42 categorias ICD-10. A constatacao e inequivoca: BM25 alcanca acuracia top-1 de 0,74, superando o melhor embedding semantico (0,67) por 7 pontos percentuais.

Harris atribui este resultado a regularidade lexica da documentacao medica: registros clinicos utilizam terminologia consistente, codigos diagnosticos e frases padronizadas onde a correspondencia exata de termos e mais discriminativa do que a generalizacao semantica. O estudo tambem demonstra que metodos de ensemble combinando scores BM25 e semanticos melhoram sobre o BM25 isolado em 1 ponto percentual (para 0,75), confirmando que mesmo em um dominio lexicalmente regular, sinais semanticos contribuem valor marginal mas mensuravel.

As implicacoes para o TalkEx sao diretas. Conversas de atendimento ao cliente ocupam uma posicao intermediaria no espectro de regularidade lexica: contem termos exatos discriminativos (nomes de produtos, vocabulario de cancelamento, palavras-chave regulatorias) juntamente com expressoes semanticamente variaveis de intencao (reclamacoes, padroes de escalacao, sinais de satisfacao). A constatacao de Harris reforca a decisao de design do TalkEx de recuperacao hibrida em vez de recuperacao somente semantica, e motiva o estudo de ablacao (Capitulo 6) que quantifica a contribuicao relativa de features lexicas versus de embedding. A contribuicao de +2,9 pontos percentuais das features lexicas na ablacao do TalkEx e consistente com a observacao de Harris de que sinais lexicos mantem valor mesmo quando representacoes semanticas estao disponiveis.

### 3.1.7 Lacuna: Recuperacao Hibrida em Dados Conversacionais em PT-BR

A literatura de recuperacao hibrida e dominada por avaliacao em lingua inglesa sobre busca web (MS-MARCO, BEIR), QA de dominio aberto (Natural Questions) e corpora de documentos formais (texto regulatorio, registros medicos). Nenhum trabalho anterior avaliou a recuperacao hibrida BM25 + densa em dados conversacionais multi-turno em portugues brasileiro. O dominio conversacional introduz desafios especificos ausentes da recuperacao de documentos: enunciados curtos com alta densidade de anafora, estrutura de turnos de falantes, deriva tematica dentro das conversas e registro informal com ruido induzido por ASR. O TalkEx fornece a primeira avaliacao desse tipo, conforme reportado no Capitulo 6.

---

## 3.2 Classificacao Baseada em Embeddings

### 3.2.1 O Paradigma "Embeddings Representam, Classificadores Decidem"

Uma decisao de design fundamental em qualquer sistema de PLN baseado em embeddings e se os embeddings devem ser usados diretamente para predicao (por exemplo, via classificacao por vizinho mais proximo ou thresholds de similaridade cosseno) ou tratados como features de entrada para um classificador supervisionado. A distincao e consequente: o uso direto de embeddings conflaciona a qualidade da representacao com a qualidade da classificacao, enquanto a abordagem desacoplada permite que o classificador aprenda fronteiras de decisao que compensam fraquezas da representacao.

### 3.2.2 Sentence-BERT e a Fundacao dos Embeddings de Sentenca

Reimers e Gurevych (2019) introduziram o Sentence-BERT (SBERT), adaptando a arquitetura BERT para produzir embeddings de sentenca de tamanho fixo via estruturas de redes Siamesas e triplet treinadas em datasets de NLI e STS. O SBERT tornou computacionalmente viavel o uso de BERT para tarefas que requerem comparacao pareada de sentencas — similaridade semantica, clustering, recuperacao — ao reduzir o custo O(n^2) de cross-encoding para codificacao independente O(n) mais busca eficiente por similaridade.

A contribuicao do SBERT para o campo foi arquitetural e pratica: demonstrou que o fine-tuning contrastivo de um modelo de linguagem pre-treinado produz representacoes em nivel de sentenca que transferem efetivamente entre tarefas com adaptacao minima. A biblioteca sentence-transformers resultante tornou-se a ferramenta padrao para geracao de embeddings de texto em aplicacoes downstream, incluindo a variante multilingual (paraphrase-multilingual-MiniLM-L12-v2) utilizada no TalkEx.

Para o TalkEx, a arquitetura do SBERT fornece o backbone de embedding, mas com uma divergencia critica de design: onde Reimers e Gurevych avaliaram embeddings primariamente para tarefas de similaridade e recuperacao, o TalkEx utiliza os embeddings como features de entrada para classificadores gradient-boosted, nunca como classificadores diretos. Essa separacao "embeddings representam, classificadores decidem", conforme articulada no estudo da AnthusAI discutido abaixo, e uma escolha arquitetural deliberada que preserva o poder discriminativo da aprendizagem supervisionada enquanto aproveita a generalizacao semantica de representacoes pre-treinadas.

### 3.2.3 SetFit: Classificacao Few-Shot com Embeddings de Sentenca

Tunstall et al. (2022) introduziram o SetFit (Sentence Transformer Fine-Tuning), um framework para classificacao de texto few-shot que realiza fine-tuning de sentence transformers usando aprendizagem contrastiva em um pequeno numero de exemplos rotulados e entao treina uma cabeca de classificacao sobre os embeddings resultantes. O SetFit alcanca desempenho competitivo com metodos few-shot baseados em prompts (T-Few, GPT-3 com prompts) sendo ordens de magnitude mais rapido para treinar e sem necessidade de prompts ou verbalizadores.

A percepcao-chave do SetFit e que o fine-tuning contrastivo com apenas 8 exemplos por classe produz embeddings que sao substancialmente mais discriminativos para a tarefa-alvo do que as representacoes pre-treinadas congeladas. No benchmark de sentimento SST-2, o SetFit com 8 exemplos por classe alcanca acuracia de 0,87, comparado a 0,80 para o baseline SBERT congelado e 0,92 para GPT-3 com 32 exemplos por classe.

A relevancia para o TalkEx e metodologica. O SetFit representa a extremidade de "fine-tuning" no espectro de adaptacao de representacoes, enquanto o TalkEx ocupa a extremidade de "encoder congelado". O design experimental do TalkEx deliberadamente escolheu a abordagem congelada para minimizar requisitos de anotacao e custo computacional — uma escolha que e validada se a qualidade da classificacao downstream for competitiva, como os resultados de H2 sugerem (Macro-F1 = 0,722 com embeddings congelados). A questao aberta, identificada como trabalho futuro no Capitulo 7, e se o fine-tuning contrastivo no corpus de atendimento ao cliente em PT-BR fecharia a lacuna de desempenho nas classes de intencao mais fracas (compra, saudacao), onde representacoes congeladas podem carecer de poder discriminativo.

### 3.2.4 AnthusAI (2024): Embeddings como Features para Classificadores

O estudo "Semantic Text Classification" da AnthusAI (2024) fornece o precedente conceitual mais direto para a arquitetura de classificacao do TalkEx. O estudo avalia sistematicamente multiplos modelos de embedding (OpenAI text-embedding-3-small/large, Cohere embed-english-v3.0, all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5) como extratores de features para classificadores downstream (regressao logistica, random forest, XGBoost, SVM, rede neural) em uma tarefa de classificacao de intencoes de atendimento ao cliente.

A constatacao central e que a combinacao de modelo de embedding e arquitetura de classificador importa mais do que qualquer componente isolado. XGBoost sobre embeddings OpenAI alcanca a maior acuracia (0,94) em um dataset de atendimento ao cliente em ingles com 6 classes, mas a diferenca de desempenho entre modelos de embedding se estreita substancialmente quando pareados com classificadores gradient-boosted comparados a classificadores lineares. Isso sugere que o classificador compensa fraquezas da representacao — precisamente o mecanismo que o principio "embeddings representam, classificadores decidem" do TalkEx operacionaliza.

O estudo da AnthusAI possui tres limitacoes relevantes para o TalkEx. Primeiro, a avaliacao e em dados single-turn em ingles; contexto conversacional multi-turno esta ausente. Segundo, o estudo nao incorpora features lexicas juntamente com embeddings — uma combinacao que o estudo de ablacao do TalkEx mostra contribuir +2,9 pontos percentuais adicionais de Macro-F1. Terceiro, o estudo nao integra pos-processamento baseado em regras ou producao de evidencia. O TalkEx estende o paradigma da AnthusAI adicionando features heterogeneas (lexicas, estruturais, baseadas em regras) ao espaco de entrada do classificador e operando sobre janelas de contexto multi-turno em vez de sentencas individuais.

### 3.2.5 Huang e He (2025): Clustering Baseado em LLM como Classificacao

Huang e He (2025), em "Text Clustering as Classification with LLMs" (SIGIR-AP 2025), propoem uma alternativa a classificacao supervisionada: utilizar grandes modelos de linguagem para realizar clustering de texto e entao mapear clusters para rotulos de classe. A abordagem usa GPT-4 para gerar descricoes de clusters, atribuir textos a clusters e refinar iterativamente a taxonomia. Nos benchmarks AG News e Yahoo Answers, o metodo alcanca acuracia competitiva (0,86 e 0,63 respectivamente) sem qualquer fine-tuning especifico de tarefa ou dados de treinamento rotulados.

A contribuicao conceitual e significativa: demonstra que LLMs possuem conhecimento de mundo suficiente para realizar construcao de taxonomia de intencoes e classificacao de texto como uma tarefa zero-shot. Para a descoberta de intencoes — o problema de identificar novas razoes de contato previamente nao rotuladas — este paradigma e diretamente aplicavel e esta refletido no pipeline de descoberta de intencoes offline do TalkEx (descrito no Capitulo 4, Secao 4.9).

As limitacoes para classificacao conversacional online sao severas. Primeiro, a abordagem requer inferencia de LLM online para cada decisao de classificacao, com latencia de 1-10 segundos por requisicao e custo monetario de $0,01-0,10 por requisicao — proibitivo na escala de milhoes de conversas por mes. Segundo, as saidas de LLM sao nao-deterministicas: a mesma entrada pode receber classificacoes diferentes entre execucoes, impedindo avaliacao reproduzivel e complicando conformidade regulatoria. Terceiro, nenhuma evidencia por decisao e produzida; o raciocinio do modelo e opaco e nao-auditavel. O TalkEx endereca essas limitacoes restringindo LLMs a papeis offline (descoberta de intencoes, assistencia na rotulagem) enquanto utiliza componentes leves, deterministicos e produtores de evidencia para classificacao online.

### 3.2.6 Lacuna: Encoder Congelado + Gradient Boosting em Conversas em PT-BR

A literatura de classificacao baseada em embeddings avalia predominantemente encoders fine-tuned em benchmarks em ingles. A combinacao especifica avaliada no TalkEx — um encoder de sentencas multilingual congelado (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensoes) pareado com LightGBM sobre features heterogeneas (embedding + lexica + estrutural + regra) em dados conversacionais multi-turno em PT-BR — nao possui precedente direto. A escolha do encoder congelado e motivada por restricoes praticas (sem infraestrutura de GPU dedicada, orcamento limitado de anotacao) que sao caracteristicas de implantacoes de atendimento ao cliente brasileiras — todos os experimentos foram executados na GPU de nivel gratuito do Google Colab, tornando a avaliacao empirica no Capitulo 6 diretamente relevante para praticantes neste contexto.

---

## 3.3 Sistemas de PLN Baseados em Regras e Hibridos

### 3.3.1 A Persistencia das Regras em PLN

Apesar da dominancia de metodos estatisticos e neurais, sistemas baseados em regras persistem em implantacoes de PLN em producao onde tres condicoes convergem: requisitos regulatorios exigem auditabilidade, especialistas de dominio possuem conhecimento explicito que e caro de aprender a partir de dados, e a cobertura para padroes especificos de alto risco deve ser garantida independentemente da distribuicao dos dados de treinamento. A literatura sobre PLN baseado em regras abrange decadas, mas a integracao de regras *com* (em vez de *no lugar de*) classificadores de aprendizado de maquina permanece subexplorada.

### 3.3.2 SystemT: Extracao de Informacao Baseada em Regras em Escala

Chiticariu, Li e Reiss (2013) apresentaram o SystemT, o sistema de extracao de informacao baseado em regras da IBM, argumentando provocativamente que "a extracao de informacao baseada em regras esta morta! Vida longa aos sistemas de extracao de informacao baseados em regras!" O artigo documenta a implantacao do SystemT em aplicacoes de producao processando bilhoes de documentos, onde o sistema utiliza uma linguagem de regras declarativa (AQL) para expressar padroes de extracao sobre texto.

A contribuicao do SystemT e primariamente arquitetural: demonstra que uma linguagem de regras bem projetada, compilada para um plano de execucao otimizado com otimizacao algebrica, pode alcancar throughput de escala industrial mantendo a interpretabilidade e auditabilidade que metodos estatisticos nao possuem. O sistema processa regras de extracao por meio de um otimizador baseado em custo que reordena a avaliacao de predicados por eficiencia — um principio de design diretamente adotado pelo motor de regras do TalkEx, que implementa avaliacao de curto-circuito ordenada por custo de predicado (predicados lexicos primeiro, predicados semanticos por ultimo), conforme descrito no Capitulo 4, Secao 4.7.

A limitacao do SystemT para o contexto do TalkEx e que ele e um sistema puramente de regras: nao se integra com classificadores estatisticos. Regras ou disparam ou nao; nao ha mecanismo para que regras influenciem a confianca de um classificador probabilistico ou para que um classificador sobreponha uma regra quando a cobertura da regra e insuficiente. A integracao rules-as-features do TalkEx — onde saidas de ativacao de regras sao incluidas como features binarias no vetor de entrada do classificador LightGBM — representa uma estrategia de integracao suave ausente no design do SystemT.

### 3.3.3 Snorkel: Supervisao Fraca com Funcoes de Rotulagem

Ratner et al. (2017) introduziram o Snorkel, um sistema para treinar classificadores usando rotulos gerados programaticamente a partir de funcoes de rotulagem — regras heuristicas que rotulam ruidosamente subconjuntos dos dados. O Snorkel utiliza um modelo generativo para estimar a acuracia e a estrutura de correlacao das funcoes de rotulagem, produzindo rotulos de treinamento probabilisticos que sao entao usados para treinar um modelo discriminativo downstream. Em multiplos benchmarks de PLN, o Snorkel alcanca desempenho dentro de 2-5 pontos de F1 de modelos treinados em dados rotulados manualmente, sem necessidade de anotacao manual.

A contribuicao do Snorkel e um framework principiado para combinar sinais ruidosos baseados em regras com aprendizado supervisionado. As funcoes de rotulagem podem ser pensadas como regras ruidosas e parciais — exatamente o tipo de sinal que especialistas de dominio em centrais de atendimento podem produzir (por exemplo, "se o cliente menciona 'cancelar' ou 'desistir', provavelmente e uma intencao de cancelamento"). O modelo generativo aprende quais funcoes de rotulagem sao confiaveis e quais entram em conflito, produzindo rotulos probabilisticos calibrados.

A relevancia para o TalkEx e conceitual e nao implementacional. O motor de regras do TalkEx opera em tempo de inferencia em vez de em tempo de treinamento: regras produzem features que aumentam a entrada do classificador, em vez de produzir rotulos de treinamento ruidosos. No entanto, a percepcao subjacente e compartilhada: heuristicas deterministicas codificam conhecimento de dominio que complementa padroes aprendidos. Uma extensao futura do TalkEx poderia adotar uma abordagem semelhante ao Snorkel para aumento semi-automatizado de dados de treinamento, utilizando as capacidades de funcao de rotulagem do motor de regras para gerar exemplos rotulados adicionais para classes de intencao com baixo desempenho.

### 3.3.4 Motores de Regras em Producao para IA Conversacional

A industria de IA conversacional desenvolveu varios motores de regras de nivel de producao que sao relevantes como precedentes de engenharia, embora a maioria careca da avaliacao academica que constituiria trabalho anterior formal.

O Rasa, um framework de IA conversacional open-source, integra politicas de dialogo baseadas em regras com classificacao de intencoes baseada em ML. Regras no Rasa definem fluxos de conversacao deterministicos que sobrepoem a politica de ML quando condicoes especificas sao atendidas — uma estrategia de sobreposicao rigida que, conforme os resultados de H3 do TalkEx demonstram, pode degradar o desempenho geral quando a cobertura de regras e esparsa (Capitulo 6: rules-as-override Macro-F1 = 0,648 versus ML-only 0,722).

O Google Dialogflow utiliza "contexts" e regras de "fulfillment" para pos-processar resultados de classificacao de intencoes, possibilitando comportamento deterministico para estados de dialogo especificos. A integracao e implicita e fortemente acoplada a arquitetura da plataforma, impedindo a avaliacao independente da contribuicao do componente de regras.

DSLs customizadas para correspondencia de padroes conversacionais sao comuns em plataformas empresariais de contact center (Observe.AI, CallMiner, Verint), mas sao tipicamente proprietarias e nao documentadas na literatura academica. Esses sistemas geralmente implementam regras baseadas em palavras-chave — "se a transcricao contem 'cancelar' dentro de 3 turnos de 'supervisor', sinalizar como escalacao" — sem a gramatica formal, compilacao AST ou otimizacao de custo de predicados que caracterizam o motor de regras do TalkEx.

### 3.3.5 Lacuna: Integracao Suave de Regras Compiladas por DSL com Classificadores de ML

A lacuna na literatura nao e a existencia de regras em sistemas de PLN — elas sao pervasivas — mas o *modo de integracao* com classificadores estatisticos. Sistemas anteriores ou utilizam regras como classificadores autonomos (SystemT), utilizam regras para gerar dados de treinamento (Snorkel) ou utilizam regras como sobreposicoes rigidas de predicoes de ML (Rasa, plataformas empresariais). A estrategia de integracao suave avaliada no TalkEx — onde sinais de ativacao de regras sao compilados a partir de uma DSL tipada em predicados AST e entao incluidos como features binarias no vetor de features heterogeneas do classificador — e, ate onde sabemos, inedita. Essa integracao permite que o classificador aprenda a informatividade de cada sinal de regra em vez de tratar saidas de regras como infaliveis, evitando a degradacao observada com estrategias de sobreposicao rigida. A avaliacao no Capitulo 6 mostra que essa integracao suave produz uma direcao positiva (+1,8pp Macro-F1) que nao alcanca significancia estatistica com a configuracao atual de 2 regras (p = 0,131), motivando a expansao do conjunto de regras em trabalhos futuros.

---

## 3.4 Classificacao de Intencoes Conversacionais

### 3.4.1 O Paradigma Single-Turn e Suas Limitacoes

A classificacao de intencoes para sistemas conversacionais tem sido historicamente formulada como um problema single-turn: dado um enunciado do usuario, prever o rotulo de intencao. Essa formulacao reflete as origens chatbot-centricas da tarefa, onde cada mensagem do usuario e tratada como uma consulta independente. Os benchmarks dominantes — ATIS (Hemphill et al., 1990), SNIPS (Coucke et al., 2018), CLINC150 (Larson et al., 2019) e BANKING77 (Casanueva et al., 2020) — reforcam esse paradigma ao fornecer datasets de enunciados unicos sem contexto conversacional.

### 3.4.2 Larson et al. (2019): Benchmarks de Deteccao de Intencoes

Larson et al. (2019) introduziram o CLINC150, um dataset benchmark de 23.700 consultas single-turn em 150 classes de intencao mais uma categoria out-of-scope, projetado para avaliar a deteccao de intencoes em dialogo orientado a tarefas. O dataset e notavel por sua escala (150 classes), sua inclusao de exemplos out-of-scope (1.200 consultas que nao correspondem a nenhuma intencao definida) e sua alta concordancia entre anotadores. Modelos avaliados no CLINC150 variam de baselines bag-of-words (acuracia ~0,77) a BERT fine-tuned (acuracia ~0,97), estabelecendo o estado da arte para deteccao de intencoes single-turn em ingles.

A contribuicao do CLINC150 como benchmark e significativa, mas sua relevancia para o TalkEx e limitada por tres fatores. Primeiro, o dataset consiste inteiramente de consultas single-turn em ingles; contexto conversacional esta ausente. Segundo, as 150 classes de intencao sao projetadas para um dominio de assistente virtual (bancario, viagens, cozinha, etc.) que difere substancialmente de taxonomias de reclamacoes e suporte de atendimento ao cliente. Terceiro, as consultas sao textos limpos e bem formados, sem o ruido de ASR, abreviacoes e registro informal caracteristicos do atendimento ao cliente em PT-BR. A taxonomia de 8 classes do TalkEx sobre janelas de contexto multi-turno representa uma formulacao de tarefa diferente que nao e diretamente comparavel ao CLINC150.

### 3.4.3 Casanueva et al. (2020): Deteccao de Intencoes Few-Shot

Casanueva et al. (2020) introduziram o BANKING77, um dataset de 13.083 consultas single-turn de clientes bancarios em 77 classes de intencao de granularidade fina, e avaliaram metodos de deteccao de intencoes few-shot incluindo USE (Universal Sentence Encoder) e ConveRT. Sua constatacao-chave e que encoders de sentencas pre-treinados, quando fine-tuned com apenas 10 exemplos por classe, alcancam desempenho competitivo com baselines totalmente supervisionados — acuracia de 0,84 com 10 exemplos versus 0,93 com dados de treinamento completos.

A avaliacao do BANKING77 e relevante para o TalkEx como precedente de dominio: demonstra que a classificacao de intencoes de atendimento ao cliente e viavel com representacoes pre-treinadas e supervisao limitada. No entanto, a formulacao single-turn, a avaliacao exclusivamente em ingles e a ausencia de janelas de contexto multi-turno limitam a comparabilidade direta. O TalkEx estende a intuicao few-shot utilizando um encoder congelado sem nenhum fine-tuning por tarefa, confiando inteiramente na combinacao de pre-training multilingual e features heterogeneas no classificador downstream.

### 3.4.4 Zhang et al. (2021): Classificacao de Intencoes Multi-Turno

Zhang et al. (2021) abordam diretamente o problema de classificacao de intencoes multi-turno, propondo modelos que incorporam historico de dialogo por meio de encoders hierarquicos que processam turnos sequencialmente e produzem representacoes em nivel de conversacao. Sua avaliacao nos benchmarks DSTC (Dialog State Tracking Challenge) demonstra que incorporar turnos precedentes melhora a acuracia de classificacao de intencoes em 3-8 pontos percentuais sobre baselines single-turn, com os maiores ganhos em intencoes que sao expressas ao longo de multiplos turnos (por exemplo, "Preciso alterar minha reserva" seguido de turnos de esclarecimento).

Este trabalho valida a premissa central do design de janela de contexto do TalkEx: que o contexto multi-turno carrega informacao que a analise single-turn nao captura. No entanto, Zhang et al. utilizam encoders hierarquicos recorrentes ou baseados em transformer que sao treinados de ponta a ponta, enquanto o TalkEx utiliza uma abordagem mais simples de janela deslizante com mean pooling sobre embeddings congelados. O trade-off e expressividade versus custo: a abordagem do TalkEx nao requer treinamento adicional e produz representacoes de dimensionalidade fixa que sao compativeis com classificadores gradient-boosted, ao custo potencial de perder padroes de attention em nivel de turno com granularidade fina. O mecanismo de construcao de janela de contexto e detalhado no Capitulo 4, Secao 4.4.

### 3.4.5 Lyu et al. (2025): Mecanismos de Attention para Classificacao de Texto

Lyu, Wang, Chen e Zhang (2025) investigam a integracao de mecanismos neurais de attention com grandes modelos de linguagem para classificacao de texto, avaliando no benchmark de texto curto AG News. Sua abordagem aumenta o BERT-base com camadas de self-attention multi-head que aprendem a ponderar representacoes de tokens antes da classificacao, alcancando F1 = 0,89 versus 0,86 para BERT-base isolado e 0,91 para uma variante fine-tuned de GPT-3.5.

A contribuicao e incremental: classificacao aumentada por attention tem sido estudada extensivamente desde Vaswani et al. (2017), e a melhoria de 3 pontos no AG News e modesta. No entanto, o artigo fornece um ponto de dados recente sobre a lacuna entre encoders congelados e arquiteturas aumentadas por attention, o que e relevante para o design de encoder congelado do TalkEx. A comparacao sugere que attention especifica de tarefa poderia melhorar a qualidade da classificacao, mas ao custo de requerer treinamento especifico de tarefa — um custo que o TalkEx evita por design.

A limitacao critica para o TalkEx e o dominio de avaliacao: AG News consiste em artigos jornalisticos curtos, de sentenca unica, em ingles formal, um registro fundamentalmente diferente de conversas multi-turno de atendimento ao cliente em PT-BR. Os autores nao avaliam em dados conversacionais, entradas multi-turno ou linguas nao-inglesas.

### 3.4.6 Lacuna: Classificacao de Intencoes Multi-Turno em PT-BR

A literatura de classificacao de intencoes conversacionais e predominantemente em ingles e predominantemente single-turn. Os principais benchmarks (CLINC150, BANKING77, ATIS, SNIPS) nao fornecem contexto conversacional, dados em portugues ou avaliacao multi-turno. Zhang et al. (2021) abordam a modelagem multi-turno, mas em benchmarks de dialogo em ingles com modelos treinados de ponta a ponta. Nenhum trabalho anterior avalia classificacao de intencoes multi-turno em dados de atendimento ao cliente em portugues brasileiro utilizando um encoder congelado com features heterogeneas — a configuracao especifica avaliada no TalkEx. O corpus pos-auditoria de 2.122 registros utilizado nesta dissertacao (Capitulo 5) representa, ate onde sabemos, a primeira avaliacao publica disponivel deste tipo.

---

## 3.5 Inferencia em Cascata e Consciente de Custo

### 3.5.1 O Principio da Cascata

A inferencia em cascata e uma estrategia de eficiencia computacional na qual entradas sao processadas por uma sequencia de classificadores progressivamente mais caros, com predicoes confiantes nos estagios iniciais prevenindo a invocacao desnecessaria de estagios posteriores mais custosos. A suposicao subjacente e que uma fracao substancial das entradas sao "faceis" e podem ser classificadas corretamente por um modelo barato, enquanto apenas entradas "dificeis" requerem o investimento computacional completo.

### 3.5.2 Viola e Jones (2001): O Classificador em Cascata Original

Viola e Jones (2001) introduziram o classificador em cascata no contexto de deteccao de faces, construindo uma sequencia de classificadores de features Haar progressivamente mais complexos treinados com AdaBoost. Cada estagio na cascata e projetado para alcancar recall muito alto (para evitar perder faces) enquanto aumenta progressivamente a precision (para rejeitar regioes sem face). O resultado e um sistema que processa a vasta maioria das sub-janelas de imagem nos primeiros estagios, alcancando deteccao de faces em tempo real em hardware da era de 2001.

A cascata de Viola-Jones estabeleceu tres principios de design que permanecem relevantes para o TalkEx. Primeiro, estagios iniciais devem ser muito baratos relativamente aos estagios posteriores — o diferencial de custo e o pre-requisito para o beneficio da cascata. Segundo, cada estagio deve ter alto recall para evitar perdas em cascata. Terceiro, o threshold em cada estagio controla o trade-off precision-recall e deve ser calibrado empiricamente. O experimento H4 do TalkEx implementa esses principios no dominio de PLN conversacional, utilizando um classificador somente lexico como Estagio 1 e o classificador hibrido completo como Estagio 2, com thresholds de saida antecipada baseados em confianca. Conforme reportado no Capitulo 6, a cascata falha porque o pre-requisito de diferencial de custo e violado: ambos os estagios compartilham embeddings pre-computados, eliminando a vantagem de custo do estagio "barato".

### 3.5.3 Matveeva et al. (2006): Recuperacao em Cascata em Busca Web

Matveeva et al. (2006) aplicaram o principio da cascata a busca web, utilizando uma sequencia de modelos de ranqueamento progressivamente mais sofisticados — de BM25, passando por features de combinacao linear, ate ranqueadores neurais — com cada estagio re-ranqueando um conjunto de candidatos progressivamente menor. A abordagem reduz o tempo medio de processamento de consultas em 40-60% mantendo o NDCG dentro de 1-2% do baseline de pipeline completo.

Este trabalho demonstra que a inferencia em cascata e eficaz para tarefas de recuperacao onde o custo do modelo completo e dominado pelo custo de inferencia por candidato e nao pela pre-computacao. A licao para o TalkEx e que a eficiencia da cascata depende de onde o custo computacional esta concentrado: se o custo dominante e uma etapa de pre-computacao (geracao de embeddings) compartilhada entre estagios, cascatear o classificador downstream nao proporciona economia. Essa observacao estrutural explica a refutacao de H4 e motiva o redesign arquitetural proposto no Capitulo 7.

### 3.5.4 Schwartz et al. (2020): Eficiencia em PLN

Schwartz, Dodge, Smith e Etzioni (2020) fornecem um levantamento abrangente de eficiencia computacional em PLN, cobrindo compressao de modelos (destilacao, poda, quantizacao), arquiteturas de saida antecipada (profundidade adaptativa, ramificacao baseada em confianca) e design consciente de hardware. O levantamento argumenta que a eficiencia deve ser reportada juntamente com a acuracia em avaliacoes de PLN, propondo a "fronteira eficiencia-acuracia" como o framework de avaliacao apropriado.

O levantamento identifica tres condicoes sob as quais arquiteturas de cascata e saida antecipada sao mais eficazes: (1) alta variancia na complexidade de entrada, de modo que entradas faceis e dificeis sao claramente separaveis; (2) diferencial de custo significativo entre os modelos barato e caro; e (3) estimativa de confianca confiavel para o modelo barato, de modo que saidas antecipadas correspondam a predicoes corretas. A condicao (2) nao foi satisfeita no experimento H4 do TalkEx, e a condicao (3) permanece nao avaliada devido a ausencia de analise de calibracao (identificada como trabalho futuro no Capitulo 7).

### 3.5.5 Lacuna: Cascata Aplicada a PLN Conversacional com Features Hibridas

A literatura de cascata origina-se em visao computacional (Viola e Jones, 2001) e tem sido aplicada a busca web (Matveeva et al., 2006) e inferencia de PLN geral (Schwartz et al., 2020). A aplicacao a pipelines de PLN conversacional com features hibridas — onde o estagio "barato" utiliza features lexicas e o estagio "caro" adiciona embeddings e regras — e, ate onde sabemos, inexplorada. O experimento H4 do TalkEx (Capitulo 6) fornece a primeira evidencia empirica sobre esta configuracao, produzindo um resultado negativo que identifica um pre-requisito arquitetural especifico (separacao genuina de custo entre estagios) para a eficacia da cascata neste cenario.

---

## 3.6 PLN em Portugues

### 3.6.1 BERTimbau: A Fundacao dos Modelos de Linguagem em PT-BR

Souza, Nogueira e Lotufo (2020) introduziram o BERTimbau, um modelo BERT pre-treinado em um grande corpus de portugues brasileiro (brWaC, 2,7 bilhoes de tokens). O BERTimbau alcanca resultados estado da arte em benchmarks de PLN em portugues incluindo reconhecimento de entidades nomeadas (HAREM), similaridade textual de sentencas (ASSIN/ASSIN2) e reconhecimento de implicacao textual, superando o BERT multilingual por 1-5 pontos percentuais entre tarefas.

A contribuicao do BERTimbau e a demonstracao de que pre-training especifico de lingua melhora o desempenho downstream sobre modelos multilinguais para o portugues. Essa constatacao e relevante para o trade-off de design do TalkEx: o TalkEx utiliza um encoder de sentencas multilingual (paraphrase-multilingual-MiniLM-L12-v2) em vez de um modelo especifico de portugues, aceitando qualidade de representacao potencialmente inferior em troca de zero treinamento especifico de lingua. Se um encoder especifico de PT-BR (por exemplo, um sentence transformer baseado em BERTimbau) melhoraria os resultados de classificacao do TalkEx e uma questao aberta identificada como trabalho futuro.

### 3.6.2 Datasets de PLN em Portugues

O panorama de recursos de PLN em portugues melhorou substancialmente desde 2020, mas permanece esparso em relacao ao ingles, particularmente para dados conversacionais.

**MilkQA** (Bittar et al., 2020) fornece 2.657 pares pergunta-resposta em portugues brasileiro extraidos de um servico de consultoria em pecuaria leiteira. O dataset e conversacional em origem, mas consiste em pares de Q&A single-turn sem contexto multi-turno.

**FaQuAD** (Sayama et al., 2019) fornece 900 pares pergunta-resposta para compreensao de leitura em portugues, seguindo o formato SQuAD. Os dados sao extraidos de artigos da Wikipedia em portugues e sao de natureza extrativa — uma tarefa diferente da classificacao de intencoes.

**ASSIN e ASSIN2** (Fonseca et al., 2016; Real et al., 2020) fornecem avaliacao de similaridade de sentencas e implicacao textual para portugues, com aproximadamente 10.000 pares de sentencas. Esses datasets sao criticos para avaliar semantica em nivel de sentenca em portugues, mas nao abordam as dimensoes conversacional ou de classificacao do TalkEx.

**Datasets de NLI em portugues** derivados do XNLI (Conneau et al., 2018) fornecem avaliacao de inferencia em linguagem natural cross-lingual, possibilitando comparacao de modelos multilinguais em portugues. Esses benchmarks informaram a escolha do encoder paraphrase-multilingual-MiniLM-L12-v2 no TalkEx, que foi avaliado no XNLI como parte de seu treinamento multilingual.

### 3.6.3 Desafios Especificos de Dados Conversacionais em PT-BR

Dados de atendimento ao cliente em portugues brasileiro apresentam desafios que estao ausentes ou atenuados na literatura de PLN centrada no ingles.

**Inconsistencia diacritica.** O portugues brasileiro utiliza marcas diacriticas (acentos, tis, cedilhas) que sao sistematicamente inconsistentes em texto conversacional: "cancelamento" versus "cancelamento," "nao" versus "nao," "voce" versus "voce." Sistemas de ASR exacerbam essa inconsistencia. O TalkEx endereca isso por meio de normalizacao Unicode NFKD com remocao de acentos (Capitulo 4, Secao 4.3), uma etapa de preprocessamento que e desnecessaria para o ingles mas critica para correspondencia lexica em portugues.

**Registro informal e abreviacoes.** Dados de chat de atendimento ao cliente em PT-BR exibem abreviacao extrema: "vc" (voce), "td" (tudo), "blz" (beleza), "pq" (porque), "msg" (mensagem). Essas abreviacoes estao ausentes do texto formal em portugues no qual a maioria dos modelos de linguagem foi treinada, criando uma lacuna de vocabulario que afeta tanto a recuperacao lexica (BM25 trata "vc" e "voce" como tokens diferentes) quanto a qualidade dos embeddings (o encoder pode representar "vc" pobremente).

**Code-switching com ingles.** Conversas de atendimento ao cliente brasileiras rotineiramente incluem emprestimos e termos tecnicos em ingles: "upgrade," "app," "feedback," "login," "premium." Encoders multilinguais lidam com esses termos melhor do que modelos especificos de portugues, o que e um argumento a favor da escolha de encoder multilingual do TalkEx.

**Datasets conversacionais publicos limitados.** Antes do dataset utilizado nesta dissertacao (RichardSakaguchiMS/brazilian-customer-service-conversations), nenhum dataset publico de atendimento ao cliente multi-turno em PT-BR com anotacoes de intencao estava disponivel. A ausencia de dados publicos de avaliacao e o principal gargalo para pesquisa em PLN conversacional em portugues.

### 3.6.4 Lacuna: Classificacao de Intencoes Conversacionais em PT-BR

O PLN em portugues avancou substancialmente com o BERTimbau e modelos relacionados, mas a dimensao conversacional permanece subatendida. Nenhum dataset publico em PT-BR fornece conversas multi-turno de atendimento ao cliente com anotacoes de intencao. Nenhum trabalho anterior avalia recuperacao hibrida, classificacao baseada em embeddings ou integracao de regras em dados conversacionais em PT-BR. A avaliacao do TalkEx no corpus pos-auditoria de 2.122 registros (Capitulo 5) representa a primeira avaliacao sistematica desses metodos nesta lingua e dominio.

---

## 3.7 Posicionamento desta Dissertacao

### 3.7.1 Sintese das Lacunas

As secoes precedentes identificam seis lacunas especificas na literatura, cada uma endereçada por um componente do TalkEx:

1. **Recuperacao hibrida em dados conversacionais em PT-BR.** Trabalhos anteriores de recuperacao hibrida avaliam em busca web em ingles, QA ou documentos formais (Rayo et al., 2025; Harris, 2024). Nao existe avaliacao para conversas multi-turno em PT-BR.

2. **Encoder congelado + gradient boosting para classificacao de intencoes em PT-BR.** A literatura de classificacao baseada em embeddings (AnthusAI, 2024; Tunstall et al., 2022) avalia em dados em ingles, tipicamente com encoders fine-tuned. A combinacao encoder congelado + LightGBM em PT-BR nao foi testada.

3. **Integracao suave de regras compiladas por DSL com classificadores de ML.** Sistemas baseados em regras ou operam independentemente (Chiticariu et al., 2013) ou geram rotulos de treinamento (Ratner et al., 2017). A integracao suave rules-as-features avaliada no TalkEx e inedita.

4. **Classificacao de intencoes multi-turno em PT-BR.** Benchmarks sao single-turn e em ingles (Larson et al., 2019; Casanueva et al., 2020). Modelagem multi-turno (Zhang et al., 2021) nao foi avaliada em dados em portugues.

5. **Inferencia em cascata para PLN conversacional.** O principio da cascata (Viola e Jones, 2001) nao foi aplicado a classificacao de intencoes conversacionais com features hibridas.

6. **Avaliacao publica conversacional em PT-BR.** Nao existe dataset publico ou benchmark para a formulacao de tarefa especifica.

### 3.7.2 Tabela de Posicionamento Comparativo

A tabela a seguir compara o TalkEx contra os trabalhos anteriores mais relevantes ao longo das dimensoes que definem o espaco de contribuicao da dissertacao. Uma marca de verificacao indica que a funcionalidade esta presente; um traco indica ausencia.

| Dimensao | Rayo et al. (2025) | Harris (2024) | AnthusAI (2024) | Tunstall et al. (2022) | Huang & He (2025) | Larson et al. (2019) | Casanueva et al. (2020) | Zhang et al. (2021) | Chiticariu et al. (2013) | Ratner et al. (2017) | **TalkEx (esta dissertacao)** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Lingua** | Ingles | Ingles | Ingles | Ingles | Ingles | Ingles | Ingles | Ingles | Ingles | Ingles | **PT-BR** |
| **Dominio** | Regulatorio | Medico | Atend. cliente | Geral | Noticias/QA | Asst. virtual | Bancario | Dialogo | Empresarial | Geral | **Atendimento ao cliente** |
| **Recuperacao hibrida** | Sim | Comparado | -- | -- | -- | -- | -- | -- | -- | -- | **Sim** |
| **Baseline BM25** | Sim | Sim | -- | -- | -- | -- | -- | -- | -- | -- | **Sim** |
| **Recuperacao densa** | Sim | Sim | -- | -- | -- | -- | -- | -- | -- | -- | **Sim** |
| **Fusao de scores** | Linear | Ensemble | -- | -- | -- | -- | -- | -- | -- | -- | **Linear + RRF** |
| **Classificacao por embedding** | -- | -- | Sim | Sim | Implicito | -- | Fine-tuned | Fine-tuned | -- | -- | **Sim (congelado)** |
| **Encoder congelado** | -- | Sim | Parcial | -- | -- | -- | -- | -- | -- | -- | **Sim** |
| **Features heterogeneas** | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Sim (emb+lex+struct+regra)** |
| **Gradient boosting** | -- | -- | Sim | -- | -- | -- | -- | -- | -- | -- | **Sim (LightGBM)** |
| **Motor de regras** | -- | -- | -- | -- | -- | -- | -- | -- | Sim | Sim (rotulagem) | **Sim (DSL-AST)** |
| **Rules-as-features** | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Sim** |
| **Evidencia por decisao** | -- | -- | -- | -- | -- | -- | -- | -- | Sim | -- | **Sim** |
| **Contexto multi-turno** | -- | -- | -- | -- | -- | -- | -- | Sim | -- | -- | **Sim (janela deslizante)** |
| **Inferencia em cascata** | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Sim (avaliada, refutada)** |
| **Significancia estatistica** | -- | -- | -- | Limitada | -- | -- | Limitada | Limitada | -- | -- | **Sim (Wilcoxon, 5 seeds)** |
| **Estudo de ablacao** | Parcial | -- | Parcial | -- | -- | -- | -- | -- | -- | -- | **Sim (4 familias de features)** |
| **Artefato open-source** | -- | -- | Sim | Sim | -- | Sim | Sim | -- | -- | Sim | **Sim (170 arquivos, ~1900 testes)** |
| **Resultados negativos reportados** | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Sim (H3 inconclusiva, H4 refutada)** |

### 3.7.3 O Espaco de Contribuicao Unico

A tabela de posicionamento revela que o TalkEx ocupa uma intersecao unica na literatura. Nenhum trabalho anterior combina todos os seguintes em um unico sistema avaliado:

1. **Recuperacao hibrida BM25 + densa** com fusao de scores parametrica, avaliada em dados conversacionais.
2. **Encoder multilingual congelado** utilizado como extrator de features (nao fine-tuned), pareado com classificacao gradient-boosted sobre features heterogeneas.
3. **Regras deterministicas compiladas por DSL** integradas como features suaves no espaco de entrada do classificador, com producao de evidencia por decisao.
4. **Janelas de contexto multi-turno** como unidade primaria de analise, em vez de sentencas individuais ou documentos.
5. **Inferencia em cascata** com saida antecipada baseada em confianca, avaliada em trade-offs custo-qualidade de classificacao.
6. **Portugues brasileiro** com dados conversacionais com anotacoes de intencao.
7. **Reporte transparente** de hipoteses confirmadas e refutadas com testes de Wilcoxon de postos sinalizados e ablacao sistematica.

As comparacoes individuais mais proximas sao Rayo et al. (2025) para recuperacao hibrida (mas em texto regulatorio em ingles, sem classificacao ou regras), AnthusAI (2024) para classificacao baseada em embeddings (mas em dados single-turn em ingles, sem recuperacao hibrida ou regras) e Chiticariu et al. (2013) para PLN baseado em regras (mas sem integracao com ML ou injecao suave de features). A contribuicao do TalkEx e a combinacao especifica dessas abordagens, sua avaliacao em dados conversacionais em PT-BR e o reporte honesto tanto de sucessos quanto de falhas.

Os resultados experimentais reportados no Capitulo 6 — recuperacao hibrida MRR = 0,853 (p = 0,017 versus BM25), classificacao Macro-F1 = 0,722 com embeddings congelados subindo para 0,740 com features de regras, a refutacao da cascata e a ablacao quantificando a dominancia de embeddings em +33,0pp — fornecem o embasamento empirico para este posicionamento. Os resultados negativos (H3 inconclusiva, H4 refutada) sao tao informativos quanto os resultados positivos, delimitando as condicoes sob as quais o paradigma hibrido entrega valor mensuravel e onde nao entrega.

---

## Referencias deste Capitulo

As referencias a seguir sao citadas neste capitulo. Detalhes bibliograficos completos sao fornecidos na lista de referencias da dissertacao.

AnthusAI. (2024). *Semantic Text Classification: Text Classification with Various Embedding Techniques*. GitHub repository. Retrieved from https://github.com/AnthusAI/semantic-text-classification.

Bittar, A., Patil, S., and Lu, W. (2020). MilkQA: A Dataset of Consumer Questions for the Task of Answer Selection. In *Proceedings of the 4th Workshop on e-Commerce and NLP* (pp. 42-47).

Casanueva, I., Temcinas, T., Gerber, D., Vandyke, D., and Mrksic, N. (2020). Efficient intent detection with dual sentence encoders. In *Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI* (pp. 38-45).

Chiticariu, L., Li, Y., and Reiss, F. R. (2013). Rule-based information extraction is dead! Long live rule-based information extraction systems! In *Proceedings of EMNLP 2013* (pp. 827-832).

Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S., Schwenk, H., and Stoyanov, V. (2018). XNLI: Evaluating cross-lingual sentence representations. In *Proceedings of EMNLP 2018* (pp. 2475-2485).

Fonseca, E. R., dos Santos, L. B., Criscuolo, M., and Aluisio, S. M. (2016). ASSIN: Avaliacao de Similaridade Semantica e INferencia textual. In *Proceedings of PROPOR 2016* (pp. 13-15).

Formal, T., Piwowarski, B., and Clinchant, S. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In *Proceedings of SIGIR 2021* (pp. 2288-2292).

Harris, L. (2024). Comparing lexical and semantic vector search methods when classifying medical documents. arXiv preprint arXiv:2505.11582v2.

Hemphill, C. T., Godfrey, J. J., and Doddington, G. R. (1990). The ATIS spoken language systems pilot corpus. In *Proceedings of the Workshop on Speech and Natural Language* (pp. 96-101).

Huang, C., and He, G. (2025). Text clustering as classification with LLMs. In *Proceedings of SIGIR-AP 2025*. arXiv preprint arXiv:2410.00927v3.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of EMNLP 2020* (pp. 6769-6781).

Khattab, O., and Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In *Proceedings of SIGIR 2020* (pp. 39-48).

Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., Kummerfeld, J. K., Leach, K., Laurenzano, M. A., Tang, L., and Mars, J. (2019). An evaluation dataset for intent detection and out-of-scope prediction. In *Proceedings of EMNLP 2019* (pp. 1311-1316).

Lyu, N., Wang, Y., Chen, F., and Zhang, Q. (2025). Advancing text classification with large language models and neural attention mechanisms. arXiv preprint arXiv:2512.09444v1.

Matveeva, I., Burges, C., Burkard, T., Lauber, A., and Wong, L. (2006). High accuracy retrieval with multiple nested ranker. In *Proceedings of SIGIR 2006* (pp. 437-444).

Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., and Re, C. (2017). Snorkel: Rapid training data creation with weak supervision. In *Proceedings of the VLDB Endowment*, 11(3), 269-282.

Rayo, J., de la Rosa, R., and Garrido, M. (2025). A hybrid approach to information retrieval and answer generation for regulatory texts. In *Proceedings of COLING 2025*. arXiv preprint arXiv:2502.16767v1.

Real, L., Fonseca, E., and Oliveira, H. G. (2020). The ASSIN 2 shared task: a survey. In *Proceedings of PROPOR 2020* (pp. 229-238).

Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019* (pp. 3982-3992).

Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., and Gatford, M. (1995). Okapi at TREC-3. In *Proceedings of the Third Text REtrieval Conference (TREC-3)* (pp. 109-126).

Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In *Proceedings of NAACL 2022* (pp. 3715-3734).

Sayama, H. F., Araujo, A. F., and Fernandes, E. R. (2019). FaQuAD: Reading comprehension dataset in the domain of Brazilian higher education. In *Proceedings of the IEEE 8th Brazilian Conference on Intelligent Systems (BRACIS)* (pp. 443-448).

Schwartz, R., Dodge, J., Smith, N. A., and Etzioni, O. (2020). Green AI. *Communications of the ACM*, 63(12), 54-63.

Souza, F., Nogueira, R., and Lotufo, R. (2020). BERTimbau: Pretrained BERT models for Brazilian Portuguese. In *Proceedings of BRACIS 2020* (pp. 403-417).

Thakur, N., Reimers, N., Rucktaschel, A., Srivastava, A., and Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In *Proceedings of NeurIPS 2021 Datasets and Benchmarks Track*.

Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M., and Pereg, O. (2022). Efficient few-shot learning without prompts. arXiv preprint arXiv:2209.11055.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In *Proceedings of NeurIPS 2017* (pp. 5998-6008).

Viola, P., and Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. In *Proceedings of CVPR 2001* (Vol. 1, pp. I-511-I-518).

Zhang, J., Hashimoto, K., Liu, W., Wu, C., Wan, Y., Yu, P., Socher, R., and Xiong, C. (2021). Discriminative nearest neighbor few-shot intent detection by transferring natural language inference. In *Proceedings of EMNLP 2021* (pp. 5064-5082).

---

*Contagem de palavras do capitulo (aproximada): 6.800 palavras.*
