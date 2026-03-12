# Capítulo 3 — Trabalhos Relacionados

Este capítulo posiciona a presente pesquisa frente ao estado da arte em cinco eixos que a dissertação articula de forma integrada: retrieval híbrido, classificação de texto com embeddings e mecanismos de atenção, sistemas de regras e DSLs para NLP, inteligência conversacional em contact centers e inferência em cascata. Para cada trabalho, apresentamos a contribuição principal, as limitações que delimitam seu escopo e a relação explícita com as hipóteses e decisões arquiteturais do TalkEx. O capítulo encerra com uma tabela comparativa de 15 trabalhos sobre 5 dimensões, demonstrando que a posição ocupada por esta dissertação — a integração dos três paradigmas (lexical, semântico e regras determinísticas) sobre representações conversacionais multi-nível — permanece inexplorada na literatura.

---

## 3.1 Retrieval Híbrido em Domínios Especializados

A combinação de busca lexical e busca semântica em um único pipeline de retrieval é uma direção de pesquisa que ganhou força com a demonstração empírica de que nenhum dos paradigmas, isoladamente, domina o outro em todos os cenários. Dois trabalhos recentes são particularmente relevantes para esta dissertação por investigarem a questão em domínios especializados com vocabulário técnico e estrutura textual própria.

### 3.1.1 Rayo et al. (2025) — Retrieval híbrido para textos regulatórios

Rayo, de la Rosa e Garrido [Rayo et al., 2025] propuseram uma abordagem híbrida para information retrieval e geração de respostas em textos regulatórios, apresentada no COLING 2025. O sistema combina BM25 com embeddings densos gerados por BGE fine-tuned, utilizando fusão linear ponderada com parâmetro alpha = 0.65 (65% peso semântico, 35% lexical). Os autores avaliaram o pipeline sobre 27.869 questões regulatórias e reportaram resultados que sustentam a superioridade do híbrido:

| Sistema | Recall@10 | MAP@10 |
|---------|-----------|--------|
| BM25 baseline | 0.7611 | 0.6237 |
| BM25 customizado | 0.7791 | 0.6415 |
| Semântico puro | 0.8103 | 0.6286 |
| **Híbrido** | **0.8333** | **0.7016** |

Três aspectos deste trabalho merecem destaque analítico. Primeiro, embora o semântico puro supere o lexical em Recall@10 (0.81 vs 0.78), o híbrido supera ambos tanto em recall quanto em precisão de ranking (MAP@10: 0.70 vs 0.63 do semântico). Isso sugere que o componente lexical contribui decisivamente para a qualidade do topo do ranking — termos exatos empurram documentos relevantes para posições mais altas. Segundo, o fine-tuning do modelo BGE produziu ganho significativo (Recall@10: 0.81 vs 0.70 do modelo base), indicando que embeddings off-the-shelf podem ser insuficientes em domínios especializados. Terceiro, o pipeline de normalização textual descrito pelos autores (expansão de contrações, lowercase, remoção de caracteres não-alfanuméricos, stemming, bigramas) é notavelmente similar ao que implementamos no TalkEx, o que valida nossa abordagem de pré-processamento.

**Limitações relevantes para esta dissertação.** O domínio regulatório possui características fundamentalmente distintas do domínio conversacional: textos longos, formais, com vocabulário controlado e estrutura rígida. O trabalho não inclui classificação multi-label, não emprega regras determinísticas e não opera sobre dados conversacionais multi-turn. Além disso, os autores não investigaram o impacto de representações em múltiplos níveis de granularidade — o retrieval opera sobre documentos inteiros, sem noção de turnos ou janelas de contexto.

**Relação com o TalkEx.** Este trabalho fornece evidência direta para a hipótese H1 (superioridade do retrieval híbrido sobre abordagens isoladas). A presente dissertação estende a investigação para o domínio conversacional, onde o texto é ruidoso (transcrição ASR, coloquialismos, gírias), informal e dependente de contexto multi-turn — condições nas quais a complementaridade entre lexical e semântico pode manifestar-se de forma distinta.

### 3.1.2 Gokhan et al. (2024) — BM25 como baseline robusto para textos regulatórios

Gokhan et al. [Gokhan et al., 2024] estabeleceram um baseline lexical robusto utilizando BM25 para o desafio RegNLP de textos regulatórios. O trabalho é relevante não tanto por suas contribuições técnicas inovadoras, mas por demonstrar a força persistente do BM25 como baseline competitivo — resultado consistente com observações de surveys recentes sobre retrieval neural [Lin et al., 2023] que documentam que BM25 continua competitivo mesmo diante de modelos densos sofisticados.

**Limitações.** O trabalho não incorpora componente semântico, limitando-se ao paradigma lexical. Não há classificação supervisionada nem avaliação em dados conversacionais.

**Relação com o TalkEx.** Reforça a decisão arquitetural de manter BM25 como baseline obrigatório em todas as avaliações. No TalkEx, implementamos BM25 com normalização accent-aware para PT-BR, estendendo a abordagem lexical para lidar com diacríticos — um desafio ausente nos trabalhos em inglês.

### 3.1.3 Karpukhin et al. (2020) — Dense Passage Retrieval

Karpukhin et al. [Karpukhin et al., 2020] introduziram o Dense Passage Retrieval (DPR), demonstrando que retrieval denso com bi-encoders treinados pode superar BM25 em question answering de domínio aberto. O trabalho é referência fundamental para a busca semântica moderna e documenta um resultado importante: quando BM25 e DPR são combinados, o sistema híbrido supera ambos os componentes isolados.

**Limitações.** O foco é question answering em inglês, com passages relativamente curtas e bem-definidas. O cenário conversacional introduz desafios adicionais — turnos curtos e ruidosos, dependências entre turnos, alternância de tópicos — que não são endereçados.

**Relação com o TalkEx.** A evidência de que o híbrido BM25+DPR supera ambos os isolados fundamenta diretamente H1 e a decisão de fusão de scores no pipeline de retrieval.

### 3.1.4 Ma et al. (2021) — Reciprocal Rank Fusion

Ma et al. [Ma et al., 2021] propuseram e avaliaram Reciprocal Rank Fusion (RRF) como estratégia de fusão entre rankings lexicais e semânticos. O RRF combina as posições (ranks) dos documentos nos dois rankings, independentemente dos scores brutos, evitando a necessidade de calibração entre escalas. O método é robusto, simples de implementar e demonstrou resultados competitivos com fusões mais sofisticadas.

**Relação com o TalkEx.** Adotamos RRF como alternativa à fusão linear ponderada, permitindo avaliação comparativa das duas estratégias de fusão no desenho experimental.

---

## 3.2 Busca Lexical vs Semântica em Classificação

A tensão entre abordagens lexicais e semânticas não se limita ao retrieval — ela se manifesta de forma igualmente relevante na classificação de texto. O trabalho de Harris (2025) oferece uma contribuição empírica que desafia pressupostos comuns sobre a superioridade de embeddings semânticos.

### 3.2.1 Harris (2025) — Busca lexical vs semântica para documentos médicos

Harris [Harris, 2025] conduziu um estudo comparativo rigoroso entre 7 métodos de representação textual — Term Frequency, TF-IDF, BM25, Word2Vec, Med2Vec, MiniLM e mxbai — para classificar 1.472 documentos médicos em 7 categorias, utilizando kNN como classificador. O resultado central é contraintuitivo: BM25 alcançou a maior acurácia preditiva e foi significativamente mais rápido que embeddings semânticos. Em documentos médicos altamente estruturados, com vocabulário controlado e terminologia consistente, os sinais lexicais são suficientemente discriminativos — e os embeddings semânticos off-the-shelf não capturam a especificidade do domínio.

**Três implicações analíticas emergem deste resultado.** Primeiro, a especificidade do domínio importa: em vocabulários controlados, a precisão lexical supera a generalização semântica. Segundo, o uso de kNN como único classificador limita as conclusões — com classificadores supervisionados (regressão logística, gradient boosting), embeddings poderiam demonstrar vantagem ao capturar relações semânticas que o kNN não explora plenamente. Terceiro, o resultado é específico para embeddings off-the-shelf; embeddings fine-tuned no domínio, como demonstrado por Rayo et al. [Rayo et al., 2025], podem inverter a conclusão.

**Limitações.** O domínio médico (documentos estruturados, vocabulário formal) difere fundamentalmente do domínio conversacional (texto ruidoso, informal, coloquial). O uso exclusivo de kNN como classificador não permite distinguir se a limitação é dos embeddings ou do método de classificação. Não há componente híbrido nem avaliação em dados multi-turn.

**Relação com o TalkEx.** Este trabalho reforça duas decisões arquiteturais centrais: (i) a obrigatoriedade de benchmarking contra BM25 antes de investir em abordagens semânticas — nunca assumimos superioridade semântica a priori; e (ii) a adoção de classificadores supervisionados (regressão logística, gradient boosting, MLP) sobre embeddings, em vez de depender exclusivamente de similaridade por vizinhança — o princípio "embeddings representam, classificadores decidem" que fundamenta todo o pipeline de classificação do TalkEx.

### 3.2.2 AnthusAI — Classificação semântica de texto

O projeto AnthusAI [AnthusAI, 2024] demonstrou empiricamente o princípio de separação entre representação e decisão ao comparar Word2Vec, BERT e OpenAI Ada-2 como embeddings alimentando regressão logística para classificação binária (questões sobre legislação de imigração espanhola). Os resultados mostraram que BERT e Ada-2 alcançam desempenho equivalente e superior a Word2Vec, sustentando a conclusão de que "the most powerful and expensive models may not always be necessary".

**Três contribuições analíticas.** Primeiro, a demonstração de que modelos menores e mais baratos (BERT) podem ser tão eficazes quanto modelos maiores (Ada-2) em domínios específicos justifica o uso de sentence-transformers (E5, BGE) em vez de LLMs para embeddings online. Segundo, a arquitetura embeddings + classificador supervisionado supera consistentemente a similaridade direta, validando a separação de conceitos. Terceiro, o trabalho mostra que o gargalo não está na representação em si, mas na qualidade do classificador e na riqueza das features.

**Limitações.** A tarefa era binária e relativamente simples. Em cenários com 9 classes, alta sobreposição lexical e necessidade de features heterogêneas (lexicais + estruturais + contextuais), a dificuldade é de outra magnitude. O trabalho não incorpora features além dos embeddings, não opera sobre dados conversacionais e não inclui mecanismo de regras.

**Relação com o TalkEx.** Sustenta o axioma arquitetural de separar representação de decisão e a escolha de modelos leves para inferência online. Estendemos a abordagem com features heterogêneas em múltiplos níveis de granularidade.

---

## 3.3 Clustering como Classificação com LLMs

A utilização de Large Language Models para tarefas de classificação representa uma tendência recente que desafia a necessidade de embeddings e classificadores supervisionados. O trabalho de Huang e He (2025) é representativo desta direção.

### 3.3.1 Huang & He (2025) — Clustering de texto como classificação com LLMs

Huang e He [Huang & He, 2025] propuseram um framework de dois estágios apresentado no SIGIR-AP 2025: no primeiro, um LLM (GPT-3.5-turbo) gera labels para mini-batches de textos; no segundo, o mesmo LLM classifica textos nesses labels gerados. Avaliado em 5 datasets (ArxivS2S, GoEmo, Massive-I/D, MTOP-I) com 18 a 102 clusters, o framework alcançou desempenho próximo ao upper bound teórico (LLM com labels conhecidos) e superior a K-means, DBSCAN, IDAS, PAS e ClusterLLM em acurácia, NMI e ARI.

**A elegância conceitual do trabalho reside na transformação de clustering em classificação** — em vez de agrupar por similaridade geométrica no espaço de embeddings, o LLM gera agrupamentos semanticamente coerentes a partir da compreensão do conteúdo. Isso elimina a sensibilidade a hiperparâmetros de clustering (k, epsilon, min_samples) e produz labels legíveis por humanos.

**Limitações críticas para contextos operacionais.** Primeiro, o uso de LLM online é custo-proibitivo em escala: processar milhões de conversas mensais com GPT-3.5-turbo implica custos de centenas de milhares de dólares. Segundo, o LLM não fornece evidência rastreável de suas decisões — atribui um label, mas não explica quais sinais textuais fundamentaram a escolha, o que é inaceitável em cenários de compliance e auditoria. Terceiro, não há integração com retrieval; o framework opera exclusivamente como classificador. Quarto, a reproducibilidade é limitada pela natureza não-determinística dos LLMs — a mesma conversa pode receber labels distintos em execuções diferentes.

**Relação com o TalkEx.** Adotamos a inspiração conceitual deste trabalho para o pipeline offline de intent discovery: LLMs geram e consolidam taxonomias de intents, mas classificadores leves operam online. As regras determinísticas do TalkEx preenchem a lacuna de auditabilidade que o LLM não oferece — cada decisão carrega trilha de evidência com os sinais textuais, scores e thresholds que a fundamentaram. A separação online/offline é uma decisão arquitetural central da dissertação: LLMs são ferramentas poderosas para labeling e discovery, mas não para inferência em tempo real sobre milhões de conversas.

### 3.3.2 Dial-In LLM (Hong et al., 2024) — Clustering de intents com LLM-in-the-loop

Hong et al. [Hong et al., 2024] propuseram um sistema de clustering de intents em diálogos de atendimento com LLM-in-the-loop, avaliado sobre 100.000+ conversas chinesas de customer service e apresentado no EMNLP 2025. O sistema alcançou 95%+ de alinhamento com julgamento humano na atribuição de intents.

**Contribuição.** Demonstra que LLMs podem ser utilizados para alinhar clustering automático com taxonomias definidas por especialistas de domínio — um problema recorrente em contact centers onde a taxonomia de intents evolui organicamente.

**Limitações.** Opera exclusivamente como ferramenta de clustering, sem integração com retrieval ou regras. O pipeline é offline e depende de LLM para cada decisão. Não trata representações multi-nível nem produz evidência rastreável.

**Relação com o TalkEx.** Suporta o conceito de pipeline offline para intent discovery que adotamos. A integração com o motor de regras para governança da taxonomia — regras podem validar se novos intents descobertos pelo LLM são consistentes com a taxonomia existente — é uma extensão que propomos.

---

## 3.4 Classificação com Mecanismos de Atenção e LLMs

A aplicação de mecanismos de atenção à classificação de texto tem demonstrado ganhos consistentes, particularmente quando combinada com encoders pré-treinados. Dois trabalhos investigam esta direção com implicações diretas para a estratégia de pooling adotada no TalkEx.

### 3.4.1 Lyu et al. (2025) — Classificação com LLMs e atenção neural

Lyu et al. [Lyu et al., 2025] propuseram um framework de classificação que combina encoder LLM com mecanismo de atenção e pooling combinado (mean + attention-weighted), avaliado no AG News (4 classes, ~120.000 textos):

| Modelo | Precision | Recall | F1 | AUC |
|--------|-----------|--------|-----|-----|
| BERT | 0.87 | 0.85 | 0.86 | 0.91 |
| Transformer | 0.85 | 0.83 | 0.84 | 0.90 |
| LSTM | 0.82 | 0.80 | 0.81 | 0.87 |
| GAT | 0.83 | 0.82 | 0.82 | 0.88 |
| **Proposto** | **0.90** | **0.88** | **0.89** | **0.94** |

O ganho de F1 de 0.86 (BERT puro) para 0.89 (proposto) e de AUC de 0.91 para 0.94 é atribuído ao mecanismo de atenção, que permite ao modelo ponderar tokens por sua relevância discriminativa em vez de tratar todos os tokens igualmente (como faz o mean pooling).

**Implicações analíticas.** Dois achados do paper são particularmente relevantes. Primeiro, os autores reportaram sensibilidade ao tamanho da dimensão oculta (pico em 512, queda com 768+), o que tem implicações para a escolha de modelos de embedding. Segundo, em cenários de desbalanceamento de classes (ratio 1:6), o recall caiu de 0.88 para 0.80 — um resultado que antecipamos em nosso dataset expandido, onde a classe "outros" representa apenas 3% das conversas.

**Limitações.** O AG News consiste em textos curtos de notícias (1 parágrafo), fundamentalmente distintos de conversas multi-turn de call center. Não há noção de turnos, falantes ou janelas de contexto. O framework não incorpora features lexicais, estruturais ou regras determinísticas. A avaliação limita-se a classificação unimodal sem componente de retrieval.

**Relação com o TalkEx.** Fundamenta a decisão de investigar attention pooling como alternativa ao mean pooling na construção de representações de janelas de contexto (H2). Em conversas de call center, onde turnos como "oi, tudo bem?" e "obrigado, tchau" diluem o sinal semântico no mean pooling, o mecanismo de atenção deveria ser particularmente benéfico ao concentrar peso nos turnos informativos. Estendemos a abordagem com features heterogêneas (lexicais + estruturais + contextuais) além dos embeddings puros.

### 3.4.2 Speaker-Turn Aware Hierarchical Model (2025) — Embeddings hierárquicos com consciência de turno

Um trabalho recente publicado na Expert Systems with Applications [Speaker-Turn, 2025] propõe embeddings hierárquicos que incorporam consciência de turno e de falante na construção de representações conversacionais. O modelo gera embeddings em dois níveis — turno individual e conversa completa — e utiliza a atribuição de falante como feature explícita.

**Contribuição.** Demonstra empiricamente que a consciência de turno e de papel (cliente/agente) melhora a qualidade das representações conversacionais, apoiando diretamente a hipótese H2 de que representações multi-nível superam representações de nível único.

**Limitações.** Não incorpora retrieval híbrido, regras determinísticas ou inferência em cascata. Opera exclusivamente no eixo de representação, sem integrar os demais componentes do pipeline.

**Relação com o TalkEx.** Valida a decisão de gerar embeddings por papel (customer-only, agent-only) além dos níveis padrão (turno, janela, conversa). No TalkEx, o motor de regras pode utilizar o predicado `speaker` para restringir avaliação a turnos de um papel específico, combinando a consciência de falante com regras determinísticas.

---

## 3.5 Sistemas de Regras e DSLs para NLP

A relação entre sistemas baseados em regras e modelos estatísticos/neurais é um dos debates mais antigos e persistentes do processamento de linguagem natural. Enquanto a academia migrou progressivamente para abordagens puramente baseadas em dados, a indústria manteve — e em muitos casos expandiu — o uso de sistemas de regras para tarefas onde auditabilidade, transparência e controle são requisitos não-negociáveis. Esta seção analisa os trabalhos que definem o estado da arte neste eixo.

### 3.5.1 Chiticariu et al. (2013) — "Rule-Based Information Extraction is Dead! Long Live Rule-Based Information Extraction Systems!"

O paper de Chiticariu, Deshpande e Medar [Chiticariu et al., 2013], apresentado no EMNLP 2013, é um dos trabalhos mais citados sobre o papel de regras em NLP e seu título deliberadamente provocativo encapsula uma tese central para esta dissertação. Os autores documentaram o descompasso entre a percepção acadêmica (regras como abordagem obsoleta) e a realidade industrial (regras como paradigma dominante em sistemas de produção), sustentando o argumento com evidências de surveys e entrevistas com profissionais da indústria.

**O argumento é estruturado em três níveis.** Primeiro, a transparência: em domínios regulados (financeiro, saúde, legal), cada decisão automatizada precisa ser explicável e auditável — requisito que modelos black-box não satisfazem nativamente. Segundo, o controle: regras podem ser modificadas, adicionadas ou removidas por especialistas de domínio sem retraining de modelos, permitindo resposta rápida a mudanças regulatórias ou de política. Terceiro, a confiabilidade: regras determinísticas produzem resultados idênticos para inputs idênticos — propriedade que modelos probabilísticos não garantem.

**Limitações.** O trabalho não propõe uma integração formal entre regras e modelos estatísticos — defende a relevância das regras, mas não articula como combiná-las com abordagens neurais. Além disso, os sistemas de regras discutidos (SystemT/AQL) operam exclusivamente com predicados lexicais e sintáticos, sem predicados semânticos baseados em embeddings.

**Relação com o TalkEx.** Este trabalho fornece o fundamento conceitual para o motor de regras do TalkEx. Estendemos o paradigma em duas direções: (i) adicionamos predicados semânticos (intent_score, embedding_similarity, topic_score) ao repertório de predicados da DSL, criando regras que combinam sinais lexicais e semânticos; e (ii) integramos o motor de regras em um pipeline que inclui retrieval híbrido e classificação supervisionada, em vez de operar as regras como sistema isolado.

### 3.5.2 SystemT / AQL (Chiticariu et al., 2010-2018) — Linguagem declarativa para extração de informação

O SystemT [Chiticariu et al., 2010] é o sistema de regras declarativo desenvolvido pela IBM para extração de informação em escala. A linguagem AQL (Annotation Query Language) permite que usuários especifiquem o que extrair (não como extrair) em uma sintaxe SQL-like, que o compilador transforma em planos algébricos otimizados com estimativa de custo.

**Contribuições técnicas.** O otimizador de custo do SystemT ordena a execução de regras por custo estimado — regras lexicais baratas são avaliadas primeiro, regras mais caras (como regex complexas ou dicionários extensos) são avaliadas apenas quando necessário. Este princípio de short-circuit por custo é precursor direto da avaliação ordenada que implementamos no TalkEx.

**Limitações.** O AQL opera exclusivamente com predicados lexicais e sintáticos — padrões sobre tokens, regex, dicionários, relações gramaticais. Não há suporte a predicados semânticos baseados em embeddings ou scores de similaridade vetorial. Além disso, o SystemT foi projetado para extração de informação (NER, relações), não para classificação conversacional.

**Relação com o TalkEx.** O motor de regras do TalkEx é herdeiro conceitual do SystemT na filosofia declarativa e na otimização por custo. A diferença fundamental é a extensão do repertório de predicados para incluir sinais semânticos — o predicado `intent_score >= 0.85` avalia o score de um classificador de intents, enquanto `embedding_similarity("cancelamento", threshold=0.7)` avalia a proximidade vetorial no espaço de embeddings. Esta combinação de predicados lexicais e semânticos em uma única DSL compilada para AST é, até onde identificamos na literatura, sem precedente direto.

### 3.5.3 UIMA Ruta (Kluegl et al., 2016) — Desenvolvimento rápido de aplicações de extração baseadas em regras

O UIMA Ruta [Kluegl et al., 2016], publicado na Natural Language Engineering, é uma DSL imperativa construída sobre o framework UIMA (Unstructured Information Management Architecture) que permite scripting de regras com introspecção de execução e indução automática de regras. Dos sistemas analisados, é o mais próximo do motor de regras do TalkEx em termos de expressividade da linguagem e capacidade de introspecção.

**Contribuições.** O sistema oferece três capacidades que compartilhamos: (i) regras com condições compostas (AND, OR, NOT) sobre anotações; (ii) introspecção de execução — é possível rastrear quais regras dispararam e com quais evidências; e (iii) indução de regras a partir de exemplos anotados, que reduz o esforço de autoria.

**Limitações.** Assim como o SystemT, o UIMA Ruta não dispõe de predicados semânticos. As regras operam sobre anotações textuais (tokens, sentenças, entidades) — não sobre scores de classificadores ou similaridade vetorial. Além disso, a linguagem é imperativa (scripting com loops e condicionais), o que dificulta a otimização automática por custo que linguagens declarativas permitem.

**Relação com o TalkEx.** O TalkEx DSL adota uma abordagem declarativa (RULE ... WHEN ... THEN ...) em vez da imperativa do UIMA Ruta, o que permite otimização automática da ordem de avaliação por custo de predicado. A introspecção de execução que o UIMA Ruta oferece corresponde ao sistema de evidência do TalkEx, onde cada nó da AST produz metadata rastreável (matched_words, scores, thresholds, actual_distance).

### 3.5.4 GATE/JAPE (Cunningham et al., 2000-presente) — Padrões de anotação

O GATE (General Architecture for Text Engineering) e seu componente JAPE (Java Annotation Patterns Engine) [Cunningham et al., 2002] constituem um dos frameworks mais estabelecidos para NLP baseado em regras. O JAPE opera como transdutor de estados finitos sobre anotações, com regras definidas por padrões (LHS) e ações (RHS), organizadas em cascata de fases.

**Contribuição.** A organização em cascata de fases — onde a saída de uma fase alimenta a entrada da próxima — é conceitualmente análoga ao pipeline cascateado do TalkEx, onde estágios progressivamente mais caros refinam os resultados. O GATE/JAPE demonstrou, ao longo de duas décadas de uso em pesquisa e indústria, a viabilidade de sistemas de regras em larga escala.

**Limitações.** O JAPE opera exclusivamente sobre padrões de anotação (tokens, POS tags, entidades) — sem predicados semânticos. A linguagem é orientada a IE (extração de informação), não a classificação conversacional. A cascata é de fases de processamento, não de níveis de custo.

**Relação com o TalkEx.** Referência arquitetural para a organização em cascata, mas com escopo e repertório de predicados fundamentalmente distintos.

---

## 3.6 Weak Supervision e Regras como Labeling Functions

Uma direção distinta, mas relacionada, utiliza regras não para inferência, mas para geração de labels de treinamento. O ecossistema Snorkel é representativo.

### 3.6.1 Snorkel (Ratner et al., 2017-2020) — Data Programming

Ratner et al. [Ratner et al., 2017] introduziram o paradigma de Data Programming, formalizado no framework Snorkel (NeurIPS 2016, VLDB 2018). O conceito central é que usuários escrevem labeling functions (LFs) — heurísticas ou regras que atribuem labels ruidosos a exemplos não-anotados — e um modelo generativo estima a acurácia de cada LF para agregar os labels ruidosos em labels probabilísticos. Estes são então usados para treinar um classificador supervisionado.

**O Snorkel foi deployado em escala industrial.** Bach et al. [Bach et al., 2019] descreveram o Snorkel DryBell, implementado no Google, onde classificadores treinados com weak supervision alcançaram qualidade comparável a modelos treinados com milhares de labels manuais (SIGMOD 2019). Outros deployments incluem Apple, Intel e diversas empresas de saúde.

**A distinção conceitual entre Snorkel e o TalkEx é fundamental.** No Snorkel, regras são labeling functions — operam no treinamento para gerar labels. No TalkEx, regras são mecanismos de inferência — operam em tempo real para classificar, etiquetar e produzir evidência auditável. A convergência está no reconhecimento de que regras codificam conhecimento de domínio valioso; a divergência está no momento da aplicação (treinamento vs inferência) e no requisito de auditabilidade (Snorkel não produz trilha de evidência por decisão individual).

### 3.6.2 Villena-Román et al. (2011) — Abordagem híbrida ML + regras

Villena-Román et al. [Villena-Román et al., 2011] propuseram uma das primeiras abordagens híbridas formais combinando machine learning e sistemas especialistas para categorização de texto, apresentada no AAAI FLAIRS-24. O sistema utiliza kNN como classificador primário e regras como pós-processamento — listas de termos positivos e negativos que ajustam as predições do modelo.

**Contribuição histórica.** Este trabalho é um dos precedentes mais diretos para a integração ML + regras que o TalkEx implementa. A ideia de que regras podem corrigir erros sistemáticos dos modelos (falsos positivos em classes específicas, por exemplo) é fundamental para H3.

**Limitações.** O trabalho é pré-transformer, pré-embeddings modernos e utiliza representações bag-of-words. As regras são simples listas de termos, sem a expressividade de uma DSL com predicados compostos. Não opera sobre dados conversacionais e não incorpora retrieval.

**Relação com o TalkEx.** Estabelece o precedente histórico para a combinação ML + regras. O TalkEx estende significativamente a expressividade das regras (de listas de termos para uma DSL com predicados lexicais, semânticos, estruturais e contextuais compilados em AST) e o contexto de aplicação (de documentos para conversas multi-turn).

### 3.6.3 Hybrid Rule+ML para Detecção de PII (2025)

Um trabalho recente publicado na Nature Scientific Reports [PII Detection, 2025] propõe um sistema híbrido de regras (regex, gazetteers) e ML (NER neural) para detecção de informações pessoais em documentos financeiros. O sistema é projetado para compliance regulatória, com audit trail como requisito central.

**Relação com o TalkEx.** Reforça que a combinação de regras e ML com trilha de auditoria é uma tendência em domínios regulados. A diferença é que o trabalho foca em extração de entidades, enquanto o TalkEx opera sobre classificação e retrieval conversacional.

---

## 3.7 Conversation Intelligence em Contact Centers

O domínio de NLP aplicado a contact centers é ativo mas fragmentado — nenhum trabalho propõe uma solução integrada que combine os múltiplos paradigmas que a operação real demanda. Esta seção analisa tanto o cenário acadêmico quanto o comercial.

### 3.7.1 Shah et al. (2023) — Revisão sistemática de NLP em contact centers

Shah et al. [Shah et al., 2023] publicaram na Pattern Analysis and Applications (Springer) uma revisão sistemática de 125 papers (2003-2023) sobre NLP em automação de contact centers. A revisão cobre vetorização (TF-IDF, LSI, embeddings), classificação (redes neurais, SVM, algoritmos genéticos) e integração com ASR.

**Achado crucial para esta dissertação.** Nenhum dos 125 papers revisados combina retrieval híbrido, representação multi-nível e regras determinísticas em uma arquitetura integrada. O domínio é caracterizado pela fragmentação: cada trabalho resolve uma parte do problema (classificação de intents, detecção de sentimentos, extração de entidades, resumo automático) sem articular como essas partes se integram em um pipeline coerente.

**A revisão identifica três lacunas persistentes:** (i) a maioria dos trabalhos avalia componentes isolados, sem medir o impacto de sua integração; (ii) poucos trabalhos endereçam auditabilidade das decisões — os modelos classificam, mas não explicam; e (iii) a separação entre processamento online e offline não é tratada sistematicamente.

**Relação com o TalkEx.** Esta revisão valida a lacuna que a presente dissertação preenche. O TalkEx propõe exatamente a integração que os 125 papers não articulam: um pipeline cascateado que combina retrieval, classificação e regras sobre representações multi-nível, com separação explícita online/offline.

### 3.7.2 BERTau (Finardi et al., 2021) — BERT para atendimento digital em PT-BR

Finardi et al. [Finardi et al., 2021] apresentaram o BERTau, um modelo BERT treinado do zero sobre 5GB de conversas em português brasileiro do Itaú Unibanco. Os resultados demonstraram ganhos expressivos: +22% MRR em FAQ retrieval, +2.1% F1 em análise de sentimento e +4.4% F1 em reconhecimento de entidades, comparado a BERT multilingual.

**Este é o único trabalho acadêmico encontrado sobre NLP em atendimento ao cliente em português brasileiro.** A escassez é, em si, uma lacuna significativa — um dos maiores mercados de contact center do mundo (o Brasil opera milhões de posições de atendimento) não possui uma base de pesquisa acadêmica proporcional.

**Limitações.** O BERTau foca em três tarefas específicas (FAQ retrieval, sentimento, NER) sem integrá-las em um pipeline unificado. Não há classificação multi-label de intents, não há retrieval híbrido (usa apenas retrieval semântico), não há regras determinísticas e não há representações multi-nível. Além disso, o modelo é treinado em dados proprietários do Itaú Unibanco, não sendo reproduzível pela comunidade acadêmica.

**Relação com o TalkEx.** Demonstra a viabilidade e o impacto de modelos pré-treinados em PT-BR para atendimento ao cliente. A presente dissertação estende o escopo significativamente — de tarefas isoladas para uma arquitetura integrada — e opera sobre dados públicos, garantindo reprodutibilidade.

### 3.7.3 MINT-CL (2024) — Classificação multi-turn com aprendizado contrastivo

O MINT-CL [MINT-CL, 2024], apresentado no CIKM 2025, propõe um framework para classificação de intents em diálogos multi-turn que modela a dinâmica de intents ao longo dos turnos. O sistema utiliza Chain-of-Intent (HMM + LLM) para geração de diálogos sintéticos e aprendizado contrastivo para classificação.

**Contribuição.** O trabalho demonstra que modelar a evolução temporal de intents dentro de uma conversa melhora a classificação — o intent do turno 5 pode depender da sequência de intents dos turnos 1-4. Este resultado apoia diretamente H2: representações que capturam dependências multi-turn superam representações de turno isolado.

**Limitações.** Não incorpora retrieval híbrido nem regras determinísticas. O componente LLM é usado em treinamento (geração de dados), não em inferência. A avaliação é em inglês, sem tratamento de desafios específicos de PT-BR.

**Relação com o TalkEx.** Valida a decisão de construir janelas de contexto que capturam dependências multi-turn. O predicado contextual `occurs_after` do TalkEx DSL opera sobre a mesma intuição — um padrão sequencial entre turnos é mais informativo que predicados sobre turnos isolados.

### 3.7.4 MDPI (2025) — Classificação em call centers com desbalanceamento

Um trabalho publicado na MDPI Electronics [MDPI, 2025] investigou classificação de conversas de call center coreanas com desbalanceamento severo (2%-26% por classe), utilizando KoBERT, Easy Data Augmentation (EDA) e metadados de NER. O EDA + NER metadata melhorou a classificação nas classes minoritárias.

**Limitações.** Abordagem single-paradigm (apenas classificação, sem retrieval ou regras). Dataset em coreano, sem tratamento de particularidades linguísticas de PT-BR.

**Relação com o TalkEx.** Relevante para o tratamento de desbalanceamento de classes no corpus expandido, onde a distribuição realista (dúvida ~35%, elogio ~4%, outros ~3%) introduz o mesmo desafio.

### 3.7.5 QiBERT (2024) — SBERT para classificação de conversas em português

O QiBERT [QiBERT, 2024] utilizou embeddings SBERT como features para classificação de conversas online em português europeu, alcançando mais de 0.95 de acurácia. O trabalho demonstra a eficácia de sentence-transformers como representações para classificação conversacional em português — embora em português europeu, não brasileiro.

**Relação com o TalkEx.** Apoia o axioma "embeddings representam, classificadores decidem" no contexto da língua portuguesa. A diferença entre PT-EU e PT-BR (vocabulário, informalismo, gírias) é uma variável que o TalkEx endereça com normalização accent-aware e suporte a coloquialismos brasileiros.

### 3.7.6 Cenário comercial

O mercado de conversation intelligence é dominado por ferramentas comerciais — Observe.AI, CallMiner, Verint, NICE, Genesys, entre outras — que oferecem classificação de intents, análise de sentimentos, detecção de compliance e sumarização automática. Quatro limitações sistemáticas caracterizam essas ferramentas:

1. **Modelos black-box.** As decisões de classificação não são auditáveis — o sistema atribui um label, mas não fornece evidência rastreável dos sinais que fundamentaram a decisão. Em cenários de compliance regulatória (Bacen, LGPD, Procon), isso é inaceitável.

2. **Dependência de LLM online.** As versões mais recentes dessas ferramentas integram LLMs para sumarização e classificação, o que implica custos de API proporcionais ao volume — proibitivos para operações brasileiras de grande porte (centenas de milhares de conversas por dia).

3. **Ausência de pipeline híbrido transparente.** As ferramentas não expõem como combinam sinais lexicais e semânticos, impossibilitando calibração por especialistas de domínio. O operador não pode, por exemplo, ajustar o peso do componente lexical quando identifica que termos específicos (códigos de produto, nomes de planos) são mais discriminativos que a semântica genérica.

4. **Inglês-cêntrica.** A maioria foi projetada para inglês, com suporte a PT-BR como segunda classe — normalização de diacríticos, tratamento de gírias e coloquialismos brasileiros são frequentemente deficientes.

Essas limitações não são deficiências técnicas inevitáveis — são consequências de decisões comerciais que privilegiam generalidade sobre especialização. O TalkEx, como artefato acadêmico, pode tomar decisões arquiteturais diferentes: transparência total do pipeline, regras auditáveis, custos controlados por inferência em cascata e tratamento nativo de PT-BR.

---

## 3.8 Inferência em Cascata

A inferência em cascata — aplicar processamento progressivamente mais caro conforme a dificuldade da instância — é um princípio que permeia domínios desde visão computacional até NLP moderno. Esta seção analisa os trabalhos que fundamentam a hipótese H4.

### 3.8.1 Viola & Jones (2001) — Cascata de classificadores para detecção de faces

Viola e Jones [Viola & Jones, 2001] introduziram o paradigma de cascata de classificadores para detecção de faces em tempo real — uma cadeia de classificadores de complexidade crescente onde estágios baratos rejeitam rapidamente regiões da imagem que claramente não contêm faces, e apenas as regiões ambíguas chegam aos classificadores mais caros.

**Este trabalho é a referência fundacional do paradigma cascateado.** A intuição central — a maioria das instâncias é fácil e pode ser resolvida por modelos baratos; apenas as instâncias difíceis justificam modelos caros — é universal e independente de domínio.

**Relação com o TalkEx.** O pipeline cascateado do TalkEx segue a mesma intuição: estágios baratos (filtros por canal, idioma, regras lexicais simples) resolvem a maioria das conversas; retrieval híbrido resolve as intermediárias; classificação supervisionada + regras semânticas resolvem as difíceis; LLMs offline revisam apenas as excepcionais.

### 3.8.2 Varshney & Baral (2022) — Cascata de modelos para classificação de texto

Varshney e Baral [Varshney & Baral, 2022] propuseram um sistema de cascata de modelos para classificação de texto que escala do menor ao maior modelo. O sistema alcançou **88.93% de redução de custo com menos de 2% de perda de acurácia** — resultado que sustenta diretamente a viabilidade de H4.

**O mecanismo é elegante em sua simplicidade.** Cada instância é processada pelo modelo mais barato. Se a confiança da predição excede um threshold, a predição é aceita e o processamento termina. Caso contrário, a instância é escalada para o próximo modelo. O resultado é que a maioria das instâncias fáceis é resolvida pelo modelo mais barato, e o custo médio por instância é drasticamente reduzido.

**Limitações.** O sistema opera exclusivamente como classificador cascateado — sem retrieval, sem regras, sem representações multi-nível. A avaliação foca em benchmarks padrão (SST-2, IMDB), não em dados conversacionais.

**Relação com o TalkEx.** Evidência direta para H4. O target de redução de custo >= 40% com degradação de F1 < 2% é conservador em relação ao resultado reportado (88.93%). A distinção é que o TalkEx cascateia não apenas modelos, mas paradigmas inteiros (regras → retrieval → classificação → LLM).

### 3.8.3 FrugalML (Chen et al., 2020) — Aprendizado de quando escalar

Chen et al. [Chen et al., 2020] propuseram o FrugalML, um framework que aprende quando escalar para modelos mais caros ao chamar APIs de ML. O sistema alcançou **90% de redução de custo** mantendo qualidade comparável ao melhor modelo individual.

**Contribuição conceitual.** O FrugalML formaliza a decisão de escalamento como um problema de otimização — dado um orçamento de custo, qual estratégia de roteamento maximiza a qualidade? Essa formalização é relevante para o design de thresholds de cascata no TalkEx.

**Relação com o TalkEx.** Suporta H4 e informa o design dos thresholds de confiança que determinam quando uma conversa é promovida para o estágio seguinte.

### 3.8.4 DeeBERT (Xin et al., 2020) — Saída antecipada em BERT

Xin et al. [Xin et al., 2020] introduziram o DeeBERT, que permite saída antecipada em camadas intermediárias do BERT quando a confiança da predição excede um threshold. O resultado é uma redução de ~40% no tempo de inferência com degradação mínima de qualidade.

**Trabalhos relacionados subsequentes** incluem o FastBERT [Liu et al., 2020], que utiliza self-distillation com classificadores em cada camada (speed-up de 1-12x com 0.1-1% de perda em F1), e o PABEE [Zhou et al., 2020], que adota saída por paciência (patience-based) — espera N camadas concordarem antes de aceitar a predição, produzindo resultados mais estáveis que o DeeBERT.

**Relação com o TalkEx.** Estes trabalhos operam em nível de camadas de um único modelo, enquanto o TalkEx opera em nível de estágios de um pipeline completo. A intuição é a mesma (processar o mínimo necessário), mas a granularidade é diferente.

### 3.8.5 Wang et al. (2011) — Cascata de custo crescente para detecção

Wang, Trapeznikov e Saligrama [Wang et al., 2011] formalizaram matematicamente a cascata de classificadores de custo crescente, estabelecendo limites teóricos para o tradeoff custo-qualidade. O framework demonstra que, sob certas condições, a cascata é Pareto-ótima — não é possível melhorar custo sem sacrificar qualidade e vice-versa.

**Relação com o TalkEx.** Fornece o fundamento teórico para a análise de curva de Pareto que planejamos no desenho experimental (Phase 5.2) — custo vs qualidade para diferentes configurações de cascata.

### 3.8.6 Green AI (Schwartz et al., 2020)

Schwartz et al. [Schwartz et al., 2020] argumentaram que a eficiência computacional deve ser uma métrica de primeira classe em pesquisa de IA, propondo o conceito de "Green AI" em contraste com "Red AI" (busca de SOTA sem consideração de custo). Strubell et al. [Strubell et al., 2019] quantificaram o custo ambiental: treinar um único transformer grande emite ~626.000 libras de CO2.

**Relação com o TalkEx.** Suporte conceitual para H4 — a inferência em cascata não é apenas uma otimização de custo financeiro, mas uma contribuição para eficiência ambiental. Em operações com milhões de conversas, a diferença entre processar tudo com o modelo mais caro e cascatear por complexidade é significativa.

---

## 3.9 Sistemas Neuro-Simbólicos

A integração de componentes neurais (embeddings, redes neurais) com componentes simbólicos (regras, lógica formal, ontologias) constitui o campo dos sistemas neuro-simbólicos, que oferece uma perspectiva teórica relevante para posicionar o TalkEx.

### 3.9.1 Survey de IA Neuro-Simbólica (2025)

Uma revisão sistemática recente [Neuro-Symbolic Survey, 2025] analisou 167 papers publicados entre 2020 e 2024, identificando que 63% focam em Learning/Inference e 44% em Knowledge Representation. A maioria dos trabalhos opera sobre knowledge graphs, não sobre classificação de texto.

**Relação com o TalkEx.** O TalkEx pode ser caracterizado como um sistema neuro-simbólico aplicado a conversas: o componente neural (embeddings, classificadores) fornece representação e decisão estatística, enquanto o componente simbólico (DSL compilada em AST) fornece regras determinísticas com evidência auditável. Esta dimensão adiciona uma perspectiva teórica ao posicionamento da dissertação.

### 3.9.2 Liusie et al. (2024) — Sinergia entre ML e métodos simbólicos para NLP

Liusie et al. [Liusie et al., 2024] publicaram um survey sobre abordagens híbridas em NLP, categorizando estratégias de integração: simbólico como features de input, restrições simbólicas sobre output neural e treinamento conjunto. Nenhuma das estratégias documentadas corresponde exatamente ao modelo do TalkEx — onde regras simbólicas compiladas em AST operam em paralelo com classificadores neurais, com fusão de decisões e prioridade configurável.

---

## 3.10 Síntese e Posicionamento

### 3.10.1 Lacunas identificadas na literatura

A análise dos trabalhos apresentados nas seções anteriores revela cinco lacunas que, em conjunto, definem o espaço de contribuição desta dissertação:

**Lacuna 1 — Integração de paradigmas.** Trabalhos em retrieval híbrido [Rayo et al., 2025; Karpukhin et al., 2020] não incorporam classificação supervisionada nem regras determinísticas. Trabalhos em classificação [Lyu et al., 2025; AnthusAI, 2024] não incorporam retrieval nem regras. Trabalhos em regras [Chiticariu et al., 2013; Ratner et al., 2017] não incorporam retrieval híbrido nem representações multi-nível. A integração dos três paradigmas em um único pipeline permanece inexplorada.

**Lacuna 2 — Representações multi-nível para conversas.** A maioria dos trabalhos opera sobre representações de nível único — documento inteiro [Rayo et al., 2025; Harris, 2025], turno isolado [Lyu et al., 2025] ou conversa completa [BERTau, 2021]. A construção de representações em múltiplos níveis de granularidade (turno, janela de contexto, conversa, por papel) e sua avaliação comparativa em classificação conversacional são pouco investigadas. MINT-CL [2024] e Speaker-Turn [2025] avançam nesta direção, mas sem integração com retrieval ou regras.

**Lacuna 3 — Auditabilidade com predicados semânticos.** Sistemas de regras existentes (SystemT, UIMA Ruta, GATE/JAPE) operam exclusivamente com predicados lexicais e sintáticos. Regras que combinam predicados lexicais (contains, regex, BM25 score) com predicados semânticos (intent_score, embedding_similarity) em uma DSL compilada para AST com trilha de evidência rastreável não possuem precedente direto na literatura.

**Lacuna 4 — Domínio conversacional PT-BR.** O BERTau [Finardi et al., 2021] é essencialmente o único trabalho acadêmico sobre NLP em atendimento ao cliente em português brasileiro, e foca em tarefas isoladas (FAQ retrieval, sentimento, NER) sem pipeline integrado. O QiBERT [2024] opera em português europeu. Nenhum trabalho endereça classificação multi-label de intents em conversas multi-turn PT-BR com tratamento nativo de diacríticos e coloquialismos.

**Lacuna 5 — Cascata de paradigmas (não apenas de modelos).** Trabalhos em inferência cascateada [Varshney & Baral, 2022; FrugalML, 2020; DeeBERT, 2020] cascateiam modelos de complexidade crescente dentro de um único paradigma (classificação). A cascata de paradigmas inteiros — regras baratas → retrieval híbrido → classificação supervisionada → LLM offline — não foi investigada sistematicamente.

### 3.10.2 Tabela comparativa

A Tabela 3.1 sintetiza o posicionamento de 15 trabalhos representativos frente a 5 dimensões que esta dissertação integra. Cada dimensão corresponde a um eixo da arquitetura proposta.

**Tabela 3.1** — Comparação de trabalhos relacionados em 5 dimensões. As colunas indicam: *Híbrido* = retrieval lexical + semântico com fusão; *Multi-nível* = representações em múltiplas granularidades (turno, janela, conversa); *Regras* = regras determinísticas com auditabilidade; *Conversacional* = operação sobre diálogos multi-turn; *PT-BR* = avaliação em português brasileiro.

| Trabalho | Híbrido | Multi-nível | Regras | Conversacional | PT-BR |
|----------|:-------:|:-----------:|:------:|:--------------:|:-----:|
| Rayo et al. (2025) | Sim | -- | -- | -- | -- |
| Gokhan et al. (2024) | -- | -- | -- | -- | -- |
| Harris (2025) | -- | -- | -- | -- | -- |
| Huang & He (2025) | -- | -- | -- | -- | -- |
| Lyu et al. (2025) | -- | -- | -- | -- | -- |
| AnthusAI (2024) | -- | -- | -- | -- | -- |
| Varshney & Baral (2022) | -- | Parcial | -- | -- | -- |
| FrugalML (2020) | -- | Parcial | -- | -- | -- |
| SystemT (2010) | -- | -- | Sim | -- | -- |
| Snorkel (2017) | -- | -- | Sim* | -- | -- |
| Villena-Román (2011) | -- | -- | Sim | -- | -- |
| Shah et al. (2023) | Parcial | -- | Parcial | Sim (revisão) | -- |
| BERTau (2021) | -- | -- | -- | Sim | Sim |
| QiBERT (2024) | -- | -- | -- | Sim | -- |
| MINT-CL (2024) | -- | Sim | -- | Sim | -- |
| **TalkEx (proposto)** | **Sim** | **Sim** | **Sim** | **Sim** | **Sim** |

*Snorkel utiliza regras para labeling (treinamento), não para inferência auditável.

**Observações sobre a tabela.** Varshney & Baral e FrugalML recebem "Parcial" em multi-nível porque cascateiam modelos de complexidade crescente, o que é uma forma de processamento em níveis, embora não de representação multi-nível. Shah et al. recebem "Parcial" em híbrido e regras porque a revisão menciona trabalhos que empregam cada abordagem isoladamente, mas nenhum que as integre. A coluna PT-BR registra o QiBERT como ausente por operar em português europeu, não brasileiro.

### 3.10.3 Formulação da lacuna

A análise da Tabela 3.1 permite formular a lacuna de forma precisa:

> Nenhum trabalho existente na literatura combina retrieval híbrido (lexical + semântico com fusão de scores), classificação supervisionada sobre representações multi-nível (turno, janela de contexto, conversa) e regras determinísticas auditáveis compiladas a partir de DSL em uma arquitetura integrada, operando sobre conversas multi-turn — e nenhum o faz em português brasileiro.

Esta lacuna não é um artefato de busca incompleta — é consequência da especialização natural da pesquisa, onde cada comunidade avança profundamente em um eixo (retrieval, classificação, regras, análise de diálogos) sem articular a integração entre eixos. A contribuição desta dissertação é demonstrar que a integração é viável, que os paradigmas são complementares (não redundantes), e que a arquitetura cascateada permite operar essa integração com eficiência computacional controlada.
