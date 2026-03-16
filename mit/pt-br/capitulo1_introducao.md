# Capítulo 1 — Introdução

## Resumo do Capítulo

Este capítulo estabelece o contexto da pesquisa, formula o problema central de pesquisa, deriva quatro hipóteses falsificáveis e apresenta os objetivos e contribuições da dissertação. Investigamos se uma arquitetura híbrida em cascata que combina recuperação lexical (BM25), recuperação semântica (embeddings densos de um encoder congelado), classificação supervisionada sobre features heterogêneas e um motor de regras determinístico pode melhorar a classificação de intenções conversacionais e a qualidade de recuperação em relação a paradigmas isolados, preservando a explicabilidade operacional. Nossos experimentos são conduzidos sobre um corpus publicamente disponível de atendimento ao cliente em PT-BR com 2.122 conversas distribuídas em 8 classes de intent, seguindo um protocolo rigoroso de auditoria de dados. Reportamos tanto confirmações quanto refutações com total transparência estatística.

---

## 1.1 Contexto e Motivação

### 1.1.1 A Escala dos Dados Conversacionais em Contact Centers Brasileiros

As operações de atendimento ao cliente no Brasil geram um dos maiores volumes de texto em linguagem natural produzidos no país. Operações de grande escala em telecomunicações, serviços financeiros, varejo e saúde processam individualmente entre cinco e quinze milhões de conversas por mês através de canais de voz, chat, e-mail e redes sociais (Associação Brasileira de Telesserviços, 2023). Em conjunto, esse volume constitui um corpus significativo de português brasileiro autêntico falado e escrito — e um dos mais sistematicamente subexplorados.

A perda informacional nesse contexto não é incidental; é estrutural. Estimativas da indústria indicam consistentemente que menos de cinco por cento das conversas de atendimento ao cliente recebem qualquer forma de análise além da disposição manual pós-chamada inserida pelo agente — uma categorização amplamente reconhecida como inconsistente, de granularidade grosseira e distorcida pela pressão para minimizar o tempo médio de atendimento (TMA). Os noventa e cinco por cento restantes das conversas persistem como dados brutos não estruturados, inacessíveis à busca, classificação, auditoria ou análise de tendências em escala. O resultado é um paradoxo operacional: as organizações investem pesadamente na captura de conversas (infraestrutura de telefonia, sistemas de gravação, plataformas omnichannel) enquanto extraem uma fração negligenciável do valor informacional que essas conversas contêm.

Essa lacuna entre volume de dados e utilização analítica não é meramente um problema de eficiência. Ela tem consequências operacionais concretas em pelo menos quatro dimensões:

**Conformidade e risco regulatório.** Em setores regulados — telecomunicações (Anatel), serviços financeiros (Bacen, SUSEP) e planos de saúde suplementar (ANS) — os reguladores exigem cada vez mais evidências demonstráveis de que os agentes seguem scripts de divulgação obrigatórios, que reclamações são registradas e resolvidas dentro de prazos legais, e que práticas de vendas proibidas não estão ocorrendo. A auditoria manual pode cobrir apenas uma pequena amostra das interações. Um sistema automatizado capaz de detectar menções a órgãos reguladores (Procon, Reclame Aqui, Anatel), não divulgação de taxas ou desvios do script do agente — e produzir evidências auditáveis para cada detecção — atenderia diretamente a uma necessidade concreta de conformidade regulatória.

**Retenção de clientes e previsão de churn.** Sinais de churn em conversas de atendimento ao cliente raramente são explícitos em um único enunciado. Um cliente que cancelará uma assinatura tipicamente percorre uma trajetória de múltiplos turnos: uma reclamação técnica não resolvida, seguida de uma solicitação de supervisor, seguida de uma ameaça direta de cancelamento. Classificar turnos individuais isoladamente perde essa trajetória. Um sistema capaz de modelar o contexto multi-turno e detectar padrões de escalação permite intervenção proativa antes que o pedido de cancelamento seja feito — precisamente a capacidade operacional que os gestores de contact center mais valorizam e que as ferramentas atuais mais consistentemente falham em entregar.

**Garantia de qualidade e consistência operacional.** O monitoramento de qualidade dos agentes em escala requer a capacidade de classificar cada conversa contra uma taxonomia consistente de razões de contato. A prática atual — códigos de disposição autorreportados pelo agente — é não confiável: agentes da mesma operação aplicam a mesma taxonomia de maneiras diferentes, sub-reportam contatos de reclamação e sistematicamente categorizam incorretamente interações ambíguas. A classificação automatizada com critérios consistentes aplicados uniformemente a todas as conversas não é meramente uma conveniência; é o pré-requisito para qualquer análise operacional significativa.

**Pesquisa de experiência do cliente (CX) e treinamento.** A capacidade de buscar conversas similares a um exemplo dado — "encontre todas as conversas em que o cliente mencionou um erro específico de faturamento e subsequentemente recebeu uma oferta de desconto" — viabiliza tanto fluxos de pesquisa quanto de treinamento que atualmente são realizados por meio de rotulagem manual. A qualidade da recuperação determina diretamente a utilidade desses fluxos de trabalho.

### 1.1.2 Limitações das Abordagens Atuais

As ferramentas analíticas disponíveis para análise de conversas em larga escala exibem limitações fundamentais que motivam a presente pesquisa.

**Classificação de intenção rasa baseada em turno único.** A maioria das plataformas comerciais de inteligência conversacional opera sobre turnos individuais isoladamente, sem modelar o contexto conversacional. Essa escolha de projeto reflete o domínio histórico de frameworks de detecção de intenção para chatbots, que são arquiteturalmente de turno único, sobre a tarefa mais complexa de compreensão de conversas multi-turno. A consequência é uma incapacidade sistemática de capturar os padrões estruturais que definem muitas categorias de razão de contato de alto valor: sequências de escalação, padrões de objeção-após-oferta, mudanças de tópico e a emergência distribuída de intenção ao longo de múltiplos turnos. Um cliente que diz "Quero saber meu saldo" no turno 1, "isso é muito mais do que eu esperava, isso não pode estar certo" no turno 3 e "Quero falar com um supervisor" no turno 5 está exibindo um padrão de reclamação-e-escalação que nenhum classificador de turno único pode detectar.

**A falsa dicotomia lexical-ou-semântica.** A busca sobre conversas de atendimento ao cliente historicamente se restringiu a uma de duas abordagens: correspondência lexical (BM25, TF-IDF, busca por palavras-chave) ou busca semântica (recuperação densa por vetores sobre embeddings). Ambas possuem limitações bem documentadas no domínio conversacional. Abordagens lexicais se destacam na correspondência exata de termos — códigos de produto, nomes de planos, palavras-chave regulatórias, vocabulário único — mas falham em capturar paráfrases e intenção implícita: um cliente dizendo "Não quero mais esse serviço" não será recuperado por uma busca pela palavra-chave "cancelamento". Abordagens semânticas usando encoders densos pré-treinados generalizam através de variações lexicais e paráfrases, mas são computacionalmente mais caras, menos interpretáveis e surpreendentemente frágeis em domínios especializados onde o vocabulário técnico é denso e consistente — precisamente as condições sob as quais abordagens lexicais têm melhor desempenho. Harris (2025) forneceu evidência direta disso: BM25 superou embeddings semânticos prontos em classificação de documentos médicos estruturados, um resultado que desafia a suposição de universalidade semântica. A resposta apropriada não é escolher um paradigma, mas compreender as condições sob as quais cada um se destaca e construir um sistema capaz de aproveitar ambos.

**Ausência de explicabilidade e auditabilidade.** Sistemas baseados exclusivamente em modelos estatísticos de aprendizado de máquina produzem predições sem evidência rastreável. Quando um classificador neural rotula uma conversa como "risco de churn", não existe mecanismo pelo qual um analista de qualidade ou auditor de conformidade possa verificar por que essa classificação foi feita — quais palavras a desencadearam, qual padrão contextual foi correspondido, qual threshold de confiança foi aplicado. Em domínios regulados, essa opacidade é um impedimento operacional. Decisões baseadas em modelos opacos não podem ser auditadas, questionadas ou explicadas aos consumidores conforme exigido por frameworks regulatórios em expansão, incluindo a Lei Geral de Proteção de Dados (LGPD) do Brasil e requisitos setoriais de divulgação. Um sistema destinado à implantação em produção em ambientes regulados de contact center deve produzir evidência junto com cada predição.

**Custo computacional uniforme.** Pipelines tradicionais de PLN aplicam a mesma profundidade de processamento a cada conversa, independentemente da complexidade. Uma interação simples de saudação ("Olá, gostaria de consultar meu saldo") recebe o mesmo investimento computacional que uma negociação complexa de retenção multi-turno com vinte turnos abrangendo múltiplos tópicos. Essa uniformidade é ineficiente em qualquer implantação de larga escala, onde o custo por inferência é uma restrição operacional concreta. O princípio da inferência em cascata — aplicar estágios de processamento progressivamente mais caros e resolver casos simples antecipadamente — é estabelecido em visão computacional (Viola e Jones, 2001) e recuperação de informação (Liu et al., 2011), mas sua aplicação à classificação conversacional de PLN permanece pouco estudada.

### 1.1.3 A Oportunidade de Pesquisa

As limitações enumeradas acima definem uma oportunidade de pesquisa coerente: investigar se uma única arquitetura pode simultaneamente endereçar a modelagem rasa de intenção (por meio de representações multi-nível que capturam contexto de turno, janela e conversa), o trade-off lexical-semântico (por meio de recuperação híbrida com fusão de scores), a lacuna de explicabilidade (por meio de um motor de regras determinístico com evidência rastreável) e o problema de custo computacional (por meio de inferência em cascata com saída antecipada baseada em confiança). Nenhum trabalho existente combinou todas as quatro abordagens no domínio conversacional.

Esta dissertação apresenta o TalkEx, um Motor de Inteligência Conversacional que operacionaliza essa integração. O TalkEx é um pipeline completo de PLN — desde a ingestão de texto bruto, passando pela segmentação de turnos, construção de janelas de contexto, geração de embeddings multi-nível, recuperação híbrida, classificação supervisionada, até um motor de regras semânticas baseado em uma linguagem de domínio específico compilada para uma árvore de sintaxe abstrata — implementado como um artefato de software de qualidade de produção com 170 arquivos-fonte e aproximadamente 1.900 testes unitários e de integração.

O insight de projeto central que motiva o TalkEx é que as quatro abordagens não estão em tensão, mas são complementares. Embeddings capturam generalização semântica; features lexicais capturam sinais discriminativos de correspondência exata; classificadores supervisionados com features heterogêneas combinam ambas as famílias de sinais; e regras determinísticas fornecem cobertura de decisão previsível e auditável para casos de alta criticidade. A inferência em cascata controla o custo computacional de combiná-las. A questão empírica — se essa combinação produz melhorias mensuráveis sobre paradigmas isolados em um dataset conversacional realista — é o objeto desta dissertação.

---

## 1.2 Problema de Pesquisa

### 1.2.1 Questão Central de Pesquisa

A análise precedente motiva a seguinte questão central de pesquisa:

> **Uma arquitetura híbrida em cascata que integra recuperação lexical, representações semânticas de encoder congelado, classificação supervisionada sobre features heterogêneas e um motor de regras determinístico pode melhorar a qualidade de classificação e recuperação em relação a paradigmas isolados no domínio de conversas de atendimento ao cliente em português brasileiro, preservando a explicabilidade operacional?**

Esta questão é deliberadamente delimitada. Ela pergunta se a combinação *pode* melhorar em relação a paradigmas isolados — um enquadramento investigativo — em vez de afirmar que a combinação *melhora* universalmente. Os resultados experimentais nesta dissertação confirmam alguns componentes da combinação e questionam outros; o enquadramento honesto da pergunta desde o início reflete o status epistêmico das respostas.

### 1.2.2 Três Tensões Fundamentais

A questão de pesquisa encapsula três tensões fundamentais que permeiam o processamento de linguagem natural aplicado a dados conversacionais:

**Tensão 1: Cobertura lexical versus generalização semântica.** Recuperação lexical e semântica operam sobre sinais complementares. O BM25 é preciso para termos exatos, vocabulário técnico e palavras-chave regulatórias — críticos em dados de contact center onde nomes de planos, códigos de produto e frases de conformidade aparecem literalmente. Embeddings semânticos densos generalizam através de variantes lexicais e paráfrases — essenciais para capturar intenção quando os clientes expressam a mesma necessidade subjacente em linguagem altamente variável. Rayo et al. (2025) demonstraram que uma combinação híbrida (BM25 mais encoder de embeddings com fine-tuning) superou ambas as abordagens isoladas em recuperação de texto regulatório, alcançando Recall@10 de 0,833 contra 0,761 para BM25 isolado e 0,810 para busca semântica isolada. No entanto, esse resultado foi obtido em textos regulatórios longos e formais — documentos europeus de obrigações do dataset ObliQA — um registro substancialmente diferente das conversas curtas, informais e multi-turno em PT-BR examinadas neste trabalho. A transferibilidade da vantagem híbrida para o domínio conversacional é uma questão empírica não trivial.

**Tensão 2: Granularidade no nível do turno versus modelagem contextual multi-turno.** Conversas de atendimento ao cliente possuem uma estrutura hierárquica natural: turnos individuais, janelas de contexto (sequências de turnos adjacentes) e a conversa completa. Cada nível captura informações qualitativamente diferentes. Um único turno captura uma intenção local ("Quero cancelar"). Uma janela de contexto de cinco a dez turnos captura dependências multi-turno — uma sequência de escalação, uma transição de tópico, um padrão de objeção. A conversa completa captura o objetivo dominante e o desfecho da interação. Classificadores operando em uma única granularidade perdem informação estrutural. Lyu et al. (2025) demonstraram que pooling baseado em atenção sobre representações no nível do token melhora a classificação (F1 de 0,89 versus 0,86 para BERT base), mas sua avaliação foi conduzida no dataset de textos curtos AG News — uma frase por amostra — não em conversas multi-turno onde a modelagem contextual é qualitativamente mais importante. O valor de representações conversacionais multi-nível no cenário de classificação de intenção requer investigação experimental direta.

**Tensão 3: Aprendizado estatístico versus auditabilidade determinística.** Modelos de aprendizado de máquina aprendem padrões a partir dos dados, mas não fornecem garantias formais nem rastro de decisão interpretável. Regras determinísticas fornecem previsibilidade, auditabilidade e custo zero de inferência para casos cobertos, mas requerem codificação explícita de conhecimento e não generalizam para padrões não cobertos. Huang e He (2025) demonstraram que grandes modelos de linguagem podem realizar classificação de texto competitiva sem fine-tuning, aproveitando clustering como paradigma de classificação — mas ao custo de exigir inferência online de LLM (proibitiva na escala de milhões de conversas por mês) e sem produzir evidência rastreável por decisão. Chiticariu et al. (2013) argumentaram que sistemas baseados em regras complementam abordagens estatísticas precisamente em cenários que requerem transparência e controle. Nenhum trabalho anterior integra regras determinísticas com evidência rastreável em um pipeline híbrido de recuperação e classificação para dados conversacionais multi-turno.

### 1.2.3 Desafios Específicos do Domínio

Além das três tensões fundamentais, o domínio de conversas de atendimento ao cliente em português brasileiro impõe desafios específicos que amplificam a dificuldade do problema:

**Ruído de reconhecimento automático de fala (ASR).** Interações de voz — o canal dominante em contact centers brasileiros — passam por sistemas de ASR que introduzem erros sistemáticos: palavras substituídas, limites de turno incorretos, pontuação e capitalização ausentes, e erros de transcrição de entidades nomeadas. Um sistema de classificação e recuperação deve ser robusto a esse ruído, requerendo normalização de texto agressiva sem perda de sinais discriminativos. O pipeline de normalização do TalkEx aplica normalização Unicode NFKD com remoção de acentos e transformação para minúsculas — crítico para português brasileiro, onde a presença ou ausência de diacríticos é inconsistente tanto na saída de ASR quanto no texto escrito pelo cliente ("não" versus "nao", "cancelamento" versus "cancelamênto").

**Dependências de intenção multi-turno.** A intenção do cliente frequentemente não reside em um único turno. Padrões de intenção que definem as categorias de razão de contato de maior valor — escalação de churn, relato de fraude, reclamação regulatória — são inerentemente fenômenos multi-turno. Um classificador que processa turnos isoladamente perde sistematicamente esses padrões. A abordagem de janela de contexto deslizante implementada no TalkEx, com tamanho de janela e passo configuráveis, fornece o alcance contextual necessário para capturar essas dependências, ao custo de introduzir escolhas de configuração (tamanho da janela, passo, alinhamento de falante, ponderação por recência) que afetam a qualidade da classificação.

**Variabilidade linguística do português brasileiro.** O PT-BR como falado em contact centers exibe alta variabilidade: abreviações ("vc", "td", "blz", "pq"), expressões regionais, diacríticos inconsistentes, coloquialismos e alternância de código com termos em inglês ("upgrade", "app", "feedback"). Essa variabilidade afeta tanto o componente de recuperação lexical — onde a normalização consistente determina se "cancelamento" e "cancelámento" são tratados como o mesmo token — quanto o componente de classificação, onde os padrões distribucionais de vocabulário diferem do texto em português formal sobre o qual a maioria dos modelos multilíngues disponíveis foi primariamente treinada.

**Requisitos simultâneos de qualidade, explicabilidade e custo.** Em ambientes de produção processando milhões de conversas, um sistema que meramente classifica com precisão é insuficiente. Ele também deve explicar suas classificações em termos que analistas humanos possam verificar e atuar, e deve fazê-lo a um custo computacional compatível com os requisitos de throughput de implantações de larga escala. Esses três requisitos frequentemente conflitam: modelos mais sofisticados melhoram a acurácia, mas aumentam o custo e reduzem a interpretabilidade. A arquitetura em cascata é projetada para gerenciar esse trade-off, mas fazê-lo efetivamente requer calibração empírica.

---

## 1.3 Hipóteses

A partir da questão de pesquisa e da análise da literatura, derivamos quatro hipóteses falsificáveis. Cada uma é enunciada com critérios quantitativos de sucesso que foram especificados antes da condução dos experimentos. Os critérios são rigorosos o suficiente para permitir refutação — e, como os resultados mostrarão, duas das quatro hipóteses não foram confirmadas sob esses critérios.

### H1 — Recuperação Híbrida Supera Paradigmas Isolados de Recuperação

> O sistema de recuperação híbrida (BM25 mais busca por vizinhos mais próximos aproximados sobre embeddings de encoder congelado, com fusão paramétrica de scores) alcança Mean Reciprocal Rank (MRR) estritamente superior tanto ao baseline apenas-BM25 quanto ao baseline apenas-ANN no corpus de atendimento ao cliente em PT-BR, com significância estatística em α = 0,05 sob o teste de postos com sinais de Wilcoxon sobre scores MRR por consulta.

Esta hipótese operacionaliza a tese de complementaridade: que sinais lexicais e semânticos são suficientemente diferentes em seus modos de falha para que combiná-los produza uma vantagem mensurável sobre qualquer um isoladamente. O critério de sucesso é deliberadamente específico quanto ao teste estatístico (teste de postos com sinais de Wilcoxon, um teste não paramétrico apropriado para comparações pareadas sobre scores por consulta com distribuição não normal) e o nível de significância requerido. H1 é confirmada apenas se a melhor configuração híbrida superar *todos* os baselines isolados no nível de significância especificado; um resultado significativo contra um baseline mas não contra o outro constituiria um resultado parcial.

Esta hipótese também serve a um propósito metodológico: ela instancia o princípio de que abordagens semânticas devem sempre ser comparadas contra um baseline forte de BM25 antes de serem aceitas como melhorias (Robertson et al., 1996; Luan et al., 2021). No domínio em estudo, o BM25 é um baseline não trivial porque conversas de atendimento ao cliente contêm marcadores lexicais discriminativos — vocabulário de cancelamento, vocabulário de reclamação, frases de saudação — que o BM25 captura direta e economicamente.

**Critério de sucesso:** MRR da melhor configuração híbrida > MRR do BM25 e > MRR do ANN, com teste de postos com sinais de Wilcoxon p < 0,05 para ambas as comparações, computado sobre scores MRR por consulta no conjunto de teste reservado.

### H2 — Features Combinadas Lexicais e de Embeddings Superam Features Apenas Lexicais

> Um classificador supervisionado (LightGBM) treinado com features lexicais combinadas e features densas de embeddings extraídas de um encoder multilíngue congelado alcança Macro-F1 estritamente superior ao melhor classificador treinado apenas com features lexicais, com significância estatística em α = 0,05, avaliado sobre cinco seeds aleatórias no conjunto de teste reservado.

Esta hipótese testa o valor aditivo de embeddings densos sobre representações lexicais em um cenário de classificação supervisionada. Ela operacionaliza o princípio "embeddings representam, classificadores decidem" (AnthusAI, 2024): embeddings são usados como features de entrada para um classificador supervisionado, não como um classificador direto baseado em similaridade. Essa escolha de projeto preserva a capacidade discriminativa do aprendizado supervisionado enquanto aproveita a generalização semântica de representações pré-treinadas. A hipótese especificamente requer que o sistema combinado supere *todas* as configurações apenas-lexicais (incluindo Regressão Logística, LightGBM e MLP treinados apenas com features lexicais), não meramente o baseline lexical mais fraco.

A escolha de Macro-F1 como métrica primária reflete a natureza multiclasse do problema e a importância do desempenho por classe: Macro-F1 pondera cada classe igualmente independentemente da frequência, garantindo que o desempenho em classes minoritárias (como "saudacao" e "compra" no dataset pós-auditoria) não seja ocultado pela acurácia agregada em classes majoritárias.

**Critério de sucesso:** Macro-F1 da melhor configuração combinada (lexical + embedding) > Macro-F1 da melhor configuração apenas-lexical, com teste de postos com sinais de Wilcoxon p < 0,05, computado sobre cinco execuções com seeds aleatórias no conjunto de teste reservado.

### H3 — Regras Determinísticas como Features Melhoram a Classificação em Relação ao ML Isolado

> Adicionar ativações de regras determinísticas como features ao pipeline de classificação supervisionada (integração regras-como-features) alcança Macro-F1 estritamente superior ao baseline apenas-ML (classificador sem features de regras), com significância estatística em α = 0,05, avaliado sobre predições por instância no conjunto de teste reservado.

Esta hipótese aborda a complementaridade de regras determinísticas e classificadores estatísticos. A estratégia de integração "regras-como-features" — na qual as saídas de ativação de regras são incluídas como features binárias no vetor de features do classificador — é uma abordagem de integração suave que permite ao classificador aprender quando confiar nos sinais das regras versus quando sobrescrevê-los, em oposição a uma estratégia de sobrescrita rígida (ML+Regras-override) que impõe as saídas das regras incondicionalmente. A hipótese prevê que a integração suave seja mensuravelmente superior à ausência de integração.

H3 também serve ao objetivo de explicabilidade da dissertação: mesmo um pequeno efeito positivo das regras na acurácia de classificação, combinado com a evidência rastreável que cada execução de regra produz, apoiaria o argumento de que regras determinísticas fornecem valor além do que pode ser capturado pelo pipeline estatístico isoladamente.

**Critério de sucesso:** Macro-F1 do ML+Regras-features > Macro-F1 do apenas-ML, com teste de postos com sinais de Wilcoxon p < 0,05, computado sobre predições por instância no conjunto de teste reservado.

### H4 — Inferência em Cascata Reduz Custo Computacional com Perda de Qualidade Limitada

> Um pipeline de inferência em cascata de dois estágios — aplicando um classificador leve (apenas features lexicais) no Estágio 1 e o classificador completo (features lexicais + embedding + regras) no Estágio 2, com saída antecipada baseada em confiança — alcança simultaneamente: (a) pelo menos 40% de redução no custo computacional médio por janela em comparação ao baseline uniforme de pipeline completo, e (b) degradação de Macro-F1 inferior a 2 pontos percentuais em comparação ao baseline uniforme.

Esta hipótese testa a premissa central da inferência em cascata: que a saída antecipada em thresholds de confiança mais baixos permite ao sistema processar uma fração significativa dos casos economicamente sem degradar proporcionalmente a qualidade da classificação. O critério é deliberadamente conjuntivo — tanto a condição de redução de custo quanto a de preservação de qualidade devem ser satisfeitas simultaneamente para que H4 seja confirmada. Uma configuração que alcance 40% de redução de custo com 5% de degradação de F1 não confirma H4; tampouco uma configuração que alcance 0,5% de degradação de F1 com 5% de redução de custo.

O threshold de 40% reflete a economia operacional de implantações de contact center em larga escala: uma redução de custo inferior a esta não justificaria a complexidade arquitetural adicional de manter e calibrar dois classificadores separados. O limite de 2 pontos percentuais de degradação de F1 garante que o ganho de eficiência não seja alcançado sacrificando a acurácia de classificação em uma escala que teria consequências operacionais.

Esta é a mais exigente das quatro hipóteses, por uma razão estrutural observada antes do início dos experimentos: se o classificador leve do Estágio 1 opera sobre os mesmos dados que o Estágio 2, e se o Estágio 2 tem desempenho substancialmente melhor que o Estágio 1 (como deve ter, para motivar o design em cascata), então qualquer fração significativa de casos resolvidos pelo Estágio 1 introduz erros de classificação. A hipótese prevê que esses erros estão limitados dentro do limite aceitável.

**Critério de sucesso:** Existe pelo menos uma configuração de threshold de confiança tal que: custo médio por janela < 0,6 × custo do baseline uniforme E Macro-F1 > Macro-F1 do baseline uniforme − 0,02, ambos avaliados no conjunto de teste reservado.

---

## 1.4 Objetivos

### 1.4.1 Objetivo Geral

Projetar, implementar e avaliar empiricamente uma arquitetura híbrida em cascata para classificação de intenções e recuperação em conversas de atendimento ao cliente em português brasileiro, integrando recuperação lexical, recuperação semântica densa, classificação supervisionada com features heterogêneas e um motor de regras determinístico com evidência rastreável, com relato transparente tanto de confirmações quanto de refutações.

### 1.4.2 Objetivos Específicos

**OE1 — Sistema de Recuperação Híbrida.** Implementar um sistema de recuperação híbrida que combina BM25 com normalização sensível a acentos para PT-BR e busca por vizinhos mais próximos aproximados sobre embeddings de um encoder multilíngue congelado (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensões), com estratégias paramétricas de fusão de scores (combinação linear ponderada e Reciprocal Rank Fusion), e avaliar seu desempenho de recuperação contra baselines isolados lexicais e semânticos nas métricas MRR, Precision@K e nDCG no corpus pós-auditoria.

**OE2 — Representações Conversacionais Multi-Nível.** Projetar e implementar uma abordagem de janela de contexto deslizante que constrói representações conversacionais multi-nível nas granularidades de turno, janela e conversa, e avaliar a contribuição de cada família de features (features de embedding, features lexicais, features de regras, features estruturais) por meio de ablação sistemática no corpus pós-auditoria.

**OE3 — Motor de Regras Semânticas com Evidência Rastreável.** Projetar e implementar um motor de regras semânticas baseado em uma linguagem de domínio específico (DSL) compilada para uma árvore de sintaxe abstrata (AST), suportando predicados lexicais, semânticos, estruturais e contextuais com avaliação de curto-circuito ordenada por custo do predicado, onde cada execução de regra produz metadados de evidência rastreável (termos correspondidos, scores de similaridade, thresholds aplicados, versão do modelo), e avaliar a contribuição do motor de regras ao desempenho de classificação em ambos os modos de integração suave (regras-como-features) e rígida (regras-como-override).

**OE4 — Análise de Inferência em Cascata.** Projetar e avaliar um pipeline de inferência em cascata de dois estágios com saída antecipada baseada em confiança, analisando o trade-off entre redução de custo computacional e degradação de qualidade de classificação ao longo de uma faixa de thresholds de confiança, e identificando se alguma configuração alcança o critério conjuntivo de H4 (≥40% de redução de custo com <2pp de degradação de F1).

---

## 1.5 Contribuições

Esta dissertação apresenta quatro contribuições. Delimitamos cada contribuição cuidadosamente para refletir o que a evidência sustenta.

### C1 — Um Pipeline Híbrido de PLN Completo e de Qualidade de Produção para Dados Conversacionais

O TalkEx é um pipeline de PLN totalmente implementado, documentado e testado para classificação de intenções e recuperação em conversas. O pipeline compreende: ingestão de dados com suporte a múltiplas fontes; segmentação de turnos e normalização de texto com tratamento de acentos específico para PT-BR; construção de janela de contexto deslizante com parâmetros configuráveis; geração de embeddings multi-nível usando um encoder multilíngue congelado; indexação BM25 com normalização sensível a acentos; indexação por vizinhos mais próximos aproximados usando FAISS; recuperação híbrida com fusão de scores; classificação supervisionada com LightGBM, Regressão Logística e MLP sobre features heterogêneas; e um motor de regras semânticas com compilação de DSL para AST. O sistema é implementado em Python 3.11 com modelos de dados Pydantic estritos, 170 arquivos-fonte e aproximadamente 1.900 testes unitários e de integração.

A contribuição arquitetural não é meramente a soma desses componentes, mas as decisões de projeto específicas que governam sua integração: o esquema de representação multi-nível que mapeia conversas para vetores de features nas granularidades de turno, janela e conversa; a parametrização de fusão de scores que possibilita a comparação sistemática de estratégias de fusão linear e baseada em rank; e a escolha de projeto de encoder congelado que desacopla a qualidade da representação semântica da disponibilidade de dados de treinamento específicos do domínio — uma consideração particularmente importante para línguas de baixos recursos e domínios especializados onde dados de fine-tuning são escassos. Essas decisões de projeto estão documentadas em quatro Architecture Decision Records (ADRs) incluídos como parte dos artefatos da dissertação.

Até onde sabemos, nenhum sistema open-source anterior combina recuperação híbrida, embeddings conversacionais multi-nível, classificação supervisionada com features heterogêneas e um motor de regras baseado em DSL em uma única implementação testada e pronta para produção para dados conversacionais em português brasileiro.

### C2 — Um Motor de Regras Semânticas Baseado em DSL com Evidência por Decisão

O motor de regras do TalkEx introduz uma linguagem de domínio específico para expressar regras de classificação e detecção, compilada para uma árvore de sintaxe abstrata avaliada com execução de curto-circuito ordenada por custo do predicado. O motor de regras suporta quatro famílias de predicados: lexicais (contains, regex, threshold de score BM25), semânticos (similaridade de embeddings, score de intent), estruturais (papel do falante, posição do turno, canal da conversa) e contextuais (repetição de padrão dentro de uma janela, sequenciamento temporal de eventos). Cada avaliação de predicado produz metadados de evidência estruturada — o texto específico correspondido, os scores de similaridade computados, os thresholds aplicados e a versão do modelo utilizada — garantindo total auditabilidade de cada decisão baseada em regras.

Esta contribuição endereça diretamente a lacuna de explicabilidade identificada na formulação do problema. Diferentemente de classificadores estatísticos, que produzem scores de probabilidade sem rastros de decisão interpretáveis, e diferentemente de abordagens de classificação baseadas em LLM (Huang e He, 2025), que não fornecem evidência por instância, o motor de regras do TalkEx produz um rastro de auditoria de decisão completo para cada classificação que influencia. Em contextos de implantação sensíveis à conformidade, essa propriedade é operacionalmente necessária e não meramente desejável.

A contribuição é delimitada honestamente: em nossos experimentos, o motor de regras foi avaliado com um conjunto limitado de regras direcionadas principalmente a padrões lexicais em duas classes de intent de alta frequência. Os experimentos, portanto, testam uma condição necessária mas não suficiente para o design completo: que mesmo um conjunto pequeno e predominantemente lexical de regras produz benefícios mensuráveis de classificação quando integrado como features suaves. A expressividade mais ampla do motor de regras — predicados de similaridade semântica, predicados de sequenciamento contextual — é demonstrada através da implementação, mas não é totalmente avaliada nos resultados experimentais aqui apresentados.

### C3 — Evidência Empírica sobre Recuperação Híbrida e Combinação de Features em Classificação Conversacional em PT-BR

Esta dissertação contribui evidência empírica sistemática sobre a efetividade relativa da recuperação lexical, recuperação semântica e sua combinação no domínio de classificação conversacional em PT-BR. Os resultados experimentais pós-auditoria — baseados em um corpus de 2.122 conversas com 8 classes de intent, avaliados sobre cinco seeds aleatórias, com testes estatísticos de postos com sinais de Wilcoxon — fornecem os seguintes achados:

- A recuperação híbrida (Hybrid-LINEAR-α=0.30, MRR=0,853) supera estatisticamente de forma significativa tanto o BM25 (MRR=0,835, p=0,017) quanto a recuperação apenas-ANN (MRR=0,824, p=0,030), confirmando H1.
- Features combinadas lexicais e de embeddings (LightGBM, Macro-F1=0,722) superam estatisticamente de forma significativa features apenas-lexicais (LightGBM, Macro-F1=0,334), confirmando H2 com uma margem de 38,8 pontos percentuais.
- Um estudo de ablação isola a contribuição de cada família de features: embeddings contribuem +33,0 pontos percentuais para o Macro-F1 (o componente dominante), features lexicais contribuem adicionalmente +2,9 pontos percentuais, features de regras contribuem +1,8 pontos percentuais e features estruturais contribuem +1,3 pontos percentuais.
- A integração regras-como-features produz uma direção positiva (+1,8pp Macro-F1) que não alcança significância estatística (p=0,131), deixando H3 inconclusiva.
- Nenhuma configuração em cascata alcança o critério conjuntivo de H4: todas as configurações em cascata aumentam o custo medido em vez de reduzi-lo, refutando H4.

Esses achados contribuem tanto evidência positiva quanto negativa. A refutação de H4 e a inconclusividade de H3 são reportadas com a mesma proeminência que as confirmações de H1 e H2. Resultados negativos e inconclusivos são informativos para a comunidade de pesquisa e fornecem uma representação mais acurada do estado atual da arquitetura do que um relato seletivo de confirmações.

### C4 — Um Protocolo de Auditoria de Dados para Corpora Conversacionais Sintéticos

Os resultados experimentais nesta dissertação foram obtidos sobre um corpus que passou por uma rigorosa auditoria em dois estágios antes de ser utilizado para teste de hipóteses. O protocolo de auditoria inclui: deduplicação exata com threshold de similaridade de cosseno de 0,97; detecção de quase-duplicatas com threshold de 0,92; detecção de contaminação few-shot entre splits de treinamento e teste; revisão humana sistemática de atribuições de classe ambíguas com taxa de aceitação de rótulo confirmada de ≥96,7%; e remoção de uma classe de intent inteira ("outros") considerada heterogênea demais para classificação supervisionada confiável.

O protocolo reduziu o corpus de 2.257 registros pré-auditoria (9 classes) para 2.122 registros pós-auditoria (8 classes) e alterou substancialmente as conclusões experimentais: sob os dados pré-auditoria, H1 não era estatisticamente significativa (p=0,103); sob os dados pós-auditoria, H1 é confirmada (p=0,017). Essa mudança de conclusão sob condições experimentais idênticas em outros aspectos ilustra o impacto concreto da qualidade dos dados nos resultados de avaliação de PLN.

Documentamos o protocolo de auditoria completo no Capítulo 5 (Metodologia) e disponibilizamos o código de auditoria e o rastro de decisões como material suplementar. Este protocolo é reutilizável para qualquer pesquisador trabalhando com corpora conversacionais sintéticos ou semi-sintéticos e contribui para a infraestrutura de reprodutibilidade do campo.

---

## 1.6 Organização da Dissertação

O restante desta dissertação está organizado em seis capítulos.

**Capítulo 2 — Fundamentação Teórica** estabelece o arcabouço conceitual subjacente ao trabalho. Apresentamos a representação de texto como vetores densos e a progressão de bag-of-words a sentence transformers; o modelo de recuperação lexical BM25 com sua formulação matemática; busca por vizinhos mais próximos aproximados para recuperação de vetores densos; estratégias de recuperação híbrida e métodos de fusão de scores; classificação supervisionada de texto com conjuntos heterogêneos de features; estrutura conversacional multi-turno e modelagem de janela de contexto; projeto de linguagem de domínio específico e avaliação de árvore de sintaxe abstrata; e o princípio de inferência em cascata em sistemas de PLN. Este capítulo é didático — fornece o vocabulário fundamental para as contribuições descritas nos capítulos subsequentes.

**Capítulo 3 — Trabalhos Relacionados** posiciona esta pesquisa no estado da arte. Analisamos criticamente os trabalhos anteriores mais relevantes: Harris (2025) sobre busca lexical versus semântica em domínios estruturados; Rayo et al. (2025) sobre recuperação híbrida para texto regulatório; Huang e He (2025) sobre clustering de texto baseado em LLM como classificação; Lyu et al. (2025) sobre mecanismos de atenção para classificação de texto; o estudo AnthusAI sobre classificação semântica de texto com embeddings como features para classificadores supervisionados; e a literatura sobre inteligência conversacional em ambientes de contact center, incluindo plataformas comerciais (Observe.AI, CallMiner, Verint) e sistemas acadêmicos (BERTaú). Concluímos o capítulo com uma tabela de posicionamento explícita identificando a lacuna que esta dissertação preenche: nenhum trabalho anterior combina recuperação híbrida, representações conversacionais multi-nível, classificação supervisionada com features heterogêneas e um motor de regras determinístico baseado em DSL em um único sistema avaliado para dados conversacionais em PT-BR.

**Capítulo 4 — Arquitetura Proposta: TalkEx** descreve o sistema em detalhe técnico suficiente para reprodução. Apresentamos o pipeline completo com limites de componentes e fluxo de dados; o modelo de dados conversacional (Conversation, Turn, ContextWindow, EmbeddingRecord, Prediction, RuleExecution como modelos Pydantic estritos e congelados); o módulo de normalização de texto com design específico para PT-BR; o esquema de representação multi-nível; o sistema de recuperação híbrida com variantes de fusão de scores; o pipeline de classificação supervisionada com engenharia de features; o motor de regras (gramática DSL, parser, representação AST, handlers de predicados, executor de curto-circuito, geração de evidência); a lógica de inferência em cascata; e as decisões arquiteturais documentadas em ADR-001 a ADR-004.

**Capítulo 5 — Design Experimental e Metodologia** detalha o protocolo experimental para cada hipótese. Descrevemos as estatísticas do corpus pós-auditoria e o protocolo de auditoria; os splits experimentais e a estratégia de seeds; as métricas de avaliação para recuperação (MRR, Precision@K, nDCG), classificação (Macro-F1, F1 por classe) e eficiência em cascata (custo por janela, percentual de redução de custo); o protocolo experimental para cada uma de H1 a H4; o design do estudo de ablação; a abordagem de análise estatística (teste de postos com sinais de Wilcoxon com intervalos de confiança por bootstrap); e as ameaças à validade, incluindo a natureza sintética do corpus, a avaliação em domínio único, a ausência de baselines com encoder fine-tuned e o artefato de determinismo na avaliação com múltiplas seeds.

**Capítulo 6 — Resultados e Análise** apresenta os resultados experimentais para cada hipótese e o estudo de ablação, analisa o desempenho por classe, discute os resultados da cascata e por que H4 foi refutada, e contextualiza os achados em relação aos trabalhos relacionados. Incluímos uma análise de erros por classe que identifica as duas classes de intent (saudacao e compra) onde todas as configurações apresentam F1 abaixo de 0,50, e discutimos em que medida isso pode refletir propriedades dos dados sintéticos de treinamento.

**Capítulo 7 — Conclusão** sintetiza os achados confirmados e refutados, articula as limitações do trabalho atual e identifica as direções de maior prioridade para investigação futura: fine-tuning do encoder de embeddings para o domínio conversacional; expansão do motor de regras para incluir predicados de similaridade semântica e sequenciamento contextual; avaliação leave-one-domain-out para testar generalização entre domínios; validação cruzada em substituição a splits fixos para intervalos de confiança mais estreitos; e métricas de calibração (Brier score, Expected Calibration Error) para suportar abstenção baseada em threshold em implantações de produção.

---

## Referências deste Capítulo

As seguintes referências são citadas neste capítulo. Detalhes bibliográficos completos são fornecidos na lista de referências da dissertação.

Associação Brasileira de Telesserviços. (2023). *Relatório Anual do Setor de Contact Center*. ABT.

AnthusAI. (2024). *Semantic Text Classification: Text Classification with Various Embedding Techniques*. GitHub repository. Retrieved from https://github.com/AnthusAI/semantic-text-classification.

Chiticariu, L., Li, Y., and Reiss, F. R. (2013). Rule-based information extraction is dead! Long live rule-based information extraction systems! In *Proceedings of EMNLP 2013* (pp. 827–832).

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186).

Harris, L. (2025). Comparing lexical and semantic vector search methods when classifying medical documents. arXiv preprint arXiv:2505.11582v2.

Huang, C., and He, G. (2025). Text clustering as classification with LLMs. In *Proceedings of SIGIR-AP 2025*. arXiv preprint arXiv:2410.00927v3.

Liu, T.-Y. (2011). *Learning to Rank for Information Retrieval*. Springer.

Lyu, N., Wang, Y., Chen, F., and Zhang, Q. (2025). Advancing text classification with large language models and neural attention mechanisms. arXiv preprint arXiv:2512.09444v1.

Rayo, J., de la Rosa, R., and Garrido, M. (2025). A hybrid approach to information retrieval and answer generation for regulatory texts. In *Proceedings of COLING 2025*. arXiv preprint arXiv:2502.16767v1.

Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019* (pp. 3982–3992).

Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., and Gatford, M. (1996). Okapi at TREC-4. In *Proceedings of the Fourth Text REtrieval Conference (TREC-4)* (pp. 73–96).

Viola, P., and Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. In *Proceedings of CVPR 2001* (Vol. 1, pp. I–511–I–518).

---

*Contagem de palavras do capítulo (aproximada): 5.400 palavras, equivalente a aproximadamente 18 páginas no formato padrão de espaçamento duplo ACL.*
