# Capítulo 1 — Introdução

## 1.1 Contexto e Motivação

O setor de atendimento ao cliente no Brasil movimenta bilhões de interações por ano. Operações de grande porte em telecomunicações, serviços financeiros e varejo processam individualmente entre 5 e 15 milhões de conversas por mês, distribuídas entre canais de voz, chat, e-mail e redes sociais [Associação Brasileira de Telesserviços, 2023]. Cada uma dessas conversas contém sinais sobre intenções do consumidor, níveis de satisfação, riscos de churn, oportunidades de venda e potenciais violações regulatórias. Em conjunto, esse volume constitui um dos maiores corpus de linguagem natural gerado em território brasileiro — e, paradoxalmente, um dos menos explorados.

A perda de valor informacional nesse cenário é sistemática. Estimativas da indústria indicam que menos de 5% das conversas de atendimento recebem alguma forma de análise além da classificação manual pelo agente ao final da chamada — uma categorização que é sabidamente inconsistente, superficial e enviesada pela pressão de tempo médio de atendimento (TMA). O restante das conversas permanece como dado bruto, inacessível para busca, classificação ou auditoria em escala. O resultado é um paradoxo operacional: organizações investem pesadamente na captura de conversas (infraestrutura de telefonia, sistemas de gravação, plataformas de chat) mas extraem uma fração ínfima do valor informacional contido nelas.

As abordagens atualmente disponíveis para análise de conversas em escala apresentam limitações fundamentais que motivam esta pesquisa.

**Classificação superficial de intents.** A maioria dos sistemas comerciais de conversation intelligence opera com classificação de turno único — cada mensagem é classificada isoladamente, sem considerar o contexto conversacional. Essa abordagem ignora que a intenção do cliente frequentemente emerge de múltiplos turnos: uma reclamação pode começar como dúvida, escalar após informação insatisfatória, e culminar em pedido de cancelamento. Classificar cada turno isoladamente perde essas transições.

**Busca puramente lexical ou puramente semântica.** Sistemas de busca sobre conversas tipicamente adotam um único paradigma. Abordagens lexicais, baseadas em correspondência de termos como BM25 [Robertson et al., 1996], são rápidas e interpretáveis, mas falham diante de paráfrases e intenções implícitas — um cliente que diz "não quero mais esse serviço" não será encontrado por uma busca pelo termo "cancelamento". Abordagens semânticas, baseadas em embeddings densos [Reimers e Gurevych, 2019], capturam similaridade de significado, mas são computacionalmente caras, menos interpretáveis e surpreendentemente frágeis em domínios com vocabulário técnico ou códigos específicos — onde a correspondência lexical exata é insubstituível. Harris (2025) demonstrou que BM25 supera embeddings semânticos off-the-shelf na classificação de documentos médicos estruturados, evidenciando que a superioridade semântica não é universal.

**Ausência de auditabilidade.** Sistemas baseados exclusivamente em modelos de aprendizado de máquina produzem predições sem evidência rastreável. Quando um classificador neural rotula uma conversa como "risco de churn", não há como o analista de qualidade ou o auditor de compliance verificar *por que* aquela decisão foi tomada. Em domínios regulados — como telecomunicações (Anatel), serviços financeiros (Bacen) e saúde suplementar (ANS) — essa opacidade é um impedimento operacional concreto. Decisões baseadas em modelos black-box não podem ser auditadas, contestadas ou explicadas ao consumidor, conforme exigido por marcos regulatórios crescentes.

**Custo computacional uniforme.** Pipelines tradicionais de NLP aplicam o mesmo nível de processamento a todas as conversas, independentemente de sua complexidade. Uma saudação simples ("Oi, quero saber o saldo") recebe o mesmo investimento computacional que uma negociação complexa de retenção com 20 turnos. Essa uniformidade é ineficiente: a maior parte das conversas pode ser resolvida com processamento leve, e apenas uma fração exige inferência sofisticada.

A relevância prática desta pesquisa estende-se a quatro dimensões operacionais. Em **compliance**, regras auditáveis permitem detectar menções a órgãos reguladores (Procon, Anatel, Reclame Aqui) e gerar alertas com evidência rastreável. Em **retenção**, a identificação precoce de sinais de churn em contexto multi-turn permite intervenção proativa. Em **qualidade operacional**, a classificação automatizada e consistente substitui a categorização manual e inconsistente do agente. Em **experiência do cliente (CX)**, a busca híbrida permite localizar conversas similares para análise de padrões e treinamento de equipes.

## 1.2 Problema de Pesquisa

Diante das limitações expostas, formulamos a seguinte pergunta central de pesquisa:

> **Como combinar retrieval lexical, retrieval semântico e regras determinísticas em uma arquitetura unificada que maximize qualidade, explicabilidade e eficiência na análise de conversas de atendimento?**

Esta pergunta encapsula três tensões fundamentais que permeiam o campo de processamento de linguagem natural aplicado a conversas:

**A tensão entre cobertura lexical e generalização semântica.** Busca lexical (BM25) e busca semântica (embeddings densos) operam sobre sinais complementares. BM25 é preciso para termos exatos, códigos de produto e vocabulário técnico, mas incapaz de capturar paráfrases. Embeddings semânticos generalizam para variações de linguagem, mas são computacionalmente caros e podem perder termos críticos de domínio. Rayo et al. (2025) demonstraram que a fusão híbrida (BM25 + embeddings fine-tuned com peso alfa=0,65) supera ambas as abordagens isoladas em retrieval de textos regulatórios, alcançando Recall@10 de 0,83 contra 0,76 (BM25) e 0,81 (semântico puro). Porém, esse resultado foi obtido em textos longos e formais — textos regulatórios europeus — e não em conversas informais multi-turn. A transferência dessa conclusão para o domínio conversacional brasileiro não é trivial.

**A tensão entre granularidade e contexto.** Conversas de atendimento possuem uma estrutura hierárquica natural: turnos individuais, janelas de contexto (sequências de turnos adjacentes) e a conversa completa. Cada nível de granularidade captura informações distintas. O turno individual captura a intenção local ("quero cancelar"). A janela de contexto captura dependências multi-turn ("primeiro reclamou, depois pediu supervisor, depois ameaçou cancelar"). A conversa completa captura o desfecho global e o tom dominante. Classificadores que operam em um único nível perdem informação estrutural. Lyu et al. (2025) mostraram que mecanismos de atenção melhoram a classificação de textos (F1 de 0,89 contra 0,86 do BERT base), mas avaliaram apenas textos curtos de notícias (AG News), não conversas multi-turn.

**A tensão entre automação estatística e auditabilidade determinística.** Modelos de aprendizado de máquina aprendem padrões dos dados mas não oferecem garantias formais nem evidência rastreável. Regras determinísticas oferecem previsibilidade e auditabilidade, mas dependem de conhecimento explicitamente codificado e não generalizam para padrões não previstos. Huang e He (2025) demonstraram que LLMs podem transformar clustering em classificação com desempenho próximo ao upper bound teórico, mas dependem de inferência online com LLM — custo proibitivo em escala — e não fornecem rastreabilidade de evidência. Chiticariu et al. (2013) argumentaram que sistemas baseados em regras, quando bem projetados, complementam abordagens estatísticas em cenários que exigem transparência e controle. Nenhum trabalho existente integra as duas abordagens em dados conversacionais.

Além dessas tensões fundamentais, o domínio de conversas de atendimento impõe desafios específicos que amplificam a dificuldade do problema:

**Texto ruidoso proveniente de transcrição automática (ASR).** Conversas de voz passam por sistemas de reconhecimento automático de fala que introduzem erros sistemáticos: palavras substituídas, segmentação incorreta de turnos, ausência de pontuação e capitalização. O classificador e o sistema de busca precisam ser robustos a esse ruído, o que exige normalização agressiva sem perda de sinais discriminativos.

**Dependências multi-turn.** A intenção do cliente frequentemente não está contida em um único turno. Padrões como objeção-após-oferta, escalação progressiva e mudança de tópico são fenômenos inerentemente multi-turn. Sistemas que classificam turnos isolados perdem esses padrões. A construção de janelas de contexto — sequências deslizantes de N turnos adjacentes — é necessária para capturar essas dependências, mas introduz questões de configuração (tamanho da janela, sobreposição, ponderação por recência) que afetam diretamente a qualidade.

**Variação linguística do português brasileiro.** O PT-BR falado em atendimento apresenta alta variabilidade: abreviações ("vc", "td", "blz", "pq"), gírias regionais, ausência inconsistente de diacríticos ("nao" vs "não", "numero" vs "número"), coloquialismos e code-switching com termos em inglês ("feedback", "upgrade", "app"). Essa variabilidade exige normalização textual especializada — particularmente a remoção de diacríticos (Unicode NFD) para garantir que "cancelamento" e "cancelámento" sejam tratados como equivalentes em buscas e regras lexicais.

**Exigência simultânea de qualidade, explicabilidade e custo controlado.** Em ambientes de produção com milhões de conversas, não basta um sistema que classifique bem — ele precisa classificar bem, explicar por que classificou de determinada forma, e fazer isso a um custo computacional viável. Essas três exigências frequentemente conflitam: modelos mais sofisticados (cross-encoders, LLMs) melhoram a qualidade mas aumentam o custo e reduzem a interpretabilidade. A arquitetura precisa balancear essas três dimensões simultaneamente.

## 1.3 Hipóteses

A partir do problema de pesquisa e da análise da literatura, formulamos quatro hipóteses falsificáveis, cada uma com critérios quantitativos de confirmação:

**H1 — Superioridade do retrieval híbrido sobre abordagens isoladas.**

> O retrieval híbrido (BM25 + busca por vizinhos mais próximos com fusão de scores) supera tanto o BM25 isolado quanto a busca semântica isolada em Recall@K, MRR e nDCG quando aplicado a conversas de call center em PT-BR.

Esta hipótese fundamenta-se na evidência de Rayo et al. (2025), que demonstraram superioridade híbrida em textos regulatórios, e na observação de Harris (2025) de que BM25 é competitivo mesmo em cenários favoráveis a abordagens semânticas. Propomos testar se a complementaridade se mantém no domínio conversacional, onde coexistem termos técnicos exatos (códigos, nomes de planos) e linguagem informal com paráfrases. O critério de confirmação exige que o melhor sistema híbrido supere todos os sistemas isolados em Recall@10 e MAP@10 com significância estatística (p < 0,05, teste de Wilcoxon signed-rank sobre os folds de validação cruzada). Considera-se H1 confirmada apenas se ambas as métricas apresentarem superioridade estatisticamente significativa simultaneamente; caso apenas uma atinja significância, H1 será considerada parcialmente confirmada.

**H2 — Ganho da combinação de features lexicais com embeddings densos.**

> Classificadores que combinam features lexicais com embeddings densos pré-treinados alcançam Macro-F1 superior a classificadores que utilizam apenas features lexicais, com significância estatística (p < 0,05) em pelo menos 60% das classes individuais.

Esta hipótese fundamenta-se na observação de que features lexicais (TF-IDF, contagens de termos, padrões de n-gramas) e embeddings densos pré-treinados (sentence-transformers) capturam sinais complementares sobre o texto. Features lexicais são eficazes para termos discriminativos exatos — códigos de produto, jargão técnico, palavras-chave de compliance — enquanto embeddings densos capturam relações semânticas, paráfrases e variações de linguagem. O princípio "embeddings representam, classificadores decidem" [AnthusAI] é preservado: os embeddings alimentam classificadores supervisionados (regressão logística, gradient boosting, MLP), não são usados diretamente como classificadores via similaridade. O critério de confirmação exige que a melhor configuração combinada (lexical + embeddings) supere todas as configurações apenas lexicais em Macro-F1, com significância estatística (p < 0,05) verificada por teste de Wilcoxon signed-rank sobre os folds de validação cruzada, em pelo menos 60% das classes individuais.

**H3 — Complementaridade das regras determinísticas ao pipeline estatístico.**

> A adição de um motor de regras semânticas (DSL compilada para AST) ao pipeline híbrido melhora a precision em classes críticas sem degradar o recall global, além de fornecer rastreabilidade de evidência por decisão.

Esta hipótese aborda a lacuna de auditabilidade identificada nos trabalhos existentes. Enquanto Huang e He (2025) dependem de LLMs sem rastreabilidade e Harris (2025) e Rayo et al. (2025) operam sem camada de regras, propomos que regras determinísticas expressas em uma DSL (Domain-Specific Language) compilada para árvores sintáticas abstratas (AST) complementam o pipeline estatístico em cenários críticos. Uma regra como `WHEN speaker == "customer" AND contains_any(["cancelar", "encerrar", "desistir"]) THEN tag("cancelamento")` oferece previsibilidade total, evidência rastreável (quais palavras foram encontradas, em qual turno, com qual score) e custo computacional negligível. O critério de confirmação exige melhoria de precision em classes críticas com degradação máxima de 1 ponto percentual em recall global, e que ao menos 80% das decisões em classes críticas produzam evidência rastreável.

**H4 — Eficiência da inferência em cascata.**

> Um pipeline com inferência cascateada — que aplica estágios progressivamente mais caros, permitindo resolução precoce quando a confiança é suficiente — reduz o custo computacional médio por conversa em pelo menos 40% comparado ao pipeline uniforme, com degradação de qualidade inferior a 2 pontos percentuais em F1.

Esta hipótese fundamenta-se no princípio de inferência cascateada [Viola e Jones, 2004], amplamente aplicado em visão computacional e mais recentemente em sistemas de retrieval em escala [Google, Microsoft Bing]. A intuição é que a maioria das conversas pode ser resolvida com processamento leve — uma saudação simples não precisa de embeddings multi-nível e cross-encoder reranking. Propomos um pipeline de 4 estágios: (1) filtros baratos e regras lexicais simples (~1ms), (2) retrieval híbrido com classificador leve (~10-50ms), (3) classificação completa com features heterogêneas e regras semânticas (~50-200ms), e (4) revisão excepcional para casos ambíguos (~500ms-2s). O critério de confirmação exige simultaneamente redução de custo >= 40% e degradação de F1 < 2 pontos percentuais para pelo menos uma configuração de thresholds de confiança. Esta hipótese pressupõe estágios com custos computacionais significativamente distintos — por exemplo, filtros lexicais simples (~1ms) versus classificadores baseados em embeddings (~50-200ms) — de modo que a resolução precoce nos estágios iniciais produza economia mensurável.

As quatro hipóteses foram deliberadamente formuladas com critérios numéricos para permitir refutação clara. H4 é a mais arriscada: se os thresholds de confiança por estágio forem calibrados de forma agressiva, a resolução precoce pode descartar conversas ambíguas e comprometer o recall. Os thresholds constituem o ponto crítico do experimento.

## 1.4 Objetivos

### 1.4.1 Objetivo Geral

Projetar, implementar e avaliar uma arquitetura híbrida em cascata para classificação e retrieval de conversas de atendimento ao cliente, integrando busca lexical, busca semântica e regras determinísticas auditáveis sobre representações conversacionais em múltiplos níveis de granularidade.

### 1.4.2 Objetivos Específicos

**OE1 — Retrieval híbrido.** Implementar um sistema de retrieval híbrido que combine BM25 (com normalização accent-aware para PT-BR) e busca por vizinhos mais próximos sobre embeddings densos, com fusão de scores parametrizável (combinação linear ponderada e Reciprocal Rank Fusion), e avaliar seu desempenho contra baselines lexicais e semânticos isolados em Recall@K, MRR e nDCG.

**OE2 — Representações multi-nível.** Projetar e implementar um esquema de representação conversacional em três níveis de granularidade — turno individual, janela de contexto (sequência deslizante de N turnos) e conversa completa — e avaliar o impacto de cada nível e de suas combinações na classificação supervisionada multi-classe, comparando com representações de nível único.

**OE3 — Motor de regras semânticas.** Projetar e implementar um motor de regras baseado em uma DSL compilada para AST, com suporte a predicados lexicais (correspondência de termos, expressões regulares, scores BM25), semânticos (similaridade de embeddings, scores de intent), estruturais (falante, canal, duração) e contextuais (repetição em janela, sequência temporal), que produza evidência rastreável por decisão e avalie seu efeito como complemento ao pipeline de classificação estatística.

**OE4 — Inferência cascateada.** Projetar e avaliar um pipeline de inferência em cascata com quatro estágios de custo crescente, analisando o trade-off entre redução de custo computacional e degradação de qualidade, e identificando os thresholds de confiança que otimizam a curva de Pareto custo-qualidade.

## 1.5 Contribuições

Esta dissertação apresenta quatro contribuições principais:

**C1 — Framework arquitetural com três paradigmas complementares.** Propomos uma arquitetura que integra, em um único pipeline, retrieval lexical (BM25), retrieval semântico (embeddings densos com busca ANN), classificação supervisionada sobre features heterogêneas e um motor de regras determinísticas. Até onde temos conhecimento, nenhum trabalho anterior combinou esses três paradigmas em dados conversacionais. Os trabalhos mais próximos — Rayo et al. (2025) para retrieval híbrido e Huang e He (2025) para classificação com LLMs — operam em domínios não-conversacionais e sem camada de regras auditáveis.

**C2 — DSL auditável compilada para AST com evidência por decisão.** Propomos uma Domain-Specific Language para expressão de regras de classificação e detecção que compila para árvores sintáticas abstratas com avaliação por short-circuit ordenada por custo de predicado. Cada nó da AST produz metadados de evidência (palavras encontradas, scores de similaridade, thresholds aplicados, versão do modelo), garantindo rastreabilidade completa da cadeia de decisão. Essa contribuição endereça diretamente a lacuna de auditabilidade identificada nos trabalhos existentes e responde a uma necessidade concreta de domínios regulados.

**C3 — Estudo empírico comparativo no domínio conversacional PT-BR.** Conduzimos um estudo experimental sistemático que avalia, no domínio de conversas de atendimento em português brasileiro: (a) retrieval híbrido contra baselines isolados, (b) representações multi-nível contra representações de nível único, (c) pipeline com regras contra pipeline sem regras em classes críticas, e (d) inferência cascateada contra pipeline uniforme. O estudo inclui ablation studies que isolam a contribuição de cada componente. Até onde sabemos, nenhum estudo comparativo abrangente foi conduzido neste domínio específico.

**C4 — Análise quantitativa do trade-off custo-qualidade em inferência cascateada.** Apresentamos uma análise empírica da curva de Pareto entre redução de custo computacional e degradação de qualidade em um pipeline cascateado de 4 estágios, identificando faixas de operação ótimas para diferentes perfis de carga. Essa contribuição é particularmente relevante para implantações em escala, onde o custo de inferência por conversa é uma restrição operacional concreta.

## 1.6 Organização da Dissertação

O restante desta dissertação está organizado em seis capítulos.

O **Capítulo 2 — Fundamentação Teórica** estabelece os conceitos fundamentais que sustentam o trabalho. Apresentamos representação vetorial de texto e a evolução de bag-of-words a sentence transformers; busca lexical com BM25 e sua formulação matemática; busca semântica com embeddings densos e índices ANN; estratégias de fusão de scores para busca híbrida; classificação supervisionada com features heterogêneas; análise de conversas e representação multi-nível; motores de regras baseados em DSL e AST; e o princípio de inferência em cascata.

O **Capítulo 3 — Trabalhos Relacionados** posiciona esta pesquisa no estado da arte. Analisamos criticamente os trabalhos de Harris (2025) sobre busca lexical versus semântica, Rayo et al. (2025) sobre retrieval híbrido em textos regulatórios, Huang e He (2025) sobre clustering como classificação com LLMs, Lyu et al. (2025) sobre mecanismos de atenção para classificação, e o trabalho de AnthusAI sobre embeddings para classificação. Identificamos a lacuna que esta dissertação preenche: nenhum trabalho existente combina retrieval híbrido, classificação multi-nível e regras determinísticas auditáveis em dados conversacionais.

O **Capítulo 4 — Arquitetura Proposta: TalkEx** descreve a arquitetura do sistema em detalhe técnico suficiente para reprodução. Apresentamos o pipeline completo, o modelo de dados conversacional, o módulo de normalização textual para PT-BR, as representações multi-nível, o sistema de retrieval híbrido com fusão de scores, o pipeline de classificação supervisionada, o motor de regras semânticas (DSL, parser, AST, executor com evidência), e a lógica de inferência cascateada.

O **Capítulo 5 — Desenho Experimental e Metodologia** detalha o protocolo experimental para cada hipótese. Descrevemos o dataset (corpus sintético expandido com validação de robustez), as métricas de avaliação (Recall@K, MRR, nDCG para retrieval; Macro-F1, precision, recall, AUC para classificação; precision e cobertura por regra; latência e throughput para eficiência), os protocolos para H1-H4, a análise estatística (Wilcoxon signed-rank, Friedman, bootstrap confidence intervals) e as ameaças à validade.

O **Capítulo 6 — Resultados e Análise** apresenta os resultados experimentais. Reportamos o desempenho do retrieval híbrido contra baselines, o impacto das representações multi-nível na classificação, o efeito das regras determinísticas na precision de classes críticas, e a curva de Pareto custo-qualidade da inferência cascateada. Incluímos análise de erro, ablation studies e discussão comparativa com a literatura.

O **Capítulo 7 — Conclusão** sintetiza as contribuições, discute limitações (natureza sintética do corpus, ausência de ASR real, escala limitada), e delineia trabalhos futuros, incluindo fine-tuning de embeddings para o domínio, active learning para redução de custo de anotação, e validação em produção com shadow deployment.
