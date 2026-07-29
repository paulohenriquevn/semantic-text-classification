PAULO HENRIQUE VIEIRA NASCIMENTO
CAUÊ CAVICHIOLI LEÃO
SUELE SUSAN FEITOSA SOUSA








PROJETO DE PESQUISA
TalkEx: Uma Arquitetura Híbrida Cascateada para Classificação de Intenções em Conversas
Projeto de Pesquisa apresentado ao
Programa de Capacitação e Formação do
Centro de Competências Embrapii em
Tecnologias Imersivas (AKCIT) como
parte dos requisitos para elaboração do
Trabalho de Conclusão de Curso.








GOIÂNIA
2025

SUMÁRIO
1 INTRODUÇÃO    3
2 PERGUNTA DE PESQUISA    4
3 HIPÓTESE    4
4 JUSTIFICATIVA    5
5 OBJETIVO GERAL    6
   5.1 Objetivos específicos    6
6 MÉTODO PRELIMINAR    7
7 RESULTADOS ESPERADOS    8
8 CRONOGRAMA DAS ATIVIDADES    9
REFERÊNCIAS    10

1 INTRODUÇÃO

Operações de atendimento ao cliente no Brasil geram milhões de conversas mensais por meio de canais de voz, chat, e-mail e redes sociais. Estimativas da indústria indicam que menos de 5% dessas conversas recebem tratamento analítico além dos códigos de disposição manuais dos agentes — uma categorização amplamente reconhecida como inconsistente e de baixa granularidade. Os 95% restantes persistem como dados não estruturados, inacessíveis para busca, classificação ou análise de tendências em escala.

Essa lacuna analítica tem consequências concretas: equipes de compliance não conseguem auditar o comportamento dos agentes em escala, sinais de churn que abrangem múltiplos turnos passam despercebidos, e a garantia de qualidade depende de códigos auto-reportados não confiáveis. Sistemas automatizados de inteligência conversacional que classifiquem intenções, recuperem interações similares e produzam evidências auditáveis para cada decisão atenderiam diretamente a essas necessidades operacionais.

A classificação de intenções (do inglês intent classification ou intent detection) é a tarefa de PLN que consiste em mapear automaticamente um enunciado — ou um diálogo multi-turno — a uma categoria semântica predefinida que representa o objetivo comunicativo do usuário (LARSON et al., 2019; ZHANG et al., 2021; FARFAN-ESCOBEDO; DOS REIS, 2024). Em domínios de atendimento ao cliente, essas categorias correspondem a ações ou demandas típicas — como cancelamento de contrato, registro de reclamação ou pedido de suporte técnico. A complexidade da tarefa aumenta em diálogos multi-turno, nos quais a intenção pode estar distribuída ao longo de múltiplos turnos e qualificada por mudanças contextuais de tom, histórico e papel do locutor (ZHANG et al., 2021; FARFAN-ESCOBEDO; DOS REIS, 2024), o que torna insuficientes os benchmarks baseados em enunciados isolados.

Três paradigmas de PLN (Processamento de Linguagem Natural) oferecem capacidades complementares para essa tarefa. A recuperação lexical por meio do algoritmo BM25 destaca-se na correspondência exata de termos — nomes de produtos, palavras-chave regulatórias, verbos de cancelamento — mas falha em paráfrases e intenção implícita. Embeddings densos de codificadores de sentenças pré-treinados capturam similaridade semântica e invariância a paráfrases, mas têm dificuldade com vocabulário específico do domínio. Por fim, regras determinísticas fornecem precisão controlável e evidência auditável para padrões conhecidos, mas não conseguem generalizar além de sua cobertura lexical.

Esses paradigmas são predominantemente estudados de forma isolada. Trabalhos de recuperação híbrida focam em benchmarks de recuperação de passagens em inglês (FORMAL et al., 2021; KHATTAB; ZAHARIA, 2020; KARPUKHIN et al., 2020). Classificação baseada em embeddings tipicamente realiza fine-tuning dos codificadores (TUNSTALL et al., 2022; REIMERS; GUREVYCH, 2019). Motores de regras são utilizados de forma independente ou completamente substituídos por sistemas neurais. A integração dos três paradigmas dentro de um pipeline unificado, avaliado sobre dados conversacionais multi-turno em língua portuguesa, permanece inexplorada.

Nesse contexto, o presente projeto propõe o TalkEx — uma arquitetura modular híbrida cascateada que combina BM25, embeddings multilíngues congelados, classificação supervisionada com LightGBM e um motor de regras semânticas compilado a partir de uma linguagem de domínio específico (DSL). O sistema é avaliado experimentalmente sobre um corpus curado de 2.122 registros de atendimento ao cliente em português brasileiro (PT-BR), abrangendo 8 classes de intenção, com metodologia experimental rigorosa baseada em múltiplas sementes aleatórias e testes estatísticos.

2 PERGUNTA DE PESQUISA

Como a combinação de recuperação lexical (BM25), embeddings multilíngues congelados e regras determinísticas em um pipeline híbrido cascateado pode aprimorar a classificação automática de intenções em conversas de atendimento ao cliente em português brasileiro, em comparação com abordagens isoladas?

3 HIPÓTESE

O estudo formula quatro hipóteses centrais, derivadas da pergunta de pesquisa:

- H1 — Recuperação híbrida (BM25 + busca por vizinhos aproximados via embeddings) supera a recuperação isolada em MRR (Mean Reciprocal Rank), devido à complementaridade dos sinais lexicais e semânticos.
- H2 — A combinação de features lexicais com embeddings de sentenças congelados supera features apenas lexicais em Macro-F1, pois os embeddings capturam padrões semânticos que não são acessíveis por correspondência de termos.
- H3 — Regras determinísticas compiladas via DSL complementam a classificação por aprendizado de máquina ao enriquecer o espaço de features com sinais auditáveis e interpretáveis.
- H4 — Inferência cascateada (pipeline em dois estágios: classificador leve com threshold de confiança, seguido de classificador pesado apenas para janelas de baixa confiança) reduz o custo computacional médio sem perda significativa de qualidade preditiva.

4 JUSTIFICATIVA

A classificação automática de intenções — tarefa central de compreensão de linguagem natural (NLU) em sistemas conversacionais (LARSON et al., 2019; RODRIGUES NETO et al., 2022) — em conversas de atendimento ao cliente em português brasileiro representa uma lacuna relevante tanto do ponto de vista científico quanto prático. Do ponto de vista científico, benchmarks consolidados de classificação de intenções — como CLINC150 e BANKING77 — são compostos por enunciados únicos em inglês, sem modelagem de contexto multi-turno (LARSON et al., 2019). A inexistência de um corpus público de atendimento ao cliente multi-turno com anotações de intenção em PT-BR impede a avaliação de frameworks de classificação nesse idioma — lacuna que este trabalho supre como instrumento necessário à avaliação empírica do TalkEx, e não como contribuição primária. (LARSON et al., 2019; CASANUEVA et al., 2020)

Do ponto de vista prático, a ausência de ferramentas analíticas para esse tipo de dado gera ineficiências operacionais significativas: dificuldade de auditoria de conformidade em escala, perda de sinais de churn distribuídos em múltiplos turnos e dependência de categorização manual inconsistente. Uma solução computacionalmente acessível — capaz de operar em infraestrutura gratuita como o Google Colab — tem potencial de democratizar a inteligência conversacional para organizações de médio e pequeno porte no Brasil.

A integração dos três paradigmas (recuperação lexical, embeddings densos e regras) em um único pipeline não havia sido avaliada empiricamente em dados conversacionais informais em idioma diferente do inglês. O TalkEx preenche essa lacuna ao fornecer evidências empíricas sobre a complementaridade desses métodos e ao documentar tanto resultados positivos quanto negativos com total transparência metodológica.

Trabalhos relacionados ao domínio investigado confirmam a relevância e a originalidade desta pesquisa. A classificação de intenções em conversas multi-turno em português brasileiro com janelas de contexto deslizantes foi explorada por Farfan-Escobedo e Dos Reis (2024). A avaliação de recuperação híbrida léxico-semântica (BM25 + embeddings) em corpus PT-BR é discutida por Fernandes et al. (2025). Os fundamentos dos modelos de linguagem para português brasileiro são estabelecidos em Souza, Nogueira e Lotufo (2023). A comparação sistemática de classificadores supervisionados em PT-BR com Macro-F1 é realizada por Boccardo e Feltrim (2026). O desenvolvimento de chatbots com NLU em português para atendimento institucional é documentado por Rodrigues Neto et al. (2022). Nenhum desses trabalhos integra os três paradigmas (léxico, semântico e regras) em um único pipeline cascateado avaliado sobre dados de atendimento ao cliente multi-turno em PT-BR, lacuna que o TalkEx endereça.

5 OBJETIVO GERAL

Propor, implementar e avaliar empiricamente o TalkEx, uma arquitetura híbrida cascateada para classificação automática de intenções em conversas de atendimento ao cliente em português brasileiro, combinando recuperação lexical BM25, embeddings multilíngues congelados, classificação supervisionada e um motor de regras semânticas.

5.1 Objetivos específicos

- Construir e auditar um corpus de conversas de atendimento ao cliente em PT-BR com 8 classes de intenção, garantindo qualidade dos rótulos com acurácia de pelo menos 96%;
- Implementar e comparar variantes de recuperação híbrida (BM25, ANN e fusão linear/RRF) utilizando MRR como métrica primária;
- Avaliar o impacto da adição de embeddings multilíngues congelados sobre a classificação supervisionada com LightGBM, em comparação com features apenas lexicais;
- Comparar três estratégias de integração de regras determinísticas no pipeline: como features suaves, como overrides rígidos e de forma standalone;
- Investigar a viabilidade de inferência cascateada em dois estágios para redução de custo computacional;
- Realizar estudo de ablação para quantificar a contribuição marginal de cada família de features (embeddings, lexical, regras, estrutural).

6 MÉTODO PRELIMINAR

A pesquisa adota uma abordagem experimental quantitativa, com delineamento comparativo entre variantes do pipeline proposto. O método está organizado nas seguintes etapas:

Dataset: será utilizado o dataset RichardSakaguchiMS/brazilian-customer-service-conversations (HuggingFace, licença Apache 2.0), expandido com geração sintética controlada via Claude Sonnet em modo batch offline. Uma auditoria abrangente removerá duplicatas, instâncias com contaminação few-shot e rótulos ambíguos, consolidando a taxonomia em 8 classes de intenção. A revisão humana confirmará acurácia mínima de 96,7% nos rótulos. O corpus final compreenderá 2.122 conversas (847 originais + 1.275 sintéticas). O corpus expandido é submetido a um protocolo de auditoria automatizada em oito etapas: (1) validação de esquema, verificando integridade dos campos obrigatórios e intervalos válidos (texto entre 50 e 2.000 palavras, domínio dentre oito categorias e sentimento dentre três); (2) deduplicação em dois níveis — exata, após normalização (minúsculas, remoção de marcadores de turno, colapso de espaços), e por proximidade semântica via embeddings (paraphrase-multilingual-MiniLM-L12-v2), com limiar suave a 0,92 e limiar rígido a 0,97 de similaridade cosseno; pares com rótulos discordantes são sinalizados como cross-intent para revisão humana; (3) detecção de contaminação few-shot, rastreando exemplos sintéticos cujos prompts de geração utilizam instâncias dos conjuntos de validação ou teste; (4) verificação de integridade dos splits estratificados, com tolerância máxima de 3 pp na distribuição de intenções e 5 pp na distribuição de domínios; (5) auditoria taxonômica baseada em embeddings, exigindo coerência intra-classe média ≥ 0,60 e separabilidade inter-classe (similaridade cosseno entre centróides de classes distintas) ≤ 0,90; (6) análise da categoria outros por agrupamento k-means (k=5), classificando cada registro como: A — possivelmente mal rotulado (similaridade com a intenção mais próxima ≥ 0,85, candidato à reclassificação); B — ambíguo (similaridade ∈ [0,75; 0,85), mantido para calibração de abstenção); ou C — fora do escopo (similaridade < 0,75, removido do treinamento supervisionado); (7) verificação de qualidade textual, exigindo presença de marcadores de turno ([customer]/[agent]), mínimo de dois turnos não vazios e proporção de caracteres latinos ≥ 90%; e (8) análise de distribuição multidimensional com tabulações cruzadas intenção × domínio e intenção × sentimento. Os registros sinalizados nas categorias A e C da etapa 6, bem como os pares cross-intent da etapa 2, são encaminhados para revisão humana pelos pesquisadores. A consolidação da taxonomia em 8 classes de intenção e a acurácia mínima esperada nos rótulos finais é de 96,7%. O corpus final compreenderá 2.122 conversas (847 originais + 1.275 sintéticas). (SAKAGUCHI, 2023)

Arquitetura do pipeline: o TalkEx implementa um pipeline multi-estágio com (1) ingestão e segmentação de turnos; (2) construção de janelas de contexto deslizantes de 5 turnos com stride 2; (3) normalização de texto em dois níveis — NFKC para canonicalização na segmentação e NFD para remoção de diacríticos no matching lexical; (4) geração de embeddings com paraphrase-multilingual-MiniLM-L12-v2 (384 dimensões, congelado); (5) recuperação híbrida por interpolação linear entre escores BM25 e busca exata por similaridade cosseno sobre embeddings densos; (6) classificação com LightGBM em 397 features (384 embeddings + 7 lexicais + 4 estruturais + 2 derivadas de regras); (7) motor de regras semânticas compilado via DSL para ASTs tipadas; e (8) agregação janela-para-conversa por média das probabilidades. (KE et al., 2017) (JOHNSON; DOUZE; JÉGOU, 2019) (WANG et al., 2020; REIMERS; GUREVYCH, 2019)

O motor de regras semânticas é implementado mediante uma linguagem de domínio específico (DSL) compilada para árvores sintáticas abstratas (ASTs) tipadas, suportando quatro famílias de predicados: (i) léxicos — correspondência de palavras-chave, expressões regulares, prefixos e listas, com normalização automática de acentuação (e.g., "cancelar" corresponde a "cancêlar"); (ii) semânticos — similaridade por embeddings e limiar sobre a pontuação de intenção pré-computada, sem recálculo de embeddings em tempo de avaliação; (iii) estruturais — papel do locutor (customer/agent), canal (voice/chat/email) e comparações numéricas de campos; e (iv) contextuais — padrões de repetição e sequências de ocorrência entre turnos. Os predicados combinam-se pelos operadores lógicos AND, OR e NOT, com avaliação em curto-circuito ordenada por custo crescente (léxico < estrutural < contextual < semântico), minimizando chamadas ao módulo de embeddings. A sintaxe de bloco (RULE … WHEN … THEN) permite atribuir tags, sobrescrever a pontuação final e definir prioridade de disparo. O conjunto de regras previsto compreenderá 10 regras cobrindo as 8 classes de intenção, redigidas e validadas pelos pesquisadores durante a Stage 3 do cronograma, conforme a seguinte estrutura ilustrativa: RULE risco_cancelamento / WHEN speaker == "customer" AND semantic.intent("cancelamento") > 0,82 AND lexical.contains_any(["cancelar", "encerrar", "desistir"]) / THEN tag("cancelamento_risco") score(0,95) priority("high").

Protocolo experimental: todos os experimentos utilizarão 5 sementes aleatórias [13, 42, 123, 2024, 999] com splits estratificados 70/15/15% (treino/validação/teste) no nível da conversa, prevenindo vazamento de dados no nível de janela. Os resultados serão reportados como média ± desvio padrão entre sementes. Significância estatística será avaliada via testes de Wilcoxon signed-rank (α = 0,05) com intervalos de confiança bootstrap de 95% (10.000 reamostras). Tamanhos de efeito serão reportados como correlação rank-biserial (r_rb), medida padrão para o teste de Wilcoxon, com interpretação: < 0,1 negligível, 0,1–0,3 pequeno, 0,3–0,5 médio, > 0,5 grande. (WILCOXON, 1945)

Métricas: para recuperação (H1): MRR como métrica primária, além de Recall@K e nDCG@K para K ∈ {5, 10, 20}. Para classificação (H2 e H3): Macro-F1 como métrica primária, F1 por classe e acurácia. Para inferência cascateada (H4): custo por janela em milissegundos, variação de F1 e percentual de janelas resolvidas por estágio.

Infraestrutura: todos os experimentos serão executados no Google Colab com GPU Tesla T4 (15 GB VRAM). A geração de embeddings utilizará aceleração GPU via PyTorch/CUDA; a classificação LightGBM treinará em CPU (aproximadamente 6 segundos). A suíte completa de experimentos será concluída em menos de 1 hora. Stack de software: Python 3.11+, sentence-transformers, scikit-learn, LightGBM, numpy e rank-bm25. (ROBERTSON; ZARAGOZA, 2009) (PEDREGOSA et al., 2011)

Aspectos éticos: o dataset utilizado é de domínio público (licença Apache 2.0). Os dados sintéticos são gerados com controle de qualidade e auditados manualmente. Não há coleta de dados pessoais identificáveis nem envolvimento de participantes humanos, dispensando apreciação por Comitê de Ética em Pesquisa.

7 RESULTADOS ESPERADOS

A partir dos objetivos estabelecidos e da arquitetura proposta, espera-se obter os seguintes resultados:

- Confirmação da hipótese H1: espera-se que a recuperação híbrida (fusão linear BM25 + ANN) supere a recuperação isolada em MRR, com diferença estatisticamente significativa (p < 0,05), evidenciando a complementaridade dos sinais lexicais e semânticos em conversas informais em PT-BR.
- Confirmação da hipótese H2: espera-se que a adição de embeddings multilíngues congelados ao conjunto de features produza ganhos expressivos em Macro-F1, superiores a 30 pontos percentuais em relação a features apenas lexicais, com efeito grande (r_rb > 0,5).
- Evidências sobre integração de regras (H3): espera-se que regras-como-features produzam ganho marginal sobre o pipeline apenas com ML, enquanto regras-como-override apresentem degradação de desempenho, fornecendo orientação prática para arquitetos de sistemas conversacionais.
- Análise de viabilidade da cascata (H4): espera-se identificar as condições estruturais necessárias para que inferência cascateada produza redução efetiva de custo, mesmo que a hipótese original seja refutada.
- Contribuição ao estado da arte em PLN para PT-BR: espera-se que o corpus auditado, o pipeline open-source e os resultados experimentais representem a primeira avaliação sistemática de classificação de intenções em conversas de atendimento ao cliente multi-turno em português brasileiro.
- Paradigma acessível de classificação: espera-se demonstrar que Macro-F1 superior a 0,70 é alcançável em 8 classes de intenção sem fine-tuning do codificador, utilizando exclusivamente infraestrutura gratuita (Google Colab).

8 CRONOGRAMA DAS ATIVIDADES

| Atividade | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage 1: Revisão de literatura e levantamento do dataset | X | X | X | | | | | | | | | | | | |
| Stage 2: Auditoria e pré-processamento do corpus | | | | X | X | X | | | | | | | | | |
| Stage 3: Implementação do pipeline TalkEx | | | | | | | X | X | X | X | | | | | |
| Stage 4: Experimentos e avaliação das hipóteses | | | | | | | | | | | X | X | X | X | |
| Stage 5: Ablação, análise de erros e discussão | | | | | | | | | | | | | X | X | X |
| Stage 6: Redação final e revisão | | | | | | | | | | | | | | X | X |

REFERÊNCIAS

FORMAL, T. et al. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv preprint arXiv:2109.10086, 2021.

KARPUKHIN, V. et al. Dense Passage Retrieval for Open-Domain Question Answering. In: Proceedings of EMNLP, 2020. p. 6769-6781.

KHATTAB, O.; ZAHARIA, M. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In: Proceedings of SIGIR, 2020. p. 39-48.

LARSON, S. et al. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. In: Proceedings of EMNLP-IJCNLP, 2019. p. 1311-1316.

RAYO, L.; DE LA ROSA, A.; GARRIDO, A. A Hybrid Approach to Information Retrieval and Answer Generation for Regulatory Texts. In: Proceedings of COLING 2025, 2025.

REIMERS, N.; GUREVYCH, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In: Proceedings of EMNLP-IJCNLP, 2019. p. 3982-3992.

SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: Pretrained BERT Models for Brazilian Portuguese. In: Proceedings of BRACIS, 2020. p. 403-417.

TUNSTALL, L. et al. Efficient Few-Shot Learning Without Prompts. arXiv preprint arXiv:2209.11055, 2022.

ZHANG, Z. et al. Multi-Turn Intent Classification with Hierarchical Attention. In: Proceedings of NAACL, 2021. p. 1845-1855.

JOHNSON, J.; DOUZE, M.; JÉGOU, H. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, v. 7, n. 3, p. 535-547, 2019.

KE, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 2017. p. 3149-3157.

PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011.

ROBERTSON, S.; ZARAGOZA, H. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, v. 3, n. 4, p. 333-389, 2009.

SAKAGUCHI, R. M. S. Brazilian Customer Service Conversations. HuggingFace, 2023. Disponivel em: https://huggingface.co/datasets/RichardSakaguchiMS/brazilian-customer-service-conversations.

WANG, W. et al. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. In: Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 2020.

WILCOXON, F. Individual Comparisons by Ranking Methods. Biometrics Bulletin, v. 1, n. 6, p. 80-83, 1945.

CASANUEVA, I. et al. Efficient Intent Detection with Dual Sentence Encoders. In: Proceedings of the 2nd Workshop on NLP for ConvAI (ACL), 2020.

BOCCARDO, M.; FELTRIM, V. D. Automatic Question Classification in Portuguese: A Large-Scale Dataset and Comparative Evaluation of Classification Strategies. In: Proceedings of PROPOR 2026 (ACL Anthology), 2026. p. 436-445.

FARFAN-ESCOBEDO, J. D.; DOS REIS, J. C. Improved intent classification based on context information using a windows-based approach. arXiv:2411.06022, 2024.

FERNANDES, L. C. et al. JurisTCU: A Brazilian Portuguese Information Retrieval Dataset with Query Relevance Judgments. Language Resources and Evaluation, 2025. arXiv:2503.08379.

RODRIGUES NETO, J. et al. Chatbot to Support Frequently Asked Questions from Students in Higher Education Institutions. In: Anais do ENIAC 2022 (SBC), 2022.

SOUZA, F. C.; NOGUEIRA, R. F.; LOTUFO, R. A. BERT Models for Brazilian Portuguese: Pretraining, Evaluation and Tokenization Analysis. Neurocomputing, v. 567, 2024.
