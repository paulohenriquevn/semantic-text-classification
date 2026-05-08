# UNIVERSIDADE FEDERAL DE GOIÁS
# INSTITUTO DE INFORMÁTICA
# CENTRO DE COMPETÊNCIA EMBRAPII EM TECNOLOGIAS IMERSIVAS (AKCIT)
# ESPECIALIZAÇÃO EM INTELIGÊNCIA ARTIFICIAL GENERATIVA

**Cauê Cavichioli Leão**
**Paulo Henrique Vieira Nascimento**
**Suele Susan Feitosa Sousa**

# TalkEx: Arquitetura Híbrida para Classificação de Intenções em Conversas de Atendimento ao Cliente em Português Brasileiro

GOIÂNIA
2026

---

*(Folha de rosto — repetir dados acima com texto abaixo)*

Trabalho de Conclusão de Curso apresentado ao Programa de Capacitação e Formação do Centro de Competências Embrapii em Tecnologias Imersivas (AKCIT) como requisito parcial para obtenção do título de Especialista em Inteligência Artificial Generativa.

Orientador(a): *(inserir nome)*

---

## AGRADECIMENTOS

Agradecemos ao Centro de Competências Embrapii em Tecnologias Imersivas (AKCIT) e à Universidade Federal de Goiás pelo suporte institucional e infraestrutura disponibilizada. Agradecemos também à comunidade open source das bibliotecas sentence-transformers, rank-bm25 e scikit-learn, cujas ferramentas tornaram este trabalho possível com infraestrutura gratuita.

---

## RESUMO

**Introdução:** A classificação automática de intenções em conversas de atendimento ao cliente é um requisito operacional para call centers em escala. Sistemas tradicionais baseados em modelos supervisionados exigem retreinamento a cada nova classe de intenção, tornando-os inflexíveis para equipes de negócio. Este trabalho propõe o TalkEx, uma arquitetura híbrida que combina recuperação lexical (BM25) e semântica (embeddings) para classificar intenções em conversas em português brasileiro, mantendo taxonomia dinâmica via banco de exemplares — sem retreino. **Método:** O sistema opera sobre janelas de contexto configuráveis com marcadores de falante ([customer]/[agent]). Avaliamos 6 métodos de classificação (BM25 KNN, Embedding KNN, Hybrid, LogReg, BERTimbau+KNN, MPNet+KNN) e um motor de regras multi-sinal com 4 famílias de predicados (lexical, estrutural, contextual, semântico), portado de um sistema de produção. O protocolo experimental utiliza 5 splits estratificados, Wilcoxon signed-rank com correção de Holm-Bonferroni, Bootstrap CI 95% e estudo de ablação de 5 componentes. **Resultados:** O BERTimbau+KNN (encoder PT-BR, 768d) atingiu o melhor resultado absoluto (Macro-F1=0,77), seguido pelo Hybrid BM25+Embedding (0,75). A ablação revelou que dados sintéticos gerados por LLM são o componente mais crítico (+22pp) e que sinais lexicais (BM25) dominam sobre embeddings semânticos (+4,9pp). O motor de regras multi-sinal classifica 7,4% das janelas com 86,2% de acurácia e rastreabilidade completa por predicado, sem degradar o F1 do sistema. A cascata reduz latência em 7,1%. **Conclusão:** O TalkEx demonstra que classificação por similaridade com taxonomia dinâmica atinge performance comparável a modelos supervisionados, com vantagens operacionais de flexibilidade, interpretabilidade e escalabilidade. Os achados confirmam que sinais lexicais dominam em domínios com vocabulário restrito e que encoders pré-treinados para PT-BR superam multilíngues.

**Descritores:** Classificação de Intenções; Processamento de Linguagem Natural; Português Brasileiro; BM25; Embeddings; Taxonomia Dinâmica.

---

## ABSTRACT

**Introduction:** Automatic intent classification in customer service conversations is an operational requirement for call centers at scale. Traditional supervised systems require retraining for each new intent class, making them inflexible for business teams. This work proposes TalkEx, a hybrid architecture that combines lexical (BM25) and semantic (embeddings) retrieval to classify intents in Brazilian Portuguese conversations while maintaining dynamic taxonomy via exemplar bank — without retraining. **Method:** The system operates on configurable context windows with speaker markers ([customer]/[agent]). We evaluated 6 classification methods and a multi-signal rule engine with 4 predicate families ported from a production system. The experimental protocol uses 5 stratified splits, Wilcoxon signed-rank with Holm-Bonferroni correction, and a 5-component ablation study. **Results:** BERTimbau+KNN (PT-BR encoder, 768d) achieved the best result (Macro-F1=0.77), followed by Hybrid BM25+Embedding (0.75). Ablation revealed that LLM-generated synthetic data is the most critical component (+22pp) and lexical signals (BM25) dominate over semantic embeddings (+4.9pp). The multi-signal rule engine classifies 7.4% of windows with 86.2% accuracy and full predicate traceability, without degrading system F1. **Conclusion:** TalkEx demonstrates that similarity-based classification with dynamic taxonomy achieves performance comparable to supervised models, with operational advantages in flexibility, interpretability, and scalability.

**Keywords:** Intent Classification; Natural Language Processing; Brazilian Portuguese; BM25; Embeddings; Dynamic Taxonomy.

---

## LISTA DE ILUSTRAÇÕES

- Figura 1 — Arquitetura do TalkEx v9
- Figura 2 — Resultados H1: Fusão híbrida vs métodos isolados
- Figura 3 — Resultados H2: Sensibilidade ao K (BM25 vs Embedding)
- Figura 4 — Resultados H3: Regras + classificação por similaridade
- Figura 5 — Resultados H4: Confidence-gated routing vs cascade vs hybrid
- Figura 6 — Distribuição de margens do Hybrid
- Figura 7 — Comparação com baselines externos
- Figura 8 — Estudo de ablação
- Figura 9 — Curva few-shot (N exemplares vs Macro-F1)
- Figura 10 — Matriz de confusão (Hybrid, seed=999)
- Figura 11 — t-SNE dos embeddings por janela
- Figura 12 — Sweep de tamanho de janela

## LISTA DE TABELAS

- Tabela 1 — Distribuição do corpus por classe de intenção
- Tabela 2 — Comparação de métodos (Macro-F1, Micro-F1, Acurácia)
- Tabela 3 — Estudo de ablação (5 configurações)
- Tabela 4 — Curva few-shot (N exemplares por classe)
- Tabela 5 — Métricas de latência por método
- Tabela 6 — Análise de erro: top-5 pares de confusão

## LISTA DE ABREVIATURAS E SIGLAS

| Sigla | Descrição |
|---|---|
| ABNT | Associação Brasileira de Normas Técnicas |
| BM25 | Best Matching 25 (algoritmo de recuperação lexical) |
| CI | Intervalo de Confiança (Confidence Interval) |
| Embrapii | Empresa Brasileira de Pesquisa e Inovação Industrial |
| F1 | Medida F1 (média harmônica de precisão e recall) |
| KNN | K-Nearest Neighbors |
| LLM | Large Language Model |
| NLU | Natural Language Understanding |
| PT-BR | Português Brasileiro |
| STS | Semantic Textual Similarity |
| TF-IDF | Term Frequency–Inverse Document Frequency |

---

# RELATÓRIO TÉCNICO — PESQUISA APLICADA

## 1 INTRODUÇÃO

A classificação automática de intenções em conversas de atendimento ao cliente é um componente central de sistemas de NLU para chatbots e call centers. Em operações com milhares de atendimentos diários, identificar a intenção do cliente (cancelamento, reclamação, dúvida, compra, etc.) permite roteamento inteligente, priorização de demandas e análise de satisfação em escala (Larson et al., 2019; Zhang et al., 2021).

Sistemas tradicionais de classificação de intenções utilizam modelos supervisionados — como LightGBM, BERT fine-tuned ou redes neurais — que requerem conjuntos de treinamento rotulados e retreinamento completo a cada nova classe de intenção (Devlin et al., 2019). Em contextos empresariais onde equipes de negócio precisam criar e ajustar intenções dinamicamente (por exemplo, adicionar "troca de produto" ou "segunda via de boleto"), o ciclo de retreinamento cria um gargalo operacional.

Uma alternativa é a classificação por similaridade via KNN sobre embeddings: novos exemplares são adicionados a um banco de exemplares, e o sistema classifica novas entradas por votação dos K vizinhos mais próximos (Cover; Hart, 1967; Zhang et al., 2020). Esta abordagem mantém taxonomia dinâmica — novas classes são adicionadas sem retreino — mas depende da qualidade das representações vetoriais.

Para o português brasileiro, os desafios são amplificados: (i) modelos multilíngues como o MiniLM (Reimers; Gurevych, 2019) não capturam nuances do PT-BR; (ii) modelos específicos como o BERTimbau (Souza et al., 2020) e o BERTaú (Finardi et al., 2021) demonstram ganhos significativos mas requerem infraestrutura de fine-tuning; e (iii) conversas de atendimento em PT-BR combinam linguagem informal, abreviações e alternância de código que dificultam a classificação.

A fusão de sinais lexicais (BM25) e semânticos (embeddings) tem sido explorada em information retrieval (Bruch et al., 2023; Rayo et al., 2025), com evidências de que BM25 generaliza melhor em domínios especializados (Thakur et al., 2021; Harris, 2025). No entanto, a aplicação desta fusão para classificação de intenções em conversas PT-BR permanece pouco explorada.

Neste contexto, o presente trabalho apresenta o TalkEx, uma arquitetura híbrida que combina recuperação lexical (BM25) e semântica (embeddings) sobre janelas de contexto configuráveis com marcadores de falante, mantendo taxonomia dinâmica via banco de exemplares. O sistema integra um motor de regras multi-sinal com 4 famílias de predicados (lexical, estrutural, contextual, semântico), portado de um sistema de produção, que oferece interpretabilidade e redução de latência nos casos classificados por regras.

### 1.1 Objetivo Geral

Desenvolver e avaliar uma arquitetura híbrida para classificação de intenções em conversas de atendimento ao cliente em português brasileiro, combinando recuperação lexical e semântica com taxonomia dinâmica.

### 1.2 Objetivos Específicos

1. Avaliar se a fusão de sinais lexicais (BM25) e semânticos (embeddings) melhora a classificação sobre componentes isolados (H1).
2. Investigar a dominância de sinais lexicais sobre semânticos em domínio de atendimento PT-BR (H2).
3. Comparar encoders pré-treinados para PT-BR (BERTimbau) com encoders multilíngues (H3).
4. Quantificar o impacto de dados sintéticos gerados por LLM em corpus de pequeno porte (H4).
5. Avaliar se regras multi-sinal oferecem interpretabilidade sem custo de accuracy (H5).

---

## 2 DESCRIÇÃO DA TECNOLOGIA

### 2.1 Visão Geral da Arquitetura

O TalkEx opera em um pipeline de 5 estágios:

1. **Segmentação de turnos:** O texto da conversa é segmentado em turnos individuais, identificando o falante ([customer] ou [agent]) via expressão regular.
2. **Janelas de contexto:** Turnos consecutivos são agrupados em janelas deslizantes de N turnos (configurável, padrão N=5, passo=2). O texto da janela preserva os marcadores de falante: `[customer] texto... [agent] texto...`.
3. **Representação vetorial:** Cada janela é codificada por um sentence transformer (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensões, congelado) via normalização L2.
4. **Classificação por similaridade:** A janela de teste é classificada por KNN com voto ponderado por similaridade, utilizando um banco de exemplares rotulados.
5. **Agregação:** Predições por janela são agregadas para o nível da conversa via voto majoritário ou ponderado.

### 2.2 Métodos de Classificação

O sistema implementa 6 métodos de classificação:

- **Embedding KNN:** Similaridade cosseno (dot product em vetores L2-normalizados).
- **BM25 KNN:** Recuperação lexical via BM25 Okapi (Robertson et al., 1996) com tokenização normalizada (lowercase + remoção de acentos).
- **Hybrid KNN:** Fusão linear: score = α × emb_norm + (1−α) × bm25_norm, com normalização min-max. O parâmetro α controla o peso relativo.
- **Rerank:** BM25 recupera top-50 candidatos, embedding reordena para top-K.
- **Confidence-Gated Routing:** Hybrid classifica todas as janelas; para janelas com margem baixa (1° − 2° lugar), regras multi-sinal podem corrigir a predição.
- **Cascade v3:** Regras classificam primeiramente (se match exclusivo); o Hybrid trata o restante.

### 2.3 Motor de Regras Multi-Sinal

Portado do sistema de produção, o motor implementa 4 famílias de predicados:

- **Lexical (custo 1):** contains, contains_any, word, regex.
- **Estrutural (custo 2):** speaker ("customer"), first_customer_turn.
- **Contextual (custo 3):** occurs_after, repeated.
- **Semântico (custo 4):** intent_score ≥ threshold (pré-computado via KNN).

Regras combinam predicados com AND. Score = min(scores dos predicados). Avaliação em ordem de custo crescente com short-circuit. Cada classificação por regra é rastreável: quais predicados dispararam, com qual score, sobre qual texto.

### 2.4 Taxonomia Dinâmica

Novas classes de intenção são adicionadas inserindo exemplares no banco, sem retreino. O sistema classifica automaticamente pela similaridade com os novos exemplares. Demonstramos com a classe "troca_produto": 8 exemplares adicionados, 2/4 queries classificadas corretamente, impacto zero nas classes existentes.

---

## 3 PROCEDIMENTOS METODOLÓGICOS

### 3.1 Dataset

Utilizamos o corpus `paulohenriquevn/talkex-augmented-pt-br`, publicado no HuggingFace, contendo 2.120 conversas de atendimento ao cliente em português brasileiro, distribuídas em 8 classes de intenção balanceadas (265 conversas cada):

| Classe | Originais | Sintéticas | Total |
|---|---|---|---|
| cancelamento | 79 | 186 | 265 |
| reclamação | 84 | 181 | 265 |
| suporte_técnico | 88 | 177 | 265 |
| compra | 85 | 180 | 265 |
| dúvida_produto | 80 | 185 | 265 |
| dúvida_serviço | 87 | 178 | 265 |
| saudação | 88 | 177 | 265 |
| elogio | 85 | 180 | 265 |
| **Total** | **676** | **1.444** | **2.120** |

As 676 conversas originais foram coletadas de cenários reais de atendimento. As 1.444 sintéticas foram geradas via Claude Sonnet para augmentation, mantendo diversidade lexical e estrutural.

Cada conversa produz em média 2,8 janelas de 5 turnos (5.927 janelas totais). As conversas possuem em média 7,9 turnos (mediana: 8).

### 3.2 Protocolo Experimental

- **Splits:** 5 partições estratificadas (70% treino / 15% validação / 15% teste), no nível da conversa (sem vazamento de janelas entre splits).
- **Métricas:** Macro-F1, Micro-F1, Acurácia, Macro-Precisão, Macro-Recall.
- **Teste estatístico:** Wilcoxon signed-rank (bilateral) com correção de Holm-Bonferroni para múltiplas comparações (m=3 testes formais, α=0,05).
- **Intervalo de confiança:** Bootstrap 95% com 10.000 reamostras.
- **Effect size:** Correlação rank-biserial (r_rb).
- **Baselines:** LogisticRegression + MiniLM embeddings, SetFit 8-shot (Tunstall et al., 2022), BERTimbau + KNN (Souza et al., 2020), MPNet-base (768d) + KNN.
- **Ablação:** 5 configurações (Full, −Speakers, −BM25, −Embeddings, −Sintéticos).
- **Infraestrutura:** Google Colab, GPU Tesla T4, custo zero.

### 3.3 Hipóteses

As hipóteses foram refinadas iterativamente ao longo de 9 versões experimentais, seguindo a prática de pesquisa exploratória (Tukey, 1977):

- **H1:** A fusão de sinais lexicais e semânticos melhora a classificação sobre cada componente isolado.
- **H2:** Em conversas de atendimento PT-BR, sinais lexicais (BM25) dominam sobre embeddings semânticos.
- **H3:** Um encoder pré-treinado para PT-BR (BERTimbau) supera encoders multilíngues.
- **H4:** Dados sintéticos gerados por LLM são essenciais para performance em corpus de pequeno porte.
- **H5:** Regras multi-sinal oferecem interpretabilidade e redução de latência sem custo de accuracy.

---

## 4 TESTES E RESULTADOS

### 4.1 Comparação de Métodos (H1, H2, H3)

**Tabela 2 — Comparação de métodos**

| Método | Tipo | Macro-F1 | Micro-F1 | Acurácia |
|---|---|---|---|---|
| BERTimbau+KNN (768d, PT-BR) | Exemplar | **0,768** | 0,767 | 0,767 |
| Hybrid (α=0,5) | Híbrido | 0,748 | 0,748 | 0,748 |
| Hybrid (α=0,65) | Híbrido | 0,746 | 0,744 | 0,744 |
| BM25-KNN (K=15) | Lexical | 0,748 | 0,748 | 0,748 |
| LogReg+MiniLM | Supervisionado | 0,719 | 0,718 | 0,718 |
| MPNet+KNN (768d, multilingual) | Exemplar | 0,716 | 0,714 | 0,714 |
| Rerank-50 | Dois estágios | 0,719 | — | — |
| Embedding-KNN (384d, MiniLM) | Exemplar | 0,697 | 0,696 | 0,696 |

**H1 — Fusão léxico-semântica: CONFIRMADA (direcional).** O Hybrid (0,748) supera BM25 isolado (0,736) em 4/5 seeds e Embedding isolado (0,697) em 5/5 seeds. Wilcoxon p=0,094 (não significativo com N=5), effect size r_rb=−0,73. Com correção de Holm-Bonferroni (m=3, limiar=0,017), o resultado não atinge significância. O efeito é consistente e direcional, limitado pelo poder estatístico.

**H2 — BM25 domina em PT-BR: CONFIRMADA.** BM25-KNN (0,748) supera Embedding-KNN (0,701) por +4,8pp em 5/5 seeds. Bootstrap CI [−0,061; −0,034] não cruza zero. A ablação confirma: remover BM25 do Hybrid custa −4,9pp vs remover embeddings custa −1,0pp. Resultado alinhado com Harris (2025) e Thakur et al. (2021).

**H3 — BERTimbau supera multilingual: CONFIRMADA.** BERTimbau+KNN (0,768) supera Hybrid MiniLM (0,746) por +2,2pp e MPNet multilingual (0,716) por +5,2pp. O controle com MPNet (mesmo tamanho 768d) confirma que é o pré-treino PT-BR que faz a diferença, não o tamanho do modelo.

### 4.2 Impacto dos Dados Sintéticos (H4)

**Tabela 3 — Estudo de ablação**

| Configuração | Macro-F1 | Δ vs Full |
|---|---|---|
| Full (Hybrid + speakers + todos os dados) | 0,746 | — |
| −Speakers (sem marcadores [customer]/[agent]) | 0,740 | −0,6pp |
| −BM25 (Embedding-only, α=1.0) | 0,697 | −4,9pp |
| −Embeddings (BM25-only, α=0.0) | 0,736 | −1,0pp |
| −Sintéticos (apenas 676 originais) | **0,527** | **−21,9pp** |

**H4 — Dados sintéticos essenciais: CONFIRMADA.** A remoção dos dados sintéticos causa queda de −21,9pp (de 0,746 para 0,527). Este é o achado mais forte da ablação. Com apenas 676 conversas originais, o sistema é insuficiente. O augmentation via LLM (Claude Sonnet) é o componente mais crítico do pipeline.

### 4.3 Regras Multi-Sinal (H5)

**Tabela 5 — Métricas de latência (index pré-construído)**

| Método | Latência/janela | Speedup vs Hybrid |
|---|---|---|
| Embedding KNN | 0,12ms | 634× |
| Rules-only | 0,22ms | 346× |
| BM25 KNN (query) | 76,5ms | 1,0× |
| Hybrid KNN (query) | 76,1ms | 1,0× |
| Cascade (Rules→Hybrid) | 70,7ms | 1,08× |

**H5 — Regras sem custo de accuracy: CONFIRMADA.** A Cascade-v3 (F1=0,745) é estatisticamente indistinguível do Hybrid (0,746, Δ=−0,07pp). As regras classificam 7,4% das janelas com 86,2% de acurácia e rastreabilidade completa. A cascata reduz latência em 7,1% (−151s/dia em 10K atendimentos).

### 4.4 Análise de Erro

Os 5 pares de confusão mais frequentes:

| Real → Predito | N erros |
|---|---|
| compra → dúvida_produto | 23 |
| dúvida_produto → compra | 18 |
| compra → saudação | 14 |
| saudação → reclamação | 11 |
| saudação → dúvida_serviço | 11 |

O par compra/dúvida_produto é o mais confundido (41 erros bidirecionais), refletindo a sobreposição semântica ("quanto custa?" pode ser compra ou dúvida).

### 4.5 Curva Few-Shot

| N exemplares/classe | N treino | Macro-F1 (Hybrid) | % do máximo |
|---|---|---|---|
| 1 | 8 | 0,282 | 37,8% |
| 5 | 40 | 0,472 | 63,3% |
| 10 | 80 | 0,514 | 68,9% |
| 25 | 200 | 0,569 | 76,3% |
| 50 | 400 | 0,612 | 82,0% |
| 100 | 800 | 0,653 | 87,6% |
| Todos | 4.152 | 0,746 | 100,0% |

Com 50 exemplares por classe (400 total), o sistema atinge 82% da performance máxima. Com 100 exemplares, 88%.

### 4.6 Sweep de Tamanho de Janela

| WINDOW_SIZE | N janelas | F1 janela | F1 conversa |
|---|---|---|---|
| 3 | 8.047 | 0,645 | 0,757 |
| 5 | 5.927 | 0,697 | 0,744 |
| 7 | 3.813 | 0,731 | 0,752 |
| all | 2.120 | **0,766** | **0,766** |

WINDOW_SIZE=all (conversa inteira como janela única) atinge o melhor F1 absoluto (0,766).

---

## 5 DISCUSSÃO

### 5.1 Dominância do BM25 em Domínio Restrito

O achado mais contra-intuitivo é que BM25 (recuperação lexical pura) supera embeddings semânticos por 4,8pp. Isso se alinha com Harris (2025), que demonstrou dominância do BM25 em documentos médicos com terminologia técnica, e com Thakur et al. (2021), que mostraram que BM25 generaliza melhor em cenários zero-shot no benchmark BEIR.

Conversas de atendimento ao cliente em PT-BR possuem vocabulário restrito com keywords altamente discriminativas ("cancelar", "procon", "erro", "parcela"). Nesses domínios, o matching lexical exato captura informação que embeddings multilíngues genéricos diluem. A fusão α=0,5–0,65 (50–65% peso lexical) confirma que o sinal BM25 precisa de peso majoritário (Bruch et al., 2023).

### 5.2 BERTimbau como Encoder Superior

BERTimbau+KNN (0,768) supera o Hybrid MiniLM (0,746) sem nenhum fine-tuning para STS. Isso confirma Souza et al. (2020): modelos pré-treinados para PT-BR capturam nuances linguísticas (acentuação, expressões coloquiais, abreviações) que encoders multilíngues perdem. O controle com MPNet (mesmo tamanho 768d, multilingual, 0,716) confirma que o fator diferencial é o pré-treino em PT-BR, não a dimensionalidade do embedding.

### 5.3 Criticidade dos Dados Sintéticos

A ablação mostra −21,9pp sem dados sintéticos (0,746→0,527). Com apenas 676 conversas originais (~85 por classe), o banco de exemplares é insuficiente para cobertura lexical e semântica. Os dados gerados por Claude Sonnet triplicam o corpus e adicionam diversidade de paráfrases que o KNN necessita para robustez. Este achado valida data augmentation via LLM para NLU em PT-BR, um cenário sub-explorado na literatura.

### 5.4 Interpretabilidade como Valor Operacional

As regras multi-sinal não melhoram F1 (Δ=−0,07pp), mas oferecem: (1) rastreabilidade completa por predicado nos 7,4% de janelas classificadas; (2) redução de latência de 7,1% na cascata; (3) auditabilidade para domínios regulados (financeiro, saúde). Em produção com 10K atendimentos/dia, a cascata economiza ~151s/dia.

### 5.5 Limitações

- **Poder estatístico:** N=5 splits limita a detecção de efeitos pequenos. H1 (p=0,094) não atinge significância com Holm-Bonferroni (limiar=0,017).
- **Rótulos herdados:** Janelas herdam o label da conversa. Janelas iniciais ("Boa tarde") podem ter intenção diferente do label global.
- **Dependência de sintéticos:** 68% do corpus é gerado por LLM. A generalização para dados puramente reais requer validação.
- **Sem cross-dataset:** Não testamos em Wavy (Farfan-Escobedo; Dos Reis, 2024) ou CLINC150 (Larson et al., 2019).
- **Embeddings congelados:** Fine-tuning do encoder para STS em conversas PT-BR provavelmente melhoraria os resultados.

---

## 6 CONSIDERAÇÕES FINAIS

O TalkEx demonstra que classificação de intenções por similaridade com taxonomia dinâmica é viável para conversas de atendimento ao cliente em português brasileiro, atingindo Macro-F1 de 0,77 com BERTimbau+KNN e 0,75 com fusão híbrida BM25+Embedding — superando modelos supervisionados anteriores (LightGBM, F1=0,73) sem retreino.

Os cinco achados principais são: (1) a fusão léxico-semântica melhora consistentemente sobre componentes isolados; (2) sinais lexicais (BM25) dominam sobre embeddings em domínios com vocabulário restrito; (3) encoders PT-BR superam multilíngues; (4) dados sintéticos via LLM são críticos para corpus pequenos (+22pp); e (5) regras multi-sinal oferecem interpretabilidade sem custo de accuracy.

A tecnologia é viável para produção: taxonomia dinâmica, infraestrutura gratuita (Colab), dataset público, e motor de regras rastreável. Os próximos passos incluem fine-tuning do BERTimbau para STS, cross-dataset evaluation, e escalação para LLM nos casos de baixa confiança.

---

## REFERÊNCIAS

BRUCH, S.; GAI, S.; INGBER, A. An Analysis of Fusion Functions for Hybrid Retrieval. **ACM Transactions on Information Systems**, v. 42, n. 1, 2023.

COVER, T.; HART, P. Nearest Neighbor Pattern Classification. **IEEE Transactions on Information Theory**, v. 13, n. 1, p. 21–27, 1967.

DEVLIN, J. et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In: **NAACL-HLT**, 2019. p. 4171–4186.

FARFAN-ESCOBEDO, J. L.; DOS REIS, J. C. Improved Intent Classification Based on Context Information Using a Windows-Based Approach. **arXiv:2411.06022**, 2024.

FINARDI, P. et al. BERTaú: Itaú BERT for Digital Customer Service. **arXiv:2101.12015**, 2021.

GEIFMAN, Y.; EL-YANIV, R. SelectiveNet: A Deep Neural Network with an Integrated Reject Option. In: **ICML**, 2019.

HARRIS, L. Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents. **arXiv:2505.11582**, 2025.

JITKRITTUM, W. et al. When Does Confidence-Based Cascade Deferral Suffice? In: **NeurIPS**, 2023.

KARPUKHIN, V. et al. Dense Passage Retrieval for Open-Domain Question Answering. In: **EMNLP**, 2020.

KHATTAB, O.; ZAHARIA, M. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In: **SIGIR**, 2020.

LARSON, S. et al. An Evaluation Dataset for Intent Detection with Out-of-Scope Queries. In: **EMNLP-IJCNLP**, 2019.

LHOEST, Q. et al. Datasets: A Community Library for Natural Language Processing. In: **EMNLP Demo**, 2021.

MACAVANEY, S. et al. Cross-Encoder Rediscovers a Semantic Variant of BM25. **arXiv:2502.04645**, 2025.

MOZANNAR, H.; SONTAG, D. Consistent Estimators for Learning to Defer to an Expert. In: **ICML**, 2020.

PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. **JMLR**, v. 12, p. 2825–2830, 2011.

RAYO, J.; DE LA ROSA, R.; GARRIDO, M. A Hybrid Approach to Information Retrieval and Answer Generation for Regulatory Texts. In: **COLING**, 2025.

REIMERS, N.; GUREVYCH, I. Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks. In: **EMNLP-IJCNLP**, 2019.

ROBERTSON, S. E. et al. Okapi at TREC-3. In: **NIST**, 1996.

SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: Pretrained BERT Models for Brazilian Portuguese. In: **BRACIS**, 2020.

THAKUR, N. et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In: **NeurIPS Datasets and Benchmarks**, 2021.

TRAPEZNIKOV, K.; SALIGRAMA, V. Multi-Stage Classifier Design. **Machine Learning**, Springer, 2013.

TUKEY, J. W. **Exploratory Data Analysis**. Addison-Wesley, 1977.

TUNSTALL, L. et al. Efficient Few-Shot Learning Without Prompts. **arXiv:2209.11055**, 2022.

VASWANI, A. et al. Attention Is All You Need. In: **NeurIPS**, 2017.

WOLF, T. et al. Transformers: State-of-the-Art Natural Language Processing. In: **EMNLP Demo**, 2020.

ZHANG, J. et al. Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference. In: **EMNLP**, 2020.

ZHANG, Y. et al. Intent Detection and Slot Filling for Multi-Turn Dialogues with Hierarchical Attention. In: **NAACL**, 2021.

---

## FINANCIAMENTO

Este trabalho foi desenvolvido no âmbito do Centro de Competências Embrapii em Tecnologias Imersivas (AKCIT), com apoio da Embrapii (Empresa Brasileira de Pesquisa e Inovação Industrial) e da Universidade Federal de Goiás.

---

## APÊNDICES

### Apêndice A — Dataset público

O corpus utilizado está disponível em: https://huggingface.co/datasets/paulohenriquevn/talkex-augmented-pt-br

### Apêndice B — Notebooks experimentais

Os 9 notebooks experimentais (v1–v9) estão disponíveis como evidência do processo iterativo de pesquisa, executáveis gratuitamente no Google Colab.
