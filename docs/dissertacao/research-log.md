# Caderno de Pesquisa — Dissertação de Mestrado

**Pesquisador:** Paulo
**Projeto:** TalkEx — Conversation Intelligence Engine
**Início:** 2026-03-10
**Fase atual:** Construção do problema e definição metodológica

**Documentos complementares:**
- Protocolo de execução (passos reprodutíveis): [`steps-log.md`](steps-log.md)
- Estrutura de capítulos: [`estrutura-capitulos.md`](estrutura-capitulos.md)
- Desenho experimental: [`desenho-experimental.md`](desenho-experimental.md)

---

## Pergunta de Pesquisa

**Como projetar um sistema de análise de conversas de atendimento que combine efetivamente retrieval lexical, retrieval semântico e regras determinísticas auditáveis, mantendo qualidade, explicabilidade e eficiência computacional?**

---

## Diário de Pesquisa

### 2026-03-10 — Sessão 1: Do artefato à pergunta

#### O ponto de partida

O projeto TalkEx já existia como plataforma de engenharia — um pipeline NLP com módulos de ingestão, embeddings, retrieval BM25, classificação e um motor de regras baseado em DSL/AST. A pergunta que iniciou esta sessão foi: **este artefato pode sustentar uma dissertação de mestrado?**

A resposta é sim, mas exigiu uma inversão de perspectiva. Na engenharia, o código é o fim. Na pesquisa, o código é o meio — o artefato que operacionaliza uma hipótese. A dissertação não é sobre o TalkEx; é sobre **o que o TalkEx permite demonstrar**.

#### Construção da lacuna

Analisei 5 trabalhos que formam a base teórica do projeto. O que me chamou atenção ao lê-los em conjunto:

**Harris (2025)** mostrou que BM25 supera embeddings semânticos off-the-shelf em documentos médicos estruturados. Resultado contraintuitivo — mas faz sentido: vocabulário consistente + estrutura rígida = sinais lexicais fortes. Limitação: usou apenas kNN para classificar, sem classificadores supervisionados. E o domínio é médico, não conversacional.

**Rayo et al. (2025)** demonstrou que o híbrido BM25+semântico supera ambos os isolados em textos regulatórios (Recall@10: 0.83 vs 0.81 vs 0.76). Resultado esperado, mas importante ter evidência. Usaram α=0.65 (65% peso semântico). Limitação: sem classificação, sem regras, sem dados conversacionais.

**Huang & He (2025)** propuseram usar LLMs para transformar clustering em classificação — gerar labels e depois classificar. Resultados próximos ao upper bound teórico. Mas usam LLM online, o que é proibitivo em escala. E não há rastreabilidade de evidência.

**Lyu et al. (2025)** mostraram que attention pooling melhora classificação (F1: 0.89, AUC: 0.94). Mas testaram apenas em AG News (textos curtos de notícias), não em conversas multi-turn.

**AnthusAI** demonstrou o princípio "embeddings representam, classificadores decidem" — BERT+LogReg alcança acurácia comparável a Ada-2+LogReg. Mas testaram classificação binária simples.

**A lacuna que vi:** nenhum desses trabalhos combina as três abordagens (lexical + semântico + regras determinísticas) em dados conversacionais multi-turn. Cada um resolve uma parte do problema. A contribuição possível é demonstrar que a combinação é maior que a soma das partes.

#### Formulação da tese

A tese emergiu da lacuna: uma arquitetura que integra os 3 paradigmas, operando sobre representações multi-nível de conversas, deve superar abordagens isoladas.

Desdobrei em 4 hipóteses falsificáveis — deliberadamente com critérios numéricos para que possam ser refutadas:

- **H1:** Híbrido > isolados em retrieval (testável com Recall@K)
- **H2:** Multi-nível > nível único em classificação (testável com Macro-F1)
- **H3:** Regras melhoram precision em classes críticas sem degradar recall global (testável)
- **H4:** Cascata reduz custo ≥40% com Δ F1 < 2% (testável)

**Reflexão importante:** H4 é a mais arriscada. Se a cascata descartar conversas ambíguas nos estágios baratos, o recall pode sofrer. Os thresholds de confiança por estágio serão o ponto crítico.

#### O dataset — uma tensão metodológica

Encontrei `RichardSakaguchiMS/brazilian-customer-service-conversations`. PT-BR, conversacional, papéis customer/agent, 9 intents, 3 sentimentos, 8 setores. Parece perfeito à primeira vista.

A análise exploratória revelou problemas que exigem reflexão:

**1. É sintético.** Todas as 944 conversas foram geradas por LLM NVIDIA. Isso não é necessariamente fatal — muita pesquisa em NLP usa dados sintéticos. Mas muda o que posso afirmar. Com dados sintéticos, a contribuição é **metodológica** (a arquitetura funciona, o pipeline é coerente, as hipóteses se confirmam neste cenário). Com dados reais, a contribuição seria **empírica** (funciona no mundo real). A dissertação precisa ser honesta sobre isso.

**2. É pequeno.** 944 conversas, 95 no teste. Com 9 classes, isso dá ~10 por classe no teste. Intervalos de confiança enormes. Um único erro muda precision em 10 pontos percentuais. Para qualquer afirmação estatisticamente significativa, preciso expandir.

**3. É artificialmente uniforme.** ~108 conversas por intent, ~313 por sentimento, ~15 por combinação intent×setor. Dados reais de call center são brutalmente desbalanceados — 40% dúvidas, 5% cancelamento, 1% fraude. A uniformidade facilita o treino mas esconde problemas reais como desempenho em classes raras.

**4. Os turnos são quase fixos.** 90% das conversas têm exatamente 8 turnos. Conversas reais variam de 2 a 50+. Isso prejudica H2, que depende de janelas de tamanho variável.

Porém, o que o dataset faz bem:
- **Linguagem brasileira informal autêntica** — "vc", "td", "show", "blz", "pq"
- **Diacríticos presentes** — 1.019 palavras com acentos, validando a normalização
- **Sinais lexicais discriminativos** — "cancelar" é quase exclusivo da classe cancelamento
- **Alta sobreposição em palavras comuns** — "obrigado", "bom", "fazer" em 9/9 intents, o que demonstra que BM25 puro terá dificuldade (bom para a tese)

**Tensão não resolvida:** expandir com mais dados sintéticos resolve o volume mas não resolve a artificialidade. Expandir com dados semi-reais (Reclame Aqui) resolve parcialmente a artificialidade mas introduz inconsistência de formato. Preciso decidir qual limitação é mais aceitável.

#### O que aprendi nesta sessão

1. A passagem de projeto de engenharia para pesquisa acadêmica requer inversão: o código serve à hipótese, não o contrário.
2. A lacuna existe e é defensável — ninguém combinou os 3 paradigmas em dados conversacionais.
3. O dataset tem limitações sérias que precisam ser endereçadas com honestidade, não contornadas com truques.
4. As 4 hipóteses são testáveis e falsificáveis — isso é o mais importante.

#### Questões em aberto após esta sessão

- **Epistemológica:** Até que ponto resultados em dados sintéticos sustentam uma tese sobre "conversas de atendimento"? Onde está a linha entre "prova de conceito" e "evidência empírica"?
- **Metodológica:** Qual estratégia de expansão do dataset preserva melhor a validade dos resultados? Mais dados sintéticos (consistência) ou dados semi-reais (realismo)?
- **Prática:** O programa de mestrado e o orientador podem ter expectativas específicas sobre formato, normas, comitê. Essas variáveis ainda não são conhecidas.
- **Técnica:** O TalkEx já tem muitos módulos implementados. Os experimentos devem usar o código existente ou implementações limpas isoladas para o paper? (Código existente = mais realista mas mais variáveis; isolado = mais controlado mas menos representativo.)

---

## Revisão de Literatura — Leituras e Reflexões

### Harris (2025) — "Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents"
**arXiv:2505.11582v2**

**O que fez:** Comparou 7 métodos de embedding (TF, TF-IDF, BM25, word2vec, med2vec, MiniLM, mxbai) para classificar 1.472 documentos médicos em 7 classes usando kNN.

**Resultado central:** BM25 alcançou a maior acurácia preditiva e foi significativamente mais rápido que embeddings semânticos. Lexical superou semântico em dados altamente estruturados.

**O que me fez pensar:**
- O resultado é específico para documentos médicos (vocabulário controlado, estrutura rígida). Em conversas informais de call center, com gírias e paráfrases, o equilíbrio pode ser diferente.
- Usou apenas kNN como classificador. Com classificadores supervisionados (LogReg, GBM), embeddings poderiam se sair melhor — o embedding captura a representação, o classificador aprende os limites de decisão.
- Confirma nossa decisão de **sempre** ter BM25 como baseline. Nunca assumir superioridade semântica.

**Relevância para a tese:** Sustenta H1 indiretamente — se BM25 é forte mesmo em domínios favoráveis a ele, a combinação com semântico deveria ser ainda mais forte em domínios onde lexical sozinho falha (como conversas com paráfrases).

---

### Rayo et al. (2025) — "A Hybrid Approach to Information Retrieval and Answer Generation for Regulatory Texts"
**COLING 2025, arXiv:2502.16767v1**

**O que fez:** Híbrido BM25 + BGE fine-tuned para retrieval em 27.869 questões regulatórias. Score = α·semântico + (1-α)·lexical, com α=0.65. Também testou RAG com GPT-3.5 Turbo, GPT-4o Mini e Llama 3.1.

**Resultados centrais:**
| Sistema | Recall@10 | MAP@10 |
|---------|-----------|--------|
| BM25 baseline | 0.7611 | 0.6237 |
| BM25 custom | 0.7791 | 0.6415 |
| Semântico puro | 0.8103 | 0.6286 |
| **Híbrido** | **0.8333** | **0.7016** |

O híbrido superou em ambas as métricas. MAP@10 do híbrido foi significativamente superior ao semântico puro (0.70 vs 0.63), mesmo com o semântico tendo Recall@10 próximo.

**O que me fez pensar:**
- O α=0.65 (65% semântico, 35% lexical) sugere que mesmo quando o semântico domina, o lexical contribui decisivamente para a **precisão do ranking** (MAP). Isso faz sentido: termos exatos no topo do ranking melhoram MAP.
- Fine-tuning do embedding (BGE) fez diferença significativa (Recall@10: 0.81 vs 0.70 do modelo base). Para a dissertação, se usar embeddings off-the-shelf, preciso reconhecer essa limitação.
- O pipeline de normalização deles (expand contractions, lowercase, remove non-alphanumeric, stemming, bigrams) é similar ao que implementamos no TalkEx. Boa validação.

**Relevância para a tese:** Evidência direta para H1. Se funciona em textos regulatórios, o experimento em conversas testará se a conclusão se generaliza.

---

### Huang & He (2025) — "Text Clustering as Classification with LLMs"
**SIGIR-AP 2025, arXiv:2410.00927v3**

**O que fez:** Framework de 2 estágios — LLM gera labels em mini-batches, depois classifica textos nesses labels. Testado em 5 datasets (ArxivS2S, GoEmo, Massive-I/D, MTOP-I) com 18-102 clusters.

**Resultado central:** Performance próxima ao upper bound teórico (LLM_known_labels) e superior a K-means, DBSCAN, IDAS, PAS, ClusterLLM em ACC, NMI e ARI.

**O que me fez pensar:**
- A ideia de transformar clustering em classificação é elegante. Mas depende de LLM online (GPT-3.5-turbo), o que tem custo proibitivo para milhões de conversas.
- Para a dissertação, a aplicação é no **pipeline offline**: LLM gera taxonomia de intents, depois classificadores baratos operam online. Isso se alinha com o princípio "LLMs offline only" do TalkEx.
- O paper não discute explicabilidade — o LLM atribui um label, mas não fornece evidência rastreável de por quê. As regras AST do TalkEx preenchem essa lacuna.

**Relevância para a tese:** Não diretamente testado nas hipóteses, mas fundamenta a estratégia de discovery offline e reforça a necessidade de auditabilidade que as regras fornecem.

---

### Lyu et al. (2025) — "Advancing Text Classification with Large Language Models and Neural Attention Mechanisms"
**arXiv:2512.09444v1**

**O que fez:** Framework de classificação com encoder LLM + attention mechanism + combined pooling (mean + attention-weighted). Testado no AG News (4 classes, ~120K textos).

**Resultados:**
| Modelo | Precision | Recall | F1 | AUC |
|--------|-----------|--------|-----|-----|
| BERT | 0.87 | 0.85 | 0.86 | 0.91 |
| Transformer | 0.85 | 0.83 | 0.84 | 0.90 |
| **Proposto** | **0.90** | **0.88** | **0.89** | **0.94** |

Attention pooling melhorou F1 de 0.86 (BERT) para 0.89, e AUC de 0.91 para 0.94.

**O que me fez pensar:**
- AG News são textos curtos de notícias (1 parágrafo). Conversas de call center são multi-turn com 8+ turnos. O attention pooling deveria ser **ainda mais** benéfico em conversas longas, onde mean pooling dilui sinais críticos em meio a turnos irrelevantes ("oi, td bem?", "obrigado").
- O paper mostrou sensibilidade ao hidden dimension (peak em 512, queda com 768+) e ao class imbalance (recall cai de 0.88 para 0.80 com imbalance 1:6). Ambos são relevantes para nosso contexto.
- Não usa features lexicais nem estruturais — só embeddings. Nossa abordagem com features heterogêneas pode ir além.

**Relevância para a tese:** Fundamenta a decisão de testar attention pooling em H2, especialmente em janelas de contexto.

---

### AnthusAI — Semantic Text Classification (GitHub)

**O que fez:** Comparou Word2Vec, BERT e OpenAI Ada-2 como embeddings + logistic regression para classificação binária (questões sobre lei de imigração espanhola).

**Resultado central:** BERT e Ada-2 alcançaram desempenho equivalente e superior a Word2Vec. A conclusão: "the most powerful and expensive models may not always be necessary."

**O que me fez pensar:**
- Confirma o axioma "embeddings representam, classificadores decidem". Embeddings sem classificador supervisionado são insuficientes.
- Em um domínio específico, modelos menores (BERT) podem ser tão bons quanto modelos maiores (Ada-2). Para a dissertação, isso justifica usar sentence-transformers (E5, BGE) em vez de LLMs para embeddings online.
- A tarefa era binária e simples. Em 9 classes com sobreposição lexical alta (como no nosso dataset), a dificuldade é outra. Features adicionais (lexicais, estruturais) provavelmente são necessárias.

**Relevância para a tese:** Sustenta a decisão arquitetural de separar representação (embeddings) de decisão (classificadores) e de usar modelos leves para inferência online.

---

## Análise Exploratória do Dataset

**Dataset:** `RichardSakaguchiMS/brazilian-customer-service-conversations`
**Data da análise:** 2026-03-10
**Licença:** Apache 2.0
**Origem:** Sintético (gerado por LLM NVIDIA)

### Estatísticas Estruturais

| Dimensão | Valor |
|----------|-------|
| Total de conversas | 944 |
| Splits | Train 755 / Val 94 / Test 95 |
| Total de mensagens | ~7.370 |
| Turnos por conversa | 6-8 (90.4% = 8, mediana 8, stdev 0.6) |
| Palavras por mensagem | Média 23.6 (customer: 16.6, agent: 30.7) |
| Palavras por conversa | Média 184.6 (min 79, max 321) |
| Palavras únicas com diacríticos | 1.019 |
| Idioma | PT-BR com informalismo (vc, td, show, blz, pq) |

### Distribuição de Intents

| Intent | Total | % | Train | Test |
|--------|-------|---|-------|------|
| reclamacao | 108 | 11.4% | 84 | 11 |
| duvida_produto | 108 | 11.4% | 80 | 11 |
| saudacao | 108 | 11.4% | 88 | 12 |
| suporte_tecnico | 108 | 11.4% | 88 | 10 |
| duvida_servico | 108 | 11.4% | 87 | 13 |
| compra | 107 | 11.3% | 85 | 8 |
| cancelamento | 106 | 11.2% | 79 | 17 |
| elogio | 98 | 10.4% | 85 | 6 |
| outros | 93 | 9.9% | 79 | 7 |

Sentimentos: positive 33.7%, neutral 33.3%, negative 33.1% (uniforme).
Setores: 8 setores de 9.4% a 14.3% (quase uniforme).

### Sinais Lexicais por Intent

Palavras com alto poder discriminativo (exclusivas no top-20):
- **cancelamento**: "cancelar" (184 ocorrências), "email", "internet"
- **compra**: "compra", "contratar", "online"
- **reclamacao**: "ontem", "tive", "errado", "deu"
- **suporte_tecnico**: "app", "dando", "consigo", "tentar", "erro" (123x)
- **elogio**: "elogio", "atendimento", "super"

Palavras sem poder discriminativo (aparecem em 9/9 intents): "obrigado", "bom", "fazer".

### Amostra: Conversa de Cancelamento

```
[CUSTOMER] Olá, vc pode me ajudar a cancelar o meu plano de internet?
           Td isso ta muito caro e o sinal ta muito ruim
[AGENT   ] Beleza, posso te ajudar com isso. Qual é o seu número de
           telefone ou código do cliente...
[CUSTOMER] Meu numero é 123456789, show, espero que vc consiga resolver
           isso rápido
[AGENT   ] Ta bom, encontrei o seu cadastro. Para cancelar o plano,
           preciso saber se você tem algum equipamento...
```

Linguagem informal autêntica: "vc", "td", "ta", "show", "beleza". Diacríticos misturados: "número" vs "numero". Perfeito para testar normalização.

---

### 2026-03-10 — Sessão 2: O problema do dataset

#### Mapeamento do cenário de dados PT-BR

Fiz uma busca sistemática por datasets PT-BR de atendimento ao cliente. O resultado foi revelador: **não existe nenhum dataset público de conversas multi-turn de call center em português brasileiro** além do que já temos (RichardSakaguchiMS). O que existe:

- **Consumidor.gov.br** — Massivo (325k+ reclamações por trimestre), labels oficiais (segmento, assunto, problema), gratuito. Mas são reclamações em texto único, sem estrutura de turnos customer/agent.
- **Reclame Aqui** — 7k reclamações de telecom com 14 categorias (rdemarqui). Também texto único, e legalmente cinza (scraping).
- **B2W-Reviews01** — 130k reviews de produtos, CC BY-NC-SA. Ratings, não intents.
- **Bitext** — 27 intents de customer service, Apache 2.0, mas em inglês. Excelente taxonomia de referência.
- **AxonData** — Multi-turn call center real, mas comercial/restrito.

**Achado importante para a dissertação:** A inexistência de dados conversacionais abertos PT-BR é, em si, uma lacuna que justifica a criação do corpus experimental. Isso fortalece a contribuição — não estamos apenas propondo uma arquitetura, estamos criando o cenário de avaliação para ela.

#### A decisão de expansão

Considerei 4 estratégias:

**Estratégia 1 — Expansão puramente sintética.** Gerar mais conversas com LLM seguindo os mesmos padrões. Resolve o volume mas não a artificialidade. Internamente consistente, mas as afirmações permanecem limitadas a "funciona em dados sintéticos".

**Estratégia 2 — Corpus semi-real (Consumidor.gov.br).** Usar reclamações reais como base. Linguagem autêntica, labels oficiais. Mas são textos únicos — sem turnos, sem interação customer/agent. Não suportam H2 (multi-nível depende de context windows multi-turn) nem H1 em sua forma completa (retrieval sobre janelas de contexto).

**Estratégia 3 — Conversacionalização de reclamações reais.** Pegar reclamações do Consumidor.gov.br e usar LLM para transformar cada reclamação em uma conversa multi-turn simulada (customer expõe o problema, agent responde). Híbrido interessante mas introduz um viés duplo: o texto original é real, a estrutura conversacional é sintética.

**Estratégia 4 — Expansão sintética controlada + validação em dados reais.** Duas camadas:
- **Corpus primário** (sintético expandido): 3.000-4.000 conversas geradas com variabilidade controlada — turnos variáveis (4-20), distribuição de classes realista (desbalanceada), ruído lexical, níveis de formalidade mistos. Para H1-H4.
- **Corpus secundário** (real, validação): Amostra do Consumidor.gov.br como teste de generalização. Classificadores treinados no corpus primário, avaliados no corpus secundário (single-text). Não testa H1/H2 completos mas testa a transferência dos modelos.

#### Por que escolhi a Estratégia 4

A Estratégia 4 é a mais defensável metodologicamente por três razões:

1. **Controle experimental.** O corpus primário sintético permite controlar variáveis que dados reais não permitem: distribuição de classes, comprimento de turnos, nível de ruído. Isso é essencial para ablation studies e para isolar o efeito de cada componente (H1-H4). Sem controle, não se sabe se o resultado vem da arquitetura ou da distribuição dos dados.

2. **Validação externa.** O corpus secundário real (Consumidor.gov.br) fornece uma medida de generalização. Se um classificador treinado em conversas sintéticas consegue classificar reclamações reais com desempenho razoável, isso sugere que as representações aprendidas capturam semântica genuína, não artefatos do gerador. Se falhar, isso é igualmente informativo — e é honesto.

3. **Posicionamento acadêmico claro.** A dissertação se posiciona como **contribuição metodológica com verificação de transferência**: "propusemos uma arquitetura, validamos em cenário controlado, e verificamos a transferência para dados reais com [resultado X]". Isso é mais forte que "testamos em dados sintéticos" e mais honesto que "testamos em dados reais" (quando o corpus é semi-real).

#### Especificação da expansão sintética

O corpus expandido deve corrigir as 4 fragilidades identificadas no dataset original:

| Fragilidade | Correção | Implementação |
|-------------|----------|---------------|
| **Tamanho (944)** | Expandir para ~3.500 | Geração em batches por intent |
| **Turnos fixos (90% = 8)** | Turnos variáveis 4-20 | Distribuição log-normal, média 8, stdev 4 |
| **Distribuição uniforme** | Distribuição realista | dúvida ~35%, reclamação ~20%, cancelamento ~8%, elogio ~5%, etc. |
| **Uniformidade lexical** | Variabilidade controlada | Personas variadas, níveis de formalidade, erros de digitação |

**Parâmetros de geração:**
- **Modelo gerador:** Claude ou GPT-4 (offline, batch mode)
- **Seed conversations:** usar as 944 existentes como exemplos few-shot
- **Taxonomia de intents:** manter as 9 existentes (consistência com análise exploratória)
- **Personas:** 5 perfis de cliente (formal, informal, irritado, idoso, jovem/gírias)
- **Setores:** manter os 8 existentes
- **Sentimentos:** distribuição realista (30% neutral, 40% negative, 30% positive — mais negativos em call center)

**Distribuição-alvo de intents (inspirada em call centers reais):**

| Intent | % alvo | Conversas (~3.500) |
|--------|--------|-------------------|
| duvida_produto | 18% | ~630 |
| duvida_servico | 17% | ~595 |
| reclamacao | 20% | ~700 |
| suporte_tecnico | 15% | ~525 |
| compra | 10% | ~350 |
| cancelamento | 8% | ~280 |
| saudacao | 5% | ~175 |
| elogio | 4% | ~140 |
| outros | 3% | ~105 |

**Splits do corpus expandido:**
- Train: 70% (~2.450)
- Val: 15% (~525)
- Test: 15% (~525)
- Stratified split preservando distribuição de classes
- Seed fixo: 42

#### Consumidor.gov.br como corpus de validação

**Plano de uso:**
1. Baixar 1 trimestre de reclamações (CSV, ~325k registros)
2. Filtrar por setores compatíveis com os 8 setores do corpus primário
3. Mapear categorias oficiais (segmento + assunto + problema) para as 9 intents
4. Selecionar amostra estratificada: ~2.000 reclamações (seguindo distribuição-alvo)
5. Avaliar classificadores treinados no corpus primário

**Limitação importante:** Consumidor.gov.br não tem estrutura multi-turn. Os classificadores serão avaliados em modo "Conv-only" (embedding da conversa/texto completo), não em modo multi-nível. Isso testa H2 parcialmente — se multi-nível supera single-level no corpus primário mas transfere pior para texto único, a interpretação requer nuance.

#### Reflexão epistemológica

Usar dados sintéticos em pesquisa de NLP é cada vez mais aceito — vários papers recentes (incluindo Huang & He 2025) usam labels gerados por LLM. A questão não é se o dado é sintético, mas se o pesquisador é **honesto** sobre as limitações que isso impõe.

O que posso afirmar com dados sintéticos:
- A arquitetura é coerente e funcional
- Os componentes se complementam (ou não) de forma mensurável
- A cascata reduz custo (ou não) com degradação controlada
- As regras melhoram precision (ou não) em cenários específicos

O que NÃO posso afirmar:
- Que os resultados se replicam com dados reais de call center
- Que os thresholds ótimos (α, thresholds de cascata) transferem para outros domínios
- Que a taxonomia de intents é adequada para operações reais

A verificação com Consumidor.gov.br é um passo na direção da segunda afirmação, mas não a resolve completamente. A dissertação deve ser explícita sobre isso no Cap. 7 (Limitações e Trabalhos Futuros).

---

## Questões Abertas

### Resolvidas nesta sessão

1. ~~**Dataset:** Qual estratégia de expansão usar?~~ → **Estratégia 4** — Expansão sintética controlada + validação em dados reais (Consumidor.gov.br)
2. ~~**Escopo de afirmações:** Prova de conceito ou estudo empírico?~~ → **Contribuição metodológica com verificação de transferência**

### Ainda abertas

1. **Código existente vs isolado:** Usar módulos do TalkEx diretamente nos experimentos ou reimplementar de forma controlada?
2. **Programa e orientador:** Há definição? Impacta normas, formato, expectativas da banca.
3. **Modelo gerador:** Claude vs GPT-4 vs outro para expansão sintética? Custo, qualidade, licença?
4. **Mapeamento Consumidor.gov.br → 9 intents:** É viável? As categorias oficiais (segmento/assunto/problema) mapeiam para intents de conversa?

### Para a revisão de literatura (Cap. 3)

- Buscar trabalhos específicos em **conversation intelligence para call centers** — o que existe academicamente?
- Buscar trabalhos em **cascaded inference** para NLP — quem mais fez isso e com quais resultados?
- Buscar trabalhos em **DSL para regras semânticas** — existe precedente acadêmico ou é prática industrial?
- Quantificar a lacuna: existe algum trabalho que combine os 3 paradigmas? Se sim, a tese precisa ser ajustada.

---

## Artefatos Produzidos

| # | Artefato | Caminho | Natureza |
|---|----------|---------|----------|
| 1 | Estrutura de capítulos | `docs/dissertacao/estrutura-capitulos.md` | Planejamento |
| 2 | Desenho experimental | `docs/dissertacao/desenho-experimental.md` | Metodologia |
| 3 | Caderno de pesquisa | `docs/dissertacao/research-log.md` | Processo |
