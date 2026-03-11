# Capítulo 7 — Conclusão

Este capítulo sintetiza as contribuições da dissertação, reconhece suas limitações, identifica direções para trabalhos futuros e encerra com considerações sobre o impacto potencial da pesquisa.

---

## 7.1 Síntese das Contribuições

Esta dissertação investigou a tese de que uma arquitetura híbrida em cascata — combinando retrieval lexical (BM25), retrieval semântico (embeddings densos), classificação supervisionada e regras determinísticas auditáveis — alcança qualidade superior às abordagens que dependem de qualquer paradigma isolado na análise de conversas de atendimento. A avaliação experimental sobre um corpus de 2.257 conversas em português brasileiro, com 9 classes de intent, produziu quatro contribuições principais.

**Contribuição 1: Evidência empírica da complementaridade híbrida.** Demonstramos que o retrieval híbrido (Hybrid-RRF, MRR=0,826) supera os componentes isolados (BM25: 0,802, ANN: 0,799), embora a diferença em relação ao BM25 não tenha atingido significância estatística (p=0,103). Este resultado, longe de invalidar a abordagem híbrida, revela que o BM25 é surpreendentemente forte em domínios conversacionais com marcadores lexicais explícitos — achado consistente com Harris (2025). A significância contra o ANN isolado (p=0,032) confirma que a fusão estabiliza o retrieval semântico.

**Contribuição 2: Classificação eficaz sem treinamento de modelos de linguagem.** A combinação de embeddings pré-treinados multilinguais (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensões) com LightGBM atinge Macro-F1=0,715 em 9 classes, sem nenhum fine-tuning ou treinamento de modelo neural. Enquanto abordagens como BERTaú (Finardi et al., 2021) requerem treinamento from scratch com 14,5 GB de dados proprietários e infraestrutura GPU, o TalkEx treina seus classificadores em ~6 segundos em CPU. O estudo de ablação confirmou que embeddings são o componente mais crítico (contribuição de +35,0pp), enquanto features lexicais adicionam +1,5pp complementar.

**Contribuição 3: Regras determinísticas como features de classificação.** Propusemos e avaliamos uma estratégia de integração ML+Regras onde o resultado da avaliação de regras DSL serve como feature adicional do classificador (rules-as-feature), em vez de substituir a predição do ML (override). Esta estratégia elevou o Macro-F1 de 0,709 (ML-only) para 0,714, com impacto desproporcional em classes críticas: cancelamento atingiu F1=1,000 (precision e recall perfeitos). A alternativa de override, em contrapartida, degradou o Macro-F1 para 0,624, demonstrando que a decisão final deve permanecer com o classificador.

**Contribuição 4: Framework arquitetural com inferência cascata.** A implementação de inferência em cascata demonstrou viabilidade conceitual: com threshold t=0,90, o estágio leve resolve 2,7% das amostras com degradação desprezível de F1 (-0,003). Embora os benefícios de custo sejam limitados quando os estágios compartilham geração de embeddings, a arquitetura é extensível para cenários de produção onde o estágio leve utiliza regras simples (microsegundos) e o estágio completo requer inferência neural (centenas de milissegundos).

---

## 7.2 Limitações

Identificamos cinco limitações que circunscrevem o escopo das conclusões:

1. **Corpus sintético.** O dataset foi expandido via geração por LLM, o que pode produzir conversas com linguagem mais regular e menos ruidosa que interações reais de call center. Embora o protocolo de validação Phase 0.5 não tenha identificado riscos críticos, a generalização dos resultados para dados de produção requer validação adicional.

2. **Domínio e idioma únicos.** Os experimentos foram conduzidos exclusivamente em conversas de atendimento em português brasileiro. A transferência para outros domínios (jurídico, saúde, financeiro) e idiomas não foi avaliada, embora o uso de embeddings multilinguais sugira potencial de generalização.

3. **Escala limitada.** Com 2.257 conversas, o corpus é substancialmente menor que os utilizados por trabalhos correlatos (BERTaú: 14,5 GB; Rayo: 27.869 perguntas). Em cenários de produção com milhões de conversas, o comportamento dos componentes pode diferir — particularmente o BM25, que se beneficia de corpora maiores.

4. **Regras definidas pelo pesquisador.** As regras DSL foram elaboradas pelo autor com base em análise qualitativa, não por especialistas do domínio de atendimento. Em operação real, a qualidade das regras depende criticamente do conhecimento do negócio de quem as define.

5. **Ausência de ASR real.** Todas as conversas utilizam texto limpo, sem ruído de transcrição automática (ASR). Em produção, erros de transcrição podem degradar tanto o BM25 (termos errados) quanto embeddings (contexto corrompido).

---

## 7.3 Trabalhos Futuros

Os resultados e limitações desta dissertação sugerem diversas direções para trabalhos futuros:

**Fine-tuning de embeddings no domínio.** Os embeddings pré-treinados genéricos já alcançam F1=0,715. Fine-tuning com dados conversacionais de atendimento — via contrastive learning com pares (query, conversa relevante) — pode melhorar a separação inter-classe, especialmente para pares de difícil discriminação como "outros↔saudacao" (similaridade 0,972) e "compra↔duvida_produto" (0,897).

**Validação com dados reais de produção.** Parceria com operadores de contact center para avaliar o pipeline em conversas reais com ruído de ASR, linguagem informal e distribuição natural de classes. Isso endereçaria diretamente a principal limitação do estudo.

**Expansão do motor de regras.** Adição de predicados semânticos (e.g., `semantic.intent_score("cancelamento") > 0.8`) e contextuais (e.g., `context.turn_window(3).contains("procon")`) às regras DSL, aumentando sua expressividade e reduzindo dependência de keywords literais.

**Cascata com estágios de custo diferenciado.** Implementação de um estágio leve baseado exclusivamente em regras determinísticas e features lexicais (sem embeddings), seguido de estágio com inferência neural completa. Isso amplificaria os benefícios de custo demonstrados conceitualmente na H4.

**Intent discovery offline com LLMs.** Utilização de modelos de linguagem de grande porte em modo offline para descoberta de novos intents emergentes via clustering de embeddings, seguido de revisão humana e promoção ao taxonomy — conforme proposto por Huang e He (2023).

**Active learning para redução de custo de anotação.** Seleção inteligente de amostras para anotação humana baseada na incerteza do classificador, priorizando conversas próximas à fronteira de decisão entre classes confusas.

---

## 7.4 Considerações Finais

Esta dissertação demonstrou que é possível construir um sistema de análise de conversas de atendimento com qualidade competitiva — Macro-F1=0,715 em 9 classes, com precision perfeita em classes críticas — sem treinamento de modelos neurais de linguagem e sem infraestrutura GPU. A arquitetura proposta (TalkEx) combina o melhor de três paradigmas — retrieval lexical para robustez, embeddings pré-treinados para representação semântica, e regras determinísticas para auditabilidade — em um pipeline cascata que prioriza eficiência.

Os resultados reforçam uma mensagem pragmática para a comunidade de NLP aplicado: investimentos massivos em treinamento de modelos nem sempre são necessários. Embeddings pré-treinados multilinguais, classificadores gradient boosting e regras determinísticas auditáveis constituem uma combinação poderosa — acessível a organizações sem infraestrutura de GPU — que coloca a qualidade analítica ao alcance de qualquer operação de atendimento.

No contexto brasileiro, onde a maioria das operações de contact center não dispõe de equipes de ML dedicadas ou infraestrutura de treinamento, esta abordagem de "ML sem treinamento de modelo" representa uma contribuição com potencial de impacto prático direto: permite que organizações extraiam inteligência de suas conversas utilizando hardware convencional, com a garantia de auditabilidade que compliance e governança exigem.
