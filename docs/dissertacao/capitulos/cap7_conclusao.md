# Capítulo 7 — Conclusão

Este capítulo sintetiza as contribuições da dissertação, reconhece suas limitações, identifica direções para trabalhos futuros e encerra com considerações sobre o impacto potencial da pesquisa.

---

## 7.1 Síntese das Contribuições

Esta dissertação investigou em que medida uma arquitetura híbrida em cascata — combinando retrieval lexical (BM25), retrieval semântico (embeddings densos), classificação supervisionada e regras determinísticas auditáveis — oferece vantagens sobre abordagens que dependem de um paradigma isolado na análise de conversas de atendimento. A avaliação experimental sobre um corpus de 2.257 conversas em português brasileiro, com 9 classes de intent, produziu evidências mistas: confirmação forte para a representação multi-nível (H2), resultados inconclusivos para retrieval híbrido (H1) e regras como features (H3), e refutação da cascata sob as condições testadas (H4). As quatro contribuições principais são detalhadas a seguir.

**Contribuição 1: Evidência empírica da complementaridade híbrida.** O Hybrid-RRF apresentou MRR numericamente superior ao BM25 (0,826 vs 0,802), porém a diferença não atingiu significância estatística (p=0,103) neste corpus. Não se pode, portanto, afirmar superioridade com confiança estatística, embora a consistência do ganho em todas as métricas de ranking sugira um efeito real que um corpus maior poderia confirmar. Este resultado revela que o BM25 é surpreendentemente forte em domínios conversacionais com marcadores lexicais explícitos — achado consistente com Harris (2025). A significância contra o ANN isolado (p=0,032) confirma que a fusão estabiliza o retrieval semântico.

**Contribuição 2: Classificação eficaz sem treinamento de modelos de linguagem.** A combinação de embeddings multilinguais (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensões) com LightGBM atinge Macro-F1=0,715 em 9 classes, beneficiando-se de modelos pré-treinados publicamente disponíveis, sem necessidade de fine-tuning ou treinamento próprio de modelos neurais. Enquanto abordagens como BERTaú (Finardi et al., 2021) requerem treinamento from scratch com 14,5 GB de dados proprietários e infraestrutura GPU, o TalkEx treina seus classificadores em ~6 segundos em CPU. O estudo de ablação confirmou que embeddings são o componente mais crítico (contribuição de +35,0pp), enquanto features lexicais adicionam +1,5pp complementar.

**Contribuição 3: Investigação de regras determinísticas como features de classificação.** Propusemos e avaliamos uma estratégia de integração ML+Regras onde o resultado da avaliação de regras DSL serve como feature adicional do classificador (rules-as-feature), em vez de substituir a predição do ML (override). A estratégia rules-as-feature apresentou ganho direcional de Macro-F1 de 0,709 para 0,714 (+0,5pp), com impacto expressivo em classes críticas (cancelamento: F1 0,951 → 1,000). Contudo, a diferença global não atingiu significância estatística (Wilcoxon p=0,4669, IC 95% [−0,015; +0,033]), tornando o resultado inconclusivo quanto à confirmação da hipótese H3. O achado qualitativo permanece relevante: a arquitetura rules-as-feature demonstra viabilidade da integração sem degradação, enquanto a estratégia de override degradou o Macro-F1 para 0,624 — evidenciando que a decisão final deve permanecer com o classificador. A confirmação estatística requer corpus maior, particularmente nas classes críticas onde o efeito é mais pronunciado (n=32 para cancelamento).

**Contribuição 4: Framework arquitetural com inferência cascata.** H4 foi refutada em seu critério primário: nenhuma configuração de cascata atingiu a redução de custo de 40%. A razão de custo entre estágios (1,5×) foi insuficiente, dado que ambos utilizam os mesmos embeddings pré-computados. O resultado secundário — degradação mínima de F1 (Δ=0,003) com roteamento por confiança — demonstra a viabilidade do princípio de inferência cascateada para cenários com diferencial de custo mais expressivo.

---

## 7.2 Limitações

Identificamos cinco limitações que circunscrevem o escopo das conclusões:

1. **Corpus sintético.** O dataset foi expandido via geração por LLM, o que pode produzir conversas com linguagem mais regular e menos ruidosa que interações reais de call center. Embora o protocolo de validação Phase 0.5 não tenha identificado riscos críticos, a generalização dos resultados para dados de produção requer validação adicional.

2. **Domínio e idioma únicos.** Os experimentos foram conduzidos exclusivamente em conversas de atendimento em português brasileiro. A transferência para outros domínios (jurídico, saúde, financeiro) e idiomas não foi avaliada, embora o uso de embeddings multilinguais sugira potencial de generalização.

3. **Escala limitada.** O corpus final de 2.257 conversas ficou aquém das ~3.500 inicialmente projetadas (Capítulo 5), devido à execução parcial da estratégia de expansão sintética. Embora a distribuição de classes tenha permanecido estratificada, o tamanho reduzido pode ter limitado o poder estatístico dos testes de hipóteses — particularmente para H1, onde a diferença Hybrid-RRF vs BM25 não atingiu significância (p=0,103). Além disso, o corpus é substancialmente menor que os utilizados por trabalhos correlatos (BERTaú: 14,5 GB; Rayo: 27.869 perguntas). Em cenários de produção com milhões de conversas, o comportamento dos componentes pode diferir — particularmente o BM25, que se beneficia de corpora maiores.

4. **Regras definidas pelo pesquisador.** As regras DSL foram elaboradas pelo autor com base em análise qualitativa, não por especialistas do domínio de atendimento. Em operação real, a qualidade das regras depende criticamente do conhecimento do negócio de quem as define.

5. **Ausência de ASR real.** Todas as conversas utilizam texto limpo, sem ruído de transcrição automática (ASR). Em produção, erros de transcrição podem degradar tanto o BM25 (termos errados) quanto embeddings (contexto corrompido).

---

## 7.3 Trabalhos Futuros

Os resultados e limitações desta dissertação sugerem diversas direções para trabalhos futuros:

**Fine-tuning de embeddings no domínio.** Os embeddings pré-treinados genéricos já alcançam F1=0,715. Fine-tuning com dados conversacionais de atendimento — via contrastive learning com pares (query, conversa relevante) — pode melhorar a separação inter-classe, especialmente para pares de difícil discriminação como "outros↔saudacao" (similaridade 0,972) e "compra↔duvida_produto" (0,897).

**Validação com dados reais de produção.** Parceria com operadores de contact center para avaliar o pipeline em conversas reais com ruído de ASR, linguagem informal e distribuição natural de classes. Isso endereçaria diretamente a principal limitação do estudo.

**Expansão do motor de regras.** Adição de predicados semânticos (e.g., `semantic.intent_score("cancelamento") > 0.8`) e contextuais (e.g., `context.turn_window(3).contains("procon")`) às regras DSL, aumentando sua expressividade e reduzindo dependência de keywords literais.

**Cascata com estágios de custo diferenciado.** Implementação de um estágio leve baseado exclusivamente em regras determinísticas e features lexicais (sem embeddings), seguido de estágio com inferência neural completa. Isso amplificaria os benefícios de custo demonstrados conceitualmente na H4.

**Intent discovery offline com LLMs.** Utilização de modelos de linguagem de grande porte em modo offline para descoberta de novos intents emergentes via clustering de embeddings, seguido de revisão humana e promoção ao taxonomy — conforme proposto por Huang e He (2025).

**Active learning para redução de custo de anotação.** Seleção inteligente de amostras para anotação humana baseada na incerteza do classificador, priorizando conversas próximas à fronteira de decisão entre classes confusas.

---

## 7.4 Considerações Finais

Esta dissertação demonstrou que é possível construir um sistema de análise de conversas de atendimento com qualidade competitiva — Macro-F1=0,715 em 9 classes — sem treinamento de modelos neurais de linguagem e sem infraestrutura GPU. A contribuição mais robusta é a evidência de que embeddings pré-treinados multilinguais, combinados com features lexicais e classificadores gradient boosting, constituem uma representação eficaz para classificação conversacional (H2, p<0,05 em 9/9 classes). A integração de regras determinísticas como features mostrou-se promissora mas estatisticamente inconclusiva no corpus atual (H3), e a inferência cascata não atingiu os critérios de eficiência projetados (H4). A arquitetura proposta (TalkEx) oferece um framework modular que integra retrieval lexical, embeddings pré-treinados e regras determinísticas auditáveis — cujo potencial completo requer validação em corpora maiores.

Os resultados reforçam uma mensagem pragmática para a comunidade de NLP aplicado: investimentos massivos em treinamento de modelos nem sempre são necessários. Embeddings pré-treinados multilinguais, classificadores gradient boosting e regras determinísticas auditáveis constituem uma combinação poderosa — acessível a organizações sem infraestrutura de GPU — que coloca a qualidade analítica ao alcance de qualquer operação de atendimento.

No contexto brasileiro, onde a maioria das operações de contact center não dispõe de equipes de ML dedicadas ou infraestrutura de treinamento, esta abordagem de "ML sem treinamento de modelo" representa uma contribuição com potencial de impacto prático direto: permite que organizações extraiam inteligência de suas conversas utilizando hardware convencional, com a garantia de auditabilidade que compliance e governança exigem.
