# RESUMO EXPANDIDO
# RELATÓRIO TÉCNICO — PESQUISA APLICADA

## AUTORIA

Leão, Cauê Cavichioli¹; Nascimento, Paulo Henrique Vieira¹; Sousa, Suele Susan Feitosa¹

¹Centro de Competência Embrapii em Tecnologias Imersivas (AKCIT), Universidade Federal de Goiás.

E-mail: *(inserir e-mail do autor correspondente)*

Orientador(a): *(inserir nome)*

---

## 1 INTRODUÇÃO

A classificação automática de intenções em conversas de atendimento ao cliente é um requisito operacional para call centers em escala. Sistemas tradicionais baseados em modelos supervisionados (LightGBM, BERT fine-tuned) exigem retreinamento a cada nova classe de intenção, criando um gargalo operacional quando equipes de negócio precisam ajustar taxonomias dinamicamente. A classificação por similaridade via KNN sobre embeddings (Zhang et al., 2020) permite taxonomia dinâmica — novas classes são adicionadas com exemplares, sem retreino — mas depende da qualidade das representações. A fusão de sinais lexicais (BM25) e semânticos (embeddings) tem demonstrado ganhos em information retrieval (Bruch et al., 2023), mas sua aplicação para classificação de intenções em conversas em português brasileiro (PT-BR) permanece sub-explorada.

O objetivo deste trabalho é desenvolver e avaliar o TalkEx, uma arquitetura híbrida que combina BM25 e embeddings sobre janelas de contexto com marcadores de falante para classificação de intenções em conversas PT-BR, mantendo taxonomia dinâmica. Investigamos 5 hipóteses: (H1) fusão léxico-semântica melhora sobre componentes isolados; (H2) BM25 domina sobre embeddings em PT-BR; (H3) encoder PT-BR (BERTimbau) supera multilíngues; (H4) dados sintéticos via LLM são essenciais; (H5) regras multi-sinal oferecem interpretabilidade sem custo de accuracy.

---

## 2 DESCRIÇÃO DA TECNOLOGIA

O TalkEx opera em pipeline: (1) segmentação de turnos com identificação de falante; (2) janelas deslizantes de N turnos com marcadores `[customer]`/`[agent]`; (3) codificação via sentence transformer (MiniLM, 384d, congelado); (4) classificação por KNN com voto ponderado; (5) agregação janela→conversa. O sistema implementa 6 métodos: Embedding KNN, BM25 KNN, Hybrid (fusão linear α·emb + (1−α)·bm25), Rerank (BM25→embedding), Cascade (regras→Hybrid) e Routing (Hybrid→regras corrigem). Um motor de regras multi-sinal com 4 famílias de predicados (lexical, estrutural, contextual, semântico) foi portado de um sistema de produção para oferecer interpretabilidade.

---

## 3 PROCEDIMENTOS METODOLÓGICOS

**Dataset:** 2.120 conversas PT-BR (676 originais + 1.444 sintéticas via Claude Sonnet), 8 classes balanceadas, publicado no HuggingFace (`paulohenriquevn/talkex-augmented-pt-br`). Cada conversa gera em média 2,8 janelas de 5 turnos (5.927 janelas totais).

**Protocolo:** 5 splits estratificados (70/15/15, nível conversa), Macro-F1/Micro-F1/Acurácia, Wilcoxon signed-rank com Holm-Bonferroni (m=3, α=0,05), Bootstrap CI 95%. Baselines: LogReg, SetFit, BERTimbau+KNN, MPNet+KNN. Ablação de 5 componentes.

---

## 4 TESTES E RESULTADOS

| Método | Macro-F1 | Tipo |
|---|---|---|
| **BERTimbau+KNN** | **0,768** | Exemplar (PT-BR, 768d) |
| Hybrid (α=0,5) | 0,748 | Híbrido |
| BM25-KNN | 0,748 | Lexical |
| LogReg+MiniLM | 0,719 | Supervisionado |
| MPNet+KNN | 0,716 | Exemplar (multilingual, 768d) |
| Emb-KNN (MiniLM) | 0,697 | Exemplar (384d) |

**H1 (Confirmada, direcional):** Hybrid supera BM25 em 4/5 seeds (+1,1pp, p=0,094). **H2 (Confirmada):** BM25 supera Embedding por +4,8pp em 5/5 seeds. Ablação: −BM25 custa −4,9pp vs −Embeddings custa −1,0pp. **H3 (Confirmada):** BERTimbau (0,768) supera Hybrid MiniLM (0,746) por +2,2pp. **H4 (Confirmada):** Remoção de sintéticos causa −21,9pp (0,746→0,527). **H5 (Confirmada):** Cascade (0,745) ≈ Hybrid (0,746); regras com 86,2% accuracy e −7,1% latência.

**Curva few-shot:** 50 exemplares/classe atingem 82% do máximo. **WINDOW_SIZE=all** atinge F1=0,766.

---

## 5 DISCUSSÃO

Os resultados confirmam que sinais lexicais (BM25) dominam em domínios com vocabulário restrito (Harris, 2025; Thakur et al., 2021). Encoders PT-BR superam multilíngues (+2,2pp vs Hybrid, +5,2pp vs MPNet), validando Souza et al. (2020). Dados sintéticos via LLM são o componente mais crítico (+22pp), demonstrando a viabilidade de data augmentation para NLU em PT-BR. Limitações: N=5 limita poder estatístico; 68% do corpus é sintético; sem cross-dataset evaluation.

---

## 6 CONSIDERAÇÕES FINAIS

O TalkEx demonstra que classificação por similaridade com taxonomia dinâmica é viável para conversas PT-BR, atingindo Macro-F1=0,77 (BERTimbau+KNN) e 0,75 (Hybrid) sem retreino. Os achados oferecem direções práticas para produção: usar BM25 como componente principal, encoder PT-BR quando disponível, e dados sintéticos para bootstrap de corpus pequenos.

---

## REFERÊNCIAS

BRUCH, S.; GAI, S.; INGBER, A. An Analysis of Fusion Functions for Hybrid Retrieval. **ACM TOIS**, v. 42, 2023.

HARRIS, L. Comparing Lexical and Semantic Vector Search Methods. **arXiv:2505.11582**, 2025.

SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: Pretrained BERT Models for Brazilian Portuguese. **BRACIS**, 2020.

THAKUR, N. et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models. **NeurIPS**, 2021.

TUNSTALL, L. et al. Efficient Few-Shot Learning Without Prompts. **arXiv:2209.11055**, 2022.

ZHANG, J. et al. Discriminative Nearest Neighbor Few-Shot Intent Detection. **EMNLP**, 2020.

---

## FINANCIAMENTO

Centro de Competências Embrapii em Tecnologias Imersivas (AKCIT), Universidade Federal de Goiás.
