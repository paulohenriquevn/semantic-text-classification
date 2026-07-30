---
slug: realtime-attendance-monitoring
date: 2026-07-29
generated_by: roadmap-init
questions_answered: 7
unresolved_dims: []
status: completed
---

# Roadmap grill: realtime-attendance-monitoring

### Q1/7: Problema-raiz e para quem dói

**Question:** Qual é o problema-raiz que este produto resolve, e para quem ele dói hoje?

**Recommended:** Operação de call center tem visibilidade quase zero do conteúdo e da trajetória emocional das conversas ao vivo. QA audita 1–2% das chamadas de forma retrospectiva; sinais críticos (intenção de cancelamento, escalonamento, quebra de compliance, raiva do cliente) só são detectados depois que o cliente saiu ou a falha regulatória aconteceu. Com o Macaw produzindo transcrição on-device de 100% das ligações a custo marginal ~$0, falta uma plataforma que transforme esse firehose em monitoramento acionável em tempo real. Dói para: supervisores/QA (cegos ao vivo, amostragem tardia), o negócio (churn evitável + risco de compliance), atendentes (sem coaching em tempo real).

**User answer:** Aceita a recomendação integralmente.

### Q2/7: Usuários primários

**Question:** Quem são os usuários primários (papel, interno vs externo, um time ou muitos)?

**Recommended:** Produto interno/B2B com três papéis: (1) Supervisor de operação — usuário-âncora, monitora N atendentes ao vivo, recebe alertas priorizados, drill-down na conversa ao vivo; se ele não usa, o produto falhou. (2) Analista de QA — de 1–2% para 100% de cobertura, busca semântica + regras, audita/rotula, consome mais o near-real-time/histórico 30d. (3) Gestor/Head — só dashboards agregados (tendências de sentimento, taxa de cancelamento por fila, aderência de script). Os três mapeiam nas três camadas do ADR-005 (quente ao vivo / near-real-time 30d / rollups). Atendente não é usuário direto na V1.

**User answer:** Aceita a recomendação integralmente.

### Q3/7: Escopo de V1 (must-have)

**Question:** O que é must-have para o produto estar vivo em V1?

**Recommended:** A linha completa "transcrição ao vivo → insight acionável na tela do supervisor", aproveitando o TalkEx existente: (1) ingestão em streaming das transcrições do Macaw (fan-in) → segmentação em turnos/janelas; (2) storage do ADR-005 operante (Postgres/Timescale 30d + pgvector + BM25, retention/expurgo nativo); (3) classificação online em cascata — regras/lexical (DSL existente) instantâneo + sentimento (ML) por turno/janela; (4) alerta em tempo real com evidência quando regra crítica dispara (cancelamento, escalonamento, sentimento negativo em queda); (5) tela de monitoramento ao vivo do supervisor (chamadas ativas + estado + drill-down com trechos destacados); (6) busca híbrida BM25+semântica sobre a janela de 30d para o QA; (7) dashboard agregado mínimo (continuous aggregates: taxa de cancelamento + distribuição de sentimento por fila/domínio). ML sofisticado (intenção fina, tópicos emergentes, resumo automático, score preditivo) fica pós-V1.

**User answer:** Aceita a recomendação integralmente.

### Q4/7: Fora de escopo (explícito)

**Question:** O que é tentador mas NÃO é para esta V1?

**Recommended:** (1) ASR/transcrição — é o Macaw, repo separado; a plataforma consome, não produz. (2) Coaching ao vivo para o próprio atendente — pós-V1 (muda o produto). (3) Discagem/PABX-CTI/barge-in — telefonia, não NLP; fora. (4) ML pesado offline (descoberta de intenção não supervisionada, clustering de tópicos, resumo por LLM, score preditivo) — pipeline offline do KB_Complementar, pós-V1. (5) LLM no caminho online — axioma "LLMs só offline"; nada de LLM no tempo real da V1. (6) Compliance/LGPD como produto (consentimento, retenção configurável por cliente, RBAC fino, multi-tenant) — V1 assume um tenant + retenção fixa 30d. (7) Mobile / alertas externos (push/Slack/e-mail) — V1 é web. (8) Fine-tuning de modelo de sentimento próprio — V1 usa modelo de sentimento pré-treinado PT-BR; treinar do zero é pós-V1/dissertação.

**User answer:** Aceita a recomendação integralmente.

### Q5/7: Restrições rígidas

**Question:** Quais são as restrições rígidas (stack, compliance, prazo, time, alvos de runtime)?

**Recommended:** Stack: Python + FastAPI (já no projeto); banco do ADR-005 (Postgres + TimescaleDB + pgvector + BM25 via pg_search/tsvector, sem ClickHouse/Qdrant na V1); frontend React+Vite+TS+Tailwind (reaproveitar o demo); modelos leves online (embedding paraphrase-multilingual-MiniLM-L12-v2 já no projeto + sentimento PT-BR pré-treinado), CPU-first (coerência com Macaw), GPU não é premissa. Runtime: retrieval híbrido p95 < 200 ms; alerta chegada→tela < ~2 s [A VALIDAR]; ingestão ~150–300 turnos/s. Compliance: LGPD como postura de arquitetura (retenção 30d + expurgo + anonimização de retraining), não feature de produto na V1. Time pequeno; disciplina de medir antes de escalar e sempre comparar contra baseline BM25.

**User answer:** Aceita a recomendação integralmente.

### Q6/7: Critério de sucesso mensurável (V1 shipada)

**Question:** Qual é o critério mensurável para dar V1 como shipada?

**Recommended:** Num piloto real (uma fila/domínio, dezenas de atendentes ao vivo), todas passam: (1) cobertura 100% das chamadas ativas na tela do supervisor em tempo real; (2) latência de alerta turno crítico→tela p95 < ~2 s [A VALIDAR]; (3) retrieval híbrido p95 < 200 ms sob carga concorrente (piloto ADR-005); (4) sentimento macro-F1 ≥ ~0,70 no rotulado PT-BR, batendo baseline BM25/léxico (alinhado a H2 ≈ 0,72 dos experimentos); (5) utilidade validada — supervisor age sobre ≥ N alertas/dia com evidência e precisão dos alertas críticos (cancelamento/escalonamento) ≥ 0,8 (não afogar em falso-positivo); (6) expurgo funcionando — bruto > 30d dropado automaticamente + amostra anonimizada de retraining exportada antes do drop.

**User answer:** Aceita a recomendação integralmente.

### Q7/7: North-star metric

**Question:** Qual métrica, quando se move, diz que o produto está vencendo (diferente de shipado)?

**Recommended:** Taxa de "intervenção acionada por alerta que mudou o desfecho da chamada" — % de chamadas de risco em que um alerta em tempo real levou a uma ação e o desfecho melhorou (retenção, de-escalonamento, compliance corrigido na hora). Em uma linha: "conversas salvas por unidade de tempo". Resiste a gaming (não é nº de alertas nem acurácia do modelo, é impacto no desfecho) e conecta com a economia do Macaw (100% das chamadas visíveis a custo ~0 → quanto disso vira valor). Proxy inicial enquanto "desfecho" não é instrumentável ponta-a-ponta: taxa de alertas acionados pelo supervisor com evidência.

**User answer:** Aceita a recomendação integralmente.
