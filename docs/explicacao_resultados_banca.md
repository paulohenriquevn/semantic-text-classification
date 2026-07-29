# TalkEx — Explicação dos Resultados para a Banca

**Documento didático para compreensão dos resultados experimentais**

---

## O que é o TalkEx?

Imagine que uma empresa de telecom recebe **milhões de mensagens por mês** de clientes pedindo ajuda, reclamando, querendo cancelar, comprando produtos etc. Hoje, **menos de 5%** dessas conversas são analisadas — o resto fica perdido.

O TalkEx é um sistema que **lê essas conversas automaticamente** e classifica o que o cliente quer (sua "intenção"). Ele faz isso combinando 3 técnicas diferentes de inteligência artificial:

1. **Busca por palavras-chave** (BM25) — como o Google: procura termos exatos
2. **Compreensão de significado** (Embeddings) — entende que "quero desistir" = "cancelar"
3. **Regras escritas por humanos** (DSL) — ex: "se menciona 'cancelar' E é o cliente falando → provável cancelamento"

---

## O Dataset (os dados usados)

| Item | Valor |
|------|-------|
| Fonte original | Dataset `RichardSakaguchiMS/brazilian-customer-service-conversations` (HuggingFace, licença Apache 2.0) |
| Conversas originais | **847** (extraídas do HuggingFace) |
| Conversas sintéticas | **1.275** (geradas via Claude Sonnet em modo batch, com auditoria de qualidade) |
| **Total do corpus** | **2.122 conversas** |
| Classes de intenção | **8** (cancelamento, reclamação, suporte técnico, compra, dúvida produto, dúvida serviço, saudação, elogio) |
| Idioma | Português brasileiro (PT-BR) |
| Setores | Telecom, varejo, financeiro, saúde, educação, energia, seguros, governo |

### De onde vêm os dados?

O dataset base foi publicado no HuggingFace por Sakaguchi (2023) e contém conversas simuladas de atendimento ao cliente em PT-BR. Como o dataset original era pequeno demais para treinar modelos com confiança, ele foi **expandido com geração sintética controlada** (usando o modelo Claude Sonnet em modo batch offline) e submetido a um **protocolo de auditoria em 8 etapas**:

1. Validação de esquema (campos obrigatórios, intervalos válidos)
2. Deduplicação em dois níveis (exata + similaridade semântica ≥ 0.97)
3. Detecção de contaminação few-shot (evita vazamento entre splits)
4. Verificação de integridade dos splits estratificados
5. Auditoria taxonômica por embeddings (coerência intra-classe ≥ 0.60)
6. Análise da categoria "outros" (removida — ficamos com 8 classes)
7. Verificação de qualidade textual (marcadores de turno, mínimo 2 turnos)
8. Análise de distribuição multidimensional (intenção × domínio × sentimento)

A acurácia final dos rótulos após revisão humana foi de **96.7%**.

### Splits (divisão dos dados)

Os dados são divididos de forma estratificada (mesma proporção de cada classe em cada parte):

| Split | Conversas | Para que serve |
|-------|-----------|----------------|
| Treino (70%) | 1.250 | O modelo aprende com esses dados |
| Validação (15%) | 404 | Ajuste fino de parâmetros (o modelo NÃO vê esses dados no treino) |
| Teste (15%) | 468 | Avaliação final (o modelo NUNCA viu esses dados) |

A divisão é feita **no nível da conversa** — todas as janelas de uma mesma conversa ficam no mesmo split, evitando vazamento de informação.

### O que é uma "janela de contexto"?

Uma conversa longa é dividida em pedaços de **5 turnos** (falas). O sistema analisa esses pedaços, não a conversa inteira de uma vez. Isso é como ler um livro parágrafo por parágrafo em vez de tentar entender tudo de uma vez.

- **4.750 janelas** foram criadas a partir das 2.122 conversas
- Cada janela se sobrepõe em 2 turnos com a anterior (stride = 2), para não perder contexto

---

## As 4 Hipóteses Testadas

Pensem em hipóteses como "apostas científicas" — afirmações que testamos com dados reais para ver se são verdadeiras.

---

## Hipótese H1: Busca Híbrida é Melhor que Busca Simples

### O que testamos?

"Se combinarmos busca por palavras (BM25) com busca por significado (embeddings), encontramos conversas relevantes mais rápido?"

### O que é MRR?

**MRR (Mean Reciprocal Rank)** = "Em média, quão rápido achamos o resultado certo?"

- Se o resultado certo aparece em **1o lugar** → pontuação = 1.0
- Se aparece em **2o lugar** → pontuação = 0.5
- Se aparece em **3o lugar** → pontuação = 0.33
- Quanto **mais alto** o MRR, **melhor** o sistema de busca

### Resultados

| Método | MRR | O que faz |
|--------|-----|-----------|
| BM25 (palavras) | 0.8354 | Busca apenas por palavras exatas |
| ANN (significado) | 0.8242 | Busca apenas por significado semântico |
| **Híbrido LINEAR (α=0.5)** | **0.8482** | Combina 50% palavras + 50% significado |
| **Híbrido RRF** | **0.8516** | Combina os rankings de ambos os métodos |

### Interpretação (para leigos)

Imagine que você busca "cliente quer desistir do plano". A busca por palavras encontra mensagens com "desistir" e "plano". A busca por significado também encontra "quero cancelar meu contrato" (que significa a mesma coisa mas usa palavras diferentes).

**Resultado: Combinando as duas, acertamos mais.** A busca híbrida (RRF) encontra o resultado relevante em 1o ou 2o lugar **85% das vezes**, contra 82-83% dos métodos isolados.

### Veredicto: ✅ CONFIRMADA

A melhoria é pequena mas consistente. Em um sistema com milhões de buscas, 2% a mais de acerto = milhares de respostas melhores por dia.

---

## Hipótese H2: Embeddings Fazem Enorme Diferença na Classificação

### O que testamos?

"Se adicionarmos 'compreensão de significado' (embeddings) às features simples (contagem de palavras etc.), a classificação melhora muito?"

### O que é Macro-F1?

**Macro-F1** = média de acerto equilibrada entre todas as 8 classes.

- F1 = 1.0 → perfeito
- F1 = 0.5 → acerta metade
- F1 = 0.125 → equivalente a chutar aleatoriamente (1/8 classes)

"Macro" significa que classes raras (poucas amostras) pesam igual às classes frequentes. Isso é importante porque não queremos um sistema que só acerta as classes fáceis.

### Resultados

| Features usadas | Classificador | Macro-F1 | Acurácia |
|-----------------|--------------|----------|----------|
| Só lexicais (13 features) | Regressão Logística | **0.2011** | 25.6% |
| Só lexicais (13 features) | LightGBM | **0.3344** | 34.6% |
| Lexicais + Embeddings (397 features) | Regressão Logística | **0.6098** | 64.9% |
| **Lexicais + Embeddings (397 features)** | **LightGBM** | **0.7216** | **73.9%** |

### Interpretação (para leigos)

Sem embeddings, o sistema é como alguém que só sabe contar palavras — consegue acertar ~33% das vezes (pouco melhor que chutar). Com embeddings, ele **entende o significado** e acerta **72% das vezes**.

O ganho é de **+38.7 pontos percentuais** (de 0.3344 para 0.7216). Isso é enorme — mais que dobrou a qualidade.

### Para cada classe:

| Intenção | F1 sem embeddings | F1 com embeddings | Melhoria |
|----------|-------------------|-------------------|----------|
| Cancelamento | 0.25 | **0.91** | +264% |
| Suporte técnico | 0.31 | **0.84** | +175% |
| Elogio | 0.40 | **0.87** | +118% |
| Reclamação | 0.42 | **0.80** | +90% |
| Dúvida serviço | 0.38 | **0.75** | +97% |
| Dúvida produto | 0.35 | **0.68** | +94% |
| Saudação | 0.38 | **0.48** | +27% |
| Compra | 0.19 | **0.44** | +130% |

Cancelamento e suporte técnico são as mais beneficiadas — faz sentido porque clientes expressam essas intenções de muitas formas diferentes ("quero sair", "desisto", "cancela", "não quero mais"), e os embeddings capturam todas essas variações.

### Veredicto: ✅ CONFIRMADA (com folga!)

Ganho de +38.7pp, muito acima dos 30pp esperados. Os embeddings são o ingrediente mais importante do sistema.

---

## Hipótese H3: Regras Ajudam (mas só como features, não como overrides)

### O que testamos?

"Se adicionarmos regras escritas por humanos (ex: 'se contém a palavra cancelar → é cancelamento'), o sistema melhora?"

Testamos 3 estratégias:

1. **ML-only** — só machine learning, sem regras
2. **ML+Rules-override** — se uma regra dispara, ela SOBRESCREVE a decisão do ML
3. **ML+Rules-feature** — o disparo de regras vira uma feature EXTRA para o ML decidir

### Resultados

| Estratégia | Macro-F1 | vs ML-only |
|-----------|----------|------------|
| Rules-only (só regras) | **0.1366** | -81% (péssimo!) |
| ML+Rules-override | **0.6796** | **-5.8%** (piorou!) |
| ML-only (baseline) | **0.7216** | — |
| **ML+Rules-feature** | **0.7400** | **+2.5%** (melhorou!) |

### Interpretação (para leigos)

Imagine um médico (ML) e um manual de sintomas (regras):

- **Só o manual** (Rules-only): é como diagnosticar olhando só uma lista de sintomas — funciona para casos óbvios, mas erra tudo que não está no manual. F1 = 0.14 (horrível).

- **Manual manda no médico** (override): quando o manual "acha" que sabe, ele manda no médico. Problema: o manual tem cobertura limitada (só detecta 2 das 8 classes). Quando ele erra, não tem como o médico corrigir. F1 caiu 5.8%.

- **Manual como segunda opinião** (feature): o médico recebe uma anotação "o manual acha que é cancelamento" e usa como UMA informação dentre muitas. Se fizer sentido no contexto, ele concorda. Se não, ele ignora. F1 subiu 2.5%.

### Classes mais beneficiadas pelas regras:

| Classe | ML-only | ML+Rules-feature | Melhoria |
|--------|---------|-------------------|----------|
| Cancelamento | 0.909 | **0.946** | +4% |
| Saudação | 0.478 | **0.522** | +9% |
| Compra | 0.442 | **0.488** | +10% |

### Veredicto: ✅ CONFIRMADA

Regras como features suaves ajudam (+2.5%). Regras como overrides rígidos prejudicam (-5.8%). A lição: **deixe o ML decidir, mas dê informações extras para ele**.

---

## Hipótese H4: Cascata para Economizar Processamento

### O que testamos?

"Podemos usar um modelo LEVE primeiro e só acionar o modelo PESADO quando o leve não tem certeza?"

A ideia é como triagem em hospital: um enfermeiro resolve casos simples; só chama o médico para casos complexos.

- **Estágio 1** (leve): Regressão Logística — rápida mas menos precisa
- **Estágio 2** (pesado): LightGBM — mais lento mas mais preciso

Se a confiança do Estágio 1 ≥ threshold → aceita; senão → escala para Estágio 2.

### Resultados

| Threshold | Macro-F1 | % que vai pro Estágio 2 | Custo (ms) |
|-----------|----------|--------------------------|------------|
| Uniforme (sempre pesado) | **0.7216** | 100% | 159 ms |
| 0.50 | 0.7050 | 48.6% | 252 ms |
| 0.70 | 0.7180 | 76.3% | 296 ms |
| 0.80 | **0.7241** | 88.3% | 315 ms |
| 0.90 | 0.7200 | 95.3% | 327 ms |

### Interpretação (para leigos)

**A cascata NÃO economizou custo neste experimento.** Por quê?

1. O Estágio 1 (LogReg) e o Estágio 2 (LightGBM) usam as **mesmas features** (397 dimensões). O custo de preparar as features (gerar embeddings) é o mesmo para ambos.

2. O "custo" real é dominado pela geração de embeddings, não pela classificação. Trocar o classificador economiza microsegundos, não segundos.

3. Com threshold alto (0.80), a cascata atinge F1 levemente MELHOR (0.7241 vs 0.7216) porque combina duas opiniões. Mas o custo não reduz — na verdade aumenta porque roda os dois modelos.

### Quando a cascata funcionaria de verdade?

Se o Estágio 1 usasse apenas features BARATAS (lexicais, 13 dims) e só escalasse para embeddings completos quando não tivesse certeza. Isso não foi testado neste experimento.

### Veredicto: ⚠️ NÃO CONFIRMADA (mas aprendizado valioso)

A cascata não reduz custo na configuração testada. O paper identifica que a condição necessária — custos de features assimétricos entre estágios — não foi satisfeita.

---

## Estudo de Ablação: "O Que Importa Mais?"

### O que é ablação?

É como descobrir qual ingrediente faz a receita funcionar: tiramos um de cada vez e vemos o quanto o prato piora.

### Resultados

| Configuração | Features | Macro-F1 | Queda vs Full |
|-------------|----------|----------|---------------|
| **Full pipeline** | **397** | **0.7400** | — |
| -Embeddings | 13 | 0.4102 | **-44.6%** |
| -Lexicais | 390 | 0.7112 | -3.9% |
| -Regras | 395 | 0.7216 | -2.5% |
| -Estruturais | 393 | 0.7267 | -1.8% |
| Embedding-only | 384 | 0.7084 | -4.3% |
| Lexical-only | 11 | 0.3344 | **-54.8%** |

### Interpretação (para leigos)

Pense nos ingredientes de uma pizza:

| Ingrediente | Analogia | Impacto |
|-------------|----------|---------|
| **Embeddings (384 features)** | A massa — sem ela não é pizza | **Essencial** (remover perde 44.6%) |
| Lexicais (7 features) | O molho — complementa | Moderado (remover perde 3.9%) |
| Regras (2 features) | A borda recheada — um extra | Pequeno mas positivo (remover perde 2.5%) |
| Estruturais (4 features) | O sal — sutil | Mínimo (remover perde 1.8%) |

**Ranking de importância:**

```
Embeddings >>>>>>> Lexicais > Regras > Estruturais
(imprescindível)   (ajuda)    (ajuda)   (marginal)
```

### O que isso significa na prática?

Se você tiver que escolher APENAS UM componente, escolha embeddings. Com apenas embeddings (384 features), o sistema já atinge F1 = 0.71. Adicionar os outros 13 features eleva para 0.74 — uma melhoria de 4% que vale o esforço mínimo.

---

## Resumo Executivo para a Banca

### O TalkEx funciona?

**Sim.** Macro-F1 = **0.74** em 8 classes de intenção, usando apenas embeddings congelados (sem fine-tuning) e infraestrutura gratuita (Google Colab).

### Principais descobertas:

| # | Descoberta | Significado Prático |
|---|-----------|---------------------|
| 1 | Embeddings são o componente mais importante (+44.6%) | Investir em bons modelos de linguagem vale mais que qualquer outra otimização |
| 2 | Busca híbrida supera busca simples (MRR 0.85 vs 0.82) | Combinar paradigmas sempre ajuda na recuperação de informação |
| 3 | Regras ajudam como features (+2.5%), prejudicam como overrides (-5.8%) | Nunca dê poder de veto a regras simples sobre modelos treinados |
| 4 | Cascata não economiza quando features são compartilhadas | Otimização de custo requer features assimétricos entre estágios |

### Limitações honestas:

- Dataset de 2.122 conversas — resultados podem diferir em escala de milhões
- Apenas 8 classes — cenários reais podem ter 50+ intenções
- Sem fine-tuning — com fine-tuning do encoder provavelmente subiria para F1 > 0.85
- Classes "compra" e "saudação" ainda têm F1 < 0.55 — confundem-se com outras intenções

### Para onde o TalkEx pode ir?

1. Fine-tuning do modelo de embeddings no domínio específico
2. Mais regras com melhor cobertura
3. Cascata com features assimétricos (lexical-first, embedding-second)
4. Escalar para datasets de milhões de conversas
5. Adicionar detecção de múltiplas intenções por conversa

---

## Glossário de Termos Técnicos

| Termo | Explicação Simples |
|-------|-------------------|
| **Embedding** | Representação numérica (vetor de 384 números) que captura o "significado" de um texto |
| **BM25** | Algoritmo de busca por palavras — como o Google faz busca textual |
| **LightGBM** | Algoritmo de classificação que aprende a decidir com base em exemplos |
| **Macro-F1** | Nota de 0 a 1 que mede quão bem o sistema acerta, equilibrando todas as classes |
| **MRR** | Nota de 0 a 1 que mede quão rápido a busca encontra o resultado certo |
| **Seed** | Número que garante reprodutibilidade — mesma seed = mesmos resultados |
| **Feature** | Uma informação numérica que o modelo usa para decidir (ex: "número de palavras") |
| **Threshold** | Limiar de confiança — "só aceite se tiver X% de certeza" |
| **DSL** | Linguagem para escrever regras de forma estruturada (como uma receita) |
| **Janela de contexto** | Pedaço de 5 turnos consecutivos de uma conversa |
| **Stride** | Quantos turnos a janela avança — stride 2 = sobrepõe 3 turnos com a anterior |
| **Ablação** | Técnica de remover um componente por vez para medir sua importância |
| **Wilcoxon** | Teste estatístico que verifica se uma diferença é real ou por acaso |
| **Bootstrap CI** | Técnica para estimar a margem de erro de um resultado |
| **Rank-biserial (r_rb)** | Medida de "quão forte" é o efeito — 0.5+ = forte, 0.3-0.5 = médio |
