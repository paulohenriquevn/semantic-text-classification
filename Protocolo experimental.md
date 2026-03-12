# Protocolo experimental para avaliação de generalização entre cenários

## 1. Objetivo da avaliação

A avaliação experimental do TalkEx tem como objetivo investigar a capacidade de sistemas de classificação de intenções baseados em **encoders semânticos congelados** de manter desempenho competitivo em múltiplos cenários conversacionais distintos, sem necessidade de fine-tuning específico por domínio.

Diferentemente de avaliações tradicionais de classificação de texto, que frequentemente consideram apenas desempenho dentro de um único domínio, este trabalho adota um protocolo experimental voltado para **medir generalização entre cenários**.

Esse protocolo busca responder às seguintes perguntas de pesquisa:

* Um sistema baseado em encoder congelado consegue manter desempenho competitivo em diferentes cenários conversacionais?
* Qual é o impacto do fine-tuning específico por domínio na capacidade de generalização do sistema?
* Existe diferença significativa na degradação de desempenho entre modelos universalistas e modelos especializados?

---

## 2. Estrutura do benchmark multi-cenário

Para avaliar generalização entre domínios, foi construído um benchmark composto por múltiplos cenários conversacionais distintos.

Cada cenário representa um contexto de interação específico entre usuário e sistema, caracterizado por vocabulário, objetivos comunicativos e tipos de intenção potencialmente distintos.

Exemplos de cenários considerados incluem:

* atendimento ao cliente em telecomunicações;
* suporte técnico para serviços digitais;
* comércio eletrônico;
* serviços educacionais;
* serviços financeiros.

Cada cenário contém um conjunto independente de conversas rotuladas segundo uma taxonomia de intenções previamente definida.

Sempre que possível, essa taxonomia é mantida consistente entre cenários para permitir comparações diretas entre modelos.

---

## 3. Estratégia de avaliação cross-domain

Para avaliar a capacidade de generalização do sistema, foi adotada uma estratégia de **leave-one-domain-out evaluation**.

Nesse protocolo, cada cenário é alternadamente utilizado como conjunto de teste, enquanto os demais cenários compõem o conjunto de treinamento.

Formalmente, considerando um conjunto de cenários:

[
D = {D_1, D_2, ..., D_n}
]

em cada rodada de avaliação é realizado o seguinte procedimento:

1. selecionar um cenário (D_i) como conjunto de teste;
2. utilizar os demais cenários (D \setminus D_i) para treinamento;
3. treinar o modelo utilizando apenas os dados dos cenários de treinamento;
4. avaliar o desempenho no cenário (D_i), completamente não visto durante o treinamento.

Esse processo é repetido para todos os cenários do benchmark.

Esse protocolo permite estimar de forma mais realista a capacidade do sistema de operar em contextos não observados durante o treinamento.

---

## 4. Regimes experimentais comparados

A avaliação considera três regimes experimentais principais.

### 4.1 Modelo universal sem fine-tuning

Nesse regime, o encoder semântico permanece completamente congelado.

As representações textuais são geradas por um modelo pré-treinado generalista e utilizadas como entrada para um classificador supervisionado leve.

Esse regime corresponde à arquitetura proposta pelo TalkEx.

### 4.2 Modelo especializado com fine-tuning

Nesse regime, o encoder é ajustado por meio de fine-tuning utilizando dados do domínio específico.

Esse modelo representa o estado da prática em muitos sistemas de classificação de intenções.

A avaliação desse regime permite medir o ganho potencial de especialização em relação ao modelo universal.

### 4.3 Modelo universal com adaptação mínima

Nesse regime intermediário, o encoder permanece congelado, mas o classificador final é treinado utilizando dados do domínio de interesse.

Esse cenário representa um nível moderado de adaptação, sem alterar as representações semânticas do modelo base.

---

## 5. Métricas de avaliação

A avaliação utiliza múltiplas métricas para capturar diferentes aspectos do comportamento do sistema.

### Macro-F1

A métrica principal utilizada é Macro-F1, que considera o desempenho médio entre todas as classes de intenção, tratando cada classe com igual importância.

Essa métrica é particularmente adequada em cenários com distribuição de classes potencialmente desbalanceada.

### Generalization gap

Para medir a degradação de desempenho entre cenários, é calculada a diferença entre:

* desempenho médio em cenários de treinamento (in-domain);
* desempenho no cenário não observado (out-of-domain).

Essa diferença é denominada **generalization gap**.

### Estabilidade entre cenários

Também é analisada a variância do desempenho entre diferentes cenários de teste, permitindo avaliar a estabilidade do sistema em contextos distintos.

---

## 6. Testes estatísticos

Para avaliar a significância estatística das diferenças entre modelos, são utilizados testes não paramétricos apropriados para comparações emparelhadas.

Entre os testes considerados estão:

* **teste de Wilcoxon signed-rank**, aplicado às métricas obtidas em diferentes cenários;
* **intervalos de confiança obtidos por bootstrap**, calculados sobre o conjunto de conversas avaliadas.

Esses procedimentos permitem verificar se diferenças observadas entre modelos são estatisticamente significativas.

---

## 7. Interpretação dos resultados

A interpretação dos resultados considera não apenas o desempenho absoluto dos modelos, mas também sua capacidade de generalização entre cenários.

Em particular, são analisados os seguintes aspectos:

* desempenho médio entre cenários;
* degradação de desempenho sob mudança de domínio;
* variabilidade entre cenários;
* custo computacional de adaptação do modelo.

Espera-se que modelos especializados apresentem melhor desempenho em cenários nos quais foram treinados, enquanto sistemas baseados em encoders congelados apresentem **maior robustez sob mudança de domínio**.

---

## 8. Critério de sucesso da abordagem

A abordagem proposta será considerada bem-sucedida se demonstrar:

1. desempenho competitivo em relação a modelos especializados;
2. menor degradação de desempenho em cenários não observados durante o treinamento;
3. capacidade de operar em múltiplos cenários sem necessidade de fine-tuning do encoder.

Esses critérios refletem o objetivo principal do TalkEx: desenvolver um sistema de inteligência conversacional que combine desempenho adequado com **portabilidade, interpretabilidade e baixo custo de adaptação**.

---

Se quiser, eu também posso escrever a **seção que normalmente vem logo depois desta na dissertação**, que é extremamente importante para convencer a banca:

**“Ameaças à validade experimental”**

Essa é a seção onde você discute:

* dataset sintético
* risco de overfitting
* limitações do benchmark
* validade externa

E uma banca de mestrado/doutorado **quase sempre cobra isso**.
