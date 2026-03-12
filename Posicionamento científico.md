# Posicionamento científico do TalkEx frente a modelos especializados

## 1. Motivação

Grande parte dos sistemas contemporâneos de classificação de intenções baseia-se em **fine-tuning de modelos de linguagem para domínios específicos**. Essa estratégia geralmente permite alcançar altos níveis de desempenho quando o modelo é treinado e avaliado dentro do mesmo domínio de aplicação. No entanto, essa abordagem apresenta limitações relevantes em cenários reais de implantação.

Primeiramente, modelos especializados frequentemente dependem de **datasets rotulados específicos para cada domínio**, o que implica custos significativos de coleta, anotação e manutenção. Em ambientes organizacionais com múltiplos produtos ou serviços, isso pode levar à necessidade de treinar e manter diversos modelos independentes.

Em segundo lugar, modelos altamente especializados tendem a apresentar **degradação significativa de desempenho quando aplicados fora do domínio de treinamento**, fenômeno amplamente conhecido como *domain shift* ou *distribution shift*. Nesse contexto, a portabilidade do sistema entre diferentes cenários torna-se limitada.

Diante dessas limitações, o TalkEx propõe uma abordagem alternativa baseada no princípio **frozen-first**, no qual o encoder semântico permanece congelado e o sistema opera majoritariamente por meio de:

* representações semânticas generalistas,
* classificadores leves,
* mecanismos de calibração,
* regras auditáveis,
* e estratégias de inferência em cascata.

Essa arquitetura busca equilibrar desempenho, interpretabilidade e custo operacional.

---

## 2. Hipótese de pesquisa

A proposta do TalkEx não parte da premissa de que um sistema universal sem fine-tuning necessariamente superará modelos especializados em seus respectivos domínios.

Em vez disso, a hipótese central da pesquisa é formulada da seguinte maneira:

> **Hipótese principal:** sistemas baseados em encoders generalistas congelados podem manter desempenho competitivo em múltiplos cenários distintos, apresentando menor degradação sob mudança de domínio e menor custo de adaptação do que sistemas dependentes de fine-tuning específico.

Essa hipótese desloca o foco da avaliação de **performance absoluta em um único domínio** para **robustez e transferibilidade entre domínios**.

---

## 3. Limitações dos modelos especializados

Modelos treinados com fine-tuning apresentam vantagens claras em cenários de domínio fechado. Entretanto, sua utilização em ambientes operacionais amplos apresenta desafios relevantes.

Entre os principais problemas estão:

### Dependência de dados rotulados

O processo de fine-tuning exige datasets rotulados específicos para cada cenário. Em aplicações reais, esse processo pode envolver:

* coleta de dados conversacionais,
* anotação manual por especialistas,
* revisão iterativa de taxonomias de intenção.

Esse ciclo pode ser oneroso e difícil de escalar.

### Baixa portabilidade entre cenários

Modelos ajustados para um domínio frequentemente capturam **padrões semânticos e pragmáticos específicos daquele contexto**. Como consequência, quando aplicados a cenários distintos, esses modelos podem apresentar quedas substanciais de desempenho.

Esse fenômeno é especialmente problemático em aplicações conversacionais, nas quais:

* vocabulário,
* estrutura de diálogo,
* intenções dos usuários

podem variar significativamente entre setores.

### Custos operacionais de manutenção

Cada novo cenário pode demandar:

* novos datasets,
* novo treinamento,
* validação adicional,
* monitoramento independente.

Essa abordagem pode levar à manutenção simultânea de múltiplos modelos especializados, aumentando a complexidade operacional.

---

## 4. Abordagem universal baseada em encoder congelado

A arquitetura proposta no TalkEx adota uma estratégia distinta. Em vez de adaptar continuamente o modelo base por meio de fine-tuning, o sistema utiliza **encoders semânticos generalistas congelados**, treinados previamente em grandes corpora multilíngues.

A adaptação ao problema específico ocorre por meio de componentes mais leves e interpretáveis, tais como:

* classificadores supervisionados de baixa complexidade;
* agregação de evidências ao longo da conversa;
* regras heurísticas auditáveis;
* mecanismos de calibração de probabilidade;
* inferência em cascata.

Essa abordagem oferece diversas vantagens:

* **redução da dependência de dados rotulados específicos por domínio**;
* **maior reutilização do mesmo modelo em diferentes cenários**;
* **menor custo computacional para adaptação**;
* **maior transparência no processo decisório**.

---

## 5. Critério de avaliação

Dado esse posicionamento, a avaliação do TalkEx não deve ser baseada exclusivamente em métricas de desempenho em um único domínio.

Em vez disso, o sistema deve ser avaliado considerando múltiplas dimensões, incluindo:

* desempenho médio entre diferentes cenários;
* degradação de desempenho sob mudança de domínio;
* custo computacional de adaptação;
* necessidade de dados rotulados adicionais;
* interpretabilidade das decisões do sistema.

Esse tipo de avaliação permite comparar de forma mais justa sistemas universais e sistemas especializados.

Em particular, espera-se que:

* modelos especializados apresentem **maior desempenho in-domain**;
* o TalkEx apresente **maior estabilidade e generalização entre cenários**.

---

## 6. Contribuição científica

A contribuição do TalkEx não reside apenas na obtenção de melhorias incrementais de acurácia em um benchmark específico.

Sua principal contribuição está na investigação de um paradigma alternativo para sistemas de inteligência conversacional, caracterizado por:

* forte reutilização de representações semânticas generalistas;
* redução da dependência de fine-tuning específico por domínio;
* integração de componentes simbólicos e estatísticos;
* e maior transparência no processo de inferência.

Ao explorar esse paradigma, o trabalho busca contribuir para o desenvolvimento de sistemas conversacionais mais **robustos, portáveis e auditáveis**, capazes de operar em múltiplos cenários com menor custo de adaptação.

---

