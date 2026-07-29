# TalkEx Dataset Audit — Human Review Sample

**Total records:** 90
**Date:** 2026-03-12
**Reviewer:** _________________

## Instructions

For each record, read the conversation text and validate the automatic decision.
Fill in `human_decision` with one of: `confirm`, `reclassify`, `restore`, `unsure`.
Add notes in `human_notes` if needed.

**Acceptance criterion:** ≥ 90% decisions confirmed → approve Phase 1.

---

## Section 1: Near-Duplicates Removed (5 records)

### Record #1: `conv_00003`

- **Original label:** `duvida_produto`
- **Action taken:** `removed_duplicate`
- **Nearest intent (embedding):** `duvida_produto` (similarity: 0.9822)
- **Reason:** Hard near-duplicate of conv_00119 (cosine similarity=0.9822)
- **Kept counterpart:** `conv_00119`

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olá, vc pode me dizer se o smartphone samsung galaxy s22 ta disponivel?
[agent] Beleza! Sim, o Samsung Galaxy S22 está disponível em nosso site. Qual é a cor e a capacidade que você está procurando?
[customer] td bem, eu quero saber o preco tb, por favor
[agent] Claro! O preço do Samsung Galaxy S22 varia de acordo com a capacidade e a cor. Por exemplo, o modelo de 128GB custa R$ 3.500,00 e o de 256GB custa R$ 4.000,00. Qual é a sua opção?
[customer] show, eu acho que vou levar o de 128GB, mas vc pode me dizer se tem alguma promocao?
[agent] Sim, estamos com uma promoção especial nes...
```
</details>

<details>
<summary>Kept counterpart text (click to expand)</summary>

```
[customer] oi, vc pode me dizer se o celular samsung galaxy s22 ta disponivel?
[agent] Beleza! Sim, o Samsung Galaxy S22 está disponível em nosso estoque. Qual é a cor que você está interessado?
[customer] td bem, eu queria saber o preco tb, vc pode me passar?
[agent] O preço do Samsung Galaxy S22 varia de acordo com a capacidade de armazenamento e a cor. O modelo de 128GB custa R$ 3.500,00 e o de 256GB custa R$ 4.000,00. Qual é a capacidade que você está procurando?
[customer] show, eu acho que vou optar pelo de 128gb, vc pode me dizer se tem algum desconto?
[agent] Ta bom! Atualmente, estamo...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #2: `conv_00036`

- **Original label:** `reclamacao`
- **Action taken:** `removed_duplicate`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.9786)
- **Reason:** Hard near-duplicate of conv_00246 (cosine similarity=0.9786)
- **Kept counterpart:** `conv_00246`

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? vc pode me ajudar com um problema que eu tive no meu cartao?
[agent] Beleza! Claro que posso ajudar. Qual foi o problema que você teve com o cartão?
[customer] eu tentei fazer uma compra online e o cartao foi recusado, mas eu tenho td o dinheiro necessário na conta
[agent] Entendi, show! Vou verificar o que aconteceu. Você pode me informar o valor da compra e a data que tentou fazer a transação?
[customer] a compra foi de R$ 250,00 e foi ontem a noite, por volta das 22h. eu ja verifiquei o extrato e o dinheiro ta la
[agent] Ok, td bem! Eu já encontrei o problema. Parec...
```
</details>

<details>
<summary>Kept counterpart text (click to expand)</summary>

```
[customer] Oi, tudo bem? vc pode me ajudar com um problema que eu tive no meu cartao?
[agent] Beleza! Claro que posso te ajudar. Qual foi o problema que você teve com o cartão?
[customer] eu fiz uma compra online e o cartao foi recusado, td foi muito estranho pq eu tinha saldo suficiente na conta
[agent] Entendi, show, vamos verificar o que aconteceu. Você pode me informar o valor da compra e a data em que ela foi feita?
[customer] a compra foi de R$ 200,00 e foi feita ontem, vc acha que pode ser algum problema com o cartao ou com a conta?
[agent] Vou verificar tudo para você. Pode ser que ten...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #3: `conv_00116`

- **Original label:** `duvida_produto`
- **Action taken:** `removed_duplicate`
- **Nearest intent (embedding):** `duvida_produto` (similarity: 0.9783)
- **Reason:** Hard near-duplicate of conv_00117 (cosine similarity=0.9783)
- **Kept counterpart:** `conv_00117`

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc tem o smartphone samsung galaxy s22 ultra disponivel?
[agent] Beleza! Sim, temos o Samsung Galaxy S22 Ultra disponível. Qual é a cor e a capacidade que você está procurando?
[customer] td bem, eu quero saber o preco tb, ta bom?
[agent] Claro! O preço do Samsung Galaxy S22 Ultra varia de acordo com a capacidade e a cor. Por exemplo, o modelo de 128GB custa R$ 4.500,00 e o de 256GB custa R$ 5.000,00. Quer que eu te envie um link para você conferir os preços e as opções de cor?
[customer] show, pode enviar o link, por favor?
[agent] Aqui está o link: https://lojaonline.com.br/sa...
```
</details>

<details>
<summary>Kept counterpart text (click to expand)</summary>

```
[customer] oi, vc tem o celular samsung galaxy s22 ultra disponivel?
[agent] Beleza! Sim, temos o Samsung Galaxy S22 Ultra disponível. Qual é a cor que você está procurando?
[customer] td bem, eu quero saber o preco tb, ta bom?
[agent] Show! O preço do Samsung Galaxy S22 Ultra varia de acordo com a cor e a capacidade de armazenamento. Por exemplo, o modelo de 128GB custa R$ 4.500,00 e o de 256GB custa R$ 5.000,00. Qual é a capacidade que você está procurando?
[customer] eu quero o de 256gb, mas o preco ta um pouco alto, vc tem alguma promoção?
[agent] Entendi, sim, estamos com uma promoção esp...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #4: `conv_00930`

- **Original label:** `cancelamento`
- **Action taken:** `removed_duplicate`
- **Nearest intent (embedding):** `cancelamento` (similarity: 0.9715)
- **Reason:** Hard near-duplicate of conv_00928 (cosine similarity=0.9715)
- **Kept counterpart:** `conv_00928`

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Quero cancelar o servico de vc, ta bom?
[agent] Beleza! Claro, posso te ajudar com isso. Qual o motivo do cancelamento? E qual o seu ID de cliente ou e-mail associado à conta?
[customer] Td bem, obrigado! O motivo é que eu não estou utilizando o servico como gostaria, e o meu e-mail é fulano@example.com
[agent] Entendi, show! Vou verificar a sua conta. Sim, encontrei. Para cancelar, preciso que você confirme que deseja cancelar o serviço e entender que não terá mais acesso às funcionalidades. Posso encaminhar um e-mail para você confirmar?
[customer] Sim, pode ser, vc ...
```
</details>

<details>
<summary>Kept counterpart text (click to expand)</summary>

```
[customer] Oi, tudo bem? Quero cancelar o servico de vc, ta bom?
[agent] Beleza! Claro, posso ajudar com isso. Qual o motivo do cancelamento? E pode me informar o seu ID de cliente ou o email associado à conta para que eu possa localizar seus dados?
[customer] td bem, o servico esta show, mas eu não estou usando mais. Meu email é fulano@gmail.com, vc consegue achar?
[agent] Encontrei sua conta, obrigado! Vou iniciar o processo de cancelamento. Você receberá um email de confirmação assim que tudo for finalizado. Além disso, gostaríamos de saber se você tem alguma sugestão para melhorarmos nosso...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #5: `conv_00177`

- **Original label:** `cancelamento`
- **Action taken:** `removed_duplicate`
- **Nearest intent (embedding):** `cancelamento` (similarity: 0.9710)
- **Reason:** Hard near-duplicate of conv_00176 (cosine similarity=0.971)
- **Kept counterpart:** `conv_00176`

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc pode me ajudar a cancelar um pedido?
[agent] Beleza! Claro, posso te ajudar com isso. Qual é o número do seu pedido?
[customer] td bem, o numero do pedido é #1234. eu queria cancelar td
[agent] Show, vou verificar isso para você. Você fez o pedido há pouco tempo, então ainda dá para cancelar. Você gostaria de cancelar o pedido inteiro ou apenas um item?
[customer] eu quero cancelar td, mas vc pode me dizer se vai ter algum custo adicional?
[agent] Ta bom, vou te explicar. Se você cancelar o pedido agora, não vai ter custo adicional. Mas se o pedido já tiver sido enviado, vamo...
```
</details>

<details>
<summary>Kept counterpart text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar a cancelar um pedido?
[agent] Beleza, tudo bem! Claro, posso te ajudar com isso. Qual é o número do seu pedido?
[customer] td bem, o numero do pedido é #1234. Eu quero cancelar td porque mudei de ideia
[agent] Show, entendi! Vou verificar o status do seu pedido. Você fez o pedido há pouco tempo, então ainda dá para cancelar. Você gostaria de cancelar o pedido inteiro ou apenas um item?
[customer] vc acha que da para cancelar so um item? Eu quero cancelar so um relógio
[agent] Sim, dá para cancelar apenas um item. Vou verificar se o relógio já foi enviado. Se nã...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

## Section 2: Reclassified Records (1 record)

### Record #6: `conv_00204`

- **Original label:** `outros`
- **Action taken:** `reclassified`
- **Recommended label:** `reclamacao`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.8582)
- **Reason:** Mislabeled as 'outros', nearest intent is 'reclamacao' (cosine similarity=0.8582)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc pode me ajudar? eu fiz uma compra na sua loja online e td deu errado, o produto veio danificado e a entrega demorou muito
[agent] Olá, sinto muito pelo problema. Posso ajudar a resolver isso para você. Você poderia me informar o número do pedido para que eu possa verificar o que aconteceu?
[customer] ta bom, o numero do pedido é #1234. eu ja tentei entrar em contato com vc por email, mas ninguem respondeu, show de bola
[agent] Peço desculpas pelo descaso. Vou verificar o que aconteceu com o seu pedido. Você gostaria de receber um novo produto ou fazer um reembolso?
[customer]...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

## Section 3: Abstention Candidates (30 records)

### Record #7: `conv_00206`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.7694)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=suporte_tecnico, similarity=0.7694). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc pode me ajudar com um problema? comprei um celular na loja vc e ele veio com defeito, td errado
[agent] Olá, lamentamos muito ouvir isso. Posso ajudar a resolver o problema. Você pode me fornecer o número do pedido e mais detalhes sobre o defeito do celular?
[customer] ta bom, o numero do pedido é #1234. o celular tem problema na tela, show, nem consigo usar
[agent] Entendi, vamos resolver isso. Você já tentou entrar em contato com o nosso suporte técnico? Se não, posso criar um ticket para você e nosso time vai avaliar o problema e encontrar uma solução
[customer] nao, ainda...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #8: `conv_synth_02472`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.7737)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=suporte_tecnico, similarity=0.7737). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi gente, vc podem me ajudar? eu to tentando entrar no sistema da empresa aqui mas ta dando um erro esquisito e meu chefe ta pegando no meu pe pq nao consigo trabalhar
[agent] Oi! Claro, posso te ajudar sim. Qual erro está aparecendo quando você tenta acessar o sistema? E qual sistema você está tentando usar?
[customer] olha, é o FlexiCorp que a gente usa aqui na empresa. ta dando uma mensagem tipo 'erro 404' ou coisa assim... cara, isso ta me deixando muito estressado pq ja perdi a manha toda
[agent] Entendo sua frustração! Erro 404 geralmente indica que a página não foi encontrada...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #9: `conv_00422`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.7629)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=suporte_tecnico, similarity=0.7629). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olá, vc pode me ajudar? Estou tendo um problema com a minha internet e não consigo resolver td sozinho
[agent] Claro, beleza! Posso tentar ajudar. Qual é o problema que você está tendo com a sua internet? É uma questão de velocidade, conexão ou algo mais?
[customer] Na verdade, a internet ta caindo o tempo todo. Já resetiei o roteador e td mais, mas não adiantou
[agent] Entendi, show! Vou precisar verificar alguns dados da sua conta. Você pode me informar o seu número de telefone ou o código do seu serviço de internet?
[customer] Ok, é 98876-5432. E o código é IL123456
[agent] Certo...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #10: `conv_00529`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7593)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7593). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com um problema que eu estou tendo com o meu plano de saude?
[agent] Claro, beleza! Qual é o problema que você está enfrentando? Pode me contar um pouco mais sobre o que está acontecendo?
[customer] Eu tentei marcar uma consulta ontem e td deu errado. O sistema disse que não tinha vagas disponiveis, mas eu sei que tem um medico que atende naquela hora. Show, eu estou muito frustrado
[agent] Sinto muito pelo incômodo. Vou verificar o que aconteceu. Você pode me dar o seu número de cadastro e a data que você tentou marcar a consulta?
[customer] Ta bom, meu numero...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #11: `conv_00420`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.8109)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.8109). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olha, vc pode me ajudar com um problema que ta acontecendo com a minha internet? Ta muito lento e td
[agent] Beleza, posso sim! Desculpe pelo incômodo. Você poderia me informar o seu número de telefone ou código do cliente para que eu possa verificar o que está acontecendo?
[customer] Show, o meu numero e 123456789, pode checar o que ta rolando?
[agent] Tudo bem, já encontrei o seu cadastro. Parece que há uma manutenção programada na sua área hoje, o que pode estar causando a lentidão. Mas vou verificar mais a fundo para ter certeza. Você está tendo algum outro problema com a linha ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #12: `conv_00312`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7842)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7842). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com um problema no meu cartao? td mundo me disse que eu estava com o cartao bloqueado e eu nao sei pq
[agent] Claro, posso ajudar sim! Lamento ouvir que você está tendo problemas com o cartão. Você poderia me contar mais sobre o que aconteceu? Quando foi que você percebeu que o cartão estava bloqueado?
[customer] Foi ontem, quando eu tentei fazer uma compra online. O site disse que o cartao nao era valido, mas eu sei que tem dinheiro na conta
[agent] Entendi, show, vamos verificar isso. Você pode me confirmar o número do cartão e a data de validade para eu pode...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #13: `conv_synth_02487`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7770)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.777). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] EU QUERO FALAR COM ALGUÉM AGORA! Vocês estão de brincadeira comigo? Liguei 5 vezes hoje e NINGUÉM atende direito!
[agent] Oi! Entendo sua frustração e peço desculpas pela demora no atendimento. Estou aqui agora para te ajudar. Pode me contar qual é a sua necessidade?
[customer] Olha só, eu só queria saber os horários de funcionamento da loja da Barra da Tijuca porque preciso ir lá HOJE resolver uma coisa do meu plano. É SIMPLES!
[agent] Perfeito! A loja da Barra da Tijuca funciona de segunda a sexta das 9h às 18h e aos sábados das 9h às 14h. Fica no Shopping Downtown, piso L2. Posso...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #14: `conv_00415`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `compra` (similarity: 0.8273)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=compra, similarity=0.8273). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com uma duvida sobre a minha conta?
[agent] Claro, beleza! Estou aqui para ajudar. Qual é a sua dúvida sobre a conta?
[customer] Eu quero saber se vc td sobre os planos de internet que vc oferece, tipo, os preços e td
[agent] Sim, show! Nós oferecemos vários planos de internet com preços competitivos. Você pode acessar nosso site para ver os detalhes de cada plano ou eu posso te passar as informações por aqui
[customer] Beleza, pode me passar as infos por aqui, ta bom?
[agent] Certo, nossos planos de internet começam em R$ 50,00 por mês, com velocidades de até ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #15: `conv_00309`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7989)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7989). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com uma duvida sobre o meu cartao?
[agent] Beleza, claro! Estou aqui para ajudar. Qual é a sua dúvida sobre o cartão?
[customer] Eu estava tentando fazer um pagamento online e deu erro. Eu ja verifiquei td e ta tudo certo, mas nao sei o que ta acontecendo
[agent] Entendi, vamos tentar resolver isso. Você pode me informar o código do erro que apareceu na tela? Isso vai me ajudar a entender melhor o problema
[customer] Naum lembro exatamente, mas acho que era um erro de autorizacao. Show, eu posso tentar novamente e vejo se consigo lembrar
[agent] Tudo bem, se vo...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #16: `conv_00313`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7961)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7961). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar? Estou tendo um problema com o meu cartao, td mundo me cobrando taxas e eu nao entendi pq
[agent] Beleza, posso tentar ajudar sim! Qual o problema exatamente com o cartao? Você recebeu alguma notificacao sobre essas taxas?
[customer] Naum, eu so vi qdo fui verificar o extrato, ta com varias cobranças de taxa de manutenção e eu nem sabia q isso existia
[agent] Entendi, show, vamos verificar isso. Você pode me informar o numero do seu cartao ou o CPF para eu consultar o seu caso?
[customer] Ta bom, o meu cpf e 12345678901, mas eu espero q vc consiga resolver isso...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #17: `conv_synth_02458`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7502)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7502). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] QUE ABSURDO! Liguei pro numero de vcs e caiu numa pizzaria! Que palhaçada é essa??
[agent] Oi! Peço desculpas pela confusão. Você deve ter ligado para um número antigo ou incorreto. Nosso atendimento é feito principalmente pelo chat e WhatsApp. Em que posso ajudá-lo hoje?
[customer] Peguei o numero no google e tava escrito que era de vcs! INADMISSÍVEL uma empresa dessas não ter nem telefone direito!
[agent] Entendo sua irritação e realmente peço desculpas por isso. Às vezes informações desatualizadas ficam no Google. Vou encaminhar para nossa equipe corrigir essas informações. Posso...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #18: `conv_00746`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7830)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.783). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Ei, vc pode me ajudar? Eu fiz um pedido pelo app de vc e td deu errado, show
[agent] Olá! Claro, posso ajudar sim. Qual foi o problema com o seu pedido? Pode me contar mais sobre o que aconteceu?
[customer] Bom, eu pedi uma pizza e um lanche, mas quando chegou em casa, a pizza tava errada e o lanche td errado, td bagunçado
[agent] Sinto muito pelo ocorrido. Vou verificar o que aconteceu com o seu pedido. Você pode me informar o número do pedido, por favor?
[customer] Ta bom, o numero do pedido é 1234. Espero que vc consiga resolver isso, ta me deixando louco
[agent] Entendi, vamos r...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #19: `conv_00528`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7561)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7561). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, td bem? Eu estou com um problema aqui, vc podem me ajudar?
[agent] Bem-vindo à nossa clinica! Claro, estou aqui para ajudar. Qual é o problema que você está enfrentando?
[customer] Eu fiz um exame aqui na clinica ha 2 dias e ainda não recebi o resultado. Isso ta demorando muito, beleza?
[agent] Peço desculpas pelo inconveniente. Você pode me fornecer o seu número de protocolo ou o nome do médico que solicitou o exame? Vou verificar o que está acontecendo.
[customer] O numero do protocolo é 12345. E o medico que solicitou foi o dr. joao. Show, espero que vc consiga resolver isso ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #20: `conv_00079`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `compra` (similarity: 0.7700)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=compra, similarity=0.77). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc pode me ajudar com uma duvida sobre a minha conta
[agent] beleza, claro, estou aqui para ajudar! qual é a sua duvida sobre a conta
[customer] td bem, eu queria saber se vc tem algum plano que inclui internet e celular juntos, ta bom
[agent] sim, temos alguns planos que incluem internet e celular, posso te passar as opcoes, mas preciso saber um pouco mais sobre o que vc está procurando, tipo, quantos gigas vc precisa por mes
[customer] ah, td bem, eu acho que 10 gigas seria o suficiente, mas eu não sei se é melhor pagar à vista ou parcelado
[agent] entendi, vamos ver... com 10...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #21: `conv_00414`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `compra` (similarity: 0.8203)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=compra, similarity=0.8203). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com uma duvida sobre a minha conta?
[agent] Claro, beleza! Estou aqui para ajudar. Qual é a sua dúvida sobre a conta?
[customer] Td bem, obrigado! Eu estava pensando em mudar o meu plano de internet, pq o atual ta um pouco caro pro que eu uso. Vc acha que eu posso ter um desconto ou algo assim?
[agent] Show, vamos dar uma olhada! Posso verificar se há alguma promoção ou plano que seja mais adequado para o seu uso. Qual é o seu plano atual e qual é o seu uso médio de internet por mês?
[customer] Eu uso mais ou menos 500gb por mes, e o meu plano atual é o de 1tb....
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #22: `conv_00846`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `elogio` (similarity: 0.8215)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=elogio, similarity=0.8215). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Quero fazer um elogio sobre o atendimento que eu recebi na semana passada
[agent] Beleza! Tudo bem sim, obrigado por entrar em contato! Qual foi o motivo da sua visita na semana passada? Comprou ou alugou algum imóvel conosco?
[customer] Sim, eu aluguei um apartamento em SP e td foi muito show! Quero agradecer ao João que me atendeu, ele foi super prestativo e me ajudou muito
[agent] Muito obrigado por suas palavras sobre o nosso atendimento! Vou passar o seu elogio para o João, tenho certeza que ele vai ficar muito feliz em saber disso. Se precisar de mais alguma cois...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #23: `conv_00053`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.8229)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.8229). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olha, vc pode me ajudar com um problema que ta acontecendo com a minha conta? td mundo me ligou hoje falando que recebei uma cobranca de R$ 500,00 e eu nao fiz nenhuma compra nesse valor
[agent] Beleza, posso sim ajudar com isso. Você poderia me informar o número da sua conta ou do seu cartão para que eu possa verificar o que está acontecendo?
[customer] ta bom, o numero da conta e 123456, e o cartao e 9876. show, eu estou muito preocupado com isso, pq eu nao tenho esse dinheiro na conta
[agent] Entendi, vamos verificar isso. Você pode me aguardar um minutinho enquanto eu consulto o...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #24: `conv_00080`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.8335)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=suporte_tecnico, similarity=0.8335). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olá, vc pode me ajudar com um problema? Eu estou tendo dificuldade em acessar a internet e td que eu faço é dar erro
[agent] Claro, beleza! Eu posso te ajudar com isso. Você pode me informar qual é o erro que está aparecendo e se você já tentou reiniciar o seu roteador?
[customer] Sim, eu já reiniciei o roteador e td, mas o problema persiste. O erro que aparece é algo sobre conexão não estabelecida
[agent] Entendi, show! Vou verificar se há algum problema com a nossa rede na sua área. Você pode me informar qual é o seu endereço e o número do seu plano de internet?
[customer] Eu moro...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #25: `conv_00418`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7938)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7938). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com uma duvida sobre a minha conta?
[agent] Claro, beleza! Estou aqui para ajudar. Qual é a sua dúvida sobre a conta?
[customer] Eu queria saber se td bem com a minha fatura do mes passado, pq eu ainda nao recebi
[agent] Show, vamos dar uma olhada! Você pode me informar o seu número de telefone ou código de cliente para eu verificar?
[customer] meu num é 123456, td bem
[agent] Ta bom, encontrei a sua conta. Parece que houve um problema com o envio da fatura, mas eu já estou resolvendo. Você receberá uma nova fatura em breve, ok?
[customer] ok, obrigado! td bem ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #26: `conv_00207`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7864)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7864). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc pode me ajudar com um problema que estou tendo com a minha conta? td que eu faco da erro
[agent] Claro, beleza! Eu estou aqui para ajudar. Pode me contar mais sobre o problema que você está enfrentando com a sua conta? Qual é o erro que você está vendo?
[customer] eu tentei comprar um produto ontem e deu erro no pagamento, agora eu to tentando novamente e ta dando erro de novo, show, isso ta me deixando louco
[agent] Sinto muito pelo inconveniente. Vou verificar o que pode estar acontecendo. Você pode me informar o número do pedido ou o e-mail que você usou para fazer a compr...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #27: `conv_00527`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7856)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7856). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com um problema? td que eu faço para marcar uma consulta no site da clinica da um erro
[agent] Claro, beleza! Posso tentar ajudar sim. Qual é o erro que você está vendo quando tenta marcar a consulta? Você consegue me passar a mensagem de erro exata?
[customer] nao sei, show, é algo tipo 'erro interno' e não da mais info. ja tentei limpar o cache e td mais
[agent] Entendi, vamos tentar resolver isso. Você pode me informar qual é o seu número de cadastro ou CPF para que eu possa verificar o que está acontecendo?
[customer] ta bom, meu cpf é 12345678901. vc acha ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #28: `conv_synth_02500`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.7687)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=saudacao, similarity=0.7687). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] OI! Quero saber uma coisa... vocês atendem até que horas hoje? Preciso resolver uma coisa URGENTE e não quero perder viagem!
[agent] Olá! Nosso atendimento funciona de segunda a sexta das 8h às 20h, e aos sábados das 8h às 14h. Hoje é quinta-feira, então estamos abertos até às 20h. Em que posso ajudá-lo?
[customer] Tá bom, pelo menos isso vocês fazem direito! Mas escuta, queria saber se vocês tem loja na região da Vila Madalena também, pq aqui no Butantã sempre tem fila absurda
[agent] Sim, temos uma loja na Vila Madalena! Fica na Rua Harmonia, 456. Geralmente tem menos movimento qu...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #29: `conv_00197`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `elogio` (similarity: 0.7836)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=elogio, similarity=0.7836). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Eu fiz uma compra na loja vc e td chegou certinho, show!
[agent] Beleza! Fico feliz em saber que tudo chegou conforme esperado. Obrigado por comprar conosco! Tem algo mais que possamos ajudar ou é só um contato amigável?
[customer] Na verdade, eu queria saber se vc td tem algum programa de fidelidade ou algo assim
[agent] Sim, temos um programa de fidelidade que oferece descontos e benefícios exclusivos para nossos clientes regulares. Posso te enviar mais informações sobre isso por e-mail, ta bom?
[customer] Ta bom, sim, por favor! Eu adoraria saber mais sobre como fun...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #30: `conv_00411`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.8098)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.8098). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com uma duvida sobre a minha conta?
[agent] Claro, beleza! Estou aqui para ajudar. Qual é a sua dúvida?
[customer] Eu queria saber se td bem com a minha fatura do mes passado, pq eu ainda nao recebi
[agent] Show, vamos verificar isso para você. Você pode me passar o seu número de telefone ou código de cliente para eu consultar?
[customer] Ta bom, meu numero é 999999999, e o codigo é 123456
[agent] Ok, já encontrei a sua conta. Verifiquei e tudo está em ordem com a sua fatura do mês passado. Ela foi emitida e deve chegar até o dia 15 deste mês. Se tiver mais alg...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #31: `conv_00026`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7900)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.79). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] vc pode me ajudar com um problema que eu estou tendo com a minha conta? td que eu faço da erro
[agent] Claro, beleza! Eu posso te ajudar com isso. Pode me contar mais sobre o erro que você está tendo? Qual é o mensagem de erro que aparece?
[customer] nao sei, show... eu apenas sei que quando eu vou fazer um pedido, da um erro e nao deixa eu finalizar a compra. vc acha que pode ser problema com o meu cartao?
[agent] Sim, pode ser um problema com o cartão, mas também pode ser um problema com o nosso sistema. Vou verificar isso para você. Você pode me dar o número do seu pedido ou o e-...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #32: `conv_00051`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.8174)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=saudacao, similarity=0.8174). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Eae, tudo bem? Quero saber se vc pode me ajudar com uma coisa
[agent] Claro, beleza! Estou aqui para ajudar. Qual é o problema ou dúvida que você tem?
[customer] Td bem, obrigado! Eu queria saber se é possivel fazer um pedido de 2° via do cartao, pq eu perdi o meu
[agent] Show, não se preocupe! Posso ajudar com isso. Você pode fazer o pedido de 2ª via pelo nosso app ou eu posso fazer aqui mesmo. Qual é a sua preferência?
[customer] Beleza, vc pode fazer aí, por favor? E quanto vai demorar pra chegar?
[agent] Ok, já fiz o pedido. O cartão deve chegar em até 10 dias úteis. Se tiver al...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #33: `conv_00852`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7702)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7702). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olha, eu estou td frustrado com a imobiliaria vc! Eu comprei um apto com vc e agora estou tendo problemas com a documentacao, show, ta demorando muito pra resolver
[agent] Sinto muito em saber que você está tendo problemas com a documentação do seu apartamento. Posso ajudar a resolver isso para você? Qual é o número do seu processo ou o endereço do imóvel para que eu possa verificar o que está acontecendo?
[customer] O numero do processo é 1234, vc pode verificar o que ta acontecendo? Eu ja liguei varias vezes e ninguem me deu uma resposta clara, beleza?
[agent] Verifiquei o seu pro...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #34: `conv_00205`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7978)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.7978). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, eu estou tendo um problema com a minha conta, vc pode me ajudar?
[agent] Claro, beleza! Estou aqui para te ajudar. Qual é o problema que você está tendo com a sua conta?
[customer] eu tentei fazer um pedido ontem e deu erro, agora td o que eu faco da erro, show
[agent] Peço desculpas pelo inconveniente. Você pode me fornecer mais detalhes sobre o erro que está aparecendo? Qual é a mensagem exata que você está vendo?
[customer] nao lembro direito, mas era algo sobre problema no servidor, vc pode verificar o que aconteceu com o meu pedido?
[agent] Vou verificar isso para você. Voc...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #35: `conv_00314`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.8081)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.8081). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] E aí, td bem? Estou com um problema aqui, vc pode me ajudar?
[agent] Beleza! Claro que posso tentar ajudar. Qual é o problema que você está enfrentando?
[customer] Eu tentei fazer um pagamento ontem e deu erro, agora o dinheiro ta pendente e eu não sei o que fazer. Já perdi td o dia ligando pra vocês e ninguém resolveu
[agent] Sinto muito pelo inconveniente. Posso tentar ajudar a resolver isso para você. Você pode me informar o número do pagamento ou a data que tentou fazer o pagamento?
[customer] era ontem, por volta das 15h. Eu não tenho o numero do pagamento, mas posso verificar ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #36: `conv_00744`

- **Original label:** `outros`
- **Action taken:** `moved_to_abstention`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.8123)
- **Reason:** Ambiguous 'outros' record near intent boundary (nearest=reclamacao, similarity=0.8123). Retained for abstention threshold calibration.

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olha, eu fiz um pedido pelo app de vc e td deu errado, o delivery demorou mais de 2h e o lanche veio errado, ta bom?
[agent] Desculpe muito pelo ocorrido, beleza. Posso ajudar a resolver isso para você. Você poderia me fornecer o número do pedido para que eu possa verificar o que aconteceu?
[customer] O numero do pedido e 1234, vc pode resolver isso ai?
[agent] Obrigado por fornecer o número do pedido. Já verifiquei e parece que houve um problema com a entrega. Vou oferecer um desconto de 20% no seu próximo pedido e também vamos reembolsar o valor do pedido errado. Você gostaria que...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

## Section 4: Out-of-Scope Removed (54 records)

### Record #37: `conv_00845`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `elogio` (similarity: 0.7470)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=elogio, similarity=0.747)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Ola, td bem? Quero fazer um elogio sobre o atendimento da sua imobiliaria, vc me ajudou a encontrar um apto show na zona sul
[agent] Olá! Muito obrigado pelo elogio, beleza! Fico feliz em saber que conseguimos ajudar você a encontrar o apartamento perfeito. Qual foi o nome do nosso consultor que te atendeu?
[customer] Ah, foi o Joao, ele foi super atencioso e me mostrou varios imoveis até achar o que eu queria, td give
[agent] Certo, vamos passar um elogio para o Joao, ele vai ficar muito feliz! Se precisar de mais alguma coisa, não hesite em entrar em contato, ta bom?
[customer] Ta...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #38: `conv_synth_03429`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7386)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.7386)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, desculpa o incomodo mas acho que liguei no numero errado, queria falar com a central do meu cartao mas ta dando ocupado direto, que saco
[agent] Oi! Tudo bem, sem problema algum! Você ligou para o Banco Digital Plus. Qual cartão você está tentando entrar em contato? Posso te passar o número correto
[customer] eh o cartao vermelho, aquele do banco nacional, ta uma droga tentar falar com eles hoje
[agent] Entendi! O número da central do Banco Nacional é 0800-123-4567. Mas se quiser, posso tentar te ajudar também - às vezes conseguimos resolver algumas questões básicas aqui
[custom...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #39: `conv_00739`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.7384)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.7384)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc td bem?
[agent] Tudo bem, obrigado! E você? Em que posso ajudar hoje?
[customer] td bem, show! eu queria saber se vc tem algum programa de fidelidade
[agent] Sim, temos um programa de fidelidade que oferece descontos e promoções exclusivas para nossos clientes frequentes. Você pode encontrar mais informações sobre isso na nossa página de perfil no app
[customer] beleza, vou dar uma olhada. e como eu consigo os pontos?
[agent] Você consegue os pontos realizando pedidos através do nosso app. A cada pedido, você ganha um determinado número de pontos, que podem ser trocados por p...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #40: `conv_00530`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7330)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.733)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Ola, vc pode me ajudar com um problema? Estou td frustrado com o atendimento da clinica
[agent] Claro, beleza! Eu estou aqui para ajudar. Qual é o problema que você está enfrentando? Pode me contar mais sobre o que aconteceu?
[customer] Eu tive um exame marcado pra semana passada e ate agora nao recebi o resultado. Liguei varias vezes e ninguem me respondeu. Isso ta demorando td
[agent] Sinto muito pelo transtorno. Vou verificar o que aconteceu com o seu exame. Você pode me fornecer o numero do seu protocolo ou o nome do medico que solicitou o exame?
[customer] O numero do protocolo...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #41: `conv_00522`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `elogio` (similarity: 0.7321)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=elogio, similarity=0.7321)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olá, tudo bem? Quero saber sobre os exames que fiz na clinica na semana passada, vc pode me dar alguma info?
[agent] Bem-vindo! Claro, posso te ajudar com isso. Qual é o seu nome e a data em que você realizou os exames?
[customer] Meu nome é João, e eu fiz os exames no dia 10. Td foi muito show, os funcionarios foram super atenciosos, beleza!
[agent] Obrigado por elogiar nossa equipe, João! Já encontrei seus exames. Você gostaria de saber os resultados por aqui ou preferiria que eu enviasse por e-mail?
[customer] Pode enviar por e-mail, ta bom? E vc pode me dizer se td tá ok com meu...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #42: `conv_00738`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_servico` (similarity: 0.7299)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_servico, similarity=0.7299)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Eae, vc poderia me falar sobre como funciona o programa de fidelidade do restaurante?
[agent] Beleza, tudo bem? Nosso programa de fidelidade é bem simples, a cada 10 pedidos você ganha um desconto de 10% no próximo pedido, e também recebe ofertas exclusivas por e-mail.
[customer] Show, td bem simples mesmo! E como eu faço pra saber quantos pedidos eu ainda tenho pra chegar nos 10?
[agent] Você pode verificar isso no nosso app, na seção 'Meus Pedidos', ou então eu posso verificar aqui para você, basta me informar o seu número de telefone ou e-mail cadastrado.
[customer] Ta bom, eu vo...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #43: `conv_00421`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.7274)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=suporte_tecnico, similarity=0.7274)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olha, vc pode me ajudar com um problema que eu estou tendo com a minha internet? Ta bem lenta
[agent] Claro, beleza! Posso tentar ajudar sim. Qual é o problema exatamente? Você está tendo dificuldades para acessar sites ou é uma questão de velocidade mesmo?
[customer] Na verdade, td foi bem até ontem, mas hoje de manha acordei e tava tudo muito lento. Show, eu fiz td certo, reinicie o roteador e td mais, mas nao adiantou
[agent] Entendi, vamos tentar resolver isso. Você pode me informar o seu número de conta ou o CPF que está vinculado à conta para que eu possa verificar o que está ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #44: `conv_00854`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7249)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.7249)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com um problema? Eu comprei um apartamento com vc e agora estou tendo td tipo de problema com a documentacao, ta bom?
[agent] Claro, beleza! Lamento muito pelos problemas que você está enfrentando. Você pode me fornecer mais detalhes sobre o que está acontecendo com a documentação do seu apartamento? Qual é o número do seu contrato ou o endereço do imóvel?
[customer] o contrato e 1234 e o endereço e rua exemplo, 123. td parece que foi feito errado, show
[agent] Entendi, obrigado por fornecer essas informações. Vou verificar o que está acontecendo e já ligo para...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #45: `conv_synth_02490`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.7235)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.7235)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vocês atendem que horas mesmo? Preciso resolver uma coisa urgente
[agent] Olá! Nosso atendimento funciona de segunda a sexta das 8h às 20h, e sábados das 8h às 14h. Em que posso ajudá-lo hoje?
[customer] Ah que droga, então domingo não tem atendimento? ABSURDO isso, banco que se preze tinha que funcionar 24h
[agent] Entendo sua preocupação. Infelizmente aos domingos não temos atendimento presencial, mas você pode usar nosso internet banking ou app para a maioria dos serviços. Qual seria a urgência?
[customer] Não é nada do app não, é uma coisa que SÓ resolve falando com vocês. Q...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #46: `conv_00419`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.7204)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=suporte_tecnico, similarity=0.7204)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com um problema que ta acontecendo com a minha internet? td estava funcionando bem, mas agora ta muito lento
[agent] Claro, beleza! Lamento ouvir que você está tendo problemas com a internet. Você pode me informar qual é o seu plano de internet e se você já reiniciou o roteador?
[customer] Eu tenho o plano de 500mb e sim, eu já reiniciei o roteador, mas td continua igual. acho que o problema ta com a sua empresa, show
[agent] Entendi, vamos verificar então. Você pode me fornecer o seu número de telefone ou o código do seu plano para que eu possa verificar o que...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #47: `conv_00745`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7163)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.7163)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar? Fiz um pedido no app de vc e td deu errado
[agent] Claro, beleza! Peço desculpas pelo inconveniente. Pode me contar mais sobre o que aconteceu com o seu pedido? Qual foi o número do pedido?
[customer] era o pedido 1234, eu pedi uma pizza e td veio errado, a pizza veio queimada e o delivery demorou td
[agent] Sinto muito pelo ocorrido. Vou verificar o que aconteceu com o seu pedido. Você gostaria de uma nova pizza ou um reembolso?
[customer] eu acho que prefiro o reembolso, ta bom?
[agent] Tudo bem, show! Vou solicitar o reembolso do valor do seu pedido. Você r...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #48: `conv_00635`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.7155)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=suporte_tecnico, similarity=0.7155)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olha, vc pode me ajudar com um problema? Eu fiz uma inscricao no curso de ingles online e td deu errado, a plataforma nao funciona direito e eu nao consigo assistir as aulas
[agent] Claro, beleza! Desculpe pelo transtorno. Pode me dar mais detalhes sobre o problema que você está enfrentando? Qual é o erro exato que você está vendo na plataforma?
[customer] nao sei, show... eu clico em assistir aula e da um erro de servidor, ja tentei em outros computadores e td, mas nao funciona, eu estou muito frustrado pq eu preciso aprender ingles pro trabalho
[agent] Entendi, vamos tentar resolv...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #49: `conv_synth_02485`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7147)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.7147)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] VCS ESTAO ABERTOS OU NAO??? Liguei 15 vezes e ninguem atende, ABSURDO!
[agent] Oi! Sim, estamos funcionando normalmente. Peço desculpas pela dificuldade em nos contatar por telefone. Nosso horário de funcionamento é das 18h às 23h. Em que posso te ajudar?
[customer] EU TO TENTANDO FALAR COM VCS HA 2 HORAS! Quero saber se vcs fazem entrega no bairro Vila Nova, mas pelo visto o atendimento de vcs e uma PORCARIA
[agent] Entendo sua frustração e realmente peço desculpas pela dificuldade. Sim, fazemos entrega na Vila Nova! A taxa é R$ 8,00 e o tempo médio de entrega é de 45 minutos. Gost...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #50: `conv_synth_02497`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.7116)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.7116)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, eu liguei pra vcs mas ninguem atendeu... qual o horario de funcionamento do atendimento mesmo?
[agent] Oi! Desculpa por isso. Nosso atendimento telefônico funciona de segunda a sexta das 8h às 18h, e aos sábados das 9h às 15h. Pelo chat aqui você consegue falar conosco 24h. Em que posso te ajudar?
[customer] cara, que sacanagem... eu tentei ligar umas 5 vezes ontem a noite e nada. pq vcs nao colocam isso no site de forma mais clara?
[agent] Entendo sua frustração e peço desculpas pela inconveniência. Vou anotar sua sugestão sobre deixar os horários mais visíveis no site. Era alg...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #51: `conv_00412`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `compra` (similarity: 0.7077)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=compra, similarity=0.7077)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Quero saber se vc tem algum plano de internet que seja mais barato e com mais dados
[agent] Beleza! Sim, temos algumas opções de planos que podem atender às suas necessidades. Qual é o seu atual plano e quanto você paga por mês?
[customer] Eu pago R$ 120 por mês e tenho 100gb de dados. Mas td bem, eu quero algo mais barato e com mais dados, ta bom?
[agent] Entendi! Nós temos um plano que pode ser interessante para você. É o nosso plano 'Internet Plus' que oferece 200gb de dados por R$ 100 por mês. Você gostaria de saber mais sobre ele?
[customer] Sim, show! Quero saber...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #52: `conv_00311`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.7068)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=suporte_tecnico, similarity=0.7068)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Ola, vc pode me ajudar? Estou tendo um problema com o meu cartao, td mundo me disse que ele ta funcionando normal, mas eu nao consigo fazer compras online
[agent] Beleza, vamos dar uma olhada nisso. Você pode me informar o erro que está aparecendo quando você tenta fazer uma compra online? É algum código ou mensagem de erro específica?
[customer] Na verdade, nao da nenhum erro, simplesmente nao finaliza a compra. Eu ja tentei em varios sites e td mundo me disse que o cartao ta ok, mas eu nao consigo
[agent] Entendi, show, vamos investigar mais a fundo. Você pode me confirmar se o se...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #53: `conv_00743`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.7059)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.7059)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Ei, vc pode me ajudar? Fiz um pedido pelo app ontem e ainda não recebi
[agent] Claro, beleza! Posso ajudar com isso. Qual o número do seu pedido, por favor?
[customer] td bem, o numero é #1234. E td isso ta demorando muito, show
[agent] Peço desculpas pelo inconveniente. Vou verificar o que aconteceu com o seu pedido. Você pode me informar qual o endereço de entrega, por favor?
[customer] ta bom, é rua das flores, 123. Espero que vc consiga resolver isso, ta me deixando louco
[agent] Entendi, vamos resolver isso. Encontrei o problema e o seu pedido será entregue ainda hoje. Se não r...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #54: `conv_synth_02478`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.7053)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.7053)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, sera que vc pode me falar o horario de funcionamento de vcs? preciso saber pq to pensando em passar la hoje a noite
[agent] Olá! Claro, posso sim te ajudar com essa informação. Nosso restaurante funciona de segunda a domingo das 11h às 23h. Hoje estamos abertos normalmente até às 23h. Você pretende vir para jantar?
[customer] show, obrigado! e ate que horas vcs fazem pedido de delivery? pq as vezes da vontade de pedir mas nao sei se ainda ta funcionando
[agent] Beleza! Para delivery fazemos pedidos até às 22h30, para dar tempo da entrega chegar antes do fechamento. Nosso tempo m...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #55: `conv_00848`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_servico` (similarity: 0.6615)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_servico, similarity=0.6615)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me dar alguma informacao sobre como funciona o processo de compra de um imovel na sua imobiliaria?
[agent] Beleza! Claro, posso te explicar tudo. Nosso processo começa com uma consulta ao nosso portifolio de imoveis, onde vc pode escolher o que melhor se adequa as suas necessidades. Em seguida, nos fazemos uma visita ao imovel, e se vc estiver interessado, podemos prosseguir com a proposta de compra.
[customer] Show, ta bom! E como funciona o pagamento? Vc aceita pagamento a prazo?
[agent] Sim, aceitamos pagamento a prazo. Nossa equipe de financiamento pode te ajudar a e...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #56: `conv_synth_02468`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.6608)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.6608)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, eu to meio perdido aqui... sera que vc pode me ajudar? Eu liguei pra outro numero e me passaram pra ca
[agent] Oi! Claro, posso sim te ajudar. Aqui é o atendimento da EduTech Online. Qual era a informação que você estava procurando?
[customer] Ah cara, que droga... eu queria falar sobre um curso de programacao mas acho que liguei no lugar errado mesmo. Vcs nao tem nada a ver com isso ne?
[agent] Na verdade temos sim! A EduTech oferece vários cursos de programação online. Que tipo de programação você estava procurando? Python, Java, desenvolvimento web?
[customer] Serio? Que mass...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #57: `conv_00735`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `elogio` (similarity: 0.6538)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=elogio, similarity=0.6538)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pessoal ta bom? Fiz um pedido ontem e queria saber se vc tem alguma promoção nova
[agent] Bem-vindo ao nosso restaurante! Sim, estamos bem, obrigado por perguntar! Sim, temos algumas promoções novas que podem interessar, como nosso combo de sexta-feira com 10% de desconto em todos os pedidos acima de R$50
[customer] Show, td bem! Eu adorei o pedido que fiz ontem, o delivery foi super rapido e o lanche estava uma delicia! Quero saber mais sobre essa promoção de sexta-feira, como funciona exatamente?
[agent] Muito obrigado pelo feedback, ficamos felizes em saber que você gostou...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #58: `conv_synth_02474`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.6514)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.6514)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Bom dia, minha filha. Por gentileza, poderia me informar qual é o horário de funcionamento da agência da Rua das Flores?
[agent] Bom dia! Claro, posso ajudar sim. A agência da Rua das Flores funciona de segunda a sexta das 10h às 16h. Aos sábados não temos atendimento nessa unidade.
[customer] Muito obrigado pela informação, querida. E me diga uma coisa, se não for incômodo, vocês atendem no horário de almoço também?
[agent] Sim senhor, atendemos normalmente no horário de almoço. A agência não fecha para o intervalo, então o senhor pode ir a qualquer momento entre 10h e 16h que tere...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #59: `conv_synth_02467`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.6504)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.6504)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, td bem? Vc pode me falar qual o horario de funcionamento das agencias do banco? Preciso ir la pessoalmente amanha
[agent] Olá! Tudo bem sim, obrigado! Claro, posso te ajudar. Nossas agências funcionam de segunda a sexta das 10h às 16h. Aos sábados algumas agências abrem das 9h às 12h. Você precisa de alguma agência específica?
[customer] Show! Preciso da agencia da Av Paulista, vc sabe se ela abre no sabado tb? E qual o melhor horario pra ir pra nao pegar fila?
[agent] A agência da Av Paulista funciona sim aos sábados das 9h às 12h! Para evitar filas, recomendo ir pela manhã ent...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #60: `conv_00196`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.6403)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.6403)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Eu estava navegando pelo site e achei super show! Vc tem alguma promoção legal nos proximos dias?
[agent] Tudo bem, obrigado por entrar em contato! Sim, temos algumas promoções interessantes nos próximos dias. Posso te informar sobre elas, ou você tem algo específico em mente?
[customer] Beleza, td bem! Eu estava procurando por um novo smartphone, e vi que vc tem uma linha de celulares com desconto. Pode me falar mais sobre isso?
[agent] Claro! Nossa linha de smartphones com desconto é muito popular. Estamos com uma promoção de até 20% off em alguns modelos. Qual é o s...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #61: `conv_00630`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.6393)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.6393)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Estou curtindo muito o curso de ingles online que estou fazendo aqui
[agent] Beleza! Fico feliz em saber que vc está gostando do curso. Como posso ajudar vc hoje?
[customer] Na verdade, eu tava com uma duvida sobre como acessar os materiais extras do curso. Vc pode me ajudar com isso?
[agent] Claro, td bem simples. Basta vc clicar no botão de 'recursos adicionais' na página do curso e vc terá acesso a todos os materiais. Se tiver alguma dificuldade, é só me chamar
[customer] Show, funcionou! Muito obrigado pela ajuda, vc é um cara muito atencioso
[agent] De nada, foi u...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #62: `conv_synth_02462`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.6366)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.6366)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] VOCÊS SÃO UMA PIADA! Liguei no número que tava no Google e caí aqui, que ABSURDO! Quero falar com alguém da secretaria da Escola Técnica São Paulo AGORA!
[agent] Oi! Peço desculpas pela confusão, mas aqui é o atendimento da UniCursos Online. Parece que você ligou no número errado mesmo. A Escola Técnica São Paulo tem um número diferente do nosso.
[customer] QUE PALHAÇADA! Perdi 20 minutos esperando na linha pra descobrir que é número errado?! Vcs não podem atualizar essas informações no Google não?!
[agent] Entendo sua irritação, realmente é frustrante quando isso acontece. Infelizm...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #63: `conv_00306`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `compra` (similarity: 0.6365)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=compra, similarity=0.6365)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Quero saber se vc tem alguma promocao para novos clientes
[agent] Tudo bem, sim! Estamos com uma promoção especial para novos clientes, você pode ganhar até 50 reais de bônus na sua conta corrente. Quer saber mais sobre como funciona?
[customer] Beleza, ta bom! Quero saber mais, show! Como eu consigo esse bônus?
[agent] Para conseguir o bônus, você precisa realizar um depósito inicial de pelo menos 100 reais e manter a conta ativa por pelo menos 30 dias. Além disso, você precisa fazer pelo menos 5 transações com o cartão dentro desse período. Quer que eu te envie um li...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #64: `conv_00850`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_servico` (similarity: 0.6318)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_servico, similarity=0.6318)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olá, vc pode me informar sobre como funciona o processo de compra de um imovel na sua imobiliaria?
[agent] Beleza! Claro, posso te explicar tudo sobre o nosso processo de compra. Basicamente, começamos com uma consulta sobre o que você está procurando, seja um apartamento, casa ou terreno. Em seguida, nossa equipe vai te apresentar opções que se encaixam no seu perfil e orçamento. Você também pode fazer uma simulação de financiamento conosco. Quer saber mais sobre algum desses passos em particular?
[customer] td bem, show! Quero saber mais sobre a simulação de financiamento. Como fu...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #65: `conv_00024`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `elogio` (similarity: 0.6314)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=elogio, similarity=0.6314)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Fiz uma compra na semana passada e td deu certo, show!
[agent] Olá! Tudo bem sim, obrigado por entrar em contato! Fico feliz em saber que a compra deu certo. Qual o motivo da sua mensagem hoje?
[customer] Na verdade, eu queria saber se vc td os produtos que eu comprei tem garantia, beleza?
[agent] Sim, todos os produtos que vendemos têm garantia. A garantia varia de acordo com o tipo de produto, mas em geral, é de 1 a 2 anos. Se você tiver alguma dúvida específica sobre a garantia de um produto, posso verificar para você.
[customer] Beleza, vc pode me enviar um e-mail ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #66: `conv_00742`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_produto` (similarity: 0.6208)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_produto, similarity=0.6208)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Eae, vc td bem? Quero saber se vc tem alguma promocao no app
[agent] Beleza! Tudo bem aqui, obrigado por perguntar! Sim, temos algumas promoções ativas no app, como descontos em pedidos acima de R$50 e um programa de fidelidade que oferece pontos por cada pedido realizado
[customer] Show, td bem! Eu quero saber mais sobre esse programa de fidelidade, como funciona exatamente?
[agent] Claro, ta bom! O programa de fidelidade é simples: você ganha pontos por cada pedido, e esses pontos podem ser trocados por descontos ou até mesmo por pedidos gratuitos. Você pode ver mais detalhes sobr...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #67: `conv_00847`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_servico` (similarity: 0.6170)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_servico, similarity=0.617)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olá, vc pode me informar sobre como funciona o processo de compra de um imovel na sua imobiliaria?
[agent] Beleza! Claro, posso te explicar tudo. Nosso processo começa com a escolha do imóvel, depois temos a avaliação do crédito e a aprovação do financiamento, se necessário. Em seguida, realizamos a negociação do preço e, por fim, a assinatura do contrato. Show, né? Quer saber mais sobre algum desses passos?
[customer] Sim, td bem. Mas eu queria saber sobre a documentacao necessaria, ta bom?
[agent] Tá bom, sem problemas! A documentação necessária inclui RG, CPF, comprovante de rend...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #68: `conv_00524`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `compra` (similarity: 0.6149)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=compra, similarity=0.6149)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me informar sobre como funciona o plano de saude da sua clinica?
[agent] Beleza! Claro, posso te explicar tudo sobre nossos planos de saude. Nossa clinica oferece varios tipos de planos, desde o basico até o mais completo, com cobertura para td tipo de procedimento medico.
[customer] Show, td bem! E como eu faço para escolher o plano certo pra mim?
[agent] Ta bom, para escolher o plano certo, precisamos saber um pouco mais sobre vc. Pode me informar sua idade, quantidade de dependentes e quais sao suas necessidades medicas atuais?
[customer] Ok, eu tenho 35 anos, sou cas...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #69: `conv_synth_02484`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.6107)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.6107)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Boa tarde. Gostaria de saber qual é o horário de funcionamento do escritório de vocês na Rua Augusta.
[agent] Boa tarde! Claro, posso informar sim. Nosso escritório na Rua Augusta funciona de segunda a sexta das 8h às 18h, e aos sábados das 8h às 14h. Domingos não atendemos. Posso ajudar com mais alguma coisa?
[customer] Perfeito, muito obrigado pela informação. E vocês atendem por ordem de chegada ou é necessário agendar horário?
[agent] Atendemos por ordem de chegada para consultas gerais, mas para visitas a imóveis específicos recomendamos agendar com antecedência para garantir q...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #70: `conv_00201`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.6086)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.6086)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc pode me ajudar com uma coisa? td bem?
[agent] Tudo bem, sim! Estou aqui para ajudar. O que você precisa saber ou resolver?
[customer] eu to procurando um presente pra minha irmã e encontrei um relógio aqui na loja online, mas não sei se é bom, vc tem alguma info sobre ele?
[agent] Claro, posso te ajudar com isso! Qual é o modelo do relógio que você encontrou? Assim posso verificar as especificações e o que os outros clientes acham dele.
[customer] acho que é um da marca xyz, mas não lembro o modelo, pode ser que seja o xyz123, td bem, vc pode me falar um pouco sobre a marca e...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #71: `conv_synth_02492`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.6065)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.6065)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Bom dia, minha filha. Por gentileza, eu preciso de uma informação. Eu liguei para esse número porque estava procurando o telefone da Secretaria da Educação, mas parece que me enganei. Vocês são de uma escola particular, não é mesmo?
[agent] Bom dia! Isso mesmo, o senhor ligou para o Instituto Educacional São Paulo, somos uma escola de cursos técnicos e preparatórios. Acredito que tenha sido um engano mesmo. O senhor estava procurando especificamente a Secretaria da Educação?
[customer] Ah, que aborrecimento! Eu estava tentando resolver uma questão sobre a aposentadoria do meu filho ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #72: `conv_00521`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.6039)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.6039)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? vc pode me ajudar com uma coisa?
[agent] Beleza! Claro que posso te ajudar. O que está acontecendo?
[customer] Eu estava procurando infos sobre os exames que oferecem aí na clinica e td mais
[agent] Show! Nós oferecemos uma variedade de exames e serviços. Você pode acessar nosso site ou eu posso te encaminhar para alguém que pode te dar mais infos específicas
[customer] Ah, beleza! Eu acho que vou dar uma olhada no site primeiro, mas se tiver alguma dúvida, eu volto a entrar em contato, ta bom?
[agent] Tudo certo! Fique à vontade para nos contatar se precisar de mais a...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #73: `conv_00853`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `compra` (similarity: 0.5335)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=compra, similarity=0.5335)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olha, vc pode me ajudar com um problema? Eu estou tentando encontrar um imovel para alugar e td que eu encontro e caro demais ou nao tem o que eu preciso
[agent] Claro, beleza! Posso te ajudar a encontrar um imovel que atenda as suas necessidades. Pode me dar mais detalhes sobre o que vc esta procurando? Qual o bairro, quantidade de quartos, valor maximo que vc ta disposto a pagar?
[customer] Ta bom, eu estou procurando algo em Sao Paulo, preferencia pelo bairro de Pinheiros, 2 quartos, e o maximo que eu posso pagar e R$ 2.500,00. Ja procurei td e nao encontrei nada que se encaixe n...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #74: `conv_00526`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_produto` (similarity: 0.5327)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_produto, similarity=0.5327)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc poderia me informar sobre os exames que oferecem na clinica? td bem
[agent] Beleza! Claro, podemos oferecer uma variedade de exames, desde os basicos até os mais complexos. Qual o tipo de exame que você está procurando?
[customer] nao sei, td bem... eu quero saber sobre os preços tambem, por favor
[agent] Show, entendi! Nossos preços variam de acordo com o tipo de exame e também com o seu plano de saúde. Você poderia me informar qual o seu plano de saúde atual?
[customer] eu tenho o plano gold, vc acha que eu consigo fazer um exame de ressonancia magnetica com ele?
[agent] Si...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #75: `conv_synth_02465`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.5303)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.5303)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi mano, vc sabe me dizer se a clínica abre no sabado? preciso marcar uns exames mas só consigo ir fim de semana tlgd
[agent] Oi! Sim, funcionamos aos sábados das 8h às 14h. Que tipo de exames você precisa fazer? Posso te ajudar com as informações de agendamento.
[customer] massa! preciso fazer hemograma completo e uns outros exames de sangue que o médico pediu. é tranquilo marcar pra sabado que vem?
[agent] Perfeito! Exames de sangue são bem tranquilos de agendar. Para sábado que vem temos horários disponíveis das 8h às 10h. O hemograma completo custa R$ 35 e precisa de jejum de 12...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #76: `conv_00849`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `compra` (similarity: 0.5287)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=compra, similarity=0.5287)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Olá, eu estou procurando por uma imobiliária que possa me ajudar a encontrar um imóvel para comprar ou alugar. Vc tem algum tipo de consulta ou coisa assim?
[agent] Beleza! Claro, podemos te ajudar com isso. Nossa imobiliária oferece uma variedade de opções para compra e aluguel. Você está procurando por algo específico, como um apartamento ou uma casa?
[customer] Na verdade, td depende do preço e da localização. Eu estou procurando por algo em São Paulo, mas não sei se é melhor comprar ou alugar. Você pode me dar algumas dicas?
[agent] Show, é uma boa pergunta! Comprar ou alugar de...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #77: `conv_00632`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.5266)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.5266)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc poderia me dar info sobre como funciona o curso de ingles online?
[agent] Beleza! Claro, posso te dar todas as informações sobre o curso de inglês online. O curso é dividido em módulos e você pode acessar as aulas a qualquer momento, desde que tenha internet. Qual é o seu nível de inglês atual?
[customer] ta bom, eu sei falar um pouco, mas quero melhorar td. Como eu sei que estou progredindo?
[agent] Show, que ótimo! Nós temos um sistema de avaliação contínua, onde você pode acompanhar seu progresso por meio de testes e exercícios. Além disso, você também terá acesso a um tut...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #78: `conv_00627`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.5252)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.5252)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Ola, td bem? Estou aqui pq quero saber mais sobre o curso de ingles online vc oferece
[agent] Tudo bem, obrigado! Claro, posso te ajudar com isso. Nosso curso de ingles online é muito completo e tem varios niveis, desde iniciante até avançado. Qual o seu nivel de ingles atualmente?
[customer] Beleza, show! Eu já tenho um nivel intermediario, mas quero melhorar minha pronuncia e vocabulario. Vc acha que o curso pode me ajudar nisso?
[agent] Com certeza! Nosso curso tem uma grande enfase na pronuncia e vocabulario, com aulas interativas e exercicios práticos. Além disso, vc terá acess...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #79: `conv_synth_02481`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.5051)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.5051)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Boa tarde, minha filha. Por gentileza, poderia me informar qual é o horário de funcionamento das lojas da Vivo aqui na cidade?
[agent] Boa tarde! Claro, posso ajudá-lo com essa informação. Nossas lojas Vivo funcionam de segunda a sexta das 9h às 18h, e aos sábados das 9h às 14h. Aos domingos ficamos fechados. O senhor está procurando alguma loja específica?
[customer] Que bom, obrigado pela informação. Eu estava pensando em ir na loja do shopping Iguatemi, se não for incômodo, a senhora poderia me confirmar se essa loja tem o mesmo horário?
[agent] Perfeitamente! A loja do Shopping ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #80: `conv_00304`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `reclamacao` (similarity: 0.5000)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=reclamacao, similarity=0.5)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Eae, tudo bem? vc pode me ajudar com uma duvida sobre o meu cartao?
[agent] Beleza! Claro, estou aqui para ajudar. Qual é a sua dúvida sobre o cartão?
[customer] td bem, eu estava pensando em viajar e queria saber se o cartao funciona fora do brasil, ta bom?
[agent] Show! Sim, o nosso cartão é internacional, então você pode usá-lo em qualquer lugar do mundo onde seja aceito. Você precisa apenas ter saldo na conta e não esquecer de informar sobre suas viagens para não ter problemas com o nosso sistema de segurança.
[customer] que show, isso é ótimo! e como eu faço para informar sobre...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #81: `conv_00633`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_servico` (similarity: 0.4942)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_servico, similarity=0.4942)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc tem algum curso de ingles para iniciantes?
[agent] Olá! Sim, temos um curso de inglês para iniciantes que é show, aborda td o básico e é muito prático. Você gostaria de saber mais sobre o currículo e os horários das aulas?
[customer] sim, td bem, quero saber mais sobre o curso, como funciona e quanto custa
[agent] Beleza! Nosso curso de inglês para iniciantes é online, com aulas ao vivo e gravadas, então você pode assistir quando e onde quiser. O custo é de R$ 299 por mês, mas temos promoções e descontos para quem paga à vista ou faz um pacote de meses. Você gostaria de fazer...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #82: `conv_00634`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_servico` (similarity: 0.4844)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_servico, similarity=0.4844)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc poderia me ajudar com uma duvida sobre o curso de ingles online?
[agent] Beleza! Claro, estou aqui para ajudar. Qual é a sua dúvida sobre o curso de inglês online?
[customer] td bem, eu queria saber se eu posso fazer o curso em qualquer horario, tipo, td q eu quiser
[agent] Sim, é possível! Nosso curso de inglês online é bem flexível e você pode acessar o material em qualquer horário que seja conveniente para você. No entanto, é importante notar que alguns recursos, como as aulas ao vivo, têm horários específicos.
[customer] show, isso ta bom pra mim! e como eu faço pra saber...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #83: `conv_00631`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.4833)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.4833)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Estou procurando um curso de ingles online, mas td bem confuso, vc pode me ajudar?
[agent] Beleza! Claro que posso te ajudar. Qual é o seu objetivo com o curso de inglês? Você quer aprender para viagem, trabalho ou algo mais?
[customer] Na verdade, eu quero aprender para uma oportunidade de trabalho. O curso vc oferece é bem completo?
[agent] Show! Nossa escola de idiomas oferece cursos completos com aulas ao vivo, material didático e suporte dos professores. Posso te enviar um link com mais informações sobre o curso que acho que pode ser o melhor para você.
[customer]...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #84: `conv_00629`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.4699)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.4699)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, tudo bem? Quero saber sobre os cursos de ingles que vc oferece
[agent] Tudo bem, sim! Nossa escola de idiomas oferece cursos de inglês para todos os níveis, do básico ao avançado. Qual é o seu objetivo com o curso de inglês?
[customer] Quero melhorar meu ingles para viajar, td bem, mas não sei qual curso escolher. Vc pode me ajudar?
[agent] Claro, beleza! Nossa equipe pode te ajudar a escolher o curso certo para você. Você já tem algum conhecimento de inglês ou é iniciante?
[customer] Sou iniciante, mas estou ansioso para aprender. O que vc acha que eu devo fazer?
[agent] Show! ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #85: `conv_00523`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `duvida_produto` (similarity: 0.4467)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=duvida_produto, similarity=0.4467)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me informar sobre os exames que oferecem na clinica?
[agent] Beleza! Claro, oferecemos uma variedade de exames, desde os mais simples como exame de sangue até os mais complexos como ressonancia magnetica. Qual tipo de exame vc está procurando?
[customer] Td bem, eu estava procurando por um exame de sangue para verificar meus niveis de colesterol. Show, vc pode me dizer quanto custa?
[agent] Ta bom, o exame de sangue para verificar os niveis de colesterol custa em torno de R$ 50,00 a R$ 100,00, dependendo do tipo de exame e do laboratorio. Vou te encaminhar para o nosso s...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #86: `conv_synth_02455`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `suporte_tecnico` (similarity: 0.4371)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=suporte_tecnico, similarity=0.4371)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] ei mano, vcs tem app pra iOS? meu celular é iPhone 12 e n to achando na App Store
[agent] Oi! Temos sim! O TechFlow Pro está disponível na App Store. Você tentou buscar por "TechFlow" ou "TechFlow Pro"? Às vezes demora um pouco pra aparecer nos resultados.
[customer] ah vlw cara! achei aqui kk era só "TechFlow" mesmo. btw, o app é de graça ou tem que pagar?
[agent] Legal que achou! O download é gratuito, mas temos planos premium. O básico é free mesmo, com funcionalidades limitadas. O premium custa R$ 29,90/mês e libera todas as features.
[customer] show de bola! vou baixar aqui e t...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #87: `conv_00025`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `compra` (similarity: 0.4290)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=compra, similarity=0.429)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] oi, vc poderia me falar um pouco sobre como vc ta lidando com a questao do meio ambiente na empresa?
[agent] Beleza! Nossa empresa tem um compromisso forte com a sustentabilidade, show! Nossa politica de meio ambiente visa minimizar o impacto ambiental de nossas operacoes, desde a escolha de fornecedores até a entrega dos produtos aos clientes.
[customer] td bem, mas como vc faz para reduzir o uso de plástico nas embalagens?
[agent] Excelente pergunta! Estamos trabalhando para reduzir o uso de plástico nas embalagens, substituindo por materiais reciclados e biodegradaveis sempre que...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #88: `conv_synth_02476`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `compra` (similarity: 0.4194)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=compra, similarity=0.4194)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] eae mano, vcs tem loja na zona norte? quero ver uns celulares pessoalmente tlgd
[agent] E aí! Temos sim, várias lojas na zona norte. Tem uma no Shopping Center Norte, outra na Rua Voluntários da Pátria no Santana, e uma no Shopping Lar Center. Qual região fica melhor pra você?
[customer] massa! a do center norte é show, fico perto. vcs tem aqueles iphones novos lá? o 15 pro?
[agent] Sim, temos o iPhone 15 Pro em estoque na loja do Center Norte! Tem nas cores azul titânio, titânio natural, titânio branco e titânio preto. Quer que eu reserve um pra você dar uma olhada?
[customer] nao ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #89: `conv_00198`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `saudacao` (similarity: 0.3586)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=saudacao, similarity=0.3586)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] Oi, vc pode me ajudar com uma coisa? Eu estou procurando por um relógio que eu comprei na sua loja há uns 2 anos e agora eu não consigo achar o manual dele
[agent] Claro, beleza! Eu posso tentar ajudar. Você tem o número do pedido ou o modelo do relógio? Isso vai me ajudar a encontrar o manual para você
[customer] Ah, td bem, eu acho que eu tenho o numero do pedido em algum lugar... mas não sei se eu consigo achar. O modelo é um daqueles da marca xyz, td preto
[agent] Show, isso já é um começo! Se você não tiver o número do pedido, não tem problema. Eu posso tentar encontrar o manua...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

### Record #90: `conv_00203`

- **Original label:** `outros`
- **Action taken:** `removed_out_of_scope`
- **Nearest intent (embedding):** `elogio` (similarity: 0.2344)
- **Reason:** Out-of-scope 'outros' record, not close to any intent (nearest=elogio, similarity=0.2344)

<details>
<summary>Conversation text (click to expand)</summary>

```
[customer] ei, vc pode me ajudar com uma coisa? estou procurando por um relógio que eu vi aqui na loja online td bem recente
[agent] Claro, beleza! Eu posso ajudar com isso. Você lembra mais ou menos quando foi que viu o relógio ou alguma característica específica dele, como marca ou cor?
[customer] ah, td bem, foi ontem a noite, acho que era um relógio da marca fossil, mas não sei se era preto ou azul, pq eu não anotei
[agent] Show, vamos ver o que eu consigo encontrar. Você pode me dar um minuto enquanto eu verifico nossos produtos?
[customer] ta bom, demora um pouco, não tem problema, vc é ...
```
</details>

| Decision | Notes |
|---|---|
| `confirm` / `reclassify` / `restore` / `unsure` | |

---

## Review Summary

| Type | Total | Confirmed | Reclassified | Restored | Unsure |
|---|---|---|---|---|---|
| Near-duplicates | 5 | | | | |
| Reclassified | 1 | | | | |
| Abstention | 30 | | | | |
| Out-of-scope | 54 | | | | |
| **Total** | **90** | | | | |

**Acceptance rate:** _____ / 90 = _____% (≥ 90% required)

**Phase 1 status:** `COMPLETE` / `REQUIRES_ADJUSTMENT`