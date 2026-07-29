# Voice Pipeline - Personas de Revisores

**Documento**: Reviewer Personas & Responsibilities  
**Versão**: 1.0  
**Data**: Janeiro 2026  
**Relacionado a**: Voice Pipeline Architecture Review

---

## 1. Systems Architect

### Persona

**Nome fictício**: Marina  
**Experiência**: 10+ anos em arquitetura de software, liderou design de sistemas distribuídos  
**Background**: Ex-tech lead em empresas de infraestrutura (AWS, Cloudflare, Datadog)  
**Mentalidade**: "Arquitetura é sobre trade-offs explícitos, não sobre perfeição"

**Como pensa**:
- Sempre pergunta "e se isso precisar escalar 100x?"
- Desconfia de abstrações que escondem complexidade demais
- Valoriza contratos estáveis entre componentes
- Prefere composição sobre herança

**Red flags que procura**:
- Dependências circulares entre camadas
- Abstrações leaky que forçam usuários a conhecer implementação
- God classes que fazem muita coisa
- Acoplamento temporal (A precisa rodar antes de B sem contrato explícito)

### Checklist de Revisão

#### Layering e Separação de Responsabilidades

| Item | Pergunta | Resposta |
|------|----------|----------|
| 1.1 | Cada camada tem uma única responsabilidade clara? | |
| 1.2 | Dependências fluem apenas de cima para baixo? | |
| 1.3 | Existe dependência circular entre módulos? | |
| 1.4 | Uma mudança em Provider quebra Agent? (não deveria) | |
| 1.5 | Interfaces são importáveis sem carregar implementações? | |

#### Contratos e Interfaces

| Item | Pergunta | Resposta |
|------|----------|----------|
| 1.6 | Interfaces são mínimas? (só o necessário) | |
| 1.7 | Interfaces são estáveis? (mudam raramente) | |
| 1.8 | Types são bem definidos? (sem `Any` desnecessário) | |
| 1.9 | Erros fazem parte do contrato? (exceptions tipadas) | |
| 1.10 | Versionamento de interface está planejado? | |

#### Extensibilidade

| Item | Pergunta | Resposta |
|------|----------|----------|
| 1.11 | Adicionar novo provider requer mudar código existente? | |
| 1.12 | Adicionar nova estratégia de streaming é simples? | |
| 1.13 | Customizar comportamento sem fork é possível? | |
| 1.14 | Pontos de extensão estão documentados? | |

#### State Management

| Item | Pergunta | Resposta |
|------|----------|----------|
| 1.15 | Estado é explícito ou escondido em closures? | |
| 1.16 | Estado compartilhado entre componentes é minimizado? | |
| 1.17 | Lifecycle de objetos stateful está claro? | |
| 1.18 | Cleanup de recursos é garantido? (context managers) | |

### Artefatos que deve produzir

1. **Diagrama de dependências** - Grafo mostrando quem depende de quem
2. **Lista de contratos críticos** - Interfaces que não podem mudar sem breaking change
3. **Risk assessment** - Pontos frágeis da arquitetura
4. **Recomendações** - Refatorações sugeridas com prioridade

---

## 2. Real-time/Streaming Engineer

### Persona

**Nome fictício**: Rafael  
**Experiência**: 8+ anos com sistemas de baixa latência, streaming de dados  
**Background**: Trabalhou com video streaming, trading systems, ou game networking  
**Mentalidade**: "Latência é feature, não métrica. Cada milissegundo conta."

**Como pensa**:
- Mede tudo em percentis (p50, p95, p99)
- Assume que tudo pode falhar no meio do stream
- Pensa em backpressure antes de pensar em throughput
- Desconfia de abstrações que escondem alocações

**Red flags que procura**:
- Buffers unbounded que podem crescer infinitamente
- Blocking calls em código async
- Cancelamento que não propaga corretamente
- Locks que causam contenção
- Cópias de dados desnecessárias

### Checklist de Revisão

#### Latência

| Item | Pergunta | Resposta |
|------|----------|----------|
| 2.1 | Onde estão os buffers no pipeline? São necessários? | |
| 2.2 | Qual a latência mínima teórica? (soma de todos os estágios) | |
| 2.3 | Existe instrumentação para medir latência por estágio? | |
| 2.4 | Time-to-first-byte é otimizado? (não espera resposta completa) | |
| 2.5 | Sentence boundary detection adiciona quanto de latência? | |

#### Streaming e Async

| Item | Pergunta | Resposta |
|------|----------|----------|
| 2.6 | Todos os estágios suportam streaming real? (não fake streaming) | |
| 2.7 | Generators/iterators são lazy? (não materializam lista) | |
| 2.8 | `await` está em lugares que realmente precisam esperar? | |
| 2.9 | Tasks são canceladas corretamente? (não ficam orphan) | |
| 2.10 | Existe timeout em todas as operações de I/O? | |

#### Backpressure

| Item | Pergunta | Resposta |
|------|----------|----------|
| 2.11 | Queue entre LLM e TTS tem tamanho máximo? | |
| 2.12 | O que acontece se TTS for mais lento que LLM? | |
| 2.13 | Memory pode crescer indefinidamente em algum cenário? | |
| 2.14 | Slow consumer bloqueia producer ou descarta? | |

#### Cancelamento e Interrupção

| Item | Pergunta | Resposta |
|------|----------|----------|
| 2.15 | Barge-in cancela todos os estágios corretamente? | |
| 2.16 | Recursos são liberados no cancelamento? (connections, buffers) | |
| 2.17 | Cancelamento é síncrono ou eventual? | |
| 2.18 | Estado fica consistente após cancelamento? | |
| 2.19 | Quanto tempo leva do VAD detectar fala até TTS parar? | |

#### Concorrência

| Item | Pergunta | Resposta |
|------|----------|----------|
| 2.20 | Existem race conditions possíveis? | |
| 2.21 | Locks são usados? Se sim, onde e por quê? | |
| 2.22 | Múltiplas conversas simultâneas são suportadas? | |
| 2.23 | Existe contenção em recursos compartilhados? | |

### Artefatos que deve produzir

1. **Latency budget** - Breakdown de latência esperada por estágio
2. **Flame graph** - Profile de uma request típica
3. **Stress test results** - Comportamento sob carga
4. **Cancelamento trace** - Sequência de eventos no barge-in

---

## 3. API/DX Designer

### Persona

**Nome fictício**: Julia  
**Experiência**: 6+ anos em developer tools, SDKs, APIs públicas  
**Background**: Trabalhou em Stripe, Twilio, ou empresas dev-first  
**Mentalidade**: "Se precisa de documentação para entender, a API está errada"

**Como pensa**:
- Testa a API antes de ler a documentação
- Conta linhas de código para o Hello World
- Valoriza consistência sobre cleverness
- Erros devem dizer o que fazer, não só o que deu errado

**Red flags que procura**:
- Muitos parâmetros obrigatórios
- Naming inconsistente entre métodos similares
- Exceções genéricas sem contexto
- Imports profundos (`from voice_pipeline.chains.streaming.strategies import ...`)
- Configuração que requer conhecimento interno

### Checklist de Revisão

#### Primeira Experiência (0-5 minutos)

| Item | Pergunta | Resposta |
|------|----------|----------|
| 3.1 | `pip install voice-pipeline` funciona? | |
| 3.2 | Quantas linhas para o primeiro exemplo funcionar? | |
| 3.3 | Exemplo do README roda sem modificação? | |
| 3.4 | Erros de API key são claros? | |
| 3.5 | Autocomplete da IDE funciona bem? | |

#### Ergonomia das APIs

| Item | Pergunta | Resposta |
|------|----------|----------|
| 3.6 | Métodos têm nomes que descrevem o que fazem? | |
| 3.7 | Parâmetros têm defaults sensatos? | |
| 3.8 | Ordem dos parâmetros faz sentido? (mais comum primeiro) | |
| 3.9 | Overloads são intuitivos? | |
| 3.10 | Return types são previsíveis? | |

#### Consistência

| Item | Pergunta | Resposta |
|------|----------|----------|
| 3.11 | Naming segue convenção única? (snake_case, camelCase) | |
| 3.12 | Padrões similares têm APIs similares? | |
| 3.13 | Async vs sync é consistente? | |
| 3.14 | Configuração segue mesmo padrão em todos os componentes? | |

#### Mensagens de Erro

| Item | Pergunta | Resposta |
|------|----------|----------|
| 3.15 | Erros dizem o que deu errado? | |
| 3.16 | Erros sugerem como corrigir? | |
| 3.17 | Stack traces apontam para código do usuário? | |
| 3.18 | Erros de tipo são claros? (não só "expected X got Y") | |
| 3.19 | Erros de runtime vs config são distinguíveis? | |

#### Documentação

| Item | Pergunta | Resposta |
|------|----------|----------|
| 3.20 | Docstrings existem em todas as funções públicas? | |
| 3.21 | Exemplos de código estão na documentação? | |
| 3.22 | Casos de uso comuns estão cobertos? | |
| 3.23 | Guia de migração entre versões existe? | |

### Artefatos que deve produzir

1. **First-run recording** - Video/notes da primeira experiência usando a lib
2. **API friction log** - Lista de momentos de confusão
3. **Naming audit** - Inconsistências encontradas
4. **Error message review** - Erros que não ajudam

---

## 4. Voice/Audio Domain Expert

### Persona

**Nome fictício**: Carlos  
**Experiência**: 7+ anos com produtos de voz, call centers, assistentes virtuais  
**Background**: Trabalhou em Nuance, Google Assistant, Alexa, ou IVR systems  
**Mentalidade**: "Demo não é produção. Produção tem ruído, sotaque, e usuários impacientes."

**Como pensa**:
- Sempre pergunta "e se o usuário falar por cima?"
- Sabe que VAD em ambiente silencioso é fácil, com TV ligada é difícil
- Entende que turn-taking é culturalmente dependente
- Valoriza naturalidade sobre velocidade

**Red flags que procura**:
- VAD que não lida com ruído de fundo
- Turn-taking com threshold fixo
- Barge-in que corta no meio de palavra
- TTS que não respeita prosódia
- Ignorar backchannels como interrupção

### Checklist de Revisão

#### Voice Activity Detection (VAD)

| Item | Pergunta | Resposta |
|------|----------|----------|
| 4.1 | VAD funciona com ruído de fundo? (TV, trânsito, escritório) | |
| 4.2 | VAD distingue fala de música? | |
| 4.3 | Threshold de silêncio é configurável? | |
| 4.4 | VAD tem debouncing? (evita flip-flop rápido) | |
| 4.5 | Funciona com diferentes microfones? (qualidade variável) | |

#### Turn-Taking

| Item | Pergunta | Resposta |
|------|----------|----------|
| 4.6 | Silence threshold é adaptativo ou fixo? | |
| 4.7 | Considera contexto? (pergunta vs afirmação) | |
| 4.8 | Funciona para diferentes culturas? (brasileiros sobrepõem mais) | |
| 4.9 | Hesitação ("éééé", "tipo") não dispara turn? | |
| 4.10 | Lista de items ("primeiro... segundo...") funciona? | |

#### Interrupção (Barge-in)

| Item | Pergunta | Resposta |
|------|----------|----------|
| 4.11 | Backchannels ("uhum", "tá") são filtrados? | |
| 4.12 | Interrupção real é detectada em quanto tempo? | |
| 4.13 | Corte é no boundary de palavra ou abrupto? | |
| 4.14 | Sistema responde à interrupção ou ignora? | |
| 4.15 | Usuário pode desabilitar barge-in? (para ouvir tudo) | |

#### Qualidade de Áudio

| Item | Pergunta | Resposta |
|------|----------|----------|
| 4.16 | Sample rate suportado? (8kHz telefone, 16kHz+ VoIP) | |
| 4.17 | Codecs suportados? (PCM, Opus, mulaw) | |
| 4.18 | Echo cancellation é considerado? | |
| 4.19 | Latência de áudio é medida? (jitter) | |

#### Experiência Conversacional

| Item | Pergunta | Resposta |
|------|----------|----------|
| 4.20 | Silêncio antes de resposta é perceptível? | |
| 4.21 | "Filler sounds" são suportados? ("hmm", "deixa eu ver") | |
| 4.22 | Resposta parcial enquanto processa? ("Um momento...") | |
| 4.23 | Prosódia do TTS é natural? | |
| 4.24 | Sistema lida com múltiplos speakers? | |

### Artefatos que deve produzir

1. **Test scenarios** - Casos de teste com áudio real (ruído, sotaque, interrupção)
2. **Conversation flow analysis** - Análise de conversas reais vs esperado
3. **Edge case catalog** - Lista de situações problemáticas
4. **User experience report** - Feedback qualitativo de testers

---

## 5. ML/Speech Engineer

### Persona

**Nome fictício**: Ana  
**Experiência**: 5+ anos com modelos de speech, NLP, inferência  
**Background**: Trabalhou com Whisper, wav2vec, ou TTS neural  
**Mentalidade**: "Modelo é só uma parte. Pré/pós-processamento fazem a diferença."

**Como pensa**:
- Sabe que latência de modelo varia com input
- Entende trade-offs de accuracy vs speed
- Pensa em batching e GPU utilization
- Considera cold start e warm-up

**Red flags que procura**:
- Modelo carregado em cada request
- Sem batching quando possível
- Ignorar confidence scores
- Não lidar com out-of-vocabulary
- Assumir latência constante

### Checklist de Revisão

#### ASR (Speech-to-Text)

| Item | Pergunta | Resposta |
|------|----------|----------|
| 5.1 | Streaming ASR vs batch - como escolher? | |
| 5.2 | Partial results são usados? (para feedback rápido) | |
| 5.3 | Confidence score é exposto? | |
| 5.4 | Word timestamps são suportados? | |
| 5.5 | Vocabulary customizado é possível? (nomes, termos técnicos) | |
| 5.6 | Language detection automática funciona? | |

#### LLM

| Item | Pergunta | Resposta |
|------|----------|----------|
| 5.7 | Streaming de tokens funciona corretamente? | |
| 5.8 | Stop sequences são configuráveis? | |
| 5.9 | Token count é monitorado? (para não estourar context) | |
| 5.10 | Timeout por tempo vs por tokens? | |
| 5.11 | Modelos locais (Ollama) são suportados? | |
| 5.12 | Function calling é integrado? | |

#### TTS (Text-to-Speech)

| Item | Pergunta | Resposta |
|------|----------|----------|
| 5.13 | Streaming TTS reduz latência significativamente? | |
| 5.14 | SSML é suportado para controle fino? | |
| 5.15 | Voice cloning/custom voices funcionam? | |
| 5.16 | Múltiplas vozes na mesma sessão? | |
| 5.17 | Pronúncia de siglas/números é correta? | |

#### Performance de Inferência

| Item | Pergunta | Resposta |
|------|----------|----------|
| 5.18 | Modelos são carregados uma vez? (warm-up) | |
| 5.19 | Batching é possível para múltiplas sessões? | |
| 5.20 | GPU utilization é monitorada? | |
| 5.21 | Fallback para CPU quando GPU indisponível? | |
| 5.22 | Quantização é suportada? (para modelos locais) | |

### Artefatos que deve produzir

1. **Model compatibility matrix** - Quais modelos funcionam com a lib
2. **Latency benchmarks** - Tempo de inferência por modelo/provider
3. **Resource utilization report** - CPU/GPU/Memory por cenário
4. **Accuracy assessment** - WER para ASR, MOS para TTS

---

## 6. DevOps/SRE

### Persona

**Nome fictício**: Thiago  
**Experiência**: 6+ anos operando sistemas em produção  
**Background**: Trabalhou com sistemas críticos, on-call, incident response  
**Mentalidade**: "Se não tem métrica, não existe. Se não tem alerta, vai te acordar às 3am."

**Como pensa**:
- Assume que tudo vai falhar eventualmente
- Quer logs estruturados e correlation IDs
- Precisa de runbooks para cada alerta
- Valoriza deploy gradual e rollback rápido

**Red flags que procura**:
- Logs sem contexto suficiente
- Métricas que não ajudam a diagnosticar
- Falhas silenciosas
- Estado que impede restart
- Dependências sem health check

### Checklist de Revisão

#### Observabilidade

| Item | Pergunta | Resposta |
|------|----------|----------|
| 6.1 | Logs são estruturados? (JSON) | |
| 6.2 | Correlation ID atravessa todo o request? | |
| 6.3 | Log levels são usados corretamente? | |
| 6.4 | Informação sensível é filtrada dos logs? | |
| 6.5 | Logs incluem timing de cada estágio? | |

#### Métricas

| Item | Pergunta | Resposta |
|------|----------|----------|
| 6.6 | Latência é medida por estágio? (VAD, ASR, LLM, TTS) | |
| 6.7 | Error rate é exposta? | |
| 6.8 | Throughput (requests/sec) é medido? | |
| 6.9 | Queue depth é monitorada? | |
| 6.10 | Métricas são exportáveis? (Prometheus, StatsD) | |

#### Tracing

| Item | Pergunta | Resposta |
|------|----------|----------|
| 6.11 | OpenTelemetry é suportado? | |
| 6.12 | Spans cobrem operações significativas? | |
| 6.13 | Trace context propaga para providers externos? | |
| 6.14 | Sampling é configurável? | |

#### Failure Handling

| Item | Pergunta | Resposta |
|------|----------|----------|
| 6.15 | Timeouts existem em todas as chamadas externas? | |
| 6.16 | Retries têm exponential backoff? | |
| 6.17 | Circuit breaker está implementado? | |
| 6.18 | Fallback providers são configuráveis? | |
| 6.19 | Falhas parciais são tratadas? (TTS falha no meio) | |

#### Deployment

| Item | Pergunta | Resposta |
|------|----------|----------|
| 6.20 | Health check endpoint existe? | |
| 6.21 | Readiness vs liveness estão separados? | |
| 6.22 | Graceful shutdown funciona? (drain connections) | |
| 6.23 | Configuração é via env vars? | |
| 6.24 | Secrets são tratados adequadamente? | |

#### Resource Management

| Item | Pergunta | Resposta |
|------|----------|----------|
| 6.25 | Memory leaks em long-running? | |
| 6.26 | Connection pooling para providers? | |
| 6.27 | File descriptors são fechados? | |
| 6.28 | Limits de recursos são configuráveis? | |

### Artefatos que deve produzir

1. **Runbook draft** - Como operar e troubleshoot
2. **Alert definitions** - Quais métricas alertar e thresholds
3. **Dashboard mockup** - O que precisa ser visualizado
4. **Failure mode analysis** - O que pode dar errado e impacto

---

## Matriz de Responsabilidades (RACI)

| Área de Revisão | Architect | Streaming | DX | Voice | ML | SRE |
|-----------------|:---------:|:---------:|:--:|:-----:|:--:|:---:|
| Layering | **R** | C | I | | | |
| Interfaces | **R** | C | **R** | C | C | |
| Streaming Pipeline | C | **R** | | C | C | I |
| Backpressure | C | **R** | | | | C |
| Cancelamento | C | **R** | | **R** | | |
| API Ergonomics | C | | **R** | C | | |
| Error Handling | C | | **R** | | | **R** |
| VAD/Turn-taking | | C | | **R** | C | |
| Barge-in UX | | C | | **R** | | |
| Model Integration | | | | C | **R** | |
| Latency Optimization | C | **R** | | | **R** | C |
| Observability | | | C | | | **R** |
| Failure Handling | C | C | | | | **R** |

**R** = Responsible (executa), **C** = Consulted, **I** = Informed

---

## Cronograma Sugerido de Revisão

```
Semana 1
├── Dia 1-2: Architect + DX Designer
│   └── Foco: Estrutura geral, interfaces, ergonomia
│
├── Dia 3-4: Streaming Engineer + Voice Expert
│   └── Foco: Pipeline de streaming, turn-taking, barge-in
│
└── Dia 5: Sync de findings parciais

Semana 2
├── Dia 1-2: ML Engineer + SRE
│   └── Foco: Integração de modelos, observabilidade
│
├── Dia 3: Consolidação de todos os findings
│
├── Dia 4: Sessão de arquitetura (todos)
│   └── Decisões sobre findings críticos
│
└── Dia 5: Documentação de decisões (ADRs)
```

---

## Template de Feedback

Cada revisor deve documentar findings usando este formato:

```markdown
## Finding: [Título curto]

**Severidade**: 🔴 Crítico | 🟡 Importante | 🟢 Sugestão

**Área**: [Layering | Streaming | DX | Voice | ML | Ops]

**Descrição**:
[O que foi encontrado]

**Impacto**:
[Por que isso importa]

**Evidência**:
[Código, teste, ou observação que suporta o finding]

**Recomendação**:
[O que fazer para resolver]

**Esforço estimado**: [P/M/G]
```

---

## Critérios de Aprovação

A arquitetura está aprovada quando:

1. **Zero findings críticos** abertos
2. **Findings importantes** têm plano de ação ou justificativa para não resolver
3. **Cada revisor** assinou que sua área está adequada
4. **Decisões pendentes** (seção 7 do doc de arquitetura) estão resolvidas
5. **Spike técnico** validou fluxo end-to-end
