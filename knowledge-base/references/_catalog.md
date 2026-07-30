---
generated_by: roadmap-init
generated_on: 2026-07-29
slug: realtime-attendance-monitoring
peer_count_cloned: 5
peer_count_skipped: 2
---

# References catalog

State-of-the-art peer projects gathered at project inception by `/roadmap-init`.
This file is the contract `/discover-plan` reads when investigating a peer.

> **Lifecycle:** every peer below has lifecycle `cloned` (folder present under this directory) or `skipped` (rejected at license gate / de-scoped, kept here for the record).

---

## chatwoot

- **Folder:** `knowledge-base/references/chatwoot/`
- **Lifecycle:** cloned
- **Repo:** https://github.com/chatwoot/chatwoot
- **License:** `NOASSERTION` (GitHub could not classify — treat as unidentified)
- **License-gate decision:** clone-anyway-study-only
- **Last release / last commit:** 2026-07-29 (active)
- **Stars / forks at clone time:** 34878 / 8398

### Why this peer is here

The closest self-hosted, open-source platform *shape* to what TalkEx-monitoring becomes: an
omnichannel operations surface with an agent/supervisor inbox and real-time conversation handling.
It is the reference for how a live, multi-user conversation platform is structured end to end.

### What to study in it

- Real-time platform architecture: how live conversation state is pushed to many concurrent operators.
- Supervisor/agent inbox UX and prioritization.
- Multi-channel ingestion and the conversation/message/contact domain model.

### Supports ROADMAP milestone(s)

- M0 — *because:* the walking skeleton needs a minimal live conversation surface.
- M1 — *because:* multi-stream ingestion + persistence patterns.
- M4 — *because:* the supervisor live-monitoring UI is the anchor surface.

### Clone command used

```bash
git clone --depth 1 --filter=blob:none https://github.com/chatwoot/chatwoot knowledge-base/references/chatwoot/
```

---

## livekit-agents

- **Folder:** `knowledge-base/references/livekit-agents/`
- **Lifecycle:** cloned
- **Repo:** https://github.com/livekit/agents
- **License:** `Apache-2.0`
- **License-gate decision:** auto-approved-permissive
- **Last release / last commit:** 2026-07-29 (active)
- **Stars / forks at clone time:** 11559 / 3415

### Why this peer is here

A mature framework for real-time streaming pipelines and low-latency session orchestration. TalkEx
needs the streaming-ingest + live-session patterns; the voice/WebRTC specifics are Macaw's domain
and are deliberately ignored here.

### What to study in it

- Streaming ingestion and per-session state management under concurrency.
- Backpressure and real-time event routing.
- Pipeline orchestration for low-latency processing (NOT the ASR/voice parts).

### Supports ROADMAP milestone(s)

- M0 — *because:* the skeleton ingests a live stream end to end.
- M1 — *because:* multi-stream fan-in and backpressure.
- M3 — *because:* low-latency event routing for real-time alerts.

### Clone command used

```bash
git clone --depth 1 --filter=blob:none https://github.com/livekit/agents knowledge-base/references/livekit-agents/
```

---

## portuguese-bert

- **Folder:** `knowledge-base/references/portuguese-bert/`
- **Lifecycle:** cloned
- **Repo:** https://github.com/neuralmind-ai/portuguese-bert
- **License:** `NOASSERTION` (BERTimbau — verify the LICENSE file before reusing weights)
- **License-gate decision:** clone-anyway-study-only
- **Last release / last commit:** 2024-06-17 (stable; models are versioned artifacts)
- **Stars / forks at clone time:** 879 / 137

### Why this peer is here

BERTimbau is the SOTA pre-trained Brazilian-Portuguese BERT encoder — the natural base for the V1
sentiment and classification feature, complementing the multilingual embedding already in TalkEx.

### What to study in it

- Fine-tuning BERTimbau for PT-BR sentiment/classification.
- Tokenization and PT-BR-specific preprocessing choices.
- How its encoder relates to the existing `paraphrase-multilingual-MiniLM-L12-v2` embedding.

### Supports ROADMAP milestone(s)

- M2 — *because:* online sentiment/feature extraction is built on a PT-BR encoder.
- M7 — *because:* retraining the sentiment/intent models draws on this base.

### Clone command used

```bash
git clone --depth 1 --filter=blob:none https://github.com/neuralmind-ai/portuguese-bert knowledge-base/references/portuguese-bert/
```

---

## ai-powered-call-center-intelligence

- **Folder:** `knowledge-base/references/ai-powered-call-center-intelligence/`
- **Lifecycle:** cloned
- **Repo:** https://github.com/ReverendBayes/AI-Powered-Call-Center-Intelligence
- **License:** `MIT`
- **License-gate decision:** auto-approved-permissive
- **Last release / last commit:** 2025-11-07
- **Stars / forks at clone time:** 28 / 7

### Why this peer is here

Despite low adoption, this is the *closest product shape*: it transcribes calls, redacts PII,
extracts emotional tone, classifies issues, and renders insight-rich dashboards with a React+TS
frontend, running locally. Study the end-to-end shape — but note it uses GPT-3.5 in the online path,
which TalkEx explicitly rejects (LLMs offline only). Learn the shape, reject the online-LLM choice.

### What to study in it

- End-to-end architecture: transcript → sentiment/emotion → issue classification → dashboard.
- Dashboard UX for call-center intelligence.
- Issue-classification taxonomy (Billing, Connectivity, Retention, Inquiry, Cancel) as prior art.

### Supports ROADMAP milestone(s)

- M2 — *because:* sentiment/emotion extraction shape.
- M3 — *because:* issue classification driving alerts.
- M4 — *because:* the supervisor dashboard/UX.
- M6 — *because:* aggregated insight dashboards.

### Clone command used

```bash
git clone --depth 1 --filter=blob:none https://github.com/ReverendBayes/AI-Powered-Call-Center-Intelligence knowledge-base/references/ai-powered-call-center-intelligence/
```

---

## portuguese-nlp

- **Folder:** `knowledge-base/references/portuguese-nlp/`
- **Lifecycle:** cloned
- **Repo:** https://github.com/ajdavidl/Portuguese-NLP
- **License:** `NONE` (curated list / awesome-list — content is references, not reusable code)
- **License-gate decision:** clone-anyway-study-only
- **Last release / last commit:** 2026-06-25 (active)
- **Stars / forks at clone time:** 365 / 37

### Why this peer is here

A maintained index of the Brazilian-Portuguese NLP landscape — models, datasets, and tools. It is
the map for discovering PT-BR sentiment datasets and models needed by the ML milestones.

### What to study in it

- PT-BR sentiment datasets available for training/evaluation.
- PT-BR model options beyond BERTimbau.
- Tooling for PT-BR preprocessing/normalization.

### Supports ROADMAP milestone(s)

- M2 — *because:* it points to the PT-BR datasets/models the sentiment feature needs.
- M7 — *because:* retraining depends on discovering additional PT-BR labeled data.

### Clone command used

```bash
git clone --depth 1 --filter=blob:none https://github.com/ajdavidl/Portuguese-NLP knowledge-base/references/portuguese-nlp/
```

---

## Skipped peers (license gate / de-scoped)

> Peers identified during SOTA discovery but not cloned. Listed here so the decision is auditable and not repeated next time.

| Peer | Repo | License | Reason for skip |
|---|---|---|---|
| collabora/WhisperLive | https://github.com/collabora/WhisperLive | MIT | de-scoped: ASR is Macaw's domain (out of scope per grill Q4), not the platform's |
| alvaroarcelus/Sentiment-Analysis-Pipeline-for-Call-Center-Calls | https://github.com/alvaroarcelus/Sentiment-Analysis-Pipeline-for-Call-Center-Calls | NONE | de-scoped: educational pipeline, stale (2023), strictly inferior to ai-powered-call-center-intelligence |

---

## Cleanup protocol

- **Remove a peer:** delete its folder under this directory AND remove its entry from this catalog in the same commit.
- **Update a peer (refresh clone):** `cd knowledge-base/references/<peer>/ && git pull` — record the new commit SHA in this catalog.
- **Replace a peer with a better one:** treat as remove + add. Do NOT rename folders; symbolic continuity is meaningless when the underlying repo changed.
