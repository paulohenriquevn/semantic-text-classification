# Roadmap SOTA — TalkEx Conversation Intelligence Engine

> **Objetivo:** Elevar o TalkEx de Macro-F1=0,659 para ≥0,850 em classificação de intents e MRR ≥0,900 em retrieval híbrido, mantendo a proposta CPU-friendly e auditável.
>
> **Baseline atual:** 2026-03-12
> **Documento para:** Engenheiros NLP, Data Engineers, QA

---

## Baseline de Referência (Ponto de Partida)

Todos os experimentos A/B devem comparar contra este baseline:

| Métrica | Valor | Configuração |
|---|---|---|
| **Macro-F1 (classificação)** | 0,659 | lexical+emb LightGBM, 100t/31l, window 5t/stride 2 |
| **Accuracy** | 0,722 | Mesmo |
| **MRR (retrieval)** | 0,826 | Hybrid-RRF (k=60) |
| **nDCG@10** | 0,613 | Hybrid-RRF |
| **Embedding model** | paraphrase-multilingual-MiniLM-L12-v2 | 384 dims, L2 norm |
| **Corpus** | 2.257 conversas (sintéticas) | 9 classes, PT-BR |
| **Windows** | 6.583 train / 1.429 val / 1.368 test | window=5, stride=2 |
| **Treinamento** | ~12s (CPU) | Sem GPU |
| **Inferência** | 0,08 ms/janela | CPU |

### Per-Class F1 Baseline (classes ordenadas por dificuldade)

| Classe | F1 | N (teste) | Diagnóstico |
|---|---|---|---|
| cancelamento | 0,968 | 32 | Quase perfeita — vocabulário único |
| suporte_tecnico | 0,833 | 46 | Bom — termos técnicos discriminativos |
| duvida_servico | 0,796 | 50 | Bom — semântica clara |
| elogio | 0,778 | 22 | Precision=1,0 mas recall=0,636 |
| reclamacao | 0,722 | 55 | Recall alto (0,945) mas precision baixa (0,584) |
| duvida_produto | 0,694 | 51 | Confusão com duvida_servico |
| saudacao | 0,541 | 26 | Precision=0,909 mas recall=0,385 |
| compra | 0,500 | 36 | Confusão com duvida_produto |
| outros | 0,095 | 20 | **Colapsada** — classe residual sem padrão |

### Contribuição dos Componentes (Ablação)

| Componente | Δ Macro-F1 | Prioridade para melhoria |
|---|---|---|
| Embeddings | +25,8pp | **MÁXIMA** — alavanca dominante |
| Lexical | +2,0pp | Moderada — complemento barato |
| Structural | +1,3pp | Baixa |
| Rules | −0,5pp | Requer redesign (semântico) |

---

## Arquitetura de Experimentação A/B

### Protocolo Obrigatório

Todo experimento DEVE seguir este protocolo para ser considerado válido:

```
1. DEFINIR hipótese: "Técnica X melhora métrica Y em ≥Z pp"
2. MANTER baseline inalterado (mesmo split, mesmo seed, mesma avaliação)
3. VARIAR apenas UMA dimensão por experimento
4. EXECUTAR com 5 seeds (42, 123, 456, 789, 0)
5. REPORTAR: média ± std, Wilcoxon signed-rank, Bootstrap CI 95%
6. REGISTRAR em experiments/results/<track>/<experiment_id>/
```

### Estrutura de Resultados

```
experiments/results/
├── baseline/                    # Baseline congelado (não editar)
│   ├── results.json
│   └── per_class.json
├── track1_embeddings/           # Experimentos de embedding
│   ├── exp001_setfit/
│   │   ├── config.json          # Hiperparâmetros exatos
│   │   ├── results.json         # Métricas agregadas
│   │   ├── per_seed_results.json
│   │   ├── per_class.json       # F1 por classe
│   │   ├── statistical_tests.json
│   │   └── NOTES.md             # Observações do engenheiro
│   ├── exp002_contrastive/
│   └── ...
├── track2_data/
├── track3_classification/
├── track4_retrieval/
├── track5_rules/
└── track6_infrastructure/
```

### Critérios de Promoção (Gate)

Um experimento só é promovido ao pipeline principal se:

| Critério | Threshold |
|---|---|
| Δ Macro-F1 vs baseline | ≥ +2,0pp (significativo a α=0,05) |
| Nenhuma classe regride > 3pp | Obrigatório |
| Wilcoxon p-value | < 0,05 |
| Bootstrap CI 95% | Exclui zero |
| Latência de inferência | ≤ 2× baseline (0,16 ms/janela) |
| Tempo de treinamento | ≤ 10× baseline (120s) em CPU |

---

## Track 1 — Embedding Fine-Tuning (Impacto Esperado: +10–20pp)

> **Racional:** Embeddings contribuem +25,8pp (ablação). O modelo atual é genérico (paraphrase-multilingual-MiniLM-L12-v2). Fine-tuning no domínio deve ser a alavanca de maior impacto.

### 1.1 Trocar Base Model (Quick Win)

**O que:** Avaliar modelos maiores/melhores sem fine-tuning.

**Modelos candidatos:**

| Modelo | Dims | Línguas | Tamanho | Por que testar |
|---|---|---|---|---|
| `intfloat/multilingual-e5-large` | 1024 | 100+ | 560M | SOTA em MTEB multilingual |
| `BAAI/bge-m3` | 1024 | 100+ | 568M | Hybrid dense+sparse+ColBERT |
| `intfloat/multilingual-e5-base` | 768 | 100+ | 278M | Trade-off tamanho vs qualidade |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | 50+ | 278M | Upgrade direto do MiniLM |

**Implementação:**
```python
# Em experiments/scripts/run_experiment.py, parametrizar:
EMBEDDING_MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", 384),   # baseline
    ("intfloat/multilingual-e5-large", 1024),
    ("BAAI/bge-m3", 1024),
    ("intfloat/multilingual-e5-base", 768),
    ("paraphrase-multilingual-mpnet-base-v2", 768),
]

# Para cada modelo, re-executar H2 completo (6 variantes × 5 seeds)
```

**Protocolo de teste:**
- Manter features lexicais + estruturais idênticas
- Substituir apenas o vetor de embedding
- Medir: Macro-F1, per-class F1, tempo de encoding, memória

**Acceptance criteria:**
- ≥ +3pp Macro-F1 com modelo maior
- Encoding time ≤ 5s por batch de 64 janelas em CPU
- Se modelo ≥ 1024 dims: verificar se LightGBM escala (pode precisar de mais árvores)

**Risco:** Modelos maiores podem ser lentos demais em CPU. Mitigação: testar com `ONNX Runtime` para inferência otimizada.

---

### 1.2 SetFit (Few-Shot Contrastive Fine-Tuning)

**O que:** Fine-tuning contrastivo leve que adapta embeddings ao domínio com poucos exemplos. SetFit treina em minutos em CPU.

**Por que:** SetFit (Tunstall et al., 2022) demonstrou performance competitiva com fine-tuning completo usando apenas 8-64 exemplos por classe. Com 2.257 conversas, temos dados de sobra.

**Implementação:**

```python
# Nova dependência: pip install setfit
from setfit import SetFitModel, SetFitTrainer, TrainingArguments

# 1. Preparar dados no formato SetFit
train_examples = []
for conv_id, label in train_labels.items():
    # Usar texto renderizado da janela central (ou concatenação)
    text = render_conversation_text(conv_id)
    train_examples.append({"text": text, "label": label})

# 2. Treinar
model = SetFitModel.from_pretrained(
    "paraphrase-multilingual-MiniLM-L12-v2",
    labels=INTENT_LABELS,
)
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    num_iterations=20,        # Pares contrastivos por época
    num_epochs=1,             # Épocas de fine-tuning
    batch_size=16,
)
trainer.train()

# 3. Extrair embeddings fine-tuned para usar com LightGBM
embeddings = model.model_body.encode(texts, normalize_embeddings=True)
```

**Variantes a testar:**

| Experimento | Descrição | Hipótese |
|---|---|---|
| SetFit-8shot | 8 exemplos por classe | Baseline few-shot |
| SetFit-32shot | 32 exemplos por classe | Melhoria com mais dados |
| SetFit-full | Todos os exemplos de treino | Ceiling de performance |
| SetFit→LightGBM | Embeddings SetFit como features para LightGBM | Combina fine-tuning com ensemble |

**Acceptance criteria:**
- SetFit-full: ≥ +5pp Macro-F1 vs baseline
- SetFit→LightGBM: ≥ +8pp (combina representação + classificador)
- Treinamento: ≤ 30 minutos em CPU

---

### 1.3 Contrastive Learning Manual (Sentence-Transformers)

**O que:** Fine-tuning do embedding model com pares/triplets do domínio usando a API nativa de sentence-transformers.

**Por que:** Mais controle que SetFit. Permite definir hard negatives específicos do domínio.

**Implementação:**

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# 1. Gerar pares contrastivos
train_pairs = []
for i, (text_a, label_a) in enumerate(train_data):
    # Positivo: mesma classe
    positive = random.choice([t for t, l in train_data if l == label_a and t != text_a])
    train_pairs.append(InputExample(texts=[text_a, positive], label=1.0))

    # Hard negative: classe mais confusa (baseado em confusion matrix)
    confused_class = CONFUSION_MAP[label_a]  # e.g., compra → duvida_produto
    hard_neg = random.choice([t for t, l in train_data if l == confused_class])
    train_pairs.append(InputExample(texts=[text_a, hard_neg], label=0.0))

# 2. Treinar com CosineSimilarityLoss
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="models/finetuned-miniml-talkex",
)
```

**Hard negatives prioritários (baseados na confusion matrix atual):**

| Par confuso | Similaridade | Prioridade |
|---|---|---|
| compra ↔ duvida_produto | Alta | P0 — maior confusão |
| duvida_produto ↔ duvida_servico | Alta | P0 |
| reclamacao ↔ suporte_tecnico | Moderada | P1 |
| saudacao ↔ outros | Alta | P1 |
| elogio ↔ saudacao | Moderada | P2 |

**Acceptance criteria:**
- ≥ +8pp Macro-F1 global
- ≥ +15pp F1 em "compra" (classe mais confusa: F1=0,500)
- ≥ +10pp F1 em "saudacao" (F1=0,541)

---

### 1.4 ONNX Runtime para Inferência Otimizada

**O que:** Converter modelos sentence-transformers para ONNX para inferência 2-4× mais rápida em CPU.

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model paraphrase-multilingual-MiniLM-L12-v2 ./onnx_model/
```

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
model = ORTModelForFeatureExtraction.from_pretrained("./onnx_model/")
```

**Acceptance criteria:**
- Encoding ≥ 2× mais rápido que PyTorch
- Diferença de embedding ≤ 1e-5 (numericamente equivalente)

---

## Track 2 — Qualidade de Dados (Impacto Esperado: +5–15pp)

> **Racional:** O corpus é 100% sintético (gerado por LLM). Dados reais e estratégias de augmentation podem melhorar a generalização significativamente.

### 2.1 Análise de Erros Sistemática

**O que:** Antes de mudar qualquer coisa, entender ONDE e POR QUE o modelo erra.

**Implementação:**

```python
# Gerar confusion matrix detalhada
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Para cada erro:
errors = []
for conv_id, pred, true in zip(test_ids, predictions, ground_truth):
    if pred != true:
        errors.append({
            "conv_id": conv_id,
            "predicted": pred,
            "true_label": true,
            "confidence": max_prob,
            "text_snippet": first_100_chars,
            "n_windows": n_windows_for_conv,
            "window_agreement": fraction_windows_agreeing,
        })

error_df = pd.DataFrame(errors)

# Análises obrigatórias:
# 1. Confusion matrix normalizada (quais classes confundem?)
# 2. Distribuição de confiança nos erros (modelo erra com alta confiança?)
# 3. Correlação entre n_windows e acerto (conversas curtas erram mais?)
# 4. Window agreement (todas as janelas concordam no erro?)
# 5. Texto dos erros (padrões qualitativos)
```

**Entregas:**
- `experiments/analysis/error_analysis.json` — erros anotados
- `experiments/analysis/confusion_matrix.png` — heatmap
- `experiments/analysis/confidence_calibration.png` — reliability diagram
- `experiments/analysis/FINDINGS.md` — achados qualitativos

---

### 2.2 Tratamento da Classe "outros"

**O que:** A classe "outros" (F1=0,095) é residual e prejudica o Macro-F1. Estratégias:

| Estratégia | Descrição | Quando usar |
|---|---|---|
| **Remoção** | Excluir "outros" e avaliar com 8 classes | Se "outros" não tem valor de negócio |
| **Threshold-based rejection** | Classificar como "outros" quando max_prob < threshold | Se "outros" = "incerto" |
| **Agrupamento** | Mesclar com classe mais próxima | Se análise qualitativa justificar |
| **Coleta direcionada** | Gerar mais exemplos de "outros" com variância | Se "outros" tem valor de negócio |

**Experimento:**
```python
# Testar rejection threshold:
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds_with_rejection = []
    for prob_vector in all_probabilities:
        if max(prob_vector) < threshold:
            preds_with_rejection.append("outros")
        else:
            preds_with_rejection.append(class_names[argmax(prob_vector)])
```

**Acceptance criteria:**
- Com remoção de "outros": Macro-F1 (8 classes) ≥ 0,750
- Com rejection: F1 de "outros" ≥ 0,300 sem degradar outras classes > 2pp

---

### 2.3 Aumento de Dados Direcionado

**O que:** Expandir classes fracas com dados sintéticos mais variados.

**Estratégia por classe:**

| Classe | F1 atual | Ação | Meta |
|---|---|---|---|
| compra | 0,500 | Gerar conversas com vocabulário diverso de compra | +200 conversas |
| saudacao | 0,541 | Gerar saudações variadas (formais, informais, regionais) | +100 conversas |
| outros | 0,095 | Gerar conversas genuinamente ambíguas | +150 conversas |

**Protocolo de geração:**
1. Analisar erros da classe → identificar padrões faltantes
2. Gerar com LLM usando prompts específicos para variância
3. Validar manualmente 10% da amostra
4. Re-treinar e comparar com baseline

---

### 2.4 Label Noise Reduction (Window-Level)

**O que:** Mitigar o ruído da supervisão fraca (windows herdam label da conversa).

**Estratégias:**

```python
# Estratégia 1: Confidence-weighted training
# Janelas com alta confiança do classificador anterior recebem peso maior
sample_weights = []
for window in train_windows:
    # Usar predição do baseline como proxy de qualidade do label
    baseline_conf = baseline_model.predict_proba([window_features])[0].max()
    sample_weights.append(baseline_conf)

lgbm.fit(X_train, y_train, sample_weight=sample_weights)

# Estratégia 2: Multi-instance learning (MIL)
# Tratar cada conversa como "bag" de janelas
# Ao menos 1 janela deve ser positiva para o label
# Usar attention-based MIL pooling

# Estratégia 3: Label smoothing
# Em vez de one-hot, usar: (1 - ε) para label correto, ε/(K-1) para outros
LABEL_SMOOTHING = 0.1
```

**Acceptance criteria:**
- ≥ +2pp Macro-F1 com qualquer estratégia de noise reduction
- Classe "outros" não piora

---

## Track 3 — Arquitetura de Classificação (Impacto Esperado: +3–8pp)

> **Racional:** LightGBM com configuração default. Otimização de hiperparâmetros, calibração e agregação podem extrair mais performance da mesma representação.

### 3.1 Hyperparameter Optimization com Optuna

**O que:** Otimização automática de hiperparâmetros do LightGBM.

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = LGBMClassifier(**params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = aggregate_window_to_conversation(model.predict_proba(X_val))
    return macro_f1(y_val_conv, y_pred)

study = optuna.create_study(direction="maximize", n_trials=200)
study.optimize(objective)
```

**Acceptance criteria:**
- ≥ +2pp Macro-F1 vs default (100t/31l)
- Validar em test set (não otimizar no test!)
- Reportar: melhores hiperparâmetros, importância dos parâmetros

---

### 3.2 Probability Calibration

**O que:** LightGBM probabilities não são calibradas. Calibração melhora tanto a confiança quanto o threshold-based routing.

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling (sigmoid) ou Isotonic regression
calibrated_lgbm = CalibratedClassifierCV(
    lgbm_model,
    method="isotonic",  # ou "sigmoid"
    cv="prefit",        # usar validação set
)
calibrated_lgbm.fit(X_val, y_val)

# Medir calibração:
from sklearn.calibration import calibration_curve
for cls in range(9):
    prob_true, prob_pred = calibration_curve(
        y_test == cls, probs[:, cls], n_bins=10
    )
    # Plotar reliability diagram
```

**Acceptance criteria:**
- ECE (Expected Calibration Error) ≤ 0,05
- Melhoria no reliability diagram vs baseline não calibrado
- Macro-F1 não degrada (calibração afeta ranking, não classificação diretamente)

**Impacto em H4 (cascata):** Probabilidades calibradas melhoram o roteamento por confiança — thresholds se tornam mais significativos.

---

### 3.3 Window Aggregation Aprimorada

**O que:** Atualmente usa média simples de probabilidades. Estratégias melhores:

```python
# Estratégia 1: Weighted average (recency bias)
# Janelas finais da conversa são mais informativas
weights = np.array([0.5, 0.7, 0.9, 1.0])  # Crescente para janelas finais
weighted_probs = np.average(window_probs, axis=0, weights=weights[:len(window_probs)])

# Estratégia 2: Max-pooling por classe
# Para cada classe, usar a probabilidade máxima entre janelas
max_probs = np.max(window_probs, axis=0)

# Estratégia 3: Confidence-weighted
# Janelas onde o modelo tem alta confiança pesam mais
confidences = np.max(window_probs, axis=1)  # max prob por janela
weights = confidences ** 2  # quadrado amplifica janelas confiantes
weighted_probs = np.average(window_probs, axis=0, weights=weights)

# Estratégia 4: Majority voting (discreto)
window_preds = np.argmax(window_probs, axis=1)
final_pred = scipy.stats.mode(window_preds).mode

# Estratégia 5: Trainable attention
# MLP pequeno que aprende pesos de atenção por janela
# Input: [prob_vector, window_position, confidence]
# Output: weight para cada janela
```

**Variantes a testar:**

| Estratégia | Complexidade | Hipótese |
|---|---|---|
| Mean (baseline) | Nenhuma | Referência |
| Recency-weighted | Baixa | Janelas finais mais informativas |
| Max-pooling | Baixa | Basta uma janela forte |
| Confidence-weighted | Baixa | Janelas confiantes dominam |
| Majority voting | Baixa | Robustez a outliers |
| Learned attention | Média | Optimal combination |

**Acceptance criteria:**
- ≥ +2pp Macro-F1 com qualquer estratégia vs mean
- Classe "outros" não piora (agregação afeta especialmente classes difusas)

---

### 3.4 Cross-Validation para Estimativa Robusta

**O que:** Atual: single split (70/15/15). CV dá estimativas mais confiáveis.

```python
from sklearn.model_selection import StratifiedKFold

# K-Fold estratificado no nível da CONVERSA (não janela!)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for fold, (train_idx, test_idx) in enumerate(cv.split(conversations, labels)):
    # Gerar janelas DENTRO de cada fold
    train_windows = generate_windows(conversations[train_idx])
    test_windows = generate_windows(conversations[test_idx])

    # Treinar e avaliar
    model.fit(train_windows_features, train_windows_labels)
    fold_score = evaluate_at_conversation_level(model, test_windows, conversations[test_idx])
    cv_scores.append(fold_score)

print(f"CV Macro-F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

**IMPORTANTE:** O split para CV deve ser no nível da conversa, NUNCA no nível da janela (evitar leakage).

---

### 3.5 Ensemble de Classificadores

**O que:** Combinar LightGBM + LogReg + MLP via stacking ou voting.

```python
from sklearn.ensemble import StackingClassifier, VotingClassifier

# Stacking: meta-learner sobre outputs dos base classifiers
stacking = StackingClassifier(
    estimators=[
        ("lgbm", LGBMClassifier(n_estimators=100, num_leaves=31)),
        ("logreg", LogisticRegression(max_iter=1000)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64))),
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method="predict_proba",
)

# Soft voting (mais simples)
voting = VotingClassifier(
    estimators=[...],
    voting="soft",
    weights=[3, 1, 2],  # LightGBM pesa mais
)
```

**Acceptance criteria:**
- ≥ +2pp Macro-F1 vs LightGBM isolado
- Latência ≤ 3× baseline (aceitável se precisão justificar)

---

## Track 4 — Retrieval Enhancement (Impacto Esperado: +3–8pp MRR)

> **Racional:** H1 mostrou que Hybrid-RRF atinge MRR=0,826 sem significância vs BM25. Retrieval melhor beneficia todo o pipeline downstream.

### 4.1 Cross-Encoder Reranking

**O que:** Reranker neural sobre top-K do retrieval. O TalkEx já tem o protocol `Reranker` definido mas NÃO invocado.

```python
from sentence_transformers import CrossEncoder

# Reranker multilingual
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Pipeline: BM25+ANN top-50 → Reranker top-10
candidates = hybrid_retriever.retrieve(query, top_k=50)
pairs = [(query.text, hit.text) for hit in candidates.hits]
scores = reranker.predict(pairs)

# Re-ordenar por score do cross-encoder
reranked = sorted(zip(candidates.hits, scores), key=lambda x: -x[1])[:10]
```

**Implementação no TalkEx:**
- Implementar `CrossEncoderReranker` conformando ao protocol `Reranker`
- Ativar invocação no `SimpleHybridRetriever` (slot já existe)

**Acceptance criteria:**
- MRR ≥ 0,870 (+4pp vs baseline)
- Wilcoxon p < 0,05 vs Hybrid-RRF sem reranker
- Latência total ≤ 500ms por query (reranking é caro)

---

### 4.2 BM25 com Stemming/Lemmatization

**O que:** O BM25 atual usa tokenização simples (whitespace + lowercase). Stemming PT-BR pode melhorar recall.

```python
import nltk
from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()  # Stemmer para português

def tokenize_with_stemming(text: str) -> list[str]:
    tokens = simple_tokenize(text)  # atual
    return [stemmer.stem(t) for t in tokens]

# Ou: usar spaCy pt_core_news_sm para lemmatization
import spacy
nlp = spacy.load("pt_core_news_sm")

def tokenize_with_lemma(text: str) -> list[str]:
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_punct]
```

**Acceptance criteria:**
- MRR BM25-stemmed ≥ MRR BM25-base + 1pp
- Não degradar hybrid (stemming pode piorar exact match)

---

### 4.3 Sparse-Dense Hybrid com BGE-M3

**O que:** BGE-M3 gera embeddings dense + sparse (learned sparse) em uma única passada. Substitui BM25+ANN por um modelo unificado.

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

# Gera dense + sparse + ColBERT em uma chamada
output = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)

# Dense: output["dense_vecs"] — para ANN search
# Sparse: output["lexical_weights"] — substitui BM25
# ColBERT: output["colbert_vecs"] — late interaction
```

**Acceptance criteria:**
- MRR ≥ 0,860 com BGE-M3 hybrid
- Substitui necessidade de BM25 separado + ANN separado

---

## Track 5 — Evolução do Rule Engine (Impacto Esperado: Variável)

> **Racional:** H3 mostrou que regras lexicais são redundantes com embeddings ricos. Para regras serem úteis, precisam operar em dimensões que o classificador NÃO acessa.

### 5.1 Semantic Predicates (Integração com Embeddings)

**O que:** Regras que usam similaridade semântica em vez de keywords.

**Implementação no TalkEx:**

```python
# No evaluator.py, handler para SEMANTIC predicates:
class SemanticPredicateHandler:
    def __init__(self, embedding_model):
        self.model = embedding_model
        # Pré-computar protótipos por classe
        self.prototypes = {
            "cancelamento": self.model.encode("quero cancelar meu plano"),
            "reclamacao": self.model.encode("estou insatisfeito com o atendimento"),
        }

    def evaluate(self, predicate: PredicateNode, window: ContextWindow) -> PredicateResult:
        window_emb = self.model.encode(window.rendered_text)
        target_emb = self.prototypes[predicate.value]
        similarity = cosine_similarity(window_emb, target_emb)
        return PredicateResult(
            matched=similarity >= predicate.threshold,
            score=similarity,
            evidence=f"semantic similarity: {similarity:.3f} (threshold: {predicate.threshold})",
        )
```

**Regras DSL com predicados semânticos:**
```
RULE high_cancel_intent
  WHEN semantic.similarity("cancelamento") > 0.85
   AND structural.speaker == "customer"
  THEN tag("high_confidence_cancel"), score(0.95)

RULE complaint_escalation
  WHEN semantic.similarity("reclamacao") > 0.80
   AND context.turn_count > 10
   AND lexical.contains("procon|ouvidoria|processo")
  THEN tag("escalation_risk"), priority("high")
```

**Acceptance criteria:**
- Semantic predicates invocáveis via DSL
- Testes positivos e negativos para cada predicado
- Quando usadas como features: ≥ +1pp Macro-F1 vs ML-only

---

### 5.2 Contextual Predicates (Cross-Window)

**O que:** Regras que avaliam padrões entre janelas, não apenas dentro de uma janela.

```python
# Predicados contextuais:
# - "intent mudou entre janela N e N+1?"
# - "sentiment degradou ao longo da conversa?"
# - "cliente repetiu a mesma reclamação em 3+ janelas?"

class ContextualPredicateHandler:
    def evaluate_cross_window(
        self, predicate: PredicateNode, windows: list[ContextWindow]
    ) -> PredicateResult:
        if predicate.field_name == "intent_shift":
            # Detectar mudança de intent entre janelas consecutivas
            ...
        elif predicate.field_name == "repeated_complaint":
            # Mesmo tema em N+ janelas
            ...
```

---

## Track 6 — Infraestrutura de Experimentação (Fundação)

> **Racional:** Antes de correr, é preciso saber medir. Infraestrutura robusta de experimentação evita falsos positivos e acelera iteração.

### 6.1 A/B Testing Framework

**O que:** Framework automatizado para comparar experimentos.

```python
# experiments/framework/ab_test.py

@dataclass
class ExperimentResult:
    experiment_id: str
    config: dict
    metrics: dict[str, float]          # macro_f1, accuracy, etc.
    per_class_f1: dict[str, float]
    per_seed_metrics: list[dict]
    training_time_ms: float
    inference_time_ms: float
    timestamp: str

class ABTestRunner:
    def __init__(self, baseline: ExperimentResult):
        self.baseline = baseline

    def compare(self, candidate: ExperimentResult) -> ABTestReport:
        """Gera relatório completo de comparação."""
        return ABTestReport(
            macro_f1_diff=candidate.metrics["macro_f1"] - self.baseline.metrics["macro_f1"],
            wilcoxon_p=wilcoxon_test(baseline_scores, candidate_scores),
            bootstrap_ci=bootstrap_ci(baseline_scores, candidate_scores),
            per_class_regression=check_per_class_regression(
                self.baseline.per_class_f1, candidate.per_class_f1, threshold=0.03
            ),
            promotion_eligible=self._check_promotion_gate(candidate),
        )

    def _check_promotion_gate(self, candidate) -> bool:
        """Verifica se candidato passa no gate de promoção."""
        return (
            candidate.metrics["macro_f1"] - self.baseline.metrics["macro_f1"] >= 0.02
            and self.compare(candidate).wilcoxon_p < 0.05
            and not self.compare(candidate).per_class_regression
            and candidate.inference_time_ms <= 2 * self.baseline.inference_time_ms
        )
```

---

### 6.2 Holm-Bonferroni Correction

**O que:** Correção para múltiplas comparações. Com N experimentos, o risco de falso positivo cresce.

```python
from statsmodels.stats.multitest import multipletests

# Quando comparar K configurações contra baseline:
p_values = [wilcoxon_p_for_each_config]
rejected, corrected_p, _, _ = multipletests(p_values, method="holm")
```

**Regra:** Toda bateria com ≥ 3 comparações simultâneas DEVE usar Holm-Bonferroni.

---

### 6.3 Feature Importance & Interpretability

**O que:** Entender quais features o LightGBM usa para decidir.

```python
import shap

# SHAP values para LightGBM
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test)

# Summary plot: quais features mais importam globalmente
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Per-class: quais features discriminam cada classe
for cls_idx, cls_name in enumerate(class_names):
    shap.summary_plot(shap_values[cls_idx], X_test, feature_names=feature_names)
```

**Entregas:**
- SHAP summary plot global
- SHAP per-class plots (9 classes)
- Top-20 features por classe
- Identificar features redundantes para simplificação

---

## Priorização e Sequência de Execução

### Fase 1 — Fundação (Semanas 1-2)

| # | Task | Track | Impacto esperado | Dependência |
|---|---|---|---|---|
| 1.1 | Error analysis sistemática | T2.1 | Diagnóstico | Nenhuma |
| 1.2 | A/B testing framework | T6.1 | Infra | Nenhuma |
| 1.3 | Cross-validation | T3.4 | Estimativa robusta | Nenhuma |
| 1.4 | Feature importance (SHAP) | T6.3 | Diagnóstico | Nenhuma |

**Gate:** Diagnóstico completo antes de mudar qualquer coisa.

### Fase 2 — Quick Wins (Semanas 3-4)

| # | Task | Track | Impacto esperado | Dependência |
|---|---|---|---|---|
| 2.1 | Trocar base model (e5-large, bge-m3) | T1.1 | +3-5pp | T6.1 |
| 2.2 | Hyperparameter tuning (Optuna) | T3.1 | +2-4pp | T6.1 |
| 2.3 | Window aggregation experiments | T3.3 | +2-3pp | T6.1 |
| 2.4 | Tratamento classe "outros" | T2.2 | +2-5pp (Macro) | T2.1 |

**Gate:** Cada experimento passa pelo ABTestRunner. Promover os que passam no gate.

### Fase 3 — Fine-Tuning (Semanas 5-8)

| # | Task | Track | Impacto esperado | Dependência |
|---|---|---|---|---|
| 3.1 | SetFit fine-tuning | T1.2 | +5-10pp | T1.1 (melhor base) |
| 3.2 | Contrastive learning com hard negatives | T1.3 | +8-15pp | T2.1 (confusion matrix) |
| 3.3 | Probability calibration | T3.2 | Melhora routing | T3.1 |
| 3.4 | Ensemble stacking | T3.5 | +2-4pp | T3.1, T1.2 |
| 3.5 | Aumento de dados direcionado | T2.3 | +3-8pp | T2.1 |

**Gate:** F1 ≥ 0,800 antes de prosseguir para Track 4-5.

### Fase 4 — Retrieval & Rules (Semanas 9-12)

| # | Task | Track | Impacto esperado | Dependência |
|---|---|---|---|---|
| 4.1 | Cross-encoder reranking | T4.1 | +4pp MRR | Nenhuma |
| 4.2 | BM25 com stemming PT-BR | T4.2 | +1-2pp MRR | Nenhuma |
| 4.3 | BGE-M3 sparse-dense | T4.3 | +3-5pp MRR | T1.1 |
| 4.4 | Semantic predicates no rule engine | T5.1 | Qualitativo | T1.2/T1.3 |
| 4.5 | Label noise reduction | T2.4 | +2-3pp | T3.2 |

### Fase 5 — Consolidação e SOTA (Semanas 13-16)

| # | Task | Track | Impacto esperado | Dependência |
|---|---|---|---|---|
| 5.1 | ONNX Runtime otimização | T1.4 | 2-4× speedup | T1.2/T1.3 |
| 5.2 | Full pipeline benchmark (todas as melhorias) | Todos | Consolidação | Todas |
| 5.3 | Holm-Bonferroni sobre toda a bateria | T6.2 | Rigor estatístico | 5.2 |
| 5.4 | Documentação final (dissertação update) | — | — | 5.2 |

---

## Metas por Fase

| Fase | Meta Macro-F1 | Meta MRR | Status |
|---|---|---|---|
| Baseline | 0,659 | 0,826 | Atual |
| Pós-Fase 2 | ≥ 0,720 | 0,826 | Quick wins |
| Pós-Fase 3 | ≥ 0,800 | 0,826 | Fine-tuning |
| Pós-Fase 4 | ≥ 0,830 | ≥ 0,870 | Retrieval + Rules |
| Pós-Fase 5 | **≥ 0,850** | **≥ 0,900** | **SOTA target** |

---

## Dependências Técnicas

```bash
# Novas dependências por track:
# Track 1
pip install setfit                    # SetFit fine-tuning
pip install optimum[onnxruntime]      # ONNX inference
pip install FlagEmbedding             # BGE-M3

# Track 3
pip install optuna                    # Hyperparameter optimization
pip install shap                      # Feature importance

# Track 4
pip install nltk                      # RSLP stemmer (PT-BR)
# ou
pip install spacy && python -m spacy download pt_core_news_sm

# Track 6
pip install statsmodels               # Holm-Bonferroni correction
```

---

## Anti-Patterns (NÃO Fazer)

1. **NÃO otimizar no test set.** Hiperparâmetros selecionados no validation set. Test set é sacrossanto.
2. **NÃO mudar múltiplas variáveis simultaneamente.** Cada experimento varia UMA dimensão.
3. **NÃO declarar melhoria sem teste estatístico.** Wilcoxon + Bootstrap CI obrigatórios.
4. **NÃO ignorar regressões per-class.** Macro-F1 sobe mas uma classe colapsa = não promover.
5. **NÃO fazer fine-tuning antes de trocar o base model.** Modelo maior primeiro, fine-tuning depois.
6. **NÃO misturar janelas de conversas diferentes no mesmo fold de CV.** Split sempre por conversa.
7. **NÃO descartar "outros" sem análise de impacto.** Pode ser requisito de negócio.
8. **NÃO usar GPU como requisito.** CPU-friendly é proposta de valor. GPU opcional para aceleração.
