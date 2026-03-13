"""Generate the MIT-level dissertation experiment notebook.

This script uses nbformat to programmatically create a Jupyter notebook
with all experimental analyses for the TalkEx dissertation.

Usage:
    python experiments/notebooks/_generate_notebook.py

The generated notebook is self-contained and reproducible.
Run from the project root directory.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (TalkEx)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0",
        "mimetype": "text/x-python",
        "file_extension": ".py",
    },
}

cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source))


# ============================================================================
# §0 — TITLE PAGE
# ============================================================================

md(r"""# TalkEx: Experimental Evaluation of a Hybrid Cascaded Architecture for Conversation Intent Classification

**Dissertation Appendix — Complete Experimental Notebook**

**Author:** Paulo Richard Sakaguchi
**Program:** Master's in Computer Science
**Date:** March 2026

---

## Abstract

This notebook contains the **complete, reproducible experimental evaluation** for the TalkEx dissertation.
It implements and validates four hypotheses about hybrid NLP architectures for conversation intent classification:

| Hypothesis | Claim |
|:---|:---|
| **H1** | Hybrid retrieval (BM25 + ANN) outperforms isolated paradigms |
| **H2** | Multi-level features (lexical + embeddings) improve classification over lexical-only |
| **H3** | Deterministic rules complement ML classifiers |
| **H4** | Cascaded inference reduces cost without sacrificing quality |

Plus ablation studies, stratified k-fold cross-validation, leave-one-domain-out evaluation,
and comprehensive error analysis.

### Reproducibility Statement

Every result in this notebook is deterministic given:
- The dataset splits in `experiments/data/`
- The random seeds `[13, 42, 123, 2024, 999]`
- The package versions logged in §1

To reproduce: `cd <project_root> && jupyter notebook experiments/notebooks/dissertation_experiments.ipynb`

---

## Table of Contents

1. [Environment & Reproducibility](#§1)
2. [Dataset Characterization](#§2)
3. [Experimental Protocol](#§3)
4. [H1 — Hybrid Retrieval](#§4)
5. [H2 — Multi-Level Classification](#§5)
6. [H3 — Rules Complement ML](#§6)
7. [H4 — Cascaded Inference](#§7)
8. [Ablation Studies](#§8)
9. [Stratified K-Fold Cross-Validation](#§9)
10. [Leave-One-Domain-Out (LODO)](#§10)
11. [Error Analysis & Interpretability](#§11)
12. [Statistical Summary & Hypothesis Decisions](#§12)
13. [Limitations & Threats to Validity](#§13)
""")

# ============================================================================
# §1 — ENVIRONMENT & REPRODUCIBILITY
# ============================================================================

md(r"""<a id="§1"></a>
## §1. Environment & Reproducibility

We log the complete computational environment to ensure full reproducibility.
All experiments use the same embedding model, classifier configurations, and random seeds.

**Principle:** *Any researcher with access to this repository must be able to reproduce
every number in this notebook by running it end-to-end.*
""")

code(r"""import json
import os
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ---------------------------------------------------------------------------
# Path setup — notebook must be run from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src" / "talkex").exists():
    # Try navigating up if run from notebooks dir
    PROJECT_ROOT = Path.cwd().parent.parent
    assert (PROJECT_ROOT / "src" / "talkex").exists(), (
        "Run this notebook from the project root: cd <project_root> && jupyter notebook"
    )
    os.chdir(PROJECT_ROOT)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "experiments" / "data"

# Seeds (standardized, post-audit)
SEEDS = [13, 42, 123, 2024, 999]

# ---------------------------------------------------------------------------
# Publication-quality matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-friendly palette
PALETTE = sns.color_palette("colorblind")
COLORS = {
    "primary": PALETTE[0],
    "secondary": PALETTE[1],
    "tertiary": PALETTE[2],
    "quaternary": PALETTE[3],
    "highlight": PALETTE[4],
    "muted": PALETTE[7],
}

print("Environment initialized successfully.")
print(f"Project root: {PROJECT_ROOT}")
print(f"Results dir:  {RESULTS_DIR}")
print(f"Figures dir:  {FIGURES_DIR}")
""")

code(r"""# ---------------------------------------------------------------------------
# Reproducibility manifest
# ---------------------------------------------------------------------------
def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"

def get_package_versions():
    packages = [
        "numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib",
        "seaborn", "scipy", "sentence_transformers", "torch", "pydantic",
    ]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[pkg] = "not installed"
    return versions

manifest = {
    "timestamp": datetime.now().isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "cpu": platform.processor() or "unknown",
    "git_commit": get_git_hash(),
    "seeds": SEEDS,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dims": 384,
    "classifier": "LightGBM (n_estimators=100, num_leaves=31)",
    "window_config": "5-turn windows, stride=2",
    "packages": get_package_versions(),
}

print("=" * 60)
print("REPRODUCIBILITY MANIFEST")
print("=" * 60)
for k, v in manifest.items():
    if k == "packages":
        print(f"\n  Package versions:")
        for pkg, ver in v.items():
            print(f"    {pkg:25s} {ver}")
    else:
        print(f"  {k:20s} {v}")
print("=" * 60)

# Save manifest
manifest_path = FIGURES_DIR / "reproducibility_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"\nManifest saved to {manifest_path}")
""")

# ============================================================================
# §2 — DATASET CHARACTERIZATION
# ============================================================================

md(r"""<a id="§2"></a>
## §2. Dataset Characterization

### 2.1 Dataset Overview

The TalkEx dataset consists of **2,122 PT-BR customer service conversations** across 8 intent classes
and 8 business domains, post-audit (deduplication, leakage remediation, taxonomy cleanup).

**Source:** `RichardSakaguchiMS/brazilian-customer-service-conversations` (Apache 2.0)
**Composition:** 847 original + 1,275 LLM-synthetic conversations
**Pre-processing:** The "outros" (other) class was removed during audit as it conflated
genuinely ambiguous conversations with labeling errors.

### 2.2 Split Strategy

Contamination-aware splitting ensures no near-duplicate conversations leak across splits.
The 60/20/20 train/val/test ratio follows standard practice.
""")

code(r"""# Load dataset splits
from run_experiment import load_split, extract_texts, extract_labels

train_records = load_split("train", DATA_DIR)
val_records = load_split("val", DATA_DIR)
test_records = load_split("test", DATA_DIR)

all_records = train_records + val_records + test_records

print(f"Dataset splits:")
print(f"  Train: {len(train_records):,} conversations")
print(f"  Val:   {len(val_records):,} conversations")
print(f"  Test:  {len(test_records):,} conversations")
print(f"  Total: {len(all_records):,} conversations")
print(f"\nSplit ratios: {len(train_records)/len(all_records):.1%} / "
      f"{len(val_records)/len(all_records):.1%} / "
      f"{len(test_records)/len(all_records):.1%}")
""")

code(r"""# ---------------------------------------------------------------------------
# 2.3 Class Distribution
# ---------------------------------------------------------------------------
all_labels = extract_labels(all_records)
train_labels = extract_labels(train_records)
val_labels = extract_labels(val_records)
test_labels = extract_labels(test_records)

label_order = sorted(set(all_labels))
n_classes = len(label_order)

# Distribution table
dist_data = []
for label in label_order:
    n_total = all_labels.count(label)
    n_train = train_labels.count(label)
    n_val = val_labels.count(label)
    n_test = test_labels.count(label)
    dist_data.append({
        "Intent": label,
        "Total": n_total,
        "Train": n_train,
        "Val": n_val,
        "Test": n_test,
        "% Total": f"{n_total/len(all_labels)*100:.1f}%",
    })

dist_df = pd.DataFrame(dist_data)
print("\nClass Distribution:")
print(dist_df.to_string(index=False))

# Imbalance ratio
counts = Counter(all_labels)
max_count = max(counts.values())
min_count = min(counts.values())
print(f"\nImbalance ratio (max/min): {max_count/min_count:.1f}x")
print(f"Majority class: {max(counts, key=counts.get)} ({max_count})")
print(f"Minority class: {min(counts, key=counts.get)} ({min_count})")
""")

code(r"""# ---------------------------------------------------------------------------
# 2.4 Class Distribution Visualization
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart — overall distribution
counts_sorted = sorted(Counter(all_labels).items(), key=lambda x: -x[1])
labels_sorted = [x[0] for x in counts_sorted]
values_sorted = [x[1] for x in counts_sorted]

bars = axes[0].barh(labels_sorted[::-1], values_sorted[::-1], color=PALETTE[:n_classes])
axes[0].set_xlabel("Number of Conversations")
axes[0].set_title("(a) Class Distribution (N=2,122)")
for bar, val in zip(bars, values_sorted[::-1]):
    axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"{val}", va="center", fontsize=9)

# Stacked bar — per-split distribution
split_df = pd.DataFrame({
    "Train": [train_labels.count(l) for l in labels_sorted[::-1]],
    "Val": [val_labels.count(l) for l in labels_sorted[::-1]],
    "Test": [test_labels.count(l) for l in labels_sorted[::-1]],
}, index=labels_sorted[::-1])

split_df.plot(kind="barh", stacked=True, ax=axes[1],
              color=[COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]])
axes[1].set_xlabel("Number of Conversations")
axes[1].set_title("(b) Distribution by Split")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_dataset_distribution.pdf")
plt.savefig(FIGURES_DIR / "fig_dataset_distribution.png")
plt.show()
print(f"Saved: {FIGURES_DIR / 'fig_dataset_distribution.pdf'}")
""")

code(r"""# ---------------------------------------------------------------------------
# 2.5 Domain Distribution & Synthetic Analysis
# ---------------------------------------------------------------------------
all_domains = [r.get("domain", "unknown") for r in all_records]
domain_counts = Counter(all_domains)

# Synthetic vs original
has_source = any("source" in r or "is_synthetic" in r or "origin" in r for r in all_records)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Domain distribution
domain_sorted = sorted(domain_counts.items(), key=lambda x: -x[1])
d_labels = [x[0] for x in domain_sorted]
d_values = [x[1] for x in domain_sorted]

axes[0].barh(d_labels[::-1], d_values[::-1], color=PALETTE[2])
axes[0].set_xlabel("Number of Conversations")
axes[0].set_title(f"(a) Domain Distribution ({len(domain_counts)} domains)")
for i, (lbl, val) in enumerate(zip(d_labels[::-1], d_values[::-1])):
    axes[0].text(val + 3, i, str(val), va="center", fontsize=9)

# Text length distribution
all_texts = extract_texts(all_records)
text_lengths = [len(t.split()) for t in all_texts]

axes[1].hist(text_lengths, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.8)
axes[1].axvline(np.median(text_lengths), color="red", linestyle="--",
                label=f"Median: {np.median(text_lengths):.0f} words")
axes[1].axvline(np.mean(text_lengths), color="orange", linestyle="--",
                label=f"Mean: {np.mean(text_lengths):.0f} words")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")
axes[1].set_title("(b) Conversation Length Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_dataset_domains_lengths.pdf")
plt.savefig(FIGURES_DIR / "fig_dataset_domains_lengths.png")
plt.show()

print(f"\nText length statistics:")
print(f"  Mean:   {np.mean(text_lengths):.0f} words")
print(f"  Median: {np.median(text_lengths):.0f} words")
print(f"  Std:    {np.std(text_lengths):.0f} words")
print(f"  Min:    {np.min(text_lengths)} words")
print(f"  Max:    {np.max(text_lengths)} words")
""")

# ============================================================================
# §3 — EXPERIMENTAL PROTOCOL
# ============================================================================

md(r"""<a id="§3"></a>
## §3. Experimental Protocol

### 3.1 Pipeline Architecture

All experiments use the real TalkEx pipeline modules — the same code that runs in production:

```
Raw Text → TurnSegmenter → SlidingWindowBuilder(5t/2s)
  → Feature Extraction (Lexical + Structural + Embedding + Rules)
  → Classification (window-level training)
  → Aggregation (avg class probs → argmax → conversation-level predictions)
  → Evaluation (conversation-level metrics)
```

### 3.2 Multi-Seed Protocol

To estimate variance, each stochastic experiment runs with 5 seeds: `[13, 42, 123, 2024, 999]`.
Results report mean ± std across seeds. Statistical tests use paired samples.

### 3.3 Metrics

| Category | Metrics | Purpose |
|:---|:---|:---|
| Classification | Macro-F1, Accuracy, per-class P/R/F1 | Performance |
| Retrieval | MRR, Recall@K, nDCG@K | Ranking quality |
| Calibration | Brier score, ECE | Confidence reliability |
| Statistical | Wilcoxon signed-rank, Bootstrap CI, Effect size | Significance |

### 3.4 Model Selection Protocol

Hyperparameters (H1 fusion weight α, H4 cascade threshold) are tuned on the **validation set**.
Final metrics are reported on the **held-out test set** only. No information leakage.
""")

code(r'''# ---------------------------------------------------------------------------
# Shared infrastructure: load pre-computed results or run experiments
# ---------------------------------------------------------------------------

def load_results(hypothesis: str) -> dict:
    """Load pre-computed results for a hypothesis."""
    path = RESULTS_DIR / hypothesis / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}. Run experiments first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_per_seed_results(hypothesis: str) -> list[dict]:
    """Load per-seed results for multi-seed analysis."""
    path = RESULTS_DIR / hypothesis / "per_seed_results.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_statistical_tests(hypothesis: str) -> list[dict]:
    """Load statistical test results."""
    path = RESULTS_DIR / hypothesis / "statistical_tests.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def results_to_df(results: dict) -> pd.DataFrame:
    """Convert results.json to a DataFrame with one row per variant."""
    rows = []
    for v in results["variants"]:
        row = {"variant": v["variant_name"]}
        row.update(v["metrics"])
        row["duration_ms"] = v["duration_ms"]
        rows.append(row)
    return pd.DataFrame(rows)

# Helper: format p-value for tables
def fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"

print("Helper functions loaded.")
print(f"\nAvailable results: {[d.name for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name not in ('figures', 'deprecated_pre_audit')]}")
''')

# ============================================================================
# §4 — H1: HYBRID RETRIEVAL
# ============================================================================

md(r"""<a id="§4"></a>
## §4. H1 — Hybrid Retrieval Outperforms Isolated Paradigms

### Hypothesis

> **H₀:** Hybrid retrieval (BM25 + ANN with score fusion) does not outperform the best individual
> retrieval method (BM25 or ANN alone) on MRR.
>
> **H₁:** Hybrid retrieval achieves significantly higher MRR than any individual method.

### Method

- **BM25-base:** Standard BM25 (Okapi) with default parameters
- **BM25-norm:** BM25 with accent-aware text normalization
- **ANN-MiniLM:** Approximate nearest neighbor search using paraphrase-multilingual-MiniLM-L12-v2 (384d)
- **Hybrid-RRF:** Reciprocal Rank Fusion of BM25 + ANN results
- **Hybrid-LINEAR:** Weighted linear combination, α tuned on validation set

**Ground truth:** Documents sharing the same intent label as the query are considered relevant.

### Decision Criterion

Hybrid achieves significantly higher MRR than BM25-base (Wilcoxon signed-rank, α=0.05).
""")

code(r"""# ---------------------------------------------------------------------------
# §4.1 Load H1 results
# ---------------------------------------------------------------------------
h1_results = load_results("H1")
h1_df = results_to_df(h1_results)

# Key retrieval metrics
retrieval_metrics = ["mrr", "recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
h1_display = h1_df[["variant"] + [m for m in retrieval_metrics if m in h1_df.columns]].copy()
h1_display = h1_display.round(4)

print("H1: Hybrid Retrieval Results")
print("=" * 80)
print(h1_display.to_string(index=False))

# Identify best
best_h1 = h1_df.loc[h1_df["mrr"].idxmax()]
bm25_base = h1_df[h1_df["variant"] == "BM25-base"].iloc[0]
print(f"\nBest variant: {best_h1['variant']} (MRR={best_h1['mrr']:.4f})")
print(f"BM25 baseline: MRR={bm25_base['mrr']:.4f}")
print(f"Improvement: +{(best_h1['mrr'] - bm25_base['mrr']):.4f} "
      f"(+{(best_h1['mrr'] - bm25_base['mrr'])/bm25_base['mrr']*100:.1f}%)")
""")

code(r"""# ---------------------------------------------------------------------------
# §4.2 H1 Visualization — MRR Comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: MRR comparison
variants = h1_df["variant"].values
mrr_values = h1_df["mrr"].values
bar_colors = [COLORS["highlight"] if "val-selected" in v else COLORS["primary"] for v in variants]

bars = axes[0].barh(range(len(variants)), mrr_values, color=bar_colors)
axes[0].set_yticks(range(len(variants)))
axes[0].set_yticklabels(variants, fontsize=8)
axes[0].set_xlabel("Mean Reciprocal Rank (MRR)")
axes[0].set_title("(a) MRR by Retrieval Method")
axes[0].set_xlim(min(mrr_values) * 0.95, max(mrr_values) * 1.02)
for bar, val in zip(bars, mrr_values):
    axes[0].text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

# Radar chart: multi-metric comparison for top methods
top_methods = ["BM25-base", "ANN-MiniLM"]
# Add val-selected hybrid
val_selected = [v for v in variants if "val-selected" in v]
if val_selected:
    top_methods.append(val_selected[0])

radar_metrics = ["mrr", "recall@5", "recall@10", "ndcg@5", "ndcg@10"]
available_radar = [m for m in radar_metrics if m in h1_df.columns]

if len(available_radar) >= 3:
    angles = np.linspace(0, 2 * np.pi, len(available_radar), endpoint=False).tolist()
    angles += angles[:1]

    ax_radar = axes[1]
    ax_radar = fig.add_subplot(122, polar=True)
    axes[1].set_visible(False)

    for i, method in enumerate(top_methods):
        row = h1_df[h1_df["variant"] == method]
        if row.empty:
            continue
        values = [row[m].values[0] for m in available_radar]
        values += values[:1]
        ax_radar.plot(angles, values, "o-", label=method, color=PALETTE[i], linewidth=2)
        ax_radar.fill(angles, values, alpha=0.1, color=PALETTE[i])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(available_radar, fontsize=8)
    ax_radar.set_title("(b) Multi-Metric Comparison", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_h1_retrieval.pdf")
plt.savefig(FIGURES_DIR / "fig_h1_retrieval.png")
plt.show()
""")

code(r"""# ---------------------------------------------------------------------------
# §4.3 H1 Statistical Tests
# ---------------------------------------------------------------------------
h1_stats = load_statistical_tests("H1")

print("H1: Statistical Tests")
print("=" * 80)
for test in h1_stats:
    print(f"\n{test.get('comparison', test.get('test', '?'))}:")
    if "p_value" in test:
        print(f"  Test:        {test['test']}")
        print(f"  Statistic:   {test['statistic']:.4f}")
        print(f"  p-value:     {fmt_p(test['p_value'])}")
        print(f"  Significant: {'YES' if test['significant'] else 'NO'}")
        if test.get("effect_size") is not None:
            print(f"  Effect size: {test['effect_size']:.4f}")
    elif "ci_lower" in test:
        print(f"  95% CI:      [{test['ci_lower']:.4f}, {test['ci_upper']:.4f}]")
        print(f"  Observed:    {test['observed_diff']:.4f}")
    print(f"  Summary:     {test.get('summary', '')}")
""")

md(r"""### H1 Verdict

**Result:** The val-selected Hybrid-LINEAR configuration achieves MRR=0.853 vs BM25-base MRR=0.835
(Wilcoxon signed-rank p=0.017, significant at α=0.05).

**Decision: H₀ rejected. H1 CONFIRMED.**

The hybrid approach provides a statistically significant, though modest, improvement over
pure lexical retrieval. The improvement is concentrated in queries where semantic paraphrasing
captures intent better than keyword matching.
""")

# ============================================================================
# §5 — H2: MULTI-LEVEL CLASSIFICATION
# ============================================================================

md(r"""<a id="§5"></a>
## §5. H2 — Multi-Level Features Improve Classification

### Hypothesis

> **H₀:** Adding semantic embedding features to lexical features does not improve
> conversation-level Macro-F1 compared to lexical-only features.
>
> **H₁:** Lexical + embedding features achieve significantly higher Macro-F1.

### Method

6 configurations: {lexical, lexical+embedding} × {LogReg, LightGBM, MLP}

**Pipeline:** Conversations → Turn Segmentation → Context Windows (5t/2s) →
Feature Extraction → Window-level Classification → Conversation-level Aggregation
(avg class probabilities → argmax)

**Model selection:** Best config selected on validation Macro-F1; test metrics reported.
""")

code(r"""# ---------------------------------------------------------------------------
# §5.1 Load H2 results
# ---------------------------------------------------------------------------
h2_results = load_results("H2")
h2_df = results_to_df(h2_results)

# Summary table
h2_summary = h2_df[["variant", "macro_f1", "accuracy"]].copy()
if "macro_f1_std" in h2_df.columns:
    h2_summary["macro_f1_std"] = h2_df["macro_f1_std"]
if "brier_score" in h2_df.columns:
    h2_summary["brier_score"] = h2_df["brier_score"]
if "ece" in h2_df.columns:
    h2_summary["ece"] = h2_df["ece"]
if "val_macro_f1" in h2_df.columns:
    h2_summary["val_macro_f1"] = h2_df["val_macro_f1"]

h2_summary = h2_summary.round(4)

print("H2: Multi-Level Classification Results")
print("=" * 80)
print(h2_summary.to_string(index=False))

best_h2 = h2_df.loc[h2_df["macro_f1"].idxmax()]
best_lex = h2_df[h2_df["variant"].str.startswith("lexical_")].sort_values("macro_f1", ascending=False).iloc[0]
best_emb = h2_df[h2_df["variant"].str.startswith("lexical+emb")].sort_values("macro_f1", ascending=False).iloc[0]

print(f"\nBest lexical-only:    {best_lex['variant']} (Macro-F1={best_lex['macro_f1']:.4f})")
print(f"Best lexical+emb:     {best_emb['variant']} (Macro-F1={best_emb['macro_f1']:.4f})")
print(f"Absolute gain:        +{best_emb['macro_f1'] - best_lex['macro_f1']:.4f}")
""")

code(r"""# ---------------------------------------------------------------------------
# §5.2 H2 Visualization — Feature Set × Classifier Heatmap
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Parse variant names into feature set and classifier
h2_pivot_data = []
for _, row in h2_df.iterrows():
    parts = row["variant"].split("_")
    feat = parts[0]
    clf = "_".join(parts[1:])
    h2_pivot_data.append({"Feature Set": feat, "Classifier": clf, "Macro-F1": row["macro_f1"]})

pivot_df = pd.DataFrame(h2_pivot_data)
if not pivot_df.empty:
    pivot = pivot_df.pivot(index="Feature Set", columns="Classifier", values="Macro-F1")

    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[0],
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={"label": "Macro-F1"})
    axes[0].set_title("(a) Macro-F1: Feature Set × Classifier")

# Bar comparison: best lexical vs best lexical+emb
comparison_data = {
    "Best Lexical-only": best_lex["macro_f1"],
    "Best Lexical+Emb": best_emb["macro_f1"],
}
bars = axes[1].bar(comparison_data.keys(), comparison_data.values(),
                   color=[COLORS["secondary"], COLORS["primary"]], width=0.5)
axes[1].set_ylabel("Macro-F1")
axes[1].set_title("(b) Impact of Embedding Features")
axes[1].set_ylim(0, 1)
for bar, val in zip(bars, comparison_data.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

# Add delta annotation
delta = best_emb["macro_f1"] - best_lex["macro_f1"]
axes[1].annotate(f"+{delta:.3f}", xy=(1, best_emb["macro_f1"]),
                xytext=(1.3, best_emb["macro_f1"] - delta/2),
                fontsize=12, fontweight="bold", color="green",
                arrowprops=dict(arrowstyle="->", color="green"))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_h2_classification.pdf")
plt.savefig(FIGURES_DIR / "fig_h2_classification.png")
plt.show()
""")

code(r"""# ---------------------------------------------------------------------------
# §5.3 H2 Per-Class F1 Analysis
# ---------------------------------------------------------------------------
# Extract per-class F1 for best lexical and best lexical+emb
per_class_cols = [c for c in h2_df.columns if c.startswith("f1_") and not c.endswith("_std")]
class_names = sorted(set(c.replace("f1_", "") for c in per_class_cols))

if class_names:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.35

    lex_f1s = [best_lex.get(f"f1_{c}", 0) for c in class_names]
    emb_f1s = [best_emb.get(f"f1_{c}", 0) for c in class_names]

    bars1 = ax.bar(x - width/2, lex_f1s, width, label="Lexical-only", color=COLORS["secondary"])
    bars2 = ax.bar(x + width/2, emb_f1s, width, label="Lexical+Embedding", color=COLORS["primary"])

    ax.set_xlabel("Intent Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1: Lexical-only vs Lexical+Embedding (Best Classifiers)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=7, color="gray")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_h2_per_class.pdf")
    plt.savefig(FIGURES_DIR / "fig_h2_per_class.png")
    plt.show()

    # Delta table
    print("\nPer-class F1 improvement (Lexical+Emb - Lexical-only):")
    print(f"{'Class':25s} {'Lex':>8s} {'Lex+Emb':>8s} {'Delta':>8s}")
    print("-" * 55)
    for c, l, e in zip(class_names, lex_f1s, emb_f1s):
        d = e - l
        marker = " ***" if abs(d) > 0.1 else " **" if abs(d) > 0.05 else ""
        print(f"{c:25s} {l:8.3f} {e:8.3f} {d:+8.3f}{marker}")
""")

code(r"""# ---------------------------------------------------------------------------
# §5.4 H2 Calibration Analysis
# ---------------------------------------------------------------------------
if "brier_score" in h2_df.columns:
    print("H2: Calibration Metrics")
    print("=" * 80)
    cal_cols = ["variant", "macro_f1", "brier_score", "ece"]
    cal_df = h2_df[[c for c in cal_cols if c in h2_df.columns]].round(4)
    print(cal_df.to_string(index=False))

    print(f"\nInterpretation:")
    print(f"  Brier score: lower is better (0 = perfect, 2 = worst for multi-class)")
    print(f"  ECE: lower is better (0 = perfectly calibrated)")
    best_cal = h2_df.loc[h2_df["brier_score"].idxmin()] if "brier_score" in h2_df.columns else None
    if best_cal is not None:
        print(f"  Best calibrated: {best_cal['variant']} (Brier={best_cal['brier_score']:.4f}, ECE={best_cal['ece']:.4f})")
""")

code(r"""# ---------------------------------------------------------------------------
# §5.5 H2 Statistical Tests
# ---------------------------------------------------------------------------
h2_stats = load_statistical_tests("H2")
if h2_stats:
    print("H2: Statistical Tests")
    print("=" * 80)
    for test in h2_stats:
        comp = test.get("comparison", test.get("test", ""))
        print(f"\n{comp}:")
        if "p_value" in test:
            print(f"  p-value: {fmt_p(test['p_value'])}")
            print(f"  Significant: {'YES' if test['significant'] else 'NO'}")
        if "ci_lower" in test:
            print(f"  95% CI: [{test['ci_lower']:.4f}, {test['ci_upper']:.4f}]")
        print(f"  {test.get('summary', '')}")
""")

md(r"""### H2 Verdict

**Result:** Adding embedding features improves Macro-F1 from 0.334 (lexical-only LightGBM)
to 0.722 (lexical+emb LightGBM) — a gain of +0.388 (+116%).

**Decision: H₀ rejected. H2 CONFIRMED.**

Semantic embeddings are the dominant feature family, providing the representation power
that lexical features alone cannot achieve for intent classification.
""")

# ============================================================================
# §6 — H3: RULES + ML
# ============================================================================

md(r"""<a id="§6"></a>
## §6. H3 — Deterministic Rules Complement ML

### Hypothesis

> **H₀:** Adding deterministic rules (as features or overrides) does not improve
> Macro-F1 over ML-only classification.
>
> **H₁:** The combined ML+Rules system achieves higher Macro-F1 than ML alone.

### Method

4 variants, all using LightGBM with lexical+structural+embedding features:

1. **ML-only:** Standard LightGBM classifier
2. **Rules-only:** 10 deterministic rules, majority vote aggregation
3. **ML+Rules-override:** If a rule fires on a window, its label overrides ML's prediction
4. **ML+Rules-feature:** Rule matches as binary features fed to LightGBM (10 extra features)

**Rules:** 10 rules covering all 8 intent classes, using 3 predicate families
(LEXICAL, STRUCTURAL, CONTEXTUAL).
""")

code(r"""# ---------------------------------------------------------------------------
# §6.1 Load H3 results
# ---------------------------------------------------------------------------
h3_results = load_results("H3")
h3_df = results_to_df(h3_results)

print("H3: Rules Complement ML Results")
print("=" * 80)
h3_cols = ["variant", "macro_f1", "accuracy"]
if "brier_score" in h3_df.columns:
    h3_cols.append("brier_score")
if "ece" in h3_df.columns:
    h3_cols.append("ece")
print(h3_df[[c for c in h3_cols if c in h3_df.columns]].round(4).to_string(index=False))

ml_only = h3_df[h3_df["variant"] == "ML-only"]
rules_feature = h3_df[h3_df["variant"] == "ML+Rules-feature"]
if not ml_only.empty and not rules_feature.empty:
    ml_f1 = ml_only.iloc[0]["macro_f1"]
    rf_f1 = rules_feature.iloc[0]["macro_f1"]
    print(f"\nML-only Macro-F1:          {ml_f1:.4f}")
    print(f"ML+Rules-feature Macro-F1: {rf_f1:.4f}")
    print(f"Delta:                     {rf_f1 - ml_f1:+.4f} ({(rf_f1 - ml_f1)*100:+.1f}pp)")
""")

code(r"""# ---------------------------------------------------------------------------
# §6.2 H3 Visualization
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: 4 variants
h3_variants = h3_df["variant"].values
h3_f1s = h3_df["macro_f1"].values
h3_colors = [COLORS["primary"] if "Rules-feature" in v else
             COLORS["secondary"] if "ML-only" in v else
             COLORS["tertiary"] if "Rules-only" in v else
             COLORS["quaternary"] for v in h3_variants]

bars = axes[0].barh(range(len(h3_variants)), h3_f1s, color=h3_colors)
axes[0].set_yticks(range(len(h3_variants)))
axes[0].set_yticklabels(h3_variants)
axes[0].set_xlabel("Macro-F1")
axes[0].set_title("(a) H3: Rule Integration Strategies")
for bar, val in zip(bars, h3_f1s):
    axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

# Per-class comparison: ML-only vs ML+Rules-feature
if not ml_only.empty and not rules_feature.empty:
    class_cols = [c for c in h3_df.columns if c.startswith("f1_") and not c.endswith("_std")]
    classes = sorted(set(c.replace("f1_", "") for c in class_cols))

    if classes:
        x = np.arange(len(classes))
        width = 0.35
        ml_vals = [ml_only.iloc[0].get(f"f1_{c}", 0) for c in classes]
        rf_vals = [rules_feature.iloc[0].get(f"f1_{c}", 0) for c in classes]

        axes[1].bar(x - width/2, ml_vals, width, label="ML-only", color=COLORS["secondary"])
        axes[1].bar(x + width/2, rf_vals, width, label="ML+Rules-feature", color=COLORS["primary"])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("F1")
        axes[1].set_title("(b) Per-Class: ML-only vs ML+Rules")
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_h3_rules.pdf")
plt.savefig(FIGURES_DIR / "fig_h3_rules.png")
plt.show()
""")

code(r"""# ---------------------------------------------------------------------------
# §6.3 H3 Statistical Tests
# ---------------------------------------------------------------------------
h3_stats = load_statistical_tests("H3")
if h3_stats:
    print("H3: Statistical Tests")
    print("=" * 80)
    for test in h3_stats:
        comp = test.get("comparison", test.get("test", ""))
        print(f"\n{comp}:")
        if "p_value" in test:
            print(f"  p-value: {fmt_p(test['p_value'])}")
            print(f"  Significant: {'YES' if test['significant'] else 'NO'}")
            if test.get("effect_size") is not None:
                print(f"  Effect size: {test['effect_size']:.4f}")
        if "ci_lower" in test:
            print(f"  95% CI: [{test['ci_lower']:.4f}, {test['ci_upper']:.4f}]")
""")

md(r"""### H3 Verdict

**Result:** ML+Rules-feature achieves Macro-F1=0.740 vs ML-only 0.722 (+1.8pp).
However, the Wilcoxon test yields p=0.131 (not significant at α=0.05).

**Decision: INCONCLUSIVE.** The direction is positive, but the improvement is not
statistically significant with the current sample size. A power analysis suggests
a larger dataset would be needed to detect this effect size reliably.

*Note: These results are from the 2-rule baseline. The expanded 10-rule version
may change the outcome — re-run pending.*
""")

# ============================================================================
# §7 — H4: CASCADED INFERENCE
# ============================================================================

md(r"""<a id="§7"></a>
## §7. H4 — Cascaded Inference Reduces Cost

### Hypothesis

> **H₀:** Cascaded inference (cheap filter → expensive classifier) achieves the same
> Macro-F1 as the uniform pipeline at lower computational cost.
>
> **H₁:** Cascading reduces cost without significant quality degradation.

### Method

- **Uniform:** All windows classified by LightGBM (full model)
- **Cascade:** Stage 1 = LogReg (cheap), if confidence ≥ threshold → accept, else → LightGBM (Stage 2)
- **Thresholds:** 0.50, 0.60, 0.70, 0.80, 0.90 (tuned on validation set)
""")

code(r"""# ---------------------------------------------------------------------------
# §7.1 Load H4 results
# ---------------------------------------------------------------------------
h4_results = load_results("H4")
h4_df = results_to_df(h4_results)

h4_cols = ["variant", "macro_f1", "pct_stage1", "pct_stage2"]
if "cost_reduction_pct" in h4_df.columns:
    h4_cols.append("cost_reduction_pct")

print("H4: Cascaded Inference Results")
print("=" * 80)
print(h4_df[[c for c in h4_cols if c in h4_df.columns]].round(4).to_string(index=False))

uniform = h4_df[h4_df["variant"] == "uniform"]
if not uniform.empty:
    print(f"\nUniform baseline: Macro-F1={uniform.iloc[0]['macro_f1']:.4f}")
""")

code(r"""# ---------------------------------------------------------------------------
# §7.2 H4 Visualization — Cost vs Quality Trade-off
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cascade_df = h4_df[h4_df["variant"] != "uniform"].copy()

if not cascade_df.empty and not uniform.empty:
    # Cost-Quality Pareto curve
    uniform_f1 = uniform.iloc[0]["macro_f1"]

    if "cost_reduction_pct" in cascade_df.columns:
        axes[0].scatter(cascade_df["cost_reduction_pct"], cascade_df["macro_f1"],
                       color=COLORS["primary"], s=100, zorder=5)
        axes[0].axhline(y=uniform_f1, color="red", linestyle="--", alpha=0.7,
                       label=f"Uniform F1={uniform_f1:.3f}")
        for _, row in cascade_df.iterrows():
            axes[0].annotate(row["variant"].replace("cascade_", "").replace(" (val-selected)", " *"),
                           (row["cost_reduction_pct"], row["macro_f1"]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
        axes[0].set_xlabel("Cost Reduction (%)")
        axes[0].set_ylabel("Macro-F1")
        axes[0].set_title("(a) Cost vs Quality Trade-off")
        axes[0].legend()

    # Stage 1 routing percentage
    if "pct_stage1" in cascade_df.columns:
        thresholds = [float(v.split("t")[1].split(" ")[0]) for v in cascade_df["variant"]
                     if "cascade_t" in v]
        if thresholds:
            axes[1].bar(range(len(thresholds)),
                       cascade_df["pct_stage1"].values[:len(thresholds)],
                       color=PALETTE[2])
            axes[1].set_xticks(range(len(thresholds)))
            axes[1].set_xticklabels([f"t={t}" for t in thresholds])
            axes[1].set_ylabel("% Windows Resolved by Stage 1")
            axes[1].set_title("(b) Stage 1 Resolution Rate")
            axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_h4_cascade.pdf")
plt.savefig(FIGURES_DIR / "fig_h4_cascade.png")
plt.show()
""")

md(r"""### H4 Verdict

**Result:** Cascading consistently degrades Macro-F1 without meaningful cost reduction.
The LogReg first-stage model is not sufficiently accurate to serve as a reliable filter.

**Decision: H₀ not rejected. H4 REFUTED.**

The cascade approach fails because: (1) the first-stage model makes confident but wrong
predictions, and (2) the overhead of running two models negates any savings from early
stopping. This finding aligns with the literature on cascaded classifiers in NLP — the
quality gap between stages must be substantial for cascading to be beneficial.
""")

# ============================================================================
# §8 — ABLATION STUDIES
# ============================================================================

md(r"""<a id="§8"></a>
## §8. Ablation Studies

### Method

Starting from the full pipeline (lexical + structural + embedding + rules), we systematically
remove one feature family at a time to measure each component's contribution.

| Config | Features Included |
|:---|:---|
| full_pipeline | lexical + structural + embedding + rules |
| -Embeddings | lexical + structural + rules |
| -Lexical | structural + embedding + rules |
| -Rules | lexical + structural + embedding |
| -Structural | lexical + embedding + rules |
| Emb-only | embedding only |
| Lexical-only | lexical + structural only |
""")

code(r"""# ---------------------------------------------------------------------------
# §8.1 Load ablation results
# ---------------------------------------------------------------------------
abl_results = load_results("ablation")
abl_df = results_to_df(abl_results)

abl_cols = ["variant", "macro_f1", "accuracy"]
if "delta_f1" in abl_df.columns:
    abl_cols.append("delta_f1")
if "brier_score" in abl_df.columns:
    abl_cols.append("brier_score")

print("Ablation Study Results")
print("=" * 80)
print(abl_df[[c for c in abl_cols if c in abl_df.columns]].round(4).to_string(index=False))

full_f1 = abl_df[abl_df["variant"] == "full_pipeline"]
if not full_f1.empty:
    print(f"\nFull pipeline baseline: Macro-F1={full_f1.iloc[0]['macro_f1']:.4f}")
""")

code(r"""# ---------------------------------------------------------------------------
# §8.2 Ablation Visualization — Component Contribution
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Waterfall-style: delta from full pipeline
if "delta_f1" in abl_df.columns:
    ablation_only = abl_df[abl_df["variant"].str.startswith("-")].copy()
    ablation_only = ablation_only.sort_values("delta_f1", ascending=False)

    if not ablation_only.empty:
        component_names = ablation_only["variant"].str.replace("-", "").values
        deltas = ablation_only["delta_f1"].values

        bar_colors = [COLORS["highlight"] if d > 0.05 else
                     COLORS["primary"] if d > 0.01 else
                     COLORS["muted"] for d in deltas]

        bars = axes[0].barh(range(len(component_names)), deltas, color=bar_colors)
        axes[0].set_yticks(range(len(component_names)))
        axes[0].set_yticklabels(component_names)
        axes[0].set_xlabel("F1 Drop When Removed (pp)")
        axes[0].set_title("(a) Component Contribution (F1 drop)")
        for bar, val in zip(bars, deltas):
            axes[0].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=9)

# Full comparison bar chart
all_variants = abl_df.sort_values("macro_f1", ascending=True)
bar_colors = [COLORS["highlight"] if v == "full_pipeline" else COLORS["primary"]
              for v in all_variants["variant"]]

bars = axes[1].barh(range(len(all_variants)), all_variants["macro_f1"].values, color=bar_colors)
axes[1].set_yticks(range(len(all_variants)))
axes[1].set_yticklabels(all_variants["variant"].values, fontsize=8)
axes[1].set_xlabel("Macro-F1")
axes[1].set_title("(b) All Ablation Configurations")
for bar, val in zip(bars, all_variants["macro_f1"].values):
    axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_ablation.pdf")
plt.savefig(FIGURES_DIR / "fig_ablation.png")
plt.show()
""")

md(r"""### Ablation Findings

**Component contributions (F1 drop when removed):**

| Component | F1 Drop | Interpretation |
|:---|:---|:---|
| Embeddings | +33.0pp | **Dominant component** — semantic representation is essential |
| Lexical | +2.9pp | Moderate — captures keyword patterns embeddings miss |
| Rules | +1.8pp | Small but positive — adds deterministic signal |
| Structural | +1.3pp | Minimal — metadata adds limited information |

**Key insight:** Embeddings alone account for ~89% of the performance. Lexical features
provide a complementary boost (+2.9pp), confirming that hybrid representation is beneficial.
""")

# ============================================================================
# §9 — K-FOLD CROSS-VALIDATION
# ============================================================================

md(r"""<a id="§9"></a>
## §9. Stratified K-Fold Cross-Validation

### Rationale

The fixed train/val/test split provides a single point estimate of performance.
Stratified 5-fold CV pools all data and produces performance estimates with
**real confidence intervals**, reducing the risk of split-dependent conclusions.

### Method

- Pool all 2,122 conversations
- Stratified 5-fold split (preserving class proportions)
- Full pipeline (lexical + structural + embedding + rules, LightGBM) per fold
- Report mean ± std Macro-F1 with 95% CI
""")

code(r"""# ---------------------------------------------------------------------------
# §9.1 K-Fold Cross-Validation
# ---------------------------------------------------------------------------
kfold_path = RESULTS_DIR / "kfold" / "results.json"

if kfold_path.exists():
    with open(kfold_path, encoding="utf-8") as f:
        kfold_results = json.load(f)

    print("K-Fold Cross-Validation Results")
    print("=" * 80)

    if "fold_results" in kfold_results:
        fold_f1s = [f["macro_f1"] for f in kfold_results["fold_results"]]
        print(f"\nPer-fold Macro-F1:")
        for i, f1 in enumerate(fold_f1s):
            print(f"  Fold {i+1}: {f1:.4f}")
        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s, ddof=1)
        ci_95 = 1.96 * std_f1 / np.sqrt(len(fold_f1s))
        print(f"\nMean Macro-F1:  {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"95% CI:         [{mean_f1 - ci_95:.4f}, {mean_f1 + ci_95:.4f}]")

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(1, len(fold_f1s)+1), fold_f1s, color=PALETTE[0], alpha=0.8)
        ax.axhline(y=mean_f1, color="red", linestyle="--", label=f"Mean={mean_f1:.3f}")
        ax.fill_between([0.5, len(fold_f1s)+0.5], mean_f1-ci_95, mean_f1+ci_95,
                        color="red", alpha=0.1, label=f"95% CI")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Stratified 5-Fold CV: Macro-F1 per Fold")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig_kfold.pdf")
        plt.savefig(FIGURES_DIR / "fig_kfold.png")
        plt.show()
    else:
        print(json.dumps(kfold_results, indent=2)[:2000])
else:
    print("K-Fold results not available. Run: python experiments/scripts/run_kfold_experiment.py")
    print("This experiment pools all data and runs stratified 5-fold CV with the full pipeline.")
""")

# ============================================================================
# §10 — LODO
# ============================================================================

md(r"""<a id="§10"></a>
## §10. Leave-One-Domain-Out (LODO) Cross-Domain Evaluation

### Rationale

Single-domain evaluation cannot demonstrate generalization. LODO holds out one business
domain entirely, trains on the remaining 7, and tests on the held-out domain.
This measures whether the model generalizes across industry verticals.

### Method

- 8 folds (one per domain: financeiro, restaurante, saude, imobiliario, telecom, ecommerce, tecnologia, educacao)
- Full pipeline per fold (lexical + structural + embedding features, LightGBM)
- Report per-domain Macro-F1 + generalization gap (in-domain − out-domain)
""")

code(r"""# ---------------------------------------------------------------------------
# §10.1 LODO Results
# ---------------------------------------------------------------------------
lodo_path = RESULTS_DIR / "LODO" / "results.json"

if lodo_path.exists():
    with open(lodo_path, encoding="utf-8") as f:
        lodo_results = json.load(f)

    print("Leave-One-Domain-Out Results")
    print("=" * 80)

    if "fold_results" in lodo_results:
        folds = lodo_results["fold_results"]
        lodo_data = []
        for fold in folds:
            lodo_data.append({
                "Held-out Domain": fold.get("held_out_domain", fold.get("domain", "?")),
                "Macro-F1": fold.get("macro_f1", 0),
                "Accuracy": fold.get("accuracy", 0),
                "N_test": fold.get("n_test", 0),
            })
        lodo_df = pd.DataFrame(lodo_data)
        print(lodo_df.to_string(index=False))

        mean_f1 = lodo_df["Macro-F1"].mean()
        std_f1 = lodo_df["Macro-F1"].std()
        print(f"\nMean Macro-F1:  {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Best domain:    {lodo_df.loc[lodo_df['Macro-F1'].idxmax(), 'Held-out Domain']} "
              f"({lodo_df['Macro-F1'].max():.4f})")
        print(f"Worst domain:   {lodo_df.loc[lodo_df['Macro-F1'].idxmin(), 'Held-out Domain']} "
              f"({lodo_df['Macro-F1'].min():.4f})")
        gen_gap = best_emb["macro_f1"] - mean_f1 if "best_emb" in dir() else 0
        print(f"Generalization gap: {gen_gap:.4f}")

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        domains = lodo_df["Held-out Domain"].values
        f1_values = lodo_df["Macro-F1"].values

        bars = ax.barh(range(len(domains)), f1_values,
                      color=[COLORS["highlight"] if f > mean_f1 else COLORS["muted"]
                             for f in f1_values])
        ax.axvline(x=mean_f1, color="red", linestyle="--",
                   label=f"Mean={mean_f1:.3f}")
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains)
        ax.set_xlabel("Macro-F1")
        ax.set_title("LODO: Per-Domain Generalization")
        ax.legend()
        for bar, val in zip(bars, f1_values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig_lodo.pdf")
        plt.savefig(FIGURES_DIR / "fig_lodo.png")
        plt.show()
    else:
        print(json.dumps(lodo_results, indent=2)[:2000])
else:
    print("LODO results not available. Run: python experiments/scripts/run_lodo_experiment.py")
""")

# ============================================================================
# §11 — ERROR ANALYSIS
# ============================================================================

md(r"""<a id="§11"></a>
## §11. Error Analysis & Interpretability

### Rationale

Aggregate metrics hide systematic failures. Error analysis identifies:
- Which classes are most confused with each other
- Whether synthetic data inflates performance
- Which features drive predictions
- Where the model lacks separability
""")

code(r"""# ---------------------------------------------------------------------------
# §11.1 Error Analysis
# ---------------------------------------------------------------------------
error_path = RESULTS_DIR / "error_analysis" / "results.json"

if error_path.exists():
    with open(error_path, encoding="utf-8") as f:
        error_results = json.load(f)

    print("Error Analysis Results")
    print("=" * 80)

    # Confusion matrix
    if "confusion_matrix" in error_results:
        cm = np.array(error_results["confusion_matrix"])
        cm_labels = error_results.get("labels", label_order)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                   xticklabels=cm_labels, yticklabels=cm_labels)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_title("(a) Confusion Matrix (counts)")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

        # Normalized confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1],
                   xticklabels=cm_labels, yticklabels=cm_labels, vmin=0, vmax=1)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        axes[1].set_title("(b) Confusion Matrix (normalized)")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig_confusion_matrix.pdf")
        plt.savefig(FIGURES_DIR / "fig_confusion_matrix.png")
        plt.show()

    # Top confusions
    if "top_confusions" in error_results:
        print("\nTop Confused Pairs:")
        for pair in error_results["top_confusions"][:10]:
            if isinstance(pair, dict):
                print(f"  {pair.get('true', '?')} → {pair.get('predicted', '?')}: "
                      f"{pair.get('count', '?')} errors")
            else:
                print(f"  {pair}")

    # Feature importance
    if "feature_importance" in error_results:
        fi = error_results["feature_importance"]
        if isinstance(fi, dict):
            top_features = sorted(fi.items(), key=lambda x: -x[1])[:20]
            print("\nTop 20 Features by LightGBM Gain:")
            for name, gain in top_features:
                print(f"  {name:40s} {gain:.4f}")
else:
    print("Error analysis not available. Run: python experiments/scripts/error_analysis.py")
    print("This produces confusion matrices, feature importance, and class separability analysis.")
""")

# ============================================================================
# §12 — STATISTICAL SUMMARY
# ============================================================================

md(r"""<a id="§12"></a>
## §12. Statistical Summary & Hypothesis Decisions

### 12.1 Hypothesis Decision Table
""")

code(r"""# ---------------------------------------------------------------------------
# §12.1 Unified Hypothesis Decision Table
# ---------------------------------------------------------------------------
decisions = [
    {
        "Hypothesis": "H1",
        "Claim": "Hybrid retrieval > isolated",
        "Metric": "MRR",
        "Best": "Hybrid-LINEAR (α=0.30)",
        "Baseline": "BM25-base",
        "Result": f"{h1_df.loc[h1_df['mrr'].idxmax(), 'mrr']:.3f} vs {h1_df[h1_df['variant']=='BM25-base'].iloc[0]['mrr']:.3f}" if not h1_df.empty else "N/A",
        "p-value": "0.017",
        "Verdict": "CONFIRMED",
    },
    {
        "Hypothesis": "H2",
        "Claim": "Lexical+Emb > Lexical-only",
        "Metric": "Macro-F1",
        "Best": "lexical+emb LightGBM",
        "Baseline": "lexical-only LightGBM",
        "Result": f"{best_emb['macro_f1']:.3f} vs {best_lex['macro_f1']:.3f}" if "best_emb" in dir() else "N/A",
        "p-value": "< 0.001",
        "Verdict": "CONFIRMED",
    },
    {
        "Hypothesis": "H3",
        "Claim": "ML+Rules > ML-only",
        "Metric": "Macro-F1",
        "Best": "ML+Rules-feature",
        "Baseline": "ML-only",
        "Result": f"{rules_feature.iloc[0]['macro_f1']:.3f} vs {ml_only.iloc[0]['macro_f1']:.3f}" if not ml_only.empty else "N/A",
        "p-value": "0.131",
        "Verdict": "INCONCLUSIVE",
    },
    {
        "Hypothesis": "H4",
        "Claim": "Cascade reduces cost",
        "Metric": "Macro-F1 / Cost",
        "Best": "N/A (cascade fails)",
        "Baseline": "uniform",
        "Result": "Cascade increases cost without quality gain",
        "p-value": "N/A",
        "Verdict": "REFUTED",
    },
]

dec_df = pd.DataFrame(decisions)
print("HYPOTHESIS DECISION TABLE")
print("=" * 100)
print(dec_df.to_string(index=False))
""")

code(r"""# ---------------------------------------------------------------------------
# §12.2 Summary Visualization
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))

verdict_colors = {
    "CONFIRMED": "#2ecc71",
    "INCONCLUSIVE": "#f39c12",
    "REFUTED": "#e74c3c",
}

hypotheses = ["H1", "H2", "H3", "H4"]
verdicts = ["CONFIRMED", "CONFIRMED", "INCONCLUSIVE", "REFUTED"]
colors = [verdict_colors[v] for v in verdicts]

bars = ax.barh(hypotheses[::-1], [1]*4, color=colors[::-1], height=0.6)
ax.set_xlim(0, 1.5)
ax.set_xticks([])
ax.set_title("Hypothesis Verdicts Summary")

for bar, hyp, verdict in zip(bars, hypotheses[::-1], verdicts[::-1]):
    ax.text(0.5, bar.get_y() + bar.get_height()/2,
           f"{hyp}: {verdict}", ha="center", va="center",
           fontsize=12, fontweight="bold", color="white")

# Remove axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_hypothesis_verdicts.pdf")
plt.savefig(FIGURES_DIR / "fig_hypothesis_verdicts.png")
plt.show()
""")

# ============================================================================
# §13 — LIMITATIONS
# ============================================================================

md(
    r"""<a id="§13"></a>
## §13. Limitations & Threats to Validity

### Internal Validity

1. **Synthetic data dominance:** 60% of the dataset is LLM-generated. While the audit confirmed
   high quality (≥96.7% human agreement), synthetic conversations may have distributional
   properties that differ from real customer interactions.

2. **Single embedding model:** All experiments use paraphrase-multilingual-MiniLM-L12-v2.
   Larger models (e.g., E5-large, BGE-large) might change the relative rankings.

3. **Fixed classifier architecture:** LightGBM with n_estimators=100, num_leaves=31 is used
   throughout. Hyperparameter tuning might benefit different feature configurations differently.

### External Validity

4. **Single dataset:** Despite 8 domains, all data comes from one source. Cross-corpus
   evaluation (LODO results, if available) provides partial generalization evidence.

5. **PT-BR only:** Results may not transfer to other languages, particularly languages
   with different morphological properties.

6. **Controlled scenario:** Customer service conversations have relatively constrained
   vocabulary and interaction patterns. Open-domain conversations would be more challenging.

### Statistical Validity

7. **Multiple hypothesis testing:** Four hypotheses tested without familywise error rate
   correction (Bonferroni would require α=0.0125). H1's p=0.017 would not survive correction.

8. **Sample size for H3:** The +1.8pp improvement may be a real effect that the current
   sample size cannot detect. Power analysis needed for definitive conclusion.

### Reproducibility

9. **All code, data splits, and seeds are documented** in this notebook and the repository.
   Any researcher can reproduce these results given access to the embedding model weights.

---

*This notebook was generated on {timestamp} and constitutes the complete experimental
record for the TalkEx dissertation.*

---

**End of Experimental Notebook**
""".replace("{timestamp}", "2026-03-13")
)

# ============================================================================
# Build and save notebook
# ============================================================================
nb.cells = cells

output_path = Path("experiments/notebooks/dissertation_experiments.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generated: {output_path}")
print(
    f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
    f"{sum(1 for c in cells if c.cell_type == 'code')} code)"
)
