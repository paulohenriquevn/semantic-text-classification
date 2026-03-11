"""Visualization generator for experiment results.

Reads structured results from experiments/results/H{1,2,3,4}/
and generates publication-ready plots for the dissertation.

Usage:
    python experiments/scripts/plot_results.py --hypothesis H1
    python experiments/scripts/plot_results.py --hypothesis all
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")

# Consistent style for all plots
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)
sns.set_theme(style="whitegrid", palette="Set2")


def load_results(hypothesis: str) -> dict:
    """Load experiment results from JSON."""
    path = RESULTS_DIR / hypothesis / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}. Run experiments first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# H1: Retrieval plots
# ---------------------------------------------------------------------------


def plot_h1(data: dict, output_dir: Path) -> None:
    """Generate H1 (hybrid retrieval) plots."""
    variants = data["variants"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Recall@K curves ---
    fig, ax = plt.subplots()
    k_values = [5, 10, 20]
    for v in variants:
        name = v["variant_name"]
        recalls = [v["metrics"].get(f"recall@{k}", 0) for k in k_values]
        ax.plot(k_values, recalls, marker="o", label=name, linewidth=2)

    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("H1: Recall@K por Sistema de Retrieval")
    ax.legend(loc="lower right")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "h1_recall_at_k.png")
    plt.close(fig)
    logger.info("Saved h1_recall_at_k.png")

    # --- Bar chart: MRR comparison ---
    fig, ax = plt.subplots()
    names = [v["variant_name"] for v in variants]
    mrr_values = [v["metrics"].get("mrr", 0) for v in variants]
    colors = sns.color_palette("Set2", len(names))
    bars = ax.bar(names, mrr_values, color=colors)
    ax.set_ylabel("MRR")
    ax.set_title("H1: Mean Reciprocal Rank por Sistema")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, mrr_values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "h1_mrr_comparison.png")
    plt.close(fig)
    logger.info("Saved h1_mrr_comparison.png")


# ---------------------------------------------------------------------------
# H2: Classification plots
# ---------------------------------------------------------------------------


def plot_h2(data: dict, output_dir: Path) -> None:
    """Generate H2 (multi-level classification) plots."""
    variants = data["variants"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Bar chart: Macro-F1 by classifier ---
    fig, ax = plt.subplots()
    names = [v["variant_name"] for v in variants]
    f1_values = [v["metrics"].get("macro_f1", 0) for v in variants]
    colors = sns.color_palette("Set2", len(names))
    bars = ax.bar(names, f1_values, color=colors)
    ax.set_ylabel("Macro-F1")
    ax.set_title("H2: Macro-F1 por Representação x Classificador")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, f1_values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "h2_macro_f1_comparison.png")
    plt.close(fig)
    logger.info("Saved h2_macro_f1_comparison.png")

    # --- Heatmap: F1 per class ---
    # Extract per-label F1 from metrics
    all_labels = set()
    for v in variants:
        for k in v["metrics"]:
            if k.startswith("f1_"):
                all_labels.add(k[3:])

    if all_labels:
        labels = sorted(all_labels)
        matrix = []
        row_names = []
        for v in variants:
            row = [v["metrics"].get(f"f1_{label}", 0) for label in labels]
            matrix.append(row)
            row_names.append(v["variant_name"])

        fig, ax = plt.subplots(figsize=(12, max(4, len(row_names) * 0.6)))
        sns.heatmap(
            np.array(matrix),
            xticklabels=labels,
            yticklabels=row_names,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title("H2: F1 por Classe x Configuração")
        fig.tight_layout()
        fig.savefig(output_dir / "h2_f1_heatmap.png")
        plt.close(fig)
        logger.info("Saved h2_f1_heatmap.png")


# ---------------------------------------------------------------------------
# H4: Cascaded inference plots
# ---------------------------------------------------------------------------


def plot_h4(data: dict, output_dir: Path) -> None:
    """Generate H4 (cascaded inference) plots."""
    variants = data["variants"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Pareto curve: cost reduction vs F1 delta ---
    cascade_variants = [v for v in variants if v["variant_name"] != "uniform"]
    uniform = next((v for v in variants if v["variant_name"] == "uniform"), None)

    if cascade_variants and uniform:
        fig, ax = plt.subplots()
        for v in cascade_variants:
            cost_red = v["metrics"].get("cost_reduction_pct", 0)
            f1_delta = v["metrics"].get("f1_delta", 0)
            ax.scatter(cost_red, f1_delta, s=100, zorder=5)
            ax.annotate(
                v["variant_name"].replace("cascade_t", "t="),
                (cost_red, f1_delta),
                textcoords="offset points",
                xytext=(5, 5),
            )

        # Reference lines
        ax.axhline(y=0.02, color="red", linestyle="--", alpha=0.5, label="F1 delta limit (2%)")
        ax.axvline(x=40, color="green", linestyle="--", alpha=0.5, label="Cost reduction target (40%)")

        ax.set_xlabel("Redução de Custo (%)")
        ax.set_ylabel("Degradação F1 (pontos)")
        ax.set_title("H4: Curva de Pareto — Custo vs Qualidade")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "h4_pareto.png")
        plt.close(fig)
        logger.info("Saved h4_pareto.png")

    # --- Stacked bar: % resolved per stage ---
    fig, ax = plt.subplots()
    names = [v["variant_name"] for v in variants]
    stage1 = [v["metrics"].get("pct_stage1", 0) for v in variants]
    stage2 = [v["metrics"].get("pct_stage2", 0) for v in variants]

    x = np.arange(len(names))
    width = 0.6
    ax.bar(x, stage1, width, label="Estágio 1 (leve)", color=sns.color_palette("Set2")[0])
    ax.bar(x, stage2, width, bottom=stage1, label="Estágio 2 (completo)", color=sns.color_palette("Set2")[1])

    ax.set_ylabel("% Conversas")
    ax.set_title("H4: Distribuição por Estágio da Cascata")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "h4_stage_distribution.png")
    plt.close(fig)
    logger.info("Saved h4_stage_distribution.png")


# ---------------------------------------------------------------------------
# H3: Rules complement ML plots
# ---------------------------------------------------------------------------


def plot_h3(data: dict, output_dir: Path) -> None:
    """Generate H3 (rules complement ML) plots."""
    variants = data["variants"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Grouped bar chart: Precision vs Recall on critical classes ---
    critical_classes = ["cancelamento", "reclamacao"]
    fig, axes = plt.subplots(1, len(critical_classes), figsize=(14, 6))
    if len(critical_classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, critical_classes, strict=True):
        names = [v["variant_name"] for v in variants]
        precisions = [v["metrics"].get(f"precision_{cls}", 0) for v in variants]
        recalls = [v["metrics"].get(f"recall_{cls}", 0) for v in variants]

        x = np.arange(len(names))
        width = 0.35
        bars1 = ax.bar(x - width / 2, precisions, width, label="Precision", color=sns.color_palette("Set2")[0])
        bars2 = ax.bar(x + width / 2, recalls, width, label="Recall", color=sns.color_palette("Set2")[1])

        for bar_group in [bars1, bars2]:
            for bar in bar_group:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Score")
        ax.set_title(f"H3: {cls}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend()

    fig.suptitle("H3: Precision vs Recall em Classes Criticas", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "h3_critical_classes.png")
    plt.close(fig)
    logger.info("Saved h3_critical_classes.png")

    # --- Macro-F1 comparison bar chart ---
    fig, ax = plt.subplots()
    names = [v["variant_name"] for v in variants]
    f1_values = [v["metrics"].get("macro_f1", 0) for v in variants]
    colors = sns.color_palette("Set2", len(names))
    bars = ax.bar(names, f1_values, color=colors)
    ax.set_ylabel("Macro-F1")
    ax.set_title("H3: Macro-F1 por Configuracao")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, f1_values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "h3_macro_f1.png")
    plt.close(fig)
    logger.info("Saved h3_macro_f1.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--hypothesis", required=True, type=click.Choice(["H1", "H2", "H3", "H4", "all"]))
@click.option("--results-dir", default="experiments/results", help="Base results directory.")
def main(hypothesis: str, results_dir: str) -> None:
    """Generate plots for experiment results."""
    global RESULTS_DIR
    RESULTS_DIR = Path(results_dir)

    hypotheses = [hypothesis] if hypothesis != "all" else ["H1", "H2", "H3", "H4"]

    for h in hypotheses:
        try:
            data = load_results(h)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", h, e)
            continue

        output_dir = RESULTS_DIR / h / "plots"
        logger.info("Generating plots for %s...", h)

        if h == "H1":
            plot_h1(data, output_dir)
        elif h == "H2":
            plot_h2(data, output_dir)
        elif h == "H3":
            plot_h3(data, output_dir)
        elif h == "H4":
            plot_h4(data, output_dir)
        else:
            logger.warning("Plot generation not yet implemented for %s", h)

    logger.info("All plots generated!")


if __name__ == "__main__":
    main()
