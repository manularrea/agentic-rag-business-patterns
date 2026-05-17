from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ORDER = ["single_agent", "parent_child", "mixture_of_agents", "handoff"]
PALETTE = {
    "single_agent": "#4C78A8",
    "parent_child": "#59A14F",
    "mixture_of_agents": "#F28E2B",
    "handoff": "#B07AA1",
}


def _setup():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"


def create_all_figures(detailed: pd.DataFrame, aggregate: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup()
    plot_quality_radar(aggregate, output_dir / "quality_radar.png")
    plot_cost_latency(aggregate, output_dir / "cost_latency_tradeoff.png")
    plot_completeness_by_type(detailed, output_dir / "completeness_by_question_type.png")
    plot_hypothesis_heatmap(aggregate, output_dir / "hypothesis_evidence_heatmap.png")
    plot_balanced_index(aggregate, output_dir / "balanced_index.png")


def plot_quality_radar(aggregate: pd.DataFrame, path: Path) -> None:
    import numpy as np

    metrics = ["factuality", "clarity", "traceability", "completeness", "decision_accuracy"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = plt.subplot(111, polar=True)
    for _, row in aggregate.set_index("pattern").loc[ORDER].reset_index().iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, label=row["pattern"], linewidth=2.4, color=PALETTE[row["pattern"]])
        ax.fill(angles, values, alpha=0.08, color=PALETTE[row["pattern"]])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Índice de calidad por patrón RAG", pad=25, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=True)
    fig.savefig(path)
    plt.close(fig)


def plot_cost_latency(aggregate: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for _, row in aggregate.iterrows():
        ax.scatter(row["total_tokens"], row["latency_ms"], s=850 * row["quality_index"],
                   color=PALETTE[row["pattern"]], alpha=0.82, edgecolor="white", linewidth=1.8)
        ax.text(row["total_tokens"] + 3, row["latency_ms"] + 2, row["pattern"].replace("_", " "), fontsize=10)
    ax.set_xlabel("Tokens medios por respuesta")
    ax.set_ylabel("Latencia media estimada (ms)")
    ax.set_title("Frontera calidad-coste-latencia", weight="bold")
    ax.text(0.01, 0.01, "El tamaño de la burbuja representa el índice de calidad.", transform=ax.transAxes, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_completeness_by_type(detailed: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=detailed,
        x="question_type",
        y="completeness",
        hue="pattern",
        hue_order=ORDER,
        palette=PALETTE,
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Tipo de pregunta")
    ax.set_ylabel("Completitud media")
    ax.set_title("Completitud por tipo de tarea empresarial", weight="bold")
    ax.legend(title="Patrón", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def plot_hypothesis_heatmap(aggregate: pd.DataFrame, path: Path) -> None:
    cols = ["quality_index", "traceability", "completeness", "decision_accuracy", "efficiency_index", "balanced_index"]
    frame = aggregate.set_index("pattern").loc[ORDER, cols]
    fig, ax = plt.subplots(figsize=(11, 6.8))
    sns.heatmap(frame, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Evidencia cuantitativa para H1-H5", weight="bold")
    ax.set_xlabel("Dimensión experimental")
    ax.set_ylabel("Patrón")
    fig.savefig(path)
    plt.close(fig)


def plot_balanced_index(aggregate: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    frame = aggregate.sort_values("balanced_index", ascending=False)
    sns.barplot(data=frame, y="pattern", x="balanced_index", palette=PALETTE, hue="pattern", legend=False, ax=ax)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Índice balanceado")
    ax.set_ylabel("Patrón")
    ax.set_title("Equilibrio entre calidad, trazabilidad y coste", weight="bold")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=4)
    fig.savefig(path)
    plt.close(fig)
