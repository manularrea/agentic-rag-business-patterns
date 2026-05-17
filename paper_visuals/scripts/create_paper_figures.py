#!/usr/bin/env python3
"""Create publication-grade visual assets for the Agentic RAG benchmark.

The script reads the reproducible benchmark outputs from ``results/`` and
creates a dedicated scientific visual layer under ``paper_visuals/``. It also
exports mathematical cost and semantic-geometry tables that can be cited
directly in a paper.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
OUT = ROOT / "paper_visuals"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
PALETTE_PATH = OUT / "styles" / "paper_palette.json"
EPS = 1e-9

PATTERN_LABELS = {
    "single_agent": "Single-Agent\nRAG",
    "parent_child": "Parent-Child\nRAG",
    "mixture_of_agents": "Mixture-of-\nAgents",
    "handoff": "Handoff\nRAG",
}

METRIC_LABELS = {
    "quality_index": "Índice de calidad",
    "cost_composite": "Coste compuesto",
    "net_utility": "Utilidad neta",
    "quality_per_1k_tokens": "Calidad / 1k tokens",
    "quality_per_second": "Calidad / segundo",
    "token_efficiency_norm": "Eficiencia tokens",
    "latency_efficiency_norm": "Eficiencia temporal",
    "decision_accuracy": "Decisión segura",
}

QUESTION_TYPE_LABELS = {
    "simple": "Simple",
    "medium": "Complejidad media",
    "synthesis": "Síntesis documental",
    "ambiguous": "Ambigua",
    "sensitive": "Sensible",
}


def load_palette() -> dict[str, str]:
    with PALETTE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA5B1")
    ax.spines["bottom"].set_color("#9AA5B1")
    ax.grid(True, axis="y", color="#D9DEE8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, colors="#1F2933")


def minmax(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + EPS)


def compute_cost_metrics() -> pd.DataFrame:
    agg = pd.read_csv(RESULTS / "aggregate_metrics.csv").copy()
    agg["tokens_norm"] = minmax(agg["total_tokens"])
    agg["latency_norm"] = minmax(agg["latency_ms"])
    agg["trace_steps_norm"] = minmax(agg["trace_steps"])
    agg["cost_composite"] = (
        0.40 * agg["tokens_norm"]
        + 0.40 * agg["latency_norm"]
        + 0.20 * agg["trace_steps_norm"]
    )
    agg["net_utility"] = agg["quality_index"] - 0.35 * agg["cost_composite"]
    agg["quality_per_1k_tokens"] = agg["quality_index"] / (agg["total_tokens"] / 1000.0)
    agg["quality_per_second"] = agg["quality_index"] / (agg["latency_ms"] / 1000.0)
    agg["token_efficiency_norm"] = 1.0 - agg["tokens_norm"]
    agg["latency_efficiency_norm"] = 1.0 - agg["latency_norm"]
    agg["is_cost_quality_pareto"] = False
    for idx, row in agg.iterrows():
        dominated = False
        for jdx, other in agg.iterrows():
            if idx == jdx:
                continue
            better_or_equal = (
                other["quality_index"] >= row["quality_index"]
                and other["cost_composite"] <= row["cost_composite"]
            )
            strictly_better = (
                other["quality_index"] > row["quality_index"]
                or other["cost_composite"] < row["cost_composite"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        agg.loc[idx, "is_cost_quality_pareto"] = not dominated
    cols = [
        "pattern",
        "quality_index",
        "total_tokens",
        "latency_ms",
        "trace_steps",
        "tokens_norm",
        "latency_norm",
        "trace_steps_norm",
        "cost_composite",
        "net_utility",
        "quality_per_1k_tokens",
        "quality_per_second",
        "token_efficiency_norm",
        "latency_efficiency_norm",
        "decision_accuracy",
        "is_cost_quality_pareto",
    ]
    return agg[cols].sort_values("net_utility", ascending=False)


def save_cost_tables(cost: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    cost.to_csv(TABLES / "paper_cost_metrics.csv", index=False)
    rounded = cost.copy()
    numeric = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric] = rounded[numeric].round(4)
    rounded.to_markdown(TABLES / "paper_cost_metrics.md", index=False)


def plot_cost_quality_frontier(cost: pd.DataFrame, palette: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=160)
    for _, row in cost.iterrows():
        pattern = row["pattern"]
        color = palette[pattern]
        size = 430 if row["is_cost_quality_pareto"] else 280
        marker = "*" if row["is_cost_quality_pareto"] else "o"
        ax.scatter(
            row["cost_composite"],
            row["quality_index"],
            s=size,
            c=color,
            marker=marker,
            edgecolor="#111827",
            linewidth=1.2,
            alpha=0.95,
            zorder=3,
        )
        ax.annotate(
            PATTERN_LABELS[pattern].replace("\n", " "),
            (row["cost_composite"], row["quality_index"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#1F2933",
        )
    pareto = cost[cost["is_cost_quality_pareto"]].sort_values("cost_composite")
    if len(pareto) > 1:
        ax.plot(
            pareto["cost_composite"],
            pareto["quality_index"],
            color="#111827",
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            label="Frontera no dominada",
        )
    ax.set_title("Frontera calidad-coste para patrones Agentic RAG", fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("Coste operativo compuesto normalizado\n0.4·tokens + 0.4·latencia + 0.2·traza", fontsize=12)
    ax.set_ylabel("Índice de calidad experimental", fontsize=12)
    ax.set_xlim(-0.05, max(1.05, cost["cost_composite"].max() + 0.1))
    ax.set_ylim(max(0, cost["quality_index"].min() - 0.08), min(1.0, cost["quality_index"].max() + 0.08))
    style_axes(ax)
    ax.grid(True, axis="both", color="#D9DEE8", linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.text(0.01, 0.01, "Fuente: benchmark reproducible; tabla matemática en paper_visuals/tables/paper_cost_metrics.csv", fontsize=9, color="#52606D")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "paper_cost_quality_frontier.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_net_utility(cost: pd.DataFrame, palette: dict[str, str]) -> None:
    data = cost.sort_values("net_utility", ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=160)
    bars = ax.barh(
        [PATTERN_LABELS[p].replace("\n", " ") for p in data["pattern"]],
        data["net_utility"],
        color=[palette[p] for p in data["pattern"]],
        edgecolor="#111827",
        linewidth=0.7,
    )
    for bar, value in zip(bars, data["net_utility"]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Utilidad neta: calidad descontando coste operativo", fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("U = quality_index − 0.35·coste_compuesto", fontsize=12)
    ax.set_ylabel("")
    style_axes(ax)
    ax.grid(True, axis="x", color="#D9DEE8", linewidth=0.8, alpha=0.75)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_net_utility.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_cost_decomposition(cost: pd.DataFrame, palette: dict[str, str]) -> None:
    data = cost.sort_values("cost_composite", ascending=False)
    labels = [PATTERN_LABELS[p].replace("\n", " ") for p in data["pattern"]]
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=160)
    bottom = np.zeros(len(data))
    components = [
        ("tokens_norm", 0.40, "Tokens", "#9B2226"),
        ("latency_norm", 0.40, "Latencia", "#CA6702"),
        ("trace_steps_norm", 0.20, "Trazas", "#005F73"),
    ]
    for col, weight, label, color in components:
        values = data[col].to_numpy() * weight
        ax.bar(labels, values, bottom=bottom, label=f"{label} (w={weight:.1f})", color=color, alpha=0.88, edgecolor="white", linewidth=0.8)
        bottom += values
    for x, value in enumerate(data["cost_composite"]):
        ax.text(x, value + 0.025, f"C={value:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Descomposición del coste operativo compuesto", fontsize=18, fontweight="bold", pad=18)
    ax.set_ylabel("Contribución normalizada ponderada", fontsize=12)
    ax.set_ylim(0, max(1.08, data["cost_composite"].max() + 0.12))
    ax.tick_params(axis="x", rotation=0)
    style_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_cost_decomposition.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_hypothesis_matrix(palette: dict[str, str]) -> None:
    h = pd.read_csv(RESULTS / "hypothesis_support.csv")
    scores = [0.85 if "matiz" in f.lower() else 1.0 for f in h["finding"]]
    fig, ax = plt.subplots(figsize=(14.8, 8.0), dpi=160)
    ax.axis("off")
    ax.text(0.02, 0.97, "Matriz de evidencia para H1-H5", fontsize=22, fontweight="bold", transform=ax.transAxes, va="top", color="#1F2933")
    ax.text(0.89, 0.905, "Soporte", fontsize=11, fontweight="bold", transform=ax.transAxes, color="#1F2933", ha="center")
    y = 0.83
    for i, row in h.iterrows():
        color = "#F2A65A" if scores[i] < 1 else "#2E8B57"
        evidence = "\n".join(textwrap.wrap(str(row["evidence"]), width=105))
        ax.add_patch(plt.Rectangle((0.02, y - 0.065), 0.11, 0.075, color=color, transform=ax.transAxes, alpha=0.95))
        ax.text(0.075, y - 0.027, row["hypothesis"], ha="center", va="center", fontsize=17, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(0.16, y, row["finding"], fontsize=13, fontweight="bold", transform=ax.transAxes, color="#1F2933")
        ax.text(0.16, y - 0.041, evidence, fontsize=10.6, transform=ax.transAxes, color="#323F4B", va="top")
        ax.add_patch(plt.Rectangle((0.87, y - 0.060), 0.085 * scores[i], 0.035, color=color, transform=ax.transAxes, alpha=0.95))
        ax.text(0.965, y - 0.043, f"{scores[i]:.2f}", ha="right", fontsize=10.5, fontweight="bold", transform=ax.transAxes)
        y -= 0.145
    ax.text(0.02, 0.055, "Escala: 1.00 = soporte completo; 0.85 = soporte con matiz por trade-off coste/calidad.", fontsize=10, color="#52606D", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_hypothesis_matrix.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_quality_cost_radar(cost: pd.DataFrame, palette: dict[str, str]) -> None:
    metrics = ["quality_index", "token_efficiency_norm", "latency_efficiency_norm", "decision_accuracy", "net_utility"]
    data = cost.copy()
    data["net_utility"] = minmax(data["net_utility"])
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10.8, 9.6), dpi=160, subplot_kw={"polar": True})
    for _, row in data.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, color=palette[row["pattern"]], linewidth=2.4, label=PATTERN_LABELS[row["pattern"]].replace("\n", " "))
        ax.fill(angles, values, color=palette[row["pattern"]], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10, color="#52606D")
    ax.set_ylim(0, 1.0)
    ax.set_title("Perfil integrado: calidad, coste y seguridad de decisión", fontsize=16, fontweight="bold", pad=28, loc="center")
    ax.grid(color="#D9DEE8", linewidth=0.9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.13), frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0.03, 0.02, 0.88, 0.96))
    fig.savefig(FIGURES / "paper_quality_cost_radar.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_domain_heatmap(palette: dict[str, str]) -> None:
    by_type = pd.read_csv(RESULTS / "metrics_by_question_type.csv")
    pivot = by_type.pivot(index="question_type", columns="pattern", values="completeness")
    order_rows = [r for r in ["simple", "medium", "synthesis", "ambiguous", "sensitive"] if r in pivot.index]
    order_cols = ["single_agent", "parent_child", "mixture_of_agents", "handoff"]
    pivot = pivot.loc[order_rows, order_cols]
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=160)
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order_cols)))
    ax.set_xticklabels([PATTERN_LABELS[c].replace("\n", " ") for c in order_cols], fontsize=10.5)
    ax.set_yticks(range(len(order_rows)))
    ax.set_yticklabels([QUESTION_TYPE_LABELS.get(r, r.capitalize()) for r in order_rows], fontsize=11)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            color = "white" if value > 0.65 else "#1F2933"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    ax.set_title("Completitud por tipo de tarea y patrón RAG", fontsize=18, fontweight="bold", pad=18)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Completitud", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_task_type_completeness_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_embedding_records() -> pd.DataFrame:
    answers = pd.read_csv(RESULTS / "answers.csv")
    metrics = pd.read_csv(RESULTS / "detailed_metrics.csv")
    with (DATA / "questions.json").open("r", encoding="utf-8") as fh:
        questions = pd.DataFrame(json.load(fh))
    questions = questions.rename(columns={"id": "question_id"})
    df = answers.merge(questions[["question_id", "question", "question_type", "risk", "domains", "expected_decision"]], on="question_id", how="left")
    metric_cols = [
        "question_id",
        "pattern",
        "factuality",
        "clarity",
        "traceability",
        "completeness",
        "decision_accuracy",
        "hallucination_rate",
        "latency_ms",
        "total_tokens",
        "trace_steps",
    ]
    df = df.merge(metrics[metric_cols], on=["question_id", "pattern"], how="left")
    df["domains_text"] = df["domains"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    df["semantic_text"] = (
        "Pregunta: " + df["question"].fillna("")
        + " | Tipo: " + df["question_type"].fillna("")
        + " | Riesgo: " + df["risk"].fillna("")
        + " | Dominios: " + df["domains_text"].fillna("")
        + " | Patrón: " + df["pattern"].fillna("")
        + " | Decisión: " + df["decision"].fillna("")
        + " | Respuesta: " + df["content"].fillna("")
        + " | Evidencia: " + df["fact_ids"].fillna("")
        + " | Citas: " + df["citations"].fillna("")
        + " | Trazabilidad: " + df["trace"].fillna("")
    )
    return df


def compute_semantic_geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_embedding_records()
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=768,
    )
    x = vectorizer.fit_transform(df["semantic_text"])
    x_norm = normalize(x, norm="l2", copy=True)
    svd = TruncatedSVD(n_components=3, random_state=42)
    coords = svd.fit_transform(x_norm)
    df["embedding_x"] = coords[:, 0]
    df["embedding_y"] = coords[:, 1]
    df["embedding_z"] = coords[:, 2]

    dense = x_norm.toarray()
    df["semantic_shift_from_single"] = np.nan
    baseline_index = {
        row.question_id: idx for idx, row in df[df["pattern"] == "single_agent"].iterrows()
    }
    for idx, row in df.iterrows():
        base_idx = baseline_index.get(row["question_id"])
        if base_idx is not None:
            distance = pairwise_distances(dense[idx : idx + 1], dense[base_idx : base_idx + 1], metric="cosine")[0, 0]
            df.loc[idx, "semantic_shift_from_single"] = distance

    centroids = {}
    for pattern in PATTERN_LABELS:
        idx = df.index[df["pattern"] == pattern].tolist()
        centroids[pattern] = dense[idx].mean(axis=0)
    centroid_matrix = np.vstack([centroids[p] for p in PATTERN_LABELS])
    centroid_dist = pairwise_distances(centroid_matrix, metric="cosine")

    rows: list[dict[str, float | str]] = []
    for i, pattern in enumerate(PATTERN_LABELS):
        idx = df.index[df["pattern"] == pattern].tolist()
        centroid = centroids[pattern].reshape(1, -1)
        compactness = float(pairwise_distances(dense[idx], centroid, metric="cosine").mean())
        separation = float(np.delete(centroid_dist[i], i).mean())
        shift = float(df.loc[df["pattern"] == pattern, "semantic_shift_from_single"].fillna(0).mean())
        quality = float(df.loc[df["pattern"] == pattern, ["factuality", "clarity", "traceability", "completeness", "decision_accuracy"]].mean(axis=1).mean())
        safe_decision = float(df.loc[df["pattern"] == pattern, "decision_accuracy"].mean())
        rows.append(
            {
                "pattern": pattern,
                "semantic_compactness": compactness,
                "semantic_separation": separation,
                "semantic_shift_from_single": shift,
                "semantic_separation_ratio": separation / (compactness + EPS),
                "mean_quality_local": quality,
                "decision_accuracy": safe_decision,
                "svd_explained_variance_ratio_3d": float(svd.explained_variance_ratio_.sum()),
                "n_points": len(idx),
            }
        )
    geometry = pd.DataFrame(rows).sort_values("semantic_separation_ratio", ascending=False)
    return df, geometry


def save_semantic_geometry_tables(embeddings: pd.DataFrame, geometry: pd.DataFrame) -> None:
    export_cols = [
        "pattern",
        "question_id",
        "question_type",
        "risk",
        "decision",
        "embedding_x",
        "embedding_y",
        "embedding_z",
        "semantic_shift_from_single",
        "factuality",
        "clarity",
        "traceability",
        "completeness",
        "decision_accuracy",
        "latency_ms",
        "total_tokens",
        "trace_steps",
    ]
    embeddings[export_cols].to_csv(TABLES / "paper_semantic_embeddings_3d.csv", index=False)
    geometry.to_csv(TABLES / "paper_semantic_geometry_metrics.csv", index=False)
    rounded = geometry.copy()
    numeric = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric] = rounded[numeric].round(4)
    rounded.to_markdown(TABLES / "paper_semantic_geometry_metrics.md", index=False)


def plot_semantic_embedding_3d(embeddings: pd.DataFrame, geometry: pd.DataFrame, palette: dict[str, str]) -> None:
    fig = plt.figure(figsize=(13.5, 9.2), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    markers = {"simple": "o", "medium": "s", "synthesis": "^", "ambiguous": "D", "sensitive": "X"}
    for pattern, group in embeddings.groupby("pattern"):
        for qtype, subgroup in group.groupby("question_type"):
            ax.scatter(
                subgroup["embedding_x"],
                subgroup["embedding_y"],
                subgroup["embedding_z"],
                s=85 + 130 * subgroup["trace_steps"].fillna(1).to_numpy() / max(1, embeddings["trace_steps"].max()),
                c=palette[pattern],
                marker=markers.get(qtype, "o"),
                edgecolor="#111827",
                linewidth=0.55,
                alpha=0.82,
            )
        centroid = group[["embedding_x", "embedding_y", "embedding_z"]].mean().to_numpy()
        ax.scatter(*centroid, s=520, c=palette[pattern], marker="*", edgecolor="#111827", linewidth=1.4, alpha=1.0)
        ax.text(*(centroid + np.array([0.01, 0.01, 0.01])), PATTERN_LABELS[pattern].replace("\n", " "), fontsize=9.5, fontweight="bold", color="#111827")

    ax.set_title("Geometría semántica 3D de respuestas por patrón Agentic RAG", fontsize=17, fontweight="bold", pad=24)
    ax.set_xlabel("SVD-1: eje semántico dominante", labelpad=12)
    ax.set_ylabel("SVD-2: especialización / dominio", labelpad=12)
    ax.set_zlabel("SVD-3: decisión / trazabilidad", labelpad=12)
    ax.view_init(elev=23, azim=38)
    ax.grid(True, color="#D9DEE8")
    ax.xaxis.pane.set_facecolor((0.98, 0.99, 1.0, 0.85))
    ax.yaxis.pane.set_facecolor((0.98, 0.99, 1.0, 0.85))
    ax.zaxis.pane.set_facecolor((0.98, 0.99, 1.0, 0.85))

    pattern_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[p], markeredgecolor="#111827", markersize=10, label=PATTERN_LABELS[p].replace("\n", " "))
        for p in PATTERN_LABELS
    ]
    type_handles = [
        plt.Line2D([0], [0], marker=m, color="#111827", linestyle="None", markerfacecolor="white", markersize=8, label=QUESTION_TYPE_LABELS.get(t, t))
        for t, m in markers.items()
    ]
    first = ax.legend(handles=pattern_handles, title="Patrón", loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=9.5, title_fontsize=10.5)
    ax.add_artist(first)
    ax.legend(handles=type_handles, title="Tipo de tarea", loc="lower left", bbox_to_anchor=(0.02, 0.08), frameon=False, fontsize=9.0, title_fontsize=10.0)

    top = geometry.sort_values("semantic_separation_ratio", ascending=False).iloc[0]
    note = (
        f"Mayor separación/compacidad: {PATTERN_LABELS[top['pattern']].replace(chr(10), ' ')} "
        f"(S/C={top['semantic_separation_ratio']:.2f}).\n"
        "Tamaño del punto ∝ pasos de traza; estrella = centroide del patrón."
    )
    fig.text(0.55, 0.055, note, fontsize=10.2, color="#323F4B", bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#CBD2D9", "alpha": 0.95})
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGURES / "paper_semantic_embedding_3d.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_semantic_geometry_bars(geometry: pd.DataFrame, palette: dict[str, str]) -> None:
    data = geometry.sort_values("semantic_separation_ratio", ascending=True)
    labels = [PATTERN_LABELS[p].replace("\n", " ") for p in data["pattern"]]
    fig, ax = plt.subplots(figsize=(12.4, 6.9), dpi=160)
    bars = ax.barh(labels, data["semantic_separation_ratio"], color=[palette[p] for p in data["pattern"]], edgecolor="#111827", linewidth=0.7)
    for bar, value in zip(bars, data["semantic_separation_ratio"]):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Separación geométrica semántica por patrón", fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("S/C = distancia media entre centroides / dispersión intra-patrón", fontsize=12)
    ax.set_ylabel("")
    style_axes(ax)
    ax.grid(True, axis="x", color="#D9DEE8", linewidth=0.8, alpha=0.75)
    ax.grid(False, axis="y")
    fig.text(0.01, 0.015, "Embeddings deterministas TF-IDF n-gram + SVD 3D sobre preguntas, respuestas, decisiones, citas y trazas reales del benchmark.", fontsize=9.2, color="#52606D")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "paper_semantic_geometry_metrics.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    palette = load_palette()
    cost = compute_cost_metrics()
    save_cost_tables(cost)
    plot_cost_quality_frontier(cost, palette)
    plot_net_utility(cost, palette)
    plot_cost_decomposition(cost, palette)
    plot_hypothesis_matrix(palette)
    plot_quality_cost_radar(cost, palette)
    plot_domain_heatmap(palette)
    embeddings, geometry = compute_semantic_geometry()
    save_semantic_geometry_tables(embeddings, geometry)
    plot_semantic_embedding_3d(embeddings, geometry, palette)
    plot_semantic_geometry_bars(geometry, palette)
    print("Generated paper visuals:")
    for path in sorted(FIGURES.glob("*.png")):
        print(f"- {path.relative_to(ROOT)}")
    print("Generated paper tables:")
    for path in sorted(TABLES.glob("paper_*")):
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
