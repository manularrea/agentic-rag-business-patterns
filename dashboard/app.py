"""Streamlit dashboard for Agentic RAG experiment analysis.

Run from the repository root with:

    streamlit run dashboard/app.py

The app is intentionally self-contained so it can consume the reproducible CSV/JSON
outputs already committed in this repository while also accepting user-uploaded runs.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PAPER_TABLES_DIR = ROOT / "paper_visuals" / "tables"

PATTERN_ORDER = ["single_agent", "parent_child", "mixture_of_agents", "handoff"]
PATTERN_LABELS = {
    "single_agent": "Single-Agent RAG",
    "parent_child": "Parent-Child",
    "mixture_of_agents": "Mixture-of-Agents",
    "handoff": "Handoff",
}
PATTERN_COLORS = {
    "single_agent": "#7dd3fc",
    "parent_child": "#a78bfa",
    "mixture_of_agents": "#f59e0b",
    "handoff": "#34d399",
}

REQUIRED_COLUMNS = {
    "question_id",
    "pattern",
    "factuality",
    "clarity",
    "traceability",
    "completeness",
    "hallucination_rate",
    "decision_accuracy",
    "latency_ms",
    "total_tokens",
}

QUALITY_COLUMNS = ["factuality", "clarity", "traceability", "completeness", "decision_accuracy"]
GROUNDING_COLUMNS = ["traceability", "completeness", "decision_accuracy"]
EFFICIENCY_COLUMNS = ["latency_ms", "total_tokens", "usd_cost", "cost_composite"]


st.set_page_config(
    page_title="Agentic RAG Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.metric-card {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 18px;
    padding: 1rem;
    background: linear-gradient(145deg, rgba(15, 23, 42, .92), rgba(30, 41, 59, .72));
    box-shadow: 0 20px 50px rgba(2, 6, 23, 0.20);
}
.small-muted {color: #94a3b8; font-size: .88rem;}
.figure-note {color: #cbd5e1; font-size: .92rem; line-height: 1.45;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_default_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detailed = pd.read_csv(RESULTS_DIR / "detailed_metrics.csv")
    answers = pd.read_csv(RESULTS_DIR / "answers.csv")
    cost = pd.read_csv(PAPER_TABLES_DIR / "paper_cost_metrics.csv")
    embeddings = pd.read_csv(PAPER_TABLES_DIR / "paper_semantic_embeddings_3d.csv")
    return detailed, answers, cost, embeddings


def _load_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix == ".json":
        raw = json.load(uploaded_file)
        if isinstance(raw, dict):
            for key in ("records", "data", "results", "metrics"):
                if key in raw and isinstance(raw[key], list):
                    return pd.DataFrame(raw[key])
            return pd.json_normalize(raw)
        return pd.DataFrame(raw)
    raise ValueError("Formato no soportado. Use CSV o JSON.")


def _safe_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_v = values.min()
    max_v = values.max()
    if pd.isna(min_v) or pd.isna(max_v) or np.isclose(max_v, min_v):
        normalized = pd.Series(np.ones(len(values)), index=series.index, dtype=float)
    else:
        normalized = (values - min_v) / (max_v - min_v)
    return 1 - normalized if invert else normalized


def validate_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    return len(missing) == 0, missing


def ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "question_type" not in out.columns:
        out["question_type"] = "unknown"
    if "risk" not in out.columns:
        out["risk"] = "unknown"
    if "trace_steps" not in out.columns:
        out["trace_steps"] = 1
    if "prompt_tokens" not in out.columns:
        out["prompt_tokens"] = out.get("total_tokens", 0) * 0.65
    if "completion_tokens" not in out.columns:
        out["completion_tokens"] = out.get("total_tokens", 0) * 0.35
    if "citations" not in out.columns:
        out["citations"] = 0
    if "faithfulness" not in out.columns:
        out["faithfulness"] = out.get("traceability", 0)
    if "context_precision" not in out.columns:
        out["context_precision"] = out.get("traceability", 0)
    if "context_recall" not in out.columns:
        out["context_recall"] = out.get("completeness", 0)
    if "answer_relevancy" not in out.columns:
        out["answer_relevancy"] = out[[c for c in ["clarity", "completeness"] if c in out]].mean(axis=1)
    if "quality_index" not in out.columns:
        cols = [c for c in QUALITY_COLUMNS if c in out]
        out["quality_index"] = out[cols].mean(axis=1)
    return out


def derive_cost_metrics(
    df: pd.DataFrame,
    input_price_per_1k: float,
    output_price_per_1k: float,
    w_tokens: float,
    w_latency: float,
    w_trace: float,
    utility_lambda: float,
) -> pd.DataFrame:
    out = ensure_optional_columns(df)
    out["prompt_tokens"] = pd.to_numeric(out["prompt_tokens"], errors="coerce").fillna(0)
    out["completion_tokens"] = pd.to_numeric(out["completion_tokens"], errors="coerce").fillna(0)
    out["total_tokens"] = pd.to_numeric(out["total_tokens"], errors="coerce").fillna(0)
    out["latency_ms"] = pd.to_numeric(out["latency_ms"], errors="coerce").fillna(0)
    out["trace_steps"] = pd.to_numeric(out["trace_steps"], errors="coerce").fillna(1)
    out["usd_cost"] = (out["prompt_tokens"] / 1000 * input_price_per_1k) + (
        out["completion_tokens"] / 1000 * output_price_per_1k
    )
    out["tokens_norm"] = _safe_normalize(out["total_tokens"])
    out["latency_norm"] = _safe_normalize(out["latency_ms"])
    out["trace_norm"] = _safe_normalize(out["trace_steps"])
    weight_sum = max(w_tokens + w_latency + w_trace, 1e-9)
    out["cost_composite"] = (
        w_tokens * out["tokens_norm"]
        + w_latency * out["latency_norm"]
        + w_trace * out["trace_norm"]
    ) / weight_sum
    out["net_utility"] = out["quality_index"] - utility_lambda * out["cost_composite"]
    out["quality_per_1k_tokens"] = out["quality_index"] / (out["total_tokens"].replace(0, np.nan) / 1000)
    out["quality_per_second"] = out["quality_index"] / (out["latency_ms"].replace(0, np.nan) / 1000)
    return out


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "quality_index",
        "factuality",
        "clarity",
        "traceability",
        "completeness",
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
        "hallucination_rate",
        "decision_accuracy",
        "latency_ms",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "usd_cost",
        "cost_composite",
        "net_utility",
        "trace_steps",
        "citations",
    ]
    present = [m for m in metrics if m in df.columns]
    agg = df.groupby("pattern", as_index=False)[present].mean(numeric_only=True)
    agg["pattern_label"] = agg["pattern"].map(PATTERN_LABELS).fillna(agg["pattern"])
    return agg.sort_values("pattern", key=lambda s: s.map({p: i for i, p in enumerate(PATTERN_ORDER)}).fillna(99))


def fig_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.20)",
        font={"family": "Inter, Arial, sans-serif", "size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 25, "t": 70, "b": 45},
    )
    return fig


def pattern_color_sequence(patterns: Iterable[str]) -> list[str]:
    return [PATTERN_COLORS.get(p, "#e5e7eb") for p in patterns]


def plot_grouped_quality(agg: pd.DataFrame, metrics: list[str]) -> go.Figure:
    long_df = agg.melt(id_vars=["pattern", "pattern_label"], value_vars=metrics, var_name="metric", value_name="score")
    fig = px.bar(
        long_df,
        x="metric",
        y="score",
        color="pattern_label",
        barmode="group",
        color_discrete_sequence=pattern_color_sequence(agg["pattern"].tolist()),
        text_auto=".2f",
    )
    fig.update_yaxes(range=[0, max(1.05, float(long_df["score"].max()) * 1.12)])
    return fig_layout(fig, "Calidad, grounding y fidelidad por patrón")


def plot_efficiency(agg: pd.DataFrame) -> go.Figure:
    metrics = [m for m in EFFICIENCY_COLUMNS if m in agg]
    normalized = agg[["pattern", "pattern_label", *metrics]].copy()
    for metric in metrics:
        normalized[f"{metric}_norm"] = _safe_normalize(normalized[metric])
    long_df = normalized.melt(
        id_vars=["pattern", "pattern_label"],
        value_vars=[f"{m}_norm" for m in metrics],
        var_name="metric",
        value_name="normalized_cost",
    )
    long_df["metric"] = long_df["metric"].str.replace("_norm", "", regex=False)
    fig = px.bar(
        long_df,
        x="metric",
        y="normalized_cost",
        color="pattern_label",
        barmode="group",
        color_discrete_sequence=pattern_color_sequence(agg["pattern"].tolist()),
        text_auto=".2f",
    )
    fig.update_yaxes(title="Coste normalizado", range=[0, 1.08])
    return fig_layout(fig, "Eficiencia operacional normalizada")


def plot_quality_cost(df: pd.DataFrame, x_axis: str, y_axis: str, size_axis: str) -> go.Figure:
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        size=size_axis if size_axis in df else None,
        color="pattern",
        symbol="question_type",
        hover_data=["question_id", "question_type", "risk", "latency_ms", "total_tokens", "net_utility"],
        color_discrete_map=PATTERN_COLORS,
        labels={"pattern": "Patrón"},
    )
    return fig_layout(fig, f"Trade-off interactivo: {y_axis} vs {x_axis}")


def plot_latency_grounding(df: pd.DataFrame) -> go.Figure:
    y_col = "faithfulness" if "faithfulness" in df else "traceability"
    fig = px.scatter(
        df,
        x="latency_ms",
        y=y_col,
        color="pattern",
        symbol="question_type",
        hover_data=["question_id", "decision_accuracy", "total_tokens", "usd_cost"],
        color_discrete_map=PATTERN_COLORS,
    )
    return fig_layout(fig, "Latencia vs grounding/faithfulness")


def plot_radar(agg: pd.DataFrame, selected_metrics: list[str]) -> go.Figure:
    fig = go.Figure()
    radar = agg.copy()
    for metric in selected_metrics:
        invert = metric in {"latency_ms", "total_tokens", "usd_cost", "cost_composite", "hallucination_rate"}
        radar[f"{metric}_radar"] = _safe_normalize(radar[metric], invert=invert)
    theta = selected_metrics + [selected_metrics[0]]
    for _, row in radar.iterrows():
        values = [row[f"{m}_radar"] for m in selected_metrics]
        values = values + [values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                name=PATTERN_LABELS.get(row["pattern"], row["pattern"]),
                line={"color": PATTERN_COLORS.get(row["pattern"], "#e5e7eb"), "width": 3},
                opacity=0.82,
            )
        )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
    )
    return fig_layout(fig, "Radar multi-métrica normalizado")


def plot_box(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.box(
        df,
        x="pattern",
        y=metric,
        color="pattern",
        points="all",
        color_discrete_map=PATTERN_COLORS,
        labels={"pattern": "Patrón"},
    )
    fig.update_xaxes(categoryorder="array", categoryarray=PATTERN_ORDER)
    return fig_layout(fig, f"Distribución de {metric} por patrón")


def plot_3d(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_3d(
        df,
        x="cost_composite",
        y="quality_index",
        z="latency_ms",
        color="pattern",
        symbol="question_type",
        hover_data=["question_id", "total_tokens", "net_utility", "risk"],
        color_discrete_map=PATTERN_COLORS,
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.86})
    return fig_layout(fig, "Espacio 3D: calidad, coste compuesto y latencia")


def plot_parallel(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    local = df.copy()
    local["pattern_code"] = local["pattern"].astype("category").cat.codes
    dimensions = [dict(label=col, values=local[col]) for col in cols]
    fig = go.Figure(
        data=go.Parcoords(
            line={
                "color": local["pattern_code"],
                "colorscale": [[0, "#7dd3fc"], [0.33, "#a78bfa"], [0.66, "#f59e0b"], [1, "#34d399"]],
                "showscale": False,
            },
            dimensions=dimensions,
        )
    )
    return fig_layout(fig, "Coordenadas paralelas para análisis multiobjetivo")


def camera_eye_from_angles(azimuth_degrees: float, elevation_degrees: float, zoom: float) -> dict[str, float]:
    """Convert human-readable 3D camera controls into Plotly scene camera coordinates."""
    azimuth = np.deg2rad(float(azimuth_degrees))
    elevation = np.deg2rad(float(elevation_degrees))
    radius = max(float(zoom), 0.25)
    return {
        "x": radius * np.cos(elevation) * np.cos(azimuth),
        "y": radius * np.cos(elevation) * np.sin(azimuth),
        "z": radius * np.sin(elevation),
    }



def plot_embeddings(embeddings: pd.DataFrame, color_by: str, camera_eye: dict[str, float] | None = None) -> go.Figure:
    color_col = color_by if color_by in embeddings else "pattern"
    fig = px.scatter_3d(
        embeddings,
        x="embedding_x",
        y="embedding_y",
        z="embedding_z",
        color=color_col,
        symbol="pattern" if color_col != "pattern" else "question_type",
        hover_data=["pattern", "question_id", "question_type", "risk", "decision", "total_tokens"],
        color_discrete_map=PATTERN_COLORS if color_col == "pattern" else None,
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.88})
    fig = fig_layout(fig, "Explorador 3D de embeddings semánticos")
    fig.update_layout(
        dragmode="turntable",
        uirevision="semantic_embeddings_camera",
        scene={
            "xaxis_title": "embedding_x",
            "yaxis_title": "embedding_y",
            "zaxis_title": "embedding_z",
            "aspectmode": "cube",
            "camera": {
                "eye": camera_eye or {"x": 1.35, "y": 1.35, "z": 0.95},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
    )
    return fig


def plot_retrieval_heatmap(df: pd.DataFrame, answers: pd.DataFrame) -> go.Figure:
    if answers.empty or "citations" not in answers.columns:
        heat = df.groupby(["question_type", "pattern"]).size().reset_index(name="retrieval_frequency")
    else:
        merged = df[["pattern", "question_id", "question_type"]].merge(
            answers[["pattern", "question_id", "citations"]], on=["pattern", "question_id"], how="left"
        )
        rows = []
        for _, row in merged.iterrows():
            raw = str(row.get("citations", ""))
            docs = [token.strip(" []'\"") for token in raw.replace(";", ",").split(",") if token.strip(" []'\"")]
            if not docs:
                docs = ["sin_cita"]
            for doc in docs:
                rows.append({"question_type": row["question_type"], "pattern": row["pattern"], "document": doc})
        exploded = pd.DataFrame(rows)
        heat = exploded.groupby(["question_type", "document"]).size().reset_index(name="retrieval_frequency")
        pivot = heat.pivot(index="question_type", columns="document", values="retrieval_frequency").fillna(0)
        fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        return fig_layout(fig, "Mapa de calor de documentos citados por tipo de pregunta")
    pivot = heat.pivot(index="question_type", columns="pattern", values="retrieval_frequency").fillna(0)
    fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
    return fig_layout(fig, "Mapa de calor de recuperación/cobertura")


def download_plotly_html(fig: go.Figure, filename: str) -> None:
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button(
        "Descargar figura como HTML interactivo",
        data=html,
        file_name=filename,
        mime="text/html",
        use_container_width=True,
    )


def _decode_plotly_typed_array(value):
    """Convert Plotly.py typed-array JSON objects into plain arrays for browser Plotly.js."""
    if isinstance(value, dict) and "bdata" in value and "dtype" in value:
        dtype_map = {
            "f8": np.float64,
            "f4": np.float32,
            "i1": np.int8,
            "i2": np.int16,
            "i4": np.int32,
            "i8": np.int64,
            "u1": np.uint8,
            "u2": np.uint16,
            "u4": np.uint32,
            "u8": np.uint64,
        }
        dtype = dtype_map.get(str(value["dtype"]).lower())
        if dtype is None:
            return value
        decoded = np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtype)
        if "shape" in value:
            decoded = decoded.reshape(value["shape"])
        return decoded.tolist()
    if isinstance(value, dict):
        return {key: _decode_plotly_typed_array(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, list):
        return [_decode_plotly_typed_array(item) for item in value]
    return value


def _plotly_figure_spec_for_browser(fig: go.Figure) -> str:
    """Serialize a Plotly figure in a format reliably consumed by Plotly.js in HTML components."""
    browser_spec = _decode_plotly_typed_array(fig.to_plotly_json())
    return json.dumps(browser_spec, separators=(",", ":")).replace("</", "<\\/")


def render_realtime_embedding_controller(fig: go.Figure, height: int = 760) -> None:
    """Render a self-contained Plotly + webcam controller for real-time 3D navigation."""
    realtime_fig = go.Figure(fig)

    # Plotly.js can render the semantic coordinates too compressed in an embedded
    # iframe because the original PCA-like X axis has a much narrower range than
    # Y and Z. Normalize only the browser-facing coordinates so the cloud is
    # visible from the first paint while preserving the original dashboard data.
    scatter_traces = [trace for trace in realtime_fig.data if getattr(trace, "type", None) == "scatter3d"]
    axis_values: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for trace in scatter_traces:
        for axis in axis_values:
            values = getattr(trace, axis, None)
            if values is not None:
                axis_values[axis].extend([float(value) for value in values])
    axis_stats: dict[str, tuple[float, float]] = {}
    for axis, values in axis_values.items():
        series = np.asarray(values, dtype=float)
        center = float(np.nanmean(series)) if series.size else 0.0
        spread = float(np.nanstd(series)) if series.size else 1.0
        axis_stats[axis] = (center, spread if spread > 1e-9 else 1.0)
    for trace in scatter_traces:
        for axis, (center, spread) in axis_stats.items():
            values = getattr(trace, axis, None)
            if values is not None:
                normalized = ((np.asarray(values, dtype=float) - center) / spread).round(6).tolist()
                setattr(trace, axis, normalized)

    if scatter_traces:
        merged_x: list[float] = []
        merged_y: list[float] = []
        merged_z: list[float] = []
        merged_colors: list[str] = []
        merged_text: list[str] = []
        for trace in scatter_traces:
            xs = list(getattr(trace, "x", []) or [])
            ys = list(getattr(trace, "y", []) or [])
            zs = list(getattr(trace, "z", []) or [])
            count = min(len(xs), len(ys), len(zs))
            marker = getattr(trace, "marker", None)
            color = getattr(marker, "color", "#e5e7eb") if marker is not None else "#e5e7eb"
            if isinstance(color, (list, tuple, np.ndarray)):
                colors = [str(item) for item in list(color)[:count]]
            else:
                colors = [str(color)] * count
            customdata = getattr(trace, "customdata", None)
            trace_name = str(getattr(trace, "name", "embedding"))
            for index in range(count):
                merged_x.append(float(xs[index]))
                merged_y.append(float(ys[index]))
                merged_z.append(float(zs[index]))
                merged_colors.append(colors[index] if index < len(colors) else "#e5e7eb")
                if customdata is not None and index < len(customdata):
                    row = list(customdata[index])
                    label = " · ".join(str(value) for value in row[:4] if str(value))
                else:
                    label = trace_name
                merged_text.append(label)
        realtime_fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=merged_x,
                    y=merged_y,
                    z=merged_z,
                    mode="markers",
                    text=merged_text,
                    hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
                    marker={
                        "color": merged_colors,
                        "size": 8,
                        "opacity": 0.96,
                        "line": {"width": 1.4, "color": "rgba(226,232,240,0.85)"},
                    },
                    name="embeddings",
                    showlegend=False,
                )
            ],
            layout=realtime_fig.layout,
        )

    realtime_fig.update_traces(
        marker={
            "size": 8,
            "opacity": 0.96,
            "line": {"width": 1.4, "color": "rgba(226,232,240,0.85)"},
        },
        selector={"type": "scatter3d"},
    )
    realtime_fig.update_layout(
        title={"text": "Embedding 3D semántico", "x": 0.02, "xanchor": "left", "font": {"size": 15}},
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        scene={
            "domain": {"x": [0, 1], "y": [0, 1]},
            "bgcolor": "rgba(2,6,23,0.98)",
            "xaxis": {"title": "embedding_x normalizado", "range": [-2.9, 2.9], "showbackground": True, "backgroundcolor": "rgba(15,23,42,0.72)", "gridcolor": "rgba(148,163,184,0.22)", "zerolinecolor": "rgba(226,232,240,0.38)"},
            "yaxis": {"title": "embedding_y normalizado", "range": [-2.9, 2.9], "showbackground": True, "backgroundcolor": "rgba(15,23,42,0.72)", "gridcolor": "rgba(148,163,184,0.22)", "zerolinecolor": "rgba(226,232,240,0.38)"},
            "zaxis": {"title": "embedding_z normalizado", "range": [-2.9, 2.9], "showbackground": True, "backgroundcolor": "rgba(15,23,42,0.72)", "gridcolor": "rgba(148,163,184,0.22)", "zerolinecolor": "rgba(226,232,240,0.38)"},
            "aspectmode": "cube",
            "camera": {
                "eye": {"x": 1.35, "y": 1.35, "z": 0.95},
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
            },
        },
    )
    fig_spec = _plotly_figure_spec_for_browser(realtime_fig)
    component_html = f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #020617;
      --panel: rgba(15, 23, 42, 0.92);
      --border: rgba(148, 163, 184, 0.28);
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --ok: #22c55e;
      --warn: #f59e0b;
    }}
    body {{ margin: 0; background: transparent; font-family: Inter, Arial, sans-serif; color: var(--text); }}
    .shell {{ display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 14px; align-items: stretch; }}
    #plot {{ width: 100%; height: {max(height - 80, 560)}px; min-height: {max(height - 80, 560)}px; border: 1px solid var(--border); border-radius: 18px; overflow: hidden; background: rgba(2, 6, 23, 0.70); }}
    .panel {{ border: 1px solid var(--border); border-radius: 18px; padding: 14px; background: var(--panel); box-shadow: 0 20px 50px rgba(2, 6, 23, 0.22); }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 8px 0; color: var(--muted); font-size: 13px; line-height: 1.45; }}
    video {{ width: 100%; aspect-ratio: 4/3; border-radius: 14px; background: #000; object-fit: cover; transform: scaleX(-1); border: 1px solid var(--border); }}
    .row {{ display: flex; gap: 8px; margin: 10px 0; }}
    button {{ flex: 1; cursor: pointer; border: 1px solid rgba(56, 189, 248, 0.45); color: #e0f2fe; background: rgba(14, 165, 233, 0.16); border-radius: 12px; padding: 9px 10px; font-weight: 700; }}
    button:hover {{ background: rgba(14, 165, 233, 0.26); }}
    button.stop {{ border-color: rgba(248, 113, 113, 0.45); color: #fee2e2; background: rgba(239, 68, 68, 0.16); }}
    button.secondary {{ width: 100%; margin: 0 0 2px; border-color: rgba(148, 163, 184, 0.42); color: #e5e7eb; background: rgba(30, 41, 59, 0.66); }}
    label {{ display: block; color: var(--text); font-size: 12px; font-weight: 700; margin-top: 10px; }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    .metric {{ display: grid; grid-template-columns: 86px 1fr; gap: 8px; margin-top: 8px; font-size: 12px; }}
    .metric span:first-child {{ color: var(--muted); }}
    .status {{ margin-top: 10px; padding: 9px 10px; border-radius: 12px; border: 1px solid var(--border); background: rgba(15, 23, 42, 0.72); color: var(--muted); font-size: 12px; line-height: 1.4; }}
    .status.ok {{ color: #bbf7d0; border-color: rgba(34, 197, 94, 0.45); }}
    .status.warn {{ color: #fde68a; border-color: rgba(245, 158, 11, 0.45); }}
    canvas#sample {{ display: none; }}
    @media (max-width: 980px) {{ .shell {{ grid-template-columns: 1fr; }} #plot {{ min-height: 560px; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <div id="plot" aria-label="Embedding 3D controlado por cámara"></div>
    <aside class="panel">
      <h3>Control en tiempo real</h3>
      <p>Mueva la mano u objeto dominante frente a la cámara. La posición horizontal controla el azimuth, la vertical controla la elevación. Acerque la mano para hacer zoom in y aléjela para hacer zoom out. La leyenda se oculta en este modo para priorizar que los puntos 3D sean visibles.</p>
      <video id="video" autoplay playsinline muted></video>
      <canvas id="sample" width="96" height="72"></canvas>
      <div class="row">
        <button id="start">Iniciar</button>
        <button id="stop" class="stop">Detener</button>
      </div>
      <button id="resetZoom" type="button" class="secondary">Calibrar zoom actual</button>
      <label>Sensibilidad: <span id="sensVal">1.00</span></label>
      <input id="sensitivity" type="range" min="0.35" max="2.00" step="0.05" value="1.00" />
      <label>Suavizado: <span id="smoothVal">0.72</span></label>
      <input id="smooth" type="range" min="0.20" max="0.92" step="0.02" value="0.72" />
      <label>Fuerza zoom mano: <span id="zoomStrengthVal">1.20</span></label>
      <input id="zoomStrength" type="range" min="0.40" max="2.60" step="0.05" value="1.20" />
      <div class="metric"><span>Azimuth</span><strong id="azimuth">45°</strong></div>
      <div class="metric"><span>Elevación</span><strong id="elevation">28°</strong></div>
      <div class="metric"><span>Zoom</span><strong id="zoom">1.85</strong></div>
      <div class="metric"><span>Gesto zoom</span><strong id="zoomGesture">calibrando</strong></div>
      <div class="metric"><span>Modo</span><strong id="mode">esperando cámara</strong></div>
      <div id="status" class="status">Pulse Iniciar. El navegador pedirá permiso de cámara; la vista se actualizará de forma continua.</div>
    </aside>
  </div>
  <script>
    const fig = {fig_spec};
    const initialCamera = (fig.layout && fig.layout.scene && fig.layout.scene.camera) || {{eye: {{x: 1.35, y: 1.35, z: 0.95}}, up: {{x: 0, y: 0, z: 1}}}};
    const plotDiv = document.getElementById('plot');
    function applyExplicitPlotSize() {{
      const rect = plotDiv.getBoundingClientRect();
      const width = Math.max(640, Math.floor(rect.width || plotDiv.clientWidth || 900));
      const plotHeight = Math.max(560, Math.floor(rect.height || plotDiv.clientHeight || 720));
      plotDiv.style.width = width + 'px';
      plotDiv.style.height = plotHeight + 'px';
      fig.layout.width = width;
      fig.layout.height = plotHeight;
    }}
    applyExplicitPlotSize();
    Plotly.newPlot(plotDiv, fig.data, fig.layout, {{displayModeBar: true, scrollZoom: true, responsive: true}}).then(() => {{
      Plotly.Plots.resize(plotDiv);
      setTimeout(() => {{ applyExplicitPlotSize(); Plotly.relayout(plotDiv, {{width: fig.layout.width, height: fig.layout.height}}); }}, 120);
    }});

    const video = document.getElementById('video');
    const canvas = document.getElementById('sample');
    const ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
    const startButton = document.getElementById('start');
    const stopButton = document.getElementById('stop');
    const resetZoomButton = document.getElementById('resetZoom');
    const statusBox = document.getElementById('status');
    const sensitivityInput = document.getElementById('sensitivity');
    const smoothInput = document.getElementById('smooth');
    const zoomStrengthInput = document.getElementById('zoomStrength');
    const sensVal = document.getElementById('sensVal');
    const smoothVal = document.getElementById('smoothVal');
    const zoomStrengthVal = document.getElementById('zoomStrengthVal');
    const azimuthEl = document.getElementById('azimuth');
    const elevationEl = document.getElementById('elevation');
    const zoomEl = document.getElementById('zoom');
    const zoomGestureEl = document.getElementById('zoomGesture');
    const modeEl = document.getElementById('mode');
    let stream = null;
    let rafId = null;
    let lastUpdate = 0;
    let smoothState = {{ azimuth: 45, elevation: 28, zoom: 1.85 }};
    let baselineCoverage = null;
    let latestCoverage = null;

    function setStatus(message, kind='') {{
      statusBox.textContent = message;
      statusBox.className = 'status' + (kind ? ' ' + kind : '');
    }}

    function cameraEye(azimuthDeg, elevationDeg, zoom) {{
      const az = azimuthDeg * Math.PI / 180;
      const el = elevationDeg * Math.PI / 180;
      return {{
        x: zoom * Math.cos(el) * Math.cos(az),
        y: zoom * Math.cos(el) * Math.sin(az),
        z: zoom * Math.sin(el)
      }};
    }}

    function inferControlsFromFrame() {{
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let lum = new Float32Array(canvas.width * canvas.height);
      let sumLum = 0;
      for (let i = 0, p = 0; i < frame.length; i += 4, p++) {{
        const r = frame[i] / 255, g = frame[i + 1] / 255, b = frame[i + 2] / 255;
        const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        lum[p] = y;
        sumLum += y;
      }}
      const meanLum = sumLum / lum.length;
      let total = 0, xSum = 0, ySum = 0, selected = 0, skinSelected = 0;
      for (let y = 0, p = 0; y < canvas.height; y++) {{
        for (let x = 0; x < canvas.width; x++, p++) {{
          const i = p * 4;
          const r = frame[i] / 255, g = frame[i + 1] / 255, b = frame[i + 2] / 255;
          const skin = r > 0.32 && g > 0.18 && b > 0.10 && r > g * 1.05 && r > b * 1.18 && (r - g) < 0.45;
          const foreground = Math.abs(lum[p] - meanLum) > 0.10;
          const active = skin || foreground;
          if (skin) skinSelected++;
          if (active) {{
            const weight = (skin ? 1.55 : 0.75) * (0.25 + lum[p]);
            total += weight;
            xSum += x * weight;
            ySum += y * weight;
            selected++;
          }}
        }}
      }}
      if (total < 0.0001 || selected < 12) {{
        return null;
      }}
      const xCenter = xSum / total / (canvas.width - 1);
      const yCenter = ySum / total / (canvas.height - 1);
      const coverage = selected / lum.length;
      const sensitivity = parseFloat(sensitivityInput.value);
      const centeredX = Math.max(0, Math.min(1, 0.5 + (xCenter - 0.5) * sensitivity));
      const centeredY = Math.max(0, Math.min(1, 0.5 + (yCenter - 0.5) * sensitivity));
      return {{
        azimuth: Math.max(0, Math.min(360, centeredX * 360)),
        elevation: Math.max(-20, Math.min(90, 80 - centeredY * 100)),
        mode: skinSelected > 18 ? 'mano/rostro' : 'primer plano',
        coverage
      }};
    }}

    function tick(now) {{
      rafId = requestAnimationFrame(tick);
      if (!stream || video.readyState < 2 || now - lastUpdate < 80) return;
      lastUpdate = now;
      const detected = inferControlsFromFrame();
      if (!detected) {{
        modeEl.textContent = 'sin objetivo';
        setStatus('Cámara activa. Coloque una mano u objeto contrastante dentro del encuadre.', 'warn');
        return;
      }}
      if (baselineCoverage === null) {{
        baselineCoverage = detected.coverage;
      }}
      latestCoverage = detected.coverage;
      const keep = parseFloat(smoothInput.value);
      const inject = 1 - keep;
      const zoomStrength = parseFloat(zoomStrengthInput.value);
      const safeBaseline = Math.max(0.012, baselineCoverage);
      const coverageDelta = (detected.coverage - baselineCoverage) / safeBaseline;
      const deadZone = 0.12;
      let gestureZoom = 1.85;
      let gestureLabel = 'mantener';
      if (coverageDelta > deadZone) {{
        gestureZoom = Math.max(0.55, 1.85 - (coverageDelta - deadZone) * zoomStrength * 0.90);
        gestureLabel = 'zoom in';
      }} else if (coverageDelta < -deadZone) {{
        gestureZoom = Math.min(3.35, 1.85 + (Math.abs(coverageDelta) - deadZone) * zoomStrength * 0.90);
        gestureLabel = 'zoom out';
      }} else {{
        baselineCoverage = baselineCoverage * 0.985 + detected.coverage * 0.015;
      }}
      smoothState.azimuth = smoothState.azimuth * keep + detected.azimuth * inject;
      smoothState.elevation = smoothState.elevation * keep + detected.elevation * inject;
      smoothState.zoom = smoothState.zoom * keep + gestureZoom * inject;
      const eye = cameraEye(smoothState.azimuth, smoothState.elevation, smoothState.zoom);
      Plotly.relayout(plotDiv, {{
        'scene.camera': {{
          eye,
          up: initialCamera.up || {{x: 0, y: 0, z: 1}},
          center: initialCamera.center || {{x: 0, y: 0, z: 0}}
        }}
      }});
      azimuthEl.textContent = `${{Math.round(smoothState.azimuth)}}°`;
      elevationEl.textContent = `${{Math.round(smoothState.elevation)}}°`;
      zoomEl.textContent = smoothState.zoom.toFixed(2);
      zoomGestureEl.textContent = `${{gestureLabel}} · ${{(coverageDelta * 100).toFixed(0)}}%`;
      modeEl.textContent = detected.mode;
      setStatus('Tiempo real activo: acerque la mano para zoom in y aléjela para zoom out.', 'ok');
    }}

    async function startCamera() {{
      try {{
        stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'user', width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }}, audio: false }});
        video.srcObject = stream;
        await video.play();
        Plotly.Plots.resize(plotDiv);
        baselineCoverage = null;
        latestCoverage = null;
        zoomGestureEl.textContent = 'calibrando';
        setStatus('Cámara iniciada. Mantenga la mano visible; acérquela para zoom in y aléjela para zoom out.', 'ok');
        if (!rafId) rafId = requestAnimationFrame(tick);
      }} catch (err) {{
        setStatus('No se pudo acceder a la cámara: ' + err.message + '. Use localhost/HTTPS y revise permisos del navegador.', 'warn');
      }}
    }}

    function stopCamera() {{
      if (stream) {{
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }}
      video.srcObject = null;
      if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
      baselineCoverage = null;
      latestCoverage = null;
      zoomGestureEl.textContent = 'detenido';
      setStatus('Cámara detenida. El gráfico queda interactivo con mouse, trackpad o controles de Plotly.');
    }}

    sensitivityInput.addEventListener('input', () => {{ sensVal.textContent = Number(sensitivityInput.value).toFixed(2); }});
    smoothInput.addEventListener('input', () => {{ smoothVal.textContent = Number(smoothInput.value).toFixed(2); }});
    zoomStrengthInput.addEventListener('input', () => {{ zoomStrengthVal.textContent = Number(zoomStrengthInput.value).toFixed(2); }});
    resetZoomButton.addEventListener('click', () => {{
      baselineCoverage = latestCoverage;
      smoothState.zoom = 1.85;
      zoomGestureEl.textContent = 'recalibrado';
      setStatus('Zoom calibrado: esta distancia de mano queda como punto neutro.', 'ok');
    }});
    startButton.addEventListener('click', startCamera);
    stopButton.addEventListener('click', stopCamera);
    window.addEventListener('resize', () => Plotly.Plots.resize(plotDiv));
    setTimeout(() => Plotly.Plots.resize(plotDiv), 150);
    window.addEventListener('beforeunload', stopCamera);
  </script>
</body>
</html>
"""
    components.html(component_html, height=height, scrolling=False)


# ---------- Sidebar: data and controls ----------

def main() -> None:
    detailed_default, answers_default, cost_default, embeddings_default = read_default_data()

    st.sidebar.title("Controles del experimento")
    st.sidebar.caption("Carga datos, filtra el workload y ajusta el modelo de coste.")

    uploaded = st.sidebar.file_uploader("Cargar resultados CSV/JSON adicionales", type=["csv", "json"], accept_multiple_files=True)
    frames = [detailed_default]
    upload_messages: list[str] = []
    if uploaded:
        for file in uploaded:
            try:
                candidate = _load_uploaded_file(file)
                ok, missing = validate_data(candidate)
                if ok:
                    candidate["run_source"] = file.name
                    frames.append(candidate)
                    upload_messages.append(f"{file.name}: cargado correctamente")
                else:
                    upload_messages.append(f"{file.name}: faltan columnas {missing}")
            except Exception as exc:  # noqa: BLE001 - displayed to user intentionally
                upload_messages.append(f"{file.name}: error de lectura: {exc}")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["run_source"] = raw.get("run_source", "repository_default")

    with st.sidebar.expander("Validación de datos", expanded=bool(upload_messages)):
        ok, missing = validate_data(raw)
        if ok:
            st.success("Dataset válido para análisis principal.")
        else:
            st.error(f"Faltan columnas requeridas: {missing}")
        for message in upload_messages:
            st.write(message)
        st.write(f"Filas activas antes de filtros: {len(raw):,}")

    st.sidebar.subheader("Modelo de coste")
    input_price = st.sidebar.number_input("Precio input USD / 1K tokens", min_value=0.0, value=0.010, step=0.001, format="%.4f")
    output_price = st.sidebar.number_input("Precio output USD / 1K tokens", min_value=0.0, value=0.030, step=0.001, format="%.4f")
    w_tokens = st.sidebar.slider("Peso tokens en Cₚ", 0.0, 1.0, 0.4, 0.05)
    w_latency = st.sidebar.slider("Peso latencia en Cₚ", 0.0, 1.0, 0.4, 0.05)
    w_trace = st.sidebar.slider("Peso traza en Cₚ", 0.0, 1.0, 0.2, 0.05)
    utility_lambda = st.sidebar.slider("Penalización λ en utilidad neta", 0.0, 1.0, 0.35, 0.05)

    data = derive_cost_metrics(raw, input_price, output_price, w_tokens, w_latency, w_trace, utility_lambda)

    st.sidebar.subheader("Filtros")
    patterns = st.sidebar.multiselect(
        "Patrones",
        options=sorted(data["pattern"].dropna().unique().tolist()),
        default=[p for p in PATTERN_ORDER if p in set(data["pattern"])],
        format_func=lambda p: PATTERN_LABELS.get(p, p),
    )
    question_types = st.sidebar.multiselect(
        "Tipos de pregunta",
        options=sorted(data["question_type"].dropna().unique().tolist()),
        default=sorted(data["question_type"].dropna().unique().tolist()),
    )
    risks = st.sidebar.multiselect(
        "Riesgo",
        options=sorted(data["risk"].dropna().unique().tolist()),
        default=sorted(data["risk"].dropna().unique().tolist()),
    )

    filtered = data[
        data["pattern"].isin(patterns)
        & data["question_type"].isin(question_types)
        & data["risk"].isin(risks)
    ].copy()
    agg = aggregate_metrics(filtered) if not filtered.empty else pd.DataFrame()

    st.title("Agentic RAG Experiments Dashboard")
    st.markdown(
        """
        Este dashboard ofrece una vista interactiva de los experimentos de **Agentic RAG Business Patterns**. Integra evaluación de calidad, grounding, eficiencia, coste matemático, comportamiento agentic y geometría semántica para apoyar análisis científico y toma de decisiones de ingeniería.
        """
    )

    if filtered.empty:
        st.warning("Los filtros seleccionados no dejan registros disponibles.")
        return

    # ---------- KPIs ----------
    kpi_cols = st.columns(5)
    kpis = {
        "Calidad media": filtered["quality_index"].mean(),
        "Factualidad": filtered["factuality"].mean(),
        "Coste USD/query": filtered["usd_cost"].mean(),
        "Latencia ms": filtered["latency_ms"].mean(),
        "Utilidad neta": filtered["net_utility"].mean(),
    }
    for col, (label, value) in zip(kpi_cols, kpis.items()):
        with col:
            if "USD" in label:
                st.metric(label, f"${value:.5f}")
            elif "Latencia" in label:
                st.metric(label, f"{value:.1f}")
            else:
                st.metric(label, f"{value:.3f}")

    tabs = st.tabs([
        "Resumen",
        "Calidad y grounding",
        "Coste y eficiencia",
        "Trade-offs",
        "Embeddings 3D",
        "Exportar y metodología",
    ])

    # ---------- Summary ----------
    with tabs[0]:
        st.subheader("Métricas agregadas por patrón")
        display_cols = [
            "pattern_label",
            "quality_index",
            "factuality",
            "traceability",
            "completeness",
            "decision_accuracy",
            "hallucination_rate",
            "latency_ms",
            "total_tokens",
            "usd_cost",
            "cost_composite",
            "net_utility",
        ]
        st.dataframe(
            agg[[c for c in display_cols if c in agg]].style.format(precision=4),
            use_container_width=True,
        )
        st.download_button(
            "Descargar métricas agregadas filtradas",
            data=agg.to_csv(index=False).encode("utf-8"),
            file_name="dashboard_aggregate_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(
            """
            > La tabla debe leerse como una matriz de decisión. Un patrón domina si mejora la calidad sin incrementar de forma desproporcionada coste, tokens, latencia o complejidad de traza.
            """
        )

    # ---------- Quality ----------
    with tabs[1]:
        metric_options = [
            "quality_index",
            "factuality",
            "clarity",
            "traceability",
            "completeness",
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
            "decision_accuracy",
        ]
        selected_quality = st.multiselect(
            "Métricas de calidad/grounding",
            options=[m for m in metric_options if m in agg],
            default=[m for m in ["quality_index", "factuality", "traceability", "completeness", "decision_accuracy"] if m in agg],
        )
        if selected_quality:
            fig_quality = plot_grouped_quality(agg, selected_quality)
            st.plotly_chart(fig_quality, use_container_width=True)
            download_plotly_html(fig_quality, "quality_grounding_by_pattern.html")

        metric_box = st.selectbox("Métrica para distribución", options=[m for m in metric_options if m in filtered])
        fig_box = plot_box(filtered, metric_box)
        st.plotly_chart(fig_box, use_container_width=True)

        fig_heat = plot_retrieval_heatmap(filtered, answers_default)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ---------- Cost ----------
    with tabs[2]:
        st.subheader("Modelo matemático de coste")
        st.latex(r"USD_i = \frac{prompt\_tokens_i}{1000}p_{in} + \frac{completion\_tokens_i}{1000}p_{out}")
        st.latex(r"C_p = \frac{w_T\hat{T}_p + w_L\hat{L}_p + w_S\hat{S}_p}{w_T+w_L+w_S}")
        st.latex(r"U_p = Q_p - \lambda C_p")
        fig_eff = plot_efficiency(agg)
        st.plotly_chart(fig_eff, use_container_width=True)
        download_plotly_html(fig_eff, "efficiency_cost_by_pattern.html")

        st.markdown("**Ejemplo configurable:** 1.000 tokens input a USD 0.01/1K y 500 tokens output a USD 0.03/1K producen USD 0.025 por consulta.")
        example_cost = (1000 / 1000 * input_price) + (500 / 1000 * output_price)
        st.info(f"Con los precios actuales del panel, el ejemplo cuesta USD {example_cost:.5f}.")

        cost_cols = ["pattern_label", "total_tokens", "prompt_tokens", "completion_tokens", "latency_ms", "trace_steps", "usd_cost", "cost_composite", "net_utility"]
        st.dataframe(agg[[c for c in cost_cols if c in agg]].style.format(precision=5), use_container_width=True)

    # ---------- Trade-offs ----------
    with tabs[3]:
        numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
        default_x = "cost_composite" if "cost_composite" in numeric_cols else numeric_cols[0]
        default_y = "quality_index" if "quality_index" in numeric_cols else numeric_cols[0]
        x_axis = st.selectbox("Eje X", numeric_cols, index=numeric_cols.index(default_x))
        y_axis = st.selectbox("Eje Y", numeric_cols, index=numeric_cols.index(default_y))
        size_axis = st.selectbox("Tamaño de punto", numeric_cols, index=numeric_cols.index("total_tokens") if "total_tokens" in numeric_cols else 0)
        fig_scatter = plot_quality_cost(filtered, x_axis, y_axis, size_axis)
        st.plotly_chart(fig_scatter, use_container_width=True)
        download_plotly_html(fig_scatter, "quality_cost_scatter.html")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_latency = plot_latency_grounding(filtered)
            st.plotly_chart(fig_latency, use_container_width=True)
        with col_b:
            fig_3d = plot_3d(filtered)
            st.plotly_chart(fig_3d, use_container_width=True)

        radar_metrics = st.multiselect(
            "Métricas para radar normalizado",
            options=[c for c in ["quality_index", "factuality", "traceability", "completeness", "decision_accuracy", "latency_ms", "total_tokens", "cost_composite", "hallucination_rate", "trace_steps"] if c in agg],
            default=[c for c in ["quality_index", "traceability", "completeness", "decision_accuracy", "cost_composite", "latency_ms"] if c in agg],
        )
        if len(radar_metrics) >= 3:
            fig_radar = plot_radar(agg, radar_metrics)
            st.plotly_chart(fig_radar, use_container_width=True)

        par_cols = st.multiselect(
            "Variables para coordenadas paralelas",
            options=numeric_cols,
            default=[c for c in ["quality_index", "factuality", "completeness", "latency_ms", "total_tokens", "cost_composite", "net_utility"] if c in numeric_cols],
        )
        if len(par_cols) >= 2:
            fig_parallel = plot_parallel(filtered, par_cols)
            st.plotly_chart(fig_parallel, use_container_width=True)

    # ---------- Embeddings ----------
    with tabs[4]:
        st.subheader("Geometría semántica y exploración 3D")
        st.markdown(
            """
            La proyección 3D reutiliza las coordenadas semánticas reproducibles calculadas en `paper_visuals/tables/paper_semantic_embeddings_3d.csv`. Al rotar, acercar y filtrar la nube de puntos se puede estudiar si cada patrón induce regiones semánticas más compactas o separadas.
            """
        )
        embedding_filtered = embeddings_default[
            embeddings_default["pattern"].isin(patterns)
            & embeddings_default["question_type"].isin(question_types)
            & embeddings_default["risk"].isin(risks)
        ].copy()

        st.markdown("### Cámara 3D e interacción inmersiva")
        st.info(
            "El panel de cámara ya está visible en esta pestaña. Use los sliders para fijar la cámara virtual del gráfico; "
            "también puede rotar la nube directamente con mouse, trackpad o pantalla táctil, hacer zoom con la rueda/gesto de pellizco "
            "y usar doble clic para recentrar la escena."
        )

        camera_presets = {
            "Isométrica para paper": {"azimuth": 45, "elevation": 28, "zoom": 1.85},
            "Frontal": {"azimuth": 0, "elevation": 12, "zoom": 1.75},
            "Lateral": {"azimuth": 90, "elevation": 12, "zoom": 1.75},
            "Superior": {"azimuth": 45, "elevation": 82, "zoom": 1.95},
            "Detalle cercano": {"azimuth": 35, "elevation": 18, "zoom": 1.15},
        }
        st.session_state.setdefault("embedding_camera_azimuth", camera_presets["Isométrica para paper"]["azimuth"])
        st.session_state.setdefault("embedding_camera_elevation", camera_presets["Isométrica para paper"]["elevation"])
        st.session_state.setdefault("embedding_camera_zoom", camera_presets["Isométrica para paper"]["zoom"])

        control_col, browser_camera_col = st.columns([1.15, 0.85])

        with browser_camera_col:
            st.markdown("#### Modo cámara en tiempo real")
            st.markdown(
                "El control por cámara opera en vivo. Seleccione **Tiempo real con cámara** debajo del panel manual y pulse "
                "**Iniciar** dentro del componente embebido. Ese componente contiene la cámara y el gráfico 3D en el mismo contexto web, "
                "por lo que puede actualizar Plotly continuamente sin recargar Streamlit."
            )
            st.caption(
                "Los sliders de la izquierda siguen disponibles como fallback manual y como punto inicial de cámara antes de activar el modo tiempo real."
            )

        with control_col:
            st.markdown("#### Cámara virtual del gráfico 3D")
            color_options = [c for c in ["pattern", "question_type", "risk", "decision"] if c in embedding_filtered]
            color_by = st.selectbox("Colorear embeddings por", color_options or ["pattern"])
            preset_col, apply_col = st.columns([0.65, 0.35])
            with preset_col:
                camera_preset = st.selectbox("Vista rápida", list(camera_presets.keys()))
            with apply_col:
                st.write("")
                if st.button("Aplicar vista", use_container_width=True):
                    defaults = camera_presets[camera_preset]
                    st.session_state["embedding_camera_azimuth"] = defaults["azimuth"]
                    st.session_state["embedding_camera_elevation"] = defaults["elevation"]
                    st.session_state["embedding_camera_zoom"] = defaults["zoom"]
            azimuth = st.slider(
                "Azimuth horizontal de cámara",
                min_value=0,
                max_value=360,
                step=5,
                key="embedding_camera_azimuth",
                help="Gira la cámara alrededor del eje vertical del espacio semántico.",
            )
            elevation = st.slider(
                "Elevación vertical de cámara",
                min_value=-20,
                max_value=90,
                step=2,
                key="embedding_camera_elevation",
                help="Sube o baja el punto de vista para inspeccionar la separación entre clusters.",
            )
            zoom = st.slider(
                "Zoom / distancia de cámara",
                min_value=0.60,
                max_value=3.00,
                step=0.05,
                key="embedding_camera_zoom",
                help="Valores menores acercan la cámara; valores mayores muestran más contexto.",
            )
            chart_height = st.slider("Altura del gráfico", min_value=520, max_value=920, value=720, step=20)
            camera_eye = camera_eye_from_angles(azimuth, elevation, zoom)
            st.caption(
                f"Vector de cámara aplicado: x={camera_eye['x']:.2f}, y={camera_eye['y']:.2f}, z={camera_eye['z']:.2f}."
            )

        if not embedding_filtered.empty:
            fig_embeddings = plot_embeddings(embedding_filtered, color_by, camera_eye=camera_eye)
            fig_embeddings.update_layout(height=chart_height)
            interaction_mode = st.radio(
                "Modo de interacción del embedding",
                ["Tiempo real con cámara", "Manual con sliders"],
                horizontal=True,
                help="El modo tiempo real usa un componente web que contiene el gráfico y la cámara dentro del mismo iframe para poder actualizar Plotly continuamente.",
            )
            if interaction_mode == "Tiempo real con cámara":
                st.info(
                    "Pulse **Iniciar** dentro del panel embebido. La cámara del navegador controlará el gráfico 3D en tiempo real; "
                    "no hace falta tomar fotos. Si el navegador bloquea permisos, abra el dashboard en localhost o HTTPS."
                )
                render_realtime_embedding_controller(fig_embeddings, height=max(chart_height + 120, 760))
            else:
                st.plotly_chart(
                    fig_embeddings,
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
                )
            download_plotly_html(fig_embeddings, "semantic_embeddings_3d.html")
        else:
            st.warning("No hay embeddings disponibles bajo los filtros actuales.")

    # ---------- Export and methodology ----------
    with tabs[5]:
        st.subheader("Exportación y reproducibilidad")
        st.download_button(
            "Descargar registros filtrados con métricas derivadas",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="dashboard_filtered_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Descargar configuración de coste actual",
            data=json.dumps(
                {
                    "input_price_per_1k": input_price,
                    "output_price_per_1k": output_price,
                    "w_tokens": w_tokens,
                    "w_latency": w_latency,
                    "w_trace": w_trace,
                    "utility_lambda": utility_lambda,
                    "patterns": patterns,
                    "question_types": question_types,
                    "risks": risks,
                },
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8"),
            file_name="dashboard_cost_configuration.json",
            mime="application/json",
            use_container_width=True,
        )

        with st.expander("Definiciones de métricas", expanded=True):
            st.markdown(
                """
                **Factualidad** mide si la respuesta coincide con hechos esperados; **traceability** aproxima capacidad de auditar citas y pasos; **completeness** mide cobertura de hechos relevantes; **decision_accuracy** evalúa decisiones de responder, aclarar o rechazar; **faithfulness**, **context_precision** y **context_recall** se derivan de trazabilidad y completitud cuando el dataset cargado no aporta métricas RAGAS explícitas.
                """
            )
        with st.expander("Resumen de patrones harness"):
            st.markdown(
                """
                **Single-Agent RAG** ejecuta recuperación y generación en un único flujo. **Parent-Child** delega subtareas a agentes especializados bajo control central. **Mixture-of-Agents** consulta agentes en paralelo y agrega respuestas para maximizar completitud. **Handoff** transfiere preguntas ambiguas, sensibles o multi-dominio a rutas especializadas o rechaza cuando corresponde.
                """
            )
        with st.expander("Provenance de ejecución"):
            st.write(
                {
                    "repository_root": str(ROOT),
                    "default_results": str(RESULTS_DIR),
                    "paper_tables": str(PAPER_TABLES_DIR),
                    "rows_after_filters": int(len(filtered)),
                    "patterns_after_filters": sorted(filtered["pattern"].dropna().unique().tolist()),
                }
            )


if __name__ == "__main__":
    main()
