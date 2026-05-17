"""Streamlit dashboard for Agentic RAG experiment analysis.

Run from the repository root with:

    streamlit run dashboard/app.py

The app is intentionally self-contained so it can consume the reproducible CSV/JSON
outputs already committed in this repository while also accepting user-uploaded runs.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


def plot_embeddings(embeddings: pd.DataFrame, color_by: str) -> go.Figure:
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
    return fig_layout(fig, "Explorador 3D de embeddings semánticos")


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
        color_by = st.selectbox("Colorear embeddings por", [c for c in ["pattern", "question_type", "risk", "decision"] if c in embedding_filtered])
        if not embedding_filtered.empty:
            fig_embeddings = plot_embeddings(embedding_filtered, color_by)
            st.plotly_chart(fig_embeddings, use_container_width=True)
            download_plotly_html(fig_embeddings, "semantic_embeddings_3d.html")
        else:
            st.warning("No hay embeddings disponibles bajo los filtros actuales.")

        with st.expander("Control táctil, gestos y privacidad"):
            st.markdown(
                """
                Plotly permite rotar y hacer zoom en la nube 3D con mouse, trackpad o pantalla táctil. Para demostraciones inmersivas, esta sección puede activar una captura de cámara local mediante el navegador. La imagen no se envía a servicios externos por esta aplicación; el usuario debe conceder permiso explícito en el navegador y puede omitir la función.
                """
            )
            enable_camera = st.toggle("Activar panel opcional de cámara para prototipos de gestos", value=False)
            if enable_camera:
                st.camera_input("Captura local opcional para calibración visual de gestos")

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
