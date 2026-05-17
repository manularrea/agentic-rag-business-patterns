# Especificación técnica del dashboard Streamlit

Este documento define la arquitectura de la carpeta `dashboard/` para explorar resultados del benchmark **Agentic RAG Business Patterns** mediante una interfaz Streamlit reproducible, científica y orientada a paper. La aplicación consume los archivos reales del repositorio y permite cargar datos adicionales en CSV o JSON para comparar ejecuciones futuras.

## Objetivo funcional

El dashboard debe permitir que investigadores y practicantes inspeccionen la relación entre calidad, factualidad, trazabilidad, grounding, latencia, tokens, coste matemático y comportamiento agentic de los patrones **Single-Agent**, **Parent-Child**, **Mixture-of-Agents** y **Handoff**. La interfaz prioriza visualizaciones interactivas y explicaciones metodológicas que puedan respaldar hipótesis experimentales y exportarse para productos científicos.

## Arquitectura prevista

| Capa | Archivo o componente | Responsabilidad |
|---|---|---|
| Entrada de datos | `dashboard/app.py` | Cargar CSV/JSON desde `results/` y `paper_visuals/tables/`, aceptar cargas del usuario y validar columnas. |
| Transformación | Funciones internas de `app.py` | Normalizar métricas, calcular coste por tokens, coste compuesto, utilidad neta, KPIs agregados y campos derivados. |
| Visualización | Plotly + Streamlit | Barras, scatter, radar, box plot, 3D scatter, coordenadas paralelas, heatmaps y embeddings 3D. |
| Estilo | `dashboard/.streamlit/config.toml` | Tema oscuro, colores primarios y legibilidad para uso editorial. |
| Documentación | `dashboard/README.md` | Instrucciones de instalación, ejecución, fuentes de datos y limitaciones. |
| Dependencias | `dashboard/requirements.txt` y `pyproject.toml` | Dependencias explícitas para ejecución aislada o instalación opcional. |

## Métricas matemáticas

El coste monetario configurable por consulta se define como:

\[
\mathrm{USD}_i = \frac{\mathrm{prompt\_tokens}_i}{1000}p_{in} + \frac{\mathrm{completion\_tokens}_i}{1000}p_{out}
\]

El índice compuesto reutiliza la formulación ya incluida en el repositorio:

\[
C_p = 0.4\hat{T}_p + 0.4\hat{L}_p + 0.2\hat{S}_p
\]

La utilidad neta se calcula como:

\[
U_p = Q_p - \lambda C_p
\]

Donde \(Q_p\) es un índice de calidad promedio, \(\hat{T}_p\) son tokens normalizados, \(\hat{L}_p\) es latencia normalizada, \(\hat{S}_p\) es complejidad de traza normalizada y \(\lambda\) es la penalización operacional configurable desde la interfaz.

## Visualizaciones incluidas

| Visualización | Propósito |
|---|---|
| KPIs por patrón | Resumir calidad, factualidad, completitud, latencia, tokens, coste y utilidad. |
| Barras de calidad y grounding | Comparar factualidad, claridad, trazabilidad, completitud y métricas RAG derivadas. |
| Barras de eficiencia | Comparar latencia, tokens y coste por patrón. |
| Scatter calidad-coste | Identificar frontera de Pareto y outliers costosos o subóptimos. |
| Scatter latencia-grounding | Diagnosticar si mayor latencia aporta mejor grounding o decisión. |
| Radar multi-métrica | Comparar perfiles normalizados de calidad, coste, latencia y comportamiento agentic. |
| Box plots | Mostrar dispersión por pregunta y patrón. |
| 3D calidad-coste-latencia | Analizar objetivos simultáneos en un espacio interactivo. |
| Coordenadas paralelas | Explorar trade-offs multivariados. |
| Embeddings 3D | Reutilizar coordenadas semánticas para analizar geometría por patrón y categoría. |
| Heatmap de recuperación | Aproximar frecuencia de citas/documentos por tipo de pregunta y patrón. |

## Decisiones técnicas

La implementación será deliberadamente autocontenida dentro de `dashboard/` para no alterar el núcleo experimental. Streamlit se usará como capa de presentación; Plotly se usará para interactividad, exportación HTML y control táctil nativo. La interacción por gestos se documentará como soporte de rotación/zoom en Plotly y como panel opcional de cámara con `st.camera_input`, manteniendo privacidad por defecto porque el navegador siempre debe solicitar consentimiento explícito para la cámara.

## Referencias

[1]: https://docs.streamlit.io/ "Streamlit Documentation"
[2]: https://plotly.com/python/ "Plotly Python Graphing Library"
[3]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
