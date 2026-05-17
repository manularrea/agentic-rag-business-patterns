# Dashboard interactivo para Agentic RAG

La carpeta `dashboard/` contiene una aplicación **Streamlit** para explorar, filtrar y exportar los resultados del benchmark de patrones Agentic RAG. La interfaz está diseñada para apoyar análisis científico, discusión de hipótesis y preparación de figuras interactivas para presentaciones o anexos del paper.

## Propósito

El dashboard convierte los resultados tabulares del repositorio en una capa exploratoria. Su objetivo no es reemplazar las figuras estáticas de `paper_visuals/`, sino complementarlas con análisis interactivo de calidad, coste, latencia, tokens, trazabilidad, recuperación y geometría semántica. Streamlit permite construir aplicaciones de datos en Python con una API declarativa, mientras que Plotly aporta gráficos interactivos con rotación 3D, zoom, selección y exportación HTML.[1] [2]

| Componente | Archivo | Función |
|---|---|---|
| Aplicación principal | `app.py` | Carga datos, calcula métricas derivadas y renderiza pestañas interactivas. |
| Tema visual | `.streamlit/config.toml` | Define apariencia oscura, contraste editorial y color primario. |
| Dependencias aisladas | `requirements.txt` | Lista paquetes necesarios para ejecutar solo el dashboard. |
| Exportaciones | `exports/` | Carpeta reservada para resultados exportados localmente por el usuario. |
| Activos | `assets/` | Carpeta reservada para capturas, íconos o recursos visuales futuros. |

## Ejecución

Desde la raíz del repositorio, instale dependencias y ejecute la aplicación:

```bash
python -m pip install -e .[dashboard]
streamlit run dashboard/app.py
```

También puede ejecutarse con dependencias aisladas:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

La aplicación usa por defecto los archivos versionados en el repositorio:

| Fuente | Uso en dashboard |
|---|---|
| `results/detailed_metrics.csv` | Métricas por pregunta y patrón. |
| `results/answers.csv` | Contenido de respuestas, citas y trazas para mapa de recuperación. |
| `paper_visuals/tables/paper_cost_metrics.csv` | Métricas agregadas de coste usadas como referencia científica. |
| `paper_visuals/tables/paper_semantic_embeddings_3d.csv` | Coordenadas 3D de geometría semántica por pregunta y patrón. |

## Carga de datos propios

El panel lateral permite subir archivos **CSV** o **JSON**. Para incorporarse al análisis principal, cada archivo debe contener como mínimo las columnas siguientes.

| Columna requerida | Significado |
|---|---|
| `question_id` | Identificador de la pregunta evaluada. |
| `pattern` | Patrón evaluado, por ejemplo `single_agent` o `parent_child`. |
| `factuality` | Puntaje de factualidad de la respuesta. |
| `clarity` | Puntaje de claridad de la respuesta. |
| `traceability` | Puntaje de trazabilidad o auditabilidad. |
| `completeness` | Puntaje de completitud. |
| `hallucination_rate` | Tasa de alucinación observada o estimada. |
| `decision_accuracy` | Exactitud de la decisión de responder, aclarar o rechazar. |
| `latency_ms` | Latencia por consulta en milisegundos. |
| `total_tokens` | Tokens totales usados en la consulta. |

Si el archivo no contiene `prompt_tokens`, `completion_tokens`, `trace_steps`, `faithfulness`, `context_precision`, `context_recall` o `answer_relevancy`, la aplicación calcula aproximaciones reproducibles a partir de los campos disponibles. Esta decisión preserva la compatibilidad con ejecuciones futuras sin sacrificar trazabilidad metodológica.

## Modelo matemático de coste

El dashboard expone un modelo configurable para estudiar trade-offs económicos y operacionales. El coste monetario por consulta se calcula como:

\[
\mathrm{USD}_i = \frac{\mathrm{prompt\_tokens}_i}{1000}p_{in} + \frac{\mathrm{completion\_tokens}_i}{1000}p_{out}.
\]

El coste compuesto por patrón pondera tokens, latencia y complejidad de traza:

\[
C_p = \frac{w_T\hat{T}_p + w_L\hat{L}_p + w_S\hat{S}_p}{w_T+w_L+w_S}.
\]

La utilidad neta se define como una función de calidad penalizada por coste:

\[
U_p = Q_p - \lambda C_p.
\]

Esta formulación permite probar si patrones multi-agente mejoran calidad y trazabilidad de forma suficiente para justificar mayor latencia o consumo de tokens.

## Visualizaciones incluidas

| Pestaña | Visualizaciones | Uso analítico |
|---|---|---|
| Resumen | KPIs y tabla agregada | Comparar patrones bajo filtros activos. |
| Calidad y grounding | Barras, box plots, heatmap de citas | Analizar factualidad, completitud, trazabilidad y recuperación. |
| Coste y eficiencia | Fórmulas, barras normalizadas, tabla de coste | Examinar tokens, latencia, coste USD y utilidad neta. |
| Trade-offs | Scatter, 3D calidad-coste-latencia, radar, coordenadas paralelas | Identificar fronteras de Pareto y perfiles multiobjetivo. |
| Embeddings 3D | Nube semántica 3D con filtros | Interpretar geometría semántica por patrón, pregunta y riesgo. |
| Exportar y metodología | Descarga de CSV, configuración JSON y definiciones | Reproducir análisis y anexar evidencia al paper. |

## Gestos e interacción 3D

Los gráficos 3D de Plotly soportan rotación, paneo, zoom y selección mediante mouse, trackpad o pantalla táctil. La pestaña de embeddings incluye además un panel opcional de cámara con `st.camera_input` para prototipos locales de interacción basada en gestos. Esta función está desactivada por defecto y requiere consentimiento explícito del navegador.[1] [2]

## Buenas prácticas para uso en paper

El dashboard debe usarse como entorno de exploración y validación. Las figuras estáticas finales deben exportarse desde `paper_visuals/` o capturarse desde el dashboard solo después de fijar filtros, parámetros de coste y versión del commit. Para mantener reproducibilidad, se recomienda descargar el JSON de configuración de coste junto con cualquier CSV filtrado.

## References

[1]: https://docs.streamlit.io/ "Streamlit Documentation"
[2]: https://plotly.com/python/ "Plotly Python Graphing Library"
[3]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
