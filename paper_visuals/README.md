# Paper Visuals: gráficos científicos estilizados

**Autor:** manularrea  
**Objetivo:** concentrar en una carpeta independiente los activos visuales, tablas matemáticas y scripts reproducibles para apoyar el paper sobre patrones Agentic RAG.

## Contenido de la carpeta

| Ruta | Descripción |
|---|---|
| `scripts/create_paper_figures.py` | Script reproducible que calcula métricas matemáticas de coste y genera figuras de alto impacto. |
| `styles/paper_palette.json` | Paleta institucional y consistente por patrón experimental. |
| `tables/paper_cost_metrics.csv` | Tabla cuantitativa con coste compuesto, utilidad neta y eficiencias. |
| `tables/paper_cost_metrics.md` | Versión legible de la tabla matemática para insertar en documentos. |
| `tables/paper_semantic_embeddings_3d.csv` | Coordenadas 3D de cada respuesta para reproducir la geometría semántica. |
| `tables/paper_semantic_geometry_metrics.csv` | Métricas de compacidad, separación, desplazamiento y cociente geométrico por patrón. |
| `tables/paper_semantic_geometry_metrics.md` | Versión legible de las métricas geométricas para insertar en documentos. |
| `figures/*.png` | Figuras científicas exportadas a 300 dpi para paper, póster o presentación. |

## Ejecución

Desde la raíz del repositorio:

```bash
python paper_visuals/scripts/create_paper_figures.py
```

El script lee `results/aggregate_metrics.csv`, `results/metrics_by_question_type.csv`, `results/hypothesis_support.csv`, `results/answers.csv`, `results/detailed_metrics.csv` y `data/questions.json`. No requiere datos externos ni aleatoriedad: los embeddings se calculan de forma determinista con TF-IDF n-gram + SVD 3D sobre las preguntas, respuestas, decisiones, citas, hechos y trazas reales del benchmark.

## Modelo matemático resumido

El coste operativo compuesto se define como:

\[
C_p = 0.4\hat{T}_p + 0.4\hat{L}_p + 0.2\hat{S}_p
\]

Donde \(\hat{T}_p\) es tokens normalizados, \(\hat{L}_p\) es latencia normalizada y \(\hat{S}_p\) es complejidad de traza normalizada. La utilidad neta usada para comparar patrones es:

\[
U_p = Q_p - 0.35 C_p
\]

Donde \(Q_p\) corresponde al índice de calidad experimental. También se exportan dos métricas de eficiencia: calidad por 1k tokens y calidad por segundo.

| Métrica | Uso en el paper |
|---|---|
| `cost_composite` | Defender el trade-off coste-calidad de H1, H2 y H3. |
| `net_utility` | Comparar patrones con una función objetivo penalizada por coste. |
| `quality_per_1k_tokens` | Relacionar calidad con consumo de tokens. |
| `quality_per_second` | Relacionar calidad con latencia operacional. |
| `is_cost_quality_pareto` | Identificar patrones no dominados en la frontera calidad-coste. |

## Modelo de geometría semántica

La geometría semántica convierte cada respuesta evaluada en un vector textual construido a partir de pregunta, tipo de tarea, riesgo, dominio, patrón, decisión, respuesta, hechos, citas y traza. Sobre esa matriz se aplica TF-IDF con n-gramas y una reducción SVD a tres dimensiones. La figura 3D no cambia los embeddings originales de recuperación; representa cómo **la salida generada por cada patrón ocupa una región semántica diferente** cuando se incorporan las decisiones y trazas propias del patrón.

Para cada patrón \(p\), el script calcula un centroide \(\mu_p\), una compacidad intra-patrón y una separación inter-patrón:

\[
K_p = \frac{1}{|X_p|}\sum_{x_i \in X_p} d_{cos}(x_i, \mu_p)
\]

\[
S_p = \frac{1}{|P|-1}\sum_{q \neq p} d_{cos}(\mu_p, \mu_q)
\]

\[
G_p = \frac{S_p}{K_p + \epsilon}
\]

Donde \(d_{cos}\) es distancia coseno y \(G_p\) es el cociente separación/compacidad. También se calcula el desplazamiento semántico medio frente al baseline Single-Agent por pregunta: \(\Delta_p = \mathbb{E}_{q}[d_{cos}(x_{p,q}, x_{single,q})]\). En la ejecución incluida, Mixture-of-Agents alcanza el mayor \(G_p\) con 1.43, seguido por Parent-Child con 1.25, lo que respalda la interpretación de que los patrones multi-agente inducen regiones semánticas más diferenciadas para tareas complejas.

| Métrica geométrica | Uso en el paper |
|---|---|
| `semantic_compactness` | Medir dispersión intra-patrón; valores menores implican respuestas más concentradas. |
| `semantic_separation` | Medir distancia promedio entre el centroide del patrón y los demás centroides. |
| `semantic_shift_from_single` | Medir cuánto cambia la respuesta del patrón frente al baseline para la misma pregunta. |
| `semantic_separation_ratio` | Comparar especialización geométrica; valores mayores indican mejor separación relativa. |
| `svd_explained_variance_ratio_3d` | Reportar cuánta varianza textual captura la proyección 3D. |

## Figuras recomendadas para el paper

| Figura | Uso recomendado |
|---|---|
| `paper_cost_quality_frontier.png` | Figura principal para mostrar la frontera calidad-coste y patrones no dominados. |
| `paper_net_utility.png` | Figura para argumentar la selección óptima cuando se penaliza coste. |
| `paper_cost_decomposition.png` | Figura técnica para explicar si el coste proviene de tokens, latencia o trazas. |
| `paper_hypothesis_matrix.png` | Resumen visual de evidencia H1-H5. |
| `paper_quality_cost_radar.png` | Comparación multidimensional de calidad, eficiencia y seguridad. |
| `paper_task_type_completeness_heatmap.png` | Evidencia visual para H2, H3 y H5 por tipo de tarea. |
| `paper_semantic_embedding_3d.png` | Figura de impacto para mostrar la geometría semántica de respuestas por patrón agentic. |
| `paper_semantic_geometry_metrics.png` | Figura matemática para comparar separación/compacidad semántica entre patrones. |

## Nota de interpretación

Estas figuras no reemplazan el benchmark original; lo complementan. La carpeta `results/` conserva las salidas experimentales base, mientras que `paper_visuals/` transforma esas salidas en material visual y matemático listo para discusión científica.
