# Control de calidad de figuras científicas estilizadas

La figura `paper_cost_quality_frontier.png` es adecuada para el paper porque muestra de manera clara la relación entre coste compuesto normalizado e índice de calidad, identifica visualmente los patrones no dominados y explicita la fórmula de coste en el eje x. La cercanía visual entre Handoff y Single-Agent está bien representada y permite discutir eficiencia frente a seguridad de decisión.

La figura `paper_net_utility.png` es legible y potente para argumentar una función objetivo penalizada por coste. El orden de utilidad neta es interpretable, los valores están anotados y la fórmula \(U = quality\_index - 0.35\cdot coste\_compuesto\) aparece directamente en el eje, lo que facilita soporte matemático en el paper.

La figura `paper_cost_decomposition.png` comunica bien el modelo matemático de coste, porque separa tokens, latencia y trazas con pesos explícitos. Es especialmente útil para justificar por qué Mixture-of-Agents obtiene alta calidad pero paga el máximo coste compuesto.

La figura `paper_quality_cost_radar.png` es conceptualmente sólida, aunque requiere un ajuste menor de composición para evitar que el título quede demasiado cerca del borde izquierdo y para dar mayor aire a la leyenda. Se corregirá en el script antes de publicar la versión final.

Tras regenerar `paper_quality_cost_radar.png`, la composición es suficientemente clara para discusión científica; el título ya no aparece comprimido y la leyenda se mantiene separada del gráfico. La figura permite comparar simultáneamente calidad, eficiencia de tokens, eficiencia temporal, utilidad neta y seguridad de decisión.

La figura `paper_task_type_completeness_heatmap.png` comunica bien diferencias por tipo de tarea y respalda especialmente H2, H3 y H5. Como mejora editorial final, se traducirán las etiquetas de tipo de pregunta al español para mantener consistencia lingüística con el resto del repositorio y con la narrativa del paper.

La versión regenerada de `paper_task_type_completeness_heatmap.png` ya usa etiquetas en español y mantiene buena legibilidad numérica. La figura es apta para el paper y puede usarse para explicar diferencias por complejidad de tarea.

La figura `paper_hypothesis_matrix.png` resume correctamente H1-H5, pero requiere una corrección editorial: algunos textos de evidencia se acercan demasiado a la barra de soporte de la derecha. Se ajustará el script para envolver texto, reducir longitud por línea y desplazar la barra de soporte a una zona separada.


## Revisión adicional: geometría semántica 3D

Se añadieron dos entregables visuales orientados al paper: `paper_semantic_embedding_3d.png` y `paper_semantic_geometry_metrics.png`. La figura 3D comunica una lectura de alto impacto porque posiciona cada respuesta del benchmark en un espacio semántico reducido mediante TF-IDF n-gram + SVD 3D, colorea por patrón agentic, usa marcadores por tipo de tarea y muestra centroides por patrón. El valor científico principal es que la visualización no es decorativa: está respaldada por una tabla reproducible de coordenadas y métricas geométricas.

La figura de barras de separación/compacidad es editorialmente limpia y permite defender matemáticamente la geometría semántica mediante el cociente `S/C = separación inter-centroide / dispersión intra-patrón`. En la revisión visual se observa que Mixture-of-Agents presenta el mayor cociente geométrico, consistente con su comportamiento de síntesis multi-documento. La figura 3D es apta para paper, aunque la leyenda de tipo de tarea queda cerca del eje inferior; se conserva porque no bloquea la interpretación, pero puede moverse en una versión de cámara alternativa si el formato final de revista exige más margen.
