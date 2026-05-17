# Resultados y visualizaciones

**Autor:** manularrea  
**Segmento:** Presentación visual, interpretación de resultados y soporte bibliográfico  
**Uso sugerido:** Secciones de resultados, discusión, figuras y referencias del paper

## Propósito de la sección de resultados

La sección de resultados debe demostrar si las hipótesis H1-H5 son defendibles con evidencia cuantitativa, matemática y visual. Para lograrlo, el paper no debe limitarse a una tabla agregada; debe mostrar comparaciones por patrón, por tipo de pregunta, por coste, por latencia, por decisión y por geometría semántica. Este enfoque evita conclusiones absolutas y permite justificar cuándo un harness multi-agente es necesario y cuándo un baseline Single-Agent RAG es suficiente.

> La regla editorial central es reportar siempre calidad junto con coste. Una mejora de completitud o trazabilidad no es automáticamente superior si su coste en tokens, latencia o complejidad operacional elimina la utilidad práctica.

## Tablas principales recomendadas

Las tablas deben organizar el argumento empírico del paper. La primera tabla debe mostrar métricas agregadas por patrón; la segunda debe mostrar resultados por tipo de pregunta; la tercera debe resumir soporte por hipótesis; y la cuarta debe presentar coste compuesto y utilidad neta. Todas estas tablas ya cuentan con fuentes reproducibles dentro del repositorio.

| Tabla | Fuente del repositorio | Propósito |
|---|---|---|
| Métricas agregadas por patrón | `results/aggregate_metrics.csv` | Comparar factualidad, claridad, trazabilidad, completitud, alucinación, decisión, latencia y tokens. |
| Métricas por tipo de pregunta | `results/metrics_by_question_type.csv` | Mostrar que el patrón ganador depende de `simple`, `medium`, `synthesis`, `ambiguous` o `sensitive`. |
| Soporte por hipótesis | `results/hypothesis_support.csv` | Conectar resultados empíricos con H1-H5. |
| Coste y utilidad neta | `paper_visuals/tables/paper_cost_metrics.csv` | Evaluar si la mejora justifica el coste. |
| Geometría semántica | `paper_visuals/tables/paper_semantic_geometry_metrics.csv` | Medir separación, compacidad y desplazamiento por patrón. |

## Figuras de comparación entre patrones

Los gráficos deben estar diseñados para responder preguntas científicas específicas. Un gráfico de barras de factualidad no basta si el argumento central es multiobjetivo; por eso se combinan barras, dispersión, radar, heatmaps y proyección 3D. La carpeta `paper_visuals/` fue creada para centralizar figuras editoriales de alto impacto y estilo consistente.

| Figura | Archivo | Pregunta científica que responde |
|---|---|---|
| Frontera calidad-coste | `paper_visuals/figures/paper_cost_quality_frontier.png` | ¿Qué patrón ofrece mayor calidad por penalización operacional? |
| Utilidad neta | `paper_visuals/figures/paper_net_utility.png` | ¿Qué patrón conserva valor después de descontar coste? |
| Descomposición de coste | `paper_visuals/figures/paper_cost_decomposition.png` | ¿El coste proviene de tokens, latencia o complejidad de trazas? |
| Radar calidad-coste-seguridad | `paper_visuals/figures/paper_quality_cost_radar.png` | ¿Cómo se comporta cada patrón en múltiples objetivos simultáneos? |
| Heatmap por tipo de tarea | `paper_visuals/figures/paper_task_type_completeness_heatmap.png` | ¿En qué categorías se justifica cada patrón? |
| Matriz de hipótesis | `paper_visuals/figures/paper_hypothesis_matrix.png` | ¿Qué evidencia soporta H1-H5? |
| Embeddings 3D | `paper_visuals/figures/paper_semantic_embedding_3d.png` | ¿La geometría de respuestas cambia según el patrón agentic? |
| Métricas geométricas | `paper_visuals/figures/paper_semantic_geometry_metrics.png` | ¿Qué patrón produce mayor separación semántica ajustada por compacidad? |

## Visualización 3D de geometría semántica

La figura de embeddings 3D aporta una capa de soporte matemático y visual. Cada punto representa una salida patrón-pregunta convertida en vector textual mediante TF-IDF y proyectada a tres dimensiones con SVD. Esta técnica se relaciona con el análisis semántico latente, que usa descomposición en valores singulares para reducir matrices término-documento de alta dimensionalidad [1].

La interpretación debe ser cuidadosa: la figura no afirma que el índice de recuperación cambie físicamente sus embeddings según el patrón; muestra que las **respuestas, decisiones y trazas generadas por cada patrón** ocupan regiones semánticas distintas. Esa diferencia geométrica es evidencia de que la orquestación modifica el comportamiento observable del sistema.

| Métrica geométrica | Fórmula conceptual | Interpretación para el paper |
|---|---|---|
| Compacidad | \(C_p=\frac{1}{|Q|}\sum_q \lVert z_{p,q}-\mu_p\rVert\) | Menor valor implica respuestas más consistentes dentro del patrón. |
| Separación | \(S_p=\frac{1}{|P|-1}\sum_{r\neq p}\lVert \mu_p-\mu_r\rVert\) | Mayor valor implica comportamiento más distinguible entre patrones. |
| Desplazamiento vs baseline | \(D_p=\lVert \mu_p-\mu_{single}\rVert\) | Mide cuánto cambia el patrón respecto a Single-Agent. |
| Cociente S/C | \(R_p=S_p/(C_p+\epsilon)\) | Mide separación ajustada por dispersión interna. |

En los resultados generados, Mixture-of-Agents presenta el mayor cociente separación/compacidad, lo que es consistente con su rol de síntesis multi-documento. Parent-Child muestra una geometría diferenciada sin perder balance operativo. Handoff se desplaza respecto al baseline porque altera la decisión en consultas ambiguas o sensibles, lo que debe interpretarse como señal de control de riesgo y no como pérdida de calidad.

## Interpretación por hipótesis

La discusión debe organizarse por hipótesis y evitar presentar un ranking único. Cada patrón responde a una función distinta dentro del espacio calidad-coste-riesgo.

| Hipótesis | Resultado esperado | Lectura visual recomendada |
|---|---|---|
| H1 | Multi-agente mejora calidad/trazabilidad con mayor coste. | Frontera calidad-coste y matriz de hipótesis. |
| H2 | Parent-Child ofrece mejor equilibrio en complejidad media. | Utilidad neta y métricas por tipo de pregunta. |
| H3 | Mixture-of-Agents mejora completitud en síntesis. | Heatmap por tipo de tarea y geometría 3D. |
| H4 | Handoff reduce alucinaciones y mejora decisión en riesgo alto. | Matriz de hipótesis y exactitud de decisión. |
| H5 | Single-Agent compite en preguntas simples con menor coste. | Coste-latencia y tabla por tipo `simple`. |

Una diferencia debe interpretarse como relevante si es consistente con la hipótesis, aparece en la categoría de pregunta esperada y no se explica únicamente por mayor verbosidad. Por ejemplo, Mixture-of-Agents puede tener mayor completitud en síntesis, pero si su coste crece de forma desproporcionada, debe recomendarse solo cuando la completitud multi-documento sea el criterio dominante.

## Significancia, intervalos e incertidumbre

En el benchmark actual, las ejecuciones son deterministas y el conjunto de preguntas es pequeño. Por ello, la palabra “significativo” debe usarse con cuidado. El paper puede reportar diferencias cuantitativas como evidencia experimental controlada y proponer intervalos bootstrap o pruebas pareadas para una ampliación con más preguntas o múltiples corridas.

| Criterio | Recomendación editorial |
|---|---|
| Diferencia de medias | Reportar \(\Delta\) frente a Single-Agent. |
| Intervalos de confianza | Usar bootstrap por pregunta cuando exista mayor tamaño muestral. |
| Tests estadísticos | Preferir pruebas pareadas porque todos los patrones responden las mismas preguntas. |
| Tamaño de efecto | Reportar magnitud además de p-valor. |
| Robustez | Repetir con más preguntas, dominios y modelos si se busca generalización externa. |

## Fuentes y citas para respaldar el análisis

El paper debe respaldar la separación entre retrieval y generación con literatura de RAG y RAGAS [2] [3]. Debe respaldar el análisis de patrones multi-agente con literatura de agentes autónomos y Mixture-of-Agents [4] [5]. Debe respaldar la discusión de riesgo y handoff con marcos de gobernanza de IA como NIST AI RMF [6]. Para las ecuaciones de coste por tokens, se pueden citar páginas oficiales de pricing de proveedores al momento de hacer una evaluación económica concreta [7] [8] [9].

| Tema | Fuente sugerida | Uso en el paper |
|---|---|---|
| RAG | Lewis et al. | Fundamentar generación aumentada por recuperación. |
| RAGAS | Es et al. y documentación RAGAS | Separar métricas de contexto y respuesta. |
| Agentes LLM | Survey de agentes autónomos | Justificar orquestación, delegación y control. |
| Mixture-of-Agents | Wang et al. | Justificar agregación de múltiples agentes. |
| Riesgo de IA | NIST AI RMF | Justificar handoff, rechazo y controles. |
| Coste por tokens | Pricing oficial de modelos | Calcular coste monetario por consulta. |

## Recomendación de narrativa final

La narrativa del paper debe presentar los resultados como una decisión de ingeniería de IA: seleccionar el patrón según el tipo de tarea y la función objetivo. Si el objetivo es bajo coste y la pregunta es simple, Single-Agent es suficiente. Si la tarea es media y necesita trazabilidad sin paralelismo excesivo, Parent-Child es preferible. Si la pregunta requiere síntesis amplia, Mixture-of-Agents se justifica. Si la pregunta es ambigua o sensible, Handoff es necesario para controlar riesgo.

Esta narrativa es científicamente más sólida que afirmar superioridad universal, porque reconoce el trade-off central de los sistemas Agentic RAG: **calidad, coste, latencia, trazabilidad y seguridad no se optimizan simultáneamente con el mismo patrón**.

## Referencias

[1]: https://doi.org/10.1137/1.9780898719769.ch5 "Latent Semantic Indexing via Singular Value Decomposition"  
[2]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"  
[3]: https://arxiv.org/abs/2309.15217 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"  
[4]: https://arxiv.org/abs/2308.08155 "A Survey on Large Language Model based Autonomous Agents"  
[5]: https://arxiv.org/abs/2402.01680 "Mixture-of-Agents Enhances Large Language Model Capabilities"  
[6]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"  
[7]: https://platform.openai.com/docs/pricing "OpenAI API Pricing"  
[8]: https://docs.anthropic.com/en/docs/about-claude/pricing "Anthropic Claude Pricing"  
[9]: https://cloud.google.com/vertex-ai/generative-ai/pricing "Google Vertex AI Generative AI Pricing"
