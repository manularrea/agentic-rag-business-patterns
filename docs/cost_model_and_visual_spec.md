# Modelo matemático de coste y especificación visual científica

**Autor:** manularrea  
**Propósito:** documentar las métricas matemáticas y las reglas de visualización usadas para construir figuras publicables del benchmark Agentic RAG.

## Modelo de coste

El benchmark mide patrones RAG con métricas de calidad y coste. Para que la comparación sea defendible en un paper, el coste no se interpreta como una variable única, sino como un vector operativo que combina consumo de tokens, latencia y complejidad de orquestación. Dado un patrón \(p\) y una pregunta \(q\), se define el vector de coste como:

\[
\mathbf{c}_{p,q}=\left[T_{p,q}, L_{p,q}, S_{p,q}\right]
\]

Donde \(T_{p,q}\) representa tokens totales, \(L_{p,q}\) representa latencia estimada en milisegundos y \(S_{p,q}\) representa pasos de traza u operaciones de coordinación. El coste normalizado por dimensión se calcula con normalización min-max sobre los patrones evaluados:

\[
\hat{x}_{p}=\frac{x_{p}-\min_{p'}(x_{p'})}{\max_{p'}(x_{p'})-\min_{p'}(x_{p'})+\epsilon}
\]

A partir de estas variables se definen cuatro indicadores para el paper.

| Indicador | Fórmula | Interpretación |
|---|---|---|
| Coste operativo compuesto | \(C_p = w_T\hat{T}_p + w_L\hat{L}_p + w_S\hat{S}_p\) | Penaliza patrones con más tokens, más latencia y mayor complejidad de coordinación. |
| Utilidad neta | \(U_p = Q_p - \lambda C_p\) | Estima la utilidad de calidad descontando coste operativo. |
| Calidad por 1k tokens | \(E^T_p = \frac{Q_p}{T_p/1000}\) | Mide cuánta calidad se obtiene por cada mil tokens. |
| Calidad por segundo | \(E^L_p = \frac{Q_p}{L_p/1000}\) | Mide cuánta calidad se obtiene por segundo de latencia. |

En la implementación se usa \(Q_p\) como `quality_index`, construido desde factualidad, claridad, trazabilidad, completitud y exactitud de decisión. Para evitar arbitrariedad excesiva, el coste compuesto usa ponderaciones simétricas \(w_T=0.4\), \(w_L=0.4\) y \(w_S=0.2\). La utilidad neta usa \(\lambda=0.35\), de modo que el coste tiene fuerza suficiente para diferenciar patrones con calidad similar sin dominar completamente el análisis.

> En términos prácticos, el modelo permite distinguir tres situaciones científicamente relevantes: patrones de alta calidad pero alto coste, patrones de menor calidad pero muy eficientes y patrones que se ubican cerca de una frontera de Pareto calidad-coste.

## Modelo de geometría semántica

Además del coste operativo, el paper requiere una lectura matemática de cómo cambia la **geometría semántica** de las respuestas cuando se usa un patrón agentic u otro. La geometría no se interpreta como una modificación del embedding usado por el recuperador, sino como una proyección de las salidas reales del benchmark. Cada observación textual integra pregunta, tipo de tarea, riesgo, dominio, patrón, decisión, respuesta, hechos citados, documentos citados y traza de ejecución.

Sea \(x_{p,q}\) el vector textual normalizado de la respuesta del patrón \(p\) para la pregunta \(q\). El script construye \(x_{p,q}\) con TF-IDF de n-gramas y distancia coseno. Para visualización, la matriz documento-término se reduce a tres dimensiones mediante SVD, lo que permite mostrar una nube 3D y centroides por patrón. Para soporte cuantitativo, se calculan cuatro métricas.

| Indicador geométrico | Fórmula | Interpretación |
|---|---|---|
| Compacidad intra-patrón | \(K_p = \frac{1}{|X_p|}\sum_{x_i\in X_p} d_{cos}(x_i,\mu_p)\) | Dispersión de las respuestas de un patrón alrededor de su centroide. Menor valor implica comportamiento semántico más concentrado. |
| Separación inter-patrón | \(S_p = \frac{1}{|P|-1}\sum_{r\neq p} d_{cos}(\mu_p,\mu_r)\) | Distancia promedio del centroide de un patrón frente a los centroides de los demás patrones. |
| Cociente separación/compacidad | \(G_p = \frac{S_p}{K_p+\epsilon}\) | Medida sintética de especialización geométrica. Valores mayores indican regiones semánticas más distinguibles. |
| Desplazamiento frente a baseline | \(\Delta_p = \mathbb{E}_q[d_{cos}(x_{p,q},x_{single,q})]\) | Cuánto cambia semánticamente la respuesta del patrón frente a Single-Agent para la misma pregunta. |

En la ejecución incluida, el cociente \(G_p\) ubica a Mixture-of-Agents como el patrón más separado y compacto en términos relativos con 1.43, seguido de Parent-Child con 1.25, Handoff con 1.14 y Single-Agent con 1.07. Esta evidencia es útil para discutir que la orquestación multi-agente no solo cambia métricas agregadas de calidad y coste, sino también la estructura semántica observable de las respuestas.

> La lectura geométrica debe interpretarse como evidencia complementaria: apoya la hipótesis de especialización semántica, pero no sustituye las métricas directas de factualidad, claridad, trazabilidad, completitud, latencia y tokens.

## Especificación visual

Las figuras estilizadas se ubican en `paper_visuals/figures/` y se generan desde `paper_visuals/scripts/create_paper_figures.py`. El diseño visual prioriza legibilidad, reproducibilidad y sobriedad científica. Todas las figuras se exportan en alta resolución para uso directo en paper, presentación o póster.

| Regla visual | Decisión |
|---|---|
| Paleta | Azul, verde, naranja y violeta con contraste alto y consistencia por patrón. |
| Tipografía | Sans serif de alta legibilidad; tamaños grandes para ejes, títulos y anotaciones. |
| Resolución | PNG a 300 dpi para publicación y revisión editorial. |
| Composición | Fondos blancos, grilla ligera, anotaciones cuantitativas y leyendas externas cuando mejoran lectura. |
| Semántica | Calidad se representa hacia arriba o hacia la derecha; coste se representa como penalización o eje separado. |

## Figuras científicas propuestas

| Archivo | Pregunta científica que responde |
|---|---|
| `paper_visuals/figures/paper_cost_quality_frontier.png` | ¿Qué patrones están más cerca de una frontera calidad-coste defendible? |
| `paper_visuals/figures/paper_net_utility.png` | ¿Qué patrón maximiza utilidad al penalizar coste matemáticamente? |
| `paper_visuals/figures/paper_cost_decomposition.png` | ¿Qué componente de coste explica la penalización de cada patrón? |
| `paper_visuals/figures/paper_hypothesis_matrix.png` | ¿Cómo se resume la evidencia H1-H5 en una figura compacta para el paper? |
| `paper_visuals/figures/paper_quality_cost_radar.png` | ¿Cómo se comparan calidad, eficiencia de tokens, eficiencia temporal y decisión segura? |
| `paper_visuals/figures/paper_semantic_embedding_3d.png` | ¿Cómo cambia la ubicación semántica de las respuestas por patrón agentic? |
| `paper_visuals/figures/paper_semantic_geometry_metrics.png` | ¿Qué patrón presenta mejor separación geométrica relativa frente a su dispersión interna? |

## Referencias

[1]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[2]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"
[3]: https://doi.org/10.1137/1.9780898719769.ch5 "Latent Semantic Indexing via Singular Value Decomposition"
