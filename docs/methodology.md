# Metodología experimental

Este proyecto implementa un **benchmark controlado y reproducible** para comparar un baseline de RAG mono-agente con tres patrones multi-agente: **Parent-Child**, **Mixture-of-Agents** y **Handoff**. La evaluación se centra en tareas empresariales donde la respuesta debe balancear factualidad, claridad, trazabilidad, completitud, coste de tokens y latencia.

> La unidad experimental es una pregunta empresarial con evidencia documental explícita, hechos esperados, citas esperadas y una decisión esperada de respuesta, aclaración o rechazo.

## Diseño

El corpus incluye documentos de finanzas, seguridad, operaciones, gobierno, talento humano, experiencia de cliente, estrategia y legal. Cada documento contiene hechos atómicos identificables, lo que permite medir cobertura y factualidad sin depender de juicios subjetivos. Esta decisión hace que el experimento sea defendible, auditable y repetible en entornos donde no se pueden publicar datos empresariales reales.

| Componente | Implementación | Justificación |
|---|---|---|
| Recuperación | Similitud lexical ponderada por dominio | Determinismo y trazabilidad del ranking documental. |
| Generación | Síntesis extractiva controlada por hechos | Permite medir factualidad y completitud de forma reproducible. |
| Patrones | Single-Agent, Parent-Child, Mixture-of-Agents, Handoff | Cobertura directa de H1-H5. |
| Métricas | Factualidad, claridad, trazabilidad, completitud, alucinación, decisión, latencia y tokens | Balance entre calidad científica y coste operativo. |

## Métricas

La **factualidad** se calcula como proporción de afirmaciones soportadas sobre afirmaciones totales. La **completitud** corresponde a la proporción de hechos esperados cubiertos por la respuesta. La **trazabilidad** combina coincidencia de citas esperadas y pasos de trazabilidad registrados en la ejecución. La **claridad** se estima mediante concisión y densidad de evidencia. Para preguntas ambiguas o sensibles, la **exactitud de decisión** mide si el sistema responde, solicita aclaración o rechaza según lo esperado.

| Métrica | Rango | Interpretación |
|---|---:|---|
| factuality | 0-1 | Mayor valor indica menos afirmaciones no soportadas. |
| completeness | 0-1 | Mayor valor indica más hechos esperados cubiertos. |
| traceability | 0-1 | Mayor valor indica mejor evidencia y trazas. |
| hallucination_rate | 0-1 | Menor valor indica menos contenido no soportado. |
| total_tokens | entero | Aproximación determinista al coste de consumo. |
| latency_ms | ms | Latencia estimada por complejidad del patrón. |

## Relación con hipótesis

H1 se evalúa comparando índices de calidad de patrones multi-agente contra Single-Agent y observando el coste adicional en tokens y latencia. H2 se evalúa mediante el índice balanceado de Parent-Child en preguntas de complejidad media. H3 se evalúa con completitud en preguntas de síntesis. H4 se evalúa con la exactitud de decisión y la tasa de alucinación en preguntas ambiguas o sensibles. H5 se evalúa con desempeño en preguntas simples frente a coste.

## Referencias

[1]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[2]: https://arxiv.org/abs/2402.01680 "Mixture-of-Agents Enhances Large Language Model Capabilities"
[3]: https://arxiv.org/abs/2308.08155 "A Survey on Large Language Model based Autonomous Agents"
[4]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"
