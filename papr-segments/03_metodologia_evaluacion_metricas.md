# Metodología de evaluación y métricas

**Autor:** manularrea  
**Segmento:** Evaluación de retrieval, generación, decisión y trazabilidad  
**Uso sugerido:** Sección de metodología de evaluación y validez estadística

## Principio metodológico

La evaluación de un sistema RAG debe separar los fallos de recuperación de los fallos de generación. Si el retriever no recupera evidencia suficiente, el generador puede producir una respuesta incompleta aunque el razonamiento sea adecuado; si el retriever recupera evidencia correcta pero el generador inventa, omite o distorsiona información, el fallo está en la etapa generativa. Por esta razón, el experimento reporta métricas de factualidad, completitud, trazabilidad, claridad, alucinación, exactitud de decisión, latencia y tokens, y recomienda complementar el benchmark con métricas RAGAS como `context precision`, `context recall`, `answer relevancy` y `faithfulness` [1] [2].

> La pregunta central no es únicamente “qué patrón responde mejor”, sino **qué componente del sistema explica la mejora o el fallo**: recuperación, generación, orquestación, decisión de seguridad o coste operativo.

## Métricas implementadas en el repositorio

El repositorio calcula métricas deterministas usando el corpus y las respuestas ideales. Las métricas se exportan en `results/detailed_metrics.csv` para análisis por pregunta y en `results/aggregate_metrics.csv` para análisis por patrón. La tabla siguiente resume el rol de cada métrica en la evaluación de hipótesis.

| Métrica | Definición operacional | Hipótesis donde aporta evidencia |
|---|---|---|
| Factualidad | Proporción de afirmaciones soportadas por hechos del corpus. | H1, H2, H4, H5. |
| Completitud | Cobertura de hechos requeridos por la pregunta. | H1, H2, H3, H5. |
| Trazabilidad | Cobertura de citas esperadas y riqueza de pasos de orquestación. | H1, H2, H4. |
| Claridad | Concisión y densidad de evidencia en la respuesta. | H1, H2, H5. |
| Alucinación | Presencia de contenido no soportado o decisión inadecuada. | H1, H4. |
| Exactitud de decisión | Coincidencia con `answer`, `clarify` o `refuse`. | H4. |
| Latencia | Penalización temporal estimada por patrón. | H1, H2, H3, H5. |
| Tokens | Aproximación de consumo textual por ejecución. | H1, H2, H3, H5. |

## Métricas RAGAS y extensión recomendada

RAGAS propone métricas automáticas que evalúan componentes específicos de sistemas RAG, especialmente el contexto recuperado y la fidelidad de la respuesta respecto a ese contexto [1] [2]. Aunque el repositorio actual usa evaluación determinista sobre hechos atómicos, las métricas RAGAS son una extensión natural para corpora más grandes o respuestas generadas por LLMs reales.

| Métrica RAGAS | Qué mide | Equivalente o complemento en el repositorio |
|---|---|---|
| `context_precision` | Si los contextos recuperados relevantes aparecen en posiciones útiles. | Complementa trazabilidad y citas esperadas. |
| `context_recall` | Si el contexto recuperado cubre información necesaria para la respuesta ideal. | Complementa completitud de hechos. |
| `answer_relevancy` | Si la respuesta responde la pregunta del usuario. | Complementa claridad y decisión esperada. |
| `faithfulness` | Si las afirmaciones de la respuesta están soportadas por el contexto. | Complementa factualidad y alucinación. |

Para integrar RAGAS en una versión posterior, cada fila de evaluación debe incluir pregunta, respuesta, contextos recuperados y ground truth. En ese escenario, se recomienda conservar las métricas deterministas existentes como evaluación primaria y usar RAGAS como evaluación secundaria de robustez, porque los evaluadores automáticos también pueden introducir variabilidad.

## Métricas humanas y LLM-as-judge

En un paper científico, la evaluación automática debe complementarse cuando las dimensiones evaluadas son subjetivas, por ejemplo claridad, utilidad ejecutiva, calidad argumentativa o calidad de citas. Un protocolo humano puede asignar puntuaciones ciegas por respuesta sin revelar el patrón, mientras que un evaluador LLM-as-judge puede actuar como segunda señal siempre que se controle temperatura, prompt, orden de respuestas y consistencia inter-evaluador.

| Dimensión | Evaluador recomendado | Escala sugerida | Control de sesgo |
|---|---|---:|---|
| Claridad ejecutiva | Humano experto o LLM-as-judge | 1 a 5 | Presentar respuestas anonimizadas. |
| Utilidad empresarial | Humano experto | 1 a 5 | Definir rúbrica por dominio. |
| Calidad de citas | Automático + humano | 0 a 1 o 1 a 5 | Verificar documento y hecho citado. |
| Seguridad de decisión | Automático + experto de riesgo | 0 a 1 | Comparar contra `expected_decision`. |
| Coherencia multi-documento | Humano experto | 1 a 5 | Evaluar cobertura sin premiar verbosidad. |

El uso de LLM-as-judge debe reportarse de forma transparente. Debe indicarse modelo, versión, prompt, temperatura, número de corridas, criterio de desempate y si el juez tuvo acceso al contexto. Esta transparencia evita que la evaluación se convierta en una caja negra adicional.

## Agregación por categoría de pregunta y patrón

Las métricas se agregan primero por pregunta y patrón, luego por tipo de pregunta y finalmente por patrón global. Esta jerarquía evita que una métrica global oculte el comportamiento especializado de cada harness. Por ejemplo, Mixture-of-Agents puede ser superior en síntesis pero innecesariamente costoso en preguntas simples; Handoff puede parecer menos completo globalmente, pero ser el único correcto en preguntas sensibles.

| Nivel de agregación | Fórmula conceptual | Uso |
|---|---|---|
| Por caso | \(m_{p,q}\) | Métrica del patrón \(p\) sobre pregunta \(q\). |
| Por categoría | \(\bar{m}_{p,c}=\frac{1}{|Q_c|}\sum_{q\in Q_c}m_{p,q}\) | Comparar patrones dentro de `simple`, `medium`, `synthesis`, `ambiguous` y `sensitive`. |
| Global por patrón | \(\bar{m}_{p}=\frac{1}{|Q|}\sum_{q\in Q}m_{p,q}\) | Comparar rendimiento promedio de cada patrón. |
| Diferencia frente a baseline | \(\Delta m_p=\bar{m}_{p}-\bar{m}_{single}\) | Medir ganancia incremental de multi-agente. |

El paper debe reportar tanto métricas globales como métricas estratificadas por categoría. Esta práctica es esencial para defender H5, porque el resultado esperado es que Single-Agent sea suficiente en preguntas simples aunque otros patrones sean superiores en escenarios complejos.

## Control de varianza e inferencia estadística

Aunque el benchmark actual es pequeño y determinista, el paper debe anticipar extensiones con múltiples corridas, LLMs no deterministas o corpora ampliados. En ese caso, se recomienda reportar intervalos de confianza bootstrap, tamaños de efecto y pruebas pareadas por pregunta. Las pruebas pareadas son más adecuadas que comparaciones independientes porque cada patrón responde el mismo conjunto de preguntas.

| Técnica | Uso recomendado | Interpretación |
|---|---|---|
| Bootstrap por pregunta | Estimar intervalo de confianza de la media o diferencia. | Robustez de métricas agregadas. |
| Test pareado | Comparar patrón multi-agente contra Single-Agent por pregunta. | Evidencia de mejora consistente. |
| Tamaño de efecto | Cuantificar magnitud de diferencia. | Evita depender solo de significancia. |
| Análisis por categoría | Evaluar heterogeneidad de efectos. | Identifica cuándo conviene cada patrón. |

Una formulación mínima para el intervalo bootstrap de una diferencia \(\Delta\) consiste en re-muestrear preguntas con reemplazo, recalcular \(\Delta_b\) para cada muestra y reportar percentiles 2.5 y 97.5. Si el intervalo no cruza cero para una métrica de calidad, se obtiene soporte estadístico adicional para la mejora del patrón.

## Relación entre métricas y las hipótesis H1-H5

| Hipótesis | Métricas primarias | Métricas de coste | Evidencia visual recomendada |
|---|---|---|---|
| H1 | Calidad, factualidad, claridad, trazabilidad. | Tokens y latencia. | `paper_cost_quality_frontier.png`, `paper_hypothesis_matrix.png`. |
| H2 | Trazabilidad, completitud y calidad en `medium`. | Coste compuesto y utilidad neta. | `paper_net_utility.png`, `balanced_index.png`. |
| H3 | Completitud en `synthesis`. | Tokens, latencia y llamadas. | `paper_task_type_completeness_heatmap.png`. |
| H4 | Exactitud de decisión y alucinación en alto riesgo. | Coste de rechazo y latencia. | `hypothesis_evidence_heatmap.png`. |
| H5 | Factualidad y completitud en `simple`. | Tokens mínimos y latencia mínima. | `cost_latency_tradeoff.png`. |

## Recomendación de reporte

El paper debe presentar cada métrica con una definición formal, una interpretación operacional y una justificación de por qué se vincula con las hipótesis. Además, debe evitar afirmar que un patrón es universalmente superior. La conclusión defendible es condicional: **Mixture-of-Agents** maximiza completitud en síntesis, **Parent-Child** ofrece balance en complejidad media, **Handoff** controla riesgo y **Single-Agent** es competitivo en tareas simples.

## Referencias

[1]: https://docs.ragas.io/ "RAGAS Documentation"  
[2]: https://arxiv.org/abs/2309.15217 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"  
[3]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"  
[4]: https://arxiv.org/abs/2407.12831 "A Survey on Evaluation of Large Language Models"  
[5]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"
