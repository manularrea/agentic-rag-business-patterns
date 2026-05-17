# Ecuaciones de coste y eficiencia

**Autor:** manularrea  
**Segmento:** Modelo matemático de coste, eficiencia y utilidad neta  
**Uso sugerido:** Sección de análisis económico, eficiencia computacional y discusión de trade-offs

## Motivación

Los patrones multi-agente pueden mejorar completitud, trazabilidad o seguridad, pero esa mejora no es gratuita. Cada subagente, agregador, reintento, verificación de seguridad o transferencia de control incrementa tokens, latencia, número de llamadas o complejidad de implementación. Por esta razón, el paper debe evaluar la calidad junto con coste operativo. La conclusión defendible no es que un patrón multi-agente sea siempre superior, sino que su uso se justifica cuando la ganancia marginal de calidad, seguridad o completitud supera la penalización de coste.

> En sistemas Agentic RAG, el coste relevante no es solo el precio de una llamada al modelo; también incluye latencia, número de pasos, reintentos, llamadas externas, complejidad de trazas y coste de mantenimiento.

## Coste monetario por consulta basado en tokens

La fórmula estándar de coste por consulta suma el coste de tokens de entrada y tokens de salida. Si \(T^{in}_{p,q}\) representa los tokens de entrada del patrón \(p\) en la pregunta \(q\), \(T^{out}_{p,q}\) representa tokens de salida, \(P_{in}\) es el precio por mil tokens de entrada y \(P_{out}\) el precio por mil tokens de salida, entonces:

\[
Coste_{p,q}^{USD}=\frac{T^{in}_{p,q}}{1000}P_{in}+\frac{T^{out}_{p,q}}{1000}P_{out}
\]

Por ejemplo, si una petición consume 1 000 tokens de entrada a 0.01 USD por mil tokens y 500 tokens de salida a 0.03 USD por mil tokens, entonces:

\[
Coste=\frac{1000}{1000}(0.01)+\frac{500}{1000}(0.03)=0.01+0.015=0.025\;USD
\]

Esta ecuación debe aplicarse por patrón y por pregunta. En patrones como Mixture-of-Agents, el coste total debe sumar todas las llamadas de especialistas y agregador, no solamente la respuesta final visible al usuario.

| Variable | Definición | Unidad |
|---|---|---|
| \(T^{in}_{p,q}\) | Tokens de entrada acumulados por patrón y pregunta. | Tokens. |
| \(T^{out}_{p,q}\) | Tokens de salida acumulados por patrón y pregunta. | Tokens. |
| \(P_{in}\) | Precio por mil tokens de entrada del modelo usado. | USD / 1K tokens. |
| \(P_{out}\) | Precio por mil tokens de salida del modelo usado. | USD / 1K tokens. |
| \(N_{calls}\) | Número de llamadas internas o externas. | Conteo. |

## Coste acumulado en cadenas de agentes

En arquitecturas agentic, el consumo crece con la cantidad de pasos. Si \(A_p\) es el conjunto de agentes o pasos invocados por el patrón \(p\), el coste acumulado se modela como:

\[
Coste_{p,q}^{total}=\sum_{a\in A_p}\left(\frac{T^{in}_{a,q}}{1000}P_{in,a}+\frac{T^{out}_{a,q}}{1000}P_{out,a}\right)+Coste_{APIs}+Coste_{retry}
\]

Esta formulación captura el efecto multiplicador de Parent-Child y Mixture-of-Agents. Parent-Child agrega coste por delegación a hijos especializados; Mixture-of-Agents agrega coste por paralelismo y agregación; Handoff puede reducir tokens en ciertos casos porque no genera una respuesta extensa cuando debe aclarar o rechazar, pero añade coste lógico por clasificación de riesgo.

| Patrón | Fuente principal de coste | Riesgo económico |
|---|---|---|
| Single-Agent | Una recuperación y una generación. | Bajo coste, pero menor robustez en tareas complejas. |
| Parent-Child | Delegaciones controladas a especialistas. | Coste moderado si se limita el número de hijos. |
| Mixture-of-Agents | Varios agentes paralelos y agregador. | Alto coste por multiplicación de tokens y llamadas. |
| Handoff | Clasificación, transferencia o rechazo. | Coste bajo-medio; eficiente cuando evita respuestas inseguras largas. |

## Métricas derivadas de eficiencia

Además del coste monetario, el repositorio reporta métricas de eficiencia que permiten comparar configuraciones sin depender de un proveedor específico de LLM. Estas métricas son apropiadas para un experimento reproducible y pueden convertirse a USD cuando se elijan precios reales.

| Métrica | Fórmula | Interpretación |
|---|---|---|
| Coste medio por pregunta | \(\bar{C}_{p}=\frac{1}{|Q|}\sum_{q\in Q}C_{p,q}\) | Coste esperado del patrón por consulta. |
| Tokens por respuesta | \(\bar{T}_{p}=\frac{1}{|Q|}\sum_{q\in Q}(T^{in}_{p,q}+T^{out}_{p,q})\) | Verbosidad y carga computacional. |
| Latencia media | \(\bar{L}_{p}=\frac{1}{|Q|}\sum_{q\in Q}L_{p,q}\) | Tiempo esperado de respuesta. |
| Llamadas medias | \(\bar{N}_{p}=\frac{1}{|Q|}\sum_{q\in Q}N_{p,q}\) | Complejidad operacional. |
| Calidad por token | \(QPT_p=\frac{Q_p}{\bar{T}_p}\) | Eficiencia semántica de la respuesta. |
| Calidad por latencia | \(QPL_p=\frac{Q_p}{\bar{L}_p}\) | Eficiencia temporal. |

## Coste compuesto normalizado

La carpeta `paper_visuals/` implementa un coste compuesto para comparar patrones de forma visual y matemática. Primero se normalizan tokens, latencia y complejidad de traza con min-max:

\[
\hat{x}_p=\frac{x_p-\min(x)}{\max(x)-\min(x)+\epsilon}
\]

Después se calcula el coste operacional compuesto:

\[
C_p=0.4\hat{T}_p+0.4\hat{L}_p+0.2\hat{S}_p
\]

Aquí \(\hat{T}_p\) es tokens normalizados, \(\hat{L}_p\) es latencia normalizada y \(\hat{S}_p\) es complejidad normalizada de la traza. La ponderación asigna el mismo peso a tokens y latencia porque ambos afectan coste económico y experiencia de usuario, mientras que la complejidad de traza actúa como penalización operativa adicional.

## Utilidad neta

Para decidir si un patrón justifica su coste, se define una utilidad neta:

\[
U_p=Q_p-\lambda C_p
\]

Donde \(Q_p\) es el índice de calidad del patrón y \(\lambda\) es un parámetro de aversión al coste. En la implementación editorial se usa \(\lambda=0.35\), lo que penaliza patrones costosos sin anular mejoras reales de calidad. Si \(U_p\) aumenta frente al baseline, el patrón ofrece una mejora eficiente; si \(Q_p\) aumenta pero \(U_p\) cae, la mejora puede no justificar el coste.

| Resultado posible | Interpretación |
|---|---|
| \(Q_p>Q_{single}\) y \(U_p>U_{single}\) | Mejora técnica y económicamente defendible. |
| \(Q_p>Q_{single}\) y \(U_p\leq U_{single}\) | Mejora de calidad con coste posiblemente excesivo. |
| \(Q_p\approx Q_{single}\) y \(C_p>C_{single}\) | Arquitectura innecesaria para ese tipo de tarea. |
| \(Q_p<Q_{single}\) pero mejor decisión de riesgo | Patrón útil solo en escenarios de seguridad o cumplimiento. |

## Costes ocultos en workflows agentic

Los costes ocultos deben mencionarse en la discusión porque no siempre aparecen en tokens. En despliegues empresariales, un workflow agentic puede requerir monitoreo, trazabilidad, auditoría de herramientas, reintentos, límites de seguridad, pruebas de regresión y mantenimiento de prompts. Estos costes son coherentes con guías de gobernanza y gestión de riesgo de IA, que recomiendan controles durante el ciclo de vida y monitoreo operacional [1].

| Coste oculto | Descripción | Relevancia para el paper |
|---|---|---|
| Latencia p95 | Peor experiencia en colas o tareas complejas. | Afecta viabilidad de producción. |
| Reintentos | Llamadas adicionales por fallos o baja confianza. | Incrementa coste real frente a coste nominal. |
| APIs externas | Herramientas, bases de datos o servicios conectados. | Multiplica puntos de fallo y facturación. |
| Desarrollo | Diseño de agentes, tests, prompts y evaluadores. | Afecta coste total de propiedad. |
| Gobernanza | Auditorías, versionado y revisión humana. | Necesario para dominios regulados. |
| Observabilidad | Logs, trazas y monitoreo. | Permite reproducibilidad y mitigación de riesgo. |

## Figuras recomendadas

Las visualizaciones generadas en `paper_visuals/figures/` conectan estas ecuaciones con resultados. La frontera calidad-coste muestra qué patrones son eficientes; la utilidad neta resume el valor ajustado por coste; la descomposición de coste explica qué componente penaliza a cada patrón.

| Figura | Archivo | Lectura principal |
|---|---|---|
| Frontera calidad-coste | `paper_cost_quality_frontier.png` | Muestra trade-off entre índice de calidad y coste compuesto. |
| Utilidad neta | `paper_net_utility.png` | Identifica patrones con mejor valor ajustado por coste. |
| Descomposición de coste | `paper_cost_decomposition.png` | Separa penalización de tokens, latencia y traza. |
| Radar calidad-coste | `paper_quality_cost_radar.png` | Integra calidad, eficiencia y seguridad en una vista comparativa. |

## Conclusión matemática

El modelo de coste permite defender que la selección de patrón debe formularse como un problema de optimización multiobjetivo. Single-Agent minimiza coste en preguntas simples; Parent-Child maximiza balance en complejidad media; Mixture-of-Agents mejora completitud cuando la síntesis multi-documento vale más que la eficiencia; Handoff optimiza seguridad de decisión en tareas ambiguas o sensibles. Por tanto, el harness complejo se justifica solo cuando el valor marginal \(\Delta Q\) o la reducción de riesgo compensa \(\Delta C\).

## Referencias

[1]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"  
[2]: https://platform.openai.com/docs/pricing "OpenAI API Pricing"  
[3]: https://docs.anthropic.com/en/docs/about-claude/pricing "Anthropic Claude Pricing"  
[4]: https://cloud.google.com/vertex-ai/generative-ai/pricing "Google Vertex AI Generative AI Pricing"  
[5]: https://arxiv.org/abs/2402.01680 "Mixture-of-Agents Enhances Large Language Model Capabilities"
