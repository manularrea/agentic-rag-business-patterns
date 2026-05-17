# Implementación de los harness y pipeline

**Autor:** manularrea  
**Segmento:** Implementación reproducible de patrones Agentic RAG  
**Uso sugerido:** Sección de arquitectura experimental, implementación y reproducibilidad

## Propósito de la implementación

La implementación del repositorio convierte los cuatro patrones evaluados en harness comparables, ejecutables y medibles. Cada harness recibe una pregunta, consulta el mismo corpus, genera una respuesta con evidencia y registra métricas operativas. El objetivo es aislar el efecto del patrón de orquestación: un **Single-Agent RAG** como baseline, un **Parent-Child** con delegación controlada, un **Mixture-of-Agents** con especialistas paralelos y agregador, y un **Handoff** orientado a transferencia explícita o rechazo seguro.

> La comparación es válida porque cada patrón opera sobre el mismo corpus, el mismo conjunto de preguntas, el mismo evaluador y la misma lógica de exportación. Lo que cambia es la estructura de control, no el dataset.

## Pipeline general

El pipeline experimental sigue una secuencia común para todos los harness. Primero se cargan los documentos y preguntas. Luego se inicializa el recuperador y se ejecutan los patrones sobre cada pregunta. Después se evalúan factualidad, completitud, trazabilidad, claridad, alucinación, exactitud de decisión, tokens y latencia. Finalmente se exportan resultados detallados y agregados para análisis estadístico y visual.

| Etapa | Entrada | Salida | Función metodológica |
|---|---|---|---|
| Carga de corpus | `data/corpus.json` | Documentos con hechos atómicos | Garantizar evidencia auditable. |
| Carga de preguntas | `data/questions.json` | Consultas con metadatos | Definir tareas, dominios, riesgo y respuesta ideal. |
| Recuperación | Pregunta + corpus | Documentos candidatos | Seleccionar evidencia base para generación. |
| Orquestación | Evidencia + patrón | Respuesta, citas y traza | Ejecutar Single-Agent, Parent-Child, Mixture-of-Agents o Handoff. |
| Evaluación | Respuesta + verdad esperada | Métricas por caso | Medir calidad, seguridad y coste. |
| Exportación | Métricas por caso | CSV y figuras | Facilitar reproducibilidad y análisis del paper. |

## Harness Single-Agent RAG

El harness **Single-Agent RAG** representa el baseline. Recupera documentos relevantes y genera una respuesta única a partir del contexto disponible. Es útil para preguntas simples o de bajo riesgo, porque evita sobrecarga de coordinación y minimiza latencia y tokens. En el paper, este patrón opera como punto de comparación contra el cual se evalúa si los sistemas multi-agente justifican su coste adicional.

| Elemento | Diseño del baseline |
|---|---|
| Orquestación | Una sola unidad de razonamiento. |
| Recuperación | Búsqueda directa sobre corpus compartido. |
| Decisión | Responder por defecto si encuentra evidencia suficiente. |
| Trazabilidad | Citas recuperadas y pasos mínimos. |
| Riesgo esperado | Puede responder cuando debería aclarar o rechazar si no existe lógica explícita de handoff. |

## Harness Parent-Child

El patrón **Parent-Child** introduce un orquestador padre que delega subtareas a agentes especializados por dominio. La delegación es controlada: el padre conserva la responsabilidad de consolidar la respuesta final, mientras los hijos recuperan o sintetizan evidencia específica. Este patrón está alineado con la hipótesis H2, porque busca mejorar calidad y trazabilidad sin activar una agregación paralela excesiva.

| Componente | Responsabilidad | Efecto esperado |
|---|---|---|
| Parent | Recibe pregunta, identifica dominios, coordina subtareas y consolida. | Mejora coherencia global y control de salida. |
| Child especializado | Atiende dominio o subproblema asignado. | Mejora cobertura de hechos relevantes. |
| Delegación controlada | Limita agentes a dominios esperados o recuperados. | Reduce expansión innecesaria de tokens. |
| Consolidación | Integra citas y hechos de los hijos. | Mejora trazabilidad y completitud en tareas medias. |

La decisión técnica más importante es que Parent-Child no intenta maximizar exhaustividad en todos los documentos, sino cubrir los dominios necesarios con una coordinación más económica que Mixture-of-Agents. En tareas de complejidad media, esta restricción favorece el equilibrio calidad-coste.

## Harness Mixture-of-Agents

El patrón **Mixture-of-Agents** ejecuta múltiples agentes en paralelo y posteriormente agrega sus respuestas. Este diseño se inspira en enfoques donde la diversidad de agentes o modelos puede mejorar la respuesta final al combinar perspectivas complementarias [1]. En el experimento, Mixture-of-Agents se espera especialmente fuerte en preguntas de síntesis multi-documento, donde la completitud depende de recuperar y combinar evidencia dispersa.

| Componente | Responsabilidad | Efecto esperado |
|---|---|---|
| Agentes paralelos | Exploran dominios, evidencias o perspectivas. | Aumentan cobertura de hechos. |
| Agregador | Consolida resultados, elimina duplicados y organiza citas. | Mejora completitud y claridad final. |
| Registro de trazas | Conserva contribuciones por agente. | Mejora auditabilidad del proceso. |
| Coste operativo | Multiplica llamadas, tokens y latencia. | Penaliza eficiencia frente a Single-Agent y Parent-Child. |

La principal contrapartida de este patrón es que el paralelismo lógico no elimina el coste total de cómputo. Aunque algunas ejecuciones puedan realizarse concurrentemente, el consumo de tokens y la complejidad de coordinación aumentan, lo que afecta la utilidad neta.

## Harness Handoff

El patrón **Handoff** incorpora transferencia explícita de control. Cuando una pregunta es ambigua, sensible o pertenece a un dominio de alto riesgo, el harness puede delegarla a una lógica especializada, solicitar aclaración o rechazar. Esta conducta es esencial para H4, porque un buen sistema empresarial no debe responder siempre: debe distinguir entre responder, clarificar y bloquear.

| Condición | Acción esperada | Ejemplo experimental |
|---|---|---|
| Pregunta clara y de bajo riesgo | Responder con evidencia. | Consulta simple de finanzas u operaciones. |
| Pregunta ambigua | Solicitar aclaración. | “Haz lo necesario con los datos del cliente”. |
| Pregunta sensible o insegura | Rechazar y citar política. | Solicitud para extraer credenciales internas. |
| Dominio regulado | Priorizar trazabilidad y evidencia. | Legal, talento humano o gobierno. |

El handoff reduce alucinaciones en escenarios de alto riesgo porque evita fabricar una respuesta cuando la consulta no está suficientemente especificada o viola controles documentados. Este comportamiento es consistente con marcos de gestión de riesgo de IA que recomiendan gobernanza, monitoreo y mitigación proporcional al contexto [2].

## Instrumentación de ejecuciones

Cada ejecución registra métricas de calidad y coste. La instrumentación no se limita al texto final: también captura citas, hechos usados, decisión, latencia estimada, tokens y traza. Esto permite comparar no solo qué patrón responde mejor, sino también cuánto cuesta llegar a esa respuesta y qué tan auditable es el camino.

| Señal instrumentada | Definición | Uso en el análisis |
|---|---|---|
| Latencia | Tiempo estimado o medido por ejecución. | Evaluar penalización temporal por orquestación. |
| Tokens | Aproximación de tokens consumidos por entrada, salida y trazas. | Estimar coste económico y complejidad. |
| Llamadas | Número de pasos/agentes invocados. | Medir multiplicación operacional en patrones complejos. |
| Citas | Documentos usados en la respuesta. | Calcular trazabilidad y soporte factual. |
| Hechos cubiertos | Hechos requeridos presentes en la respuesta. | Calcular completitud. |
| Decisión | `answer`, `clarify` o `refuse`. | Evaluar seguridad y adecuación. |
| Traza | Secuencia de pasos de orquestación. | Auditar delegaciones, agregaciones y handoff. |

## Configuración del modelo base y retriever

La versión actual es determinista y no depende de llamadas externas a LLMs, lo cual fortalece reproducibilidad. En lugar de una API generativa no determinista, el repositorio implementa generación controlada por reglas, hechos esperados y documentos recuperados. Esta decisión permite que las pruebas, métricas y visualizaciones se regeneren de forma estable. En una extensión con LLM real, deben reportarse familia de modelo, versión, temperatura, `top_p`, semilla, política de reintentos, límites de tokens y fecha de ejecución.

| Parámetro | Configuración actual | Requisito si se usa LLM externo |
|---|---|---|
| Familia de LLM | Generador determinista local. | Nombre exacto del modelo y proveedor. |
| Temperatura | No aplica; ejecución determinista. | Recomendado: 0.0 para evaluación comparativa. |
| `top_p` | No aplica. | Reportar valor exacto. |
| Semilla | No aplica o fija por implementación. | Reportar semilla y variabilidad entre corridas. |
| Retriever | Lexical determinista. | Reportar modelo de embeddings, dimensión, normalización y scoring. |
| Scoring | Coincidencia lexical y dominio. | Definir BM25, dense retrieval, híbrido o reranker. |

La documentación de parámetros es crítica porque pequeñas variaciones en recuperación o generación pueden modificar las métricas RAG. Por ello, cualquier versión del paper que incorpore LLM-as-judge, embeddings comerciales o APIs externas debe registrar precios, versión de modelo y fecha de consulta.

## Archivos relevantes del repositorio

| Archivo o carpeta | Rol |
|---|---|
| `scripts/run_experiment.py` | Punto de entrada para ejecutar el benchmark. |
| `src/agentic_rag/` | Paquete Python con patrones, recuperación, métricas y utilidades. |
| `data/corpus.json` | Corpus controlado de documentos empresariales. |
| `data/questions.json` | Preguntas con metadatos y respuestas ideales. |
| `results/detailed_metrics.csv` | Métricas por pregunta y patrón. |
| `results/aggregate_metrics.csv` | Métricas agregadas por patrón. |
| `paper_visuals/scripts/create_paper_figures.py` | Generación de figuras editoriales y métricas matemáticas. |

## Referencias

[1]: https://arxiv.org/abs/2402.01680 "Mixture-of-Agents Enhances Large Language Model Capabilities"  
[2]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"  
[3]: https://arxiv.org/abs/2308.08155 "A Survey on Large Language Model based Autonomous Agents"  
[4]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
