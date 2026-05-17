# Marco experimental y corpus

**Autor:** manularrea  
**Segmento:** Marco experimental, corpus y recuperación  
**Uso sugerido:** Sección de método experimental del paper

## Objetivo del marco experimental

El marco experimental se diseñó para comparar cuatro configuraciones de Agentic RAG bajo condiciones controladas: **Single-Agent**, **Parent-Child**, **Mixture-of-Agents** y **Handoff**. La comparación se fundamenta en el principio de Retrieval-Augmented Generation, donde la respuesta generativa se condiciona por evidencia recuperada desde una base documental, reduciendo la dependencia de memoria paramétrica y permitiendo trazabilidad hacia documentos fuente [1]. En este repositorio, el foco no es maximizar el tamaño del corpus, sino crear un banco auditable de hechos empresariales que permita medir factualidad, completitud, decisión, trazabilidad, coste y latencia de forma reproducible.

> El corpus actúa como una cámara de prueba controlada: todos los patrones reciben exactamente el mismo conjunto documental y el mismo conjunto de preguntas, por lo que las diferencias observadas se atribuyen al patrón de orquestación y no a variaciones de datos.

## Construcción del corpus

El corpus se almacena en `data/corpus.json` y contiene **8 documentos empresariales**, cada uno asociado con un dominio funcional, un identificador documental, un título, un texto fuente y un diccionario de hechos atómicos verificables. Esta estructura permite evaluar si una respuesta recupera y utiliza los hechos correctos, y también permite calcular métricas de soporte documental por cita.

| Dimensión | Valor implementado | Justificación científica |
|---|---:|---|
| Número de documentos | 8 | Suficiente para cubrir dominios empresariales heterogéneos y permitir síntesis multi-documento. |
| Hechos por documento | 3 | Cada documento contiene hechos discretos que pueden verificarse automáticamente. |
| Total de hechos atómicos | 24 | Permite evaluar cobertura, omisiones y alucinaciones por patrón. |
| Formato | JSON estructurado | Facilita reproducibilidad, versionado y evaluación determinista. |
| Unidad documental | Documento corto por dominio | Reduce ruido experimental y permite atribución clara de evidencias. |

Los dominios incluidos cubren finanzas, seguridad, operaciones, gobierno, talento humano, experiencia de cliente, estrategia y legal. Esta selección representa tareas empresariales donde los agentes RAG suelen requerir factualidad, control de riesgo, recuperación de fuentes y capacidad de síntesis transversal.

| Dominio | Documento | Tipo de evidencia contenida |
|---|---|---|
| Finanzas | `DOC_FIN_01` | Umbrales de aprobación, conciliación diaria y auditoría de excepciones. |
| Seguridad | `DOC_SEC_01` | Mínimo privilegio, auditoría de acciones y rechazo de exfiltración. |
| Operaciones | `DOC_OPS_01` | Tiempo de ciclo, rollback documentado y latencia p95. |
| Gobierno | `DOC_GOV_01` | Fuente, versión de prompt, responsable, fecha, justificación y explicabilidad. |
| Talento humano | `DOC_HR_01` | Sesgo, revisión humana y prohibición de disciplina automática. |
| Experiencia de cliente | `DOC_CX_01` | Consistencia, escalamiento humano y citas de políticas internas. |
| Estrategia | `DOC_STR_01` | Valor esperado, riesgo, datos, integración y complejidad media. |
| Legal | `DOC_LEG_01` | Minimización, cifrado, base legal y rechazo de secretos. |

## Generación del conjunto de preguntas

El conjunto de preguntas se almacena en `data/questions.json` y contiene **10 preguntas** con metadatos explícitos. Cada pregunta define su identificador, texto, tipo de tarea, nivel de riesgo, dominios relevantes, hechos requeridos, citas esperadas y decisión esperada. Esta estructura convierte cada consulta en un caso de prueba evaluable, no solamente en una interacción abierta.

| Tipo de pregunta | Cantidad | Riesgo predominante | Propósito experimental |
|---|---:|---|---|
| `simple` | 2 | Bajo | Validar si un RAG mono-agente bien ajustado es suficiente para tareas directas. |
| `medium` | 2 | Medio | Medir si Parent-Child logra equilibrio entre delegación, trazabilidad y coste. |
| `synthesis` | 2 | Medio | Evaluar si Mixture-of-Agents mejora completitud multi-documento. |
| `ambiguous` | 2 | Alto | Evaluar si Handoff solicita aclaración en consultas insuficientemente especificadas. |
| `sensitive` | 2 | Alto | Evaluar si Handoff rechaza solicitudes inseguras o no permitidas. |

La variable `expected_decision` codifica si el sistema debe responder, pedir aclaración o rechazar. Esta señal es clave para H4, porque una arquitectura segura no solo debe responder correctamente cuando existe evidencia, sino también detectar cuándo responder sería riesgoso o inválido. La literatura sobre gestión de riesgo de IA recomienda controles proporcionales al contexto y mecanismos de gobernanza que reduzcan resultados inseguros en dominios sensibles [5].

## Control de comparabilidad entre patrones

La validez interna del experimento depende de que los cuatro harness operen sobre el mismo corpus, las mismas preguntas y el mismo mecanismo de evaluación. En el repositorio, la ejecución experimental centraliza la carga de `data/corpus.json` y `data/questions.json`, y luego ejecuta cada patrón sobre la misma lista de consultas. Esto evita que una variante reciba más información o un subconjunto distinto de preguntas.

| Control | Implementación | Riesgo mitigado |
|---|---|---|
| Corpus único | Todos los patrones leen `data/corpus.json`. | Evita sesgo por diferencias de evidencia disponible. |
| Preguntas únicas | Todos los patrones leen `data/questions.json`. | Evita sesgo por tareas no comparables. |
| Métricas compartidas | La evaluación usa el mismo módulo de métricas. | Evita criterios de scoring específicos por patrón. |
| Registro de trazas | Cada patrón produce pasos o decisiones trazables. | Permite auditar la fuente de mejoras o fallos. |
| Exportación común | Resultados en `results/*.csv`. | Facilita análisis agregado y visualización reproducible. |

## Índices, fragmentación y recuperación

El repositorio implementa un recuperador reproducible de tipo lexical (`KeywordRetriever`) sobre documentos cortos. En esta versión, cada documento funciona como una unidad semántica completa y no se fragmenta en chunks adicionales. Esta decisión es deliberada: como cada documento contiene tres hechos atómicos y un dominio claro, el chunking por documento preserva contexto suficiente sin introducir solapamientos que puedan complicar la atribución de hechos.

En una extensión con corpus empresarial de mayor escala, el chunking debería parametrizarse con tamaño de ventana, solapamiento, normalización, identificador de documento padre y metadatos de dominio. En sistemas RAG reales, la calidad del retrieval depende tanto de la segmentación como del scoring y del tamaño de `k`, porque el generador solo puede fundamentarse en la evidencia que llega al contexto [2] [3].

| Parámetro | Valor actual | Extensión recomendada para corpus real |
|---|---|---|
| Unidad de indexación | Documento completo | Chunks de 300 a 800 tokens con solapamiento controlado. |
| Modelo de embeddings | No aplica en recuperación base; se usa lexical determinista. | Embeddings densos con modelo documentado y versionado. |
| Scoring | Coincidencia lexical y dominio cuando corresponde. | Scoring híbrido BM25 + embeddings densos. |
| `k` | Parametrizable por patrón/harness. | Reportar `k`, umbral de similitud y estrategia de reranking. |
| Metadatos | `id`, `title`, `domain`, `facts`. | Añadir fuente, fecha, versión, confidencialidad y propietario. |

## Relación con la geometría semántica 3D

La visualización de geometría semántica integrada en `paper_visuals/figures/paper_semantic_embedding_3d.png` no sustituye al recuperador; es una capa analítica posterior. En esa capa, preguntas, respuestas, decisiones, citas, trazas y métricas se transforman en representaciones TF-IDF y se proyectan a 3D mediante reducción SVD. Esto permite observar si cada patrón ocupa regiones semánticas distintas y calcular métricas como compacidad, separación inter-centroide y desplazamiento respecto al baseline.

| Métrica geométrica | Interpretación para el corpus |
|---|---|
| Compacidad intra-patrón | Qué tan homogéneas son las respuestas de un patrón ante tareas heterogéneas. |
| Separación inter-patrón | Qué tan distinguible es el comportamiento semántico de cada patrón. |
| Desplazamiento vs Single-Agent | Qué tanto cambia la salida cuando se introduce orquestación agentic. |
| Cociente separación/compacidad | Indicador de diferenciación semántica ajustada por dispersión. |

## Reproducibilidad

Para reproducir el marco experimental completo se deben ejecutar los siguientes comandos desde la raíz del repositorio:

```bash
pip install -e .[dev]
python scripts/run_experiment.py
python paper_visuals/scripts/create_paper_figures.py
```

La primera ejecución genera métricas base en `results/`; la segunda genera tablas y figuras editoriales en `paper_visuals/`. Para asegurar comparabilidad, cualquier cambio en corpus, preguntas, `k`, scoring o fragmentación debe registrarse como una nueva versión experimental.

## Referencias

[1]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"  
[2]: https://docs.ragas.io/ "RAGAS Documentation"  
[3]: https://arxiv.org/abs/2309.15217 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"  
[4]: https://doi.org/10.1145/3209978.3210063 "A reproducibility checklist for machine learning research"  
[5]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"
