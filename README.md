# Agentic RAG Business Patterns

Este repositorio contiene un **harness experimental reproducible** para defender hipótesis sobre patrones multi-agente aplicados a RAG empresarial. El proyecto compara un baseline **Single-Agent RAG** contra **Parent-Child**, **Mixture-of-Agents** y **Handoff**, midiendo factualidad, claridad, trazabilidad, completitud, alucinaciones, latencia y consumo de tokens. El diseño se apoya en la literatura de RAG, agentes autónomos y mezclas de agentes [1] [2] [3].

> El objetivo no es maximizar una métrica aislada, sino estimar cuándo conviene usar orquestación multi-agente y cuándo un RAG mono-agente bien afinado es suficiente.

## Hipótesis evaluadas

| Hipótesis | Resultado esperado evaluable | Evidencia generada |
|---|---|---|
| H1 | Los patrones multi-agente mejoran calidad y trazabilidad frente a Single-Agent, con mayor coste. | `results/aggregate_metrics.csv`, radar de calidad y frontera coste-latencia. |
| H2 | Parent-Child ofrece el mejor equilibrio en complejidad media. | `balanced_index` y métricas filtrables por `question_type=medium`. |
| H3 | Mixture-of-Agents mejora completitud en síntesis, con más latencia y tokens. | `completeness_by_question_type.png` y métricas agregadas. |
| H4 | Handoff mejora preguntas ambiguas, sensibles o multi-dominio mediante transferencia o rechazo. | `decision_accuracy`, `hallucination_rate` y trazas por respuesta. |
| H5 | Single-Agent es competitivo en preguntas simples y de bajo riesgo con menor coste. | Detalle por `question_type=simple` en `detailed_metrics.csv`. |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

También puede ejecutarse directamente en un entorno con dependencias instaladas:

```bash
pip install -e .
python scripts/run_experiment.py
```

## Ejecución del benchmark

```bash
agentic-rag-experiment run --output-dir results
```

El comando genera métricas detalladas, métricas agregadas, respuestas trazadas y figuras listas para incluir en un producto científico.

| Archivo | Contenido |
|---|---|
| `results/detailed_metrics.csv` | Métrica por pregunta y patrón. |
| `results/aggregate_metrics.csv` | Métricas promedio por patrón e índices compuestos. |
| `results/answers.csv` | Respuestas, citas, hechos usados y trazas. |
| `results/figures/*.png` | Visualizaciones publicables. |
| `docs/final_report.md` | Interpretación científica de H1-H5. |

## Resultados principales

La ejecución incluida en este repositorio soporta las cinco hipótesis de trabajo. En promedio, los patrones multi-agente alcanzan mayor índice de calidad que Single-Agent, aunque Parent-Child y Mixture-of-Agents incrementan latencia y tokens. Handoff sobresale cuando la respuesta correcta es pedir aclaración o rechazar una instrucción sensible.

| Hipótesis | Veredicto | Evidencia resumida |
|---|---|---|
| H1 | Soportada con matiz | Calidad multi-agente 0.749 vs Single-Agent 0.644; tokens 131.6 vs 89.0; latencia 431.7 ms vs 220.3 ms. |
| H2 | Soportada | En tareas medium, Parent-Child logra factualidad, trazabilidad y completitud de 1.00 con 185.5 tokens, frente a 338.0 de Mixture-of-Agents. |
| H3 | Soportada | En síntesis, Mixture-of-Agents alcanza completitud 0.929, superior a Parent-Child 0.464 y Single-Agent 0.309. |
| H4 | Soportada | En ambiguas/sensibles, Handoff logra exactitud de decisión 1.00 y alucinación 0.00. |
| H5 | Soportada | En preguntas simples, Single-Agent logra factualidad y completitud 1.00 con coste bajo. |

![Resumen de hipótesis](results/figures/hypothesis_summary.png)

El análisis completo está en [`docs/final_report.md`](docs/final_report.md), y las tablas reproducibles están en [`results/`](results/).

## Arquitectura

```text
data/corpus.json + data/questions.json
        │
        ▼
KeywordRetriever ──► Single-Agent RAG
        │            Parent-Child RAG
        │            Mixture-of-Agents RAG
        │            Handoff RAG
        ▼
Métricas reproducibles ──► CSV + visualizaciones + reporte
```

## Principales decisiones técnicas

El proyecto usa recuperación lexical determinista y generación extractiva controlada por hechos para que las conclusiones sean **auditables y repetibles**. Esta aproximación evita que variaciones externas de un proveedor LLM alteren los resultados experimentales y facilita prácticas de gestión de riesgo alineadas con marcos de gobierno de IA [4]. En una fase posterior, el mismo harness puede conectarse a modelos reales manteniendo las mismas interfaces de evaluación.

## Referencias

[1]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[2]: https://arxiv.org/abs/2402.01680 "Mixture-of-Agents Enhances Large Language Model Capabilities"
[3]: https://arxiv.org/abs/2308.08155 "A Survey on Large Language Model based Autonomous Agents"
[4]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"
