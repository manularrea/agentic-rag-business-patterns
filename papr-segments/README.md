# Segmentos independientes para el paper

**Autor:** manularrea  
**Proyecto:** `agentic-rag-business-patterns`  
**Propósito:** Esta carpeta contiene segmentos Markdown independientes, redactados para ser reutilizados como secciones autónomas del paper científico sobre patrones Agentic RAG en tareas empresariales.

Los archivos separan el marco experimental, la implementación de los harness, la metodología de evaluación, el modelo de coste y la estrategia de resultados/visualizaciones. Esta separación permite integrar cada bloque en una estructura IMRaD o en anexos técnicos sin mezclar resultados, método y discusión.

| Archivo | Apartado cubierto | Uso recomendado en el paper |
|---|---|---|
| `01_marco_experimental_y_corpus.md` | Corpus, preguntas, metadatos, control de comparabilidad, recuperación y fragmentación. | Método experimental y descripción del dataset. |
| `02_implementacion_harness_pipeline.md` | Implementación reproducible de Single-Agent, Parent-Child, Mixture-of-Agents y Handoff. | Método, arquitectura de sistemas y reproducibilidad. |
| `03_metodologia_evaluacion_metricas.md` | Métricas RAG, RAGAS, evaluación por categoría, agregación y control de varianza. | Metodología de evaluación y validez interna. |
| `04_ecuaciones_coste_eficiencia.md` | Fórmulas de coste por tokens, latencia, llamadas, utilidad neta y coste compuesto. | Modelo matemático, eficiencia y discusión económica. |
| `05_resultados_visualizaciones_citas.md` | Plan visual, interpretación de resultados, significancia, fuentes y citas. | Resultados, discusión y material visual del paper. |

> Nota editorial: se conserva el nombre de carpeta `papr-segments` porque fue solicitado explícitamente así. Si se desea una variante ortográfica para publicación final, puede crearse posteriormente un alias `paper-segments`, pero no se modifica el nombre pedido para evitar romper referencias.

## Relación con el repositorio

Los segmentos se apoyan en los activos ya generados por el repositorio: los datos controlados están en `data/`, los resultados en `results/`, las figuras base en `results/figures/`, las figuras editoriales en `paper_visuals/figures/` y las tablas matemáticas en `paper_visuals/tables/`. Para regenerar resultados debe ejecutarse `python scripts/run_experiment.py`; para regenerar las figuras del paper debe ejecutarse `python paper_visuals/scripts/create_paper_figures.py`.

## Referencias

[1]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"  
[2]: https://docs.ragas.io/ "RAGAS Documentation"  
[3]: https://arxiv.org/abs/2309.15217 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"  
[4]: https://arxiv.org/abs/2402.01680 "Mixture-of-Agents Enhances Large Language Model Capabilities"  
[5]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"
