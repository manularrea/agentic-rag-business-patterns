from __future__ import annotations

from .schemas import Answer, Document, EvaluationQuestion


def count_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _select_facts(documents: list[Document], required: list[str], coverage_limit: int | None = None) -> list[str]:
    available: list[str] = []
    for doc in documents:
        for fact_id in doc.facts:
            if fact_id in required and fact_id not in available:
                available.append(fact_id)
    if coverage_limit is not None:
        return available[:coverage_limit]
    return available


def synthesize_answer(
    *,
    pattern: str,
    question: EvaluationQuestion,
    documents: list[Document],
    fact_ids: list[str],
    trace: list[str],
    decision: str = "answer",
    latency_ms: float,
    prompt_multiplier: float = 1.0,
    completion_multiplier: float = 1.0,
    unsupported_claims: int = 0,
) -> Answer:
    if decision == "clarify":
        content = (
            "La consulta es ambigua para el contexto disponible. Se requiere precisar alcance, "
            "dominio afectado y periodo antes de emitir una respuesta ejecutable."
        )
    elif decision == "refuse":
        content = (
            "No es apropiado responder con una instrucción operativa porque la consulta puede inducir "
            "una acción sensible o insegura. Se recomienda reformularla como una solicitud de gobierno, "
            "cumplimiento o mitigación de riesgos."
        )
    else:
        fact_text: list[str] = []
        by_fact = {fid: text for doc in documents for fid, text in doc.facts.items()}
        for fid in fact_ids:
            if fid in by_fact:
                fact_text.append(f"{by_fact[fid]} [{fid}]")
        if fact_text:
            content = " ".join(fact_text)
            content += " La conclusión se fundamenta exclusivamente en la evidencia citada."
        else:
            content = "No se encontró evidencia suficiente en el corpus recuperado para responder."
        if unsupported_claims:
            content += " Además, se infiere una recomendación no validada por el corpus."
    citations = []
    for doc in documents:
        if doc.id not in citations:
            citations.append(doc.id)
    prompt_tokens = int(count_tokens(question.question + " " + " ".join(d.content for d in documents)) * prompt_multiplier)
    completion_tokens = int(count_tokens(content) * completion_multiplier)
    metadata = {"unsupported_claims": unsupported_claims, "retrieved_documents": len(documents)}
    return Answer(
        pattern=pattern,
        question_id=question.id,
        content=content,
        citations=citations,
        fact_ids=fact_ids,
        trace=trace,
        decision=decision,  # type: ignore[arg-type]
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        metadata=metadata,
    )


def select_required_facts(documents: list[Document], question: EvaluationQuestion, coverage_limit: int | None = None) -> list[str]:
    return _select_facts(documents, question.required_fact_ids, coverage_limit=coverage_limit)
