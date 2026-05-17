from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .schemas import Answer, EvaluationQuestion


@dataclass
class RowMetrics:
    question_id: str
    pattern: str
    question_type: str
    risk: str
    expected_decision: str
    decision: str
    factuality: float
    clarity: float
    traceability: float
    completeness: float
    hallucination_rate: float
    decision_accuracy: float
    latency_ms: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    citations: int
    trace_steps: int


def score_answer(answer: Answer, question: EvaluationQuestion) -> RowMetrics:
    required = set(question.required_fact_ids)
    found = set(answer.fact_ids)
    unsupported = int(answer.metadata.get("unsupported_claims", 0))
    supported_claims = len(found)
    factuality = supported_claims / max(1, supported_claims + unsupported)
    completeness = len(found & required) / max(1, len(required))
    expected_citations = set(question.expected_citations)
    citation_hit = len(set(answer.citations) & expected_citations) / max(1, len(expected_citations))
    trace_density = min(1.0, len(answer.trace) / 5.0)
    traceability = 0.65 * citation_hit + 0.35 * trace_density
    words = max(1, len(answer.content.split()))
    evidence_ratio = min(1.0, len(answer.fact_ids) / max(1, len(question.required_fact_ids)))
    concision = 1.0 if 18 <= words <= 115 else max(0.55, 1 - abs(words - 66) / 160)
    clarity = 0.72 * concision + 0.28 * max(evidence_ratio, 0.55 if answer.decision != "answer" else 0.0)
    hallucination_rate = unsupported / max(1, supported_claims + unsupported)
    decision_accuracy = 1.0 if answer.decision == question.expected_decision else 0.0
    return RowMetrics(
        question_id=question.id,
        pattern=answer.pattern,
        question_type=question.question_type,
        risk=question.risk,
        expected_decision=question.expected_decision,
        decision=answer.decision,
        factuality=round(factuality, 4),
        clarity=round(clarity, 4),
        traceability=round(traceability, 4),
        completeness=round(completeness, 4),
        hallucination_rate=round(hallucination_rate, 4),
        decision_accuracy=round(decision_accuracy, 4),
        latency_ms=round(answer.latency_ms, 2),
        total_tokens=answer.total_tokens,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        citations=len(answer.citations),
        trace_steps=len(answer.trace),
    )


def rows_to_frame(rows: list[RowMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "factuality", "clarity", "traceability", "completeness", "hallucination_rate",
        "decision_accuracy", "latency_ms", "total_tokens", "citations", "trace_steps",
    ]
    grouped = df.groupby("pattern", as_index=False)[metrics].mean()
    quality_cols = ["factuality", "clarity", "traceability", "completeness", "decision_accuracy"]
    grouped["quality_index"] = grouped[quality_cols].mean(axis=1)
    min_cost = grouped["total_tokens"].min()
    min_latency = grouped["latency_ms"].min()
    grouped["efficiency_index"] = 0.5 * (min_cost / grouped["total_tokens"]) + 0.5 * (min_latency / grouped["latency_ms"])
    grouped["balanced_index"] = 0.72 * grouped["quality_index"] + 0.28 * grouped["efficiency_index"]
    return grouped.round(4)
