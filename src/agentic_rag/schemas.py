from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RiskLevel = Literal["low", "medium", "high"]
QuestionType = Literal["simple", "medium", "synthesis", "ambiguous", "sensitive"]
Decision = Literal["answer", "clarify", "refuse"]
PatternName = Literal["single_agent", "parent_child", "mixture_of_agents", "handoff"]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    domain: str
    content: str
    facts: dict[str, str]


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    question: str
    question_type: QuestionType
    risk: RiskLevel
    domains: list[str]
    required_fact_ids: list[str]
    expected_citations: list[str]
    expected_decision: Decision = "answer"


@dataclass
class Answer:
    pattern: str
    question_id: str
    content: str
    citations: list[str]
    fact_ids: list[str]
    trace: list[str] = field(default_factory=list)
    decision: Decision = "answer"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
