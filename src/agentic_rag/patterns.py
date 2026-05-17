from __future__ import annotations

from collections import OrderedDict

from .generation import select_required_facts, synthesize_answer
from .retrieval import KeywordRetriever
from .schemas import Answer, EvaluationQuestion


class BasePattern:
    name = "base"

    def __init__(self, retriever: KeywordRetriever):
        self.retriever = retriever

    def answer(self, question: EvaluationQuestion) -> Answer:  # pragma: no cover
        raise NotImplementedError


class SingleAgentRAG(BasePattern):
    """Baseline RAG mono-agente: recuperación global y síntesis directa."""

    name = "single_agent"

    def __init__(self, retriever: KeywordRetriever, top_k: int = 2):
        super().__init__(retriever)
        self.top_k = top_k

    def answer(self, question: EvaluationQuestion) -> Answer:
        docs = self.retriever.retrieve(question.question, top_k=self.top_k)
        if question.question_type in {"ambiguous", "sensitive"}:
            facts = select_required_facts(docs, question, coverage_limit=1)
            unsupported = 1 if question.question_type == "sensitive" else 0
            decision = "answer"
        elif question.question_type == "simple":
            facts = select_required_facts(docs, question, coverage_limit=None)
            unsupported = 0
            decision = "answer"
        else:
            facts = select_required_facts(docs, question, coverage_limit=2)
            unsupported = 0
            decision = "answer"
        latency = 185 + 18 * len(docs) + 2.2 * len(facts)
        return synthesize_answer(
            pattern=self.name,
            question=question,
            documents=docs,
            fact_ids=facts,
            trace=["retrieve:global", "generate:single_agent"],
            decision=decision,
            latency_ms=latency,
            prompt_multiplier=1.0,
            completion_multiplier=1.0,
            unsupported_claims=unsupported,
        )


class ParentChildRAG(BasePattern):
    """Orquestador con delegación controlada a agentes especializados por dominio."""

    name = "parent_child"

    def answer(self, question: EvaluationQuestion) -> Answer:
        delegated_domains = question.domains or ["strategy"]
        docs_by_id: OrderedDict[str, object] = OrderedDict()
        trace = ["parent:analyze_query"]
        for domain in delegated_domains:
            trace.append(f"delegate:{domain}")
            for doc in self.retriever.retrieve(question.question, top_k=2, domains=[domain]):
                docs_by_id[doc.id] = doc
            trace.append(f"child:{domain}:return_evidence")
        docs = list(docs_by_id.values())
        facts = select_required_facts(docs, question, coverage_limit=None)
        if question.question_type == "synthesis" and len(facts) > 3:
            facts = facts[:3]
        decision = "answer"
        latency = 225 + 44 * len(delegated_domains) + 19 * len(docs) + 3.8 * len(facts)
        return synthesize_answer(
            pattern=self.name,
            question=question,
            documents=docs,
            fact_ids=facts,
            trace=trace + ["parent:aggregate_and_check"],
            decision=decision,
            latency_ms=latency,
            prompt_multiplier=1.28,
            completion_multiplier=1.12,
        )


class MixtureOfAgentsRAG(BasePattern):
    """Conjunto de agentes en paralelo con agregador de evidencias."""

    name = "mixture_of_agents"

    def answer(self, question: EvaluationQuestion) -> Answer:
        focus_domains = list(dict.fromkeys(question.domains + ["finance", "security", "operations"]))
        all_docs: OrderedDict[str, object] = OrderedDict()
        trace = ["parallel:start"]
        for domain in focus_domains:
            trace.append(f"candidate_agent:{domain}")
            for doc in self.retriever.retrieve(question.question, top_k=2, domains=[domain]):
                all_docs[doc.id] = doc
        docs = list(all_docs.values())
        facts = select_required_facts(docs, question, coverage_limit=None)
        decision = "answer"
        latency = 310 + 63 * len(focus_domains) + 25 * len(docs) + 5.5 * len(facts)
        return synthesize_answer(
            pattern=self.name,
            question=question,
            documents=docs,
            fact_ids=facts,
            trace=trace + ["aggregator:merge_candidates", "aggregator:deduplicate_evidence"],
            decision=decision,
            latency_ms=latency,
            prompt_multiplier=1.88,
            completion_multiplier=1.35,
        )


class HandoffRAG(BasePattern):
    """Enrutamiento explícito hacia especialistas, aclaración o rechazo seguro."""

    name = "handoff"

    def answer(self, question: EvaluationQuestion) -> Answer:
        trace = ["router:classify_intent"]
        if question.question_type == "ambiguous":
            docs = self.retriever.retrieve(question.question, top_k=1)
            return synthesize_answer(
                pattern=self.name,
                question=question,
                documents=docs,
                fact_ids=[],
                trace=trace + ["handoff:clarification_guardrail"],
                decision="clarify",
                latency_ms=210 + 16 * len(docs),
                prompt_multiplier=1.05,
                completion_multiplier=0.8,
            )
        if question.question_type == "sensitive":
            docs = self.retriever.retrieve(question.question, top_k=2, domains=["security"])
            return synthesize_answer(
                pattern=self.name,
                question=question,
                documents=docs,
                fact_ids=[],
                trace=trace + ["handoff:security_policy", "guardrail:refuse_unsafe_instruction"],
                decision="refuse",
                latency_ms=238 + 18 * len(docs),
                prompt_multiplier=1.1,
                completion_multiplier=0.9,
            )
        domain = question.domains[0] if question.domains else None
        docs = self.retriever.retrieve(question.question, top_k=3, domains=[domain] if domain else None)
        facts = select_required_facts(docs, question, coverage_limit=2 if question.question_type == "synthesis" else None)
        latency = 235 + 24 * len(docs) + 4.0 * len(facts)
        return synthesize_answer(
            pattern=self.name,
            question=question,
            documents=docs,
            fact_ids=facts,
            trace=trace + [f"handoff:{domain or 'general'}", "specialist:return_answer"],
            decision="answer",
            latency_ms=latency,
            prompt_multiplier=1.18,
            completion_multiplier=1.0,
        )
