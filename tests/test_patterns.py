from agentic_rag.data import load_documents, load_questions
from agentic_rag.patterns import HandoffRAG, MixtureOfAgentsRAG, ParentChildRAG, SingleAgentRAG
from agentic_rag.retrieval import KeywordRetriever


def _pattern_instances():
    retriever = KeywordRetriever(load_documents())
    return [SingleAgentRAG(retriever), ParentChildRAG(retriever), MixtureOfAgentsRAG(retriever), HandoffRAG(retriever)]


def test_all_patterns_answer_all_questions():
    questions = load_questions()
    for pattern in _pattern_instances():
        for question in questions:
            answer = pattern.answer(question)
            assert answer.pattern == pattern.name
            assert answer.content
            assert answer.total_tokens > 0
            assert answer.latency_ms > 0


def test_handoff_uses_guardrails_for_high_risk_questions():
    questions = {q.id: q for q in load_questions()}
    handoff = HandoffRAG(KeywordRetriever(load_documents()))
    assert handoff.answer(questions["Q_AMBIG_01"]).decision == "clarify"
    assert handoff.answer(questions["Q_SENS_01"]).decision == "refuse"
