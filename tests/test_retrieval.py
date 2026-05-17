from agentic_rag.data import load_documents
from agentic_rag.retrieval import KeywordRetriever


def test_retriever_respects_domain_filter():
    retriever = KeywordRetriever(load_documents())
    docs = retriever.retrieve("pagos automatizados aprobación", top_k=2, domains=["finance"])
    assert docs
    assert all(doc.domain == "finance" for doc in docs)
