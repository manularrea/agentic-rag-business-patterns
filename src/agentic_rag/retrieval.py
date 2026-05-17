from __future__ import annotations

import math
import re
from collections import Counter

from .schemas import Document

TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)
STOPWORDS = {
    "de", "la", "el", "y", "en", "un", "una", "para", "con", "que", "del", "los",
    "las", "al", "por", "se", "a", "o", "es", "sobre", "como", "qué", "cual", "cuál",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


class KeywordRetriever:
    """Retriever determinista basado en similitud lexical ponderada por dominio."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._doc_terms = {doc.id: Counter(tokenize(doc.content + " " + doc.title)) for doc in documents}
        df: Counter[str] = Counter()
        for terms in self._doc_terms.values():
            df.update(terms.keys())
        n = len(documents)
        self._idf = {term: math.log((1 + n) / (1 + freq)) + 1 for term, freq in df.items()}

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        domains: list[str] | None = None,
    ) -> list[Document]:
        query_terms = Counter(tokenize(query))
        allowed = set(domains or [])
        scored: list[tuple[float, Document]] = []
        for doc in self.documents:
            if allowed and doc.domain not in allowed:
                continue
            terms = self._doc_terms[doc.id]
            score = 0.0
            for term, qtf in query_terms.items():
                score += qtf * terms.get(term, 0) * self._idf.get(term, 1.0)
            if allowed and doc.domain in allowed:
                score *= 1.25
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        return [doc for _, doc in scored[:top_k]]
