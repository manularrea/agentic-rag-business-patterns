from __future__ import annotations

import json
from pathlib import Path

from .schemas import Document, EvaluationQuestion

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_documents(path: Path | None = None) -> list[Document]:
    source = path or DATA_DIR / "corpus.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    return [Document(**item) for item in raw]


def load_questions(path: Path | None = None) -> list[EvaluationQuestion]:
    source = path or DATA_DIR / "questions.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    return [EvaluationQuestion(**item) for item in raw]
