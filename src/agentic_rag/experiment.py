from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_documents, load_questions
from .metrics import aggregate_results, rows_to_frame, score_answer
from .patterns import HandoffRAG, MixtureOfAgentsRAG, ParentChildRAG, SingleAgentRAG
from .retrieval import KeywordRetriever
from .visualization import create_all_figures


def build_patterns(retriever: KeywordRetriever):
    return [
        SingleAgentRAG(retriever),
        ParentChildRAG(retriever),
        MixtureOfAgentsRAG(retriever),
        HandoffRAG(retriever),
    ]


def run_experiment(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = load_documents()
    questions = load_questions()
    retriever = KeywordRetriever(documents)
    rows = []
    answer_records = []
    for pattern in build_patterns(retriever):
        for question in questions:
            answer = pattern.answer(question)
            rows.append(score_answer(answer, question))
            answer_records.append(
                {
                    "pattern": answer.pattern,
                    "question_id": answer.question_id,
                    "decision": answer.decision,
                    "content": answer.content,
                    "citations": ";".join(answer.citations),
                    "fact_ids": ";".join(answer.fact_ids),
                    "trace": " -> ".join(answer.trace),
                }
            )
    detailed = rows_to_frame(rows)
    aggregate = aggregate_results(detailed)
    detailed.to_csv(output_dir / "detailed_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    pd.DataFrame(answer_records).to_csv(output_dir / "answers.csv", index=False)
    create_all_figures(detailed, aggregate, output_dir / "figures")
    return detailed, aggregate
