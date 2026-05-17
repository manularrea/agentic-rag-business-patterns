from pathlib import Path

from agentic_rag.experiment import run_experiment

if __name__ == "__main__":
    detailed, aggregate = run_experiment(Path("results"))
    print(f"Evaluaciones: {len(detailed)}")
    print(aggregate.to_string(index=False))
