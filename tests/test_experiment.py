from pathlib import Path

from agentic_rag.experiment import run_experiment


def test_experiment_generates_expected_outputs(tmp_path: Path):
    detailed, aggregate = run_experiment(tmp_path)
    assert len(detailed) == 40
    assert set(aggregate["pattern"]) == {"single_agent", "parent_child", "mixture_of_agents", "handoff"}
    assert (tmp_path / "detailed_metrics.csv").exists()
    assert (tmp_path / "aggregate_metrics.csv").exists()
    assert (tmp_path / "figures" / "balanced_index.png").exists()
