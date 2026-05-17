from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .experiment import run_experiment

app = typer.Typer(help="Benchmark reproducible de patrones Agentic RAG empresariales.")
console = Console()


@app.command()
def run(output_dir: Path = typer.Option(Path("results"), help="Directorio de resultados.")) -> None:
    """Ejecuta el benchmark completo y genera métricas, respuestas y figuras."""
    detailed, aggregate = run_experiment(output_dir)
    console.print(f"Filas evaluadas: {len(detailed)}")
    console.print(aggregate.to_string(index=False))


if __name__ == "__main__":
    app()
