"""Command-line entrypoint for SparkScope.

Day 1: the `analyze` command is a working stub that validates its input path
and reports that parsing is not yet wired up. Week-1 work replaces the stub
body with the real parse -> model -> report pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from sparkscope import __version__

app = typer.Typer(
    name="sparkscope",
    help="Diagnose Apache Spark performance issues (skew, spill, shuffle, joins) "
    "from event logs. No running cluster required.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"sparkscope {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """SparkScope — point it at a Spark event log and get ranked diagnostics."""


@app.command()
def analyze(
    event_log: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a Spark event-log file (or a History Server directory).",
    ),
    top: int = typer.Option(
        10, "--top", "-n", min=1, help="Show at most this many of the worst findings."
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a terminal report."
    ),
) -> None:
    """Analyze a Spark event log and report performance diagnostics."""
    # Day 1 stub: validate input, prove the wiring, then exit.
    # Day 3-7 replaces this with: parse -> build model -> run detectors -> report.
    console.print(f"[bold]SparkScope[/bold] {__version__}")
    console.print(f"Target : [cyan]{event_log}[/cyan]")
    console.print(f"Top N  : {top}   JSON: {output_json}")
    console.print(
        "[yellow]Parser not wired yet (Day 1 scaffold). "
        "Coming in Week 1: stage/task model + detectors.[/yellow]"
    )


if __name__ == "__main__":
    app()
