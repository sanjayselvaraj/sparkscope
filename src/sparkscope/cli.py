"""Command-line entrypoint for SparkScope.

Day 1: the `analyze` command is a working stub that validates its input path
and reports that parsing is not yet wired up. Week-1 work replaces the stub
body with the real parse -> model -> report pipeline.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sparkscope import __version__
from sparkscope.parser.event_log import EventLogParseError, parse_file

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
    _version: bool | None = typer.Option(
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
    # Day 2: parse -> build execution model -> show a per-stage summary.
    # Detectors (Day 8+) will consume this same model and add a findings report.
    try:
        run = parse_file(event_log)
    except EventLogParseError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    app_name = run.app_name or "(unknown)"
    app_id = run.app_id or "n/a"
    console.print(f"[bold]SparkScope[/bold] {__version__}")
    console.print(f"Application : [cyan]{app_name}[/cyan]  ({app_id})")
    console.print(f"Jobs: {len(run.jobs)}   Stages: {len(run.stages)}\n")

    table = Table(title="Stage summary")
    table.add_column("Stage", justify="right")
    table.add_column("Name", overflow="fold")
    table.add_column("Tasks", justify="right")
    table.add_column("Median ms", justify="right")
    table.add_column("Max ms", justify="right")
    table.add_column("Skew x", justify="right")

    for stage in run.stage_list():
        table.add_row(
            f"{stage.stage_id}.{stage.attempt_id}",
            stage.name or "-",
            str(len(stage.tasks)),
            f"{stage.median_task_ms:.0f}",
            str(stage.max_task_ms),
            f"{stage.skew_ratio:.1f}",
        )
    console.print(table)

    if output_json:
        console.print("[dim](JSON reporter arrives with the detectors in Week 2.)[/dim]")


if __name__ == "__main__":
    app()
