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

# Importing each detector module registers it in the detector registry via the
# @register side-effect. run_all() then runs whatever is registered. Keep these
# imports explicit (not a dynamic scan) so the active detector set is obvious.
from sparkscope.analysis import join as _join  # noqa: F401
from sparkscope.analysis import partition as _partition  # noqa: F401
from sparkscope.analysis import shuffle as _shuffle  # noqa: F401
from sparkscope.analysis import skew as _skew  # noqa: F401
from sparkscope.analysis import spill as _spill  # noqa: F401
from sparkscope.analysis.base import run_all
from sparkscope.analysis.compare import compare_runs
from sparkscope.parser.event_log import EventLogParseError, parse_file
from sparkscope.report.comparison import render_comparison_json, render_comparison_terminal
from sparkscope.report.findings import render_json, render_terminal

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
    # Pipeline: parse -> build execution model -> run detectors -> report.
    try:
        run = parse_file(event_log)
    except EventLogParseError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    findings = run_all(run)

    if output_json:
        # JSON mode prints only the machine-readable payload (nothing else to
        # stdout) so it can be piped into jq / a CI gate without stray text.
        print(render_json(findings, top))
        return

    app_name = run.app_name or "(unknown)"
    app_id = run.app_id or "n/a"
    console.print(f"[bold]SparkScope[/bold] {__version__}")
    console.print(f"Application : [cyan]{app_name}[/cyan]  ({app_id})")
    console.print(f"Jobs: {len(run.jobs)}   Stages: {len(run.stages)}\n")

    # Surface partial-corruption so data loss is never silently invisible.
    if run.skipped_lines:
        console.print(
            f"[yellow]warning:[/yellow] skipped {run.skipped_lines} unparseable "
            "event-log line(s); results may be incomplete.\n"
        )

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
    console.print()
    render_terminal(console, findings, top)


@app.command()
def compare(
    baseline: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Baseline Spark event-log file.",
    ),
    current: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Current Spark event-log file to compare against the baseline.",
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a terminal report."
    ),
) -> None:
    """Compare two Spark event logs and report what changed (facts only)."""
    try:
        base_run = parse_file(baseline)
    except EventLogParseError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    try:
        cur_run = parse_file(current)
    except EventLogParseError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    c = compare_runs(base_run, cur_run)

    if output_json:
        # JSON mode prints only the machine-readable payload (nothing else to
        # stdout) so it can be piped into jq / a CI gate without stray text.
        print(render_comparison_json(c))
        return

    base_name = base_run.app_name or "(unknown)"
    cur_name = cur_run.app_name or "(unknown)"
    console.print(
        f"[bold]{base_name}[/bold] (baseline) vs [bold]{cur_name}[/bold] (current)\n"
    )

    # Surface partial-corruption on either side so data loss is never invisible.
    if base_run.skipped_lines:
        console.print(
            f"[yellow]warning:[/yellow] baseline skipped {base_run.skipped_lines} "
            "unparseable event-log line(s); results may be incomplete.\n"
        )
    if cur_run.skipped_lines:
        console.print(
            f"[yellow]warning:[/yellow] current skipped {cur_run.skipped_lines} "
            "unparseable event-log line(s); results may be incomplete.\n"
        )

    render_comparison_terminal(console, c)


if __name__ == "__main__":
    app()
