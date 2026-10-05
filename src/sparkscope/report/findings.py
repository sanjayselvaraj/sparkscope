"""Reporters: render a list of :class:`Finding` for humans or machines.

Kept separate from the CLI so presentation logic is testable without invoking
Typer, and so adding a format later (markdown, SARIF, ...) is a new function
here rather than a change to the command.
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from sparkscope.analysis.finding import Finding, Severity

_SEVERITY_STYLE = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}


def render_terminal(console: Console, findings: list[Finding], top: int) -> None:
    """Print findings worst-first as readable panels (at most ``top``)."""
    if not findings:
        console.print("[green]No performance issues detected.[/green]")
        return

    shown = findings[:top]
    console.print(f"[bold]{len(findings)} finding(s)[/bold] (showing {len(shown)}):\n")

    for f in shown:
        style = _SEVERITY_STYLE.get(f.severity, "white")
        body = Text()
        body.append(
            f"stage {f.stage_label}   detector: {f.detector}   "
            f"confidence: {f.confidence.label()}\n",
            style="dim",
        )
        body.append("Evidence (observed):\n", style="bold")
        for line in f.evidence:
            body.append(f"  • {line}\n")
        if f.likely_cause:
            body.append("\nLikely cause (inferred): ", style="bold")
            body.append(f"{f.likely_cause}\n")
        if f.recommendation:
            body.append("Fix: ", style="bold")
            body.append(f.recommendation)
        console.print(
            Panel(
                body,
                title=f"[{style}]{f.severity.label()}[/{style}]  {f.title}",
                border_style=style,
            )
        )


def render_json(findings: list[Finding], top: int) -> str:
    """Return a JSON string of findings (worst-first, capped at ``top``)."""
    payload = {
        "finding_count": len(findings),
        "findings": [f.to_dict() for f in findings[:top]],
    }
    return json.dumps(payload, indent=2)
