"""Reporters: render a :class:`RunComparison` for humans or machines.

Presentation only. This module NEVER labels anything "regression"/"bad" or
assigns severity — it mirrors the factual, judgment-free contract of the
comparison model. Worst-first ordering below is by magnitude of change only, a
display convenience, not a verdict. Kept separate from the CLI so the rendering
is testable without invoking Typer.
"""

from __future__ import annotations

import json

from rich.console import Console

from sparkscope.analysis.compare import MetricDelta, RunComparison, StageComparison
from sparkscope.analysis.util import format_bytes, format_ms


def _fmt_value(value: float, unit: str) -> str:
    """Format a metric value by its unit, reusing the shared formatters."""
    if unit == "bytes":
        return format_bytes(value)
    if unit == "ms":
        return format_ms(value)
    if unit == "ratio":
        return f"{value:.1f}x"
    # count (and any other scalar): plain integer.
    return f"{value:.0f}"


def _fmt_pct(pct: float | None) -> str:
    """Signed integer percent, or ``n/a`` when undefined (zero baseline)."""
    if pct is None:
        return "n/a"
    return f"{pct:+.0f}%"


def _fmt_delta(delta: MetricDelta) -> str:
    base = _fmt_value(delta.baseline, delta.unit)
    cur = _fmt_value(delta.current, delta.unit)
    return f"{delta.name}: {base} -> {cur}  ({_fmt_pct(delta.pct_change)})"


def _stage_rank_key(comparison: StageComparison) -> tuple[int, float, float, str]:
    """Worst-first-by-MAGNITUDE display key (not a judgment).

    Deltas with a real ``pct_change`` rank first, ordered by the largest
    absolute pct_change among the stage's deltas (descending). Stages whose
    largest change has no percentage (zero baseline) fall back after those,
    ordered by the largest absolute ``abs_change``. ``stage_key`` is the stable
    tiebreak so ordering is deterministic. The caller sorts ascending, so the
    numeric magnitudes are negated to put the largest first.
    """
    pcts = [abs(d.pct_change) for d in comparison.metric_deltas if d.pct_change is not None]
    abs_changes = [abs(d.abs_change) for d in comparison.metric_deltas]
    has_pct = 0 if pcts else 1  # real-pct stages sort before pct-less ones
    max_pct = max(pcts) if pcts else 0.0
    max_abs = max(abs_changes) if abs_changes else 0.0
    return (has_pct, -max_pct, -max_abs, comparison.stage_key)


def render_comparison_terminal(console: Console, comparison: RunComparison) -> None:
    """Print the comparison worst-first for humans (facts only)."""
    # Application-level section.
    console.print("[bold]Application[/bold]")
    for delta in comparison.app_deltas:
        console.print(f"  {_fmt_delta(delta)}")
    console.print()

    # Per-stage sections, only for matched stages with at least one non-zero
    # delta: a matched stage whose every delta is zero is unchanged and would
    # add only noise, so it is skipped.
    matched = [
        c
        for c in comparison.stage_comparisons
        if c.status == "matched" and any(d.abs_change != 0 for d in c.metric_deltas)
    ]
    matched.sort(key=_stage_rank_key)

    if matched:
        console.print("[bold]Changed stages[/bold] (largest change first)")
        for c in matched:
            console.print(f"  [cyan]{c.stage_key}[/cyan]  (match: {c.match_method})")
            # Show only the non-zero deltas to stay concise.
            for delta in c.metric_deltas:
                if delta.abs_change != 0:
                    console.print(f"    {_fmt_delta(delta)}")
        console.print()

    added = [c.stage_key for c in comparison.stage_comparisons if c.status == "added"]
    removed = [c.stage_key for c in comparison.stage_comparisons if c.status == "removed"]
    if added:
        console.print(f"[bold]Added stages[/bold]: {', '.join(added)}")
    if removed:
        console.print(f"[bold]Removed stages[/bold]: {', '.join(removed)}")


def render_comparison_json(comparison: RunComparison) -> str:
    """Return the comparison as a stable, machine-readable JSON string."""
    return json.dumps(comparison.to_dict(), indent=2)
