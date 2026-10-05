"""Small shared helpers for detectors (formatting + simple stats).

Kept in one place so every detector formats bytes/durations identically and we
don't scatter copies of the same rounding logic across modules.
"""

from __future__ import annotations

from statistics import median


def format_ms(ms: float) -> str:
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def format_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def ratio(numerator: float, denominator: float) -> float:
    """Guarded ratio: 0.0 when the denominator is non-positive."""
    return numerator / denominator if denominator > 0 else 0.0
