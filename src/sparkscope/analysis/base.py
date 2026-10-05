"""Detector base class and registry.

A detector is a small, independent unit of analysis: given the parsed
:class:`SparkRun`, it returns a list of :class:`Finding`. Keeping detectors
behind a common ABC (rather than a pile of functions) buys three things:

* **Extensibility** -- adding the spill / shuffle / join detectors later means
  writing one subclass, not editing a central analyzer.
* **Uniform metadata** -- every detector declares a ``name`` used in findings
  and reports.
* **Isolation/testability** -- each detector is tested on its own against
  crafted stages, with no coupling to the others.

The registry is a plain module-level list populated by ``@register``. We keep
it dead simple on purpose: no plugin entry-points, no dynamic import magic.
For v0.1 the detector set is small and known; a heavier mechanism would be
premature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sparkscope.analysis.finding import Finding
from sparkscope.parser.models import SparkRun

# Ordered registry of detector classes. Order here is only a default; findings
# are ultimately ranked by severity, not by detector order.
_REGISTRY: list[type[Detector]] = []


def register(cls: type[Detector]) -> type[Detector]:
    """Class decorator that adds a detector to the registry."""
    _REGISTRY.append(cls)
    return cls


class Detector(ABC):
    """Base class for all detectors."""

    #: Short stable identifier, shown in findings and reports (e.g. "skew").
    name: str = "detector"
    #: Machine-readable category for the findings this detector produces.
    category: str = "detector"

    @abstractmethod
    def analyze(self, run: SparkRun) -> list[Finding]:
        """Inspect the run and return any findings (possibly empty)."""
        raise NotImplementedError


def all_detectors() -> list[Detector]:
    """Instantiate one of each registered detector."""
    return [cls() for cls in _REGISTRY]


def run_all(run: SparkRun) -> list[Finding]:
    """Run every registered detector and return findings, worst severity first."""
    findings: list[Finding] = []
    for detector in all_detectors():
        findings.extend(detector.analyze(run))
    findings.sort(reverse=True)  # Severity (then sort_index) descending.
    return findings
