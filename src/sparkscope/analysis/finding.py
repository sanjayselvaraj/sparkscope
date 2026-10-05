"""The structured output of a detector: a :class:`Finding`.

Every detector speaks the same language -- it returns zero or more Findings.
A Finding deliberately separates three different kinds of claim, because
conflating them is how diagnostic tools lose a user's trust:

* **evidence**  -- observed facts read directly from the event log. These are
  not inferences; they are what the log literally says.
* **likely_cause** -- an *inference* about why. We only ever phrase this as a
  hypothesis ("consistent with ...", "likely ..."), never as proven fact,
  because the event log rarely proves root cause on its own.
* **recommendation** -- a concrete action to try.

A Finding also carries machine-readable fields (``category``, ``metrics``) so
the JSON output is programmatically consumable, not just human-readable, and a
``confidence`` so a detector can say "this looks severe but I only have weak
corroboration."

Raw metrics are already in the Spark UI; the value SparkScope adds is this
honest interpretation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Severity(IntEnum):
    """How serious a finding is.

    An IntEnum (not a plain Enum) so findings sort naturally by severity:
    ``sorted(findings, reverse=True)`` puts the worst first, which is exactly
    how a diagnostics report should read.
    """

    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

    def label(self) -> str:
        return self.name


class Confidence(IntEnum):
    """How well the available evidence corroborates the finding.

    Severity answers "how bad is it?"; confidence answers "how sure are we it is
    what we say it is?". They are independent: a stage can show a large time
    imbalance (high severity signal) that we are only moderately sure is *data*
    skew rather than a straggler executor (medium confidence). Reporting both
    keeps the tool honest.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3

    def label(self) -> str:
        return self.name


@dataclass(order=True)
class Finding:
    """A single diagnosed problem.

    ``order=True`` with ``severity`` first means Findings compare by severity;
    ``sort_index`` is an explicit magnitude tiebreaker so equally severe findings
    rank by how extreme they are. All descriptive fields are ``compare=False`` so
    free text never affects ranking.
    """

    severity: Severity
    # Secondary sort key: larger magnitude ranks first within a severity band.
    sort_index: float = field(default=0.0)

    #: Machine-readable detector category, e.g. "skew" (stable across versions).
    category: str = field(default="", compare=False)
    detector: str = field(default="", compare=False)
    title: str = field(default="", compare=False)
    stage_id: int | None = field(default=None, compare=False)
    stage_attempt_id: int | None = field(default=None, compare=False)

    confidence: Confidence = field(default=Confidence.MEDIUM, compare=False)
    #: Observed facts from the log (NOT inference). Each item is a short string.
    evidence: list[str] = field(default_factory=list, compare=False)
    #: Structured numbers behind the evidence, for programmatic consumers.
    metrics: dict[str, float] = field(default_factory=dict, compare=False)
    #: Inference about the cause -- always phrased as a hypothesis, never fact.
    likely_cause: str = field(default="", compare=False)
    recommendation: str = field(default="", compare=False)

    @property
    def stage_label(self) -> str:
        if self.stage_id is None:
            return "-"
        attempt = self.stage_attempt_id if self.stage_attempt_id is not None else 0
        return f"{self.stage_id}.{attempt}"

    def to_dict(self) -> dict:
        """Flatten to a deterministic, JSON-serializable dict."""
        return {
            "category": self.category,
            "detector": self.detector,
            "severity": self.severity.label(),
            "confidence": self.confidence.label(),
            "title": self.title,
            "stage": self.stage_label if self.stage_id is not None else None,
            "metrics": dict(self.metrics),
            "evidence": list(self.evidence),
            "likely_cause": self.likely_cause,
            "recommendation": self.recommendation,
        }
