"""Data-skew detector.

Data skew is the canonical Spark performance pathology: a shuffle key whose
values are unevenly distributed sends most of the rows to one reduce task, so
that single "hot" task runs far longer than its peers while the rest of the
cluster sits idle. In the Spark UI it looks like "199 tasks done in seconds, 1
task running for minutes."

What the event log DOES and DOES NOT tell us (this governs how we word findings)
--------------------------------------------------------------------------------
The log gives us per-task *run time*, *shuffle read bytes*, and *spill bytes*.
It does NOT give us the per-key row distribution, so the log cannot by itself
*prove* that a single key is hot. A long task could instead be a straggler
executor, GC pressure, or a slow node.

So we do two things to stay honest:

1. We treat task-time imbalance (``skew_ratio`` = max/median task time) as the
   primary *signal*, not proof.
2. We look for a *corroborating* signal: is the slowest task also moving
   disproportionately more data (shuffle read / spill) than its peers? If yes,
   the imbalance is backed by a data-volume imbalance -- that is "consistent
   with data skew" and we report HIGH confidence. If the time is lopsided but
   the data is not, we drop to LOW confidence and explicitly say it may be a
   straggler rather than data skew.

Severity measures *how bad* the imbalance is; confidence measures *how sure we
are of the cause*. They are reported separately.

Thresholds (documented, not magic numbers)
-------------------------------------------
* ``MIN_TASKS``        -- below this, one slow task is noise and the median is
  unstable; we do not judge.
* ``MIN_HOT_TASK_MS``  -- an absolute floor; "30x skew" on a 300ms task costs
  nobody anything.
* ``*_RATIO``          -- time-imbalance band boundaries -> severity.
* ``DATA_SKEW_RATIO``  -- how lopsided the hot task's *data* must be to count as
  corroboration.
"""

from __future__ import annotations

from sparkscope.analysis.base import Detector, register
from sparkscope.analysis.finding import Confidence, Finding, Severity
from sparkscope.analysis.util import format_bytes, format_ms, safe_median
from sparkscope.parser.models import SparkRun, Stage, Task

#: Minimum tasks in a stage before skew is meaningful (below this, it's noise).
MIN_TASKS = 4
#: The slowest task must take at least this long for skew to matter in absolute terms.
MIN_HOT_TASK_MS = 1_000
#: Time-imbalance band boundaries -> severity.
MEDIUM_RATIO = 2.0
HIGH_RATIO = 5.0
CRITICAL_RATIO = 10.0
#: The hot task must move at least this multiple of the median task's data
#: volume for the time skew to be corroborated as *data* skew.
DATA_SKEW_RATIO = 2.0


def _severity_for(ratio: float) -> Severity | None:
    """Map a time-skew ratio to a severity band, or None if not worth flagging."""
    if ratio >= CRITICAL_RATIO:
        return Severity.CRITICAL
    if ratio >= HIGH_RATIO:
        return Severity.HIGH
    if ratio >= MEDIUM_RATIO:
        return Severity.MEDIUM
    return None


def _task_data_bytes(task: Task) -> int:
    """Data volume a task handled: shuffle read + spill (the drivers of skew)."""
    m = task.metrics
    return m.shuffle_read_bytes + m.memory_spilled_bytes + m.disk_spilled_bytes


@register
class SkewDetector(Detector):
    """Flags stages where one task runs disproportionately longer than the median."""

    name = "skew"
    category = "skew"

    def analyze(self, run: SparkRun) -> list[Finding]:
        findings: list[Finding] = []
        for stage in run.stage_list():
            finding = self._check_stage(stage)
            if finding is not None:
                findings.append(finding)
        return findings

    def _check_stage(self, stage: Stage) -> Finding | None:
        # --- gates: only judge stages where skew is meaningful -----------------
        if len(stage.tasks) < MIN_TASKS:
            return None

        time_ratio = stage.skew_ratio
        if time_ratio <= 0:
            return None
        if stage.max_task_ms < MIN_HOT_TASK_MS:
            return None  # absolute floor: trivially short hot task is noise

        severity = _severity_for(time_ratio)
        if severity is None:
            return None

        # --- corroboration: is the hot task also moving more DATA? -------------
        # Find the slowest task and compare its data volume to the median task's.
        hot_task = max(stage.tasks, key=lambda t: t.metrics.duration_ms)
        data_volumes = [_task_data_bytes(t) for t in stage.tasks]
        median_data = safe_median([float(v) for v in data_volumes])
        hot_data = _task_data_bytes(hot_task)

        data_ratio = (hot_data / median_data) if median_data > 0 else 0.0
        corroborated = data_ratio >= DATA_SKEW_RATIO

        median_ms = stage.median_task_ms
        hot_ms = stage.max_task_ms

        # --- evidence: OBSERVED FACTS only (no inference here) -----------------
        evidence = [
            f"slowest task {format_ms(hot_ms)} vs median {format_ms(median_ms)} "
            f"({time_ratio:.1f}x longer)",
            f"{len(stage.tasks)} tasks in stage",
        ]
        if median_data > 0:
            evidence.append(
                f"slowest task handled {format_bytes(hot_data)} vs median "
                f"{format_bytes(median_data)} ({data_ratio:.1f}x more data)"
            )
        elif hot_data > 0:
            evidence.append(
                f"slowest task handled {format_bytes(hot_data)} "
                "(peers reported no shuffle/spill data)"
            )

        # --- inference: phrased as hypothesis, conditioned on corroboration ----
        if corroborated:
            confidence = Confidence.HIGH
            likely_cause = (
                "The slowest task processed far more data than its peers, which is "
                "consistent with data skew: one shuffle/join key (or a few) likely "
                "holds a disproportionate share of rows, so a single task does most "
                "of the work."
            )
        else:
            # Time is lopsided but data is NOT -- do not assert data skew.
            confidence = Confidence.LOW
            likely_cause = (
                "One task ran much longer than its peers, but it did NOT handle "
                "noticeably more data. This is less consistent with classic data "
                "skew and may instead be a straggler executor, GC pause, or a slow "
                "node. Confirm in the Spark UI before salting keys."
            )
            # A time-only imbalance is weaker evidence; cap severity at HIGH so we
            # never scream CRITICAL "data skew" without data corroboration.
            if severity == Severity.CRITICAL:
                severity = Severity.HIGH

        metrics = {
            "time_skew_ratio": round(time_ratio, 2),
            "data_skew_ratio": round(data_ratio, 2),
            "max_task_ms": float(hot_ms),
            "median_task_ms": round(median_ms, 2),
            "num_tasks": float(len(stage.tasks)),
        }

        return Finding(
            severity=severity,
            sort_index=time_ratio,
            category=self.category,
            detector=self.name,
            title=f"Data skew in stage {stage.stage_id}.{stage.attempt_id}",
            stage_id=stage.stage_id,
            stage_attempt_id=stage.attempt_id,
            confidence=confidence,
            evidence=evidence,
            metrics=metrics,
            likely_cause=likely_cause,
            recommendation=(
                "If data skew is confirmed: enable Adaptive Query Execution skew-join "
                "handling (spark.sql.adaptive.enabled=true, "
                "spark.sql.adaptive.skewJoin.enabled=true), or salt the hot key to "
                "spread it across partitions. If one join side is small, a broadcast "
                "join avoids the skewed shuffle entirely."
            ),
        )
