"""Partition detector.

Partitioning problems show up as task-size extremes:

* **Too few partitions** -- a handful of very large, long-running tasks that
  under-use the cluster and risk spilling.
* **Too many tiny partitions** -- thousands of sub-second tasks whose scheduling
  overhead dominates the actual work.

The event log does NOT tell us the "right" number of partitions -- that depends
on cluster size and data, which the log doesn't contain. So we never prescribe
an exact count. We only flag *distributions* that are extreme enough to be worth
a look, and we phrase the cause as a hypothesis.

Signals
-------
* Few-large: a small number of tasks (< ``FEW_TASKS``) each running longer than
  ``LONG_TASK_MS``. Consistent with under-partitioning.
* Many-tiny: a large number of tasks (> ``MANY_TASKS``) whose median runtime is
  below ``TINY_TASK_MS``. Consistent with over-partitioning / scheduling churn.

These are intentionally conservative so we don't flag ordinary stages.
"""

from __future__ import annotations

from sparkscope.analysis.base import Detector, register
from sparkscope.analysis.finding import Confidence, Finding, Severity
from sparkscope.analysis.util import format_ms
from sparkscope.parser.models import SparkRun, Stage

#: "Few" partitions: at or below this task count, with long tasks -> under-partitioned.
FEW_TASKS = 8
#: A task is "long" above this (ms) -- used with FEW_TASKS.
LONG_TASK_MS = 60_000  # 1 minute
#: "Many" partitions: above this task count, with tiny tasks -> over-partitioned.
MANY_TASKS = 1_000
#: A task is "tiny" below this median (ms) -- used with MANY_TASKS.
TINY_TASK_MS = 100


@register
class PartitionDetector(Detector):
    """Flags stages whose task-size distribution suggests over/under-partitioning."""

    name = "partition"
    category = "partition"

    def analyze(self, run: SparkRun) -> list[Finding]:
        findings: list[Finding] = []
        for stage in run.stage_list():
            finding = self._check_stage(stage)
            if finding is not None:
                findings.append(finding)
        return findings

    def _check_stage(self, stage: Stage) -> Finding | None:
        n = len(stage.tasks)
        if n == 0:
            return None
        median_ms = stage.median_task_ms
        max_ms = stage.max_task_ms

        # --- under-partitioned: few, long tasks --------------------------------
        if n <= FEW_TASKS and max_ms >= LONG_TASK_MS:
            evidence = [
                f"only {n} task(s), slowest {format_ms(max_ms)} "
                f"(median {format_ms(median_ms)})",
            ]
            return Finding(
                severity=Severity.MEDIUM,
                sort_index=float(max_ms),
                category=self.category,
                detector=self.name,
                title=f"Possible under-partitioning in stage {stage.stage_id}.{stage.attempt_id}",
                stage_id=stage.stage_id,
                stage_attempt_id=stage.attempt_id,
                confidence=Confidence.LOW,
                evidence=evidence,
                metrics={
                    "num_tasks": float(n),
                    "max_task_ms": float(max_ms),
                    "median_task_ms": round(median_ms, 2),
                },
                likely_cause=(
                    "Few tasks each running a long time is consistent with too few "
                    "partitions, so the work is not spread across the cluster. (The "
                    "event log cannot confirm cluster size, so this is a hypothesis.)"
                ),
                recommendation=(
                    "Consider increasing parallelism (repartition, or raise "
                    "spark.sql.shuffle.partitions) so work spreads across more tasks. "
                    "Verify against your executor/core count before changing."
                ),
            )

        # --- over-partitioned: many, tiny tasks --------------------------------
        if n >= MANY_TASKS and 0 < median_ms <= TINY_TASK_MS:
            evidence = [
                f"{n} tasks with a median runtime of only {format_ms(median_ms)}",
            ]
            return Finding(
                severity=Severity.LOW,
                sort_index=float(n),
                category=self.category,
                detector=self.name,
                title=f"Possible over-partitioning in stage {stage.stage_id}.{stage.attempt_id}",
                stage_id=stage.stage_id,
                stage_attempt_id=stage.attempt_id,
                confidence=Confidence.LOW,
                evidence=evidence,
                metrics={
                    "num_tasks": float(n),
                    "median_task_ms": round(median_ms, 2),
                },
                likely_cause=(
                    "A very large number of very short tasks is consistent with too "
                    "many partitions, where per-task scheduling overhead starts to "
                    "dominate useful work."
                ),
                recommendation=(
                    "Consider coalescing to fewer partitions (coalesce, or lower "
                    "spark.sql.shuffle.partitions) so each task does more useful work "
                    "relative to scheduling overhead."
                ),
            )

        return None
