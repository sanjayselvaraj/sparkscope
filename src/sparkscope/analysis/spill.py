"""Spill detector.

Spill happens when a task cannot fit its working set in execution memory and
writes intermediate data to disk. A little spill is common and harmless; a lot
of spill means the stage is memory-starved and is paying a heavy disk/serialize
penalty.

The event log reports, per task, ``Memory Bytes Spilled`` and ``Disk Bytes
Spilled``. (Note: "memory spilled" is the in-memory size of data that was then
written to disk; "disk spilled" is the on-disk size. Both being non-zero is the
signature of real spilling.) We grade on the on-disk spill volume because that
is what actually hits the disk, and on how concentrated it is.

Bands
-----
* any spill below ``MIN_DISK_SPILL_BYTES`` -> not reported (noise)
* >= MEDIUM / HIGH / CRITICAL disk-spill totals -> graded severity
We also note how many tasks spilled, to distinguish a single memory-starved task
from a stage-wide memory shortfall.
"""

from __future__ import annotations

from sparkscope.analysis.base import Detector, register
from sparkscope.analysis.finding import Confidence, Finding, Severity
from sparkscope.analysis.util import format_bytes, ratio
from sparkscope.parser.models import SparkRun, Stage

#: Below this on-disk spill total, a stage is not flagged.
MIN_DISK_SPILL_BYTES = 128 * 1024**2  # 128 MB
MEDIUM_SPILL_BYTES = 1 * 1024**3  # 1 GB
HIGH_SPILL_BYTES = 10 * 1024**3  # 10 GB
CRITICAL_SPILL_BYTES = 50 * 1024**3  # 50 GB


def _severity_for(disk_spill: int) -> Severity | None:
    if disk_spill >= CRITICAL_SPILL_BYTES:
        return Severity.CRITICAL
    if disk_spill >= HIGH_SPILL_BYTES:
        return Severity.HIGH
    if disk_spill >= MEDIUM_SPILL_BYTES:
        return Severity.MEDIUM
    if disk_spill >= MIN_DISK_SPILL_BYTES:
        return Severity.LOW
    return None


@register
class SpillDetector(Detector):
    """Flags stages that spill a significant amount of data to disk."""

    name = "spill"
    category = "spill"

    def analyze(self, run: SparkRun) -> list[Finding]:
        findings: list[Finding] = []
        for stage in run.stage_list():
            finding = self._check_stage(stage)
            if finding is not None:
                findings.append(finding)
        return findings

    def _check_stage(self, stage: Stage) -> Finding | None:
        disk_spill = stage.total_disk_spilled_bytes
        mem_spill = stage.total_memory_spilled_bytes

        severity = _severity_for(disk_spill)
        if severity is None:
            return None

        spilling_tasks = stage.tasks_with_spill
        total_tasks = len(stage.tasks)
        spill_task_fraction = ratio(spilling_tasks, total_tasks)

        evidence = [
            f"stage spilled {format_bytes(disk_spill)} to disk "
            f"(in-memory size {format_bytes(mem_spill)})",
            f"{spilling_tasks}/{total_tasks} tasks spilled",
        ]

        # Distinguish stage-wide memory shortfall from one hot/starved task.
        if spill_task_fraction >= 0.5:
            likely_cause = (
                "Most tasks in the stage spilled, which is consistent with the stage "
                "being memory-starved overall: partitions are too large for the "
                "available execution memory per task."
            )
        else:
            likely_cause = (
                "Only a minority of tasks spilled, which is consistent with a few "
                "oversized partitions (possible skew) rather than a stage-wide memory "
                "shortfall."
            )

        metrics = {
            "disk_spilled_bytes": float(disk_spill),
            "memory_spilled_bytes": float(mem_spill),
            "spilling_tasks": float(spilling_tasks),
            "total_tasks": float(total_tasks),
            "spill_task_fraction": round(spill_task_fraction, 2),
        }

        return Finding(
            severity=severity,
            sort_index=float(disk_spill),
            category=self.category,
            detector=self.name,
            title=f"Disk spill in stage {stage.stage_id}.{stage.attempt_id}",
            stage_id=stage.stage_id,
            stage_attempt_id=stage.attempt_id,
            confidence=Confidence.HIGH,  # spill bytes are directly observed, not inferred
            evidence=evidence,
            metrics=metrics,
            likely_cause=likely_cause,
            recommendation=(
                "Increase execution memory (spark.executor.memory / memoryOverhead) "
                "or reduce partition size so each task's working set fits in memory "
                "(raise spark.sql.shuffle.partitions, or repartition). If only a few "
                "tasks spill, check for skew on the partition key."
            ),
        )
