"""The SparkScope execution model.

A Spark event log is a flat, append-only stream of JSON events. On its own it is
hard to reason about. This module defines the *structured* model those events
collapse into: a run made of jobs → stages → tasks, each carrying the metrics a
detector needs.

Design notes (worth being able to defend in an interview):

* The model is intentionally a thin, typed projection of the log -- NOT a
  faithful mirror of every Spark field. We keep only what detectors use, so the
  surface stays small and the parser stays simple.
* ``TaskMetrics`` holds the handful of fields the v0.1 detectors read
  (run time, shuffle read/write bytes, memory/disk spill). Everything else in a
  Spark ``TaskEnd`` event is discarded on purpose.
* Durations are milliseconds (Spark's native unit in the log) to avoid lossy
  conversions; formatting to human units is a reporting concern, not a model one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass
class TaskMetrics:
    """Per-task metrics distilled from a ``SparkListenerTaskEnd`` event."""

    duration_ms: int = 0
    shuffle_read_bytes: int = 0
    shuffle_write_bytes: int = 0
    memory_spilled_bytes: int = 0
    disk_spilled_bytes: int = 0
    input_bytes: int = 0
    output_bytes: int = 0


@dataclass
class Task:
    """A single task attempt within a stage."""

    task_id: int
    stage_id: int
    stage_attempt_id: int
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    failed: bool = False


@dataclass
class Stage:
    """A stage: a set of tasks separated from neighbours by a shuffle boundary."""

    stage_id: int
    attempt_id: int
    name: str = ""
    num_tasks: int = 0
    tasks: list[Task] = field(default_factory=list)

    # -- aggregate helpers the detectors lean on -------------------------------

    @property
    def task_durations(self) -> list[int]:
        return [t.metrics.duration_ms for t in self.tasks]

    @property
    def max_task_ms(self) -> int:
        durs = self.task_durations
        return max(durs) if durs else 0

    @property
    def median_task_ms(self) -> float:
        durs = self.task_durations
        return median(durs) if durs else 0.0

    @property
    def total_shuffle_read_bytes(self) -> int:
        return sum(t.metrics.shuffle_read_bytes for t in self.tasks)

    @property
    def total_shuffle_write_bytes(self) -> int:
        return sum(t.metrics.shuffle_write_bytes for t in self.tasks)

    @property
    def total_spill_bytes(self) -> int:
        return sum(
            t.metrics.memory_spilled_bytes + t.metrics.disk_spilled_bytes for t in self.tasks
        )

    @property
    def skew_ratio(self) -> float:
        """Max task time divided by median task time.

        A value near 1.0 means a balanced stage; a large value means one (or a
        few) tasks did disproportionately more work -- the signature of skew.
        Returns 0.0 when there is not enough data to judge.
        """
        med = self.median_task_ms
        if med <= 0 or len(self.tasks) < 2:
            return 0.0
        return self.max_task_ms / med


@dataclass
class Job:
    """A job: the unit Spark creates per action (e.g. ``count()``)."""

    job_id: int
    stage_ids: list[int] = field(default_factory=list)


@dataclass
class SparkRun:
    """The whole application: the root the parser produces and detectors consume."""

    app_name: str = ""
    app_id: str = ""
    jobs: dict[int, Job] = field(default_factory=dict)
    stages: dict[tuple[int, int], Stage] = field(default_factory=dict)

    def stage_list(self) -> list[Stage]:
        """All stages, in a stable order (by stage id then attempt)."""
        return [self.stages[k] for k in sorted(self.stages.keys())]
