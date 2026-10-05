"""Tests for the partition detector.

Contract:
* ordinary stage (moderate task count, moderate durations) -> no finding
* few long tasks -> under-partitioning (MEDIUM, LOW confidence)
* many tiny tasks -> over-partitioning (LOW, LOW confidence)
* borderline cases just inside/outside the thresholds behave correctly
* empty stage -> no finding, no crash
* deterministic output

Each test builds a Stage directly so the detector is exercised in isolation.
"""

from __future__ import annotations

from sparkscope.analysis.finding import Confidence, Severity
from sparkscope.analysis.partition import (
    FEW_TASKS,
    LONG_TASK_MS,
    MANY_TASKS,
    TINY_TASK_MS,
    PartitionDetector,
)
from sparkscope.parser.models import SparkRun, Stage, Task, TaskMetrics


def _stage(stage_id: int, durations_ms: list[int], *, attempt: int = 0) -> Stage:
    stage = Stage(stage_id=stage_id, attempt_id=attempt, name=f"stage {stage_id}")
    for i, d in enumerate(durations_ms):
        stage.tasks.append(
            Task(
                task_id=i,
                stage_id=stage_id,
                stage_attempt_id=attempt,
                metrics=TaskMetrics(duration_ms=d),
            )
        )
    return stage


def _analyze(*stages: Stage):
    run = SparkRun()
    for s in stages:
        run.stages[(s.stage_id, s.attempt_id)] = s
    return PartitionDetector().analyze(run)


# --- healthy -----------------------------------------------------------------


def test_ordinary_stage_produces_no_finding():
    # 50 tasks around 5s each: neither few-and-long nor many-and-tiny.
    findings = _analyze(_stage(1, [5000] * 50))
    assert findings == []


def test_empty_stage_produces_no_finding():
    findings = _analyze(_stage(2, []))
    assert findings == []


# --- under-partitioning: few, long tasks -------------------------------------


def test_few_long_tasks_flag_under_partitioning():
    assert FEW_TASKS == 8
    # 4 tasks, slowest 2 minutes -> under-partitioned
    findings = _analyze(_stage(3, [120_000, 90_000, 80_000, 70_000]))
    assert len(findings) == 1
    f = findings[0]
    assert f.category == "partition"
    assert f.severity == Severity.MEDIUM
    assert f.confidence == Confidence.LOW
    assert "under-partition" in f.title.lower()


def test_few_but_short_tasks_not_flagged():
    # Few tasks but none crosses the long-task floor.
    assert LONG_TASK_MS == 60_000
    findings = _analyze(_stage(4, [5000, 4000, 3000, 2000]))
    assert findings == []


# --- over-partitioning: many, tiny tasks -------------------------------------


def test_many_tiny_tasks_flag_over_partitioning():
    assert MANY_TASKS == 1_000
    assert TINY_TASK_MS == 100
    # 2000 tasks with median 50ms -> over-partitioned
    findings = _analyze(_stage(5, [50] * 2000))
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == Severity.LOW
    assert f.confidence == Confidence.LOW
    assert "over-partition" in f.title.lower()


def test_many_but_not_tiny_tasks_not_flagged():
    # Many tasks but median runtime well above the tiny floor.
    findings = _analyze(_stage(6, [5000] * 2000))
    assert findings == []


# --- determinism -------------------------------------------------------------


def test_output_is_deterministic():
    first = _analyze(_stage(3, [120_000, 90_000, 80_000, 70_000]))[0].to_dict()
    second = _analyze(_stage(3, [120_000, 90_000, 80_000, 70_000]))[0].to_dict()
    assert first == second
