"""Tests for the spill detector.

Contract:
* no spill / sub-threshold spill -> no finding
* MEDIUM / HIGH / CRITICAL disk-spill bands -> correct severity
* spill is HIGH confidence (bytes are directly observed, not inferred)
* stage-wide spill vs a-few-tasks spill produces different likely_cause wording
* deterministic output

Each test builds a Stage directly so the detector is exercised in isolation.
"""

from __future__ import annotations

from sparkscope.analysis.finding import Confidence, Severity
from sparkscope.analysis.spill import (
    MIN_DISK_SPILL_BYTES,
    SpillDetector,
)
from sparkscope.parser.models import SparkRun, Stage, Task, TaskMetrics

_GB = 1024**3
_MB = 1024**2


def _stage(
    stage_id: int,
    *,
    disk_per_task: list[int],
    mem_per_task: list[int] | None = None,
    attempt: int = 0,
) -> Stage:
    stage = Stage(stage_id=stage_id, attempt_id=attempt, name=f"stage {stage_id}")
    mem = mem_per_task if mem_per_task is not None else disk_per_task
    for i, (d, m) in enumerate(zip(disk_per_task, mem, strict=True)):
        stage.tasks.append(
            Task(
                task_id=i,
                stage_id=stage_id,
                stage_attempt_id=attempt,
                metrics=TaskMetrics(
                    duration_ms=1000,
                    disk_spilled_bytes=d,
                    memory_spilled_bytes=m,
                ),
            )
        )
    return stage


def _analyze(*stages: Stage):
    run = SparkRun()
    for s in stages:
        run.stages[(s.stage_id, s.attempt_id)] = s
    return SpillDetector().analyze(run)


# --- no / sub-threshold spill ------------------------------------------------


def test_no_spill_produces_no_finding():
    findings = _analyze(_stage(1, disk_per_task=[0, 0, 0, 0]))
    assert findings == []


def test_subthreshold_spill_produces_no_finding():
    # Total below the 128 MB floor.
    assert MIN_DISK_SPILL_BYTES == 128 * _MB
    findings = _analyze(_stage(2, disk_per_task=[10 * _MB, 10 * _MB]))
    assert findings == []


# --- severity bands ----------------------------------------------------------


def test_medium_spill_band():
    # ~2 GB total -> MEDIUM (>= 1 GB, < 10 GB)
    findings = _analyze(_stage(3, disk_per_task=[1 * _GB, 1 * _GB]))
    assert len(findings) == 1
    f = findings[0]
    assert f.category == "spill"
    assert f.severity == Severity.MEDIUM
    assert f.confidence == Confidence.HIGH


def test_high_spill_band():
    # ~12 GB total -> HIGH (>= 10 GB, < 50 GB)
    findings = _analyze(_stage(4, disk_per_task=[6 * _GB, 6 * _GB]))
    assert findings[0].severity == Severity.HIGH


def test_critical_spill_band():
    # ~60 GB total -> CRITICAL (>= 50 GB)
    findings = _analyze(_stage(5, disk_per_task=[30 * _GB, 30 * _GB]))
    assert findings[0].severity == Severity.CRITICAL


# --- cause wording: stage-wide vs few tasks ----------------------------------


def test_stage_wide_spill_wording():
    # All 4 tasks spill -> "memory-starved overall"
    findings = _analyze(
        _stage(6, disk_per_task=[1 * _GB, 1 * _GB, 1 * _GB, 1 * _GB])
    )
    assert "memory-starved overall" in findings[0].likely_cause
    assert findings[0].metrics["spill_task_fraction"] == 1.0


def test_few_tasks_spill_wording():
    # 1 of 4 tasks spills but crosses the volume floor -> "a few oversized partitions"
    findings = _analyze(_stage(7, disk_per_task=[2 * _GB, 0, 0, 0]))
    assert "oversized partitions" in findings[0].likely_cause
    assert findings[0].metrics["spill_task_fraction"] == 0.25


# --- determinism -------------------------------------------------------------


def test_output_is_deterministic():
    first = _analyze(_stage(3, disk_per_task=[1 * _GB, 1 * _GB]))[0].to_dict()
    second = _analyze(_stage(3, disk_per_task=[1 * _GB, 1 * _GB]))[0].to_dict()
    assert first == second
