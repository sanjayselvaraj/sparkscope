"""Tests for the shuffle detector.

Contract:
* small shuffle (below the absolute floor) -> no finding
* large shuffle with high write-amplification vs input -> MEDIUM/HIGH
* large shuffle proportionate to input -> INFO (awareness, not a defect)
* large shuffle but input unknown (0) -> no finding (no basis to interpret)
* deterministic output

Each test builds a Stage directly so the detector is exercised in isolation.
"""

from __future__ import annotations

from sparkscope.analysis.finding import Severity
from sparkscope.analysis.shuffle import (
    MIN_SHUFFLE_BYTES,
    ShuffleDetector,
)
from sparkscope.parser.models import SparkRun, Stage, Task, TaskMetrics

_GB = 1024**3
_MB = 1024**2


def _stage(
    stage_id: int,
    *,
    per_task_write: int = 0,
    per_task_read: int = 0,
    per_task_input: int = 0,
    num_tasks: int = 4,
    attempt: int = 0,
) -> Stage:
    stage = Stage(stage_id=stage_id, attempt_id=attempt, name=f"stage {stage_id}")
    for i in range(num_tasks):
        stage.tasks.append(
            Task(
                task_id=i,
                stage_id=stage_id,
                stage_attempt_id=attempt,
                metrics=TaskMetrics(
                    duration_ms=1000,
                    shuffle_write_bytes=per_task_write,
                    shuffle_read_bytes=per_task_read,
                    input_bytes=per_task_input,
                ),
            )
        )
    return stage


def _analyze(*stages: Stage):
    run = SparkRun()
    for s in stages:
        run.stages[(s.stage_id, s.attempt_id)] = s
    return ShuffleDetector().analyze(run)


# --- floor -------------------------------------------------------------------


def test_small_shuffle_produces_no_finding():
    # Well below the 1 GB floor.
    assert MIN_SHUFFLE_BYTES == _GB
    findings = _analyze(_stage(1, per_task_write=1 * _MB, per_task_input=1 * _MB))
    assert findings == []


# --- write amplification bands ----------------------------------------------


def test_high_write_amplification_is_high_severity():
    # write 10 GB total vs input 1 GB total -> amp 10x -> HIGH
    findings = _analyze(
        _stage(
            2,
            per_task_write=(10 * _GB) // 4,
            per_task_input=(1 * _GB) // 4,
        )
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.category == "shuffle"
    assert f.severity == Severity.HIGH
    assert f.metrics["write_amplification"] >= 5.0


def test_medium_write_amplification_is_medium_severity():
    # write ~3 GB vs input 1 GB -> amp 3x -> MEDIUM
    findings = _analyze(
        _stage(
            3,
            per_task_write=(3 * _GB) // 4,
            per_task_input=(1 * _GB) // 4,
        )
    )
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_large_but_proportionate_shuffle_is_info():
    # write ~1 GB vs input 2 GB -> amp 0.5x (below MEDIUM band) -> INFO
    findings = _analyze(
        _stage(
            4,
            per_task_write=(1 * _GB) // 4 + _MB,  # keep total just over the floor
            per_task_input=(2 * _GB) // 4,
        )
    )
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


# --- unknown input -----------------------------------------------------------


def test_large_shuffle_with_unknown_input_is_silent():
    # Big shuffle but no input bytes recorded -> no basis to interpret -> silent.
    findings = _analyze(
        _stage(5, per_task_write=(4 * _GB) // 4, per_task_input=0)
    )
    assert findings == []


# --- determinism -------------------------------------------------------------


def test_output_is_deterministic():
    args = {"per_task_write": (10 * _GB) // 4, "per_task_input": (1 * _GB) // 4}
    first = _analyze(_stage(2, **args))[0].to_dict()
    second = _analyze(_stage(2, **args))[0].to_dict()
    assert first == second
