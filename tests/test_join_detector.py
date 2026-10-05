"""Tests for the join detector.

The join detector is deliberately conservative: a default event log does NOT
record the join strategy, so it never claims to know the join type. It only
flags a pattern *consistent with* an expensive shuffle-based join -- a large
shuffle read AND an imbalanced task distribution -- as a LOW-confidence,
MEDIUM-severity finding that tells the user to confirm in the SQL plan.

Contract:
* small shuffle read -> no finding (even if imbalanced)
* large shuffle read but balanced tasks -> no finding
* large shuffle read AND imbalance -> MEDIUM / LOW-confidence finding
* the finding is explicitly labelled as an inference, not a confirmed join type
* deterministic output
"""

from __future__ import annotations

from sparkscope.analysis.finding import Confidence, Severity
from sparkscope.analysis.join import (
    IMBALANCE_RATIO,
    MIN_SHUFFLE_READ_BYTES,
    JoinDetector,
)
from sparkscope.parser.models import SparkRun, Stage, Task, TaskMetrics

_GB = 1024**3
_MB = 1024**2


def _stage(
    stage_id: int,
    durations_ms: list[int],
    *,
    per_task_read: int,
    attempt: int = 0,
) -> Stage:
    stage = Stage(stage_id=stage_id, attempt_id=attempt, name=f"stage {stage_id}")
    for i, d in enumerate(durations_ms):
        stage.tasks.append(
            Task(
                task_id=i,
                stage_id=stage_id,
                stage_attempt_id=attempt,
                metrics=TaskMetrics(
                    duration_ms=d, shuffle_read_bytes=per_task_read
                ),
            )
        )
    return stage


def _analyze(*stages: Stage):
    run = SparkRun()
    for s in stages:
        run.stages[(s.stage_id, s.attempt_id)] = s
    return JoinDetector().analyze(run)


# --- floor: shuffle read too small -------------------------------------------


def test_small_shuffle_read_not_flagged():
    assert MIN_SHUFFLE_READ_BYTES == _GB
    # Imbalanced but tiny shuffle read -> not a join concern.
    findings = _analyze(_stage(1, [100, 100, 100, 1000], per_task_read=1 * _MB))
    assert findings == []


# --- balanced large shuffle --------------------------------------------------


def test_large_balanced_shuffle_not_flagged():
    # 4 GB total shuffle read but perfectly balanced -> no imbalance signal.
    findings = _analyze(_stage(2, [1000, 1000, 1000, 1000], per_task_read=_GB))
    assert findings == []


# --- the pattern we DO flag --------------------------------------------------


def test_large_shuffle_plus_imbalance_flags_possible_join():
    assert IMBALANCE_RATIO == 3.0
    # 4 GB shuffle read AND 10x task imbalance -> flagged.
    findings = _analyze(_stage(3, [1000, 1000, 1000, 10_000], per_task_read=_GB))
    assert len(findings) == 1
    f = findings[0]
    assert f.category == "join"
    assert f.severity == Severity.MEDIUM
    assert f.confidence == Confidence.LOW
    # Honesty: never asserts the join type is known.
    assert "does not record the join strategy" in f.likely_cause
    assert "confirm" in f.recommendation.lower()


# --- determinism -------------------------------------------------------------


def test_output_is_deterministic():
    first = _analyze(
        _stage(3, [1000, 1000, 1000, 10_000], per_task_read=_GB)
    )[0].to_dict()
    second = _analyze(
        _stage(3, [1000, 1000, 1000, 10_000], per_task_read=_GB)
    )[0].to_dict()
    assert first == second
