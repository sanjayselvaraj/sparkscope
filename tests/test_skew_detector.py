"""Tests for the data-skew detector.

Covers the full behavior contract:
* healthy stage -> no finding
* moderate / severe skew -> correct severity bands
* corroboration: data-volume skew raises confidence; time-only skew lowers it
  AND caps severity (we never cry CRITICAL "data skew" without data evidence)
* suppression: too few tasks, trivially short hot task
* zero / invalid durations
* stage attempts handled independently
* deterministic output (same input -> identical findings)

Each test builds a Stage directly so the detector is exercised in isolation.
"""

from __future__ import annotations

from sparkscope.analysis.base import run_all
from sparkscope.analysis.finding import Confidence, Severity
from sparkscope.analysis.skew import (
    MIN_HOT_TASK_MS,
    MIN_TASKS,
    SkewDetector,
)
from sparkscope.parser.models import SparkRun, Stage, Task, TaskMetrics


def _stage(
    stage_id: int,
    durations_ms: list[int],
    *,
    shuffle_read: list[int] | None = None,
    attempt: int = 0,
) -> Stage:
    """Build a stage. ``shuffle_read`` (per task) lets a test add data-volume
    skew so the detector can corroborate time skew."""
    stage = Stage(stage_id=stage_id, attempt_id=attempt, name=f"stage {stage_id}")
    reads = shuffle_read or [0] * len(durations_ms)
    for i, (d, r) in enumerate(zip(durations_ms, reads, strict=True)):
        stage.tasks.append(
            Task(
                task_id=i,
                stage_id=stage_id,
                stage_attempt_id=attempt,
                metrics=TaskMetrics(duration_ms=d, shuffle_read_bytes=r),
            )
        )
    return stage


def _run_with(*stages: Stage) -> SparkRun:
    run = SparkRun()
    for s in stages:
        run.stages[(s.stage_id, s.attempt_id)] = s
    return run


def _analyze(*stages: Stage):
    return SkewDetector().analyze(_run_with(*stages))


# --- healthy -----------------------------------------------------------------


def test_healthy_balanced_stage_produces_no_finding():
    findings = _analyze(_stage(2, [1000, 1050, 980, 1020]))
    assert findings == []


# --- corroborated skew (time AND data lopsided) ------------------------------


def test_severe_skew_with_data_corroboration_is_critical_high_confidence():
    findings = _analyze(
        _stage(
            1,
            [100, 100, 100, 10_000],
            shuffle_read=[1_000_000, 1_000_000, 1_000_000, 90_000_000],
        )
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.category == "skew"
    assert f.severity == Severity.CRITICAL
    assert f.confidence == Confidence.HIGH
    assert f.metrics["time_skew_ratio"] == 100.0
    assert f.metrics["data_skew_ratio"] >= 2.0
    assert "consistent with data skew" in f.likely_cause
    assert any("more data" in e for e in f.evidence)


def test_moderate_skew_band():
    findings = _analyze(
        _stage(10, [2000, 2000, 2000, 5000], shuffle_read=[10, 10, 10, 100])
    )
    assert findings[0].severity == Severity.MEDIUM


def test_high_skew_band():
    findings = _analyze(
        _stage(11, [1000, 1000, 1000, 6000], shuffle_read=[10, 10, 10, 100])
    )
    assert findings[0].severity == Severity.HIGH


# --- uncorroborated skew (time lopsided, data NOT) ---------------------------


def test_time_only_skew_lowers_confidence_and_caps_severity():
    # 100x TIME skew but every task read the same data -> not classic data skew.
    findings = _analyze(
        _stage(5, [100, 100, 100, 10_000], shuffle_read=[1_000, 1_000, 1_000, 1_000])
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.confidence == Confidence.LOW
    # Severity capped at HIGH (never CRITICAL) without data corroboration.
    assert f.severity == Severity.HIGH
    assert "straggler" in f.likely_cause.lower()


# --- suppression gates -------------------------------------------------------


def test_suppressed_when_too_few_tasks():
    assert MIN_TASKS == 4
    findings = _analyze(_stage(3, [100, 100, 10_000]))
    assert findings == []


def test_suppressed_when_hot_task_trivially_short():
    assert MIN_HOT_TASK_MS == 1_000
    findings = _analyze(_stage(4, [10, 10, 10, 300]))
    assert findings == []


# --- zero / invalid durations ------------------------------------------------


def test_all_zero_durations_produces_no_finding():
    findings = _analyze(_stage(6, [0, 0, 0, 0]))
    assert findings == []


def test_zero_median_with_one_hot_task_does_not_crash():
    # median duration is 0 (three zeros), one long task; skew_ratio guards to 0.
    findings = _analyze(_stage(7, [0, 0, 0, 5000]))
    # skew_ratio returns 0.0 when median <= 0, so no finding -- and no divide error.
    assert findings == []


# --- stage attempts ----------------------------------------------------------


def test_stage_attempts_evaluated_independently():
    balanced_retry = _stage(8, [1000, 1000, 1000, 1000], attempt=1)
    skewed_first = _stage(
        8, [100, 100, 100, 9000], shuffle_read=[1, 1, 1, 90], attempt=0
    )
    findings = _analyze(skewed_first, balanced_retry)
    # Only attempt 0 is skewed; attempt 1 (the retry) is healthy.
    assert len(findings) == 1
    assert findings[0].stage_attempt_id == 0


# --- ranking + determinism ---------------------------------------------------


def test_findings_ranked_worst_first():
    run = _run_with(
        _stage(20, [2000, 2000, 2000, 5000], shuffle_read=[10, 10, 10, 100]),  # MEDIUM
        _stage(21, [1000, 1000, 1000, 20000], shuffle_read=[10, 10, 10, 500]),  # CRITICAL
    )
    findings = run_all(run)
    assert findings[0].severity == Severity.CRITICAL
    assert findings[0].stage_id == 21
    assert findings[1].severity == Severity.MEDIUM


def test_output_is_deterministic():
    stage_args = (1, [100, 100, 100, 10_000])
    kwargs = {"shuffle_read": [1_000_000, 1_000_000, 1_000_000, 90_000_000]}
    first = _analyze(_stage(*stage_args, **kwargs))[0].to_dict()
    second = _analyze(_stage(*stage_args, **kwargs))[0].to_dict()
    assert first == second
