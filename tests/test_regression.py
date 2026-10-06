"""Tests for the regression JUDGMENT engine.

Mirrors ``test_compare.py``: build ``Stage``/``SparkRun`` directly with small
local helpers and run them through ``compare_runs`` so the engine is exercised
end to end against the real factual model. Where it is clearer, construct the
``StageComparison``/``LogicalStage``/``MetricDelta`` objects directly.

Every branch of context normalization and the significance gate has a test, and
the make-or-break anti-false-positive rule (growth proportional to input is NOT
a regression) is called out explicitly.
"""

from __future__ import annotations

from sparkscope.analysis.compare import (
    LogicalStage,
    MetricDelta,
    RunComparison,
    StageComparison,
    compare_runs,
)
from sparkscope.analysis.finding import Confidence, Finding, Severity
from sparkscope.analysis.regression import (
    RegressionThresholds,
    analyze_regressions,
)
from sparkscope.parser.models import Job, SparkRun, Stage, Task, TaskMetrics

MB = 1024 * 1024


def _stage(
    stage_id: int,
    durations_ms: list[int],
    *,
    name: str = "",
    attempt: int = 0,
    shuffle_read: int = 0,
    shuffle_write: int = 0,
    mem_spill: int = 0,
    disk_spill: int = 0,
    input_bytes: int = 0,
    output_bytes: int = 0,
) -> Stage:
    """Build a stage with one Task per duration (byte metrics on the first task)."""
    stage = Stage(stage_id=stage_id, attempt_id=attempt, name=name)
    for i, d in enumerate(durations_ms):
        first = i == 0
        stage.tasks.append(
            Task(
                task_id=i,
                stage_id=stage_id,
                stage_attempt_id=attempt,
                metrics=TaskMetrics(
                    duration_ms=d,
                    shuffle_read_bytes=shuffle_read if first else 0,
                    shuffle_write_bytes=shuffle_write if first else 0,
                    memory_spilled_bytes=mem_spill if first else 0,
                    disk_spilled_bytes=disk_spill if first else 0,
                    input_bytes=input_bytes if first else 0,
                    output_bytes=output_bytes if first else 0,
                ),
            )
        )
    return stage


def _run(*stages: Stage, app_name: str = "app", jobs: int = 0) -> SparkRun:
    run = SparkRun(app_name=app_name)
    for s in stages:
        run.stages[(s.stage_id, s.attempt_id)] = s
    for j in range(jobs):
        run.jobs[j] = Job(job_id=j)
    return run


def _logical(
    stage_id: int,
    *,
    name: str = "s",
    duration_ms_total: float = 0.0,
    task_count: int = 0,
    median_task_ms: float = 0.0,
    max_task_ms: int = 0,
    skew_ratio: float = 0.0,
    shuffle_read_bytes: int = 0,
    shuffle_write_bytes: int = 0,
    memory_spilled_bytes: int = 0,
    disk_spilled_bytes: int = 0,
    input_bytes: int = 0,
) -> LogicalStage:
    return LogicalStage(
        stage_id=stage_id,
        name=name,
        attempt_count=1,
        duration_ms_total=duration_ms_total,
        task_count=task_count,
        median_task_ms=median_task_ms,
        max_task_ms=max_task_ms,
        skew_ratio=skew_ratio,
        shuffle_read_bytes=shuffle_read_bytes,
        shuffle_write_bytes=shuffle_write_bytes,
        memory_spilled_bytes=memory_spilled_bytes,
        disk_spilled_bytes=disk_spilled_bytes,
        input_bytes=input_bytes,
    )


def _matched_comparison(
    base: LogicalStage,
    cur: LogicalStage,
    *,
    match_method: str = "name",
) -> RunComparison:
    """Build a one-stage RunComparison directly with chosen fixed-order deltas."""
    names_units = [
        ("duration_ms_total", "ms"),
        ("task_count", "count"),
        ("median_task_ms", "ms"),
        ("max_task_ms", "ms"),
        ("skew_ratio", "ratio"),
        ("shuffle_read_bytes", "bytes"),
        ("shuffle_write_bytes", "bytes"),
        ("memory_spilled_bytes", "bytes"),
        ("disk_spilled_bytes", "bytes"),
        ("input_bytes", "bytes"),
    ]
    deltas = [
        MetricDelta(
            name=n,
            baseline=float(getattr(base, n)),
            current=float(getattr(cur, n)),
            unit=u,  # type: ignore[arg-type]
        )
        for n, u in names_units
    ]
    sc = StageComparison(
        stage_key=cur.name or f"stage {cur.stage_id}",
        status="matched",
        match_method=match_method,  # type: ignore[arg-type]
        baseline=base,
        current=cur,
        metric_deltas=deltas,
    )
    return RunComparison(
        baseline_app="base",
        current_app="cur",
        app_deltas=[],
        stage_comparisons=[sc],
        baseline_skipped_lines=0,
        current_skipped_lines=0,
    )


def _by_detector(findings: list[Finding]) -> dict[str, Finding]:
    return {f.detector: f for f in findings}


# --- no regression -----------------------------------------------------------


def test_identical_runs_no_findings():
    base = _run(_stage(1, [100, 100], name="s", shuffle_read=10 * MB))
    cur = _run(_stage(1, [100, 100], name="s", shuffle_read=10 * MB))
    assert analyze_regressions(compare_runs(base, cur)) == []


def test_all_zero_deltas_no_findings():
    base = _run(_stage(1, [0, 0], name="s"))
    cur = _run(_stage(1, [0, 0], name="s"))
    assert analyze_regressions(compare_runs(base, cur)) == []


# --- make-or-break: input-proportional growth is NOT a regression ------------


def test_input_proportional_growth_is_not_a_regression():
    # Input doubled AND shuffle doubled in proportion -> scaling, NOT a
    # regression. This is the central anti-false-positive rule.
    base = _logical(
        1, duration_ms_total=100_000, input_bytes=1000 * MB, shuffle_read_bytes=1000 * MB
    )
    cur = _logical(
        1, duration_ms_total=200_000, input_bytes=2000 * MB, shuffle_read_bytes=2000 * MB
    )
    findings = analyze_regressions(_matched_comparison(base, cur))
    assert [f.detector for f in findings] == []  # nothing flagged


# --- branch (ii): input flat, metric over threshold + floor ------------------


def test_shuffle_regression_input_flat():
    # Input flat; shuffle well over 30% threshold and over the 256MB floor.
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB)
    cur = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=2000 * MB)
    findings = analyze_regressions(_matched_comparison(base, cur))
    by = _by_detector(findings)
    assert "regression.shuffle" in by
    f = by["regression.shuffle"]
    assert any("input approximately flat" in e for e in f.evidence)
    # 300% / 30% = 10x excess -> CRITICAL.
    assert f.severity is Severity.CRITICAL


def test_duration_regression_input_flat():
    base = _logical(1, duration_ms_total=100_000, input_bytes=1000 * MB)
    cur = _logical(1, duration_ms_total=150_000, input_bytes=1000 * MB)
    by = _by_detector(analyze_regressions(_matched_comparison(base, cur)))
    assert "regression.duration" in by
    assert any("input approximately flat" in e for e in by["regression.duration"].evidence)


def test_spill_regression_input_flat():
    base = _logical(1, input_bytes=1000 * MB, memory_spilled_bytes=400 * MB)
    cur = _logical(1, input_bytes=1000 * MB, memory_spilled_bytes=1000 * MB)
    by = _by_detector(analyze_regressions(_matched_comparison(base, cur)))
    assert "regression.spill" in by


# --- branch (iii): input grew but metric grew faster -------------------------


def test_metric_outgrew_input():
    # Input +50%, shuffle +300% -> normalized factor well above 1 -> flagged.
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB)
    cur = _logical(1, input_bytes=1500 * MB, shuffle_read_bytes=2000 * MB)
    by = _by_detector(analyze_regressions(_matched_comparison(base, cur)))
    assert "regression.shuffle" in by
    f = by["regression.shuffle"]
    assert "normalized_factor" in f.metrics
    # metric_growth 4.0 / input_growth 1.5 = 2.67.
    assert abs(f.metrics["normalized_factor"] - (4.0 / 1.5)) < 1e-6
    assert any("normalized factor" in e.lower() for e in f.evidence)


def test_metric_grew_below_input_not_flagged():
    # Input +100%, shuffle only +50% -> grew slower than input -> benign.
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=1000 * MB)
    cur = _logical(1, input_bytes=2000 * MB, shuffle_read_bytes=1500 * MB)
    assert analyze_regressions(_matched_comparison(base, cur)) == []


# --- zero-baseline: a new cost appeared --------------------------------------


def test_zero_baseline_new_spill_cost():
    # Spill newly appeared (baseline 0) above the floor -> flagged, LOW conf.
    base = _logical(1, input_bytes=1000 * MB, memory_spilled_bytes=0)
    cur = _logical(1, input_bytes=1000 * MB, memory_spilled_bytes=1000 * MB)
    by = _by_detector(analyze_regressions(_matched_comparison(base, cur)))
    assert "regression.spill" in by
    f = by["regression.spill"]
    assert f.confidence is Confidence.LOW  # no ratio to corroborate
    assert any("baseline was zero" in e for e in f.evidence)


def test_zero_baseline_below_floor_not_flagged():
    base = _logical(1, input_bytes=1000 * MB, memory_spilled_bytes=0)
    cur = _logical(1, input_bytes=1000 * MB, memory_spilled_bytes=10 * MB)
    assert analyze_regressions(_matched_comparison(base, cur)) == []


# --- significance gate suppressions ------------------------------------------


def test_below_floor_suppressed():
    # +400% shuffle but current only 10MB (< 256MB floor) -> NOT flagged.
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=2 * MB)
    cur = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=10 * MB)
    assert analyze_regressions(_matched_comparison(base, cur)) == []


def test_below_percent_threshold_suppressed():
    # Clears the 256MB floor but only +10% (< 30% shuffle threshold) -> NOT flagged.
    base = _logical(1, input_bytes=10000 * MB, shuffle_read_bytes=1000 * MB)
    cur = _logical(1, input_bytes=10000 * MB, shuffle_read_bytes=1100 * MB)
    assert analyze_regressions(_matched_comparison(base, cur)) == []


# --- skew --------------------------------------------------------------------


def test_skew_regression_stable_task_count_hot_key():
    # skew 2x -> 30x, task count stable -> flagged, hot-key hypothesis.
    base = _logical(1, skew_ratio=2.0, task_count=200, input_bytes=1000 * MB)
    cur = _logical(1, skew_ratio=30.0, task_count=200, input_bytes=1000 * MB)
    by = _by_detector(analyze_regressions(_matched_comparison(base, cur)))
    assert "regression.skew" in by
    f = by["regression.skew"]
    assert any("task count stable" in e for e in f.evidence)
    assert "hot key" in f.likely_cause.lower()


def test_skew_task_count_changed_repartition_wording():
    base = _logical(1, skew_ratio=2.0, task_count=200, input_bytes=1000 * MB)
    cur = _logical(1, skew_ratio=30.0, task_count=50, input_bytes=1000 * MB)
    f = _by_detector(analyze_regressions(_matched_comparison(base, cur)))["regression.skew"]
    assert any("task count changed" in e for e in f.evidence)
    assert "repartition" in f.likely_cause.lower()


def test_skew_below_current_ratio_not_flagged():
    # Increased sharply but current ratio 1.8 < 2.0 min -> NOT flagged.
    base = _logical(1, skew_ratio=1.0, task_count=200)
    cur = _logical(1, skew_ratio=1.8, task_count=200)
    assert analyze_regressions(_matched_comparison(base, cur)) == []


def test_skew_increase_below_min_not_flagged():
    # Current ratio over 2.0 but increase only +25% (< 50% min) -> NOT flagged.
    base = _logical(1, skew_ratio=2.4, task_count=200)
    cur = _logical(1, skew_ratio=3.0, task_count=200)
    assert analyze_regressions(_matched_comparison(base, cur)) == []


def test_skew_zero_baseline_capped_confidence():
    # Baseline skew 0 (undefined % increase) -> current-ratio floor only, LOW conf.
    base = _logical(1, skew_ratio=0.0, task_count=200)
    cur = _logical(1, skew_ratio=5.0, task_count=200)
    f = _by_detector(analyze_regressions(_matched_comparison(base, cur)))["regression.skew"]
    assert f.confidence is Confidence.LOW


# --- added / removed ---------------------------------------------------------


def test_added_expensive_stage_flagged():
    base = _run(_stage(1, [100], name="a", input_bytes=10 * MB))
    cur = _run(
        _stage(1, [100], name="a", input_bytes=10 * MB),
        _stage(2, [100], name="b", input_bytes=1000 * MB),
    )
    by = _by_detector(analyze_regressions(compare_runs(base, cur)))
    assert "regression.new_stage" in by
    assert by["regression.new_stage"].confidence is Confidence.LOW


def test_removed_stage_not_flagged():
    base = _run(
        _stage(1, [100], name="a", input_bytes=10 * MB),
        _stage(2, [100], name="b", input_bytes=1000 * MB),
    )
    cur = _run(_stage(1, [100], name="a", input_bytes=10 * MB))
    findings = analyze_regressions(compare_runs(base, cur))
    assert all(f.detector != "regression.new_stage" for f in findings)
    # A removed stage must not produce ANY regression finding.
    assert findings == []


# --- confidence derivation ---------------------------------------------------


def test_confidence_high_on_corroboration():
    # Name match + two regressed signals (duration + shuffle) -> HIGH.
    base = _logical(
        1, duration_ms_total=100_000, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB
    )
    cur = _logical(
        1, duration_ms_total=200_000, input_bytes=1000 * MB, shuffle_read_bytes=2000 * MB
    )
    findings = analyze_regressions(_matched_comparison(base, cur, match_method="name"))
    assert len(findings) >= 2
    assert all(f.confidence is Confidence.HIGH for f in findings)


def test_confidence_medium_single_name_signal():
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB)
    cur = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=2000 * MB)
    f = _by_detector(analyze_regressions(_matched_comparison(base, cur, match_method="name")))[
        "regression.shuffle"
    ]
    assert f.confidence is Confidence.MEDIUM


def test_confidence_medium_name_plus_position():
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB)
    cur = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=2000 * MB)
    f = _by_detector(
        analyze_regressions(_matched_comparison(base, cur, match_method="name+position"))
    )["regression.shuffle"]
    assert f.confidence is Confidence.MEDIUM


def test_confidence_low_id_fallback():
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB)
    cur = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=2000 * MB)
    f = _by_detector(analyze_regressions(_matched_comparison(base, cur, match_method="id")))[
        "regression.shuffle"
    ]
    assert f.confidence is Confidence.LOW


def test_id_match_two_signals_not_upgraded():
    # Corroboration upgrade requires a NAME match; id match stays LOW.
    base = _logical(
        1, duration_ms_total=100_000, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB
    )
    cur = _logical(
        1, duration_ms_total=200_000, input_bytes=1000 * MB, shuffle_read_bytes=2000 * MB
    )
    findings = analyze_regressions(_matched_comparison(base, cur, match_method="id"))
    assert all(f.confidence is Confidence.LOW for f in findings)


# --- match-method caveat -----------------------------------------------------


def test_name_plus_position_caveat_in_evidence():
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB)
    cur = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=2000 * MB)
    f = _by_detector(
        analyze_regressions(_matched_comparison(base, cur, match_method="name+position"))
    )["regression.shuffle"]
    assert any("name+position" in e for e in f.evidence)


def test_id_match_caveat_in_evidence():
    base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=500 * MB)
    cur = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=2000 * MB)
    f = _by_detector(analyze_regressions(_matched_comparison(base, cur, match_method="id")))[
        "regression.shuffle"
    ]
    assert any("matched by id only" in e for e in f.evidence)


# --- severity bands ----------------------------------------------------------


def test_severity_bands_across_excess():
    # shuffle threshold 30%. Pick pct_change to land in each band.
    def shuffle_finding(pct: float) -> Finding:
        base = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=1000 * MB)
        cur_bytes = int(1000 * MB * (1 + pct / 100))
        cur = _logical(1, input_bytes=1000 * MB, shuffle_read_bytes=cur_bytes)
        return _by_detector(analyze_regressions(_matched_comparison(base, cur)))[
            "regression.shuffle"
        ]

    # excess = pct/30. LOW:[1,2)->pct in [30,60); MEDIUM:[2,4)->[60,120);
    # HIGH:[4,8)->[120,240); CRITICAL:>=8->>=240.
    assert shuffle_finding(45).severity is Severity.LOW
    assert shuffle_finding(90).severity is Severity.MEDIUM
    assert shuffle_finding(180).severity is Severity.HIGH
    assert shuffle_finding(300).severity is Severity.CRITICAL


# --- app-level roll-up -------------------------------------------------------


def test_app_level_duration_rollup_input_flat():
    # App duration +50% with app input flat -> one app-level duration finding.
    base = _run(_stage(1, [100, 100], name="s", input_bytes=1000 * MB))
    cur = _run(_stage(1, [150, 150], name="s", input_bytes=1000 * MB))
    findings = analyze_regressions(compare_runs(base, cur))
    app = [f for f in findings if f.detector == "regression.duration" and f.stage_id is None]
    assert len(app) == 1
    assert any("roll-up" in e for e in app[0].evidence)


def test_app_level_rollup_suppressed_when_input_grew():
    # App duration up but app input also up proportionally -> no app roll-up.
    base = _run(_stage(1, [100, 100], name="s", input_bytes=1000 * MB))
    cur = _run(_stage(1, [200, 200], name="s", input_bytes=2000 * MB))
    findings = analyze_regressions(compare_runs(base, cur))
    assert not [
        f for f in findings if f.detector == "regression.duration" and f.stage_id is None
    ]


# --- thresholds override -----------------------------------------------------


def test_threshold_override_flags_borderline():
    # +15% shuffle is below the default 30% threshold (suppressed), but a custom
    # 10% threshold flags it.
    base = _logical(1, input_bytes=10000 * MB, shuffle_read_bytes=1000 * MB)
    cur = _logical(1, input_bytes=10000 * MB, shuffle_read_bytes=1150 * MB)
    comparison = _matched_comparison(base, cur)
    assert analyze_regressions(comparison) == []  # default suppresses

    custom = RegressionThresholds(shuffle_regression_percent=10.0)
    by = _by_detector(analyze_regressions(comparison, custom))
    assert "regression.shuffle" in by


# --- metrics typing + determinism --------------------------------------------


def test_metrics_contains_only_floats():
    base = _logical(
        1,
        duration_ms_total=100_000,
        input_bytes=1000 * MB,
        shuffle_read_bytes=500 * MB,
        memory_spilled_bytes=0,
        skew_ratio=2.0,
        task_count=200,
    )
    cur = _logical(
        1,
        duration_ms_total=300_000,
        input_bytes=1000 * MB,
        shuffle_read_bytes=2000 * MB,
        memory_spilled_bytes=1000 * MB,
        skew_ratio=30.0,
        task_count=200,
    )
    findings = analyze_regressions(_matched_comparison(base, cur))
    assert findings  # several signals fired
    for f in findings:
        for key, value in f.metrics.items():
            assert isinstance(value, float), f"{f.detector}.{key} = {value!r}"


def test_deterministic_output():
    base = _logical(
        1,
        duration_ms_total=100_000,
        input_bytes=1000 * MB,
        shuffle_read_bytes=500 * MB,
        skew_ratio=2.0,
        task_count=200,
    )
    cur = _logical(
        1,
        duration_ms_total=300_000,
        input_bytes=1000 * MB,
        shuffle_read_bytes=2000 * MB,
        skew_ratio=30.0,
        task_count=200,
    )
    comparison = _matched_comparison(base, cur)
    first = [f.to_dict() for f in analyze_regressions(comparison)]
    second = [f.to_dict() for f in analyze_regressions(comparison)]
    assert first == second


# --- REGRESSION TESTS for the per-stage input-attribution bug ----------------
#
# Real Spark attributes input_bytes ONLY to scan/read stages; downstream
# shuffle/join/aggregate stages record input_bytes == 0. The old engine used a
# stage's OWN input delta as the proportionality denominator, so a pure shuffle
# stage (own input 0 -> 0) was judged "input flat" and its proportional growth
# was FALSELY flagged. The fix falls back to APPLICATION-level input for such
# stages. These tests model that real attribution (input on the scan stage,
# zero on the shuffle stage).


def test_proportional_scaling_across_stages_no_findings():
    # THE BUG. Scan stage input doubles (95MB -> 191MB); downstream join/shuffle
    # stage has own input 0 on BOTH sides, but its shuffle AND task-time double,
    # task counts unchanged. This is textbook proportional SCALING and must
    # produce ZERO findings. This test FAILS against the buggy code (which flags
    # a shuffle + duration regression on the join stage citing "input flat
    # (0.0B -> 0.0B)") and PASSES after the fix.
    base = _run(
        _stage(10, [1000, 1000], name="scan", input_bytes=95 * MB),
        _stage(
            20,
            [2000, 2000],  # join task-time sum 4s
            name="join",
            input_bytes=0,
            shuffle_read=1500 * MB,
        ),
    )
    cur = _run(
        _stage(10, [1000, 1000], name="scan", input_bytes=191 * MB),  # input doubled
        _stage(
            20,
            [4000, 4000],  # join task-time sum 8s (doubled)
            name="join",
            input_bytes=0,
            shuffle_read=3000 * MB,  # shuffle doubled
        ),
    )
    findings = analyze_regressions(compare_runs(base, cur))
    assert findings == []  # cleanly proportional job -> nothing flagged


def test_shuffle_outgrew_app_input_still_flagged():
    # Scan input +50% (95MB -> ~143MB); downstream shuffle stage (own input 0)
    # shuffle +300% -> judged against APPLICATION input, normalized factor > 1
    # -> a shuffle regression IS flagged, and the evidence references
    # application input (not the stage's own zero input).
    base = _run(
        _stage(10, [1000], name="scan", input_bytes=100 * MB),
        _stage(20, [1000], name="join", input_bytes=0, shuffle_read=500 * MB),
    )
    cur = _run(
        _stage(10, [1000], name="scan", input_bytes=150 * MB),  # +50%
        _stage(20, [1000], name="join", input_bytes=0, shuffle_read=2000 * MB),  # +300%
    )
    by = _by_detector(analyze_regressions(compare_runs(base, cur)))
    assert "regression.shuffle" in by
    f = by["regression.shuffle"]
    assert f.stage_id == 20  # the join stage, not the scan stage
    assert "normalized_factor" in f.metrics
    # shuffle growth 4.0 / app-input growth 1.5 = 2.67.
    assert abs(f.metrics["normalized_factor"] - (4.0 / 1.5)) < 1e-6
    assert any("application input" in e for e in f.evidence)
    assert not any("0.0B -> 0.0B" in e for e in f.evidence)


def test_genuine_regression_still_fires_with_app_input_flat():
    # App input flat (scan input unchanged); downstream stage develops 20x skew
    # + multi-GB spill. The skew and spill findings must still fire exactly as
    # before -- the fix must not over-suppress the genuine case.
    base = _run(
        _stage(10, [1000], name="scan", input_bytes=100 * MB),
        _stage(
            20,
            [100] * 199 + [100],  # balanced
            name="agg",
            input_bytes=0,
            shuffle_read=500 * MB,
            mem_spill=0,
        ),
    )
    cur = _run(
        _stage(10, [1000], name="scan", input_bytes=100 * MB),  # input unchanged
        _stage(
            20,
            [100] * 199 + [2000],  # one task 20x the median -> skew
            name="agg",
            input_bytes=0,
            shuffle_read=500 * MB,
            mem_spill=2000 * MB,  # multi-GB spill, newly appeared
        ),
    )
    by = _by_detector(analyze_regressions(compare_runs(base, cur)))
    assert "regression.skew" in by
    assert "regression.spill" in by


def test_app_fallback_evidence_wording_is_honest():
    # When the app-level fallback is used, the finding's evidence must NOT
    # contain the misleading "input approximately flat (0.0B -> 0.0B)" and MUST
    # reference application input. Here app input is flat but the shuffle stage
    # (own input 0) grows a genuine regression.
    base = _run(
        _stage(10, [1000], name="scan", input_bytes=1000 * MB),
        _stage(20, [1000], name="join", input_bytes=0, shuffle_read=500 * MB),
    )
    cur = _run(
        _stage(10, [1000], name="scan", input_bytes=1000 * MB),  # app input flat
        _stage(20, [1000], name="join", input_bytes=0, shuffle_read=2000 * MB),  # +300%
    )
    by = _by_detector(analyze_regressions(compare_runs(base, cur)))
    assert "regression.shuffle" in by
    f = by["regression.shuffle"]
    assert not any("0.0B -> 0.0B" in e for e in f.evidence)
    assert not any("input approximately flat (0.0B" in e for e in f.evidence)
    assert any("application input" in e for e in f.evidence)


def test_no_credible_input_context_not_flagged():
    # Edge case: a stage with no own input in a job that also records NO input
    # anywhere (tiny synthetic log). There is no credible proportionality
    # context, so byte/time growth must NOT be flagged (anti-false-positive).
    base = _run(_stage(1, [1000], name="only", input_bytes=0, shuffle_read=500 * MB))
    cur = _run(_stage(1, [1000], name="only", input_bytes=0, shuffle_read=2000 * MB))
    findings = analyze_regressions(compare_runs(base, cur))
    assert all(f.detector != "regression.shuffle" for f in findings)


def test_worst_first_ordering():
    # A CRITICAL stage and a LOW stage -> critical ranks first.
    base = _run(
        _stage(1, [100], name="big", input_bytes=1000 * MB, shuffle_read=1000 * MB),
        _stage(2, [100], name="small", input_bytes=1000 * MB, shuffle_read=1000 * MB),
    )
    cur = _run(
        _stage(1, [100], name="big", input_bytes=1000 * MB, shuffle_read=5000 * MB),
        _stage(2, [100], name="small", input_bytes=1000 * MB, shuffle_read=1400 * MB),
    )
    findings = analyze_regressions(compare_runs(base, cur))
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, reverse=True)
