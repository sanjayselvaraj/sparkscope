"""Tests for the run-comparison model (pure factual diff).

Mirrors the detector test style: build ``Stage``/``SparkRun`` directly with
small local helpers, no fixtures. Every assertion is about FACTS (metric
deltas, match methods, statuses). The comparison model must never emit
regression/severity/cause fields — several tests assert their absence.
"""

from __future__ import annotations

import io
import json

from rich.console import Console

from sparkscope.analysis.compare import (
    LogicalStage,
    MetricDelta,
    build_logical_stages,
    compare_runs,
)
from sparkscope.parser.models import Job, SparkRun, Stage, Task, TaskMetrics
from sparkscope.report.comparison import render_comparison_json, render_comparison_terminal


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
    """Build a stage with one Task per duration.

    Byte metrics are placed on the FIRST task (keeps per-task bookkeeping simple;
    the comparison only ever sums them across the stage, so distribution within
    the stage is irrelevant to these tests).
    """
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


def _delta(comparison, name: str) -> MetricDelta:
    """Fetch a single named delta from a matched StageComparison."""
    for d in comparison.metric_deltas:
        if d.name == name:
            return d
    raise AssertionError(f"no delta named {name!r}")


# --- identical runs ----------------------------------------------------------


def test_identical_runs_all_matched_zero_change():
    base = _run(_stage(1, [100, 200], name="read", shuffle_read=1000))
    cur = _run(_stage(1, [100, 200], name="read", shuffle_read=1000))
    c = compare_runs(base, cur)

    assert len(c.stage_comparisons) == 1
    sc = c.stage_comparisons[0]
    assert sc.status == "matched"
    assert sc.match_method == "name"

    # A nonzero metric: abs_change 0, pct_change exactly 0.0.
    sr = _delta(sc, "shuffle_read_bytes")
    assert sr.abs_change == 0
    assert sr.pct_change == 0.0

    # A zero-baseline metric: pct_change is None (undefined), abs_change 0.
    spill = _delta(sc, "disk_spilled_bytes")
    assert spill.baseline == 0
    assert spill.abs_change == 0
    assert spill.pct_change is None


# --- single-metric increases -------------------------------------------------


def test_duration_increase():
    base = _run(_stage(1, [100, 100], name="s"))
    cur = _run(_stage(1, [300, 300], name="s"))
    sc = compare_runs(base, cur).stage_comparisons[0]
    d = _delta(sc, "duration_ms_total")
    assert d.baseline == 200.0
    assert d.current == 600.0
    assert d.abs_change == 400.0
    assert d.pct_change == 200.0


def test_shuffle_increase():
    base = _run(_stage(1, [100], name="s", shuffle_read=1000, shuffle_write=500))
    cur = _run(_stage(1, [100], name="s", shuffle_read=3000, shuffle_write=2000))
    sc = compare_runs(base, cur).stage_comparisons[0]
    assert _delta(sc, "shuffle_read_bytes").abs_change == 2000
    assert _delta(sc, "shuffle_read_bytes").pct_change == 200.0
    assert _delta(sc, "shuffle_write_bytes").abs_change == 1500
    assert _delta(sc, "shuffle_write_bytes").pct_change == 300.0


def test_spill_increase():
    base = _run(_stage(1, [100], name="s", mem_spill=100, disk_spill=200))
    cur = _run(_stage(1, [100], name="s", mem_spill=400, disk_spill=1000))
    sc = compare_runs(base, cur).stage_comparisons[0]
    assert _delta(sc, "memory_spilled_bytes").abs_change == 300
    assert _delta(sc, "disk_spilled_bytes").abs_change == 800
    assert _delta(sc, "disk_spilled_bytes").pct_change == 400.0


def test_skew_ratio_increase():
    # Balanced baseline (skew ~1.0) vs skewed current (max/median large).
    base = _run(_stage(1, [100, 100, 100, 100], name="s"))
    cur = _run(_stage(1, [100, 100, 100, 1000], name="s"))
    sc = compare_runs(base, cur).stage_comparisons[0]
    d = _delta(sc, "skew_ratio")
    assert d.baseline == 1.0
    assert d.current == 10.0
    assert d.abs_change == 9.0


def test_input_proportional_to_shuffle_no_judgment():
    base = _run(_stage(1, [100], name="s", input_bytes=1000, shuffle_write=1000))
    cur = _run(_stage(1, [100], name="s", input_bytes=2000, shuffle_write=2000))
    sc = compare_runs(base, cur).stage_comparisons[0]
    # Both deltas present and positive; recorded as plain facts.
    assert _delta(sc, "input_bytes").abs_change == 1000
    assert _delta(sc, "shuffle_write_bytes").abs_change == 1000

    # No cause/severity/regression labeling anywhere in the serialized form.
    blob = json.dumps(compare_runs(base, cur).to_dict()).lower()
    for banned in ("regression", "severity", "likely_cause", "recommendation", "bad"):
        assert banned not in blob


# --- added / removed ---------------------------------------------------------


def test_added_stage():
    base = _run(_stage(1, [100], name="a"))
    cur = _run(_stage(1, [100], name="a"), _stage(2, [100], name="b"))
    c = compare_runs(base, cur)
    added = [sc for sc in c.stage_comparisons if sc.status == "added"]
    assert len(added) == 1
    assert added[0].stage_key == "b"
    assert added[0].match_method == "none"
    assert added[0].metric_deltas == []


def test_removed_stage():
    base = _run(_stage(1, [100], name="a"), _stage(2, [100], name="b"))
    cur = _run(_stage(1, [100], name="a"))
    c = compare_runs(base, cur)
    removed = [sc for sc in c.stage_comparisons if sc.status == "removed"]
    assert len(removed) == 1
    assert removed[0].stage_key == "b"
    assert removed[0].match_method == "none"
    assert removed[0].metric_deltas == []


def test_changed_task_count():
    base = _run(_stage(1, [100, 100], name="s"))
    cur = _run(_stage(1, [100, 100, 100, 100], name="s"))
    sc = compare_runs(base, cur).stage_comparisons[0]
    d = _delta(sc, "task_count")
    assert d.baseline == 2.0
    assert d.current == 4.0
    assert d.abs_change == 2.0


# --- stage retry collapse ----------------------------------------------------


def test_stage_retry_collapsed():
    # Baseline: two attempts of stage 1, same name. Current: single attempt.
    attempt0 = _stage(1, [100, 100], name="s", attempt=0, shuffle_read=500)
    attempt1 = _stage(1, [200, 200], name="s", attempt=1, shuffle_read=500)
    base = _run(attempt0, attempt1)
    cur = _run(_stage(1, [100], name="s"))

    logical = build_logical_stages(base)
    assert len(logical) == 1
    ls = logical[0]
    assert ls.attempt_count == 2
    # Aggregated over BOTH attempts: 4 tasks, durations 100+100+200+200=600.
    assert ls.task_count == 4
    assert ls.duration_ms_total == 600.0
    assert ls.shuffle_read_bytes == 1000  # 500 + 500

    sc = compare_runs(base, cur).stage_comparisons[0]
    assert sc.status == "matched"
    assert sc.match_method == "name"
    assert sc.baseline is not None
    assert sc.baseline.attempt_count == 2


# --- duplicate names ---------------------------------------------------------


def test_duplicate_names_name_plus_position():
    # Two baseline stages share name "dup"; current has one.
    base = _run(_stage(1, [100], name="dup"), _stage(2, [100], name="dup"))
    cur = _run(_stage(5, [100], name="dup"))
    c = compare_runs(base, cur)

    matched = [sc for sc in c.stage_comparisons if sc.status == "matched"]
    removed = [sc for sc in c.stage_comparisons if sc.status == "removed"]
    assert len(matched) == 1
    assert matched[0].match_method == "name+position"
    # Paired the first (stage_id order) baseline with the single current.
    assert matched[0].baseline is not None
    assert matched[0].baseline.stage_id == 1
    # The extra same-named baseline stage becomes removed.
    assert len(removed) == 1
    assert removed[0].baseline is not None
    assert removed[0].baseline.stage_id == 2


# --- id fallback -------------------------------------------------------------


def test_id_fallback_all_blank_names():
    base = _run(_stage(1, [100]), _stage(2, [100]))
    cur = _run(_stage(1, [100]), _stage(2, [100]))
    c = compare_runs(base, cur)
    assert len(c.stage_comparisons) == 2
    for sc in c.stage_comparisons:
        assert sc.status == "matched"
        assert sc.match_method == "id"


def test_id_fallback_added_removed():
    base = _run(_stage(1, [100]), _stage(2, [100]))
    cur = _run(_stage(1, [100]), _stage(3, [100]))
    c = compare_runs(base, cur)
    statuses = {sc.status for sc in c.stage_comparisons}
    assert statuses == {"matched", "added", "removed"}


# --- zero baseline -----------------------------------------------------------


def test_zero_baseline_pct_none_no_zerodivision():
    base = _run(_stage(1, [100], name="s", disk_spill=0))
    cur = _run(_stage(1, [100], name="s", disk_spill=5000))
    sc = compare_runs(base, cur).stage_comparisons[0]
    d = _delta(sc, "disk_spilled_bytes")
    assert d.baseline == 0
    assert d.abs_change == 5000
    assert d.pct_change is None  # undefined, not inf / not a crash


# --- app-level deltas --------------------------------------------------------


def test_app_level_deltas():
    base = _run(_stage(1, [100], name="s", input_bytes=1000), jobs=1)
    cur = _run(
        _stage(1, [100], name="s", input_bytes=1000),
        _stage(2, [100], name="t", input_bytes=500),
        jobs=3,
    )
    c = compare_runs(base, cur)
    by_name = {d.name: d for d in c.app_deltas}
    assert by_name["job_count"].baseline == 1.0
    assert by_name["job_count"].current == 3.0
    assert by_name["stage_count"].baseline == 1.0
    assert by_name["stage_count"].current == 2.0
    assert by_name["input_bytes"].abs_change == 500


# --- determinism -------------------------------------------------------------


def test_determinism():
    def build():
        return _run(
            _stage(1, [100, 200], name="read", shuffle_read=1000),
            _stage(2, [300], name="write", disk_spill=50),
        )

    c1 = compare_runs(build(), build())
    c2 = compare_runs(build(), build())
    assert c1.to_dict() == c2.to_dict()
    # And the canonical form is JSON-serializable.
    assert json.loads(json.dumps(c1.to_dict())) == c1.to_dict()


# --- reporter smoke ----------------------------------------------------------


def test_render_json_parses():
    base = _run(_stage(1, [100], name="s", shuffle_read=1000))
    cur = _run(_stage(1, [100], name="s", shuffle_read=3000))
    c = compare_runs(base, cur)
    parsed = json.loads(render_comparison_json(c))
    assert set(parsed) == {
        "baseline_app",
        "current_app",
        "baseline_skipped_lines",
        "current_skipped_lines",
        "app_deltas",
        "stage_comparisons",
    }


def test_render_terminal_runs():
    base = _run(_stage(1, [100], name="s", shuffle_read=1000), _stage(9, [100], name="old"))
    cur = _run(_stage(1, [100], name="s", shuffle_read=3000), _stage(8, [100], name="new"))
    c = compare_runs(base, cur)
    console = Console(file=io.StringIO(), width=100)
    render_comparison_terminal(console, c)
    out = console.file.getvalue()
    assert "Application" in out
    assert "regression" not in out.lower()
    assert "LogicalStage" in str(LogicalStage.__name__)  # sanity import use
