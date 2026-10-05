"""Parser + model tests.

Two layers:
* unit tests that fold hand-written event dicts (fast, pin exact field mapping);
* an end-to-end test over the committed fixture (proves file parsing + the
  skew signal the detectors will later rely on).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparkscope.parser.event_log import (
    EventLogParseError,
    parse_events,
    parse_file,
)

FIXTURE = Path(__file__).parent / "fixtures" / "skewed_app.log"


def _task_end(stage_id: int, task_id: int, run_ms: int, **metrics) -> dict:
    return {
        "Event": "SparkListenerTaskEnd",
        "Stage ID": stage_id,
        "Stage Attempt ID": 0,
        "Task Info": {"Task ID": task_id, "Launch Time": 0, "Finish Time": run_ms},
        "Task Metrics": {"Executor Run Time": run_ms, **metrics},
    }


def test_app_metadata_is_captured():
    run = parse_events(
        [
            {
                "Event": "SparkListenerApplicationStart",
                "App Name": "demo",
                "App ID": "app-1",
            }
        ]
    )
    assert run.app_name == "demo"
    assert run.app_id == "app-1"


def test_tasks_group_into_stage_with_metrics():
    run = parse_events(
        [
            _task_end(0, 0, 100, **{"Shuffle Write Metrics": {"Shuffle Bytes Written": 10}}),
            _task_end(0, 1, 300, **{"Memory Bytes Spilled": 5, "Disk Bytes Spilled": 7}),
        ]
    )
    stage = run.stages[(0, 0)]
    assert len(stage.tasks) == 2
    assert stage.max_task_ms == 300
    assert stage.median_task_ms == 200  # (100 + 300) / 2
    assert stage.total_shuffle_write_bytes == 10
    assert stage.total_spill_bytes == 12


def test_skew_ratio_math():
    # One hot task (1000ms) against three fast ones (100ms) => clear skew.
    run = parse_events(
        [
            _task_end(1, 0, 100),
            _task_end(1, 1, 100),
            _task_end(1, 2, 100),
            _task_end(1, 3, 1000),
        ]
    )
    stage = run.stages[(1, 0)]
    assert stage.median_task_ms == 100
    assert stage.max_task_ms == 1000
    assert stage.skew_ratio == pytest.approx(10.0)


def test_skew_ratio_is_zero_for_single_task():
    run = parse_events([_task_end(2, 0, 500)])
    assert run.stages[(2, 0)].skew_ratio == 0.0


def test_non_spark_dicts_are_ignored_not_fatal():
    # A dict without an "Event" field is ignored; a real event still parses.
    task_event = {
        "Event": "SparkListenerTaskEnd",
        "Stage ID": 0,
        "Stage Attempt ID": 0,
        "Task Info": {"Task ID": 0},
        "Task Metrics": {},
    }
    run = parse_events([{"not": "a spark event"}, task_event])
    assert (0, 0) in run.stages


def test_empty_or_non_spark_input_raises():
    with pytest.raises(EventLogParseError):
        parse_events([{"foo": "bar"}])


def test_fixture_end_to_end():
    run = parse_file(FIXTURE)
    assert run.app_name == "sparkscope-skew-fixture"
    assert len(run.stages) == 2

    read_stage = run.stages[(0, 0)]
    shuffle_stage = run.stages[(1, 0)]

    # Stage 0 is balanced; stage 1 has one hot task.
    assert read_stage.skew_ratio < 1.5
    assert shuffle_stage.skew_ratio > 10  # task 7 is ~90x the median

    # The hot task's spill shows up in the stage total (200MB).
    assert shuffle_stage.total_spill_bytes == 200_000_000
