"""Streaming parser: Spark event-log JSON -> :class:`SparkRun`.

A Spark event log is one JSON object per line. Real logs are large (hundreds of
MB to multiple GB), so we parse **line by line** and keep only the structured
model in memory -- never the whole file. That streaming discipline is a
deliberate design choice: memory stays flat regardless of log size.

We handle only the events the v0.1 model needs and silently skip everything
else. Spark emits dozens of event types across versions; ignoring unknown ones
(rather than failing) is what makes the parser robust across Spark 3.x releases.

Events consumed
---------------
* ``SparkListenerApplicationStart`` -> app name / id
* ``SparkListenerJobStart``         -> job -> stage mapping
* ``SparkListenerStageSubmitted``   -> stage name / planned task count
* ``SparkListenerTaskEnd``          -> per-task metrics (the detector fuel)
* ``SparkListenerStageCompleted``   -> final stage task count
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sparkscope.parser.models import Job, SparkRun, Stage, Task, TaskMetrics


class EventLogParseError(Exception):
    """Raised when the input does not look like a Spark event log at all."""


@dataclass
class _LineStats:
    """Mutable counter shared with :func:`_iter_json_lines`.

    Using a small object (rather than a return value) lets the generator report
    how many lines it skipped without breaking its lazy, one-line-at-a-time
    contract -- the caller reads ``.skipped`` after the stream is exhausted.
    """

    skipped: int = 0


def _iter_json_lines(lines: Iterable[str], stats: _LineStats) -> Iterator[dict]:
    """Yield parsed JSON objects, counting lines that fail to decode.

    A malformed line is skipped rather than fatal: real logs can contain a
    truncated final line if the application was killed mid-write, and we would
    rather analyze the 99% that parsed than refuse the whole file. Each skipped
    line increments ``stats.skipped`` so the caller can warn the user -- silent
    data loss and a genuinely healthy log must not look identical.

    Blank lines are not counted as skips: they are expected padding, not corruption.
    """
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            stats.skipped += 1
            continue
        if isinstance(obj, dict):
            yield obj
        else:
            # Valid JSON but not an object (e.g. a bare array/number) is corrupt
            # for our purposes -- count it rather than drop it silently.
            stats.skipped += 1


def _task_metrics_from_event(event: dict) -> TaskMetrics:
    """Pull the fields we care about out of a TaskEnd event's metrics block."""
    info = event.get("Task Info", {})
    metrics = event.get("Task Metrics", {}) or {}

    shuffle_read = metrics.get("Shuffle Read Metrics", {}) or {}
    shuffle_write = metrics.get("Shuffle Write Metrics", {}) or {}
    input_metrics = metrics.get("Input Metrics", {}) or {}
    output_metrics = metrics.get("Output Metrics", {}) or {}

    # Shuffle read bytes = local + remote, matching how Spark reports total read.
    read_bytes = int(shuffle_read.get("Local Bytes Read", 0)) + int(
        shuffle_read.get("Remote Bytes Read", 0)
    )

    # Duration semantics (important -- these are NOT the same measure):
    #
    # * "Executor Run Time" is the CPU/compute time the task actually spent
    #   running on the executor. It EXCLUDES scheduler delay, task
    #   deserialization, result serialization, and GC-adjacent waits. This is
    #   the right signal for *skew* -- we want to compare how much real work
    #   each task did, not how long it sat in a queue.
    #
    # * "Finish Time - Launch Time" is wall-clock elapsed time for the attempt.
    #   It INCLUDES scheduling/serialization overhead, so two tasks doing equal
    #   work can differ here purely due to cluster contention.
    #
    # We therefore prefer "Executor Run Time" and only fall back to the
    # wall-clock delta when run time is missing (malformed/partial task). The
    # fallback is best-effort: a stage mixing the two measures could compare
    # slightly apples-to-oranges, but in practice every real completed task
    # carries "Executor Run Time", so the fallback fires only for broken tasks.
    wall_clock = int(info.get("Finish Time", 0)) - int(info.get("Launch Time", 0))
    if wall_clock < 0:
        wall_clock = 0

    return TaskMetrics(
        duration_ms=int(metrics.get("Executor Run Time", wall_clock) or wall_clock),
        shuffle_read_bytes=read_bytes,
        shuffle_write_bytes=int(shuffle_write.get("Shuffle Bytes Written", 0)),
        memory_spilled_bytes=int(metrics.get("Memory Bytes Spilled", 0)),
        disk_spilled_bytes=int(metrics.get("Disk Bytes Spilled", 0)),
        input_bytes=int(input_metrics.get("Bytes Read", 0)),
        output_bytes=int(output_metrics.get("Bytes Written", 0)),
    )


def parse_events(events: Iterable[dict]) -> SparkRun:
    """Fold a stream of event dicts into a :class:`SparkRun`.

    Kept separate from file IO so it can be unit-tested with in-memory events.
    """
    run = SparkRun()
    saw_any_spark_event = False

    for event in events:
        etype = event.get("Event")
        if not etype:
            continue
        saw_any_spark_event = True

        if etype == "SparkListenerApplicationStart":
            run.app_name = event.get("App Name", "")
            run.app_id = event.get("App ID", "") or ""

        elif etype == "SparkListenerJobStart":
            job_id = int(event.get("Job ID", -1))
            stage_ids = [
                int(s.get("Stage ID"))
                for s in event.get("Stage Infos", [])
                if "Stage ID" in s
            ]
            if not stage_ids:
                stage_ids = [int(s) for s in event.get("Stage IDs", [])]
            run.jobs[job_id] = Job(job_id=job_id, stage_ids=stage_ids)

        elif etype == "SparkListenerStageSubmitted":
            info = event.get("Stage Info", {})
            _ensure_stage(run, info)

        elif etype == "SparkListenerStageCompleted":
            info = event.get("Stage Info", {})
            stage = _ensure_stage(run, info)
            if "Number of Tasks" in info:
                stage.num_tasks = int(info["Number of Tasks"])

        elif etype == "SparkListenerTaskEnd":
            stage_id = int(event.get("Stage ID", -1))
            attempt = int(event.get("Stage Attempt ID", 0))
            key = (stage_id, attempt)
            stage = run.stages.get(key)
            if stage is None:
                stage = Stage(stage_id=stage_id, attempt_id=attempt)
                run.stages[key] = stage
            info = event.get("Task Info", {})
            task = Task(
                task_id=int(info.get("Task ID", len(stage.tasks))),
                stage_id=stage_id,
                stage_attempt_id=attempt,
                metrics=_task_metrics_from_event(event),
                failed=bool(info.get("Failed", False)),
            )
            stage.tasks.append(task)

    if not saw_any_spark_event:
        raise EventLogParseError(
            "No Spark events found. Is this a Spark event log "
            "(one JSON object per line with an 'Event' field)?"
        )
    return run


def _ensure_stage(run: SparkRun, stage_info: dict) -> Stage:
    """Fetch-or-create a Stage from a Stage Info block."""
    stage_id = int(stage_info.get("Stage ID", -1))
    attempt = int(stage_info.get("Stage Attempt ID", 0))
    key = (stage_id, attempt)
    stage = run.stages.get(key)
    if stage is None:
        stage = Stage(stage_id=stage_id, attempt_id=attempt)
        run.stages[key] = stage
    if stage_info.get("Stage Name"):
        stage.name = stage_info["Stage Name"]
    if "Number of Tasks" in stage_info:
        stage.num_tasks = int(stage_info["Number of Tasks"])
    return stage


def parse_file(path: str | Path) -> SparkRun:
    """Parse a Spark event-log file into a :class:`SparkRun`, streaming line by line.

    Records the number of undecodable lines on ``run.skipped_lines`` so callers
    can warn the user when a log was partially corrupt.
    """
    path = Path(path)
    stats = _LineStats()
    with path.open("r", encoding="utf-8") as fh:
        run = parse_events(_iter_json_lines(fh, stats))
    run.skipped_lines = stats.skipped
    return run
