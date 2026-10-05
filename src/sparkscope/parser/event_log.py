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
from pathlib import Path

from sparkscope.parser.models import Job, SparkRun, Stage, Task, TaskMetrics


class EventLogParseError(Exception):
    """Raised when the input does not look like a Spark event log at all."""


def _iter_json_lines(lines: Iterable[str]) -> Iterator[dict]:
    """Yield parsed JSON objects, skipping blank lines.

    A malformed line is skipped rather than fatal: real logs can contain a
    truncated final line if the application was killed mid-write, and we would
    rather analyze the 99% that parsed than refuse the whole file.
    """
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


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

    duration = int(info.get("Finish Time", 0)) - int(info.get("Launch Time", 0))
    if duration < 0:
        duration = 0

    return TaskMetrics(
        duration_ms=int(metrics.get("Executor Run Time", duration) or duration),
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
    """Parse a Spark event-log file into a :class:`SparkRun`, streaming line by line."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return parse_events(_iter_json_lines(fh))
