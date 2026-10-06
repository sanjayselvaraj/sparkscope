"""Pure factual diff of two Spark runs — the comparison model.

Trust boundary
--------------
This module emits **facts only**: metric deltas between a baseline run and a
current run. It makes NO regression, severity, "bad", or "cause" judgments and
produces NO ``Finding`` objects — those belong to a later milestone. The words
"regression"/"bad" never appear here. Every stage match carries a
``match_method`` label stating *how* it was matched; a match is never a claim of
proven semantic equivalence.

Stage-matching strategy (and WHY)
---------------------------------
Spark assigns stage ids per-run via the DAG scheduler, so an upstream query
change renumbers downstream stages — id-equality across two separate runs would
produce meaningless matches. We therefore match by stage **name** (the Spark
"Stage Name", typically the RDD/operation call-site), the most stable semantic
anchor the log carries. Id is used only as a last-resort fallback when names are
blank. Concretely:

(a) Collapse stage attempts into one :class:`LogicalStage` per ``stage_id``
    (a retry is the same logical stage; comparing attempt-by-attempt would
    create spurious added/removed noise). Metrics are computed over the COMBINED
    task list across attempts (not averaged) so median/skew reflect the real
    retried work.
(b) Group each run's logical stages by name. For a name present on BOTH sides:
      * unique on each side  -> one match, ``match_method="name"``.
      * repeated on either side -> pair positionally in stage_id order
        (``zip`` shortest), ``match_method="name+position"``; leftover baseline
        entries become ``removed`` and leftover current entries become ``added``
        (``match_method="none"`` — they did not actually pair).
(c) Name only in baseline -> ``removed`` (``match_method="none"``);
    name only in current  -> ``added``  (``match_method="none"``).
(d) SPECIAL CASE: if EVERY logical stage on BOTH sides has a blank name there is
    no semantic anchor, so match by ``stage_id`` across runs with
    ``match_method="id"`` (explicitly weaker, and labeled as such). If only SOME
    names are blank we do NOT take this path; the blank-named ones simply group
    under name ``""`` and pair positionally like any other repeated name.
(e) Each matched pair gets metric deltas (fixed order, see ``_STAGE_METRICS``).
    Added/removed comparisons carry an empty delta list — there is nothing to
    diff against.
(f) ``stage_comparisons`` are ordered deterministically: group order
    ``matched``, then ``added``, then ``removed``; within a group sorted by
    ``(stage_key, stage_id)`` (baseline id for matched/removed, current id for
    added). The reporter does its own worst-first *display* ordering; this
    canonical order keeps ``to_dict()`` deterministic and judgment-free.

Other honesty notes
--------------------
* ``pct_change`` is ``None`` when ``baseline == 0`` — a percent change from zero
  is mathematically undefined, and forcing a number (0.0 or inf) would fabricate
  a fact. The reporter renders ``None`` as ``n/a``.
* ``duration_ms_total`` is a **sum of task durations**, not wall-clock stage
  elapsed time: the execution model carries no stage start/end timestamps, so a
  task-time sum is the honest proxy available. The app-level duration delta is
  likewise a sum of these per-stage task-time sums.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Literal

from sparkscope.parser.models import SparkRun, Task

# Metric name -> unit, in the fixed order deltas are emitted for a matched pair.
_STAGE_METRICS: list[tuple[str, str]] = [
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


@dataclass(frozen=True)
class MetricDelta:
    """A single metric compared between the baseline and current run.

    Pure fact: a before value, an after value, and the derived change. No
    judgment about whether the change is good or bad.
    """

    name: str
    baseline: float
    current: float
    unit: Literal["ms", "bytes", "count", "ratio"]

    @property
    def abs_change(self) -> float:
        return self.current - self.baseline

    @property
    def pct_change(self) -> float | None:
        """Percentage change, or ``None`` when the baseline is zero.

        A change from a zero baseline is real but has no percentage (dividing by
        zero is undefined); returning ``None`` avoids fabricating a meaningless
        or infinite number. We guard the exact ``== 0`` case here rather than
        reusing ``util.ratio`` (which returns 0.0 for a non-positive
        denominator) precisely because 0.0 would hide the undefined case.
        """
        if self.baseline == 0:
            return None
        return (self.current - self.baseline) / self.baseline * 100.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "baseline": self.baseline,
            "current": self.current,
            "abs_change": self.abs_change,
            "pct_change": self.pct_change,
        }


@dataclass(frozen=True)
class LogicalStage:
    """One stage with all its attempts collapsed into a single logical unit."""

    stage_id: int
    name: str
    attempt_count: int
    duration_ms_total: float
    task_count: int
    median_task_ms: float
    max_task_ms: int
    skew_ratio: float
    shuffle_read_bytes: int
    shuffle_write_bytes: int
    memory_spilled_bytes: int
    disk_spilled_bytes: int
    input_bytes: int


def _skew_ratio(durations: list[int], med: float) -> float:
    """Max/median skew with the same guard as ``Stage.skew_ratio``.

    0.0 when there are fewer than two tasks or the median is non-positive.
    """
    if med <= 0 or len(durations) < 2:
        return 0.0
    return max(durations) / med


def build_logical_stages(run: SparkRun) -> list[LogicalStage]:
    """Collapse every ``Stage`` attempt sharing a ``stage_id`` into one
    :class:`LogicalStage`, computing stats over the COMBINED task list.

    Returned deterministically ordered by ``stage_id``.
    """
    # stage_list() is already sorted by (stage_id, attempt_id), so iterating it
    # groups attempts of the same stage contiguously and deterministically.
    by_id: dict[int, list] = {}
    for stage in run.stage_list():
        by_id.setdefault(stage.stage_id, []).append(stage)

    logical: list[LogicalStage] = []
    for stage_id in sorted(by_id):
        attempts = by_id[stage_id]
        combined_tasks: list[Task] = []
        for s in attempts:
            combined_tasks.extend(s.tasks)

        durations = [t.metrics.duration_ms for t in combined_tasks]
        med = median(durations) if durations else 0.0
        # First non-blank name across attempts; "" when none carry a name.
        name = next((s.name for s in attempts if s.name), "")

        logical.append(
            LogicalStage(
                stage_id=stage_id,
                name=name,
                attempt_count=len({s.attempt_id for s in attempts}),
                duration_ms_total=float(sum(durations)),
                task_count=len(combined_tasks),
                median_task_ms=med,
                max_task_ms=max(durations) if durations else 0,
                skew_ratio=_skew_ratio(durations, med),
                shuffle_read_bytes=sum(t.metrics.shuffle_read_bytes for t in combined_tasks),
                shuffle_write_bytes=sum(t.metrics.shuffle_write_bytes for t in combined_tasks),
                memory_spilled_bytes=sum(t.metrics.memory_spilled_bytes for t in combined_tasks),
                disk_spilled_bytes=sum(t.metrics.disk_spilled_bytes for t in combined_tasks),
                input_bytes=sum(t.metrics.input_bytes for t in combined_tasks),
            )
        )
    return logical


def _stage_key(stage: LogicalStage) -> str:
    """A human-readable label: the stage name when present, else ``stage <id>``.

    This is a display/ordering label only, NOT a semantic-equivalence claim.
    """
    return stage.name if stage.name else f"stage {stage.stage_id}"


@dataclass
class StageComparison:
    """How one logical stage compares between the two runs.

    ``status`` is ``matched`` (present on both sides, with ``metric_deltas``),
    ``added`` (only in current), or ``removed`` (only in baseline). ``added`` and
    ``removed`` carry an empty ``metric_deltas`` list.
    """

    stage_key: str
    status: Literal["matched", "added", "removed"]
    match_method: Literal["name", "name+position", "id", "none"]
    baseline: LogicalStage | None
    current: LogicalStage | None
    metric_deltas: list[MetricDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_key": self.stage_key,
            "status": self.status,
            "match_method": self.match_method,
            "baseline_stage_id": self.baseline.stage_id if self.baseline else None,
            "current_stage_id": self.current.stage_id if self.current else None,
            "metric_deltas": [d.to_dict() for d in self.metric_deltas],
        }


@dataclass
class RunComparison:
    """The full factual diff of a baseline run against a current run."""

    baseline_app: str
    current_app: str
    app_deltas: list[MetricDelta]
    stage_comparisons: list[StageComparison]
    baseline_skipped_lines: int
    current_skipped_lines: int

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_app": self.baseline_app,
            "current_app": self.current_app,
            "baseline_skipped_lines": self.baseline_skipped_lines,
            "current_skipped_lines": self.current_skipped_lines,
            "app_deltas": [d.to_dict() for d in self.app_deltas],
            "stage_comparisons": [c.to_dict() for c in self.stage_comparisons],
        }


def _stage_metric_deltas(base: LogicalStage, cur: LogicalStage) -> list[MetricDelta]:
    """Build the fixed-order metric deltas for a matched stage pair."""
    deltas: list[MetricDelta] = []
    for name, unit in _STAGE_METRICS:
        deltas.append(
            MetricDelta(
                name=name,
                baseline=float(getattr(base, name)),
                current=float(getattr(cur, name)),
                unit=unit,  # type: ignore[arg-type]
            )
        )
    return deltas


def _matched(
    base: LogicalStage,
    cur: LogicalStage,
    match_method: Literal["name", "name+position", "id"],
) -> StageComparison:
    return StageComparison(
        stage_key=_stage_key(cur),
        status="matched",
        match_method=match_method,
        baseline=base,
        current=cur,
        metric_deltas=_stage_metric_deltas(base, cur),
    )


def _removed(base: LogicalStage) -> StageComparison:
    return StageComparison(
        stage_key=_stage_key(base),
        status="removed",
        match_method="none",
        baseline=base,
        current=None,
    )


def _added(cur: LogicalStage) -> StageComparison:
    return StageComparison(
        stage_key=_stage_key(cur),
        status="added",
        match_method="none",
        baseline=None,
        current=cur,
    )


def _match_by_id(
    base_stages: list[LogicalStage], cur_stages: list[LogicalStage]
) -> list[StageComparison]:
    """Special case: no names anywhere, so id is the only (weak) anchor."""
    base_by_id = {s.stage_id: s for s in base_stages}
    cur_by_id = {s.stage_id: s for s in cur_stages}
    comparisons: list[StageComparison] = []
    for stage_id in base_by_id.keys() | cur_by_id.keys():
        b = base_by_id.get(stage_id)
        c = cur_by_id.get(stage_id)
        if b is not None and c is not None:
            comparisons.append(_matched(b, c, "id"))
        elif b is not None:
            comparisons.append(_removed(b))
        else:
            assert c is not None
            comparisons.append(_added(c))
    return comparisons


def _match_by_name(
    base_stages: list[LogicalStage], cur_stages: list[LogicalStage]
) -> list[StageComparison]:
    """Name-first matching (steps b and c of the module docstring)."""
    base_by_name: dict[str, list[LogicalStage]] = {}
    cur_by_name: dict[str, list[LogicalStage]] = {}
    for s in base_stages:  # input is already stage_id-ordered
        base_by_name.setdefault(s.name, []).append(s)
    for s in cur_stages:
        cur_by_name.setdefault(s.name, []).append(s)

    comparisons: list[StageComparison] = []
    for name in base_by_name.keys() | cur_by_name.keys():
        b_list = base_by_name.get(name, [])
        c_list = cur_by_name.get(name, [])

        if b_list and c_list:
            if len(b_list) == 1 and len(c_list) == 1:
                comparisons.append(_matched(b_list[0], c_list[0], "name"))
            else:
                # Repeated name: pair positionally in stage_id order.
                # strict=False is intentional: unequal counts are expected and
                # the leftovers are handled below as added/removed.
                for b, c in zip(b_list, c_list, strict=False):
                    comparisons.append(_matched(b, c, "name+position"))
                # Unpaired leftovers did not actually pair -> added/removed.
                for b in b_list[len(c_list):]:
                    comparisons.append(_removed(b))
                for c in c_list[len(b_list):]:
                    comparisons.append(_added(c))
        elif b_list:
            comparisons.extend(_removed(b) for b in b_list)
        else:
            comparisons.extend(_added(c) for c in c_list)
    return comparisons


_STATUS_ORDER = {"matched": 0, "added": 1, "removed": 2}


def _sort_key(c: StageComparison) -> tuple[int, str, int]:
    """Deterministic canonical order: matched, added, removed; then
    (stage_key, stage_id) — baseline id for matched/removed, current id for added.
    """
    stage = c.baseline if c.baseline is not None else c.current
    stage_id = stage.stage_id if stage is not None else 0
    return (_STATUS_ORDER[c.status], c.stage_key, stage_id)


def _app_deltas(
    baseline: SparkRun,
    current: SparkRun,
    base_stages: list[LogicalStage],
    cur_stages: list[LogicalStage],
) -> list[MetricDelta]:
    """Application-level deltas computed from the two runs.

    ``duration_ms_total`` is the sum of per-stage task-time sums (a proxy for app
    runtime; the model lacks app wall-clock). ``output_bytes`` is summed from
    per-task metrics because ``Stage`` exposes no output aggregate.
    """

    def _sum(stages: list[LogicalStage], attr: str) -> float:
        return float(sum(getattr(s, attr) for s in stages))

    def _output_bytes(run: SparkRun) -> int:
        return sum(t.metrics.output_bytes for s in run.stage_list() for t in s.tasks)

    specs: list[tuple[str, str, float, float]] = [
        (
            "duration_ms_total",
            "ms",
            _sum(base_stages, "duration_ms_total"),
            _sum(cur_stages, "duration_ms_total"),
        ),
        ("job_count", "count", float(len(baseline.jobs)), float(len(current.jobs))),
        ("stage_count", "count", float(len(base_stages)), float(len(cur_stages))),
        (
            "input_bytes",
            "bytes",
            _sum(base_stages, "input_bytes"),
            _sum(cur_stages, "input_bytes"),
        ),
        (
            "output_bytes",
            "bytes",
            float(_output_bytes(baseline)),
            float(_output_bytes(current)),
        ),
        (
            "shuffle_read_bytes",
            "bytes",
            _sum(base_stages, "shuffle_read_bytes"),
            _sum(cur_stages, "shuffle_read_bytes"),
        ),
        (
            "shuffle_write_bytes",
            "bytes",
            _sum(base_stages, "shuffle_write_bytes"),
            _sum(cur_stages, "shuffle_write_bytes"),
        ),
    ]
    return [
        MetricDelta(name=name, baseline=b, current=c, unit=unit)  # type: ignore[arg-type]
        for name, unit, b, c in specs
    ]


def compare_runs(baseline: SparkRun, current: SparkRun) -> RunComparison:
    """Compare two runs and return a pure factual :class:`RunComparison`."""
    base_stages = build_logical_stages(baseline)
    cur_stages = build_logical_stages(current)

    # (d) all-blank names on BOTH sides -> id is the only anchor.
    base_all_blank = bool(base_stages) and all(not s.name for s in base_stages)
    cur_all_blank = bool(cur_stages) and all(not s.name for s in cur_stages)
    if base_all_blank and cur_all_blank:
        stage_comparisons = _match_by_id(base_stages, cur_stages)
    else:
        stage_comparisons = _match_by_name(base_stages, cur_stages)

    stage_comparisons.sort(key=_sort_key)

    return RunComparison(
        baseline_app=baseline.app_name,
        current_app=current.app_name,
        app_deltas=_app_deltas(baseline, current, base_stages, cur_stages),
        stage_comparisons=stage_comparisons,
        baseline_skipped_lines=baseline.skipped_lines,
        current_skipped_lines=current.skipped_lines,
    )
