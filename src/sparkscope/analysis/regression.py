"""The regression JUDGMENT engine.

This module is the judgment layer that sits on top of the pure factual
comparison model (:mod:`sparkscope.analysis.compare`). The comparison model
emits *facts* -- metric deltas between a baseline run and a current run -- and
makes no claim about whether a delta is good or bad. This module decides which
of those factual deltas are real **regressions in context**, and emits ranked,
confidence-scored, explainable :class:`~sparkscope.analysis.finding.Finding`
objects.

Pipeline (per candidate signal)
--------------------------------
observed delta -> context normalization -> significance gate ->
severity + confidence -> :class:`Finding`.

Central principle (anti-false-positive)
---------------------------------------
A difference is NOT automatically a regression. Growth in proportion to input
is *scaling*, not a regression: if a stage reads twice the data and shuffles
twice the bytes, that is expected and must be suppressed. This is the
make-or-break rule and lives in branch (i) of :func:`_eval_metric_signal`.

Trust rule
----------
Every finding separates observed FACT (``evidence``) from HYPOTHESIS
(``likely_cause``, always phrased "consistent with ..."/"likely ...") from a
concrete ``recommendation``. A non-``name`` stage match (``id`` /
``name+position``) carries its caveat in the evidence/title so a guessed stage
identity never reads as confirmed.

Thresholds
----------
All numbers in :class:`RegressionThresholds` are **DEFAULTS**, not authoritative
production values. A caller may override any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from sparkscope.analysis.compare import (
    LogicalStage,
    MetricDelta,
    RunComparison,
    StageComparison,
)
from sparkscope.analysis.finding import Confidence, Finding, Severity
from sparkscope.analysis.util import format_bytes, format_ms


@dataclass(frozen=True)
class RegressionThresholds:
    """Tunable gates for the regression engine.

    Every field is a **DEFAULT** chosen to be reasonable for a first pass, NOT
    an authoritative production value. Callers are expected to override these
    for their own workloads; the engine never treats them as ground truth.
    """

    #: DEFAULT. A stage's task-time sum must grow by at least this percent
    #: (input held flat) to count as a runtime regression.
    runtime_regression_percent: float = 20.0
    #: DEFAULT. Shuffle bytes must grow by at least this percent to be a candidate.
    shuffle_regression_percent: float = 30.0
    #: DEFAULT. Spill bytes must grow by at least this percent to be a candidate.
    spill_regression_percent: float = 50.0
    #: DEFAULT. Skew only matters if the CURRENT skew ratio is at least this high.
    skew_min_current_ratio: float = 2.0
    #: DEFAULT. ...and the skew ratio climbed at least this much (percent).
    skew_min_increase_percent: float = 50.0
    #: DEFAULT. If a metric grew within +/- this band of input growth, treat the
    #: growth as proportional (scaling, benign) and suppress it.
    input_proportional_tolerance_percent: float = 25.0
    #: DEFAULT. Ignore stages whose current task-time sum is below this (ms); a
    #: huge percent jump on a trivially short stage costs nobody anything.
    min_duration_ms_total: float = 5000.0
    #: DEFAULT. Absolute floor (bytes) for shuffle/spill deltas: 256 MB. A large
    #: percent jump on a few KB is noise.
    min_bytes: float = 256.0 * 1024 * 1024

    @classmethod
    def default(cls) -> RegressionThresholds:
        """Return the DEFAULT thresholds."""
        return cls()


# ---------------------------------------------------------------------------
# Internal helpers: delta lookup + context normalization.
# ---------------------------------------------------------------------------


def _delta(sc: StageComparison, name: str) -> MetricDelta | None:
    """Return the named :class:`MetricDelta` for a stage, or None if absent."""
    for d in sc.metric_deltas:
        if d.name == name:
            return d
    return None


def _growth(delta: MetricDelta) -> float | None:
    """Growth factor ``current / baseline``, or None when baseline is 0.

    We guard the exact zero-baseline case and return None (NOT 0.0 via
    ``util.ratio``) because the growth factor is genuinely *undefined* at a zero
    baseline, and 0.0 would silently hide that undefined case.
    """
    if delta.baseline > 0:
        return delta.current / delta.baseline
    return None


def _input_flat(input_delta: MetricDelta | None, tolerance_percent: float) -> bool:
    """True when input is roughly flat between the two runs.

    Flat means the input ``pct_change`` is defined and its magnitude is within
    the tolerance band. When the input baseline is 0 (pct_change None) and
    current input is > 0, input GREW from zero -> NOT flat. When both baseline
    and current input are 0, there is no input signal at all -> treat as flat
    (nothing scaled).
    """
    if input_delta is None:
        return True
    if input_delta.pct_change is None:
        # Zero baseline: flat only if current is also zero (no input anywhere).
        return input_delta.current == 0
    return abs(input_delta.pct_change) <= tolerance_percent


def _normalized_factor(
    metric_delta: MetricDelta, input_delta: MetricDelta | None
) -> float | None:
    """``metric_growth / input_growth`` when BOTH are defined, else None.

    A value near 1.0 means the metric grew in proportion to input (scaling); a
    value well above 1.0 means the metric outgrew its input. Both divisions are
    guarded via :func:`_growth` (returns None at a zero baseline).
    """
    metric_growth = _growth(metric_delta)
    if metric_growth is None or input_delta is None:
        return None
    input_growth = _growth(input_delta)
    if input_growth is None or input_growth == 0:
        return None
    return metric_growth / input_growth


# ---------------------------------------------------------------------------
# Severity banding.
# ---------------------------------------------------------------------------

#: Severity band boundaries keyed on an "excess" ratio (how far a context-
#: adjusted change exceeds its gate). Shared by all signal types so magnitude
#: maps consistently to severity.
def _severity_for_excess(excess: float) -> Severity | None:
    """Map an excess ratio to a severity band, or None when below the gate.

    * ``excess < 1.0``        -> below gate, not emitted.
    * ``1.0 <= excess < 2.0`` -> LOW (marginally over threshold).
    * ``2.0 <= excess < 4.0`` -> MEDIUM.
    * ``4.0 <= excess < 8.0`` -> HIGH.
    * ``excess >= 8.0``       -> CRITICAL.
    """
    if excess < 1.0:
        return None
    if excess < 2.0:
        return Severity.LOW
    if excess < 4.0:
        return Severity.MEDIUM
    if excess < 8.0:
        return Severity.HIGH
    return Severity.CRITICAL


def _match_caveat(match_method: str) -> str | None:
    """Return a caveat sentence for a non-``name`` match, else None.

    A guessed stage identity must never read as confirmed, so a weaker match
    method carries an explicit caveat in the finding's evidence.
    """
    if match_method == "name":
        return None
    if match_method == "name+position":
        return (
            "stage matched by name+position (a same-named stage appeared more "
            "than once and was paired by order) -- stage identity is inferred, "
            "not confirmed"
        )
    if match_method == "id":
        return (
            "stage matched by id only (no stage names were available) -- stage "
            "identity across runs is a weak guess, not confirmed"
        )
    return (
        f"stage matched by {match_method!r} -- stage identity is inferred, not "
        "confirmed"
    )


# ---------------------------------------------------------------------------
# Per-signal evaluators.
# ---------------------------------------------------------------------------

# Which concrete deltas back each signal, and the unit of their values.
_SHUFFLE_METRICS = ("shuffle_read_bytes", "shuffle_write_bytes")
_SPILL_METRICS = ("memory_spilled_bytes", "disk_spilled_bytes")


def _fmt_value(value: float, unit: str) -> str:
    """Format a metric value by unit, reusing the shared formatters."""
    if unit == "bytes":
        return format_bytes(value)
    if unit == "ms":
        return format_ms(value)
    return f"{value:.0f}"


def _metric_floor(detector: str, thresholds: RegressionThresholds) -> float:
    """Absolute floor for a signal: bytes for shuffle/spill, ms for duration."""
    if detector == "regression.duration":
        return thresholds.min_duration_ms_total
    return thresholds.min_bytes


def _metric_percent(detector: str, thresholds: RegressionThresholds) -> float:
    """Percent threshold for a signal."""
    if detector == "regression.duration":
        return thresholds.runtime_regression_percent
    if detector == "regression.shuffle":
        return thresholds.shuffle_regression_percent
    return thresholds.spill_regression_percent  # spill


def _signal_label(detector: str) -> str:
    """Human label for a signal type, used in titles/evidence."""
    return {
        "regression.duration": "task-time",
        "regression.shuffle": "shuffle",
        "regression.spill": "spill",
    }[detector]


def _eval_metric_signal(
    sc: StageComparison,
    detector: str,
    metric_delta: MetricDelta,
    input_delta: MetricDelta | None,
    thresholds: RegressionThresholds,
    *,
    input_is_app_fallback: bool = False,
    has_input_context: bool = True,
) -> Finding | None:
    """Evaluate one byte/time signal on a matched stage through the context
    branches, and return a :class:`Finding` or None (suppressed / sub-gate).

    ``input_delta`` is the EFFECTIVE input context for this stage, chosen by the
    caller: the stage's OWN ``input_bytes`` delta when the stage records input
    of its own, or the APPLICATION-LEVEL input delta when the stage records no
    input (a pure shuffle/compute stage whose own ``input_bytes`` is zero on
    both sides). ``input_is_app_fallback`` is True in the latter case and only
    affects the honest evidence wording -- the proportionality maths are
    identical either way. ``has_input_context`` is False only when there is no
    credible input context at all (neither the stage nor the application records
    any input); in that case byte/time GROWTH is NOT flagged (anti-false-positive
    conservatism), though a new-from-zero cost still fires via the zero-baseline
    branch.

    Context branches:

    (i)  input GREW and the metric grew IN PROPORTION (within +/- tolerance of
         input growth) -> NOT a regression; suppress. This is the make-or-break
         anti-false-positive rule: scaling is not regressing.
    (ii) input is roughly FLAT and the metric cleared BOTH its percent threshold
         and its absolute floor -> candidate; evidence states "input
         approximately flat (X -> Y)".
    (iii) input grew but the metric grew MUCH faster (normalized factor well
         above 1) and cleared its absolute floor -> candidate; evidence states
         both deltas and the normalized factor.

    ZERO-BASELINE: when the metric baseline is 0 (``pct_change`` is None, e.g.
    spill newly appeared) there is no growth factor to compute. Treat it as "a
    new cost appeared": flag only if current clears the absolute floor, CAP
    confidence (handled by the caller via a flag we encode on the finding's
    confidence), and state in evidence that the baseline was zero.
    """
    floor = _metric_floor(detector, thresholds)
    percent = _metric_percent(detector, thresholds)
    tol = thresholds.input_proportional_tolerance_percent
    unit = metric_delta.unit
    label = _signal_label(detector)
    caveat = _match_caveat(sc.match_method)

    base_s = _fmt_value(metric_delta.baseline, unit)
    cur_s = _fmt_value(metric_delta.current, unit)

    # --- ZERO-BASELINE: a new cost appeared from nothing. -------------------
    if metric_delta.pct_change is None:
        # No growth factor possible. Only the absolute floor can gate it.
        if metric_delta.current < floor:
            return None
        # Excess measures how far above the floor the new cost is.
        excess = metric_delta.current / floor if floor > 0 else 1.0
        severity = _severity_for_excess(excess)
        if severity is None:
            return None
        evidence = [
            f"{label} appeared in current with no baseline cost: "
            f"{base_s} -> {cur_s} (baseline was zero)",
        ]
        if caveat is not None:
            evidence.append(caveat)
        metrics: dict[str, float] = {
            "baseline": metric_delta.baseline,
            "current": metric_delta.current,
            "abs_change": metric_delta.abs_change,
            "floor": floor,
            "excess": excess,
        }
        return Finding(
            severity=severity,
            sort_index=excess,
            category="regression",
            detector=detector,
            title=(
                f"New {label} cost in stage {sc.stage_key} "
                "(not present in baseline)"
            ),
            stage_id=_stage_id(sc),
            stage_attempt_id=None,
            # Zero-baseline findings have NO ratio to corroborate: cap at LOW.
            confidence=Confidence.LOW,
            evidence=evidence,
            metrics=metrics,
            likely_cause=(
                f"A {label} cost appeared in the current run that the baseline "
                "did not have. With no baseline value there is no growth ratio to "
                "corroborate, so this is reported with low confidence; it is "
                "consistent with a plan change that introduced a new "
                f"{label} step."
            ),
            recommendation=(
                "Compare the query plans of the two runs to find the newly "
                f"introduced {label} step, and confirm it is intended."
            ),
        )

    # From here on pct_change is defined (baseline > 0).
    pct = metric_delta.pct_change
    norm = _normalized_factor(metric_delta, input_delta)
    input_grew = (
        input_delta is not None
        and input_delta.pct_change is not None
        and input_delta.pct_change > tol
    )

    # --- Branch (i): proportional growth -> SUPPRESS. -----------------------
    # Only applies when input grew AND we can compute a normalized factor. If
    # the metric stayed within +/- tolerance of input growth, it is scaling.
    if input_grew and norm is not None:
        lower = 1.0 - tol / 100.0
        upper = 1.0 + tol / 100.0
        if lower <= norm <= upper:
            return None

    # --- Build the candidate per the remaining branches. --------------------
    input_base_s = _fmt_value(input_delta.baseline, "bytes") if input_delta else "n/a"
    input_cur_s = _fmt_value(input_delta.current, "bytes") if input_delta else "n/a"
    # When the input context is the application-level fallback (this stage
    # records no own input), the evidence must say so truthfully: it is the
    # stage's own input that was zero, NOT the job's input, so we never print
    # "input approximately flat" for a job whose input actually grew.
    input_pct = input_delta.pct_change if input_delta is not None else None
    if input_is_app_fallback:
        input_ctx = (
            "stage records no own input; judged against application input "
            f"({input_base_s} -> {input_cur_s}"
            + (f", {input_pct:+.0f}%" if input_pct is not None else "")
            + ")"
        )
    else:
        input_ctx = None

    evidence = []
    excess = 0.0
    if input_grew and norm is not None:
        # --- Branch (iii): input grew but metric outgrew it. ----------------
        if norm < 1.0 + tol / 100.0:
            # Metric did not outgrow input by more than tolerance: not a
            # regression even though input grew (and not proportional-suppressed
            # because it is below input growth -> benign).
            return None
        if metric_delta.current < floor:
            return None
        excess = norm
        if input_is_app_fallback:
            evidence.append(
                f"{input_ctx}; {label} grew faster than application input: "
                f"{label} {base_s} -> {cur_s} ({pct:+.0f}%); "
                f"input-normalized factor {norm:.2f}x (1.0x would be proportional)"
            )
        else:
            evidence.append(
                f"{label} grew faster than input: {label} {base_s} -> {cur_s} "
                f"({pct:+.0f}%) while input {input_base_s} -> {input_cur_s}; "
                f"input-normalized factor {norm:.2f}x (1.0x would be proportional)"
            )
    else:
        # --- Branch (ii): input flat, metric over threshold + floor. --------
        if not has_input_context:
            # No credible input context (stage records no own input AND the
            # application records no input either). We cannot tell scaling from
            # a regression, so -- honouring the anti-false-positive principle --
            # we do NOT flag byte/time growth here. (A new-from-zero cost is
            # still caught by the zero-baseline branch above.)
            return None
        if not _input_flat(input_delta, tol):
            # input grew but we could not normalize (e.g. metric or input zero
            # baseline handled elsewhere) -> be conservative, do not flag.
            return None
        if pct < percent:
            return None
        if metric_delta.current < floor:
            return None
        excess = pct / percent if percent > 0 else 1.0
        if input_is_app_fallback:
            # App input is genuinely flat here (the fallback context itself is
            # flat); state it against application input, never claiming the
            # stage's own input was the flat thing.
            evidence.append(
                f"{input_ctx}, approximately flat; "
                f"{label} {base_s} -> {cur_s} ({pct:+.0f}%)"
            )
        else:
            evidence.append(
                f"input approximately flat ({input_base_s} -> {input_cur_s}); "
                f"{label} {base_s} -> {cur_s} ({pct:+.0f}%)"
            )

    severity = _severity_for_excess(excess)
    if severity is None:
        return None
    if caveat is not None:
        evidence.append(caveat)

    metrics = {
        "baseline": metric_delta.baseline,
        "current": metric_delta.current,
        "abs_change": metric_delta.abs_change,
        "pct_change": pct,
        "floor": floor,
        "excess": excess,
    }
    if input_delta is not None:
        metrics["input_baseline"] = input_delta.baseline
        metrics["input_current"] = input_delta.current
        input_growth = _growth(input_delta)
        if input_growth is not None:
            metrics["input_growth"] = input_growth
    if norm is not None:
        metrics["normalized_factor"] = norm

    return Finding(
        severity=severity,
        sort_index=excess,
        category="regression",
        detector=detector,
        title=f"{label.capitalize()} regression in stage {sc.stage_key}",
        stage_id=_stage_id(sc),
        stage_attempt_id=None,
        confidence=_base_confidence(sc.match_method),
        evidence=evidence,
        metrics=metrics,
        likely_cause=(
            f"The stage's {label} cost rose beyond what input growth explains, "
            "which is consistent with a plan or data-distribution change rather "
            "than simple scaling."
        ),
        recommendation=(
            "Diff the query plans and partitioning between the two runs for this "
            f"stage to find what drove the extra {label} cost."
        ),
    )


def _eval_skew(sc: StageComparison, thresholds: RegressionThresholds) -> Finding | None:
    """Evaluate the skew signal for a matched stage.

    ``skew_ratio`` is self-normalizing (max/median task time), so it is NOT
    divided by input growth. Flag when the CURRENT skew ratio is at least
    ``skew_min_current_ratio`` AND it increased by at least
    ``skew_min_increase_percent``. Zero-baseline skew (baseline ratio 0, so the
    percent increase is undefined) satisfies the "increased" leg by the
    current-ratio floor alone, and is reported with capped confidence.

    Context FACT: whether ``task_count`` stayed roughly stable (a skew spike with
    a stable partition count is the classic hot-key signature) or changed (which
    could instead be a repartition). This is stated as fact; the cause is a
    hypothesis.
    """
    skew = _delta(sc, "skew_ratio")
    if skew is None:
        return None
    if skew.current < thresholds.skew_min_current_ratio:
        return None

    zero_baseline = skew.pct_change is None  # baseline skew was 0.0
    if not zero_baseline:
        assert skew.pct_change is not None
        if skew.pct_change < thresholds.skew_min_increase_percent:
            return None

    excess = (
        skew.current / thresholds.skew_min_current_ratio
        if thresholds.skew_min_current_ratio > 0
        else 1.0
    )
    severity = _severity_for_excess(excess)
    if severity is None:
        return None

    tc = _delta(sc, "task_count")
    tc_stable = True
    tc_note = "task count unknown"
    if tc is not None:
        tol = thresholds.input_proportional_tolerance_percent
        if tc.pct_change is None:
            tc_stable = tc.current == tc.baseline
        else:
            tc_stable = abs(tc.pct_change) <= tol
        if tc_stable:
            tc_note = (
                f"task count stable ({tc.baseline:.0f} -> {tc.current:.0f}) -- "
                "the classic hot-key signature"
            )
        else:
            tc_note = (
                f"task count changed ({tc.baseline:.0f} -> {tc.current:.0f}) -- "
                "could instead reflect a repartition"
            )

    caveat = _match_caveat(sc.match_method)
    if zero_baseline:
        skew_fact = (
            f"skew ratio {skew.baseline:.1f}x -> {skew.current:.1f}x "
            "(baseline had no measurable skew)"
        )
    else:
        skew_fact = (
            f"skew ratio {skew.baseline:.1f}x -> {skew.current:.1f}x "
            f"({skew.pct_change:+.0f}%)"
        )
    evidence = [skew_fact, tc_note]
    if caveat is not None:
        evidence.append(caveat)

    metrics: dict[str, float] = {
        "baseline": skew.baseline,
        "current": skew.current,
        "abs_change": skew.abs_change,
        "excess": excess,
    }
    if skew.pct_change is not None:
        metrics["pct_change"] = skew.pct_change
    if tc is not None:
        metrics["task_count_baseline"] = tc.baseline
        metrics["task_count_current"] = tc.current

    # Confidence: a stable-task-count hot-key signature is a clean single
    # signal (name match -> MEDIUM). Zero-baseline skew has nothing to
    # corroborate the increase -> capped LOW.
    confidence = Confidence.LOW if zero_baseline else _base_confidence(sc.match_method)

    cause = (
        "A single task running far longer than the median, with the partition "
        "count unchanged, is consistent with a hot key: one shuffle/join key "
        "likely holds a disproportionate share of rows."
        if tc_stable
        else "The task-time imbalance grew, but the partition count also changed, "
        "so this is consistent with either an emerging hot key or a repartition; "
        "confirm before treating it as skew."
    )

    return Finding(
        severity=severity,
        sort_index=excess,
        category="regression",
        detector="regression.skew",
        title=f"Skew regression in stage {sc.stage_key}",
        stage_id=_stage_id(sc),
        stage_attempt_id=None,
        confidence=confidence,
        evidence=evidence,
        metrics=metrics,
        likely_cause=cause,
        recommendation=(
            "If a hot key is confirmed, enable AQE skew-join handling "
            "(spark.sql.adaptive.skewJoin.enabled=true) or salt the hot key to "
            "spread it across partitions."
        ),
    )


def _eval_added(sc: StageComparison, thresholds: RegressionThresholds) -> Finding | None:
    """Emit ``regression.new_stage`` for an expensive ADDED stage.

    A stage present only in the current run (``status == "added"``) is flagged
    when it clears either the duration floor or a bytes floor. A REMOVED stage
    is handled by :func:`analyze_regressions` returning nothing: work that went
    away is not a regression (documented choice for v0.1).

    Added stages carry ``match_method == "none"`` (they did not pair), so there
    is no baseline to corroborate against -> confidence LOW.
    """
    cur = sc.current
    if cur is None:
        return None

    # Cost excesses against each applicable floor; take the largest.
    excesses: list[tuple[str, float, str, str]] = []
    if thresholds.min_duration_ms_total > 0:
        excesses.append(
            (
                "task-time",
                cur.duration_ms_total / thresholds.min_duration_ms_total,
                format_ms(cur.duration_ms_total),
                "ms",
            )
        )
    bytes_cost = max(cur.input_bytes, cur.shuffle_read_bytes, cur.shuffle_write_bytes)
    if thresholds.min_bytes > 0:
        excesses.append(
            (
                "data",
                bytes_cost / thresholds.min_bytes,
                format_bytes(bytes_cost),
                "bytes",
            )
        )

    kind, excess, cost_s, _unit = max(excesses, key=lambda e: e[1])
    severity = _severity_for_excess(excess)
    if severity is None:
        return None

    metrics: dict[str, float] = {
        "current_duration_ms_total": cur.duration_ms_total,
        "current_input_bytes": float(cur.input_bytes),
        "current_shuffle_read_bytes": float(cur.shuffle_read_bytes),
        "current_shuffle_write_bytes": float(cur.shuffle_write_bytes),
        "excess": excess,
    }

    return Finding(
        severity=severity,
        sort_index=excess,
        category="regression",
        detector="regression.new_stage",
        title=f"New stage appeared: {sc.stage_key}",
        stage_id=cur.stage_id,
        stage_attempt_id=None,
        confidence=Confidence.LOW,
        evidence=[
            "a new stage appeared that did not exist in the baseline "
            f"({sc.stage_key}), carrying {kind} cost {cost_s}",
            "stage is new, so there is no baseline to corroborate against",
        ],
        metrics=metrics,
        likely_cause=(
            "A stage with no baseline counterpart is consistent with a query-plan "
            "change (e.g. an added join, aggregation, or repartition) rather than "
            "a change in the data alone."
        ),
        recommendation=(
            "Diff the query plans of the two runs to find the operation that "
            "introduced this stage, and confirm it is intended."
        ),
    )


def _eval_app_duration(
    comparison: RunComparison, thresholds: RegressionThresholds
) -> Finding | None:
    """Optional app-level duration roll-up.

    Emits ONE ``regression.duration`` finding with ``stage_id=None`` only when
    the app-level ``duration_ms_total`` regressed beyond
    ``runtime_regression_percent`` AND app ``input_bytes`` is flat. This is a
    roll-up for a quick top-line read, NOT a sum of the per-stage findings;
    per-stage findings remain primary. It is intentionally gated on flat input
    so it never double-counts a regression that is really just input scaling.
    """
    by_name = {d.name: d for d in comparison.app_deltas}
    dur = by_name.get("duration_ms_total")
    if dur is None or dur.pct_change is None:
        return None
    if dur.pct_change < thresholds.runtime_regression_percent:
        return None
    if not _input_flat(by_name.get("input_bytes"), thresholds.input_proportional_tolerance_percent):
        return None

    excess = dur.pct_change / thresholds.runtime_regression_percent
    severity = _severity_for_excess(excess)
    if severity is None:
        return None

    input_delta = by_name.get("input_bytes")
    input_base_s = _fmt_value(input_delta.baseline, "bytes") if input_delta else "n/a"
    input_cur_s = _fmt_value(input_delta.current, "bytes") if input_delta else "n/a"

    return Finding(
        severity=severity,
        sort_index=excess,
        category="regression",
        detector="regression.duration",
        title="Overall runtime regression (app-level roll-up)",
        stage_id=None,
        stage_attempt_id=None,
        confidence=Confidence.MEDIUM,
        evidence=[
            f"total task-time {format_ms(dur.baseline)} -> {format_ms(dur.current)} "
            f"({dur.pct_change:+.0f}%)",
            f"input approximately flat ({input_base_s} -> {input_cur_s})",
            "app-level roll-up (a top-line read, not a sum of per-stage findings)",
        ],
        metrics={
            "baseline": dur.baseline,
            "current": dur.current,
            "abs_change": dur.abs_change,
            "pct_change": dur.pct_change,
            "excess": excess,
        },
        likely_cause=(
            "Overall task-time grew with input held roughly flat, which is "
            "consistent with a regression somewhere in the plan rather than more "
            "data to process. See the per-stage findings for where."
        ),
        recommendation=(
            "Review the per-stage regression findings below to localize the "
            "runtime increase."
        ),
    )


# ---------------------------------------------------------------------------
# Small shared accessors.
# ---------------------------------------------------------------------------


def _stage_id(sc: StageComparison) -> int | None:
    """Current stage id for a matched stage (what the current run shows)."""
    stage: LogicalStage | None = sc.current if sc.current is not None else sc.baseline
    return stage.stage_id if stage is not None else None


def _base_confidence(match_method: str) -> Confidence:
    """Confidence from match quality alone (before corroboration upgrade).

    * ``name``          -> MEDIUM (a single clean name-matched signal).
    * ``name+position`` -> MEDIUM (positional pairing of a repeated name).
    * ``id`` / other    -> LOW (weak identity).
    """
    if match_method in ("name", "name+position"):
        return Confidence.MEDIUM
    return Confidence.LOW


def _stage_has_own_input(input_delta: MetricDelta | None) -> bool:
    """True when a stage records a MEANINGFUL input of its own.

    Spark attributes ``input_bytes`` only to scan/read stages; downstream
    shuffle/join/aggregate stages read no input and so their own
    ``input_bytes`` delta is ``0 -> 0``. A stage "has own input" when either
    its baseline or its current input is non-zero -- i.e. it is a scan-type
    stage that should be judged against its own input.
    """
    if input_delta is None:
        return False
    return input_delta.baseline != 0 or input_delta.current != 0


def _effective_input_context(
    own_input: MetricDelta | None,
    app_input: MetricDelta | None,
) -> tuple[MetricDelta | None, bool, bool]:
    """Pick the input delta a stage's proportionality should be judged against.

    Returns ``(context_delta, is_app_fallback, has_credible_context)``:

    * If the stage records meaningful OWN input (scan-type stage), use that
      delta as the context (``is_app_fallback == False``) -- today's behaviour.
      Credible context: yes.
    * Otherwise (a pure shuffle/compute stage whose own input is ``0 -> 0``),
      FALL BACK to the application-level input delta so the stage's growth is
      measured against the data volume that actually flowed into the job. This
      is the fix: a join stage whose shuffle doubled is judged against app input
      that doubled, yielding a normalized factor ~1.0 (proportional, suppressed)
      rather than being falsely flagged as "input flat". Credible context: yes.
    * When neither the stage nor the application records any input (both
      ``0 -> 0`` or absent -- e.g. a tiny synthetic log), there is NO credible
      input context at all. We honour the anti-false-positive principle and
      report ``has_credible_context == False`` so byte/time GROWTH signals are
      NOT flagged: there is no basis on which to call proportional-looking
      growth a regression. (A genuinely NEW cost from a zero baseline is still
      flagged by the zero-baseline branch, which needs no input context.)
    """
    if _stage_has_own_input(own_input):
        return own_input, False, True
    if _stage_has_own_input(app_input):
        return app_input, True, True
    # No own input and no app input signal: no credible context.
    return app_input, True, False


def _worse_metric(sc: StageComparison, names: tuple[str, ...]) -> MetricDelta | None:
    """Return the delta with the larger current value among ``names``.

    Shuffle read/write collapse into ONE shuffle signal and memory/disk spill
    into ONE spill signal; we judge the worse (larger current) of the pair so a
    stage never produces two near-duplicate findings for the same concern.
    """
    candidates = [d for n in names if (d := _delta(sc, n)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.current)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def analyze_regressions(
    comparison: RunComparison, thresholds: RegressionThresholds | None = None
) -> list[Finding]:
    """Judge a factual :class:`RunComparison` and return ranked regressions.

    Per-stage findings are primary; an optional app-level duration roll-up is
    appended. Output is deterministic: sorted worst-first by
    ``(severity, sort_index)`` with a stable string tiebreak so equal findings
    never reorder between runs.
    """
    thresholds = thresholds or RegressionThresholds.default()
    findings: list[Finding] = []

    # Application-level input delta, read ONCE from the factual comparison. A
    # downstream stage that records no input of its own is judged against this
    # (the data volume that flowed into the job), not against its own ~0 input.
    app_input_delta = next(
        (d for d in comparison.app_deltas if d.name == "input_bytes"), None
    )

    for sc in comparison.stage_comparisons:
        if sc.status == "removed":
            # Work that disappeared is not a regression (v0.1 choice).
            continue
        if sc.status == "added":
            added = _eval_added(sc, thresholds)
            if added is not None:
                findings.append(added)
            continue

        # Matched stage: evaluate each signal type. Choose the EFFECTIVE input
        # context: the stage's own input for scan-type stages, or the app-level
        # input for pure shuffle/compute stages that record no input of their
        # own (own input 0 -> 0).
        own_input = _delta(sc, "input_bytes")
        input_delta, app_fallback, has_context = _effective_input_context(
            own_input, app_input_delta
        )
        stage_candidates: list[Finding] = []

        dur = _delta(sc, "duration_ms_total")
        if dur is not None:
            f = _eval_metric_signal(
                sc,
                "regression.duration",
                dur,
                input_delta,
                thresholds,
                input_is_app_fallback=app_fallback,
                has_input_context=has_context,
            )
            if f is not None:
                stage_candidates.append(f)

        shuffle = _worse_metric(sc, _SHUFFLE_METRICS)
        if shuffle is not None:
            f = _eval_metric_signal(
                sc,
                "regression.shuffle",
                shuffle,
                input_delta,
                thresholds,
                input_is_app_fallback=app_fallback,
                has_input_context=has_context,
            )
            if f is not None:
                stage_candidates.append(f)

        spill = _worse_metric(sc, _SPILL_METRICS)
        if spill is not None:
            f = _eval_metric_signal(
                sc,
                "regression.spill",
                spill,
                input_delta,
                thresholds,
                input_is_app_fallback=app_fallback,
                has_input_context=has_context,
            )
            if f is not None:
                stage_candidates.append(f)

        skew = _eval_skew(sc, thresholds)
        if skew is not None:
            stage_candidates.append(skew)

        # Corroboration: >=2 independent regressed signals in the SAME stage, on
        # a clean name match, with each eligible to be corroborated (not a
        # capped zero-baseline-only / new-cost signal) -> upgrade to HIGH.
        if sc.match_method == "name":
            corroborating = [
                f for f in stage_candidates if f.confidence is not Confidence.LOW
            ]
            if len(corroborating) >= 2:
                for f in corroborating:
                    f.confidence = Confidence.HIGH

        findings.extend(stage_candidates)

    app_finding = _eval_app_duration(comparison, thresholds)
    if app_finding is not None:
        findings.append(app_finding)

    # Deterministic worst-first order with a stable tiebreak so equal
    # (severity, sort_index) never reorder across runs.
    findings.sort(
        key=lambda f: (f.severity, f.sort_index, _tiebreak(f)),
        reverse=True,
    )
    return findings


def _tiebreak(f: Finding) -> str:
    """Stable secondary key for findings with equal (severity, sort_index).

    Reverse-sorted alongside the numeric keys, so this orders equal findings by
    a fixed string; the exact direction does not matter, only that it is stable.
    """
    stage = f.stage_label if f.stage_id is not None else "~"
    return f"{f.detector}|{stage}|{f.title}"
