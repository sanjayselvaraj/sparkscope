"""Join diagnosis.

Honesty first: a *default* Spark event log does NOT reliably tell us the join
strategy (broadcast vs sort-merge vs shuffle-hash). That information lives in the
SQL physical plan carried by ``SparkListenerSQLExecutionStart`` events, which
v0.1 does not parse. So this detector does NOT claim to know whether a join was a
sort-merge or could have been a broadcast.

What it CAN do, defensibly, is flag a *pattern consistent with an expensive
shuffle-based join*: a stage that both reads a large shuffle AND shows a highly
imbalanced task-duration distribution. That combination frequently accompanies a
sort-merge join on a skewed key -- but we present it strictly as "consistent
with," paired with the exact observed evidence, and we recommend the engineer
confirm the join type in the SQL tab of the Spark UI.

This keeps the detector trustworthy: observed shuffle + imbalance (fact),
"consistent with an expensive join" (clearly-labelled inference), "confirm in the
SQL plan" (recommendation). We would rather under-claim than mislead.

If/when v0.1.x parses SQL plan events, this detector can be upgraded to name the
actual join strategy. That is tracked as a roadmap item, not faked here.
"""

from __future__ import annotations

from sparkscope.analysis.base import Detector, register
from sparkscope.analysis.finding import Confidence, Finding, Severity
from sparkscope.analysis.util import format_bytes
from sparkscope.parser.models import SparkRun, Stage

#: A stage must read at least this much shuffle to be considered join-heavy.
MIN_SHUFFLE_READ_BYTES = 1 * 1024**3  # 1 GB
#: Task-duration imbalance (max/median) above which the join looks expensive.
IMBALANCE_RATIO = 3.0


@register
class JoinDetector(Detector):
    """Flags stages whose shuffle + imbalance is consistent with an expensive join."""

    name = "join"
    category = "join"

    def analyze(self, run: SparkRun) -> list[Finding]:
        findings: list[Finding] = []
        for stage in run.stage_list():
            finding = self._check_stage(stage)
            if finding is not None:
                findings.append(finding)
        return findings

    def _check_stage(self, stage: Stage) -> Finding | None:
        shuffle_read = stage.total_shuffle_read_bytes
        if shuffle_read < MIN_SHUFFLE_READ_BYTES:
            return None

        imbalance = stage.skew_ratio
        if imbalance < IMBALANCE_RATIO:
            return None

        # Both signals present: large shuffle read AND imbalance.
        evidence = [
            f"stage shuffle read {format_bytes(shuffle_read)}",
            f"task-duration imbalance {imbalance:.1f}x (max/median)",
        ]
        return Finding(
            # Deliberately not CRITICAL: we are inferring from indirect evidence.
            severity=Severity.MEDIUM,
            sort_index=imbalance,
            category=self.category,
            detector=self.name,
            title=f"Possible expensive join in stage {stage.stage_id}.{stage.attempt_id}",
            stage_id=stage.stage_id,
            stage_attempt_id=stage.attempt_id,
            confidence=Confidence.LOW,
            evidence=evidence,
            metrics={
                "shuffle_read_bytes": float(shuffle_read),
                "task_imbalance_ratio": round(imbalance, 2),
            },
            likely_cause=(
                "A large shuffle read combined with an imbalanced task distribution "
                "is consistent with a shuffle-based join (e.g. sort-merge) on a "
                "skewed key. NOTE: the default event log does not record the join "
                "strategy, so this is an indirect inference, not a confirmed join type."
            ),
            recommendation=(
                "Open the SQL tab in the Spark UI to confirm the join strategy. If one "
                "side is small, a broadcast join avoids the shuffle. If the key is "
                "skewed, enable AQE skew-join handling or salt the key."
            ),
        )
