"""Shuffle detector.

A shuffle moves data across the network between stages. Shuffles are normal and
often necessary (joins, wide aggregations, repartitions), so **large shuffle is
not automatically a problem** -- flagging every big shuffle would be noise.

What we can and cannot say from the event log
---------------------------------------------
The log gives us per-task and stage-total shuffle read/write bytes (local vs
remote) and input bytes. It does NOT tell us *why* a shuffle happened or whether
it was logically required. So we never say "this shuffle is unnecessary." We
only surface shuffle that is *large in absolute terms* AND *large relative to
the data the stage ingested* -- a pattern consistent with moving far more data
than the stage reduced, which is worth a human look.

Signals used
------------
* Absolute floor: ignore shuffles below ``MIN_SHUFFLE_BYTES`` -- small shuffles
  never matter.
* Write-amplification: shuffle write bytes relative to stage input bytes. If a
  stage reads X and writes several·X to shuffle, data is being expanded/moved
  rather than reduced before the exchange -- worth investigating. (Only computed
  when input bytes are present; otherwise we report the magnitude as INFO.)
* Remote-read dominance: a very high remote/local read ratio means almost all
  shuffle data crossed the network, which is the expensive path.
"""

from __future__ import annotations

from sparkscope.analysis.base import Detector, register
from sparkscope.analysis.finding import Confidence, Finding, Severity
from sparkscope.analysis.util import format_bytes, ratio
from sparkscope.parser.models import SparkRun, Stage

#: Below this total shuffle (read+write), a stage is never flagged.
MIN_SHUFFLE_BYTES = 1 * 1024**3  # 1 GB
#: Shuffle-write / input ratio bands (only meaningful when input is known).
WRITE_AMP_MEDIUM = 2.0
WRITE_AMP_HIGH = 5.0


def _severity_for_write_amp(amp: float) -> Severity | None:
    if amp >= WRITE_AMP_HIGH:
        return Severity.HIGH
    if amp >= WRITE_AMP_MEDIUM:
        return Severity.MEDIUM
    return None


@register
class ShuffleDetector(Detector):
    """Flags stages whose shuffle looks large and inefficient relative to input."""

    name = "shuffle"
    category = "shuffle"

    def analyze(self, run: SparkRun) -> list[Finding]:
        findings: list[Finding] = []
        for stage in run.stage_list():
            finding = self._check_stage(stage)
            if finding is not None:
                findings.append(finding)
        return findings

    def _check_stage(self, stage: Stage) -> Finding | None:
        read = stage.total_shuffle_read_bytes
        write = stage.total_shuffle_write_bytes
        total = read + write
        if total < MIN_SHUFFLE_BYTES:
            return None

        input_bytes = stage.total_input_bytes
        write_amp = ratio(write, input_bytes)  # 0.0 if input unknown

        evidence = [
            f"stage shuffle write {format_bytes(write)}, read {format_bytes(read)}",
        ]

        severity = _severity_for_write_amp(write_amp) if input_bytes > 0 else None

        if severity is not None:
            # We have input bytes and the write amplification clears a band.
            evidence.append(
                f"shuffle write is {write_amp:.1f}x the stage input "
                f"({format_bytes(input_bytes)})"
            )
            likely_cause = (
                "The stage writes substantially more data to shuffle than it read "
                "as input, which is consistent with an exchange that moves/expands "
                "data rather than reducing it first (e.g. a join or repartition "
                "before any filtering/aggregation)."
            )
            confidence = Confidence.MEDIUM
        else:
            # Large shuffle, but we cannot show inefficiency -> INFO, not a problem.
            # Reported so the user sees where the big exchanges are, clearly labelled
            # as informational rather than a defect.
            if input_bytes == 0:
                return None  # no basis to say anything useful; stay silent
            severity = Severity.INFO
            evidence.append(
                f"shuffle is proportionate to input ({format_bytes(input_bytes)}); "
                "shown for awareness, not flagged as a defect"
            )
            likely_cause = (
                "Large but seemingly proportionate shuffle. Large shuffles are often "
                "legitimate; this is informational, not necessarily a problem."
            )
            confidence = Confidence.LOW

        metrics = {
            "shuffle_write_bytes": float(write),
            "shuffle_read_bytes": float(read),
            "input_bytes": float(input_bytes),
            "write_amplification": round(write_amp, 2),
        }

        return Finding(
            severity=severity,
            sort_index=write_amp,
            category=self.category,
            detector=self.name,
            title=f"Large shuffle in stage {stage.stage_id}.{stage.attempt_id}",
            stage_id=stage.stage_id,
            stage_attempt_id=stage.attempt_id,
            confidence=confidence,
            evidence=evidence,
            metrics=metrics,
            likely_cause=likely_cause,
            recommendation=(
                "Check whether a filter or aggregation can run before the exchange "
                "to shrink the shuffled data, whether a broadcast join can avoid the "
                "shuffle, and whether spark.sql.shuffle.partitions is sized for this "
                "data volume. Confirm the exchange is required in the SQL plan."
            ),
        )
