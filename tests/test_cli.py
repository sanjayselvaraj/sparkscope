"""Day 1 smoke tests: the CLI is wired and validates input.

These are intentionally thin -- they prove the package imports, the entrypoint
exists, and argument validation works. Real behavioral tests arrive with the
parser and detectors.
"""

from __future__ import annotations

from typer.testing import CliRunner

from sparkscope import __version__
from sparkscope.cli import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_analyze_rejects_missing_path():
    result = runner.invoke(app, ["analyze", "/no/such/event-log"])
    assert result.exit_code != 0  # Typer's exists=True guard fires


def test_analyze_parses_a_minimal_event_log(tmp_path):
    log = tmp_path / "event-log"
    log.write_text(
        '{"Event":"SparkListenerApplicationStart","App Name":"demo","App ID":"a1"}\n'
        '{"Event":"SparkListenerTaskEnd","Stage ID":0,"Stage Attempt ID":0,'
        '"Task Info":{"Task ID":0,"Launch Time":0,"Finish Time":100},'
        '"Task Metrics":{"Executor Run Time":100}}\n'
    )
    result = runner.invoke(app, ["analyze", str(log)])
    assert result.exit_code == 0
    assert "SparkScope" in result.stdout
    assert "demo" in result.stdout
    assert "Stage summary" in result.stdout


def test_analyze_rejects_non_spark_file(tmp_path):
    log = tmp_path / "not-a-spark-log"
    log.write_text("{}\n")  # valid JSON, but no Spark events
    result = runner.invoke(app, ["analyze", str(log)])
    assert result.exit_code == 2
    assert "error" in result.stdout.lower()
