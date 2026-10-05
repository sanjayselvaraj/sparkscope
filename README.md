# SparkScope

> Diagnose Apache Spark performance issues — data skew, disk spill, shuffle bloat, and bad joins — straight from the event log. **No running cluster required.**

Spark performance problems are the daily toil of data engineering, and the only
way to find them is usually clicking through the Spark UI stage by stage.
SparkScope reads the JSON event log Spark already writes, reconstructs the
job → stage → task model, and runs detectors that each map to a real failure
mode — then prints a ranked list of what's wrong and how to fix it.

```bash
pip install sparkscope
sparkscope analyze /path/to/event-log
```

## Status

🚧 **v0.1 in active development (14-day build).** Day 1 ships the scaffold, CLI,
and a reproducible skewed-join event-log fixture. Detectors land across Week 1–2.

## Why no Spark dependency?

SparkScope parses the event log Spark emits — it never runs or attaches to a
cluster. That means it installs in seconds, runs offline on your laptop or in
CI, and works against any log from any Spark 3.x deployment. (`pyspark` is an
optional extra, needed only to *regenerate* the example fixture.)

## Planned detectors (v0.1)

| Detector | Finds | Typical fix it suggests |
|---|---|---|
| Skew | One task running far longer than the stage median | Enable AQE skew-join / salt the key |
| Spill | Tasks spilling to disk under memory pressure | Raise memory / reduce partition size |
| Shuffle bloat | Oversized shuffle read/write | Filter earlier / reduce partitions |
| Partition sizing | Too many tiny or too few huge partitions | Tune `spark.sql.shuffle.partitions` |
| Join | Expensive sort-merge joins / missing broadcast | Broadcast small side |

## Regenerate the example fixture

```bash
pip install "sparkscope[spark]"
python examples/generate_skewed_log.py --out-dir ./tmp_eventlog
sparkscope analyze ./tmp_eventlog/<app-id>
```

## License

Apache-2.0
