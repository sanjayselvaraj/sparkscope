"""Generate a Spark event log that contains a *deliberately skewed* join.

This script is NOT part of the sparkscope package. It exists only to produce a
realistic event-log fixture that sparkscope can analyze and that we ship in
`tests/fixtures/`. Run it once; commit the resulting log.

Why a skewed join?
------------------
Data skew is the canonical Spark performance pathology: a join (or groupBy) key
whose values are wildly unbalanced forces one reduce task to process most of the
rows while the rest finish instantly. The Spark UI shows it as "199 tasks done in
seconds, 1 task running for minutes." That single lopsided task is exactly what
SparkScope's skew detector must surface from the event log.

How we force skew
-----------------
We build a `transactions` table whose `customer_id` is ~90% a single hot value
("whale") and ~10% spread across many cold values. Joining that against a
`customers` dimension on `customer_id`, after a shuffle, lands almost all rows in
one partition -> one hot task.

Usage
-----
    pip install "sparkscope[spark]"      # installs pyspark
    python examples/generate_skewed_log.py --out-dir ./tmp_eventlog

Then point sparkscope at the produced file:
    sparkscope analyze ./tmp_eventlog/<app-id>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def build_spark(event_log_dir: Path):
    """Create a local SparkSession with event logging enabled.

    `spark.eventLog.enabled=true` + `spark.eventLog.dir` is what makes Spark
    write the JSON event log we later parse. We also force a small, fixed shuffle
    partition count and DISABLE adaptive query execution (AQE) so the skew is
    visible in the log instead of being auto-mitigated by Spark's skew-join
    handling -- the whole point is to produce a log that *shows* the problem.
    """
    from pyspark.sql import SparkSession  # imported lazily; only needed here

    event_log_dir.mkdir(parents=True, exist_ok=True)

    return (
        SparkSession.builder.appName("sparkscope-skew-fixture")
        .master("local[4]")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", event_log_dir.as_uri())
        .config("spark.sql.shuffle.partitions", "16")
        # Disable AQE so Spark does NOT auto-split the skewed partition.
        .config("spark.sql.adaptive.enabled", "false")
        # Disable broadcast so the join actually shuffles (broadcast would hide skew).
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )


def generate(spark, n_transactions: int, n_customers: int, hot_fraction: float):
    """Create a skewed transactions table and a balanced customers dimension."""
    from pyspark.sql import Row

    hot_customer = 0  # the "whale": most transactions reference this one id
    rng = random.Random(42)

    def tx_rows():
        for i in range(n_transactions):
            cid = hot_customer if rng.random() < hot_fraction else rng.randint(1, n_customers - 1)
            yield Row(txn_id=i, customer_id=cid, amount=round(rng.random() * 100, 2))

    transactions = spark.createDataFrame(list(tx_rows()))
    customers = spark.createDataFrame(
        [Row(customer_id=c, name=f"customer_{c}") for c in range(n_customers)]
    )

    # The skewed shuffle: join on customer_id, then aggregate.
    joined = transactions.join(customers, on="customer_id", how="inner")
    result = joined.groupBy("customer_id").sum("amount")

    # Force execution so the stages/tasks actually run and get logged.
    count = result.count()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./tmp_eventlog"),
        help="Directory Spark writes the event log into.",
    )
    parser.add_argument("--transactions", type=int, default=2_000_000)
    parser.add_argument("--customers", type=int, default=10_000)
    parser.add_argument(
        "--hot-fraction",
        type=float,
        default=0.9,
        help="Fraction of transactions assigned to the single hot customer (drives skew).",
    )
    args = parser.parse_args()

    spark = build_spark(args.out_dir)
    try:
        rows = generate(spark, args.transactions, args.customers, args.hot_fraction)
        print(f"Done. Aggregated {rows} customer groups.")
        print(f"Event log written under: {args.out_dir.resolve()}")
        print("Find the app-id file in that directory and run:")
        print(f"    sparkscope analyze {args.out_dir}/<app-id>")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
