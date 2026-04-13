from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


@dataclass
class QualityResult:
    check_name: str
    passed: bool
    metric_value: float
    threshold: float
    details: str
    run_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("CryptoEngine_Quality")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# ── Individual Checks ─────────────────────────────────────────────────────────

def check_null_rates(df: DataFrame, max_null_rate: float = 0.05) -> list[QualityResult]:

    total = df.count()
    results = []

    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    for col_name, null_count in null_counts.items():
        rate = null_count / total if total > 0 else 0.0
        # Skip metadata/timestamp columns — nulls there are acceptable
        if col_name.startswith("_") or col_name in ("event_year", "event_month"):
            continue
        passed = rate <= max_null_rate
        results.append(QualityResult(
            check_name=f"null_rate_{col_name}",
            passed=passed,
            metric_value=round(rate, 4),
            threshold=max_null_rate,
            details=f"{null_count}/{total} nulls in '{col_name}'",
        ))

    return results


def check_rsi_range(df: DataFrame) -> QualityResult:

    if "rsi_14" not in df.columns:
        return QualityResult("rsi_range", True, 0.0, 0.0, "rsi_14 column not present — skipped")

    bad_count = df.filter(
        (F.col("rsi_14") < 0) | (F.col("rsi_14") > 100)
    ).count()

    return QualityResult(
        check_name="rsi_range_validity",
        passed=bad_count == 0,
        metric_value=float(bad_count),
        threshold=0.0,
        details=f"{bad_count} rows with RSI outside [0, 100]",
    )


def check_ohlc_consistency(df: DataFrame) -> QualityResult:

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return QualityResult("ohlc_consistency", True, 0.0, 0.0, "OHLC columns not present — skipped")

    bad_count = df.filter(
        (F.col("low") > F.col("open"))  |
        (F.col("low") > F.col("close")) |
        (F.col("high") < F.col("open")) |
        (F.col("high") < F.col("close"))
    ).count()

    return QualityResult(
        check_name="ohlc_consistency",
        passed=bad_count == 0,
        metric_value=float(bad_count),
        threshold=0.0,
        details=f"{bad_count} rows violating low <= OHLC <= high",
    )


def check_price_anomalies(df: DataFrame, multiplier: float = 10.0) -> QualityResult:

    if "avg_close_30d" not in df.columns:
        return QualityResult("price_anomaly", True, 0.0, 0.0, "avg_close_30d not present — skipped")

    anomaly_count = df.filter(
        F.col("avg_close_30d").isNotNull() &
        (F.col("close") > multiplier * F.col("avg_close_30d"))
    ).count()

    return QualityResult(
        check_name="price_anomaly_detection",
        passed=anomaly_count == 0,
        metric_value=float(anomaly_count),
        threshold=0.0,
        details=f"{anomaly_count} rows where close > {multiplier}× 30d avg",
    )


def check_zero_volume(df: DataFrame, max_zero_vol_rate: float = 0.02) -> QualityResult:

    if "total_volume_usd" not in df.columns:
        return QualityResult("zero_volume", True, 0.0, 0.0, "volume column not present — skipped")

    total = df.count()
    zero_count = df.filter(
        (F.col("total_volume_usd").isNull()) |
        (F.col("total_volume_usd") == 0)
    ).count()

    rate = zero_count / total if total > 0 else 0.0

    return QualityResult(
        check_name="zero_volume_rate",
        passed=rate <= max_zero_vol_rate,
        metric_value=round(rate, 4),
        threshold=max_zero_vol_rate,
        details=f"{zero_count}/{total} rows with zero/null volume",
    )


def check_regime_distribution(df: DataFrame, max_dominant_rate: float = 0.80) -> list[QualityResult]:

    if "market_regime" not in df.columns or "coin_id" not in df.columns:
        return []

    total_per_coin = df.groupBy("coin_id").count().withColumnRenamed("count", "total")
    regime_counts = (
        df.groupBy("coin_id", "market_regime")
        .count()
        .join(total_per_coin, on="coin_id")
        .withColumn("regime_rate", F.col("count") / F.col("total"))
        .filter(F.col("regime_rate") > max_dominant_rate)
    )

    rows = regime_counts.collect()
    results = []
    for row in rows:
        results.append(QualityResult(
            check_name=f"regime_distribution_{row['coin_id']}",
            passed=False,
            metric_value=round(row["regime_rate"], 4),
            threshold=max_dominant_rate,
            details=(
                f"Regime '{row['market_regime']}' dominates {row['regime_rate']*100:.1f}% "
                f"of days for {row['coin_id']} — consider retuning classifier thresholds"
            ),
        ))
    return results


# ── Report Writer ─────────────────────────────────────────────────────────────

def write_quality_report(results: list[QualityResult], spark: SparkSession, output_path: str) -> None:

    rows = [
        (r.check_name, r.passed, r.metric_value, r.threshold, r.details, r.run_ts)
        for r in results
    ]
    schema = ["check_name", "passed", "metric_value", "threshold", "details", "run_ts"]
    report_df = spark.createDataFrame(rows, schema=schema)

    report_df.write.mode("append").parquet(f"{output_path}/quality_report")

    # Summary to stdout
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT — {passed} passed / {failed} failed / {len(results)} total")
    print("=" * 60)
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"  {status}  {r.check_name}: {r.details}")
    print("=" * 60 + "\n")

    if failed > 0:
        print(f"[WARN] {failed} quality check(s) failed. Review before proceeding to Gold.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_quality_checks(processed_path: str = "data/processed") -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    silver_df = spark.read.format("delta").load(f"{processed_path}/silver_enriched")
    # Cache since we run multiple checks on the same DataFrame
    silver_df.cache()

    results: list[QualityResult] = []
    results += check_null_rates(silver_df)
    results.append(check_rsi_range(silver_df))
    results.append(check_ohlc_consistency(silver_df))
    results.append(check_price_anomalies(silver_df))
    results.append(check_zero_volume(silver_df))
    results += check_regime_distribution(silver_df)

    write_quality_report(results, spark, processed_path)

    silver_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    run_quality_checks()
