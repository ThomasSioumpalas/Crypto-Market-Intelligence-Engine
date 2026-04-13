import time
from contextlib import contextmanager
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  ⏱  {label}: {elapsed:.3f}s")


def count_exchanges(df: DataFrame) -> int:
   
    plan = df._jdf.queryExecution().executedPlan().toString()
    return plan.count("Exchange")


def print_section(title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print("━" * 60)


def get_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("CryptoEngine_Optimization")
        .master("spark://spark-master:7077")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "false")  # Disable AQE for fair comparison
    )

    return configure_spark_with_delta_pip(builder).getOrCreate()

# ── Analysis 1: Shuffle Count ─────────────────────────────────────────────────

def analyze_shuffle_count(silver_df: DataFrame) -> None:
   
    print_section("1. SHUFFLE ANALYSIS")

    window = Window.partitionBy("coin_id").orderBy("event_date").rowsBetween(-6, 0)
    query = silver_df.withColumn("rolling_7d", F.avg("close").over(window))

    n_exchanges = count_exchanges(query)
    print(f"  Exchange nodes in plan: {n_exchanges}")
    print("  Each Exchange = one shuffle. Window functions need exactly 1.")
    print("  If you see > 1, look for redundant sorts or unnecessary repartitions.")
    print()
    query.explain(mode="formatted")


# ── Analysis 2: Broadcast Join vs SortMergeJoin ───────────────────────────────

def analyze_join_strategies(silver_df: DataFrame, spark: SparkSession) -> None:
   
    print_section("2. BROADCAST JOIN vs SORT-MERGE JOIN")

    # Small metadata table (8 rows)
    metadata = spark.createDataFrame(
        [("bitcoin","BTC"),("ethereum","ETH"),("solana","SOL"),
         ("cardano","ADA"),("avalanche-2","AVAX"),("polkadot","DOT"),
         ("chainlink","LINK"),("litecoin","LTC")],
        ["coin_id", "ticker"]
    )

    print("  Sort-Merge Join (no broadcast hint):")
    with timer("  SortMergeJoin"):
        smj_result = silver_df.join(metadata, on="coin_id", how="left")
        smj_result.count()  # trigger action
    smj_exchanges = count_exchanges(smj_result)
    print(f"    Exchange nodes: {smj_exchanges}")

    print()
    print("  Broadcast Join (explicit broadcast hint):")
    with timer("  BroadcastHashJoin"):
        bhj_result = silver_df.join(F.broadcast(metadata), on="coin_id", how="left")
        bhj_result.count()
    bhj_exchanges = count_exchanges(bhj_result)
    print(f"    Exchange nodes: {bhj_exchanges}")
    print()
    print("  → Broadcast join should have fewer Exchange nodes (no shuffle of large table)")


# ── Analysis 3: Cache vs No-Cache ─────────────────────────────────────────────

def analyze_cache_benefit(silver_df: DataFrame) -> None:
   
    print_section("3. CACHE BENEFIT ANALYSIS")

    def run_two_aggs(df: DataFrame, label: str):
        with timer(f"{label} — agg 1 (count by coin)"):
            df.groupBy("coin_id").count().collect()
        with timer(f"{label} — agg 2 (avg close by coin)"):
            df.groupBy("coin_id").agg(F.avg("close")).collect()

    print("  Without cache:")
    run_two_aggs(silver_df, "no cache")

    print()
    print("  With cache:")
    silver_df.cache()
    silver_df.count()  # materialize cache
    run_two_aggs(silver_df, "cached")
    silver_df.unpersist()

    print()
    print("  → If data fits in memory, cached runs should be faster.")
    print("  → With tiny datasets, overhead may outweigh benefit.")


# ── Analysis 4: Partition Pruning ─────────────────────────────────────────────

def analyze_partition_pruning(processed_path: str, spark: SparkSession) -> None:
   
    print_section("4. PARTITION PRUNING VERIFICATION")

    try:
        silver_df = spark.read.format("delta").load(f"{processed_path}/silver_enriched")

        print("  Full scan (no filter):")
        full_plan = silver_df.explain(mode="formatted")

        print()
        print("  Filtered scan (coin_id = 'bitcoin'):")
        filtered = silver_df.filter(F.col("coin_id") == "bitcoin")
        filtered.explain(mode="formatted")

        print()
        print("  → Look for 'PartitionFilters: [isnotnull(coin_id), (coin_id = bitcoin)]'")
        print("  → The filtered plan should show fewer files in 'numFiles'")
    except Exception as e:
        print(f"  [SKIP] Delta table not found — run bronze/silver layers first. ({e})")


# ── Analysis 5: UDF vs Built-in ───────────────────────────────────────────────

def analyze_udf_vs_builtin(silver_df: DataFrame) -> None:
   
    print_section("5. UDF vs BUILT-IN FUNCTION")

    from pyspark.sql.types import DoubleType

    @F.udf(DoubleType())
    def abs_udf(x):
        return abs(x) if x is not None else None

    print("  Python UDF (row-level, Python ↔ JVM per row):")
    with timer("  abs_udf"):
        silver_df.withColumn("abs_change_udf", abs_udf(F.col("pct_change_1d"))).count()

    print()
    print("  Built-in F.abs() (JVM-native, no serialization):")
    with timer("  F.abs()"):
        silver_df.withColumn("abs_change_builtin", F.abs(F.col("pct_change_1d"))).count()

    print()
    print("  → Built-in functions always win on numeric ops — check Spark docs first!")


# ── Analysis 6: Skew Detection ───────────────────────────────────────────────

def analyze_data_skew(silver_df: DataFrame) -> None:
  
    print_section("6. DATA SKEW DETECTION")

    # Row count per coin — rough skew indicator
    print("  Row count per coin (should be roughly equal for same date range):")
    silver_df.groupBy("coin_id").count().orderBy("count", ascending=False).show(20, truncate=False)

    # Partition size distribution
    partition_sizes = silver_df.rdd.mapPartitions(
        lambda it: [sum(1 for _ in it)]
    ).collect()

    if partition_sizes:
        print(f"  Partition row counts: {sorted(partition_sizes, reverse=True)}")
        max_p = max(partition_sizes)
        min_p = min(partition_sizes)
        ratio = max_p / min_p if min_p > 0 else float("inf")
        print(f"  Max/Min partition ratio: {ratio:.2f}x")
        if ratio > 5:
            print("  ⚠️  HIGH SKEW DETECTED: largest partition has {ratio:.0f}× more rows than smallest")
            print("  Fix: repartition by a more evenly distributed key, or use salting for joins.")
        else:
            print("  ✅  Partition distribution looks healthy (ratio < 5×)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_optimization_analysis(processed_path: str = "data/processed") -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "=" * 60)
    print("  QUERY OPTIMIZATION ANALYSIS")
    print("=" * 60)

    try:
        silver_df = spark.read.format("delta").load(f"{processed_path}/silver_enriched")
        silver_df.cache()
        silver_df.count()  # materialize

        analyze_shuffle_count(silver_df)
        analyze_join_strategies(silver_df, spark)
        analyze_cache_benefit(silver_df)
        analyze_partition_pruning(processed_path, spark)
        analyze_udf_vs_builtin(silver_df)
        analyze_data_skew(silver_df)

        silver_df.unpersist()
    except Exception as e:
        print(f"\n[ERROR] Run bronze + silver layers first: {e}")

    spark.stop()


if __name__ == "__main__":
    run_optimization_analysis()
