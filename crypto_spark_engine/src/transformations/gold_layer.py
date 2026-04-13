from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import math
from delta import configure_spark_with_delta_pip


def get_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("CryptoEngine_Gold")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
    )

    return configure_spark_with_delta_pip(builder).getOrCreate()

# ── Metadata table (broadcast join demo) ─────────────────────────────────────
COIN_METADATA = [
    ("bitcoin",      "BTC",  "Layer 1",  2009),
    ("ethereum",     "ETH",  "Layer 1",  2015),
    ("solana",       "SOL",  "Layer 1",  2020),
    ("cardano",      "ADA",  "Layer 1",  2017),
    ("avalanche-2",  "AVAX", "Layer 1",  2020),
    ("polkadot",     "DOT",  "Layer 0",  2020),
    ("chainlink",    "LINK", "Oracle",   2017),
    ("litecoin",     "LTC",  "Layer 1",  2011),
]

METADATA_COLS = ["coin_id", "ticker", "category", "launch_year"]


def build_metadata_df(spark: SparkSession) -> DataFrame:
    """
    Creates a small lookup DataFrame from in-memory data.
    In production this would be read from a database or config file.
    """
    return spark.createDataFrame(COIN_METADATA, schema=METADATA_COLS)


# ── Gold Table 1: Daily Summary ───────────────────────────────────────────────
def build_daily_summary(silver_df: DataFrame, metadata_df: DataFrame) -> DataFrame:
    """
    One row per (coin_id, event_date) with all key enriched metrics.

    Broadcast Join Rationale:
      metadata_df has 8 rows — tiny. Joining it the standard way (SortMergeJoin)
      would trigger a shuffle of the entire silver_df just to join 8 rows.
      F.broadcast() tells Spark to replicate the small table to every executor
      so the join happens locally — no shuffle needed.

      Rule of thumb: broadcast tables < 10MB (configured via
      spark.sql.autoBroadcastJoinThreshold, default 10MB).
      With AQE enabled, Spark can auto-broadcast if it detects the table is small
      at runtime — but explicit broadcast() makes the intent clear.
    """
    return (
        silver_df
        .join(F.broadcast(metadata_df), on="coin_id", how="left")
        .select(
            "coin_id",
            "ticker",
            "category",
            "event_date",
            "open", "high", "low", "close",
            "total_volume_usd",
            "market_cap_usd",
            "avg_close_7d",
            "avg_close_14d",
            "avg_close_30d",
            "vwap_7d",
            "pct_change_1d",
            "pct_change_7d",
            "high_low_spread_pct",
            "rsi_14",
            "market_regime",
            "rank_by_mcap",
            "dense_rank_by_mcap",
        )
        .orderBy("coin_id", "event_date")
    )


# ── Gold Table 2: Weekly Performance Leaderboard ─────────────────────────────
def build_weekly_rankings(silver_df: DataFrame) -> DataFrame:
    """
    Weekly coin performance leaderboard.

    Steps:
      1. Extract ISO week (year + week number) from event_date.
      2. Aggregate: first open, last close per week per coin.
      3. Compute weekly return %.
      4. Rank coins within each week by weekly return.

    Why F.first/F.last with ignorenulls=True?
      Window-based first/last (lag/lead) would require a self-join.
      groupBy + agg with first/last is simpler and sufficient since we want
      the extreme values within the weekly bucket, not relative to other rows.
    """
    weekly = (
        silver_df
        .withColumn("iso_year",  F.year("event_date"))
        .withColumn("iso_week",  F.weekofyear("event_date"))
        .groupBy("coin_id", "iso_year", "iso_week")
        .agg(
            F.first("open",  ignorenulls=True).alias("week_open"),
            F.last("close",  ignorenulls=True).alias("week_close"),
            F.max("high").alias("week_high"),
            F.min("low").alias("week_low"),
            F.sum("total_volume_usd").alias("week_total_volume"),
            F.avg("market_cap_usd").alias("week_avg_mcap"),
        )
        .withColumn(
            "weekly_return_pct",
            (F.col("week_close") - F.col("week_open")) / F.col("week_open") * 100
        )
    )

    week_window = Window.partitionBy("iso_year", "iso_week").orderBy(
        F.col("weekly_return_pct").desc()
    )

    return (
        weekly
        .withColumn("weekly_rank", F.rank().over(week_window))
        .orderBy("iso_year", "iso_week", "weekly_rank")
    )


# ── Gold Table 3: Volatility Report with Bollinger Bands ─────────────────────
def build_volatility_report(silver_df: DataFrame) -> DataFrame:
    """
    Annualized volatility + Bollinger Bands (20-day, 2 std dev).

    Annualized Volatility:
        = stddev(daily_returns) × sqrt(365)
        This converts the daily standard deviation to an annual figure,
        assuming 365 trading days (crypto trades 24/7, unlike equity markets).
        For equity, you'd use sqrt(252).

    Bollinger Bands:
        Upper = 20-day SMA + 2 × 20-day stddev
        Lower = 20-day SMA - 2 × 20-day stddev
        Price outside bands signals potential breakout or mean reversion.

    %B indicator:
        = (close - lower_band) / (upper_band - lower_band)
        0 = at lower band, 1 = at upper band, >1 = above upper band (overbought).
    """
    TRADING_DAYS_PER_YEAR = 365
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD_MULTIPLIER = 2.0

    w30   = Window.partitionBy("coin_id").orderBy("event_date").rowsBetween(-29, 0)
    w20bb = Window.partitionBy("coin_id").orderBy("event_date").rowsBetween(-BOLLINGER_PERIOD + 1, 0)

    return (
        silver_df
        # Annualized volatility from 30-day rolling stddev of daily returns
        .withColumn(
            "annualized_vol_pct",
            F.stddev("pct_change_1d").over(w30) * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
        # Bollinger Bands
        .withColumn("bb_sma_20",    F.avg("close").over(w20bb))
        .withColumn("bb_stddev_20", F.stddev("close").over(w20bb))
        .withColumn(
            "bb_upper",
            F.col("bb_sma_20") + BOLLINGER_STD_MULTIPLIER * F.col("bb_stddev_20")
        )
        .withColumn(
            "bb_lower",
            F.col("bb_sma_20") - BOLLINGER_STD_MULTIPLIER * F.col("bb_stddev_20")
        )
        # %B indicator
        .withColumn(
            "bb_pct_b",
            (F.col("close") - F.col("bb_lower")) / (F.col("bb_upper") - F.col("bb_lower"))
        )
        # Bandwidth: how wide the bands are relative to SMA (expansion/contraction signal)
        .withColumn(
            "bb_bandwidth",
            (F.col("bb_upper") - F.col("bb_lower")) / F.col("bb_sma_20")
        )
        .select(
            "coin_id", "event_date", "close",
            "annualized_vol_pct",
            "bb_sma_20", "bb_upper", "bb_lower",
            "bb_pct_b", "bb_bandwidth",
            "rsi_14", "market_regime",
        )
        .orderBy("coin_id", "event_date")
    )


# ── Gold Table 4: Regime Frequency Summary ───────────────────────────────────
def build_regime_summary(silver_df: DataFrame) -> DataFrame:
    """
    Aggregate regime statistics per coin.

    Answers: For each coin, how often is it in each regime, and what is the
    average return during that regime? This is essentially a performance
    attribution table by market condition.

    Note: avg_return_in_regime can be used to assess whether a regime label
    predicts anything — if trending_bullish averages +2% and ranging_quiet
    averages +0.1%, the labels have signal. If not, revisit the classifier.
    """
    return (
        silver_df
        .filter(F.col("market_regime") != "unknown")
        .groupBy("coin_id", "market_regime")
        .agg(
            F.count("*").alias("days_in_regime"),
            F.avg("pct_change_1d").alias("avg_daily_return_pct"),
            F.avg("rsi_14").alias("avg_rsi"),
            F.avg("high_low_spread_pct").alias("avg_spread_pct"),
            F.min("event_date").alias("first_occurrence"),
            F.max("event_date").alias("last_occurrence"),
        )
        .withColumn(
            "regime_pct_of_history",
            F.col("days_in_regime") /
            F.sum("days_in_regime").over(Window.partitionBy("coin_id")) * 100
        )
        .orderBy("coin_id", F.col("days_in_regime").desc())
    )


# ── Caching Strategy ──────────────────────────────────────────────────────────
def cache_silver_with_rationale(silver_df: DataFrame) -> DataFrame:
    """
    Cache silver_df before building multiple Gold tables from it.

    Without caching:
      Each Gold table triggers a full re-read of the Delta Silver table + all
      transformations (window functions, UDFs) — 4x the computation.

    With caching (MEMORY_AND_DISK):
      First action materializes Silver into memory. Subsequent Gold table builds
      read from cache — no re-computation, no disk I/O.

    When NOT to cache:
      - If the DataFrame is too large for memory (cache spills to disk → slower).
      - If the DataFrame is only used once.
      - If the computation is cheap (e.g. a simple filter) — caching overhead
        may exceed the savings.

    MEMORY_AND_DISK is safer than MEMORY_ONLY — if cache overflows RAM, it
    spills to disk rather than evicting and recomputing.
    """
    from pyspark import StorageLevel
    silver_df.persist(StorageLevel.MEMORY_AND_DISK)
    print("[Gold] Silver DataFrame cached (MEMORY_AND_DISK)")
    return silver_df


# ── Main ──────────────────────────────────────────────────────────────────────
def run_gold(processed_path: str = "data/processed") -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("GOLD LAYER — Starting")
    print("=" * 60)

    # Read Silver
    silver_df = spark.read.format("delta").load(f"{processed_path}/silver_enriched")

    # Cache before building multiple Gold tables
    silver_df = cache_silver_with_rationale(silver_df)

    # Build metadata for broadcast join
    metadata_df = build_metadata_df(spark)

    # ── Build & write Gold tables ─────────────────────────────────
    tables = {
        "gold_daily_summary":    build_daily_summary(silver_df, metadata_df),
        "gold_weekly_rankings":  build_weekly_rankings(silver_df),
        "gold_volatility_report": build_volatility_report(silver_df),
        "gold_regime_summary":   build_regime_summary(silver_df),
    }

    for table_name, df in tables.items():
        output = f"{processed_path}/{table_name}"
        (
            df.write
            .mode("overwrite")
            .parquet(output)
        )
        row_count = spark.read.parquet(output).count()
        print(f"[Gold] {table_name}: {row_count} rows → {output}")

    # Unpersist cache after all Gold tables are built
    silver_df.unpersist()
    print("[Gold] Silver cache released")

    print("=" * 60)
    print("GOLD LAYER — Complete")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    run_gold()
