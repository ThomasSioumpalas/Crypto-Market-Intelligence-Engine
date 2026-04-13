from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.window import Window
import pandas as pd
from delta import configure_spark_with_delta_pip


def get_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("CryptoEngine_Silver")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "8")
        # AQE: Adaptive Query Execution — lets Spark re-plan mid-query based on
        # runtime statistics. Particularly useful for skewed joins.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    return configure_spark_with_delta_pip(builder).getOrCreate()


def coin_date_window(days: int) -> Window:

    return (
        Window
        .partitionBy("coin_id")
        .orderBy("event_date")
        .rowsBetween(-(days - 1), 0)
    )

def coin_full_window() -> Window:
      return Window.partitionBy("coin_id").orderBy("event_date")


# ── Join Logic ─────────────────────────────────────────────────────────────────
def join_bronze_tables(ohlc_df: DataFrame, market_df: DataFrame) -> DataFrame:
    market_slim = market_df.select(
        "coin_id", "event_date", "market_cap_usd", "total_volume_usd"
    )

    return ohlc_df.join(market_slim, on=["coin_id", "event_date"], how="inner")


# ── Window Function Transformations ───────────────────────────────────────────
def add_rolling_averages(df: DataFrame) -> DataFrame:

    return (
        df
        .withColumn("avg_close_7d",  F.avg("close").over(coin_date_window(7)))
        .withColumn("avg_close_14d", F.avg("close").over(coin_date_window(14)))
        .withColumn("avg_close_30d", F.avg("close").over(coin_date_window(30)))
        # Rolling stddev — needed for volatility and Bollinger Bands in Gold
        .withColumn("stddev_close_30d", F.stddev("close").over(coin_date_window(30)))
    )


def add_vwap(df: DataFrame) -> DataFrame:

    w7 = coin_date_window(7)
    return (
        df
        .withColumn("_price_vol_7d", F.sum(F.col("close") * F.col("total_volume_usd")).over(w7))
        .withColumn("_vol_7d",       F.sum("total_volume_usd").over(w7))
        .withColumn("vwap_7d", F.col("_price_vol_7d") / F.col("_vol_7d"))
        .drop("_price_vol_7d", "_vol_7d")
    )


def add_momentum_features(df: DataFrame) -> DataFrame:

    w = coin_full_window()
    return (
        df
        # Previous close prices
        .withColumn("prev_close_1d", F.lag("close", 1).over(w))
        .withColumn("prev_close_7d", F.lag("close", 7).over(w))
        # Percentage changes — null for first N rows (no prior data)
        .withColumn(
            "pct_change_1d",
            (F.col("close") - F.col("prev_close_1d")) / F.col("prev_close_1d") * 100
        )
        .withColumn(
            "pct_change_7d",
            (F.col("close") - F.col("prev_close_7d")) / F.col("prev_close_7d") * 100
        )
        # Intraday spread as % of close price
        .withColumn(
            "high_low_spread_pct",
            (F.col("high") - F.col("low")) / F.col("close") * 100
        )
        .drop("prev_close_1d", "prev_close_7d")
    )


def add_daily_rankings(df: DataFrame) -> DataFrame:
    date_window = Window.partitionBy("event_date").orderBy(F.col("market_cap_usd").desc())
    return (
        df
        .withColumn("rank_by_mcap",       F.rank().over(date_window))
        .withColumn("dense_rank_by_mcap", F.dense_rank().over(date_window))
        .withColumn("pct_rank_by_mcap",   F.percent_rank().over(date_window))
    )


# ── Pandas UDF: RSI ───────────────────────────────────────────────────────────
@F.pandas_udf(DoubleType())
def compute_rsi_udf(close_series: pd.Series) -> pd.Series:
    
    period = 14
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Exponential moving average (EMA) — standard Wilder smoothing
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50.0)   # fill initial NaN with neutral value


# ── Standard UDF: Regime Classifier ──────────────────────────────────────────
@F.udf(StringType())
def classify_regime(
    pct_change_7d: float,
    high_low_spread_pct: float,
    rsi: float
) -> str:
    
    if pct_change_7d is None or high_low_spread_pct is None or rsi is None:
        return "unknown"

    abs_momentum = abs(pct_change_7d)
    is_volatile = high_low_spread_pct > 5.0   # >5% intraday spread = high vol

    if abs_momentum > 10 and rsi > 65:
        return "trending_bullish"
    elif abs_momentum > 10 and rsi < 35:
        return "trending_bearish"
    elif is_volatile and abs_momentum < 5:
        return "ranging_volatile"
    elif abs_momentum < 2 and not is_volatile:
        return "ranging_quiet"
    elif pct_change_7d < -10 and rsi < 30:
        return "reversing_oversold"
    elif pct_change_7d > 10 and rsi > 70:
        return "reversing_overbought"
    else:
        return "neutral"


# ── Optimization: Explain Plan Logger ────────────────────────────────────────
def log_explain_plan(df: DataFrame, label: str) -> None:
   
    print(f"\n{'='*60}")
    print(f"EXPLAIN PLAN — {label}")
    print("=" * 60)
    df.explain(extended=True)   # extended=True shows logical + physical plans
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_silver(processed_path: str = "data/processed") -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("SILVER LAYER — Starting")
    print("=" * 60)

    # Read Bronze Delta tables
    ohlc_df   = spark.read.format("delta").load(f"{processed_path}/bronze_ohlc")
    market_df = spark.read.format("delta").load(f"{processed_path}/bronze_market")

    # ── Join ──────────────────────────────────────────────────────
    joined_df = join_bronze_tables(ohlc_df, market_df)

    # Log plan before window functions (shows the join strategy)
    log_explain_plan(joined_df, "After join, before window functions")

    # ── Window Features ───────────────────────────────────────────
    df = add_rolling_averages(joined_df)
    df = add_vwap(df)
    df = add_momentum_features(df)
    df = add_daily_rankings(df)

    # ── Pandas UDF: RSI ───────────────────────────────────────────
    # RSI must be computed per coin and ordered by date.
    # We sort before applying the UDF — the Pandas UDF receives all rows
    # for a partition, and RSI is order-dependent.
    df = df.sortWithinPartitions("coin_id", "event_date")
    df = df.withColumn("rsi_14", compute_rsi_udf(F.col("close")))

    # ── Standard UDF: Regime ──────────────────────────────────────
    df = df.withColumn(
        "market_regime",
        classify_regime(
            F.col("pct_change_7d"),
            F.col("high_low_spread_pct"),
            F.col("rsi_14"),
        )
    )

    # Log plan after all transformations — observe window function shuffles
    log_explain_plan(df, "Full Silver pipeline (pre-write)")

    # ── Write Silver ──────────────────────────────────────────────
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("coin_id", "event_date")
        .save(f"{processed_path}/silver_enriched")
    )

    print(f"[Silver] Row count: {df.count()}")
    print(f"[Silver] Unique regimes: {df.select('market_regime').distinct().collect()}")
    print("=" * 60)
    print("SILVER LAYER — Complete")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    run_silver()
