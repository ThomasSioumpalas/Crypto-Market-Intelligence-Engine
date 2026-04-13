# Databricks notebook source
# ─────────────────────────────────────────────────────────────────────────────
# Crypto Market Intelligence Engine — Databricks Version
#
# This notebook is the Databricks-native version of the local Docker pipeline.
# Key differences from the local version:
#
#   1. No SparkSession creation — Databricks provides `spark` automatically.
#   2. Storage paths use DBFS (dbfs:/...) or Unity Catalog volumes.
#   3. Delta Lake is built-in — no extra config needed.
#   4. Shuffle partitions tuned higher — Databricks clusters have more cores.
#   5. Cluster config matters: use a multi-node cluster for meaningful
#      performance differences in the optimization section.
#
# Recommended cluster: DBR 14.x LTS, 2 workers (i3.xlarge or equivalent)
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------
# MAGIC %md
# MAGIC # 🔧 Setup & Config

# COMMAND ----------

# Paths — change these to match your DBFS or Unity Catalog setup
RAW_PATH = "dbfs:/crypto_engine/raw"
PROCESSED_PATH = "dbfs:/crypto_engine/processed"
COINS = ["bitcoin", "ethereum", "solana", "cardano", "avalanche-2", "polkadot"]
DAYS = 365

# Tune shuffle partitions to your cluster core count
# Rule: 2-4x the number of total cores across all workers
spark.conf.set("spark.sql.shuffle.partitions", "16")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")

# COMMAND ----------
# MAGIC %md
# MAGIC # 📥 Ingestion

# COMMAND ----------

import requests
import json
import time
from datetime import datetime, timezone

BASE_URL = "https://api.coingecko.com/api/v3"
run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def fetch_and_save_coin(coin_id: str, days: int) -> None:
    """Fetch both OHLC and market data for a coin and save to DBFS."""

    def _get(url, params):
        for attempt in range(4):
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                time.sleep(2 ** attempt * 10)
            else:
                resp.raise_for_status()
        raise RuntimeError(f"Failed: {url}")

    # OHLC
    raw_ohlc = _get(f"{BASE_URL}/coins/{coin_id}/ohlc", {"vs_currency": "usd", "days": days})
    ohlc_records = [{"coin_id": coin_id, "timestamp_ms": r[0], "open": r[1],
                     "high": r[2], "low": r[3], "close": r[4]} for r in raw_ohlc]

    ohlc_path = f"{RAW_PATH}/ohlc/{coin_id}/{run_ts}.json"
    dbutils.fs.put(ohlc_path, json.dumps(ohlc_records), overwrite=True)

    time.sleep(2.5)

    # Market
    raw_market = _get(f"{BASE_URL}/coins/{coin_id}/market_chart",
                      {"vs_currency": "usd", "days": days, "interval": "daily"})
    prices, mcaps, vols = raw_market["prices"], raw_market["market_caps"], raw_market["total_volumes"]
    market_records = [{"coin_id": coin_id, "timestamp_ms": p[0], "price_usd": p[1],
                       "market_cap_usd": mc[1], "total_volume_usd": v[1]}
                      for p, mc, v in zip(prices, mcaps, vols)]

    market_path = f"{RAW_PATH}/market/{coin_id}/{run_ts}.json"
    dbutils.fs.put(market_path, json.dumps(market_records), overwrite=True)

    time.sleep(2.5)
    print(f"✅ {coin_id}: {len(ohlc_records)} OHLC + {len(market_records)} market records")


for coin in COINS:
    fetch_and_save_coin(coin, DAYS)

# COMMAND ----------
# MAGIC %md
# MAGIC # 🥉 Bronze Layer

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

OHLC_SCHEMA = StructType([
    StructField("coin_id", StringType(), False),
    StructField("timestamp_ms", LongType(), False),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
])

MARKET_SCHEMA = StructType([
    StructField("coin_id", StringType(), False),
    StructField("timestamp_ms", LongType(), False),
    StructField("price_usd", DoubleType(), True),
    StructField("market_cap_usd", DoubleType(), True),
    StructField("total_volume_usd", DoubleType(), True),
])


def add_date_columns(df):
    return (df
        .withColumn("event_ts", F.to_timestamp(F.col("timestamp_ms") / 1000))
        .withColumn("event_date", F.to_date("event_ts"))
        .withColumn("event_year", F.year("event_ts"))
        .withColumn("event_month", F.month("event_ts")))


def deduplicate(df, key_cols):
    w = Window.partitionBy(key_cols).orderBy(F.current_timestamp())
    return (df.withColumn("_rn", F.row_number().over(w))
              .filter(F.col("_rn") == 1).drop("_rn"))


# Read + process OHLC
ohlc_df = (spark.read.schema(OHLC_SCHEMA).option("multiLine", "true")
           .json(f"{RAW_PATH}/ohlc/*/*.json")
           .withColumn("_ingested_at", F.current_timestamp()))
ohlc_df = add_date_columns(ohlc_df)
ohlc_df = deduplicate(ohlc_df, ["coin_id", "timestamp_ms"])

(ohlc_df.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("coin_id", "event_date")
 .save(f"{PROCESSED_PATH}/bronze_ohlc"))

# Read + process Market
market_df = (spark.read.schema(MARKET_SCHEMA).option("multiLine", "true")
             .json(f"{RAW_PATH}/market/*/*.json")
             .withColumn("_ingested_at", F.current_timestamp()))
market_df = add_date_columns(market_df)
market_df = deduplicate(market_df, ["coin_id", "timestamp_ms"])

(market_df.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("coin_id", "event_date")
 .save(f"{PROCESSED_PATH}/bronze_market"))

print(f"Bronze OHLC: {ohlc_df.count()} rows")
print(f"Bronze Market: {market_df.count()} rows")

# COMMAND ----------
# MAGIC %md
# MAGIC # 🥈 Silver Layer — Window Functions, UDFs, RSI

# COMMAND ----------

import pandas as pd
from pyspark.sql.types import DoubleType, StringType

# Read Bronze
ohlc_b   = spark.read.format("delta").load(f"{PROCESSED_PATH}/bronze_ohlc")
market_b = spark.read.format("delta").load(f"{PROCESSED_PATH}/bronze_market")

# Join
market_slim = market_b.select("coin_id", "event_date", "market_cap_usd", "total_volume_usd")
joined = ohlc_b.join(market_slim, on=["coin_id", "event_date"], how="inner")


# ── Window Specs ──────────────────────────────────────────────────────────────
def coin_window(days):
    return (Window.partitionBy("coin_id").orderBy("event_date")
            .rowsBetween(-(days - 1), 0))

coin_full = Window.partitionBy("coin_id").orderBy("event_date")
date_rank = Window.partitionBy("event_date").orderBy(F.col("market_cap_usd").desc())


# ── Transformations ───────────────────────────────────────────────────────────
silver = (
    joined
    # Rolling averages
    .withColumn("avg_close_7d",    F.avg("close").over(coin_window(7)))
    .withColumn("avg_close_14d",   F.avg("close").over(coin_window(14)))
    .withColumn("avg_close_30d",   F.avg("close").over(coin_window(30)))
    .withColumn("stddev_close_30d",F.stddev("close").over(coin_window(30)))
    # VWAP
    .withColumn("_pv7", F.sum(F.col("close") * F.col("total_volume_usd")).over(coin_window(7)))
    .withColumn("_v7",  F.sum("total_volume_usd").over(coin_window(7)))
    .withColumn("vwap_7d", F.col("_pv7") / F.col("_v7")).drop("_pv7", "_v7")
    # Momentum
    .withColumn("prev_close_1d", F.lag("close", 1).over(coin_full))
    .withColumn("prev_close_7d", F.lag("close", 7).over(coin_full))
    .withColumn("pct_change_1d",
        (F.col("close") - F.col("prev_close_1d")) / F.col("prev_close_1d") * 100)
    .withColumn("pct_change_7d",
        (F.col("close") - F.col("prev_close_7d")) / F.col("prev_close_7d") * 100)
    .withColumn("high_low_spread_pct",
        (F.col("high") - F.col("low")) / F.col("close") * 100)
    .drop("prev_close_1d", "prev_close_7d")
    # Rankings
    .withColumn("rank_by_mcap", F.rank().over(date_rank))
    .withColumn("dense_rank_by_mcap", F.dense_rank().over(date_rank))
    .withColumn("pct_rank_by_mcap", F.percent_rank().over(date_rank))
)


# ── Pandas UDF: RSI ───────────────────────────────────────────────────────────
@F.pandas_udf(DoubleType())
def compute_rsi_udf(close_series: pd.Series) -> pd.Series:
    period = 14
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50.0)


# ── UDF: Regime Classifier ────────────────────────────────────────────────────
@F.udf(StringType())
def classify_regime(pct_change_7d, high_low_spread_pct, rsi):
    if None in (pct_change_7d, high_low_spread_pct, rsi):
        return "unknown"
    am = abs(pct_change_7d)
    vol = high_low_spread_pct > 5.0
    if am > 10 and rsi > 65:    return "trending_bullish"
    if am > 10 and rsi < 35:    return "trending_bearish"
    if vol and am < 5:          return "ranging_volatile"
    if am < 2 and not vol:      return "ranging_quiet"
    if pct_change_7d < -10 and rsi < 30: return "reversing_oversold"
    if pct_change_7d > 10  and rsi > 70: return "reversing_overbought"
    return "neutral"


silver = silver.sortWithinPartitions("coin_id", "event_date")
silver = silver.withColumn("rsi_14", compute_rsi_udf(F.col("close")))
silver = silver.withColumn("market_regime", classify_regime(
    F.col("pct_change_7d"), F.col("high_low_spread_pct"), F.col("rsi_14")))

(silver.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("coin_id", "event_date")
 .save(f"{PROCESSED_PATH}/silver_enriched"))

print(f"Silver rows: {silver.count()}")
display(silver.limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC # 🔍 Optimization Analysis

# COMMAND ----------

# ── Explain Plan: before vs after broadcast join ──────────────────────────────
silver_df = spark.read.format("delta").load(f"{PROCESSED_PATH}/silver_enriched")

metadata = spark.createDataFrame(
    [("bitcoin","BTC","Layer 1"),("ethereum","ETH","Layer 1"),
     ("solana","SOL","Layer 1"),("cardano","ADA","Layer 1"),
     ("avalanche-2","AVAX","Layer 1"),("polkadot","DOT","Layer 0")],
    ["coin_id", "ticker", "category"]
)

print("=== SortMergeJoin Plan (no broadcast) ===")
silver_df.join(metadata, on="coin_id").explain(mode="formatted")

print("\n=== BroadcastHashJoin Plan (with broadcast) ===")
silver_df.join(F.broadcast(metadata), on="coin_id").explain(mode="formatted")

# COMMAND ----------

# ── Partition Pruning Check ───────────────────────────────────────────────────
print("=== Partition Pruning: filter on coin_id ===")
silver_df.filter(F.col("coin_id") == "bitcoin").explain(mode="formatted")
# Look for PartitionFilters in the output

# COMMAND ----------

# ── Skew Detection ────────────────────────────────────────────────────────────
print("=== Row count per coin (skew check) ===")
display(silver_df.groupBy("coin_id").count().orderBy("count", ascending=False))

# COMMAND ----------
# MAGIC %md
# MAGIC # 🥇 Gold Layer

# COMMAND ----------

import math

silver_df = spark.read.format("delta").load(f"{PROCESSED_PATH}/silver_enriched")
silver_df.cache()
silver_df.count()  # materialize

metadata_df = spark.createDataFrame(
    [("bitcoin","BTC","Layer 1",2009),("ethereum","ETH","Layer 1",2015),
     ("solana","SOL","Layer 1",2020),("cardano","ADA","Layer 1",2017),
     ("avalanche-2","AVAX","Layer 1",2020),("polkadot","DOT","Layer 0",2020)],
    ["coin_id","ticker","category","launch_year"]
)

# Daily Summary with Broadcast Join
daily_summary = (silver_df
    .join(F.broadcast(metadata_df), on="coin_id", how="left")
    .select("coin_id","ticker","category","event_date","open","high","low","close",
            "total_volume_usd","market_cap_usd","avg_close_7d","avg_close_30d",
            "vwap_7d","pct_change_1d","pct_change_7d","rsi_14","market_regime",
            "rank_by_mcap"))

(daily_summary.write.mode("overwrite")
 .parquet(f"{PROCESSED_PATH}/gold_daily_summary"))

# Volatility + Bollinger Bands
BBPERIOD = 20
w20 = Window.partitionBy("coin_id").orderBy("event_date").rowsBetween(-BBPERIOD+1, 0)
w30 = Window.partitionBy("coin_id").orderBy("event_date").rowsBetween(-29, 0)

volatility = (silver_df
    .withColumn("annualized_vol_pct", F.stddev("pct_change_1d").over(w30) * math.sqrt(365))
    .withColumn("bb_sma_20",    F.avg("close").over(w20))
    .withColumn("bb_stddev_20", F.stddev("close").over(w20))
    .withColumn("bb_upper",  F.col("bb_sma_20") + 2 * F.col("bb_stddev_20"))
    .withColumn("bb_lower",  F.col("bb_sma_20") - 2 * F.col("bb_stddev_20"))
    .withColumn("bb_pct_b",  (F.col("close") - F.col("bb_lower")) /
                             (F.col("bb_upper") - F.col("bb_lower")))
    .withColumn("bb_bandwidth", (F.col("bb_upper") - F.col("bb_lower")) / F.col("bb_sma_20"))
    .select("coin_id","event_date","close","annualized_vol_pct",
            "bb_sma_20","bb_upper","bb_lower","bb_pct_b","bb_bandwidth","rsi_14","market_regime"))

(volatility.write.mode("overwrite")
 .parquet(f"{PROCESSED_PATH}/gold_volatility_report"))

# Regime Summary
regime_summary = (silver_df
    .filter(F.col("market_regime") != "unknown")
    .groupBy("coin_id", "market_regime")
    .agg(
        F.count("*").alias("days_in_regime"),
        F.avg("pct_change_1d").alias("avg_daily_return_pct"),
        F.avg("rsi_14").alias("avg_rsi"),
    ))

(regime_summary.write.mode("overwrite")
 .parquet(f"{PROCESSED_PATH}/gold_regime_summary"))

silver_df.unpersist()

print("Gold tables written:")
print(f"  daily_summary:    {spark.read.parquet(f'{PROCESSED_PATH}/gold_daily_summary').count()} rows")
print(f"  volatility:       {spark.read.parquet(f'{PROCESSED_PATH}/gold_volatility_report').count()} rows")
print(f"  regime_summary:   {spark.read.parquet(f'{PROCESSED_PATH}/gold_regime_summary').count()} rows")

# COMMAND ----------
# MAGIC %md
# MAGIC # 📊 Quick Analytics

# COMMAND ----------

# Top 5 highest volatility days per coin
vol_df = spark.read.parquet(f"{PROCESSED_PATH}/gold_volatility_report")
display(
    vol_df.filter(F.col("annualized_vol_pct").isNotNull())
    .withColumn("rn", F.row_number().over(
        Window.partitionBy("coin_id").orderBy(F.col("annualized_vol_pct").desc())))
    .filter(F.col("rn") <= 5)
    .drop("rn")
    .orderBy("coin_id", F.col("annualized_vol_pct").desc())
)

# COMMAND ----------

# Regime frequency per coin
reg_df = spark.read.parquet(f"{PROCESSED_PATH}/gold_regime_summary")
display(reg_df.orderBy("coin_id", F.col("days_in_regime").desc()))

# COMMAND ----------
# MAGIC %md
# MAGIC # ✅ Pipeline Complete
# MAGIC
# MAGIC All Gold tables available at:
# MAGIC - `dbfs:/crypto_engine/processed/gold_daily_summary`
# MAGIC - `dbfs:/crypto_engine/processed/gold_volatility_report`
# MAGIC - `dbfs:/crypto_engine/processed/gold_regime_summary`
