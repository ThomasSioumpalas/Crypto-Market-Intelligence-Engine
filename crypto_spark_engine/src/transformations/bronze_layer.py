from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from delta import configure_spark_with_delta_pip


OHLC_SCHEMA = StructType([
    StructField("coin_id", StringType(), nullable=False),
    StructField("timestamp_ms", LongType(), nullable=False),   # epoch millis
    StructField("open", DoubleType(), nullable=True),
    StructField("high", DoubleType(), nullable=True),
    StructField("low", DoubleType(), nullable=True),
    StructField("close", DoubleType(), nullable=True),
])

MARKET_SCHEMA = StructType([
    StructField("coin_id", StringType(), nullable=False),
    StructField("timestamp_ms", LongType(), nullable=False),
    StructField("price_usd", DoubleType(), nullable=True),
    StructField("market_cap_usd", DoubleType(), nullable=True),
    StructField("total_volume_usd", DoubleType(), nullable=True),
])



def get_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("CryptoEngine_Bronze")
        .master("local[*]")
        # Delta Lake extensions — required for ACID writes and time travel
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Shuffle partitions: default 200 is overkill for our dataset size.
        # At ~10 coins × 365 days = ~3,650 rows, 8 partitions is sufficient.
        .config("spark.sql.shuffle.partitions", "8")
    )

    return configure_spark_with_delta_pip(builder).getOrCreate()


def read_raw_ohlc(spark: SparkSession, raw_path: str) -> "DataFrame":
 
    df = (
        spark.read
        .schema(OHLC_SCHEMA)
        .option("multiLine", "true")
        .json(f"{raw_path}/ohlc/*/*.json")
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_ingested_at", F.current_timestamp())
    )
    return df


def read_raw_market(spark: SparkSession, raw_path: str) -> "DataFrame":
    df = (
        spark.read
        .schema(MARKET_SCHEMA)
        .option("multiLine", "true")
        .json(f"{raw_path}/market/*/*.json")
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_ingested_at", F.current_timestamp())
    )
    return df


def add_date_columns(df: "DataFrame") -> "DataFrame":
    return (
        df
        .withColumn(
            "event_ts",
            F.to_timestamp(F.col("timestamp_ms") / 1000)  # ms → seconds → timestamp
        )
        .withColumn("event_date", F.to_date(F.col("event_ts")))
        .withColumn("event_year", F.year(F.col("event_ts")))
        .withColumn("event_month", F.month(F.col("event_ts")))
    )


def quarantine_bad_rows(df: "DataFrame", output_path: str) -> "DataFrame":

    is_bad = (
        F.col("coin_id").isNull() |
        F.col("timestamp_ms").isNull() |
        F.col("event_ts").isNull()
    )

    bad_df = df.filter(is_bad).withColumn("_quarantine_reason", F.lit("null_key_field"))
    good_df = df.filter(~is_bad)

    bad_count = bad_df.count()
    if bad_count > 0:
        print(f"[WARN] Quarantining {bad_count} bad rows → {output_path}/quarantine/")
        (
            bad_df.write
            .mode("append")
            .parquet(f"{output_path}/quarantine/")
        )

    return good_df


def deduplicate(df: "DataFrame", key_cols: list[str]) -> "DataFrame":

    from pyspark.sql.window import Window

    window = Window.partitionBy(key_cols).orderBy(F.col("_ingested_at").desc())
    return (
        df
        .withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def write_bronze(df: "DataFrame", output_path: str, table_name: str) -> None:

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")   # allow schema evolution safely
        .partitionBy("coin_id", "event_date")
        .save(f"{output_path}/{table_name}")
    )
    print(f"[Bronze] Written {table_name} → {output_path}/{table_name}")


def run_bronze(raw_path: str = "data/raw", output_path: str = "data/processed") -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("BRONZE LAYER — Starting")
    print("=" * 60)

    # ── OHLC ──────────────────────────────────────────────────────
    ohlc_df = read_raw_ohlc(spark, raw_path)
    ohlc_df = add_date_columns(ohlc_df)
    ohlc_df = quarantine_bad_rows(ohlc_df, output_path)
    ohlc_df = deduplicate(ohlc_df, key_cols=["coin_id", "timestamp_ms"])
    write_bronze(ohlc_df, output_path, "bronze_ohlc")

    print(f"[Bronze] OHLC row count: {ohlc_df.count()}")

    # ── Market ─────────────────────────────────────────────────────
    market_df = read_raw_market(spark, raw_path)
    market_df = add_date_columns(market_df)
    market_df = quarantine_bad_rows(market_df, output_path)
    market_df = deduplicate(market_df, key_cols=["coin_id", "timestamp_ms"])
    write_bronze(market_df, output_path, "bronze_market")

    print(f"[Bronze] Market row count: {market_df.count()}")
    print("=" * 60)
    print("BRONZE LAYER — Complete")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    run_bronze()
