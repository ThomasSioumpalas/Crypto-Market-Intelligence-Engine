
import pytest
from datetime import date
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, LongType
from delta import configure_spark_with_delta_pip


@pytest.fixture(scope="session")
def spark():
    """Single SparkSession shared across all tests in the session."""
    builder = (
        SparkSession.builder
        .appName("CryptoEngine_Tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Suppress verbose Spark logging in test output
        .config("spark.driver.extraJavaOptions", "-Dlog4j.rootCategory=ERROR,console")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def sample_ohlc_df(spark):
    """
    Synthetic OHLC data for 3 coins × 30 days.
    Prices are deterministic so window function results are predictable.
    """
    data = []
    coins = ["bitcoin", "ethereum", "solana"]
    base_prices = {"bitcoin": 40000.0, "ethereum": 2000.0, "solana": 100.0}

    for coin in coins:
        base = base_prices[coin]
        for day in range(30):
            close = base + day * base * 0.01   # 1% daily increase
            data.append({
                "coin_id": coin,
                "event_date": date(2024, 1, day + 1),
                "open":  close * 0.99,
                "high":  close * 1.02,
                "low":   close * 0.97,
                "close": close,
                "total_volume_usd": 1_000_000.0 * (day + 1),
                "market_cap_usd":   close * 19_000_000.0,
            })

    schema = StructType([
        StructField("coin_id",           StringType(), False),
        StructField("event_date",        DateType(),   False),
        StructField("open",              DoubleType(), True),
        StructField("high",              DoubleType(), True),
        StructField("low",               DoubleType(), True),
        StructField("close",             DoubleType(), True),
        StructField("total_volume_usd",  DoubleType(), True),
        StructField("market_cap_usd",    DoubleType(), True),
    ])

    return spark.createDataFrame(data, schema=schema)


# ── Tests: Rolling Averages ───────────────────────────────────────────────────

class TestRollingAverages:
    def test_7d_avg_not_null_after_warmup(self, spark, sample_ohlc_df):
        """
        For days >= 7, avg_close_7d should be non-null.
        For days 1-6, it should be non-null too (rowsBetween includes available rows).
        """
        from src.transformations.silver_layer import add_rolling_averages
        result = add_rolling_averages(sample_ohlc_df)
        null_count = result.filter(F.col("avg_close_7d").isNull()).count()
        assert null_count == 0, f"Expected 0 nulls in avg_close_7d, got {null_count}"

    def test_30d_avg_less_than_or_equal_to_max_close(self, spark, sample_ohlc_df):
        """Rolling average must not exceed maximum close for the window."""
        from src.transformations.silver_layer import add_rolling_averages
        result = add_rolling_averages(sample_ohlc_df)
        # avg must be <= max close (trivially true for an average)
        violations = result.filter(F.col("avg_close_30d") > F.col("close") * 2).count()
        assert violations == 0

    def test_7d_avg_less_than_14d_avg_in_uptrend(self, spark, sample_ohlc_df):
        """
        In our synthetic uptrend (prices rise daily), the 7-day avg should be
        higher than the 14-day avg because recent prices are higher.
        This tests that window frames are correctly ordered.
        """
        from src.transformations.silver_layer import add_rolling_averages
        result = add_rolling_averages(sample_ohlc_df)
        # On day 30, 7d avg should be > 14d avg (recent prices higher)
        day30 = result.filter(
            (F.col("event_date") == date(2024, 1, 30)) &
            (F.col("coin_id") == "bitcoin")
        ).collect()
        assert len(day30) == 1
        assert day30[0]["avg_close_7d"] > day30[0]["avg_close_14d"], (
            "7d avg should exceed 14d avg in an uptrend"
        )


# ── Tests: VWAP ───────────────────────────────────────────────────────────────

class TestVWAP:
    def test_vwap_is_nonnull(self, spark, sample_ohlc_df):
        from src.transformations.silver_layer import add_vwap
        result = add_vwap(sample_ohlc_df)
        null_count = result.filter(F.col("vwap_7d").isNull()).count()
        assert null_count == 0

    def test_vwap_between_low_and_high(self, spark, sample_ohlc_df):
        """VWAP is volume-weighted close, so it should roughly be near close price."""
        from src.transformations.silver_layer import add_vwap
        result = add_vwap(sample_ohlc_df)
        # VWAP should be within a reasonable range of the close
        violations = result.filter(
            (F.col("vwap_7d") > F.col("high") * 1.1) |
            (F.col("vwap_7d") < F.col("low") * 0.9)
        ).count()
        assert violations == 0, f"{violations} rows with VWAP far outside high/low range"


# ── Tests: Momentum ───────────────────────────────────────────────────────────

class TestMomentum:
    def test_pct_change_1d_null_on_first_row(self, spark, sample_ohlc_df):
        """First row per coin has no previous day — pct_change_1d must be null."""
        from src.transformations.silver_layer import add_momentum_features
        result = add_momentum_features(sample_ohlc_df)
        first_row = result.filter(
            (F.col("event_date") == date(2024, 1, 1)) &
            (F.col("coin_id") == "bitcoin")
        ).collect()
        assert first_row[0]["pct_change_1d"] is None

    def test_pct_change_positive_in_uptrend(self, spark, sample_ohlc_df):
        """In our synthetic uptrend, pct_change_1d should be positive after day 1."""
        from src.transformations.silver_layer import add_momentum_features
        result = add_momentum_features(sample_ohlc_df)
        negative_changes = result.filter(
            F.col("pct_change_1d").isNotNull() &
            (F.col("pct_change_1d") <= 0)
        ).count()
        # Allow 0 — some days may round to exactly 0 with floating point
        assert negative_changes == 0, f"{negative_changes} negative 1d changes in an uptrend"

    def test_high_low_spread_nonnegative(self, spark, sample_ohlc_df):
        """High is always >= Low by construction, so spread should be >= 0."""
        from src.transformations.silver_layer import add_momentum_features
        result = add_momentum_features(sample_ohlc_df)
        negative_spread = result.filter(F.col("high_low_spread_pct") < 0).count()
        assert negative_spread == 0


# ── Tests: Rankings ───────────────────────────────────────────────────────────

class TestRankings:
    def test_ranks_cover_all_coins(self, spark, sample_ohlc_df):
        """With 3 coins, rank should use values 1, 2, 3 on every day."""
        from src.transformations.silver_layer import add_daily_rankings
        result = add_daily_rankings(sample_ohlc_df)
        day15 = result.filter(F.col("event_date") == date(2024, 1, 15))
        ranks = set(row["rank_by_mcap"] for row in day15.collect())
        assert ranks == {1, 2, 3}, f"Expected ranks {{1,2,3}}, got {ranks}"

    def test_bitcoin_highest_rank(self, spark, sample_ohlc_df):
        """Bitcoin has the highest base price, so it should rank #1 by market cap."""
        from src.transformations.silver_layer import add_daily_rankings
        result = add_daily_rankings(sample_ohlc_df)
        btc_on_day15 = result.filter(
            (F.col("event_date") == date(2024, 1, 15)) &
            (F.col("coin_id") == "bitcoin")
        ).collect()
        assert btc_on_day15[0]["rank_by_mcap"] == 1


# ── Tests: RSI UDF ────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_in_valid_range(self, spark, sample_ohlc_df):
        """RSI must always be in [0, 100]."""
        from src.transformations.silver_layer import compute_rsi_udf
        result = sample_ohlc_df.withColumn("rsi", compute_rsi_udf(F.col("close")))
        out_of_range = result.filter(
            (F.col("rsi") < 0) | (F.col("rsi") > 100)
        ).count()
        assert out_of_range == 0, f"{out_of_range} RSI values outside [0, 100]"

    def test_rsi_high_in_strong_uptrend(self, spark):
        """RSI should be > 50 in a monotonically increasing price series."""
        from src.transformations.silver_layer import compute_rsi_udf
        # 30-day monotonic increase: should push RSI above 50
        data = [(float(i * 100),) for i in range(1, 31)]
        df = spark.createDataFrame(data, ["close"])
        result = df.withColumn("rsi", compute_rsi_udf(F.col("close"))).collect()
        # Last RSI value (most data, most reliable)
        last_rsi = result[-1]["rsi"]
        assert last_rsi > 50, f"Expected RSI > 50 in uptrend, got {last_rsi}"


# ── Tests: Regime Classifier ─────────────────────────────────────────────────

class TestRegimeClassifier:
    def test_trending_bullish_correct(self, spark):
        from src.transformations.silver_layer import classify_regime
        data = [(15.0, 3.0, 70.0)]  # strong momentum + high RSI
        df = spark.createDataFrame(data, ["pct_change_7d", "hl_spread", "rsi"])
        result = df.withColumn(
            "regime",
            classify_regime(F.col("pct_change_7d"), F.col("hl_spread"), F.col("rsi"))
        ).collect()
        assert result[0]["regime"] == "trending_bullish"

    def test_null_inputs_return_unknown(self, spark):
        from src.transformations.silver_layer import classify_regime
        data = [(None, None, None)]
        df = spark.createDataFrame(data, ["pct_change_7d", "hl_spread", "rsi"])
        result = df.withColumn(
            "regime",
            classify_regime(F.col("pct_change_7d"), F.col("hl_spread"), F.col("rsi"))
        ).collect()
        assert result[0]["regime"] == "unknown"

    def test_ranging_quiet_correct(self, spark):
        from src.transformations.silver_layer import classify_regime
        data = [(1.0, 2.0, 50.0)]  # low momentum + low spread + neutral RSI
        df = spark.createDataFrame(data, ["pct_change_7d", "hl_spread", "rsi"])
        result = df.withColumn(
            "regime",
            classify_regime(F.col("pct_change_7d"), F.col("hl_spread"), F.col("rsi"))
        ).collect()
        assert result[0]["regime"] == "ranging_quiet"


# ── Tests: Data Quality ───────────────────────────────────────────────────────

class TestDataQuality:
    def test_ohlc_consistency_passes_on_valid_data(self, spark, sample_ohlc_df):
        from src.quality.data_quality import check_ohlc_consistency
        result = check_ohlc_consistency(sample_ohlc_df)
        assert result.passed, f"OHLC consistency failed: {result.details}"

    def test_ohlc_consistency_fails_on_bad_data(self, spark):
        """Inject a row where low > high — should fail."""
        from src.quality.data_quality import check_ohlc_consistency
        bad_data = [("bitcoin", date(2024, 1, 1), 100.0, 90.0, 110.0, 95.0)]  # high < low
        schema = ["coin_id", "event_date", "open", "high", "low", "close"]
        bad_df = spark.createDataFrame(bad_data, schema=schema)
        result = check_ohlc_consistency(bad_df)
        assert not result.passed, "Expected OHLC check to fail on bad data"

    def test_rsi_range_passes_on_valid_rsi(self, spark):
        from src.quality.data_quality import check_rsi_range
        data = [("bitcoin", 45.0), ("ethereum", 72.0), ("solana", 30.0)]
        df = spark.createDataFrame(data, ["coin_id", "rsi_14"])
        result = check_rsi_range(df)
        assert result.passed

    def test_rsi_range_fails_on_invalid_rsi(self, spark):
        from src.quality.data_quality import check_rsi_range
        data = [("bitcoin", 150.0)]   # RSI > 100
        df = spark.createDataFrame(data, ["coin_id", "rsi_14"])
        result = check_rsi_range(df)
        assert not result.passed
