# Crypto Market Intelligence Engine
### Advanced PySpark Portfolio Project

A data engineering project where **PySpark is the star** — window functions, vectorized Pandas UDFs, query optimization, and data quality checks over real crypto market data from the CoinGecko API.

---

## Project Structure

```
crypto_spark_engine/
├── docker/
│   └── docker-compose.yml          # Spark cluster + Jupyter Lab
├── src/
│   ├── ingestion/
│   │   └── coingecko_ingest.py     # API ingestion with rate-limit back-off
│   ├── transformations/
│   │   ├── bronze_layer.py         # Raw JSON → Delta (dedup, schema, quarantine)
│   │   ├── silver_layer.py         # Window functions, UDFs, RSI, rankings
│   │   └── gold_layer.py           # Aggregated analytics tables + broadcast join
│   ├── quality/
│   │   └── data_quality.py         # Null rates, OHLC invariants, anomaly detection
│   └── optimization/
│       └── query_optimization.py   # Shuffle analysis, cache benchmarks, skew detection
├── notebooks/
│   └── crypto_engine_databricks.py # Full pipeline as a Databricks notebook
├── tests/
│   └── test_transformations.py     # Unit tests (synthetic data, no API calls)
├── run_pipeline.py                 # CLI entry point for the full pipeline
└── requirements.txt
```

---

## Quick Start

### Local (Docker)

```bash
# 1. Start Spark cluster + Jupyter
cd docker && docker-compose up -d

# 2. Install dependencies (if running outside Docker)
pip install -r requirements.txt

# 3. Run the full pipeline
python run_pipeline.py --coins bitcoin,ethereum,solana --days 365

# 4. Run specific layers only
python run_pipeline.py --layers bronze,silver,gold --skip-ingest

# 5. Run tests
pytest tests/ -v
```

Access points:
- **Jupyter Lab**: http://localhost:8888
- **Spark Master UI**: http://localhost:8080
- **Spark App UI**: http://localhost:4040 (while a job is running)

### Databricks

1. Import `notebooks/crypto_engine_databricks.py` as a Python notebook.
2. Attach to a cluster running **DBR 14.x LTS** (Python 3.10+).
3. Update `RAW_PATH` and `PROCESSED_PATH` to your DBFS or Unity Catalog paths.
4. Run all cells top-to-bottom.

---

## Architecture — Medallion Pattern

```
CoinGecko API
      │
      ▼
[ Ingestion ]  ──→  data/raw/ohlc/{coin}/{run_ts}.json
                     data/raw/market/{coin}/{run_ts}.json
      │
      ▼
[ Bronze Layer ]  ──→  bronze_ohlc  (Delta)
                         bronze_market  (Delta)
      │  Schema enforcement, deduplication, quarantine, date columns
      ▼
[ Quality Checks ] ──→  quality_report (Parquet, append-mode)
      │
      ▼
[ Silver Layer ]  ──→  silver_enriched  (Delta)
      │  Join, rolling averages, VWAP, momentum, RSI (Pandas UDF),
      │  regime classifier (UDF), daily rankings
      ▼
[ Quality Checks ] ──→  quality_report (same table, second entry)
      │
      ▼
[ Gold Layer ]  ──→  gold_daily_summary      (Parquet)
                      gold_weekly_rankings    (Parquet)
                      gold_volatility_report  (Parquet)
                      gold_regime_summary     (Parquet)
```

---

## PySpark Techniques — Detailed Notes

### 1. Window Functions

All window specs are defined once and reused:

```python
Window.partitionBy("coin_id").orderBy("event_date").rowsBetween(-(N-1), 0)
```

**`rowsBetween` vs `rangeBetween`**: We use `rowsBetween` throughout. `rangeBetween` operates on the actual column values (requires numeric order key) and breaks with date gaps. `rowsBetween` uses physical row offsets and is safe for irregular time series.

**Window operations that trigger shuffles**: Every `Window.partitionBy("coin_id")` triggers one Exchange (shuffle) in Spark's physical plan. This is unavoidable — data must be co-located by coin for the window to be correct. The goal is to minimize *unnecessary* shuffles, not eliminate all of them.

### 2. VWAP — Ratio of Two Rolling Sums

```python
vwap_7d = Σ(close × volume) / Σ(volume)  [7-day window]
```

Computed as two `F.sum()` calls over the same window spec — efficient because Spark optimizes repeated window scans over identical specs into a single pass.

### 3. Pandas UDFs (Vectorized) vs Row-Level UDFs

| | Row-Level UDF | Pandas UDF |
|---|---|---|
| Serialization | Python ↔ JVM per row | Apache Arrow (batch) |
| Speed on numeric ops | Slow | ~10–100x faster |
| Input type | Individual Python values | `pd.Series` |
| Best for | String manipulation, simple conditionals | Numeric sequences (RSI, rolling stats) |

**RSI** is implemented as a Pandas UDF because:
1. It requires a sequence of values (not just current row).
2. The `ewm()` (exponential weighted mean) operation in pandas is highly optimized.
3. Arrow serialization makes batch transfer to Python cheap.

**Regime classifier** is a row-level UDF because:
1. It only needs the current row's values.
2. Logic is simple (conditionals) — no numeric computation.
3. The performance difference vs built-ins is acceptable for a string output.

### 4. Broadcast Join

```python
silver_df.join(F.broadcast(metadata_df), on="coin_id", how="left")
```

`metadata_df` has 8 rows (~1KB). Without `F.broadcast()`, Spark would use a SortMergeJoin, shuffling the entire `silver_df` (potentially millions of rows) to co-locate with metadata. With broadcast, the small table is copied to every executor — no shuffle of the large table.

**Auto-broadcast threshold**: `spark.sql.autoBroadcastJoinThreshold = 10MB` (default). With AQE enabled, Spark can auto-broadcast even without the hint. Explicit `F.broadcast()` makes the intent clear and works regardless of AQE settings.

### 5. Caching Strategy

```python
silver_df.persist(StorageLevel.MEMORY_AND_DISK)
silver_df.count()   # materialize — triggers actual computation
# ... build 4 Gold tables from silver_df ...
silver_df.unpersist()
```

**When to cache**: When the same DataFrame is read by multiple downstream actions. Without caching, each Gold table would re-read the Delta file + re-run all Silver transformations.

**MEMORY_AND_DISK vs MEMORY_ONLY**: If the dataset overflows RAM, MEMORY_ONLY evicts partitions (recomputed on next access — correct but slow). MEMORY_AND_DISK spills to disk instead — slower than memory but avoids recomputation.

**Always unpersist**: Leaving DataFrames in cache wastes memory for subsequent jobs. Spark does not automatically evict cached DataFrames when they go out of scope.

### 6. Partition Strategy

Silver is partitioned by `(coin_id, event_date)`.

- `coin_id` first: most queries filter on one coin — partition pruning eliminates all other coins' files.
- `event_date` second: time-range queries benefit from date pruning within a coin.
- **Over-partitioning risk**: 10 coins × 365 days = 3,650 partitions. Each partition = minimum 1 file. Too many small files hurts read performance (small file problem). At scale, consider `(coin_id, event_year, event_month)` instead.

### 7. Shuffle Partition Tuning

Default: `spark.sql.shuffle.partitions = 200`. For our dataset (~36,500 rows), 200 shuffle partitions means most are empty or tiny — huge overhead.

We set it to **8** locally (matches available cores). On Databricks, set to **2–4× total core count** across all workers.

**With AQE** (`spark.sql.adaptive.coalescePartitions.enabled = true`): Spark automatically coalesces small shuffle partitions at runtime. This makes the shuffle partition setting less critical but still worth tuning as a starting point.

---

## Query Plan Reading Guide

Run `df.explain(mode="formatted")` to inspect the physical plan. Key nodes:

| Node | Meaning | Good/Bad |
|---|---|---|
| `BroadcastHashJoin` | Small table broadcast to all executors | ✅ Good |
| `SortMergeJoin` | Both sides sorted + merged | ⚠️ Acceptable for large tables |
| `Exchange (HashPartitioning)` | Shuffle — data moved over network | ⚠️ Minimize unnecessary ones |
| `Filter` pushed below `Exchange` | Predicate pushed down before shuffle | ✅ Good |
| `PartitionFilters` | Partition pruning active | ✅ Good |
| `InMemoryTableScan` | Reading from cache | ✅ Good (if intentional) |

---

## Data Quality Framework

Quality checks run twice: after Bronze (to catch raw data issues) and after Silver (to validate derived columns). Results are appended to `quality_report` Parquet — queryable over time.

Checks:
- **Null rate per column**: configurable threshold (default 5%)
- **RSI range**: fail if any RSI outside [0, 100] — indicates UDF bug
- **OHLC consistency**: `low <= open, close <= high` — API corruption check
- **Price anomaly**: `close > 10× avg_close_30d` — API error detection
- **Zero volume**: rate > 2% is suspicious for liquid coins
- **Regime distribution**: warn if one regime > 80% of days (classifier calibration issue)

---

## Databricks vs Local Differences

| | Local (Docker) | Databricks |
|---|---|---|
| SparkSession | Must create manually | Provided as `spark` |
| Delta Lake | Requires config extensions | Built-in |
| Storage | Local filesystem | DBFS or Unity Catalog |
| Shuffle partitions | 8 (matches local cores) | 16+ (tune to cluster) |
| AQE | Optional | Recommended (enabled by default in DBR 11+) |
| File save | `data/processed/...` | `dbfs:/crypto_engine/...` |

The core PySpark code is identical between environments — only the session config and paths differ.
