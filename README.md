# Crypto Market Intelligence Engine

A production-grade PySpark data pipeline that ingests live cryptocurrency market data from the CoinGecko API and processes it through a full **Bronze → Silver → Gold** medallion architecture, producing analytics-ready tables enriched with technical indicators, market regime classification, and query optimization benchmarks.

---

## What It Does

Raw OHLCV data for any set of coins is fetched, validated, enriched with window-based features (moving averages, VWAP, RSI via Pandas UDF, momentum, volatility), classified into market regimes, and aggregated into Gold-layer analytics tables — all orchestrated through a single CLI command.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Processing | PySpark 3.5.1 (local + cluster mode) |
| Storage | Delta Lake 3.2.0 + Parquet |
| Ingestion | CoinGecko REST API |
| Orchestration | Python CLI (`run_pipeline.py`) |
| Infrastructure | Docker Compose (Spark master, worker, Jupyter) |
| Quality | Custom data quality checks (null rates, OHLC invariants, RSI range) |
| UDFs | Pandas UDF (RSI), Python UDF (regime classification) |

---

## Architecture

```
CoinGecko API
      │
      ▼
[ Ingestion ]  →  data/raw/ohlc/{coin}/  +  data/raw/market/{coin}/
      │
      ▼
[ Bronze Layer ]  →  Delta: bronze_ohlc, bronze_market
      │  Schema enforcement · Deduplication · Quarantine
      ▼
[ Quality Checks ]
      │
      ▼
[ Silver Layer ]  →  Delta: silver_enriched
      │  MA 7/14/30d · VWAP · Volatility · Momentum
      │  RSI-14 (Pandas UDF) · Regime classifier (Python UDF)
      │  Daily rankings by market cap
      ▼
[ Gold Layer ]  →  Parquet: 4 analytics tables
      │  gold_daily_summary · gold_weekly_rankings
      │  gold_volatility_report · gold_regime_summary
      ▼
[ Optimization Report ]
      AQE · Cache benchmark · Skew detection · Partition analysis
```

---

## Quick Start

```bash
# 1. Start the cluster
cd docker && docker-compose up -d

# 2. Open Jupyter at http://localhost:8888
# 3. Open Spark Master UI at http://localhost:8080

# 4. Run the full pipeline (inside Jupyter terminal)
cd /home/jovyan/project/crypto_spark_engine
python run_pipeline.py --coins bitcoin,ethereum,solana --days 365

# 5. Skip ingestion on reruns
python run_pipeline.py --coins bitcoin,ethereum,solana --days 365 --skip-ingest

# 6. Run specific layers only
python run_pipeline.py --skip-ingest --layers silver,gold,quality
```

---

## PySpark Concepts Demonstrated

- **Window functions** — rolling averages, VWAP, lag-based momentum, daily rankings
- **Pandas UDF** (Arrow-based) — vectorized RSI-14 calculation over partitioned time series
- **Python UDF** — market regime classification (bullish / bearish / overbought / oversold / neutral)
- **Delta Lake** — ACID writes, schema enforcement, partition pruning
- **Adaptive Query Execution (AQE)** — enabled for shuffle partition coalescing and skew join handling
- **Broadcast join** — OHLC (small) broadcast against market data
- **Data quality** — null rate checks, OHLC invariant validation, anomaly detection
- **Medallion architecture** — Bronze / Silver / Gold separation of concerns

---

## Project Structure

```
crypto_spark_engine/
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile.jupyter
├── src/
│   ├── ingestion/
│   │   └── coingecko_ingest.py
│   ├── transformations/
│   │   ├── bronze_layer.py
│   │   ├── silver_layer.py
│   │   └── gold_layer.py
│   ├── quality/
│   │   └── data_quality.py
│   └── optimization/
│       └── query_optimization.py
├── tests/
│   └── test_transformations.py
├── run_pipeline.py
└── requirements.txt
```
