import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ingest")

BASE_URL = "https://api.coingecko.com/api/v3"
RAW_DIR = Path("data/raw")

DEFAULT_COINS = [
    "bitcoin", "ethereum", "solana", "cardano",
    "avalanche-2", "polkadot", "chainlink", "litecoin",
]

REQUEST_DELAY_SECONDS = 2.5
MAX_RETRIES = 4


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _get(url: str, params: dict = None) -> dict:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = 2 ** attempt * 10  # 10s, 20s, 40s, 80s
            log.warning(f"Rate limited. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}")
            time.sleep(wait)
        else:
            resp.raise_for_status()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


# ── Fetch functions ───────────────────────────────────────────────────────────
def fetch_ohlc(coin_id: str, days: int) -> list[dict]:
    
    url = f"{BASE_URL}/coins/{coin_id}/ohlc"
    raw = _get(url, params={"vs_currency": "usd", "days": str(days)})

    return [
        {
            "coin_id": coin_id,
            "timestamp_ms": row[0],
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
        }
        for row in raw
    ]


def fetch_market_chart(coin_id: str, days: int) -> dict:
   
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    raw = _get(url, params={"vs_currency": "usd", "days": str(days), "interval": "daily"})

    prices = raw.get("prices", [])
    market_caps = raw.get("market_caps", [])
    volumes = raw.get("total_volumes", [])

    # Defensive: zip stops at shortest — warn if lengths differ
    if not (len(prices) == len(market_caps) == len(volumes)):
        log.warning(
            f"{coin_id}: mismatched array lengths "
            f"prices={len(prices)} market_caps={len(market_caps)} volumes={len(volumes)}"
        )

    return [
        {
            "coin_id": coin_id,
            "timestamp_ms": p[0],
            "price_usd": p[1],
            "market_cap_usd": mc[1],
            "total_volume_usd": v[1],
        }
        for p, mc, v in zip(prices, market_caps, volumes)
    ]


# ── Write helpers ─────────────────────────────────────────────────────────────
def write_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    log.info(f"Written {len(data)} records → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def ingest(coins: list[str], days: int) -> None:
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log.info(f"Starting ingestion run {run_ts} | coins={coins} | days={days}")

    for coin in coins:
        log.info(f"Processing: {coin}")
        try:
            ohlc_data = fetch_ohlc(coin, days)
            time.sleep(REQUEST_DELAY_SECONDS)

            market_data = fetch_market_chart(coin, days)
            time.sleep(REQUEST_DELAY_SECONDS)

            # Write raw — one file per coin per run. We keep all runs so
            # Bronze ingestion can detect and deduplicate across runs.
            write_json(ohlc_data, RAW_DIR / f"ohlc/{coin}/{run_ts}.json")
            write_json(market_data, RAW_DIR / f"market/{coin}/{run_ts}.json")

        except Exception as e:
            log.error(f"Failed for {coin}: {e}")
            continue  # Don't abort entire run for one bad coin

    log.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoinGecko ingestion")
    parser.add_argument(
        "--coins",
        default=",".join(DEFAULT_COINS),
        help="Comma-separated CoinGecko coin IDs",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of historical days to fetch (>= 90 for daily granularity)",
    )
    args = parser.parse_args()
    ingest(coins=args.coins.split(","), days=args.days)
