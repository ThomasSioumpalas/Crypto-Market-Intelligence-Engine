import argparse
import logging
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pipeline")


def run_layer(name: str, fn, *args, **kwargs) -> bool:
    """Run a single pipeline layer, timing it and catching errors."""
    log.info(f"{'='*50}")
    log.info(f"Starting layer: {name}")
    log.info(f"{'='*50}")
    start = time.perf_counter()
    try:
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log.info(f"✅  {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - start
        log.error(f"❌  {name} FAILED after {elapsed:.1f}s: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Crypto Spark Pipeline Runner")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip API ingestion")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization analysis")
    parser.add_argument(
        "--layers",
        default="ingest,bronze,quality,silver,gold",
        help="Comma-separated list of layers to run",
    )
    parser.add_argument(
        "--coins",
        default="bitcoin,ethereum,solana,cardano,avalanche-2,polkadot,chainlink,litecoin",
    )
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--raw-path", default="data/raw")
    parser.add_argument("--processed-path", default="data/processed")
    args = parser.parse_args()

    requested_layers = set(args.layers.split(","))
    run_start = datetime.now(timezone.utc)
    log.info(f"Pipeline started at {run_start.isoformat()}")
    log.info(f"Layers: {requested_layers} | Coins: {args.coins} | Days: {args.days}")

    results = {}

    # ── Ingestion ──────────────────────────────────────────────────
    if "ingest" in requested_layers and not args.skip_ingest:
        from src.ingestion.coingecko_ingest import ingest
        results["ingest"] = run_layer(
            "Ingestion",
            ingest,
            coins=args.coins.split(","),
            days=args.days,
        )
    else:
        log.info("Skipping ingestion")

    # ── Bronze ─────────────────────────────────────────────────────
    if "bronze" in requested_layers:
        from src.transformations.bronze_layer import run_bronze
        results["bronze"] = run_layer(
            "Bronze Layer",
            run_bronze,
            raw_path=args.raw_path,
            output_path=args.processed_path,
        )

    # ── Quality (pre-silver) ───────────────────────────────────────
    # Run quality on bronze tables first to catch raw data issues
    # before investing compute in Silver transformations.
    if "quality" in requested_layers and results.get("bronze", True):
        from src.quality.data_quality import run_quality_checks
        results["quality_bronze"] = run_layer(
            "Quality Checks (Bronze)",
            run_quality_checks,
            processed_path=args.processed_path,
        )

    # ── Silver ─────────────────────────────────────────────────────
    if "silver" in requested_layers and results.get("bronze", True):
        from src.transformations.silver_layer import run_silver
        results["silver"] = run_layer(
            "Silver Layer",
            run_silver,
            processed_path=args.processed_path,
        )

    # ── Quality (post-silver) ──────────────────────────────────────
    if "quality" in requested_layers and results.get("silver", True):
        from src.quality.data_quality import run_quality_checks
        results["quality_silver"] = run_layer(
            "Quality Checks (Silver)",
            run_quality_checks,
            processed_path=args.processed_path,
        )

    # ── Gold ───────────────────────────────────────────────────────
    if "gold" in requested_layers and results.get("silver", True):
        from src.transformations.gold_layer import run_gold
        results["gold"] = run_layer(
            "Gold Layer",
            run_gold,
            processed_path=args.processed_path,
        )

    # ── Optimization Analysis ──────────────────────────────────────
    if not args.skip_optimization and results.get("silver", True):
        from src.optimization.query_optimization import run_optimization_analysis
        results["optimization"] = run_layer(
            "Optimization Analysis",
            run_optimization_analysis,
            processed_path=args.processed_path,
        )

    # ── Summary ────────────────────────────────────────────────────
    run_end = datetime.now(timezone.utc)
    total_elapsed = (run_end - run_start).total_seconds()

    log.info("=" * 50)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 50)
    for layer, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        log.info(f"  {status}  {layer}")
    log.info(f"Total elapsed: {total_elapsed:.1f}s")
    log.info("=" * 50)

    # Exit with non-zero code if any layer failed (important for CI)
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
