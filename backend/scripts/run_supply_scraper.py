#!/usr/bin/env python3
"""
CLI entry point for the daily supply scraper.

Collects sell_listings (Steam) into the supply_snapshots table.
Designed to be called by the supply-scraper GitHub Actions workflow.

Usage:
    python scripts/run_supply_scraper.py
    python scripts/run_supply_scraper.py --burst-size 25 --burst-pause 25
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.supply_scraper import DEFAULT_BURST_PAUSE, DEFAULT_BURST_SIZE, SupplyScraper
from database import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_supply_scraper")


def run(burst_size: int = DEFAULT_BURST_SIZE, burst_pause: float = DEFAULT_BURST_PAUSE) -> dict:
    """Run the full supply scrape cycle. Returns status dict."""
    logger.info("=" * 60)
    logger.info("SUPPLY SCRAPER — Starting")
    logger.info(f"  Burst: {burst_size} req, then {burst_pause}s pause")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        scraper = SupplyScraper(db, burst_size=burst_size, burst_pause=burst_pause)
        result = scraper.run()

        if result.get("status") == "error":
            logger.error(f"Supply scraper failed: {result.get('error')}")
            return result

        logger.info(f"  Steam items: {result['steam_items']:,}")
        logger.info(f"  Duration: {result['duration_seconds']}s")
        logger.info("Supply scraper completed successfully")
        return result
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


def main():
    """CLI entry point (parses argv)."""
    parser = argparse.ArgumentParser(description="Daily supply scraper")
    parser.add_argument(
        "--burst-size", type=int, default=DEFAULT_BURST_SIZE, help=f"Requests per burst (default: {DEFAULT_BURST_SIZE})"
    )
    parser.add_argument(
        "--burst-pause",
        type=float,
        default=DEFAULT_BURST_PAUSE,
        help=f"Seconds between bursts (default: {DEFAULT_BURST_PAUSE})",
    )
    args = parser.parse_args()

    result = run(burst_size=args.burst_size, burst_pause=args.burst_pause)
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
