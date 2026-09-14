#!/usr/bin/env python3
"""
Discover new CS2 items from Steam Community Market.
This script browses Steam listings to find items not yet in the database.

Usage:
    python scripts/discover_steam_items.py [max_items]

Example:
    python scripts/discover_steam_items.py 5000
"""

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.real_data_collector import get_collector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    max_items = 5000

    if len(sys.argv) > 1:
        try:
            max_items = int(sys.argv[1])
        except ValueError:
            print(f"Invalid max_items: {sys.argv[1]}")
            sys.exit(1)

    logger.info(f"Starting Steam item discovery (max {max_items} items)...")
    logger.info("⏳ This will take a while due to Steam rate limiting (15s between requests)")
    logger.info("Please be patient...")

    collector = get_collector()
    stats = collector.discover_new_items_from_steam(max_items=max_items)

    print("\n" + "=" * 60)
    print("DISCOVERY COMPLETE")
    print("=" * 60)
    print(f"✓ Discovered items: {stats['discovered']}")
    print(f"✓ Added to database: {stats['added_to_db']}")
    print(f"⊘ Already existed: {stats['already_exists']}")
    print(f"✗ Errors: {stats['errors']}")

    if stats["items"]:
        print("\nFirst 10 new items found:")
        for item in stats["items"][:10]:
            print(f"  - {item}")
        if len(stats["items"]) > 10:
            print(f"  ... and {len(stats['items']) - 10} more")

    if stats["added_to_db"] > 0:
        print(f"\n✅ Successfully added {stats['added_to_db']} new items to database!")
        print("You can now collect prices for these items.")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
