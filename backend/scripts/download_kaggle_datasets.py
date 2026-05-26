1#!/usr/bin/env python3
"""
Find and download CS:GO/CS2 historical price datasets from Kaggle.

This script searches for price history datasets on Kaggle and helps download them.

Requirements:
    pip install kaggle

Setup:
    1. Create Kaggle account: https://kaggle.com
    2. Go to Account > API and download kaggle.json
    3. Place kaggle.json in ~/.kaggle/
    4. Run: chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_kaggle_datasets.py --search "cs2 prices"
    python scripts/download_kaggle_datasets.py --dataset "muonneutrino/steam-community-market-history"
    python scripts/download_kaggle_datasets.py --list
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Popular known CS:GO datasets on Kaggle
KNOWN_DATASETS = {
    "steam-community-market": {
        "id": "muonneutrino/steam-community-market-history",
        "description": "Steam Community Market historical data",
        "coverage": "Multiple years of game items pricing"
    },
    "csgo-prices": {
        "id": "christianrosillo/csgo-price-history",
        "description": "CS:GO skin price history",
        "coverage": "Historical prices for popular skins"
    },
    "csgo-cases": {
        "id": "mrmorj/csgo-weapon-cases",
        "description": "CS:GO weapon cases and items",
        "coverage": "Case contents and rarity data"
    }
}


def check_kaggle_installed() -> bool:
    """Check if Kaggle CLI is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        logger.error("Kaggle not installed. Run: pip install kaggle")
        return False


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        logger.error(f"Kaggle credentials not found at {kaggle_config}")
        logger.error("See setup instructions at top of this script")
        return False
    return True


def list_known_datasets():
    """List popular CS:GO datasets."""
    print("\n" + "="*80)
    print("POPULAR CS:GO/CS2 DATASETS ON KAGGLE")
    print("="*80 + "\n")

    for key, dataset in KNOWN_DATASETS.items():
        print(f"Dataset: {key}")
        print(f"  ID: {dataset['id']}")
        print(f"  Description: {dataset['description']}")
        print(f"  Coverage: {dataset['coverage']}")
        print()

    print("To download a dataset:")
    print("  python scripts/download_kaggle_datasets.py --dataset DATASET_ID\n")


def search_datasets(query: str) -> bool:
    """Search Kaggle for datasets matching query."""
    try:
        import kaggle

        logger.info(f"Searching Kaggle for: {query}")

        # Note: kaggle.api.dataset_list_files() requires authentication
        # This is a simplified version showing the concept
        logger.info("Searching... (requires Kaggle CLI setup)")

        print("\nTo search manually:")
        print(f"  1. Visit: https://kaggle.com/datasets?search={query}")
        print(f"  2. Look for CS:GO/CS2 price history datasets")
        print(f"  3. Get the dataset ID (username/dataset-name)")
        print(f"  4. Run: python scripts/download_kaggle_datasets.py --dataset DATASET_ID\n")

        return True

    except Exception as e:
        logger.error(f"Error searching: {e}")
        return False


def download_dataset(dataset_id: str, output_dir: str = "data/kaggle") -> bool:
    """
    Download a dataset from Kaggle.

    Args:
        dataset_id: Dataset ID (username/dataset-name)
        output_dir: Directory to save files

    Returns:
        True if successful, False otherwise
    """
    if not check_kaggle_installed():
        return False

    if not check_kaggle_credentials():
        return False

    try:
        import kaggle

        logger.info(f"Downloading dataset: {dataset_id}")
        logger.info(f"Output directory: {output_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_id,
            path=output_dir,
            unzip=True
        )

        logger.info(f"✓ Downloaded to: {output_path}")

        # List files
        files = list(output_path.glob("**/*"))
        print(f"\nFiles downloaded:")
        for f in files:
            if f.is_file():
                size = f.stat().st_size / 1024 / 1024  # MB
                print(f"  {f.name} ({size:.2f} MB)")

        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def format_for_import(csv_file: str, output_file: str = "data/prices.csv"):
    """
    Format downloaded Kaggle data for import.

    Takes CSV from Kaggle and formats it for the import script.
    """
    import csv

    try:
        logger.info(f"Formatting {csv_file} for import...")

        input_path = Path(csv_file)
        if not input_path.exists():
            logger.error(f"File not found: {csv_file}")
            return False

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read and reformat
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            writer = csv.DictWriter(
                outfile,
                fieldnames=['item_name', 'timestamp', 'price', 'volume', 'source']
            )
            writer.writeheader()

            for row in reader:
                # Adapt column names based on dataset structure
                # This is dataset-dependent, adjust as needed
                try:
                    writer.writerow({
                        'item_name': row.get('item_name') or row.get('name') or row.get('market_hash_name'),
                        'timestamp': row.get('timestamp') or row.get('date') or row.get('time'),
                        'price': row.get('price') or row.get('mean_price') or row.get('last_price'),
                        'volume': row.get('volume') or row.get('supply') or '0',
                        'source': 'kaggle'
                    })
                except Exception as e:
                    logger.debug(f"Skipping row: {e}")

        logger.info(f"✓ Formatted to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Formatting failed: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Download CS:GO/CS2 datasets from Kaggle',
        epilog='''
Examples:
  python download_kaggle_datasets.py --list
  python download_kaggle_datasets.py --search "cs2 prices"
  python download_kaggle_datasets.py --dataset muonneutrino/steam-community-market-history
        '''
    )

    parser.add_argument('--list', action='store_true', help='List known datasets')
    parser.add_argument('--search', help='Search for datasets')
    parser.add_argument('--dataset', help='Download specific dataset (username/dataset-name)')
    parser.add_argument('--format', help='Format CSV file for import')
    parser.add_argument('--output', default='data/kaggle', help='Output directory')

    args = parser.parse_args()

    if args.list:
        list_known_datasets()
        return 0

    elif args.search:
        return 0 if search_datasets(args.search) else 1

    elif args.dataset:
        success = download_dataset(args.dataset, args.output)
        if success:
            print("\nNext step: Format and import the data")
            print("  python scripts/import_historical_prices.py --source csv --file data/prices.csv")
        return 0 if success else 1

    elif args.format:
        return 0 if format_for_import(args.format) else 1

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
