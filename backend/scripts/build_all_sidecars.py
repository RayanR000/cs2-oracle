"""Build all four local sidecars. Downloads are the caller's job (see docstrings):
kaggle datasets download -d devynpruden/cs2-skin-price-history-2013-2026 -f daily_prices.parquet
gh api repos/atalantus/buff-price-history-archive/contents/price-history-daily.json.xz
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scripts.build_bid_panel import SIDECAR_NAME as BID
from scripts.build_bid_panel import main as bid_main
from scripts.build_stattrak_panel import SIDECAR_NAME as STATTRAK
from scripts.build_stattrak_panel import main as st_main
from scripts.ingest_supply_history import SIDECAR_NAME as SUPPLY
from scripts.ingest_supply_history import main as supply_main
from scripts.ingest_volume_panel import SIDECAR_NAME as VOL
from scripts.ingest_volume_panel import main as volume_main

# Every sidecar this orchestrator can produce, by its known filename -- not a
# glob. `-panel.parquet` doesn't match `supply-history.parquet`, so a glob
# silently drops it from the returned dict even though the file writes fine;
# this is the exact "no count fields" failure mode the repo's zero-row guard
# exists for. Enumerate explicitly so the contract holds as names evolve.
SIDECAR_NAMES = [VOL, SUPPLY, BID, STATTRAK]


def _count_sidecars(archive_dir: Path) -> dict:
    """Row counts for every known sidecar that exists under archive_dir."""
    return {name: len(pd.read_parquet(archive_dir / name)) for name in SIDECAR_NAMES if (archive_dir / name).exists()}


def run(archive_dir: Path, volume_src: Path, supply_src: Path) -> dict:
    volume_main(volume_src, archive_dir)
    supply_main(supply_src, archive_dir)
    bid_main(archive_dir)
    st_main(archive_dir / VOL, archive_dir)
    return _count_sidecars(archive_dir)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
