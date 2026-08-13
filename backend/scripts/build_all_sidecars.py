"""Build all four local sidecars. Downloads are the caller's job (see docstrings):
  kaggle datasets download -d devynpruden/cs2-skin-price-history-2013-2026 -f daily_prices.parquet
  gh api repos/atalantus/buff-price-history-archive/contents/price-history-daily.json.xz
"""
from __future__ import annotations
import sys
from pathlib import Path
from scripts.ingest_volume_panel import main as volume_main, SIDECAR_NAME as VOL
from scripts.ingest_supply_history import main as supply_main
from scripts.build_bid_panel import main as bid_main
from scripts.build_stattrak_panel import main as st_main
import pandas as pd


def run(archive_dir: Path, volume_src: Path, supply_src: Path) -> dict:
    volume_main(volume_src, archive_dir)
    supply_main(supply_src, archive_dir)
    bid_main(archive_dir)
    st_main(archive_dir / VOL, archive_dir)
    return {p.name: len(pd.read_parquet(p))
            for p in archive_dir.glob("*-panel.parquet")}


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
