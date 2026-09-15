"""Ingest the devynpruden CS2 daily-price parquet into a local volume sidecar.

Source: kaggle datasets download -d devynpruden/cs2-skin-price-history-2013-2026 \
        -f daily_prices.parquet
It reaches 2026-06-15 (covers the serving anchors) and carries dense `volume`.
Local research dataset only -- private/local training; do not redistribute rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REQUIRED = ["market_hash_name", "date", "price_median", "volume"]
SIDECAR_NAME = "volume-panel.parquet"


def build_volume_panel(src_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in src_df.columns]
    if missing:
        raise ValueError(f"volume source missing columns: {missing}")
    df = src_df[REQUIRED].copy()
    df = df.rename(columns={"market_hash_name": "item_id"})
    df = df.dropna(subset=["item_id", "date", "volume"])
    # One row per (item_id, date): mean median-price, summed volume.
    out = df.groupby(["item_id", "date"], as_index=False).agg(
        steam_volume=("volume", "sum"), steam_sale_median=("price_median", "mean")
    )
    out["steam_volume"] = out["steam_volume"].astype("int64")
    return out[["item_id", "date", "steam_volume", "steam_sale_median"]]


def main(kaggle_parquet: Path, out_dir: Path) -> Path:
    src = pd.read_parquet(kaggle_parquet)
    panel = build_volume_panel(src)
    out = out_dir / SIDECAR_NAME
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows, {panel['item_id'].nunique():,} items -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
