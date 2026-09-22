"""Compute a StatTrak-premium sidecar from paired ST / base prices.

No download: the ratio st_price/base_price on the same day is a free
usage/demand proxy. Prices come from the volume panel's steam_sale_median so
both legs share one coherent source. Private/local only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ST_PREFIX = "StatTrak™ "
SIDECAR_NAME = "stattrak-panel.parquet"


def base_name(slug: str) -> str | None:
    return slug[len(ST_PREFIX) :] if slug.startswith(ST_PREFIX) else None


def build_stattrak_panel(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices[prices["price"] > 0].copy()
    st = df[df["item_id"].str.startswith(ST_PREFIX)].copy()
    st["base_id"] = st["item_id"].map(base_name)
    merged = st.merge(
        df.rename(columns={"item_id": "base_id", "price": "base_price"}), on=["base_id", "date"], how="inner"
    )
    merged["st_premium"] = merged["price"] / merged["base_price"]
    return merged[["item_id", "date", "st_premium"]].reset_index(drop=True)


def main(volume_panel: Path, out_dir: Path) -> Path:
    vp = pd.read_parquet(volume_panel)
    prices = vp.rename(columns={"steam_sale_median": "price"})[["item_id", "date", "price"]]
    panel = build_stattrak_panel(prices)
    out = out_dir / SIDECAR_NAME
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows, {panel['item_id'].nunique():,} items -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
