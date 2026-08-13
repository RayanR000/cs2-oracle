"""Ingest atalantus/buff-price-history-archive listing counts into a sidecar.

Source: gh api repos/atalantus/buff-price-history-archive/contents/\
        price-history-daily.json.xz  (or clone). Decompress with lzma.
Format: {name: [[unix_s...],[cny*100...],[listing_count...]]}. Third array is
listing count, populated only after 2023-01-25; coverage ends 2024-01-19.
TRAINING-ONLY (no 2026 coverage). Private/local; do not redistribute rows.
"""
from __future__ import annotations
import json
import lzma
import sys
from datetime import date, timezone, datetime
from pathlib import Path
import pandas as pd

SIDECAR_NAME = "supply-history.parquet"


def parse_supply_history(raw: dict) -> pd.DataFrame:
    records = []
    for name, arrays in raw.items():
        if not isinstance(arrays, list) or len(arrays) < 3:
            continue  # no listing-count array -> drop, never zero-fill
        stamps, _prices, counts = arrays[0], arrays[1], arrays[2]
        if len(counts) != len(stamps):
            raise ValueError(f"{name}: counts/timestamps length mismatch")
        for ts, cnt in zip(stamps, counts):
            if cnt is None:
                continue
            d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            records.append((name, d, int(cnt)))
    out = pd.DataFrame(records, columns=["item_id", "date", "buff_listing_count"])
    if not out.empty:
        out = out.groupby(["item_id", "date"], as_index=False)["buff_listing_count"].max()
    return out


def main(xz_path: Path, out_dir: Path) -> Path:
    with lzma.open(xz_path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    panel = parse_supply_history(raw)
    out = out_dir / SIDECAR_NAME
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows, {panel['item_id'].nunique():,} items -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
