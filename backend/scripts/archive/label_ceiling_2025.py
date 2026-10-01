#!/usr/bin/env python3
"""Split-half reliability of the voted label on the 2025 multi-source panel.

The 2025 archive has two independent sources:
  - source IS NULL: the original Steam-derived quote (aggregator_steam_17mafo)
  - source='buff_iflow': BUFF venue via the iflow ingest path

This script computes per-horizon forward returns from each source independently,
then measures how well they agree (Pearson correlation of returns). The
Spearman-Brown corrected reliability for the 2-source average gives the MAX
achievable R^2 for the voted composite.

Compare to the earlier 87-date 2026 measurement which used up to 7 sources.

Usage:
    venv/bin/python -m scripts.archive.label_ceiling_2025 --out /tmp/label_ceiling_2025.json
"""

import json
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from db.archive import prices_relation
from models.item_parser import archive_universe_sql_filter

HORIZONS = [1, 3, 7, 14, 30]
MIN_PRICE = 1.0


def load_2025_sources():
    """Load 2025 prices split by source, filtered to >= $1 median items."""
    con = duckdb.connect()
    rel = prices_relation(con)
    uni = archive_universe_sql_filter()

    df = con.sql(f"""
        SELECT item_slug, CAST(day AS DATE) AS day, source, mean_price
        FROM {rel}
        WHERE CAST(day AS DATE) BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
          AND ({uni})
          AND mean_price > 0
    """).fetchdf()
    con.close()

    medians = df.groupby("item_slug")["mean_price"].median()
    ge1 = set(medians[medians >= MIN_PRICE].index)
    df = df[df["item_slug"].isin(ge1)].copy()

    steam = df[df["source"].isna()].rename(columns={"mean_price": "price_steam"}).drop(columns=["source"])
    buff = df[df["source"] == "buff_iflow"].rename(columns={"mean_price": "price_buff"}).drop(columns=["source"])

    merged = steam.merge(buff, on=["item_slug", "day"], how="inner")
    print(
        f"Loaded {len(merged):,} matched (item, day) pairs, "
        f"{merged['item_slug'].nunique()} items, "
        f"{merged['day'].nunique()} days"
    )
    return merged


def compute_forward_returns(df, price_col, horizon):
    """Log forward return at horizon h days."""
    df = df.sort_values(["item_slug", "day"])
    future = df.groupby("item_slug")[price_col].shift(-horizon)
    return np.log(future / df[price_col])


def measure_reliability(merged):
    results = {}
    for h in HORIZONS:
        m = merged.copy()
        m["r_steam"] = compute_forward_returns(m, "price_steam", h)
        m["r_buff"] = compute_forward_returns(m, "price_buff", h)
        valid = m.dropna(subset=["r_steam", "r_buff"])

        r_steam = valid["r_steam"].to_numpy()
        r_buff = valid["r_buff"].to_numpy()

        ok = np.isfinite(r_steam) & np.isfinite(r_buff)
        r_steam, r_buff = r_steam[ok], r_buff[ok]

        single_corr = float(np.corrcoef(r_steam, r_buff)[0, 1])
        sb_reliability = 2 * single_corr / (1 + single_corr)
        max_ic = float(np.sqrt(max(0, sb_reliability)))

        obs_sd_steam = float(np.std(r_steam))
        obs_sd_buff = float(np.std(r_buff))
        true_sd = float(np.sqrt(max(0, sb_reliability)) * (obs_sd_steam + obs_sd_buff) / 2)

        results[h] = {
            "n_pairs": int(ok.sum()),
            "n_items": int(valid["item_slug"].nunique()),
            "single_source_corr": single_corr,
            "sb_reliability": sb_reliability,
            "max_R2": sb_reliability,
            "max_IC": max_ic,
            "obs_sd_steam_pct": obs_sd_steam * 100,
            "obs_sd_buff_pct": obs_sd_buff * 100,
            "true_sd_pct": true_sd * 100,
        }
    return results


def print_table(results):
    print(f"\n{'h':>4} {'n_pairs':>10} {'r(A,B)':>8} {'SB_rel':>8} {'max_R2':>8} {'max_IC':>8} {'true_SD%':>9}")
    print("-" * 65)
    for h in HORIZONS:
        r = results[h]
        print(
            f"{h:>4} {r['n_pairs']:>10,} {r['single_source_corr']:>8.4f} "
            f"{r['sb_reliability']:>8.4f} {r['max_R2']:>8.4f} "
            f"{r['max_IC']:>8.4f} {r['true_sd_pct']:>8.2f}%"
        )
    print()


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    merged = load_2025_sources()
    results = measure_reliability(merged)
    print_table(results)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
