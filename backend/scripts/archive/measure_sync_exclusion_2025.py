"""Split-half label-noise ceiling on the 2025 two-source panel.

REVISED TASK (the original sync-exclusion design cannot run on 2025 data):
the 2025 archive carries no `aggregator_sync` feed -- that is a 2026 source.
What 2025 does carry is two independent quotes per item-day:

    source IS NULL      -> the original Steam-derived quote ("steam")
    source='buff_iflow' -> an independent BUFF venue quote ("buff")

So instead of a sync on/off comparison, this script measures the LABEL NOISE
CEILING: how well does each source's h-day forward return predict the other's?
Under parallel-measures assumptions the cross-source return correlation `r`
is the reliability of a single source, and the Spearman-Brown formula gives
the reliability of the 2-source voted average -- the max R^2 any model can
achieve predicting that composite, and sqrt of it the max rank-correlation
(IC) it can achieve.

Formulas (classical test theory, per horizon):
    r      = Pearson(return_steam, return_buff)      # single-source reliability
    rel2   = 2*r / (1+r)                             # Spearman-Brown, 2-source avg
    max_R2 = rel2                                    # ceiling predicting composite
    max_IC = sqrt(max(rel2, 0))                      # ceiling rank-correlation
    true_sd = sd_pooled * sqrt(max(r, 0))            # latent return SD implied
              with sd_pooled = sqrt((var_s + var_b) / 2)

Reads through `db/archive.py::prices_relation` (never a raw glob: a bare
`read_parquet('prices-*.parquet')` narrows to the first file's schema and
silently drops `source`) plus `archive_universe_sql_filter()` from
`models/item_parser.py` (NULL-safe bid / historical_fallback / doppler /
phantom rules -- a bare NOT IN drops 13 years of prices). The panel is then
restricted to calendar 2025 and to items whose combined-2025 median price is
>= $1.00 (the served universe).

Returns are calendar-exact: return[d] = ln(price[d+h] / price[d]) requires a
row exactly h calendar days later in the SAME source; otherwise NaN. A paired
row additionally requires both sources' returns finite (i.e. four prices:
steam[d], steam[d+h], buff[d], buff[d+h]).

Usage (run from backend/):
    venv/bin/python scripts/measure_sync_exclusion_2025.py
    venv/bin/python scripts/measure_sync_exclusion_2025.py --out /tmp/label_ceiling_2025.json

Read-only: one DuckDB scan of the local Parquet archive, no DB, no writes
except the JSON sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import numpy as np
import pandas as pd
from db.archive import prices_relation
from models.item_parser import archive_universe_sql_filter

HORIZONS = (1, 3, 7, 14, 30)
MIN_MEDIAN_PRICE = 1.0
START = "2025-01-01"
END = "2025-12-31"
DEFAULT_OUT = "/tmp/label_ceiling_2025.json"


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #


def load_source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Steam (source IS NULL) and BUFF (source='buff_iflow') 2025 frames.

    Returns (steam, buff), each with columns [item_slug, day, price].
    Universe rules applied in SQL; the >=$1 median filter applied in pandas
    on the combined panel so the definition is one line and auditable.
    """
    con = duckdb.connect()
    try:
        rel = prices_relation(con)
    except FileNotFoundError as exc:
        raise SystemExit(f"price archive not found: {exc}") from exc
    uni = archive_universe_sql_filter()

    base = (
        f"SELECT item_slug, CAST(day AS DATE) AS day, mean_price AS price "
        f"FROM {rel} "
        f"WHERE CAST(day AS DATE) BETWEEN DATE '{START}' AND DATE '{END}' "
        f"AND ({uni}) AND mean_price IS NOT NULL AND mean_price > 0"
    )
    steam = con.sql(f"{base} AND source IS NULL").fetchdf()
    buff = con.sql(f"{base} AND source = 'buff_iflow'").fetchdf()
    con.close()

    for df in (steam, buff):
        df["day"] = pd.to_datetime(df["day"])
    return steam, buff


def apply_median_filter(
    steam: pd.DataFrame, buff: pd.DataFrame, min_price: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Keep only items whose combined-2025 median price >= min_price."""
    combined_med = (
        pd.concat([steam[["item_slug", "price"]], buff[["item_slug", "price"]]]).groupby("item_slug")["price"].median()
    )
    keep = set(combined_med[combined_med >= min_price].index)
    return (
        steam[steam["item_slug"].isin(keep)].copy(),
        buff[buff["item_slug"].isin(keep)].copy(),
        len(keep),
    )


# --------------------------------------------------------------------------- #
# returns + reliability
# --------------------------------------------------------------------------- #


def forward_returns(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Per-(item, day) h-day forward log return within one source.

    Calendar-exact: needs a row exactly h days later for the same item.
    Returns columns [item_slug, day, ret].
    """
    if df.empty:
        return pd.DataFrame({"item_slug": [], "day": [], "ret": []})
    fwd = df[["item_slug", "day", "price"]].copy()
    fwd["day"] = fwd["day"] - pd.to_timedelta(h, unit="D")
    fwd = fwd.rename(columns={"price": "price_fwd"})
    m = df[["item_slug", "day", "price"]].merge(fwd, on=["item_slug", "day"], how="inner")
    m["ret"] = np.log(m["price_fwd"] / m["price"])
    m = m[np.isfinite(m["ret"])]
    return m[["item_slug", "day", "ret"]]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def horizon_stats(steam: pd.DataFrame, buff: pd.DataFrame, h: int) -> dict:
    """Split-half stats for one horizon on pairwise-complete returns."""
    sr = forward_returns(steam, h).rename(columns={"ret": "ret_steam"})
    br = forward_returns(buff, h).rename(columns={"ret": "ret_buff"})
    paired = sr.merge(br, on=["item_slug", "day"], how="inner")
    n = len(paired)
    n_items = int(paired["item_slug"].nunique()) if n else 0
    if n < 3:
        return {
            "horizon": h,
            "n": n,
            "n_items": n_items,
            "single_source_corr": float("nan"),
            "composite_reliability": float("nan"),
            "max_R2": float("nan"),
            "max_IC": float("nan"),
            "sd_steam": float("nan"),
            "sd_buff": float("nan"),
            "sd_pooled": float("nan"),
            "true_sd": float("nan"),
        }
    xs = paired["ret_steam"].to_numpy(dtype=float)
    xb = paired["ret_buff"].to_numpy(dtype=float)
    r = pearson(xs, xb)
    sd_s = float(np.std(xs, ddof=1)) if n > 1 else float("nan")
    sd_b = float(np.std(xb, ddof=1)) if n > 1 else float("nan")
    sd_p = float(np.sqrt((np.var(xs, ddof=1) + np.var(xb, ddof=1)) / 2.0))
    if np.isfinite(r) and r > 0:
        rel2 = float(2.0 * r / (1.0 + r))
        max_ic = float(np.sqrt(max(rel2, 0.0)))
        true_sd = float(sd_p * np.sqrt(r))
    else:
        rel2 = float(2.0 * r / (1.0 + r)) if np.isfinite(r) and r != -1.0 else float("nan")
        max_ic = float("nan")
        true_sd = float("nan")
    return {
        "horizon": h,
        "n": n,
        "n_items": n_items,
        "single_source_corr": r,
        "composite_reliability": rel2,
        "max_R2": rel2,
        "max_IC": max_ic,
        "sd_steam": sd_s,
        "sd_buff": sd_b,
        "sd_pooled": sd_p,
        "true_sd": true_sd,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--min-price", type=float, default=MIN_MEDIAN_PRICE)
    args = ap.parse_args()

    steam, buff = load_source_frames()
    n_items_raw = pd.concat([steam[["item_slug"]], buff[["item_slug"]]])["item_slug"].nunique()
    print(
        f"loaded 2025 universe-filtered panel: "
        f"steam {len(steam):,} rows / {steam['item_slug'].nunique():,} items, "
        f"buff {len(buff):,} rows / {buff['item_slug'].nunique():,} items, "
        f"{n_items_raw:,} distinct items"
    )
    if steam.empty or buff.empty:
        print("one source is empty -- nothing to correlate")
        return 1

    joint_days = len(
        steam[["item_slug", "day"]].merge(buff[["item_slug", "day"]], on=["item_slug", "day"], how="inner")
    )
    print(f"joint (item, day) with both sources at d: {joint_days:,}")

    steam, buff, n_kept = apply_median_filter(steam, buff, args.min_price)
    joint_kept = len(
        steam[["item_slug", "day"]].merge(buff[["item_slug", "day"]], on=["item_slug", "day"], how="inner")
    )
    print(
        f"median >= ${args.min_price:.2f} (combined 2025): {n_kept:,} items; "
        f"steam {len(steam):,} rows / {steam['item_slug'].nunique():,} items, "
        f"buff {len(buff):,} rows / {buff['item_slug'].nunique():,} items, "
        f"joint days {joint_kept:,}"
    )
    if joint_kept == 0:
        print("no overlapping (item, day) after median filter")
        return 1

    rows = [horizon_stats(steam, buff, h) for h in HORIZONS]

    header = (
        f"{'h':>3} {'n_pairs':>9} {'n_items':>8} {'single_r':>9} "
        f"{'rel2':>8} {'max_R2':>8} {'max_IC':>8} "
        f"{'sd_pool':>8} {'true_sd':>8}"
    )
    print("")
    print("split-half reliability: steam log-returns vs buff log-returns")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['horizon']:>3} {r['n']:>9,} {r['n_items']:>8,} "
            f"{r['single_source_corr']:>9.4f} {r['composite_reliability']:>8.4f} "
            f"{r['max_R2']:>8.4f} {r['max_IC']:>8.4f} "
            f"{r['sd_pooled']:>8.4f} {r['true_sd']:>8.4f}"
        )
    print("")
    print(
        "single_r = Pearson corr of the two sources' returns (= one-source "
        "reliability); rel2 = 2r/(1+r) = reliability of the 2-source "
        "average = max R^2 predicting the composite; max_IC = sqrt(rel2); "
        "true_sd = pooled observed SD * sqrt(r)."
    )

    payload = {
        "meta": {
            "start": START,
            "end": END,
            "min_median_price": args.min_price,
            "median_def": "median(mean_price) over combined 2025 universe-filtered rows (both sources)",
            "sources": {"steam": "source IS NULL", "buff": "source=buff_iflow"},
            "return_def": "ln(price[d+h]/price[d]), calendar-exact within "
            "source; paired rows require both returns finite",
            "formulas": {
                "single_source_corr": "Pearson(ret_steam, ret_buff)",
                "composite_reliability": "2r/(1+r)",
                "max_R2": "= composite_reliability",
                "max_IC": "sqrt(max(composite_reliability,0))",
                "true_sd": "sd_pooled*sqrt(max(r,0))",
            },
            "panel": {
                "n_items_kept": n_kept,
                "steam_rows": len(steam),
                "buff_rows": len(buff),
                "joint_days": joint_kept,
            },
        },
        "horizons": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
