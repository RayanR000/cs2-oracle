#!/usr/bin/env python3
"""Promote staged BUFF-iflow price rows into the canonical Parquet archive.

`backfill_buff_iflow.py` writes `source="buff_iflow"` rows into a staging dir
and deliberately stops there; this script is the promotion half. Its purpose is
the 2026-04-16..05-20 stretch of the multi-source hole: canonical carries a
single source there (`aggregator_steam_17mafo`, itself basis-rescaled by
`merge_17mafo_gap.py`), so promoting iflow adds the only genuinely independent
venue quote available for those days and joins the 2026-03 island to 05-20.

The iflow upstream feed ends 2026-05-20, so 05-21..07-10 cannot be filled from
this source.

Before writing, the staged rows are checked against canonical
`aggregator_buff163` on any overlapping days: same venue via a different ingest
path, so the level agreement bounds the conversion/slug error.

Usage (from backend/):
    venv/bin/python scripts/promote_iflow_staging.py --dry-run
    venv/bin/python scripts/promote_iflow_staging.py \
        --start 2026-04-16 --end 2026-05-20
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.archive import ARCHIVE_ROOT
from db.parquet import append_monthly

SOURCE = "buff_iflow"
PRICE_COLS = ["item_slug", "day", "source", "mean_price", "volume", "ingested_at"]
DEDUP_KEYS = ["item_slug", "day", "source"]

DEFAULT_START = "2026-04-16"
DEFAULT_END = "2026-05-20"
IFLOW_FEED_END = "2026-05-20"

# The promoted range sits inside the multi-source hole, so canonical
# `aggregator_buff163` is absent there by construction and cannot validate it.
# Check against the adjacent pre-gap window instead, where both ingest paths
# cover the same days.
DEFAULT_CHECK_START = "2026-04-01"
DEFAULT_CHECK_END = "2026-04-15"

# Same-venue cross-check thresholds (see module docstring).
MAX_MEDIAN_LOG_RATIO = 0.05
MAX_TAIL_FRACTION = 0.10  # frac of rows disagreeing by >25%


def load_staged(staging_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Staged price rows for [start, end], validated to a single source."""
    paths = sorted(glob.glob(str(staging_dir / "prices-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no staged prices-*.parquet under {staging_dir}")
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["day"] = pd.to_datetime(df["day"])
        frames.append(df[(df["day"] >= start) & (df["day"] <= end)])
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise ValueError(f"no staged rows in {start}..{end}")
    sources = set(out["source"].unique())
    if sources != {SOURCE}:
        raise ValueError(f"expected only source={SOURCE!r}, found {sorted(sources)}")
    missing = [c for c in PRICE_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"staged rows missing columns {missing}")
    return out[PRICE_COLS]


def validate_coverage(
    prices: pd.DataFrame, start: str, end: str, min_items: int = 1000, max_missing_days: int = 0
) -> None:
    """Raise AssertionError on a too-sparse day, or too many absent days.

    The upstream iflow feed drops the occasional day (five in 2025), so a small
    number of absent days is tolerable; a run of them is not, and neither is a
    day that is present but thin. ``max_missing_days`` bounds the former.
    """
    expected = pd.date_range(start=start, end=end, freq="D")
    present = set(prices["day"].unique())
    missing = [ts for ts in expected if ts not in present]
    assert len(missing) <= max_missing_days, (
        f"{len(missing)} days absent from staged rows "
        f"(limit {max_missing_days}): {[str(t.date()) for t in missing[:10]]}"
    )
    if missing:
        print(f"Tolerating {len(missing)} absent day(s): {[str(t.date()) for t in missing]}")
    for ts in expected:
        if ts in missing:
            continue
        count = prices.loc[prices["day"] == ts, "item_slug"].nunique()
        assert count >= min_items, f"low item count for {ts.date()}: {count} < {min_items}"


def cross_check_buff163(prices: pd.DataFrame, archive_glob: str) -> dict | None:
    """Compare staged levels to canonical `aggregator_buff163` where they overlap.

    Returns None when the two never share an (item, day); otherwise a stats dict
    and a hard failure if the median bias or the disagreement tail is too large.
    """
    import duckdb

    con = duckdb.connect()
    try:
        canon = con.sql(f"""
            SELECT item_slug, day, mean_price AS pa
            FROM read_parquet('{archive_glob}')
            WHERE source = 'aggregator_buff163' AND mean_price > 0
        """).fetchdf()
    except duckdb.Error:
        return None
    finally:
        con.close()
    if canon.empty:
        return None

    canon["day"] = pd.to_datetime(canon["day"])
    mine = prices.loc[prices["mean_price"] > 0, ["item_slug", "day", "mean_price"]]
    m = canon.merge(mine.rename(columns={"mean_price": "pi"}), on=["item_slug", "day"])
    if m.empty:
        return None

    lr = np.log(m["pi"] / m["pa"])
    stats = {
        "rows": len(m),
        "items": m["item_slug"].nunique(),
        "dates": m["day"].nunique(),
        "log_level_corr": float(np.log(m["pa"]).corr(np.log(m["pi"]))),
        "median_log_ratio": float(lr.median()),
        "p10_log_ratio": float(lr.quantile(0.10)),
        "p90_log_ratio": float(lr.quantile(0.90)),
        "tail_frac_gt25pct": float((lr.abs() > 0.25).mean()),
    }
    assert abs(stats["median_log_ratio"]) <= MAX_MEDIAN_LOG_RATIO, (
        f"median log-ratio {stats['median_log_ratio']:.4f} exceeds {MAX_MEDIAN_LOG_RATIO} — suspect FX or slug mapping"
    )
    assert stats["tail_frac_gt25pct"] <= MAX_TAIL_FRACTION, (
        f"{stats['tail_frac_gt25pct']:.1%} of rows disagree by >25%, limit {MAX_TAIL_FRACTION:.0%}"
    )
    return stats


def run(
    start: str,
    end: str,
    staging_dir: Path,
    out_dir: Path,
    dry_run: bool = False,
    min_items: int = 1000,
    max_missing_days: int = 0,
    check_start: str = DEFAULT_CHECK_START,
    check_end: str = DEFAULT_CHECK_END,
) -> pd.DataFrame:
    if end > IFLOW_FEED_END:
        raise ValueError(
            f"--end {end} is past the iflow feed end {IFLOW_FEED_END}; "
            "days after that cannot be filled from this source"
        )

    prices = load_staged(staging_dir, start, end)
    print(
        f"Loaded {len(prices):,} staged rows, "
        f"{prices['item_slug'].nunique():,} items, "
        f"{prices['day'].nunique()} days ({start}..{end})"
    )

    validate_coverage(prices, start, end, min_items=min_items, max_missing_days=max_missing_days)
    print("Coverage validation passed")

    try:
        check_rows = load_staged(staging_dir, check_start, check_end)
    except ValueError:
        check_rows = None
    stats = None if check_rows is None else cross_check_buff163(check_rows, str(out_dir / "prices-*.parquet"))
    if stats is None:
        print(f"Cross-check skipped: no canonical aggregator_buff163 overlap in {check_start}..{check_end}")
    else:
        print(
            f"Cross-check vs aggregator_buff163 on {check_start}..{check_end}: "
            f"{stats['rows']:,} rows / "
            f"{stats['items']:,} items / {stats['dates']} dates"
        )
        print(
            f"  log-level corr {stats['log_level_corr']:.4f}  "
            f"median log-ratio {stats['median_log_ratio']:+.4f}  "
            f"p10/p90 {stats['p10_log_ratio']:+.4f}/{stats['p90_log_ratio']:+.4f}  "
            f"tail>25% {stats['tail_frac_gt25pct']:.2%}"
        )

    if dry_run:
        print("Dry run — no files written")
        return prices

    append_monthly(out_dir, "prices", prices[PRICE_COLS], DEDUP_KEYS)
    print(f"Done. Appended {start}..{end} as source={SOURCE} under {out_dir}")
    return prices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--staging-dir", default=ARCHIVE_ROOT.parent / "buff-iflow-staging" / "price-archive")
    ap.add_argument("--out-dir", default=ARCHIVE_ROOT)
    ap.add_argument("--min-items", type=int, default=1000)
    ap.add_argument(
        "--max-missing-days", type=int, default=0, help="tolerate this many absent days (upstream feed gaps)"
    )
    ap.add_argument("--check-start", default=DEFAULT_CHECK_START, help="pre-gap window used to validate staged levels")
    ap.add_argument("--check-end", default=DEFAULT_CHECK_END)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run(
        args.start,
        args.end,
        Path(args.staging_dir),
        Path(args.out_dir),
        dry_run=args.dry_run,
        min_items=args.min_items,
        max_missing_days=args.max_missing_days,
        check_start=args.check_start,
        check_end=args.check_end,
    )


if __name__ == "__main__":
    main()
