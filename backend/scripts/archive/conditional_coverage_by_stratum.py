#!/usr/bin/env python3
"""Conditional coverage by cross-sectional stratum: the Mondrian gate.

`measure_conditional_qhat.py` refuted DATE-conditional q_hat (2026-08-12): the
band's miscalibration is not a calendar effect. Per-tier / liquidity / family
q_hat is a different axis and untested — but a Mondrian arm is only worth the
~1 day if coverage actually varies across the strata it would condition on. If
coverage is flat, skip the arm entirely.

METHOD (read-only; the archive already holds every number):

* Data: rows production actually SERVED and then resolved, >=$1 cohort,
  `excluded_forecast_date` applied — the same panel `centre_vs_lastprice.py`
  reads, including its anchor-wedge rebase (`r_hat` off the model's own quote,
  half-widths transplanted as offsets). A stratum spread measured on the wedge
  would be a data artifact, not a calibration signal.
* Strata (all knowable at serve time, all per horizon):
    tier     — `backtest.scoring.price_tier(base_price)` (a liquidity band by
               construction, not a display bucket).
    family   — `items.type` (DB) / `weapon_type` (Parquet metadata join).
    width    — served half-width tertile within the horizon (the volatility
               axis the band itself prices on).
    staleness— `base_stale_run_days`: fresh (0) / stale (>=1) / unknown (NULL),
               a frozen-anchor illiquidity proxy.
* Statistic: coverage per stratum with a 90% date-bootstrap CI (dates are the
  independent unit; a row bootstrap understates the interval). The gate read is
  the max-min spread per axis: flat means the CIs overlap and the spread sits
  inside bootstrap noise, and the verdict line says so explicitly.

Run: backend/venv/bin/python scripts/conditional_coverage_by_stratum.py
       --archive-dir ../cs2-oracle-data/price-archive
Default is prod Postgres (read-only); --archive-dir reads a Parquet copy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.scoring import MIN_HEADLINE_DATES, excluded_forecast_date, price_tier

SERVED_MIN_PRICE = 1.0
N_BOOTSTRAP = 1000
RNG_SEED = 42
MIN_DATES_FOR_CI = 3
# A spread below this is not worth a Mondrian arm even if nominally nonzero.
FLAT_SPREAD_PP = 5.0

_BASE_COLS = [
    "item_id",
    "forecast_date",
    "horizon_days",
    "base_price",
    "actual_price",
    "current_price",
    "predicted_price_low",
    "predicted_price_mid",
    "predicted_price_high",
    "base_stale_run_days",
]


def _select(db_source: str, family_expr: str) -> str:
    cols = ", ".join([*_BASE_COLS, family_expr])
    return f"""
    SELECT {cols}
    FROM {db_source}
    WHERE actual_price IS NOT NULL
      AND base_price > 0 AND actual_price > 0
      AND predicted_price_mid IS NOT NULL
      AND predicted_price_low IS NOT NULL
      AND predicted_price_high IS NOT NULL
      AND base_price >= {SERVED_MIN_PRICE}
    """


def _read_db() -> pd.DataFrame:
    from database import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        sql = _select("forecast_outcomes", "items.type AS family") + " AND forecast_outcomes.item_id = items.id"
        # The JOIN needs items in FROM; rebuild with it.
        cols = ", ".join(f"forecast_outcomes.{c}" for c in _BASE_COLS)
        sql = f"""
    SELECT {cols}, items.type AS family
    FROM forecast_outcomes JOIN items ON items.id = forecast_outcomes.item_id
    WHERE forecast_outcomes.actual_price IS NOT NULL
      AND forecast_outcomes.base_price > 0
      AND forecast_outcomes.actual_price > 0
      AND forecast_outcomes.predicted_price_mid IS NOT NULL
      AND forecast_outcomes.predicted_price_low IS NOT NULL
      AND forecast_outcomes.predicted_price_high IS NOT NULL
      AND forecast_outcomes.base_price >= {SERVED_MIN_PRICE}
    """
        rows = db.execute(text(sql)).fetchall()
    finally:
        db.close()
    return pd.DataFrame(rows, columns=[*_BASE_COLS, "family"])


def _read_parquet(archive_dir: Path) -> pd.DataFrame:
    import duckdb

    ops = archive_dir / "ops" / "forecast_outcomes.parquet"
    meta = archive_dir / "item-metadata.parquet"
    fam = f"LEFT JOIN read_parquet('{meta}') AS m ON m.item_slug = o.item_slug"
    cols = ", ".join(f"o.{c}" for c in [*_BASE_COLS, "item_slug"])
    # item-metadata.parquet carries weapon_type, not type.
    sql = f"""
    SELECT {cols}, m.weapon_type AS family
    FROM read_parquet('{ops}') AS o {fam}
    WHERE o.actual_price IS NOT NULL
      AND o.base_price > 0 AND o.actual_price > 0
      AND o.predicted_price_mid IS NOT NULL
      AND o.predicted_price_low IS NOT NULL
      AND o.predicted_price_high IS NOT NULL
      AND o.base_price >= {SERVED_MIN_PRICE}
    """
    df = duckdb.connect().sql(sql).fetchdf()
    return df.drop(columns=["item_slug"])


def bucket_staleness(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    return "fresh" if float(v) <= 0 else "stale"


def assign_width_tertile(hw: pd.Series) -> pd.Series:
    """Within-horizon half-width tertile: low / mid / high.

    qcut on ranks (not values) so ties cannot collapse a bin away; a
    degenerate (constant-width) horizon yields a single 'flat' stratum rather
    than three arbitrary ones.
    """
    if hw.nunique() <= 1:
        return pd.Series(["flat"] * len(hw), index=hw.index)
    r = hw.rank(method="first")
    try:
        return pd.qcut(r, 3, labels=["narrow", "mid", "wide"])
    except ValueError:
        return pd.Series(["flat"] * len(hw), index=hw.index)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Rebase to the model's own quote, derive coverage and strata."""
    df = df.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    for c in (
        "base_price",
        "actual_price",
        "current_price",
        "predicted_price_low",
        "predicted_price_mid",
        "predicted_price_high",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    reasons = df["forecast_date"].map(excluded_forecast_date)
    for why, n in reasons.dropna().value_counts().items():
        print(f"excluded {n:,} row(s): {why}")
    df = df[reasons.isna()].reset_index(drop=True)

    b = df["base_price"].to_numpy()
    q = df["current_price"].to_numpy(dtype=float)
    q = np.where(np.isnan(q) | (q <= 0), b, q)
    mid = df["predicted_price_mid"].to_numpy()
    r = df["actual_price"].to_numpy() / b - 1.0
    r_hat = mid / q - 1.0
    w_lo = (mid - df["predicted_price_low"].to_numpy()) / q
    w_hi = (df["predicted_price_high"].to_numpy() - mid) / q
    df["covered"] = (r >= r_hat - w_lo) & (r <= r_hat + w_hi)
    df["half_width"] = (w_lo + w_hi) / 2.0
    df["tier"] = [price_tier(p) for p in b]
    df["family"] = df["family"].fillna("unknown").astype(str)
    df["stale"] = [bucket_staleness(v) for v in df["base_stale_run_days"].to_numpy()]
    df["width"] = "flat"
    for _h, idx in df.groupby("horizon_days").groups.items():
        df.loc[idx, "width"] = assign_width_tertile(df.loc[idx, "half_width"]).astype(str).values
    return df


def _bootstrap_ci(g: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    dates = g["forecast_date"].unique()
    if len(dates) < MIN_DATES_FOR_CI:
        return (float("nan"), float("nan"))
    by_date = {d: sub for d, sub in g.groupby("forecast_date")}
    out = []
    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(dates, size=len(dates), replace=True)
        s = pd.concat([by_date[d] for d in draw], ignore_index=True)
        out.append(float(s["covered"].mean()))
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))


def stratum_table(df: pd.DataFrame, axis: str, rng: np.random.Generator) -> list:
    rows = []
    for (h, s), g in df.groupby(["horizon_days", axis]):
        lo, hi = _bootstrap_ci(g, rng)
        rows.append(
            dict(
                horizon=int(h),
                stratum=str(s),
                rows=len(g),
                dates=int(g["forecast_date"].nunique()),
                coverage=float(g["covered"].mean()),
                ci90=[lo, hi],
                median_width=float(g["half_width"].median()),
            )
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-dir", default=None, help="read a price-archive Parquet copy instead of prod Postgres.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    df = _read_parquet(Path(args.archive_dir)) if args.archive_dir else _read_db()
    print(f"source: {'parquet ' + str(args.archive_dir) if args.archive_dir else 'prod postgres'}")
    df = prepare(df)
    rng = np.random.default_rng(RNG_SEED)
    payload: dict = {}

    print(f"\nserved cohort (>=$1.0): {len(df):,} resolved rows, {df['forecast_date'].nunique()} forecast dates\n")
    for axis in ("tier", "family", "width", "stale"):
        rows = stratum_table(df, axis, rng)
        payload[axis] = rows
        print(f"=== by {axis} ===")
        print(f"{'h':>4} {'stratum':>12} {'rows':>8} {'dates':>6} {'cover':>7} {'90% CI':>22} {'med_hw%':>8}")
        for h in sorted({r["horizon"] for r in rows}):
            sub = [r for r in rows if r["horizon"] == h]
            covs = [r["coverage"] for r in sub]
            for r in sorted(sub, key=lambda d: d["stratum"]):
                ci = (
                    "[{:5.1f}, {:5.1f}]".format(r["ci90"][0] * 100, r["ci90"][1] * 100)
                    if np.isfinite(r["ci90"][0])
                    else "(<3 dates)"
                )
                print(
                    f"{h:>4} {r['stratum']:>12} {r['rows']:>8,} "
                    f"{r['dates']:>6} {r['coverage'] * 100:>6.1f}% "
                    f"{ci:>22} {r['median_width'] * 100:>7.2f}"
                )
            spread = (max(covs) - min(covs)) * 100 if covs else float("nan")
            n_dates = {r["dates"] for r in sub}
            mature = min(n_dates) >= MIN_HEADLINE_DATES if n_dates else False
            flat = spread < FLAT_SPREAD_PP
            tag = "FLAT" if flat else "SPREAD"
            pub = "" if mature else " (panel immature: <20 dates — informative, not quotable)"
            print(f"  h={h}: max-min spread {spread:.1f}pp -> {tag}{pub}")
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")
    print(
        "Gate read: an axis earns a Mondrian arm only with a SPREAD that "
        "survives its CIs. A FLAT axis at every horizon means skip the arm."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
