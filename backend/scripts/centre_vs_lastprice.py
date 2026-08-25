#!/usr/bin/env python3
"""The centre's null: does the GBM centre beat quoting the last price?

`climatology_vs_gbm.py` settled the band's SCALE — a featureless per-item
climatology is ~30% narrower at matched coverage, so the GBM's width is
decorative and `CLIMATOLOGY_SCALE` has served since 2026-08-20. That leaves the
CENTRE (`r_hat`) as the model's only load-bearing use, and it has never been
tested against the obvious null: quote today's price and predict no move.

Two existing results BOUND the centre without testing it. The deep review
measured median |r_hat| as moving the interval <5% of its own width, and the
trade hunt found direction selection null (AUC 0.51-0.53). But null ORDERING is
not null LEVEL: a centre can be useless for ranking and still beat a random walk
on average error. That gap is what this measures.

METHOD (read-only; the archive already holds every number):

* Data: `price-archive/ops/forecast_outcomes.parquet` — rows production actually
  SERVED and then resolved against the realised price. No replay, no artifact,
  no retrain, so no flag-matching hazard.
* THE ANCHOR WEDGE IS REBASED, which is the whole correctness of this script.
  `predict()` builds the triple as `current_price x (1 + ret)` while the outcome
  legs (`base_price`, `actual_price`) come from `resolve_anchors`; the two
  anchors disagree on 85% of production rows by a median 5.70%. So the model's
  return is read off its OWN quote — `r_hat = mid/current_price - 1` — exactly
  as `_derive_verdict` rebases `in_interval`. Reading it off `base_price`
  instead charges the model for the stale-quote wedge and makes it look ~2.7x
  worse than the random walk, which is measuring the wedge, not the centre.
* Both centres then share one anchor and one outcome, so the comparison is
  paired per row and nothing differs but where the centre sits.
* Error is RELATIVE (r = price/base - 1), because the served cohort spans four
  price decades and a dollar MAE would just rank items by price.
      GBM centre:        |r_actual - r_hat|
      last-price centre: |r_actual|
* Skill = 1 - MAE_gbm / MAE_naive. Positive means the model beats the random
  walk. Reported per horizon; horizons are never pooled (the geometry eras in
  docs make a pooled number uninterpretable).
* Panel: `excluded_forecast_date` is applied, the same exclusion the published
  figures use. It removes 2026-07-19 (a dead-band direction rule live for one
  run) and 2025-12-01 (a replay), so h=30 falls below MIN_FORECAST_DATES and is
  reported as UNPUBLISHABLE rather than dropped silently.
* Uncertainty: bootstrap over forecast DATES, not rows — a date's returns are
  correlated through the market factor, so a row bootstrap would understate the
  interval by roughly the square root of the cohort size.
* Coverage: the band's half-widths (low/high offsets around the mid) are
  TRANSPLANTED onto the last-price centre, holding width byte-identical, and
  coverage is recomputed. This isolates the centre: any coverage difference is
  the centre's placement alone.

Run: backend/venv/bin/python scripts/centre_vs_lastprice.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.scoring import (                       # noqa: E402
    MIN_FORECAST_DATES, excluded_forecast_date)

ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "price-archive"
SERVED_MIN_PRICE = 1.0     # the >=$1 cohort the forecast is actually served on
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _load(archive_dir: Path) -> pd.DataFrame:
    import duckdb

    p = archive_dir / "ops" / "forecast_outcomes.parquet"
    df = duckdb.connect().sql(f"""
        SELECT forecast_date, horizon_days, base_price, actual_price,
               current_price,
               predicted_price_low, predicted_price_mid, predicted_price_high,
               model_version, base_stale_run_days
        FROM read_parquet('{p}')
        WHERE actual_price IS NOT NULL
          AND base_price > 0 AND actual_price > 0
          AND predicted_price_mid IS NOT NULL
          AND base_price >= {SERVED_MIN_PRICE}
    """).fetchdf()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    reasons = df["forecast_date"].map(excluded_forecast_date)
    for why, n in reasons.dropna().value_counts().items():
        print(f"excluded {n:,} row(s): {why}")
    df = df[reasons.isna()].reset_index(drop=True)

    b = df["base_price"].to_numpy()
    # `_quote_basis`: the price the forecast was QUOTED FROM, falling back to the
    # resolved base for the legacy rows that predate the column.
    q = df["current_price"].to_numpy(dtype=float)
    q = np.where(np.isnan(q) | (q <= 0), b, q)
    mid = df["predicted_price_mid"].to_numpy()
    df["r_actual"] = df["actual_price"].to_numpy() / b - 1.0
    df["r_hat"] = mid / q - 1.0
    # Half-widths as offsets from the MID, on the band's own basis, so they can
    # be moved onto another centre without carrying the wedge with them.
    df["w_lo"] = (mid - df["predicted_price_low"].to_numpy()) / q
    df["w_hi"] = (df["predicted_price_high"].to_numpy() - mid) / q
    return df


def _mae(r_actual: np.ndarray, centre: np.ndarray) -> float:
    return float(np.mean(np.abs(r_actual - centre)))


def _covered(r: np.ndarray, centre: np.ndarray,
             w_lo: np.ndarray, w_hi: np.ndarray) -> np.ndarray:
    return ((r >= centre - w_lo) & (r <= centre + w_hi))


def _bootstrap_skill(g: pd.DataFrame, rng: np.random.Generator) -> tuple:
    """90% CI on the skill score, resampling forecast dates with replacement."""
    dates = g["forecast_date"].unique()
    by_date = {d: sub for d, sub in g.groupby("forecast_date")}
    out = []
    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(dates, size=len(dates), replace=True)
        s = pd.concat([by_date[d] for d in draw], ignore_index=True)
        r, rh = s["r_actual"].to_numpy(), s["r_hat"].to_numpy()
        naive = _mae(r, np.zeros_like(r))
        if naive <= 0:
            continue
        out.append(1.0 - _mae(r, rh) / naive)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-dir", default=str(ARCHIVE_DIR),
                    help="price-archive directory (use the CANONICAL clone)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    df = _load(Path(args.archive_dir))
    rng = np.random.default_rng(RNG_SEED)
    results = []

    print(f"served cohort (>=${SERVED_MIN_PRICE:.0f}): {len(df):,} resolved rows, "
          f"{df['forecast_date'].nunique()} forecast dates\n")
    hdr = (f"{'h':>4} {'rows':>8} {'dates':>6} {'MAE_gbm':>9} {'MAE_naive':>10} "
           f"{'skill':>8} {'90% CI':>18} {'cov_gbm':>8} {'cov_naive':>10}")
    print(hdr)
    print("-" * len(hdr))

    for h, g in df.groupby("horizon_days"):
        r = g["r_actual"].to_numpy()
        rh = g["r_hat"].to_numpy()
        w_lo, w_hi = g["w_lo"].to_numpy(), g["w_hi"].to_numpy()
        mae_gbm, mae_naive = _mae(r, rh), _mae(r, np.zeros_like(r))
        skill = 1.0 - mae_gbm / mae_naive if mae_naive > 0 else float("nan")
        lo, hi = _bootstrap_skill(g, rng)
        cov_gbm = float(_covered(r, rh, w_lo, w_hi).mean())
        cov_naive = float(_covered(r, np.zeros_like(r), w_lo, w_hi).mean())
        print(f"{h:>4} {len(g):>8,} {g['forecast_date'].nunique():>6} "
              f"{mae_gbm:>9.5f} {mae_naive:>10.5f} {skill:>+8.4f} "
              f"{'[%+.4f, %+.4f]' % (lo, hi):>18} "
              f"{cov_gbm:>8.3f} {cov_naive:>10.3f}")
        n_dates = g["forecast_date"].nunique()
        if n_dates < MIN_FORECAST_DATES:
            print(f"     ^ h={h}: {n_dates} clean date(s) < MIN_FORECAST_DATES="
                  f"{MIN_FORECAST_DATES} — NOT publishable")
        results.append(dict(horizon=int(h), rows=int(len(g)),
                            dates=int(g["forecast_date"].nunique()),
                            mae_gbm=mae_gbm, mae_naive=mae_naive, skill=skill,
                            skill_ci90=[lo, hi],
                            coverage_gbm=cov_gbm, coverage_naive=cov_naive,
                            median_abs_r_hat=float(np.median(np.abs(rh))),
                            mean_half_width=float(np.mean(w_lo + w_hi) / 2)))

    print("\nskill = 1 - MAE_gbm/MAE_naive; >0 means the GBM centre beats the "
          "random walk.\ncoverage is at IDENTICAL width (the band's half-widths "
          "moved onto each centre).")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
