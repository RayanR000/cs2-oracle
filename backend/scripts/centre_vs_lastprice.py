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

* Data: rows production actually SERVED and then resolved against the realised
  price. No replay, no artifact, no retrain, so no flag-matching hazard.
  DEFAULT IS PROD POSTGRES (read-only), per the `archive-reads` rule: neither
  copy of `ops/forecast_outcomes.parquet` is the full scored panel. The durable
  CI-written archive is fresh and its cells are complete but SHALLOW — measured
  2026-08-25 it holds 1-2 fewer clean dates per horizon than prod (h=3: 14 vs
  16, h=7: 15 vs 17, h=14: 10 vs 11) — which matters most for the maturity gate,
  where a shallow read under-reports readiness. `--archive-dir` still reads a
  Parquet copy for an offline or historical cross-check; the local working copy
  is the dangerous one and should never be used for a panel figure.
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

from backtest.scoring import MIN_FORECAST_DATES, excluded_forecast_date

ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "price-archive"
SERVED_MIN_PRICE = 1.0  # the >=$1 cohort the forecast is actually served on
N_BOOTSTRAP = 1000
RNG_SEED = 42


#: One column list, two stores. The WHERE clause is valid in both dialects.
_SELECT = """
    SELECT item_id, forecast_date, horizon_days, base_price, actual_price,
           current_price,
           predicted_price_low, predicted_price_mid, predicted_price_high,
           model_version, base_stale_run_days
    FROM {source}
    WHERE actual_price IS NOT NULL
      AND base_price > 0 AND actual_price > 0
      AND predicted_price_mid IS NOT NULL
      AND base_price >= {min_price}
"""

_COLUMNS = [
    "item_id",
    "forecast_date",
    "horizon_days",
    "base_price",
    "actual_price",
    "current_price",
    "predicted_price_low",
    "predicted_price_mid",
    "predicted_price_high",
    "model_version",
    "base_stale_run_days",
]


def _read_db() -> pd.DataFrame:
    """The scored panel from prod Postgres, read-only."""
    from database import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        sql = _SELECT.format(source="forecast_outcomes", min_price=SERVED_MIN_PRICE)
        rows = db.execute(text(sql)).fetchall()
    finally:
        db.close()
    return pd.DataFrame(rows, columns=_COLUMNS)


def _read_parquet(archive_dir: Path) -> pd.DataFrame:
    import duckdb

    p = archive_dir / "ops" / "forecast_outcomes.parquet"
    return duckdb.connect().sql(_SELECT.format(source=f"read_parquet('{p}')", min_price=SERVED_MIN_PRICE)).fetchdf()


def _load(archive_dir: Path | None) -> pd.DataFrame:
    df = _read_parquet(archive_dir) if archive_dir is not None else _read_db()
    print(f"source: {'parquet ' + str(archive_dir) if archive_dir else 'prod postgres'}")
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


def _covered(r: np.ndarray, centre: np.ndarray, w_lo: np.ndarray, w_hi: np.ndarray) -> np.ndarray:
    return (r >= centre - w_lo) & (r <= centre + w_hi)


SHRINK_GRID = np.round(np.arange(0.0, 1.05, 0.05), 2)


def _best_lambda(r: np.ndarray, rh: np.ndarray) -> tuple:
    """The shrinkage that minimises MAE, and the MAE there.

    `centre = lambda * r_hat` nests both arms already measured: lambda=0 is the
    random walk and lambda=1 is what production serves. Treating it as a dial
    rather than a keep/retire switch is the point — a negative skill at
    lambda=1 does not imply the centre carries NO information, only that it is
    over-expressed, and the minimiser says by how much.

    Grid, not a solver: MAE in lambda is piecewise linear and convex, so the
    optimum sits at a kink, and a 0.05 grid is finer than the panel can
    resolve anyway (the bootstrap spread below is far wider than one step).
    """
    maes = np.array([_mae(r, lam * rh) for lam in SHRINK_GRID])
    i = int(np.argmin(maes))
    return float(SHRINK_GRID[i]), float(maes[i])


def _bootstrap_lambda(g: pd.DataFrame, rng: np.random.Generator) -> tuple:
    """90% CI on the MAE-minimising lambda, resampling forecast DATES."""
    dates = g["forecast_date"].unique()
    by_date = {d: sub for d, sub in g.groupby("forecast_date")}
    out = []
    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(dates, size=len(dates), replace=True)
        s = pd.concat([by_date[d] for d in draw], ignore_index=True)
        lam, _ = _best_lambda(s["r_actual"].to_numpy(), s["r_hat"].to_numpy())
        out.append(lam)
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))


def _shrink_report(df: pd.DataFrame, rng: np.random.Generator) -> list:
    """Per horizon: the MAE curve over lambda, its minimiser, and coverage there.

    Coverage is reported at the SERVED width, so this answers the deployable
    question — "if I shrink the centre and change nothing else, what happens to
    the band?" — rather than a matched-coverage one.
    """
    rows = []
    print("\n=== centre shrinkage: centre = lambda * r_hat ===")
    print("lambda=0 is the random walk, lambda=1 is what production serves.\n")
    hdr = (
        f"{'h':>4} {'lam*':>6} {'90% CI':>14} {'MAE(lam*)':>10} {'MAE(1)':>9} "
        f"{'MAE(0)':>9} {'gain vs 1':>10} {'gain vs 0':>10} {'cov(lam*)':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for h, g in df.groupby("horizon_days"):
        r, rh = g["r_actual"].to_numpy(), g["r_hat"].to_numpy()
        w_lo, w_hi = g["w_lo"].to_numpy(), g["w_hi"].to_numpy()
        lam, mae_lam = _best_lambda(r, rh)
        mae_1, mae_0 = _mae(r, rh), _mae(r, np.zeros_like(r))
        lo, hi = _bootstrap_lambda(g, rng)
        cov = float(_covered(r, lam * rh, w_lo, w_hi).mean())
        print(
            f"{h:>4} {lam:>6.2f} {'[%.2f, %.2f]' % (lo, hi):>14} "
            f"{mae_lam:>10.5f} {mae_1:>9.5f} {mae_0:>9.5f} "
            f"{1 - mae_lam / mae_1:>+10.4f} {1 - mae_lam / mae_0:>+10.4f} "
            f"{cov:>10.3f}"
        )
        rows.append(
            dict(
                horizon=int(h),
                lam_star=lam,
                lam_ci90=[lo, hi],
                mae_lam=mae_lam,
                mae_served=mae_1,
                mae_naive=mae_0,
                gain_vs_served=1 - mae_lam / mae_1,
                gain_vs_naive=1 - mae_lam / mae_0,
                coverage_at_lam=cov,
                curve={float(l): _mae(r, l * rh) for l in SHRINK_GRID},
            )
        )
    print("\ngain vs 1 = what shrinking buys over today's served centre.")
    print(
        "gain vs 0 = what the shrunk centre buys over predicting no move; if "
        "the CI on lam* covers 0,\n            the centre is not "
        "distinguishable from carrying no information at all."
    )
    return rows


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
    ap.add_argument(
        "--archive-dir",
        default=None,
        help="read a price-archive Parquet copy instead of prod Postgres. Shallower; for offline cross-checks only.",
    )
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--shrink",
        action="store_true",
        help="sweep centre = lambda * r_hat and report the MAE-minimising lambda per horizon",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="exit 1 unless EVERY horizon has >= MIN_FORECAST_DATES "
        "clean dates. The panel was 8-11 dates when this was "
        "written, so the verdict is not quotable yet; run this "
        "to ask whether it has matured without reading a table.",
    )
    args = ap.parse_args()

    df = _load(Path(args.archive_dir) if args.archive_dir else None)
    rng = np.random.default_rng(RNG_SEED)
    results = []

    print(
        f"served cohort (>=${SERVED_MIN_PRICE:.0f}): {len(df):,} resolved rows, "
        f"{df['forecast_date'].nunique()} forecast dates\n"
    )
    hdr = (
        f"{'h':>4} {'rows':>8} {'dates':>6} {'MAE_gbm':>9} {'MAE_naive':>10} "
        f"{'skill':>8} {'90% CI':>18} {'cov_gbm':>8} {'cov_naive':>10}"
    )
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
        print(
            f"{h:>4} {len(g):>8,} {g['forecast_date'].nunique():>6} "
            f"{mae_gbm:>9.5f} {mae_naive:>10.5f} {skill:>+8.4f} "
            f"{'[%+.4f, %+.4f]' % (lo, hi):>18} "
            f"{cov_gbm:>8.3f} {cov_naive:>10.3f}"
        )
        n_dates = g["forecast_date"].nunique()
        if n_dates < MIN_FORECAST_DATES:
            print(f"     ^ h={h}: {n_dates} clean date(s) < MIN_FORECAST_DATES={MIN_FORECAST_DATES} — NOT publishable")
        results.append(
            dict(
                horizon=int(h),
                rows=len(g),
                dates=int(g["forecast_date"].nunique()),
                mae_gbm=mae_gbm,
                mae_naive=mae_naive,
                skill=skill,
                skill_ci90=[lo, hi],
                coverage_gbm=cov_gbm,
                coverage_naive=cov_naive,
                median_abs_r_hat=float(np.median(np.abs(rh))),
                mean_half_width=float(np.mean(w_lo + w_hi) / 2),
            )
        )

    shrink_rows = _shrink_report(df, rng) if args.shrink else None

    short = [r for r in results if r["dates"] < MIN_FORECAST_DATES]
    if short:
        worst = min(r["dates"] for r in short)
        print(
            f"\nPANEL IMMATURE: {len(short)} of {len(results)} horizon(s) below "
            f"MIN_FORECAST_DATES={MIN_FORECAST_DATES} (shallowest: {worst} "
            f"dates). The signs above are informative; the magnitudes are not "
            f"quotable."
        )
        print(
            "The panel only grows when the forecast chain runs. If "
            "item_forecasts has stopped advancing, no amount of waiting "
            "matures it — check the workflows first."
        )
    else:
        print(f"\nPANEL MATURE: every horizon has >= {MIN_FORECAST_DATES} clean dates. This verdict is quotable.")

    print(
        "\nskill = 1 - MAE_gbm/MAE_naive; >0 means the GBM centre beats the "
        "random walk.\ncoverage is at IDENTICAL width (the band's half-widths "
        "moved onto each centre)."
    )

    if args.json_out:
        payload = {"centre_vs_naive": results}
        if shrink_rows is not None:
            payload["shrinkage"] = shrink_rows
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")
    return 1 if (args.gate and short) else 0


if __name__ == "__main__":
    raise SystemExit(main())
