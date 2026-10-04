"""Outside baseline: the served band against off-the-shelf 80% intervals.

QUESTION (for PORTFOLIO.md, not for serving): on the served panel, does the served
band beat what a reviewer would build in an afternoon?

ARMS, scored on the same rows:
  served — the archived band, rebased to the model's own quote exactly as
           `scripts/archive/conditional_coverage_by_stratum.py` does.
  naive  — random walk; the band is the empirical 10th-90th percentile of the
           item's trailing h-day returns (365-day window). The floor.
  ets    — statsforecast AutoETS on log price with `ConformalIntervals` at 80%.
           The recognisable off-the-shelf model. Needs the `baseline` extra.

HONESTY:
  * Baselines see only prices at or before the forecast date.
  * They are fit on the scorer's own basis: `load_voted_prices` (the loader
    `backtest_accuracy.py` resolves actuals through), smoothed to the trailing
    median of three observations as `resolve_anchors` does for base and actual.
  * Everything is scored as a ratio to `base_price`, so items are comparable.
  * Rows where any arm is missing are dropped from every arm.
  * ETS costs ~25 ms a fit; the served universe (~1,300 items at >=$1) runs in
    ~15 minutes on 9 cores. `--items` takes a seeded, tier-stratified sample.

METRICS per horizon: coverage, median half-width, mean interval score (alpha=0.2,
`backtest.scoring.interval_score`), and served - baseline interval score as a
paired per-date difference with a date-bootstrap CI (negative = served better).
The served band over-covers, so each baseline's width is also shown rescaled to
the served coverage on the same rows. That scale is fitted in hindsight: it
reads "how wide would this baseline be at the served coverage", not a forecast.

Run (from backend/):
    venv/bin/python scripts/outside_baseline.py --archive-dir <price-archive copy>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.serving_policy import MIN_SERVED_PRICE_USD as SERVED_MIN_PRICE
from backtest.price_resolution import SMOOTH_WINDOW, load_voted_prices
from backtest.scoring import excluded_forecast_date, interval_score, price_tier

ALPHA = 0.2
LEVEL = 80  # = round((1 - ALPHA) * 100)
HORIZONS = (3, 7, 14, 30)
WINDOW_DAYS = 365
MIN_RETURNS = 30
ETS_MIN_OBS = 200  # ConformalIntervals(h=30, n_windows=5) needs > 150 points
ETS_N_WINDOWS = 5
RNG_SEED = 20261004


# --- pure pieces (tested) -------------------------------------------------------


def smoothed_series(raw: pd.Series) -> pd.Series:
    """Median of the last three observations at each observation, as `resolve_anchors`."""
    raw = raw.sort_index()
    return raw.rolling(SMOOTH_WINDOW, min_periods=1).median()


def trailing_returns(s: pd.Series, h: int, asof: date, window_days: int) -> np.ndarray:
    """Log h-calendar-day returns of `s` whose END date is in (asof - window, asof].

    The start value is the last observation at or before end - h, so gaps in the
    series stretch nothing: a return always spans h calendar days or more.
    """
    s = s[s.index <= asof]
    if len(s) < 2:
        return np.empty(0)
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    vals = np.log(s.to_numpy(dtype=float))
    ends = np.nonzero(idx > pd.Timestamp(asof - timedelta(days=window_days)))[0]
    ns = idx.as_unit("ns").asi8
    starts = np.searchsorted(ns, ns[ends] - h * 86_400 * 10**9, side="right") - 1
    ok = starts >= 0
    return vals[ends[ok]] - vals[starts[ok]]


def naive_band(returns: np.ndarray, alpha: float = ALPHA, min_returns: int = MIN_RETURNS):
    """Central (alpha/2, 1 - alpha/2) quantiles of the trailing log returns, or None."""
    if len(returns) < min_returns:
        return None
    lo, hi = np.quantile(returns, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def rescale_to_coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, target: float) -> float:
    """Scale k on a zero-centred log band so mean coverage of y equals `target`."""
    a, b = 1e-3, 100.0
    for _ in range(80):
        k = (a + b) / 2
        if np.mean((y >= k * lo) & (y <= k * hi)) < target:
            a = k
        else:
            b = k
    return (a + b) / 2


def paired_date_diff(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 2000, seed: int = RNG_SEED):
    """Mean over dates of the per-date mean (a - b), with a 90% date-bootstrap CI."""
    per = df.assign(_d=df[col_a] - df[col_b]).groupby("forecast_date")["_d"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    boots = per[rng.integers(0, len(per), (n_boot, len(per)))].mean(axis=1)
    lo, hi = np.percentile(boots, [5, 95])
    return float(per.mean()), float(lo), float(hi)


# --- panel ---------------------------------------------------------------------


def load_panel(archive_dir: Path) -> pd.DataFrame:
    """Served cohort, exclusions applied, served band rebased to the model's quote."""
    import duckdb

    ops = archive_dir / "ops" / "forecast_outcomes.parquet"
    df = (
        duckdb.connect()
        .sql(
            f"""
            SELECT item_slug, forecast_date, horizon_days, base_price, current_price, actual_price,
                   predicted_price_low, predicted_price_mid, predicted_price_high
            FROM read_parquet('{ops}')
            WHERE actual_price > 0 AND base_price >= {SERVED_MIN_PRICE}
              AND predicted_price_low IS NOT NULL AND predicted_price_mid IS NOT NULL
              AND predicted_price_high IS NOT NULL
            """
        )
        .fetchdf()
    )
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    df = df[df["forecast_date"].map(excluded_forecast_date).isna()].reset_index(drop=True)

    b = df["base_price"].to_numpy(dtype=float)
    q = df["current_price"].to_numpy(dtype=float)
    q = np.where(np.isnan(q) | (q <= 0), b, q)
    mid = df["predicted_price_mid"].to_numpy(dtype=float)
    r_hat = mid / q - 1.0
    df["y"] = df["actual_price"].to_numpy() / b  # outcome as a ratio to base
    df["served_lo"] = 1.0 + r_hat - (mid - df["predicted_price_low"].to_numpy()) / q
    df["served_hi"] = 1.0 + r_hat + (df["predicted_price_high"].to_numpy() - mid) / q
    df["tier"] = [price_tier(p) for p in b]
    return df


def sample_items(panel: pd.DataFrame, n: int, seed: int = RNG_SEED) -> list[str]:
    """Seeded sample, allocated across price tiers in proportion to item counts."""
    items = panel.groupby("item_slug")["tier"].agg(lambda t: t.mode().iloc[0]).reset_index()
    if n >= len(items):
        return sorted(items["item_slug"])
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for _, g in items.groupby("tier"):
        k = max(1, round(n * len(g) / len(items)))
        out.extend(rng.choice(g["item_slug"].to_numpy(), size=min(k, len(g)), replace=False))
    return sorted(out)


# --- arms ----------------------------------------------------------------------


def _item_job(args):
    """All baseline bands for one item: {(forecast_date, h): {arm: (lo, hi)}} in log-ratio units."""
    slug, raw, dates, with_ets = args
    s = smoothed_series(raw)
    out: dict = {}
    for fd in dates:
        bands: dict = {}
        for h in HORIZONS:
            nb = naive_band(trailing_returns(s, h, fd, WINDOW_DAYS))
            if nb is not None:
                bands.setdefault(h, {})["naive"] = nb
        if with_ets:
            ets = _ets_bands(s, fd)
            for h, band in ets.items():
                bands.setdefault(h, {})["ets"] = band
        for h, arms in bands.items():
            out[(fd, h)] = arms
    return slug, out


def _ets_bands(s: pd.Series, asof: date) -> dict:
    """AutoETS + conformal 80% on the daily log series up to `asof`; log-ratio to the last value."""
    from statsforecast.models import AutoETS
    from statsforecast.utils import ConformalIntervals

    hist = s[s.index <= asof]
    hist = hist[hist.index > asof - timedelta(days=WINDOW_DAYS)]
    if len(hist) < 2:
        return {}
    daily = hist.copy()
    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index))
    daily = daily.resample("D").last().ffill()
    if len(daily) < ETS_MIN_OBS:
        return {}
    y = np.log(daily.to_numpy(dtype=float))
    hmax = max(HORIZONS)
    try:
        model = AutoETS(season_length=1, prediction_intervals=ConformalIntervals(h=hmax, n_windows=ETS_N_WINDOWS))
        f = model.forecast(y=y, h=hmax, level=[LEVEL])
    except Exception:
        return {}
    lvl = LEVEL
    last = y[-1]
    return {h: (float(f[f"lo-{lvl}"][h - 1] - last), float(f[f"hi-{lvl}"][h - 1] - last)) for h in HORIZONS}


def build_baselines(archive_dir: Path, panel: pd.DataFrame, slugs: list[str], with_ets: bool, workers: int):
    sub = panel[panel["item_slug"].isin(slugs)]
    dates = sorted(sub["forecast_date"].unique())
    voted = load_voted_prices(archive_dir, slugs, min(dates), max(dates), max_span_days=WINDOW_DAYS + max(HORIZONS))
    print(f"voted prices: {len(voted):,} item-days for {voted['item_id'].nunique():,} items")
    want = sub.groupby("item_slug")["forecast_date"].apply(lambda d: sorted(set(d))).to_dict()
    jobs = []
    for slug, g in voted.groupby("item_id"):
        if slug in want:
            jobs.append((slug, g.set_index("date")["price"].astype(float), want[slug], with_ets))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (slug, out) in enumerate(ex.map(_item_job, jobs, chunksize=8), 1):
            for (fd, h), arms in out.items():
                row = {"item_slug": slug, "forecast_date": fd, "horizon_days": h}
                for arm, (lo, hi) in arms.items():
                    row[f"{arm}_lo"], row[f"{arm}_hi"] = lo, hi
                rows.append(row)
            if i % 200 == 0:
                print(f"  {i:,}/{len(jobs):,} items")
    return pd.DataFrame(rows)


# --- report --------------------------------------------------------------------


def score(panel: pd.DataFrame, base: pd.DataFrame, arms: list[str]) -> dict:
    df = panel.merge(base, on=["item_slug", "forecast_date", "horizon_days"], how="inner")
    need = [f"{a}_{e}" for a in arms for e in ("lo", "hi")]
    n0 = len(df)
    df = df.dropna(subset=need).reset_index(drop=True)
    print(f"\ncommon rows: {len(df):,} of {n0:,} panel rows with a baseline attempt")
    for a in arms:  # log ratio -> price ratio to base
        df[f"{a}_lo"], df[f"{a}_hi"] = np.exp(df[f"{a}_lo"]), np.exp(df[f"{a}_hi"])
    out: dict = {}
    print(
        f"\n{'h':>3} {'arm':>7} {'rows':>7} {'dates':>5} {'cover':>7} {'med_hw%':>8} {'IS':>7}"
        f" {'served-arm IS [90% CI]':>28} {'hw% @served cov':>16}"
    )
    for h, g in df.groupby("horizon_days"):
        g = g.copy()
        res: dict = {}
        served_cov = float(np.mean((g["y"] >= g["served_lo"]) & (g["y"] <= g["served_hi"])))
        for a in ["served", *arms]:
            lo, hi, y = g[f"{a}_lo"].to_numpy(), g[f"{a}_hi"].to_numpy(), g["y"].to_numpy()
            g[f"is_{a}"] = interval_score(lo, hi, y, alpha=ALPHA)
            r = {
                "rows": len(g),
                "dates": int(g["forecast_date"].nunique()),
                "coverage": float(np.mean((y >= lo) & (y <= hi))),
                "median_half_width": float(np.median((hi - lo) / 2)),
                "interval_score": float(g[f"is_{a}"].mean()),
            }
            line = (
                f"{h:>3} {a:>7} {r['rows']:>7,} {r['dates']:>5} {r['coverage'] * 100:>6.1f}% "
                f"{r['median_half_width'] * 100:>8.2f} {r['interval_score']:>7.4f}"
            )
            if a != "served":
                est, ci_lo, ci_hi = paired_date_diff(g, "is_served", f"is_{a}")
                r["served_minus_arm_is"] = [est, ci_lo, ci_hi]
                ly, llo, lhi = np.log(y), np.log(lo), np.log(hi)
                k = rescale_to_coverage(llo, lhi, ly, served_cov)
                r["k_at_served_coverage"] = k
                r["median_half_width_at_served_coverage"] = float(np.median((np.exp(k * lhi) - np.exp(k * llo)) / 2))
                line += f" {est:>+9.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {r['median_half_width_at_served_coverage'] * 100:>15.2f}"
            print(line)
            res[a] = r
        out[int(h)] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-dir", required=True, type=Path, help="price-archive copy (prices-*.parquet + ops/)")
    ap.add_argument(
        "--items", type=int, default=None, help="tier-stratified item sample size (default: every served item)"
    )
    ap.add_argument(
        "--since", type=date.fromisoformat, help="score only forecast dates on or after this (band-era split)"
    )
    ap.add_argument("--until", type=date.fromisoformat, help="score only forecast dates on or before this")
    ap.add_argument("--no-ets", action="store_true", help="naive arm only (no statsforecast needed)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    panel = load_panel(args.archive_dir)
    if args.since:
        panel = panel[panel["forecast_date"] >= args.since].reset_index(drop=True)
    if args.until:
        panel = panel[panel["forecast_date"] <= args.until].reset_index(drop=True)
    print(f"served panel (>=${SERVED_MIN_PRICE}): {len(panel):,} rows, {panel['forecast_date'].nunique()} dates")
    slugs = sample_items(panel, args.items) if args.items else sorted(panel["item_slug"].unique())
    print(f"item sample: {len(slugs):,} (seed {RNG_SEED})")
    arms = ["naive"] if args.no_ets else ["naive", "ets"]
    base = build_baselines(args.archive_dir, panel, slugs, with_ets=not args.no_ets, workers=args.workers)
    if base.empty:
        raise SystemExit("no baseline bands built — check the archive copy")
    result = score(panel[panel["item_slug"].isin(slugs)], base, arms)
    if args.json_out:
        args.json_out.write_text(json.dumps({"seed": RNG_SEED, "items": len(slugs), "by_horizon": result}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
