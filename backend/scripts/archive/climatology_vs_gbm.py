#!/usr/bin/env python3
"""The null the programme never had: does the GBM band beat climatology?

deep-model-review §9.1 / §12.12. The product is almost entirely a SCALE estimate
(median |r_hat| moves the interval <5% of its own width), so the question that
gates every other modelling item is: at matched coverage, is the GBM's band
narrower than a per-item climatological band that uses no features and no
booster? If climatology is not wider (within +/- SE), the GBM is decorative.

METHOD (read-only, no retrain, no serving-path code):

* Data: the trained artifact's `engineered_data.parquet` (the OOF population).
  Forward h-day returns are computed by exact calendar-date lookup, matching
  `prepare_targets`.
* GBM band: half-width w_gbm = clip(price_std_60d / price, floor, cap) — the
  exact `conformal.sigma_from_columns` scale the served band multiplies by q_hat.
* Climatology band: per item, the trailing dispersion of its OWN h-day returns,
  taken over the CALIBRATION period only (causal), shrunk toward the pooled
  dispersion of its price tier by n_i/(n_i+K). Items unseen in calibration fall
  back to the tier pool. No features, no booster.
* MATCHED COVERAGE. Both bands are centred at 0 (the review's null; the centre is
  <5% of width). For a per-row half-width w_i, the multiplier that yields exactly
  80% marginal coverage on the test rows is lambda* = quantile(|r_i|/w_i, 0.80),
  and the mean half-width at that coverage is lambda* * mean(w_i). So both methods
  are compared at IDENTICAL 80% coverage and only their width differs — this is
  purely a test of how well each allocates width across items.
* Split: earliest 70% of anchor dates calibrate the climatology and are excluded
  from the test read; the latest 30% are held out. lambda* is set on the held-out
  rows for BOTH methods equally, so neither gets a calibration-level edge.
* Uncertainty: a bootstrap over test DATES (returns are correlated within a date
  through the market factor), reporting the width ratio clim/gbm with a 90% CI.

Run: backend/venv/bin/python scripts/climatology_vs_gbm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARTIFACT = Path(__file__).resolve().parent.parent / "models" / "saved_models"
ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "price-archive"
SERVED_MIN_PRICE = 1.0  # the >=$1 cohort the band is actually served on
HORIZONS = (3, 7, 14, 30)
TARGET_COVERAGE = 0.80
CALIB_FRACTION = 0.70  # (artifact mode) earliest share fitting climatology
SCALE_FRACTION = 0.60  # archive mode: climatology scale fit before this
CAL_FRACTION = 0.80  # archive mode: conformal lambda fit in [scale, cal)
SHRINK_K = 20  # n_i/(n_i+K) shrink of item -> tier dispersion
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _load() -> pd.DataFrame:
    import duckdb

    p = ARTIFACT / "engineered_data.parquet"
    df = duckdb.connect().sql(f"SELECT item_id, date, price, price_std_60d FROM read_parquet('{p}')").fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    # Vectorized backtest.scoring.price_tier (liquidity bands, not display).
    px = df["price"].to_numpy()
    df["tier"] = np.select([px >= 1000, px >= 100, px >= 20, px >= 5, px >= 1], [5, 4, 3, 2, 1], default=0)
    return df


def _served_cohort_slugs() -> set:
    """The trainable/backfilled cohort production serves from — the archive
    derivation behind `_resolve_backfilled_slugs("serve")`: pre-2026 backfill
    minus iflow-only history. Pure function of the archive (the DB's
    `is_backfilled` is the synthetic local fixture, so it is not read)."""
    import duckdb
    from db.archive import prices_relation
    from models.item_parser import archive_universe_sql_filter

    con = duckdb.connect()
    try:
        rel = prices_relation(con, str(ARCHIVE_DIR), columns=["item_slug", "day", "source"])
        rows = con.sql(f"""
            SELECT DISTINCT item_slug FROM {rel} sub
            WHERE day < DATE '2026-01-01'
              AND source IS DISTINCT FROM 'buff_iflow'
              AND {archive_universe_sql_filter("sub.item_slug", "sub.source")}
        """).fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


def _load_archive(start: str, served_only: bool = False, label: str = "voted") -> pd.DataFrame:
    """Prod-faithful confirmation: vote the durable archive through the SAME
    static method the production loader calls, then engineer `price_std_60d`
    exactly as `engineer_features` does (row-based rolling 60, min_periods=1).
    Restricted to the >=$1 cohort, and to the served backfilled cohort when
    `served_only`. No model artifact is consulted here.

    `label='within-source'` replaces the voted single-price series with a
    within-source-chained index (`check_label_seams.within_source_index`) whose
    every daily step is measured inside one source, so the seams that fabricate
    market-wide moves on 2026-03-22/07-09/07-10 never enter the price, the
    rolling dispersion, or the forward return. The cohort and row count are held
    identical to the voted basis, so the two runs are the same measurement one
    label apart.

    Caveat, measured 2026-08-22: on the BROAD >=$1 universe the within-source
    index leaves a handful (<0.1%) of near-zero-price residual items whose
    re-anchored level compounds to implausible forward returns (up to ~7e6% at
    h=14), and because the gate reports a MEAN half-width those few items
    dominate the h>=14 broad-cohort climatology width. The per-item median and
    tails are unaffected (in fact tighter than voted). Run `--served-cohort`
    for the honest comparison: that is the cohort the band is served on, it
    excludes the residual tail, and both labels are stable there."""
    import duckdb
    from db.archive import prices_relation
    from models.forecaster import ItemForecaster
    from models.item_parser import archive_universe_sql_filter

    served = _served_cohort_slugs() if served_only else None

    con = duckdb.connect()
    try:
        rel = prices_relation(con, str(ARCHIVE_DIR), columns=["item_slug", "day", "mean_price", "volume", "source"])
        raw = con.sql(f"""
            SELECT item_slug AS item_id, day AS date, mean_price AS price,
                   volume, source
            FROM {rel} sub
            WHERE day >= DATE '{start}'
              AND (source IS NULL OR source NOT LIKE 'historical_fallback:%')
              AND {archive_universe_sql_filter("sub.item_slug", "sub.source")}
        """).fetchdf()
    finally:
        con.close()

    if served is not None:
        raw = raw[raw["item_id"].isin(served)].copy()

    raw["date"] = pd.to_datetime(raw["date"])
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    raw = raw.dropna(subset=["price"])

    voted = ItemForecaster._apply_multi_source_voting(raw)
    voted["date"] = pd.to_datetime(voted["date"])
    voted = voted.sort_values(["item_id", "date"]).reset_index(drop=True)
    if label == "within-source":
        from scripts.check_label_seams import within_source_index

        clean = within_source_index(raw, voted)
        voted = voted.drop(columns=["price"]).merge(clean, on=["item_id", "date"], how="inner")
        voted = voted.sort_values(["item_id", "date"]).reset_index(drop=True)
    # price_std_60d as production engineers it: row-based rolling std (forecaster
    # .py:2184). Row-based is a known defect, but we confirm the SERVED band.
    voted["price_std_60d"] = (
        voted.groupby("item_id")["price"].rolling(60, min_periods=1).std().reset_index(level=0, drop=True)
    )
    voted = voted[voted["price"] >= SERVED_MIN_PRICE].copy()
    px = voted["price"].to_numpy()
    voted["tier"] = np.select([px >= 1000, px >= 100, px >= 20, px >= 5, px >= 1], [5, 4, 3, 2, 1], default=0)
    return voted[["item_id", "date", "price", "price_std_60d", "tier"]]


def _forward_return_pct(df: pd.DataFrame, horizon: int) -> pd.Series:
    """(price[date+h]/price[date] - 1)*100 by exact calendar-date lookup."""
    future = df[["item_id", "date", "price"]].copy()
    future["date"] = future["date"] - pd.Timedelta(days=horizon)
    future = future.rename(columns={"price": "price_fwd"})
    merged = df[["item_id", "date", "price"]].merge(future, on=["item_id", "date"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (merged["price_fwd"] / merged["price"] - 1.0) * 100.0
    return r


def _gbm_sigma(df: pd.DataFrame, clip: dict) -> np.ndarray:
    from models import conformal

    return conformal.sigma_from_columns(
        df["price_std_60d"].to_numpy(),
        df["price"].to_numpy(),
        floor=clip["floor"],
        cap=clip["cap"],
        fallback=clip["fallback"],
    )


def _sigma_clip(df: pd.DataFrame, source: str, clip_source: str = "auto") -> dict:
    """Artifact mode reads the persisted bounds; archive mode derives them the
    same way training does (`conformal.sigma_bounds`, 1st/99th pct), fallback =
    median raw sigma. `clip_source=artifact` forces the persisted PROD bounds
    even in archive mode, so a loose cohort cap cannot inflate the GBM width."""
    if source == "artifact" or clip_source == "artifact":
        return json.load(open(ARTIFACT / "meta.json"))["sigma_clip"]
    from models import conformal

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = df["price_std_60d"].to_numpy() / df["price"].to_numpy()
    floor, cap = conformal.sigma_bounds(raw)
    good = np.isfinite(raw) & (raw > 0)
    return {"floor": floor, "cap": cap, "fallback": float(np.median(raw[good]))}


def _climatology_halfwidth(calib: pd.DataFrame, test: pd.DataFrame, col: str) -> np.ndarray:
    """Per-test-row half-width from the item's trailing |h-day return| quantile,
    shrunk toward its price tier's pooled quantile. Fit on calibration only."""
    q = TARGET_COVERAGE

    def _absq(s: pd.Series) -> float:
        a = s.abs().to_numpy()
        a = a[np.isfinite(a)]
        return float(np.quantile(a, q)) if a.size else np.nan

    item_hw = calib.groupby("item_id")[col].apply(_absq)
    item_n = calib.groupby("item_id")[col].apply(lambda s: int(s.notna().sum()))
    tier_hw = calib.groupby("tier")[col].apply(_absq)
    global_hw = _absq(calib[col])

    def _shrunk(item_id, tier):
        base = tier_hw.get(tier, np.nan)
        if not np.isfinite(base):
            base = global_hw
        raw = item_hw.get(item_id, np.nan)
        n = item_n.get(item_id, 0)
        if not np.isfinite(raw) or n == 0:
            return base
        w = n / (n + SHRINK_K)
        return w * raw + (1 - w) * base

    return test.apply(lambda r: _shrunk(r["item_id"], r["tier"]), axis=1).to_numpy()


def _matched_width(abs_r: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """(lambda*, mean half-width) at exactly TARGET_COVERAGE. lambda* is the
    quantile of |r|/w that puts TARGET_COVERAGE of rows inside lambda*·w."""
    ok = np.isfinite(abs_r) & np.isfinite(w) & (w > 0)
    ar, ww = abs_r[ok], w[ok]
    if ar.size == 0:
        return np.nan, np.nan
    lam = float(np.quantile(ar / ww, TARGET_COVERAGE))
    return lam, lam * float(np.mean(ww))


def _coverage_at_lambda(abs_r: np.ndarray, w: np.ndarray, lam: float) -> float:
    ok = np.isfinite(abs_r) & np.isfinite(w) & (w > 0)
    if not ok.any() or not np.isfinite(lam):
        return np.nan
    return float(np.mean(abs_r[ok] <= lam * w[ok]))


def _run_horizon(df: pd.DataFrame, horizon: int, clip: dict) -> dict:
    d = df.copy()
    d["r"] = _forward_return_pct(d, horizon)
    d["w_gbm"] = _gbm_sigma(d, clip)
    d = d[np.isfinite(d["r"])].copy()

    # Three temporal splits: climatology scale fit on the earliest, lambda
    # (the conformal level) fit on the middle, coverage+width read on the
    # latest — so both the width AND the coverage level are out of sample.
    dates = np.sort(d["date"].unique())
    c1 = dates[int(len(dates) * SCALE_FRACTION)]
    c2 = dates[int(len(dates) * CAL_FRACTION)]
    scale_df = d[d["date"] < c1]
    cal = d[(d["date"] >= c1) & (d["date"] < c2)].copy()
    ev = d[d["date"] >= c2].copy()

    scale_r = scale_df[["item_id", "tier", "r"]].rename(columns={"r": "r_h"})
    for frame in (cal, ev):
        frame["w_clim"] = _climatology_halfwidth(scale_r, frame, "r_h") if not scale_r.empty else np.nan

    def _method(wcol):
        # lambda calibrated to 80% on `cal`, then coverage AND width read on `ev`.
        lam, _ = _matched_width(cal["r"].abs().to_numpy(), cal[wcol].to_numpy())
        w_ev = ev[wcol].to_numpy()
        cov = _coverage_at_lambda(ev["r"].abs().to_numpy(), w_ev, lam)
        ok = np.isfinite(w_ev) & (w_ev > 0)
        width = float(lam * np.mean(w_ev[ok])) if ok.any() and np.isfinite(lam) else np.nan
        return lam, cov, width

    _lam_g, cov_g, _width_g = _method("w_gbm")
    _lam_c, cov_c, _width_c = _method("w_clim")

    # Matched-coverage width ratio on `ev` (both forced to exactly 80% there),
    # with a date-clustered bootstrap — the apples-to-apples width read.
    abs_ev = ev["r"].abs().to_numpy()
    _, mw_g = _matched_width(abs_ev, ev["w_gbm"].to_numpy())
    _, mw_c = _matched_width(abs_ev, ev["w_clim"].to_numpy())
    rng = np.random.default_rng(RNG_SEED)
    by_date = {dt: g for dt, g in ev.groupby("date")}
    keys = list(by_date.keys())
    ratios = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        boot = pd.concat([by_date[keys[i]] for i in pick])
        ar = boot["r"].abs().to_numpy()
        _, g = _matched_width(ar, boot["w_gbm"].to_numpy())
        _, c = _matched_width(ar, boot["w_clim"].to_numpy())
        ratios[b] = c / g if g else np.nan
    lo, hi = np.nanpercentile(ratios, [5, 95])

    return {
        "horizon": horizon,
        "n_eval": len(ev),
        "n_eval_dates": int(ev["date"].nunique()),
        "gbm_cov": round(cov_g, 4),
        "clim_cov": round(cov_c, 4),
        "gbm_mean_width_pct": round(mw_g, 3),
        "clim_mean_width_pct": round(mw_c, 3),
        "width_ratio_clim_over_gbm": round(mw_c / mw_g, 4) if mw_g else None,
        "ratio_ci90": [round(float(lo), 4), round(float(hi), 4)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument(
        "--source",
        choices=("artifact", "archive"),
        default="artifact",
        help="artifact: local trained engineered_data.parquet; archive: prod-faithful vote of the durable archive",
    )
    ap.add_argument("--start", default="2024-01-01", help="archive mode: earliest day to vote")
    ap.add_argument(
        "--clip",
        choices=("auto", "artifact"),
        default="auto",
        help="archive mode: 'artifact' forces the persisted prod sigma clip bounds instead of deriving them",
    )
    ap.add_argument(
        "--served-cohort",
        action="store_true",
        help="archive mode: restrict to the served backfilled cohort "
        "(_resolve_backfilled_slugs), not the broad >=$1 universe",
    )
    ap.add_argument(
        "--label",
        choices=("voted", "within-source"),
        default="voted",
        help="archive mode: 'within-source' rebuilds the price on a "
        "seam-free within-source chained index (the clean label)",
    )
    args = ap.parse_args()

    df = (
        _load_archive(args.start, served_only=args.served_cohort, label=args.label)
        if args.source == "archive"
        else _load()
    )
    clip = _sigma_clip(df, args.source, args.clip)
    cohort = "served" if args.served_cohort else "all"
    lbl = args.label if args.source == "archive" else "artifact"
    print(
        f"[{args.source}/{cohort}/{lbl}] {len(df):,} rows, {df['item_id'].nunique():,} "
        f"items, {df['date'].min().date()}..{df['date'].max().date()} | "
        f"sigma clip floor={clip['floor']:.4f} cap={clip['cap']:.4f}\n"
    )
    print(
        f"{'h':>3} {'n_eval':>9} {'dates':>6} {'GBMcov':>7} {'climcov':>8} "
        f"{'GBM w%':>8} {'clim w%':>9} {'ratio':>7} {'90% CI':>18}"
    )
    for h in args.horizons:
        r = _run_horizon(df, h, clip)
        ci = f"[{r['ratio_ci90'][0]:.3f}, {r['ratio_ci90'][1]:.3f}]"
        print(
            f"{r['horizon']:>3} {r['n_eval']:>9,} {r['n_eval_dates']:>6} "
            f"{r['gbm_cov']:>7} {r['clim_cov']:>8} "
            f"{r['gbm_mean_width_pct']:>8} {r['clim_mean_width_pct']:>9} "
            f"{r['width_ratio_clim_over_gbm']:>7} {ci:>18}"
        )
    print(
        "\nGBMcov/climcov = OUT-OF-SAMPLE marginal coverage (lambda fit on the "
        "middle split, read on the last); target 0.80."
    )
    print(
        "ratio = clim width / GBM width at matched 80% coverage on the eval "
        "split. <=1 (CI incl.) means climatology matches or beats the GBM."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
