#!/usr/bin/env python3
"""The band's last untested arm: does a MAGNITUDE-trained booster beat climatology?

`climatology_vs_gbm.py` retired the served band's scale — a featureless per-item
climatology is ~25-30% narrower at matched 80% coverage, and that holds on the
seam-free within-source label, so `CLIMATOLOGY_SCALE` has served since
2026-08-20. `centre_vs_lastprice.py` then retired the centre: skill is negative
at h=3/7/14 with CIs clear of zero, and the shrinkage sweep puts lambda* at 0.00
with a [0.00, 0.00] CI, so the signed centre carries no information at any scale.

Both of those verdicts are about a model trained on SIGNED returns. That leaves
one arm genuinely untested, and it is the arm the evidence actually points at:
direction is null (AUC 0.51-0.53) but MAGNITUDE is not — P(|move| > cost) is
market-orthogonal and date-stable, and it is the one signal the programme has
found. The served band never used a magnitude model. Its width is
`sigma_from_columns(price_std_60d, price)`, a clipped trailing-vol proxy with no
booster in it at all; the GBM's contribution to width was only ever that proxy.

So the question here is NOT the one climatology_vs_gbm answered. It is: if a
booster is pointed DIRECTLY at |r_h| — the quantity that is actually
predictable — does it allocate width better than a featureless climatology?
Features that were dead for direction (listing count, volume, recency) are
plausible volatility predictors, and this is the gate that says whether any of
that is worth a production retrain.

METHOD (one cheap local booster; no production retrain, no serving-path code):

* Data: the trained artifact's `engineered_data.parquet` (the OOF population),
  the same read `climatology_vs_gbm.py --source artifact` uses. Forward h-day
  returns come from the same exact-calendar-date lookup, in PERCENT, so the
  sigma and climatology arms here are directly comparable to that script's --
  which is the built-in consistency check on this one.
* Three ARMS, all centred at 0 (the review's null; the centre is <5% of width
  and now measured worthless), all compared at matched coverage:
      w_sigma  clip(price_std_60d / price) -- what production served pre-08-20.
      w_clim   per-item trailing |r_h| quantile, shrunk toward its price tier
               by n_i/(n_i+K). No features, no booster. The champion.
      w_mag    LightGBM on |r_h|, quantile objective at alpha=TARGET_COVERAGE,
               using the artifact's own 33 `feature_cols`. Predicting the 80th
               percentile of |move| IS the half-width, so the booster is fit to
               the exact quantity the band needs -- no rescaling of an L2 fit.
* THREE temporal splits, so nothing is read where it was fit:
      [0, 0.60)      fits BOTH the climatology scale AND the magnitude booster
                     (its last 15% is held back for early stopping only).
      [0.60, 0.80)   fits the conformal multiplier lambda for EVERY arm equally.
      [0.80, 1.0]    coverage and width are read here, out of sample twice over.
  An arm cannot win by being better calibrated: lambda is fit per arm on the
  middle split and only WIDTH differs at the matched coverage read.
* Matched coverage: for per-row half-width w_i, lambda* = quantile(|r|/w, 0.80)
  puts exactly 80% of rows inside lambda*·w, and mean width is lambda*·mean(w).
  Identical machinery to `climatology_vs_gbm._matched_width`, on purpose.
* Uncertainty: bootstrap over eval DATES, not rows -- a date's returns are
  correlated through the market factor, so a row bootstrap would understate the
  interval by roughly sqrt(cohort size). Reported as the width ratio
  mag/clim with a 90% CI; <1 with the CI clear of 1 means the magnitude booster
  is genuinely narrower than climatology.
* Horizons are NEVER pooled (the band-geometry eras make a pooled number
  uninterpretable).

READ IT AS A GATE, not a deployment. A single booster is a proxy for the served
ensemble, and the artifact population is not the served cohort. If mag/clim is
>= 1 (CI included), the magnitude target is dead too and the band is climatology
for good. Only a ratio clearly below 1 justifies spending a production retrain.

Run: backend/venv/bin/python scripts/magnitude_vs_climatology.py
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
SERVED_MIN_PRICE = 1.0  # the >=$1 cohort the band is actually served on
HORIZONS = (3, 7, 14, 30)
TARGET_COVERAGE = 0.80
SCALE_FRACTION = 0.60  # climatology scale + booster fit before this
CAL_FRACTION = 0.80  # conformal lambda fit in [scale, cal)
EARLY_STOP_TAIL = 0.15  # tail of the fit split held for early stopping
SHRINK_K = 20  # n_i/(n_i+K) shrink of item -> tier dispersion
N_BOOTSTRAP = 1000
RNG_SEED = 42
N_ESTIMATORS = 3000
EARLY_STOPPING_ROUNDS = 100


def _feature_cols() -> list:
    """The artifact's own served feature set, so this gate tests the features
    production actually has rather than a hand-picked set."""
    return list(json.load(open(ARTIFACT / "meta.json"))["feature_cols"])


#: Columns in `feature_cols` that `engineer_features` derives from the dollar
#: columns and that the persisted frame therefore does not carry. Rebuilt here
#: EXACTLY as forecaster.py does (:2432 for the CVs, :2531 for the MACD pair),
#: because a mismatched definition would silently hand the booster a different
#: feature than production would give it.
def _add_scale_free(df: pd.DataFrame) -> pd.DataFrame:
    px = df["price"].replace(0, np.nan)
    for window in (7, 14, 20, 30, 60):
        src = f"price_std_{window}d"
        if src in df.columns:
            df[f"price_cv_{window}d"] = df[src] / px
    if "macd_line" in df.columns:
        df["macd_line_rel"] = df["macd_line"] / px
    if "macd_histogram" in df.columns:
        df["macd_histogram_rel"] = df["macd_histogram"] / px
    return df


def _served_cohort_slugs() -> set:
    """The trainable/backfilled cohort production actually serves from.

    Imported from `climatology_vs_gbm`, never reimplemented -- the two gates
    MUST agree on what "served" means or their width numbers are not
    comparable. The artifact's `item_id` is the `market_hash_name`, which is
    the archive's `item_slug` (verified: all 5,542 artifact ids intersect the
    archive slug set exactly), so the set applies to this frame unmapped.
    """
    from climatology_vs_gbm import _served_cohort_slugs as _impl

    return _impl()


def _load(feats: list, served_only: bool = False) -> pd.DataFrame:
    import duckdb

    p = ARTIFACT / "engineered_data.parquet"
    have = {r[0] for r in duckdb.connect().sql(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()}
    # The dollar sources needed to rebuild the scale-free features, plus the
    # band's own sigma input.
    need = {"item_id", "date", "price", "price_std_60d", "macd_line", "macd_histogram"} | {
        f"price_std_{w}d" for w in (7, 14, 20, 30, 60)
    }
    need |= {c for c in feats if c in have}
    cols = ", ".join(sorted(need & have))
    df = duckdb.connect().sql(f"SELECT {cols} FROM read_parquet('{p}')").fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    df = _add_scale_free(df)
    df = df[df["price"] >= SERVED_MIN_PRICE].copy()
    if served_only:
        served = _served_cohort_slugs()
        before = df["item_id"].nunique()
        df = df[df["item_id"].isin(served)].copy()
        print(f"served cohort: {df['item_id'].nunique():,} of {before:,} artifact items retained")
    px = df["price"].to_numpy()
    # backtest.scoring.price_tier (liquidity bands, not display), vectorized --
    # same expression climatology_vs_gbm.py uses, so the tier pools match.
    df["tier"] = np.select([px >= 1000, px >= 100, px >= 20, px >= 5, px >= 1], [5, 4, 3, 2, 1], default=0)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} feature(s) unavailable, dropped from the magnitude arm: {missing}")
    return df.reset_index(drop=True)


def _forward_return_pct(df: pd.DataFrame, horizon: int) -> pd.Series:
    """(price[date+h]/price[date] - 1)*100 by exact calendar-date lookup.

    The same label `climatology_vs_gbm._forward_return_pct` builds, so the two
    scripts' shared arms are comparable; the offset is written with an explicit
    unit here only to avoid NumPy's bare-integer timedelta deprecation.
    """
    future = df[["item_id", "date", "price"]].copy()
    future["date"] = future["date"] - pd.Timedelta(horizon, unit="D")
    future = future.rename(columns={"price": "price_fwd"})
    merged = df[["item_id", "date", "price"]].merge(future, on=["item_id", "date"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        return (merged["price_fwd"] / merged["price"] - 1.0) * 100.0


def _sigma_halfwidth(df: pd.DataFrame) -> np.ndarray:
    """The served pre-2026-08-20 band scale, via the production helper and the
    artifact's persisted clip bounds -- not a local re-derivation."""
    from models import conformal

    with open(ARTIFACT / "meta.json") as f:
        clip = json.load(f)["sigma_clip"]
    # sigma is a FRACTION; the labels here are percent, so scale to match.
    return 100.0 * conformal.sigma_from_columns(
        df["price_std_60d"].to_numpy(),
        df["price"].to_numpy(),
        floor=clip["floor"],
        cap=clip["cap"],
        fallback=clip["fallback"],
    )


def _climatology_halfwidth(calib: pd.DataFrame, test: pd.DataFrame, col: str, k: float = SHRINK_K) -> np.ndarray:
    """Per-test-row half-width from the item's trailing |h-day return| quantile,
    shrunk toward its price tier's pooled quantile. Fit on `calib` only.

    Same estimator as `climatology_vs_gbm._climatology_halfwidth`; kept as a
    local copy rather than imported because that module runs work at import.

    `k` is `CLIMATOLOGY_SHRINK_K`, exposed as a parameter because the fold run
    made it the live lever: a global constant (k->inf) was 5-16% narrower than
    the production k=20 at every horizon and every fold.
    """
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
        # k=0 is the pure per-item estimate; k->inf collapses onto the pool.
        w = 1.0 if k <= 0 else n / (n + k)
        return w * raw + (1 - w) * base

    return np.array([_shrunk(i, t) for i, t in zip(test["item_id"], test["tier"])], dtype=float)


def _tuned_params(horizon: int) -> dict:
    """The horizon's own production params, with the objective repointed at the
    magnitude target. Production already trains quantile boosters, so only
    `alpha` and the label change -- keeping everything else pinned means a win
    here is attributable to the TARGET, not to a luckier hyperparameter."""
    with open(ARTIFACT / "meta.json") as f:
        tuned = json.load(f)["tuned_params"]
    base = dict((tuned.get(str(horizon), {}) or {}).get("0.5", {}))
    base.pop("alpha", None)
    base.pop("metric", None)
    base.update(
        objective="quantile",
        alpha=TARGET_COVERAGE,
        metric="quantile",
        verbosity=-1,
        n_jobs=-1,
        feature_pre_filter=False,
    )
    return base


def _fit_magnitude(fit: pd.DataFrame, feats: list, horizon: int, objective: str) -> tuple:
    """Quantile-regress |r_h| on the artifact's features. Returns (model, cols).

    The last EARLY_STOP_TAIL of the fit split (by date) is held out for early
    stopping ONLY, so no round count is chosen on the lambda or eval splits.
    """
    import lightgbm as lgb

    cols = [c for c in feats if c in fit.columns]
    d = fit[np.isfinite(fit["r"])].copy()
    d["y"] = d["r"].abs()
    dates = np.sort(d["date"].unique())
    if len(dates) < 4:
        return None, cols
    cut = dates[int(len(dates) * (1.0 - EARLY_STOP_TAIL))]
    tr, va = d[d["date"] < cut], d[d["date"] >= cut]
    if tr.empty or va.empty:
        tr, va = d, d

    params = _tuned_params(horizon)
    if objective == "l1":
        params.update(objective="regression_l1", metric="l1")
        params.pop("alpha", None)
    model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **params)
    model.fit(
        tr[cols],
        tr["y"],
        eval_set=[(va[cols], va["y"])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model, cols


MIN_FOLD_DATES = 10


def _folds(dates: np.ndarray, n_folds: int) -> list:
    """Walk-forward (fit_end, cal_end, ev_end) date cuts, one triple per fold.

    Fold 0 reproduces the single split exactly (fit < 0.60, lambda in
    [0.60, 0.80), eval >= 0.80). Later folds tile the eval region forward and
    EXPAND the fit window, which is what production would actually do -- so a
    width win has to repeat across eras rather than appear once in the final
    20% of dates. The lambda window keeps its fixed width and always sits
    immediately before its own eval tile, so no fold reads a conformal level
    fitted after the rows it scores.
    """
    n = len(dates)
    if n < MIN_FOLD_DATES or n_folds < 1:
        return []

    def _at(frac: float) -> int:
        return int(min(max(int(n * frac), 0), n - 1))

    cal_width = CAL_FRACTION - SCALE_FRACTION
    tile = (1.0 - CAL_FRACTION) / n_folds
    out = []
    for i in range(n_folds):
        ev_start = CAL_FRACTION + i * tile
        ev_end = CAL_FRACTION + (i + 1) * tile if i < n_folds - 1 else 1.0
        fit_end = ev_start - cal_width
        cuts = (dates[_at(fit_end)], dates[_at(ev_start)], dates[_at(ev_end)] if ev_end < 1.0 else dates[n - 1])
        # A fold whose windows collapse onto each other measures nothing.
        if not (cuts[0] < cuts[1] < cuts[2]):
            return []
        out.append(cuts)
    return out


def _matched_width(abs_r: np.ndarray, w: np.ndarray) -> tuple:
    """(lambda*, mean half-width) at exactly TARGET_COVERAGE."""
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


#: A GLOBAL pooled half-width: one number for every item, the fit split's
#: |r_h| quantile. Featureless AND itemless, so it is a strictly weaker null
#: than climatology. It exists because the walk-forward run showed the booster
#: winning at h=30 after only ONE boosting round -- a near-constant model. If
#: `mag` merely matches `pool`, the "magnitude model" is not using features at
#: all; it is revealing that per-item climatology OVER-FITS item-level noise at
#: long horizons, and the deployable fix would be a bigger CLIMATOLOGY_SHRINK_K,
#: not a booster.
ARMS = ("sigma", "clim", "pool", "mag")


def _run_fold(d: pd.DataFrame, horizon: int, feats: list, objective: str, cuts: tuple, is_last: bool) -> dict:
    """One walk-forward fold: fit < cuts[0], lambda in [cuts[0], cuts[1]),
    eval in [cuts[1], cuts[2]]. The final fold takes the tail inclusively so no
    dates are silently dropped."""
    c0, c1, c2 = cuts
    fit = d[d["date"] < c0].copy()
    cal = d[(d["date"] >= c0) & (d["date"] < c1)].copy()
    ev = (d[d["date"] >= c1] if is_last else d[(d["date"] >= c1) & (d["date"] < c2)]).copy()
    if fit.empty or cal.empty or ev.empty:
        return {"horizon": horizon, "skipped": "a fold window is empty"}

    # --- climatology arm: scale from the fit split only (causal) -------------
    scale_r = fit[["item_id", "tier", "r"]].rename(columns={"r": "r_h"})
    for frame in (cal, ev):
        frame["w_clim"] = _climatology_halfwidth(scale_r, frame, "r_h") if not scale_r.empty else np.nan

    # --- pooled arm: one global constant from the fit split ------------------
    pooled = fit["r"].abs().to_numpy()
    pooled = pooled[np.isfinite(pooled)]
    pool_hw = float(np.quantile(pooled, TARGET_COVERAGE)) if pooled.size else np.nan
    for frame in (cal, ev):
        frame["w_pool"] = pool_hw

    # --- magnitude arm: booster on |r_h|, fit split only ---------------------
    model, cols = _fit_magnitude(fit, feats, horizon, objective)
    for frame in (cal, ev):
        if model is None:
            frame["w_mag"] = np.nan
        else:
            pred = model.predict(frame[cols])
            # A predicted quantile of |r| is a half-width, so it must be
            # positive; the booster can undershoot into <=0 on thin leaves.
            frame["w_mag"] = np.where(np.isfinite(pred) & (pred > 0), pred, np.nan)

    def _method(wcol):
        lam, _ = _matched_width(cal["r"].abs().to_numpy(), cal[wcol].to_numpy())
        w_ev = ev[wcol].to_numpy()
        cov = _coverage_at_lambda(ev["r"].abs().to_numpy(), w_ev, lam)
        ok = np.isfinite(w_ev) & (w_ev > 0)
        width = float(lam * np.mean(w_ev[ok])) if ok.any() and np.isfinite(lam) else np.nan
        return lam, cov, width

    out = {
        "horizon": horizon,
        "ev_start": str(pd.Timestamp(c1).date()),
        "ev_stop": str(ev["date"].max().date()),
        "n_fit": len(fit),
        "n_cal": len(cal),
        "n_eval": len(ev),
        "n_eval_dates": int(ev["date"].nunique()),
        "n_features": len(cols),
        "best_iteration": (int(getattr(model, "best_iteration_", 0) or 0) if model is not None else None),
    }

    abs_ev = ev["r"].abs().to_numpy()
    for arm in ARMS:
        lam, cov, width = _method(f"w_{arm}")
        _, mw = _matched_width(abs_ev, ev[f"w_{arm}"].to_numpy())
        out[f"{arm}_lambda"] = None if not np.isfinite(lam) else round(lam, 4)
        out[f"{arm}_cov"] = None if not np.isfinite(cov) else round(cov, 4)
        out[f"{arm}_width_oos_pct"] = None if not np.isfinite(width) else round(width, 3)
        out[f"{arm}_matched_width_pct"] = None if not np.isfinite(mw) else round(mw, 3)

    # --- matched-coverage width ratios, date-clustered bootstrap -------------
    _, mw_c = _matched_width(abs_ev, ev["w_clim"].to_numpy())
    _, mw_m = _matched_width(abs_ev, ev["w_mag"].to_numpy())
    _, mw_s = _matched_width(abs_ev, ev["w_sigma"].to_numpy())
    out["ratio_mag_over_clim"] = round(mw_m / mw_c, 4) if mw_c and np.isfinite(mw_m) else None
    out["ratio_mag_over_sigma"] = round(mw_m / mw_s, 4) if mw_s and np.isfinite(mw_m) else None
    _, mw_p = _matched_width(abs_ev, ev["w_pool"].to_numpy())
    out["ratio_mag_over_pool"] = round(mw_m / mw_p, 4) if mw_p and np.isfinite(mw_m) else None
    out["ratio_pool_over_clim"] = round(mw_p / mw_c, 4) if mw_c and np.isfinite(mw_p) else None

    rng = np.random.default_rng(RNG_SEED)
    by_date = {dt: g for dt, g in ev.groupby("date")}
    keys = list(by_date.keys())
    ratios = np.full(N_BOOTSTRAP, np.nan)
    for b in range(N_BOOTSTRAP):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        boot = pd.concat([by_date[keys[i]] for i in pick])
        ar = boot["r"].abs().to_numpy()
        _, c = _matched_width(ar, boot["w_clim"].to_numpy())
        _, m = _matched_width(ar, boot["w_mag"].to_numpy())
        if c and np.isfinite(c) and np.isfinite(m):
            ratios[b] = m / c
    lo, hi = np.nanpercentile(ratios, [5, 95])
    out["ratio_ci90"] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def _run_horizon(df: pd.DataFrame, horizon: int, feats: list, objective: str, n_folds: int = 1) -> list:
    """Every fold for one horizon. A single fold is the original single split."""
    d = df.copy()
    d["r"] = _forward_return_pct(d, horizon)
    d["w_sigma"] = _sigma_halfwidth(d)
    d = d[np.isfinite(d["r"])].copy()

    dates = np.sort(d["date"].unique())
    folds = _folds(dates, n_folds)
    if not folds:
        return [{"horizon": horizon, "skipped": f"{len(dates)} date(s) will not support {n_folds} fold(s)"}]
    return [
        _run_fold(d, horizon, feats, objective, cuts, is_last=(i == len(folds) - 1)) for i, cuts in enumerate(folds)
    ]


K_GRID = (0, 10, 20, 40, 80, 160, 320, 10**9)
PROD_K = SHRINK_K  # what models/forecaster.py currently serves


def _run_k_sweep(df: pd.DataFrame, horizon: int, n_folds: int) -> dict:
    """Matched-coverage width of the climatology arm across CLIMATOLOGY_SHRINK_K.

    Booster-free and pool-free: only the shrink constant moves, so the width
    difference is attributable to shrinkage alone. Reported as a ratio against
    the production K, averaged over folds, plus how many folds each K wins --
    a K that wins on average but loses folds is not a deployable K.
    """
    d = df.copy()
    d["r"] = _forward_return_pct(d, horizon)
    d = d[np.isfinite(d["r"])].copy()
    folds = _folds(np.sort(d["date"].unique()), n_folds)
    if not folds:
        return {"horizon": horizon, "skipped": "not enough dates"}

    per_k = {k: [] for k in K_GRID}
    for i, (c0, c1, c2) in enumerate(folds):
        fit = d[d["date"] < c0]
        cal = d[(d["date"] >= c0) & (d["date"] < c1)].copy()
        ev = (d[d["date"] >= c1] if i == len(folds) - 1 else d[(d["date"] >= c1) & (d["date"] < c2)]).copy()
        if fit.empty or cal.empty or ev.empty:
            continue
        scale_r = fit[["item_id", "tier", "r"]].rename(columns={"r": "r_h"})
        abs_ev = ev["r"].abs().to_numpy()
        for k in K_GRID:
            w = _climatology_halfwidth(scale_r, ev, "r_h", k=k)
            _, mw = _matched_width(abs_ev, w)
            per_k[k].append(mw)

    base = np.array(per_k[PROD_K], dtype=float)
    out = {"horizon": horizon, "n_folds": len(base), "per_k": {}}
    for k in K_GRID:
        arr = np.array(per_k[k], dtype=float)
        ok = np.isfinite(arr) & np.isfinite(base)
        if not ok.any():
            continue
        ratio = arr[ok] / base[ok]
        out["per_k"][k] = {
            "mean_width_pct": round(float(np.mean(arr[ok])), 3),
            "mean_ratio_vs_prod_k": round(float(np.mean(ratio)), 4),
            "worst_ratio": round(float(np.max(ratio)), 4),
            "folds_narrower": int(np.sum(ratio < 1.0)),
            "n": int(ok.sum()),
        }
    return out


def _print_k_sweep(rows: list) -> None:
    print(f"\n=== CLIMATOLOGY_SHRINK_K sweep (production K = {PROD_K}) ===")
    print("ratio < 1 means NARROWER than production at the same 80% coverage.\n")
    hdr = f"{'h':>3} {'K':>11} {'width%':>8} {'ratio':>8} {'worst':>8} {'wins':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("skipped"):
            print(f"{r['horizon']:>3}  SKIPPED: {r['skipped']}")
            continue
        n = r["n_folds"]
        for k, v in r["per_k"].items():
            label = "pool(inf)" if k >= 10**9 else str(k)
            star = "  <-- prod" if k == PROD_K else ""
            print(
                f"{r['horizon']:>3} {label:>11} {v['mean_width_pct']:>8} "
                f"{v['mean_ratio_vs_prod_k']:>8.4f} {v['worst_ratio']:>8.4f} "
                f"{str(v['folds_narrower']) + '/' + str(n):>7}{star}"
            )
        print()
    print("width% = mean matched-80%-coverage half-width, averaged over folds.")
    print(
        "ratio  = mean of the PER-FOLD ratio against production K (paired, so "
        "it is not\n         distorted by folds sitting in wider eras)."
    )
    print(
        "worst  = the single worst fold's ratio. A K is only deployable if "
        "worst < 1 too --\n         a mean win with a losing fold is the "
        "shape every refuted band arm had."
    )
    print("wins   = folds where this K is narrower than production K.")


def _print_report(rows: list) -> None:
    hdr = (
        f"{'h':>3} {'eval from':>11} {'n_eval':>9} {'dates':>6} {'rounds':>7} "
        f"{'clim w%':>9} {'pool w%':>9} {'mag w%':>8} "
        f"{'mag/clim':>9} {'90% CI':>18} {'mag/pool':>9} {'pool/clim':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("skipped"):
            print(f"{r['horizon']:>3}  SKIPPED: {r['skipped']}")
            continue
        ci = f"[{r['ratio_ci90'][0]:.3f}, {r['ratio_ci90'][1]:.3f}]" if r.get("ratio_ci90") else "n/a"
        print(
            f"{r['horizon']:>3} {r['ev_start']:>11} {r['n_eval']:>9,} "
            f"{r['n_eval_dates']:>6} "
            f"{r['best_iteration']!s:>7} "
            f"{r['clim_matched_width_pct']!s:>9} "
            f"{r['pool_matched_width_pct']!s:>9} "
            f"{r['mag_matched_width_pct']!s:>8} "
            f"{r['ratio_mag_over_clim']!s:>9} {ci:>18} "
            f"{r['ratio_mag_over_pool']!s:>9} "
            f"{r['ratio_pool_over_clim']!s:>10}"
        )
    print(
        "\nw% = mean half-width in PERCENT at matched 80% coverage on the eval "
        "split.\nmag/clim < 1 means the magnitude booster is narrower than the "
        "featureless\nclimatology; the verdict is the CI, not the point estimate "
        "-- if it covers 1.0\nthe features add nothing to the band and the "
        "magnitude target is dead too."
    )
    print(
        "mag/pool ~ 1.0 means the booster is doing NO feature work -- it has "
        "collapsed to a\nglobal constant, and pool/clim then says whether "
        "per-item climatology is simply\nover-fitting item noise. Only "
        "mag/pool clearly BELOW 1 is evidence that the 33\nfeatures carry "
        "usable volatility signal."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument(
        "--objective",
        choices=("quantile", "l1"),
        default="quantile",
        help="quantile: regress the 80th pct of |r| directly (the "
        "half-width itself). l1: regress median |r| and let "
        "lambda rescale -- a robustness check, not the headline.",
    )
    ap.add_argument(
        "--served-cohort",
        action="store_true",
        help="restrict to the served backfilled cohort "
        "(_resolve_backfilled_slugs) rather than the broad >=$1 "
        "artifact population. The broad cohort carries a "
        "near-zero residual tail that flatters any width "
        "estimator, so this is the deployable read.",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=1,
        help="walk-forward eval windows. >1 makes the width win "
        "REPEAT across eras instead of appearing once in the "
        "final 20%% of dates -- the shape in which every "
        "previous band arm was CV-positive and "
        "serving-negative.",
    )
    ap.add_argument(
        "--k-sweep",
        action="store_true",
        help="sweep CLIMATOLOGY_SHRINK_K on the climatology arm "
        "alone (no booster, no pool arm) and report the "
        "matched-coverage width against the production K.",
    )
    ap.add_argument(
        "--k-grid",
        type=float,
        nargs="+",
        default=None,
        help="override the K sweep grid. The production K is "
        "always inserted, since every ratio is measured "
        "against it.",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    feats = _feature_cols()
    df = _load(feats, served_only=args.served_cohort)
    cohort = "served" if args.served_cohort else "all"
    print(
        f"[artifact/{cohort}] {len(df):,} rows, {df['item_id'].nunique():,} items, "
        f"{df['date'].min().date()}..{df['date'].max().date()} | "
        f"{len(feats)} served features | objective={args.objective}\n"
    )
    if args.k_grid:
        global K_GRID
        K_GRID = tuple(sorted({PROD_K, *(int(k) for k in args.k_grid)}))
    if args.k_sweep:
        rows = [_run_k_sweep(df, h, args.folds) for h in args.horizons]
        _print_k_sweep(rows)
    else:
        rows = [r for h in args.horizons for r in _run_horizon(df, h, feats, args.objective, args.folds)]
        _print_report(rows)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "target_coverage": TARGET_COVERAGE,
                    "objective": args.objective,
                    "cohort": cohort,
                    "folds": args.folds,
                    "horizons": rows,
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
