#!/usr/bin/env python3
"""Part B gate: does BREAK_AWARE_LOOKBACKS=1 narrow the band at matched coverage?

Implements the gate in `docs/specs/2026-09-30-composition-break-calendar-design.md`
(§Part B gate). Arms: the production training frame built with the flag off
(`control`) and on (`break_aware`). The mask runs inside `build_training_data`
before the cross-sectional rank transform, so the on-arm cannot be derived from
the off-frame -- the frame is built twice and the two must share every
(item_id, date) row and every label.

Fixed design:
  frame:   `build_training_data(days_back=1460, backfilled_only=True,
           max_feature_rows=1_200_000, min_median_price=1.0)`, as `train()` calls
           it, with the workflow's frame-affecting env (ENGINEERED_CACHE=0,
           SKIP_REGIMES=1, FEATURE_NATIVE_NAN=1).
  folds:   21-day validation windows every 21 days from FOLD_START, each trained
           on every earlier row purged by production's
           `_purge_overlapping_train_rows`, row-sampled to ROW_BUDGET with a
           per-fold seed shared by both arms.
  trainer: `_train_ensemble_member` at `_boost_rounds(h, cv=True)`, the
           horizon's production objective (`CENTRE_OBJECTIVE_MAP`), the ab_test_*
           family tree params, NaN passed through as `_impute_features` does
           under FEATURE_NATIVE_NAN.
  band:    centre +/- q * climatology scale, the scale built from the fold's fit
           labels (identical for both arms); each arm is scored at its own q80
           on the fold's eval rows (matched coverage).
  metrics: (primary) fold-paired log matched width, and row-paired interval
           score at alpha=0.2 resampled by fold (`paired_metric_difference`).

`--mde` runs the control arm twice at two seeds and reports the noise floor.
The spec stops the gate if it exceeds ~2% (width or interval score).

Usage (from backend/; reads prod Postgres for events, read-only):
    venv/bin/python -m scripts.archive.ab_test_break_aware_lookbacks \\
        --cache-dir /tmp/balb --build-cache-only
    venv/bin/python -m scripts.archive.ab_test_break_aware_lookbacks \\
        --cache-dir /tmp/balb --mde --horizon 14 --out /tmp/balb_mde_h14.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The workflow's frame-affecting flags, before the forecaster module reads them.
os.environ["ENGINEERED_CACHE"] = "0"
os.environ["SKIP_REGIMES"] = "1"
os.environ["FEATURE_NATIVE_NAN"] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
from api.serving_policy import MIN_SERVED_PRICE_USD
from backtest.paired_mde import paired_metric_difference
from database import SessionLocal
from models.forecaster import CENTRE_OBJECTIVE_MAP, ItemForecaster
from scripts.archive.anomaly_band_modulator_ab import _lookup_scale
from scripts.archive.exceedance_meta_ab import paired_fold_deltas

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ab_test_break_aware_lookbacks")

ARMS = ("control", "break_aware")
FOLD_START = "2026-02-01"
STEP_DAYS = 21
VAL_WINDOW_DAYS = 21
ROW_BUDGET = 400_000
MIN_EVAL_ROWS = 500
TARGET_COVERAGE = 0.80
ALPHA = 1.0 - TARGET_COVERAGE
SEEDS = (42, 7)
DS_PARAMS = {"max_bin": ItemForecaster.MAX_BIN, "feature_pre_filter": False}
TREE_PARAMS = {
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 5,
    "min_data_in_leaf": 15,
    "min_gain_to_split": 0.1,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbosity": -1,
    "force_row_wise": True,
}


def _objective_params(horizon):
    obj = CENTRE_OBJECTIVE_MAP.get(horizon, "quantile")
    if obj == "regression":
        return {"objective": "regression", "metric": "l2"}
    if obj == "huber":
        return {"objective": "huber", "alpha": 1.0, "metric": "huber"}
    return {"objective": "quantile", "alpha": 0.5, "metric": "quantile"}


def build_frames(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for arm in ARMS:
        if arm == "break_aware":
            os.environ["BREAK_AWARE_LOOKBACKS"] = "1"
        else:
            os.environ.pop("BREAK_AWARE_LOOKBACKS", None)
        db = SessionLocal()
        try:
            fc = ItemForecaster(db_session=db)
            t0 = time.time()
            df = fc.build_training_data(
                days_back=1460,
                backfilled_only=True,
                max_feature_rows=1_200_000,
                min_median_price=1.0,
                universe="train",
            )
        finally:
            db.close()
        logger.info(f"  {arm}: built {len(df):,} rows in {time.time() - t0:.0f}s")
        df.to_parquet(cache_dir / f"{arm}.parquet", index=False)
        meta[arm] = {
            "feature_cols": list(fc.feature_cols),
            "break_dates": sorted(d.isoformat() for d in fc.break_dates),
            "rows": len(df),
        }
    os.environ.pop("BREAK_AWARE_LOOKBACKS", None)
    if meta["control"]["feature_cols"] != meta["break_aware"]["feature_cols"]:
        raise SystemExit("Arms selected different feature columns; the pairing does not hold.")
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_frames(cache_dir: Path):
    meta = json.loads((cache_dir / "meta.json").read_text())
    frames = {arm: pd.read_parquet(cache_dir / f"{arm}.parquet") for arm in ARMS}
    keys = [f[["item_id", "date"]].reset_index(drop=True) for f in frames.values()]
    if not keys[0].equals(keys[1]):
        raise SystemExit("Arms carry different (item_id, date) rows; rebuild the cache.")
    return frames, meta["control"]["feature_cols"], meta


def _fold_bounds(dates):
    start = pd.Timestamp(FOLD_START)
    last = max(dates)
    out = []
    b = start
    while b + pd.Timedelta(days=7) <= last:
        out.append(b)
        b += pd.Timedelta(days=STEP_DAYS)
    return out


def _train_predict(horizon, X_fit, y_fit, X_val, seed, n_jobs):
    params = {**TREE_PARAMS, **_objective_params(horizon), **DS_PARAMS, "random_state": seed, "n_jobs": n_jobs}
    ItemForecaster._apply_row_sampling(params, 0.5)
    dtrain = lgb.Dataset(X_fit, y_fit, params=DS_PARAMS, free_raw_data=False)
    dtrain.construct()
    booster = ItemForecaster._train_ensemble_member(
        params,
        dtrain,
        None,
        num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
        early_stopping=ItemForecaster._early_stopping_enabled(),
    )
    return booster.predict(X_val)


def _score(abs_r, scale):
    """q80 at matched coverage, mean half-width, and per-row interval score."""
    q = float(np.quantile(abs_r / scale, TARGET_COVERAGE))
    half = q * scale
    interval = 2 * half + (2 / ALPHA) * np.maximum(0.0, abs_r - half)
    return q, float(np.mean(half)), interval


def run(frames, feature_cols, horizon, mode, n_jobs):
    """mode 'mde': control at two seeds. mode 'ab': control vs break_aware at seed 42."""
    if mode == "mde":
        legs = [("control", SEEDS[0]), ("control", SEEDS[1])]
    else:
        legs = [("control", SEEDS[0]), ("break_aware", SEEDS[0])]
    names = [f"{arm}@{seed}" for arm, seed in legs]

    db = SessionLocal()
    try:
        fc = ItemForecaster(db_session=db)
        tcol = f"target_return_{horizon}d"
        subs = {}
        for arm in {a for a, _ in legs}:
            t = fc.prepare_targets(frames[arm], horizon)
            t = t.dropna(subset=[tcol]).sort_values(["item_id", "date"]).reset_index(drop=True)
            subs[arm] = t
        ref = subs["control"]
        for arm, t in subs.items():
            if not ref[["item_id", "date", tcol]].equals(t[["item_id", "date", tcol]]):
                raise SystemExit(f"{arm} labels differ from control at {horizon}d; the pairing does not hold.")
        feats = [c for c in feature_cols if c in ref.columns]
        days = pd.to_datetime(ref["date"]).to_numpy()
        bounds = _fold_bounds(sorted(pd.to_datetime(ref["date"].unique())))
        per_fold = {n: [] for n in names}
        records = {n: [] for n in names}
        for fold_idx, b in enumerate(bounds):
            b_end = b + pd.Timedelta(days=VAL_WINDOW_DAYS)
            in_fit = days < b.to_datetime64()
            in_val = (days >= b.to_datetime64()) & (days < b_end.to_datetime64())
            fit_idx = ItemForecaster._purge_overlapping_train_rows(ref[in_fit], b, horizon).index
            if len(fit_idx) > ROW_BUDGET:
                rng = np.random.default_rng(1000 + fold_idx)
                fit_idx = pd.Index(np.sort(rng.choice(fit_idx.to_numpy(), ROW_BUDGET, replace=False)))
            val_mask = in_val & (ref["price"].to_numpy(dtype=float) >= MIN_SERVED_PRICE_USD)
            val_idx = ref.index[val_mask]
            if len(val_idx) < MIN_EVAL_ROWS or len(fit_idx) == 0:
                continue
            fit_rows = ref.loc[fit_idx]
            val_rows = ref.loc[val_idx]
            table, pool, g = ItemForecaster._build_climatology_table(fit_rows[["item_id", "price", tcol]], tcol)
            if not table:
                continue
            cfg = {"table": table, "tier_pool": pool, "global": g}
            scale = np.asarray(
                _lookup_scale(fc, horizon, cfg, val_rows["item_id"].to_numpy(), val_rows["price"].to_numpy(dtype=float)),
                dtype=float,
            )
            y_fit = fit_rows[tcol].to_numpy(dtype=float)
            y_val = val_rows[tcol].to_numpy(dtype=float)
            ok = np.isfinite(scale) & (scale > 0) & np.isfinite(y_val)
            for (arm, seed), name in zip(legs, names):
                src = subs[arm]
                X_fit = src.loc[fit_idx, feats].replace([np.inf, -np.inf], np.nan)
                X_val = src.loc[val_idx, feats].replace([np.inf, -np.inf], np.nan)
                med = X_fit.median()
                X_fit = fc._impute_features(X_fit, med)
                X_val = fc._impute_features(X_val, med)
                t0 = time.time()
                pred = _train_predict(horizon, X_fit, y_fit, X_val, seed, n_jobs)
                abs_r = np.abs(y_val - pred)
                q, w, interval = _score(abs_r[ok], scale[ok])
                nan_share = float(X_val.isna().to_numpy().mean())
                per_fold[name].append(
                    {
                        "fold": fold_idx,
                        "val_start": str(b.date()),
                        "n_fit": len(fit_idx),
                        "n_eval": int(ok.sum()),
                        "q80": q,
                        "width": w,
                        "log_width": float(np.log(w)),
                        "interval_score": float(np.mean(interval)),
                        "val_nan_share": nan_share,
                    }
                )
                vr = val_rows[ok]
                records[name].extend(
                    {"item_id": i, "forecast_date": str(d)[:10], "fold_id": fold_idx, "interval_score": float(s)}
                    for i, d, s in zip(vr["item_id"].to_numpy(), vr["date"].to_numpy(), interval)
                )
                logger.info(
                    f"    {horizon}d fold {fold_idx} {b.date()} {name}: q80={q:.3f} width={w:.3f} "
                    f"IS={np.mean(interval):.3f} nan={nan_share:.3f} ({time.time() - t0:.0f}s)"
                )
    finally:
        db.close()

    a, b_ = names
    is_ctl = float(np.mean([r["interval_score"] for r in records[a]])) if records[a] else float("nan")
    width = paired_fold_deltas(per_fold[a], per_fold[b_], "log_width")
    interval = paired_metric_difference(records[a], records[b_], value_key="interval_score", cluster_key="fold_id")
    interval.pop("records", None)
    out = {
        "horizon": horizon,
        "mode": mode,
        "legs": names,
        "n_folds": len(per_fold[a]),
        "log_width_delta": width,
        "interval_score_delta": interval,
        "interval_score_control_mean": is_ctl,
        "per_fold": per_fold,
    }
    if width:
        out["width_pct"] = {k: round(100 * width[k], 2) for k in ("mean", "ci_low", "ci_high")}
        out["width_mde_pct"] = round(100 * (width["ci_high"] - width["ci_low"]) / 2, 2)
    if interval.get("mde") is not None and np.isfinite(is_ctl) and is_ctl > 0:
        out["interval_mde_pct"] = round(100 * interval["mde"] / is_ctl, 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--build-cache-only", action="store_true")
    p.add_argument("--mde", action="store_true", help="seed-only noise floor on the control arm")
    p.add_argument("--horizon", type=int, choices=ItemForecaster.HORIZONS)
    p.add_argument("--out", type=Path)
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args()

    if args.build_cache_only:
        build_frames(args.cache_dir)
        return
    frames, feature_cols, meta = load_frames(args.cache_dir)
    logger.info(f"  break calendar: {meta['control']['break_dates']}; {len(feature_cols)} features")
    horizons = [args.horizon] if args.horizon else list(ItemForecaster.HORIZONS)
    results = {}
    for h in horizons:
        results[h] = run(frames, feature_cols, h, "mde" if args.mde else "ab", args.n_jobs)
        r = results[h]
        logger.info(
            f"  {h}d {r['mode']}: folds={r['n_folds']} width%={r.get('width_pct')} "
            f"width_mde%={r.get('width_mde_pct')} interval_mde%={r.get('interval_mde_pct')}"
        )
        if args.out:
            args.out.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
