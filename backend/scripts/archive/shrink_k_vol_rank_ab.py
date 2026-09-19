#!/usr/bin/env python3
"""Paired replay: per-item-K GBM and vol-rank GBM against the flat K=320 band.

Two band-geometry arms sit in opposite states, and both are wrong:

  SHRINK_K_GBM=1 shipped in 4949714 with NO measurement at all — the 02:59 UTC
  2026-09-09 retrain baked its per-item K table into the live artifact — while
  VOLATILITY_RANK_GBM sits off behind a commit-message-only local A/B
  (3bfd9a9: 8.1%/6.1%/3.6% narrower at matched 80% for h=3/7/14d, neutral at
  h=30). An unvalidated GBM is served while a claimed one is not, and neither
  claim meets this repo's bar: no harness, no paired interval, no changelog
  (backend/AGENTS.md: "Don't cite a stored verdict without checking").

This harness settles both against the same control on the same rows:

  control:     flat CLIMATOLOGY_SHRINK_K=320 climatology (the f02320b winner:
               narrower in 23/23 served-gated paired cells). Built with
               production's own `ItemForecaster._build_climatology_table`, so
               the control IS the served geometry, not a re-derivation.
  shrink_k_gbm: production's `_compute_per_item_optimal_k` on the fit split,
               `_fit_shrink_k_model` on those optima, `_predict_shrink_k` on
               the eval split's item stats, `_build_climatology_table_adaptive`
               — the exact production path, refit per fold. If even this
               fit-split ORACLE (train-optimal K, honestly transferred) cannot
               beat flat-320 out of sample, the served table is memorisation.
  vol_rank:    production's `_fit_vol_rank_model` (|return| regression on the
               allowlisted price_technicals) per fold, multiplier normalised to
               mean 1.0 on the fit split exactly as production does
               (forecaster.py `_vol_rank_multiplier`: clip raw at 0.01, divide
               by the fit mean, clip to [0.25, 4.0]), applied to the control
               scale on eval. Same 28 columns production serves.

Metric is mean half-width at MATCHED 80% coverage per arm per fold
(magnitude_vs_climatology `_matched_width`): each arm gets its own q80 on the
eval rows, so a reshape that needs a per-item q cannot win — only a shape a
single served q_hat can use. Reported as the paired per-fold delta on
log-width (arm - control; negative is narrower, interval-valid and symmetric)
plus the mean ratio and the win count. A mean win with a losing fold is the
shape every refuted band arm had — worst-fold is printed for that reason.

Fold machinery, item split, row budget, purge (h+13 embargo) and the paired
t-interval are imported wholesale from the harness family
(ab_test_item_metadata, exceedance_meta_ab) rather than re-derived — the point
is a paired read, which only holds if all arms see identical rows.

Usage:
    python -m scripts.archive.shrink_k_vol_rank_ab --horizon 3 \\
        --frame-cache /tmp/skvr_frame.parquet --out /tmp/skvr_h3.json
    python -m scripts.archive.shrink_k_vol_rank_ab --max-folds 2   # recent folds only
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from api.serving_policy import MIN_SERVED_PRICE_USD
from database import SessionLocal
from models.forecaster import ItemForecaster
from scripts.archive.ab_test_item_metadata import (
    ROW_BUDGET,
    STEP_DAYS,
    VAL_WINDOW_DAYS,
    _stratified_sample,
    assign_items,
    build_frame,
)
from scripts.exceedance_meta_ab import (
    TREE_PARAMS,
    paired_fold_deltas,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("shrink_k_vol_rank_ab")

TARGET_COVERAGE = 0.80

#: Fewest eval rows a fold may score. Width-at-matched-coverage is a quantile
#: read; below this the q80 is decided by tail draws rather than the arm.
MIN_EVAL_ROWS = 200


def matched_width(abs_r: np.ndarray, scale: np.ndarray):
    """(q80, mean half-width) with coverage matched exactly on these rows.

    Same definition as magnitude_vs_climatology._matched_width: each arm is
    scored at its own q80, so the comparison is width at equal coverage, never
    coverage at equal width. Returns (nan, nan) when nothing is scoreable so a
    degenerate fold drops out of the paired mean instead of voting zero.
    """
    abs_r = np.asarray(abs_r, dtype=float)
    scale = np.asarray(scale, dtype=float)
    ok = np.isfinite(abs_r) & np.isfinite(scale) & (scale > 0)
    if not ok.any():
        return float("nan"), float("nan")
    q = float(np.quantile(abs_r[ok] / scale[ok], TARGET_COVERAGE))
    return q, q * float(np.mean(scale[ok]))


def vol_rank_multiplier(raw_pred: np.ndarray, fit_mean: float) -> np.ndarray:
    """Mean-1.0 cross-sectional multiplier from a vol-rank prediction.

    Mirrors `ItemForecaster._vol_rank_multiplier` (clip raw at 0.01, divide by
    the FIT-split mean so q_hat's level is preserved, clip to [0.25, 4.0]).
    The mean MUST come from the fit split — normalising on eval would leak the
    eval volatility level into the scale. Returns None-equivalent (all-NaN)
    when the fit mean is unusable, so the fold drops out rather than serving
    an unnormalised scale.
    """
    raw = np.clip(np.asarray(raw_pred, dtype=float), 0.01, None)
    if not np.isfinite(fit_mean) or fit_mean <= 0:
        return np.full_like(raw, np.nan)
    return np.clip(raw / fit_mean, 0.25, 4.0)


def _lookup(forecaster, horizon, table_cfg, ids, prices):
    """Eval-split climatology scale through the production lookup."""
    forecaster.climatology_scale[horizon] = table_cfg
    return forecaster._climatology_lookup(horizon, ids, prices)


def run(df, pruned, horizon_filter=None, n_jobs=None, max_folds=None):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    eval_items, train_items, trained_eval = assign_items(df)

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        horizons = [h for h in ItemForecaster.HORIZONS if horizon_filter is None or h == horizon_filter]
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Band geometry {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            tcol = f"target_return_{horizon}d"
            if tcol not in tdf.columns:
                logger.warning(f"    {tcol} absent — skipping")
                continue
            tdf = tdf.dropna(subset=[tcol]).sort_values(["item_id", "date"])
            if tdf.empty:
                logger.warning(f"    No valid returns for {horizon}d")
                continue
            base_cols = [c for c in pruned if c in tdf.columns]
            if not base_cols:
                logger.warning("    no allowlisted feature columns — skipping")
                continue
            sub = tdf[["item_id", "date", "price", tcol, *base_cols]].copy()

            dates = sorted(sub["date"].unique())
            split_idx = len(dates) * 2 // 3
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            is_heldout = sub["item_id"].isin(set(eval_items)).to_numpy()
            is_trained_eval = sub["item_id"].isin(set(trained_eval)).to_numpy()
            is_train_item = sub["item_id"].isin(set(train_items)).to_numpy()

            fold_list = list(range(split_idx + 1, len(dates), STEP_DAYS))
            if max_folds is not None:
                fold_list = fold_list[-max_folds:]

            per_fold = {"control": [], "shrink_k_gbm": [], "vol_rank": []}
            for fold_idx, window_end in enumerate(fold_list):
                val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue
                in_fit = sub_days <= dates_dt[window_end - 1]
                in_val = (sub_days >= dates_dt[window_end]) & (sub_days <= dates_dt[window_end + len(val_dates) - 1])
                fit_df = ItemForecaster._purge_overlapping_train_rows(
                    sub[in_fit & is_train_item], val_dates[0], horizon
                )
                val_df = sub[in_val & (is_heldout | is_trained_eval)]
                val_df = val_df[val_df["price"] >= MIN_SERVED_PRICE_USD]
                if len(val_df) < MIN_EVAL_ROWS or fit_df.empty:
                    continue
                fit_sample = _stratified_sample(fit_df, train_items, ROW_BUDGET, fold_idx)

                fit_min = fit_df[["item_id", "price", tcol]]
                val_ids = val_df["item_id"].to_numpy()
                val_px = val_df["price"].to_numpy(dtype=float)
                abs_r = val_df[tcol].abs().to_numpy(dtype=float)

                # CONTROL: flat K=320, production's own builder on the fit.
                ctl_cfg = {
                    "table": {},
                    "tier_pool": {},
                    "global": float("nan"),
                }
                ctl_table, ctl_pool, ctl_g = ItemForecaster._build_climatology_table(fit_min, tcol)
                if not ctl_table:
                    continue
                ctl_cfg = {"table": ctl_table, "tier_pool": ctl_pool, "global": ctl_g}
                ctl_scale = _lookup(forecaster, horizon, dict(ctl_cfg), val_ids, val_px)

                # SHRINK_K_GBM: per-item oracle K on the fit, GBM transfer.
                sk_scale = None
                try:
                    item_stats = forecaster._compute_per_item_optimal_k(fit_min, tcol)
                    if not item_stats.empty:
                        sk_model = forecaster._fit_shrink_k_model(
                            item_stats, item_stats["optimal_k"].to_numpy(), tree_params={}, boosting_type="gbdt"
                        )
                        forecaster.shrink_k_models[horizon] = sk_model
                        try:
                            sk_table, sk_pool, sk_g = forecaster._build_climatology_table_adaptive(
                                fit_min, tcol, horizon
                            )
                            if sk_table:
                                sk_scale = _lookup(
                                    forecaster,
                                    horizon,
                                    {"table": sk_table, "tier_pool": sk_pool, "global": sk_g},
                                    val_ids,
                                    val_px,
                                )
                        finally:
                            forecaster.shrink_k_models.pop(horizon, None)
                except Exception as exc:
                    logger.warning(f"    fold {fold_idx}: shrink_k arm failed ({exc!r}) — fold scores control/vol only")

                # VOL_RANK: |return| regression on the fit, mean-1 mult on val.
                vr_scale = None
                try:
                    med = fit_sample[base_cols].median()
                    X_fit = fit_sample[base_cols].fillna(med).to_numpy(dtype=float)
                    y_fit = fit_sample[tcol].abs().to_numpy(dtype=float)
                    ok_fit = np.isfinite(y_fit) & np.isfinite(X_fit).all(axis=1)
                    if ok_fit.sum() >= 200:
                        # Small honest val slice off the fit tail for early
                        # stopping ONLY — never the eval rows.
                        cut = int(len(fit_sample) * 0.9)
                        tr_sl = np.zeros(len(fit_sample), dtype=bool)
                        tr_sl[:cut] = True
                        tr_ok = ok_fit & tr_sl
                        va_ok = ok_fit & ~tr_sl
                        if tr_ok.sum() >= 200 and va_ok.sum() >= 50:
                            vr_models = forecaster._fit_vol_rank_model(
                                X_fit[tr_ok],
                                y_fit[tr_ok],
                                X_fit[va_ok],
                                y_fit[va_ok],
                                "gbdt",
                                dict(TREE_PARAMS, n_jobs=n_jobs),
                                horizon=horizon,
                                num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                            )
                            forecaster.vol_rank_models[horizon] = vr_models
                            try:
                                raw_fit = forecaster._predict_vol_rank(horizon, X_fit[ok_fit])
                                X_val = val_df[base_cols].fillna(med).to_numpy(dtype=float)
                                raw_val = forecaster._predict_vol_rank(horizon, X_val)
                                if raw_fit is not None and raw_val is not None:
                                    fit_mean = float(np.mean(np.clip(raw_fit, 0.01, None)))
                                    mult = vol_rank_multiplier(raw_val, fit_mean)
                                    if np.isfinite(mult).all():
                                        vr_scale = ctl_scale * mult
                            finally:
                                forecaster.vol_rank_models.pop(horizon, None)
                except Exception as exc:
                    logger.warning(
                        f"    fold {fold_idx}: vol_rank arm failed ({exc!r}) — fold scores control/shrink only"
                    )

                row = {"fold": fold_idx, "val_start": str(val_dates[0]), "n_fit": len(fit_df), "n_eval": len(val_df)}
                for arm, scale in (("control", ctl_scale), ("shrink_k_gbm", sk_scale), ("vol_rank", vr_scale)):
                    if scale is None:
                        per_fold[arm].append(dict(row, width=None, log_width=None))
                        continue
                    _, w = matched_width(abs_r, scale)
                    per_fold[arm].append(
                        dict(
                            row,
                            width=(None if not np.isfinite(w) else float(w)),
                            log_width=(None if not np.isfinite(w) or w <= 0 else float(np.log(w))),
                        )
                    )
                forecaster.climatology_scale.pop(horizon, None)

            if not [r for r in per_fold["control"] if r["width"] is not None]:
                logger.warning(f"    no usable folds at {horizon}d")
                continue
            results[horizon] = {arm: {"per_fold": rows} for arm, rows in per_fold.items()}
            results[horizon]["_paired"] = {
                arm: paired_fold_deltas(per_fold["control"], per_fold[arm], "log_width")
                for arm in ("shrink_k_gbm", "vol_rank")
            }
            for arm in ("control", "shrink_k_gbm", "vol_rank"):
                ws = [r["width"] for r in per_fold[arm] if r["width"] is not None]
                if ws:
                    logger.info(f"      {arm:12s} mean matched width={np.mean(ws):.3f}% ({len(ws)} folds)")
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 78)
    print("BAND GEOMETRY vs FLAT K=320 — paired per-fold deltas on log-width")
    print("negative delta = NARROWER at matched 80% coverage (good)")
    print("=" * 78)
    for horizon, entry in sorted(results.items()):
        n_ctl = sum(1 for r in entry["control"]["per_fold"] if r["width"] is not None)
        print(f"\nh={horizon}d   control folds={n_ctl}")
        for arm in ("shrink_k_gbm", "vol_rank"):
            d = entry["_paired"][arm]
            rows = entry[arm]["per_fold"]
            base = {r["fold"]: r["width"] for r in entry["control"]["per_fold"]}
            ratios = [
                r["width"] / base[r["fold"]]
                for r in rows
                if r["width"] is not None and base.get(r["fold"]) is not None and base[r["fold"]] > 0
            ]
            rinfo = (
                f"mean ratio {np.mean(ratios):.4f}, "
                f"worst {np.max(ratios):.4f}, "
                f"narrower {sum(x < 1.0 for x in ratios)}/{len(ratios)}"
                if ratios
                else "no paired folds"
            )
            if d is None:
                print(f"  {arm:12s} — too few paired folds; {rinfo}")
                continue
            flag = "*" if d["excludes_zero"] else " "
            better = d["n_folds"] - d["wins"]  # wins count positive deltas
            print(
                f"  {arm:12s} d={d['mean']:+.5f} "
                f"[{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]{flag} "
                f"narrower {better}/{d['n_folds']}; {rinfo}"
            )
    print(
        "\n* = 95% interval excludes zero. An arm earns the served table only "
        "with\na negative interval AND no losing fold (worst ratio < 1)."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--metadata-parquet", default=None)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="score only the K most recent folds (quick read; the verdict needs the full run)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    df, pruned, _ = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    results = run(df, pruned, horizon_filter=args.horizon, n_jobs=args.n_jobs, max_folds=args.max_folds)
    print_summary(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
