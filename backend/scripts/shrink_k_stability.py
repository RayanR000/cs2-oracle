#!/usr/bin/env python3
"""Stability of SHRINK_K_GBM's per-item K predictions across training windows.

`scripts/shrink_k_vol_rank_ab.py` measures whether the per-item-K GBM beats
flat K=320 out of sample, but a win there could still be a lucky fold: if item
X gets K=50 in one fold and K=500 in the next, the GBM is fitting noise and
the served table will wobble on every retrain. This harness measures that
directly.

Method, per horizon in [3, 7, 14, 30]:
  - `prepare_targets(df, horizon)` for the label column (production's own).
  - Walk-forward folds identical to the harness family: train = everything up
    to the fold boundary over `train_items`, embargoed with production's own
    `_purge_overlapping_train_rows` (horizon+13, never a bare horizon).
  - Per fold: `_compute_per_item_optimal_k` on the fit split (the oracle),
    `_fit_shrink_k_model` on those optima (production's tree_params={} default),
    then `predict_shrink_k` on a FIXED probe set.
  - The probe is the held-out `eval_items` cohort with features computed ONCE
    from the full history, so fold-to-fold variation is model instability, not
    feature drift (count/raw_std drift slowly and would otherwise confound the
    read). Held-out items never appear in any fit split, so a per-fold feature
    frame cannot supply them.

Metrics:
  - Per-item CV = std/mean of predicted K across folds. Median CV is the
    headline: low = stable, high = noisy.
  - Mean pairwise Spearman rank correlation of K predictions between folds
    (plus the consecutive-fold mean): do folds agree on WHICH items need large
    vs small K, even if the level shifts?
  - Pooled distribution of predicted K (p10/p25/p50/p75/p90) plus the oracle
    distribution for context.

Usage:
    venv/bin/python scripts/shrink_k_stability.py --horizon 3 \\
        --frame-cache /tmp/sks_frame.parquet --out /tmp/shrink_k_stability.json
    venv/bin/python scripts/shrink_k_stability.py \\
        --frame-cache /tmp/sks_frame.parquet
"""

import os

os.environ["CLIMATOLOGY_SCALE"] = "1"
os.environ["SHRINK_K_GBM"] = "1"

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from database import SessionLocal
from models.forecaster import ItemForecaster
from scripts.ab_test_item_metadata import (
    ROW_BUDGET,
    STEP_DAYS,
    VAL_WINDOW_DAYS,
    _stratified_sample,
    assign_items,
    build_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("shrink_k_stability")

HORIZONS = [3, 7, 14, 30]

#: Minimum observations for an oracle K to be trusted. Mirrors the <15 rule
#: inside `_compute_per_item_optimal_k` (thin items get the global default).
MIN_OLSSON = 15


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation via pandas (no scipy dependency at call time)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    if np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman"))


def _percentiles(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: None for k in ("p10", "p25", "p50", "p75", "p90")}
    return {
        k: float(v)
        for k, v in zip(
            ("p10", "p25", "p50", "p75", "p90"),
            np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90]),
        )
    }


def run(df, horizon_filter=None):
    eval_items, train_items, _trained_eval = assign_items(df)
    logger.info(f"  ROW_BUDGET={ROW_BUDGET} (probe is item-level; no row sampling)")

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        horizons = [h for h in HORIZONS if horizon_filter is None or h == horizon_filter]
        if horizon_filter is not None and not horizons:
            logger.warning(f"  horizon {horizon_filter} not in {HORIZONS} — nothing to do")
            return {}
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Shrink-K stability {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            tcol = f"target_return_{horizon}d"
            if tcol not in tdf.columns:
                logger.warning(f"    {tcol} absent — skipping")
                continue
            tdf = tdf.dropna(subset=[tcol]).sort_values(["item_id", "date"])
            if tdf.empty:
                logger.warning(f"    No valid returns for {horizon}d")
                continue
            sub = tdf[["item_id", "date", "price", tcol]].copy()

            # Fixed probe: eval-item features computed ONCE from the full
            # history. Fold-to-fold prediction spread is then the GBM moving,
            # not the inputs drifting. (Held-out items never occur in any fit
            # split, so a per-fold frame cannot supply them anyway.)
            ref_df = sub[sub["item_id"].isin(set(eval_items))]
            if ref_df.empty:
                logger.warning("    no eval-item rows for the probe — skipping")
                continue
            ref_stats = ItemForecaster._compute_per_item_optimal_k(ref_df, tcol)
            if ref_stats.empty:
                logger.warning("    empty probe stats — skipping")
                continue
            ref_stats = ref_stats.sort_values("item_id").reset_index(drop=True)
            common_items = ref_stats["item_id"].astype(str).tolist()
            logger.info(
                f"    probe: {len(common_items)} held-out items "
                f"({int((ref_stats['count'] >= MIN_OLSSON).sum())} with "
                f"count >= {MIN_OLSSON})"
            )

            dates = sorted(sub["date"].unique())
            split_idx = len(dates) * 2 // 3
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            is_train_item = sub["item_id"].isin(set(train_items)).to_numpy()

            fold_preds = []
            fold_meta = []
            oracle_pool = []
            for fold_idx, window_end in enumerate(range(split_idx + 1, len(dates), STEP_DAYS)):
                val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue
                in_fit = sub_days <= dates_dt[window_end - 1]
                # Production's own purge: horizon + 13 embargo, never bare.
                fit_df = ItemForecaster._purge_overlapping_train_rows(
                    sub[in_fit & is_train_item], val_dates[0], horizon
                )
                if fit_df.empty:
                    continue
                fit_min = fit_df[["item_id", "price", tcol]]
                try:
                    item_stats = forecaster._compute_per_item_optimal_k(fit_min, tcol)
                    if item_stats.empty:
                        continue
                    oracle_pool.append(item_stats["optimal_k"].to_numpy(dtype=float))
                    model = forecaster._fit_shrink_k_model(
                        item_stats,
                        item_stats["optimal_k"].to_numpy(),
                        tree_params={},
                        boosting_type="gbdt",
                    )
                    forecaster.shrink_k_models[horizon] = model
                    try:
                        preds = forecaster.predict_shrink_k(horizon, ref_stats)
                    finally:
                        forecaster.shrink_k_models.pop(horizon, None)
                except Exception as exc:
                    logger.warning(f"    fold {fold_idx}: shrink-K fit failed ({exc!r}) — skipped")
                    continue
                preds = np.asarray(preds, dtype=float)
                if preds.shape[0] != len(common_items) or not np.isfinite(preds).any():
                    logger.warning(f"    fold {fold_idx}: degenerate predictions — skipped")
                    continue
                fold_preds.append(preds)
                fold_meta.append(
                    {
                        "fold": fold_idx,
                        "val_start": str(val_dates[0]),
                        "val_end": str(val_dates[-1]),
                        "n_fit_rows": len(fit_df),
                        "n_fit_items": int(item_stats["item_id"].nunique()),
                    }
                )
                logger.info(
                    f"    fold {fold_idx} ({val_dates[0]}): fit {len(fit_df):,} rows / "
                    f"{item_stats['item_id'].nunique()} items, "
                    f"probe K p50={float(np.nanmedian(preds)):.1f}"
                )

            if len(fold_preds) < 2:
                logger.warning(f"    <2 usable folds at {horizon}d — skipping")
                continue
            mat = np.vstack(fold_preds)  # (n_folds, n_items)
            means = np.nanmean(mat, axis=0)
            stds = np.nanstd(mat, axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cvs = np.where(means > 0, stds / means, np.nan)

            # Pairwise Spearman over folds; consecutive mean reported alongside.
            pair_rhos = []
            consec_rhos = []
            n_folds = mat.shape[0]
            for i in range(n_folds):
                for j in range(i + 1, n_folds):
                    r = _spearman(mat[i], mat[j])
                    if np.isfinite(r):
                        pair_rhos.append(r)
                    if j == i + 1 and np.isfinite(r):
                        consec_rhos.append(r)

            pooled_pred = mat[np.isfinite(mat)]
            pooled_oracle = np.concatenate(oracle_pool) if oracle_pool else np.array([], dtype=float)

            per_item = [
                {
                    "item_id": iid,
                    "mean_k": float(means[k]),
                    "std_k": float(stds[k]),
                    "cv": (None if not np.isfinite(cvs[k]) else float(cvs[k])),
                }
                for k, iid in enumerate(common_items)
            ]
            finite_cv = cvs[np.isfinite(cvs)]
            entry = {
                "n_folds": n_folds,
                "n_common_items": len(common_items),
                "common_items": common_items,
                "folds": fold_meta,
                "per_item": per_item,
                "median_cv": float(np.median(finite_cv)) if finite_cv.size else None,
                "mean_cv": float(np.mean(finite_cv)) if finite_cv.size else None,
                "p90_cv": float(np.quantile(finite_cv, 0.90)) if finite_cv.size else None,
                "mean_pairwise_spearman": float(np.mean(pair_rhos)) if pair_rhos else None,
                "mean_consecutive_spearman": float(np.mean(consec_rhos)) if consec_rhos else None,
                "n_pairs": len(pair_rhos),
                "pred_k_percentiles": _percentiles(pooled_pred),
                "oracle_k_percentiles": _percentiles(pooled_oracle),
                "pred_k_by_fold": [list(map(float, row)) for row in fold_preds],
            }
            results[horizon] = entry
            logger.info(
                f"      {horizon}d: folds={n_folds} items={len(common_items)} "
                f"median CV={entry['median_cv']:.3f} "
                f"mean pairwise rho={entry['mean_pairwise_spearman']:.3f}"
            )
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 88)
    print("SHRINK-K GBM STABILITY — per-item K predictions across walk-forward folds")
    print("low median CV = stable; high CV / low rank-correlation = fitting noise")
    print("=" * 88)
    print(
        f"  {'h':>4} {'folds':>6} {'items':>6} {'medianCV':>9} {'meanCV':>8} "
        f"{'p90CV':>7} {'pairRho':>8} {'consecRho':>10} "
        f"{'p10':>7} {'p50':>7} {'p90':>7}"
    )
    print(f"  {'-' * 84}")
    for horizon in sorted(results):
        e = results[horizon]
        pk = e["pred_k_percentiles"]

        def _f(v, fmt):
            return "  n/a" if v is None else format(v, fmt)

        print(
            f"  {horizon:>4}d {e['n_folds']:>6} {e['n_common_items']:>6} "
            f"{_f(e['median_cv'], '>9.3f')} {_f(e['mean_cv'], '>8.3f')} "
            f"{_f(e['p90_cv'], '>7.3f')} {_f(e['mean_pairwise_spearman'], '>8.3f')} "
            f"{_f(e['mean_consecutive_spearman'], '>10.3f')} "
            f"{_f(pk.get('p10'), '>7.1f')} {_f(pk.get('p50'), '>7.1f')} "
            f"{_f(pk.get('p90'), '>7.1f')}"
        )
    print("")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--metadata-parquet", default=None)
    parser.add_argument("--out", default="/tmp/shrink_k_stability.json")
    args = parser.parse_args()

    df, pruned, _meta_present = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    logger.info(f"  Frame: {len(df):,} rows, {len(pruned)} pruned features")
    _ = _stratified_sample  # imported for parity with the harness family
    results = run(df, horizon_filter=args.horizon)
    print_summary(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
