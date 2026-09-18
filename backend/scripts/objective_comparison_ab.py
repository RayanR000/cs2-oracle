#!/usr/bin/env python3
"""Compare LightGBM training objectives for centre prediction accuracy.

Hypothesis: the q50 quantile objective (≡ MAE) is misaligned with the
evaluation metric (within-date rank IC). LambdaRank directly optimizes
ranking; Huber smooths the gradient near zero; binary classification
predicts direction; MSE (L2) has a different gradient profile that may
capture weak signal the L1 surface misses.

Arms:
  quantile   — production baseline (α=0.5, ≡ MAE)
  huber      — smooth L1/L2 hybrid
  mse        — L2 regression
  lambdarank — pairwise ranking, grouped by date
  binary     — P(return > 0), logistic loss

Each arm trains on the same temporal split with the same tree params
(minus objective-specific ones), evaluated on within-date rank IC and DA.
Uses production's data pipeline, feature set, and evaluation methods.

Usage:
    cd backend
    venv/bin/python -m scripts.objective_comparison_ab \
        --horizon 14 --out /tmp/obj_cmp_h14.json

    # All horizons:
    venv/bin/python -m scripts.objective_comparison_ab \
        --out /tmp/obj_cmp.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from database import SessionLocal
from models.forecaster import ItemForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("objective_comparison_ab")

DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}

ARMS = ("quantile", "huber", "mse", "lambdarank", "binary")

MIN_ROWS_PER_DATE = 20
N_BOOTSTRAP = 1000
BOOT_SEED = 42


def _arm_params(arm: str, horizon: int, n_jobs: int) -> dict:
    """Build LightGBM params for each objective arm.

    Tree params match production (from tuned_params cache or defaults).
    Only the objective and its related keys differ.
    """
    base = {
        "boosting_type": "gbdt",
        "num_leaves": 47,
        "max_depth": 5,
        "min_data_in_leaf": 15,
        "min_gain_to_split": 0.1,
        "learning_rate": 0.01,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.0,
        "lambda_l2": 1.5,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": n_jobs,
        "force_row_wise": True,
        **DS_PARAMS,
    }

    if arm == "quantile":
        base["objective"] = "quantile"
        base["alpha"] = 0.5
        base["metric"] = "quantile"
    elif arm == "huber":
        base["objective"] = "huber"
        base["alpha"] = 1.0
        base["metric"] = "huber"
    elif arm == "mse":
        base["objective"] = "regression"
        base["metric"] = "l2"
    elif arm == "lambdarank":
        base["objective"] = "lambdarank"
        base["metric"] = "ndcg"
        base["eval_at"] = [10, 50]
        base["lambdarank_truncation_level"] = 100
    elif arm == "binary":
        base["objective"] = "binary"
        base["metric"] = "binary_logloss"
    else:
        raise ValueError(f"Unknown arm: {arm}")

    return base


def _within_date_rank_ic(pred, actual, dates):
    """Within-date Spearman IC — the production evaluation metric."""
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    d = np.asarray(dates)
    frame = pd.DataFrame({"d": d, "p": p, "a": a})
    ics = {}
    for day, g in frame.groupby("d"):
        if len(g) < MIN_ROWS_PER_DATE:
            continue
        if g["p"].nunique() < 2 or g["a"].nunique() < 2:
            continue
        ic = spearmanr(g["p"], g["a"]).statistic
        if np.isfinite(ic):
            ics[str(day)] = float(ic)
    return ics


def _directional_accuracy(pred, actual):
    """Fraction of rows where predicted direction matches actual."""
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a)
    p, a = p[mask], a[mask]
    if len(p) == 0:
        return None
    pred_dir = np.sign(p)
    actual_dir = np.sign(a)
    return float(np.mean(pred_dir == actual_dir))


def bootstrap_ci(values: dict, n=N_BOOTSTRAP, seed=BOOT_SEED):
    """95% CI of the mean under date resampling."""
    dates = sorted(values)
    if len(dates) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array([values[d] for d in dates], dtype=float)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)])
    return {
        "n_dates": len(dates),
        "mean": round(float(arr.mean()), 5),
        "ci_low": round(float(np.percentile(means, 2.5)), 5),
        "ci_high": round(float(np.percentile(means, 97.5)), 5),
    }


def paired_delta_ci(a: dict, b: dict, n=N_BOOTSTRAP, seed=BOOT_SEED):
    """Bootstrap CI of mean(a - b) on shared dates."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    return bootstrap_ci({d: a[d] - b[d] for d in shared}, n=n, seed=seed)


def run(horizon_filter=None, out_path=None):
    n_jobs = max(1, (os.cpu_count() or 4) // 2)
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        logger.info("Building training data (production pipeline)...")
        t0 = time.time()
        df = forecaster.build_training_data(
            days_back=1460,
            backfilled_only=True,
            max_feature_rows=1_200_000,
            min_median_price=1.0,
            universe="train",
        )
        logger.info(f"  Built frame in {time.time() - t0:.0f}s: {len(df):,} rows, {df['item_id'].nunique()} items")
    finally:
        db.close()

    feature_cols = list(forecaster.feature_cols)
    logger.info(f"  Features: {len(feature_cols)}")

    horizons = [h for h in ItemForecaster.HORIZONS if horizon_filter is None or h == horizon_filter]
    all_results = {}

    for horizon in horizons:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"HORIZON {horizon}d — objective comparison")
        logger.info(f"{'=' * 60}")

        tdf = forecaster.prepare_targets(df, horizon)
        tcol = f"target_return_{horizon}d"
        tdf = tdf.dropna(subset=[tcol]).copy()
        tdf = tdf.sort_values("date")

        train_set, val_set = forecaster._build_production_split(tdf, horizon, max_rows=1_200_000)
        logger.info(f"  {horizon}d: {len(train_set):,} train, {len(val_set):,} val")

        feats = [c for c in feature_cols if c in train_set.columns]
        X_train_raw = train_set[feats].replace([np.inf, -np.inf], np.nan)
        medians = X_train_raw.median()
        X_train = X_train_raw.fillna(medians)
        X_val = val_set[feats].replace([np.inf, -np.inf], np.nan).fillna(medians)

        y_train = train_set[tcol].to_numpy(dtype=float)
        y_val = val_set[tcol].to_numpy(dtype=float)
        val_dates = val_set["date"].to_numpy()

        boost_rounds = ItemForecaster._boost_rounds(horizon, cv=False)

        # Naive baseline (−return_1d as ranking predictor)
        naive_val_pred = -val_set["return_1d"].to_numpy(dtype=float) if "return_1d" in val_set.columns else np.zeros_like(y_val)
        naive_ics = _within_date_rank_ic(naive_val_pred, y_val, val_dates)
        naive_da = _directional_accuracy(naive_val_pred, y_val)

        arm_results = {}
        arm_ics = {}

        for arm in ARMS:
            logger.info(f"\n  --- {arm} ---")
            t_arm = time.time()
            params = _arm_params(arm, horizon, n_jobs)

            if arm == "lambdarank":
                # LambdaRank needs integer relevance labels. Discretize returns
                # into within-date decile ranks (0-9) so items with higher
                # returns get higher relevance. Per-date ranking removes the
                # market factor, matching the evaluation metric.
                def _to_relevance(returns, dates):
                    """Per-date decile rank (0-9) as relevance label."""
                    frame = pd.DataFrame({"r": returns, "d": dates})
                    frame["rel"] = 0
                    for _, g in frame.groupby("d"):
                        if len(g) < 5:
                            frame.loc[g.index, "rel"] = 5
                            continue
                        n_bins = min(10, len(g))
                        try:
                            frame.loc[g.index, "rel"] = pd.qcut(
                                g["r"], n_bins, labels=False, duplicates="drop"
                            )
                        except ValueError:
                            frame.loc[g.index, "rel"] = np.argsort(np.argsort(g["r"])) * 9 // len(g)
                    return frame["rel"].astype(int).to_numpy()

                train_order = np.argsort(train_set["date"].to_numpy(), kind="stable")
                X_tr = X_train.iloc[train_order]
                tr_dates = train_set["date"].to_numpy()[train_order]
                y_tr_rel = _to_relevance(y_train, train_set["date"].to_numpy())[train_order]
                _, group_counts = np.unique(tr_dates, return_counts=True)

                val_order = np.argsort(val_dates, kind="stable")
                X_vl = X_val.iloc[val_order]
                vl_dates = val_dates[val_order]
                y_vl_rel = _to_relevance(y_val, val_dates)[val_order]
                _, val_group_counts = np.unique(vl_dates, return_counts=True)

                n_levels = int(max(y_tr_rel.max(), y_vl_rel.max())) + 1
                params["label_gain"] = ",".join(str(i) for i in range(n_levels))

                dtrain = lgb.Dataset(X_tr, y_tr_rel, group=group_counts, params=DS_PARAMS)
                dval = lgb.Dataset(X_vl, y_vl_rel, reference=dtrain, group=val_group_counts, params=DS_PARAMS)
                dtrain.construct()
                dval.construct()
                booster = lgb.train(
                    params, dtrain, num_boost_round=boost_rounds,
                    valid_sets=[dval], callbacks=[lgb.log_evaluation(0)],
                )
                pred_val = booster.predict(X_val)
            elif arm == "binary":
                y_tr_bin = (y_train > 0).astype(float)
                dtrain = lgb.Dataset(X_train, y_tr_bin, params=DS_PARAMS)
                dval_bin = lgb.Dataset(X_val, (y_val > 0).astype(float), reference=dtrain, params=DS_PARAMS)
                dtrain.construct()
                dval_bin.construct()
                booster = lgb.train(
                    params, dtrain, num_boost_round=boost_rounds,
                    valid_sets=[dval_bin], callbacks=[lgb.log_evaluation(0)],
                )
                pred_val = booster.predict(X_val)
            else:
                dtrain = lgb.Dataset(X_train, y_train, params=DS_PARAMS)
                dval = lgb.Dataset(X_val, y_val, reference=dtrain, params=DS_PARAMS)
                dtrain.construct()
                dval.construct()
                booster = lgb.train(
                    params, dtrain, num_boost_round=boost_rounds,
                    valid_sets=[dval], callbacks=[lgb.log_evaluation(0)],
                )
                pred_val = booster.predict(X_val)

            elapsed = time.time() - t_arm

            per_date_ics = _within_date_rank_ic(pred_val, y_val, val_dates)
            da = _directional_accuracy(pred_val, y_val)
            ic_summary = bootstrap_ci(per_date_ics)
            paired_naive = paired_delta_ci(per_date_ics, naive_ics) if naive_ics else None

            fi = pd.DataFrame({
                "feature": feats,
                "gain": booster.feature_importance(importance_type="gain"),
            }).sort_values("gain", ascending=False).head(10)

            arm_results[arm] = {
                "rank_ic": ic_summary,
                "da": round(da, 4) if da is not None else None,
                "naive_da": round(naive_da, 4) if naive_da is not None else None,
                "paired_vs_naive": paired_naive,
                "train_seconds": round(elapsed, 1),
                "top_features": [(r[0], round(r[1], 1)) for r in fi.itertuples(index=False, name=None)],
            }
            arm_ics[arm] = per_date_ics

            ic_str = f"IC={ic_summary['mean']:.4f} [{ic_summary['ci_low']:.4f}, {ic_summary['ci_high']:.4f}]" if ic_summary else "IC=None"
            da_str = f"DA={da:.1%}" if da is not None else "DA=None"
            logger.info(f"    {ic_str}  {da_str}  ({elapsed:.1f}s)")

        # Summary table
        logger.info(f"\n  {'=' * 60}")
        logger.info(f"  SUMMARY {horizon}d")
        logger.info(f"  {'=' * 60}")
        logger.info(f"  {'Arm':<12} {'Rank IC':>10} {'CI':>24} {'DA':>8} {'Time':>6}")
        logger.info(f"  {'-' * 62}")

        for arm in ARMS:
            r = arm_results[arm]
            ic = r["rank_ic"]
            if ic:
                ic_str = f"{ic['mean']:>10.4f}"
                ci_str = f"[{ic['ci_low']:.4f}, {ic['ci_high']:.4f}]"
            else:
                ic_str = f"{'N/A':>10}"
                ci_str = "N/A"
            da_str = f"{r['da']:.1%}" if r["da"] is not None else "N/A"
            t_str = f"{r['train_seconds']:.0f}s"
            marker = " *" if arm == "quantile" else ""
            logger.info(f"  {arm:<12} {ic_str} {ci_str:>24} {da_str:>8} {t_str:>6}{marker}")

        # Paired deltas: every arm vs quantile baseline
        logger.info(f"\n  Paired deltas vs quantile baseline (IC_arm − IC_quantile):")
        q_ics = arm_ics.get("quantile", {})
        for arm in ARMS:
            if arm == "quantile":
                continue
            delta = paired_delta_ci(arm_ics[arm], q_ics)
            if delta:
                sig = "**" if delta["ci_low"] > 0 else ("*" if delta["ci_high"] < 0 else "")
                logger.info(
                    f"    {arm:<12} Δ={delta['mean']:+.4f} "
                    f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] "
                    f"({delta['n_dates']} dates) {sig}"
                )
                arm_results[arm]["paired_vs_quantile"] = delta
            else:
                logger.info(f"    {arm:<12} insufficient shared dates")

        naive_ic = bootstrap_ci(naive_ics)
        if naive_ic:
            logger.info(
                f"    {'naive':<12} IC={naive_ic['mean']:.4f} "
                f"[{naive_ic['ci_low']:.4f}, {naive_ic['ci_high']:.4f}] "
                f"DA={naive_da:.1%}"
            )

        all_results[horizon] = arm_results

    if out_path:
        Path(out_path).write_text(json.dumps(all_results, indent=2, default=str))
        logger.info(f"\nResults written to {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare LightGBM objectives for centre prediction")
    parser.add_argument("--horizon", type=int, default=None, help="Single horizon to test (default: all)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    run(horizon_filter=args.horizon, out_path=args.out)


if __name__ == "__main__":
    main()
