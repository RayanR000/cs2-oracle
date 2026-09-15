#!/usr/bin/env python3
"""Does isotonic calibration make the 30d anomaly head fit to DISCLOSE?

`ANOMALY_SERVED_HORIZONS = (3, 7, 14)` withholds `anomaly_p` at 30d. The reason
(docs/changelog/2026-09-09-anomaly-head-beats-its-null.md) is narrow and
specific: at 30d the head RANKS — held-out AUC +0.129 over a featureless null,
clear of zero — but its held-out LOG LOSS is a dead tie with the pooled
constant. It orders items correctly and states the wrong number, and a served
`anomaly_p` is read as a probability, not as a rank.

That is the exact shape of defect isotonic calibration fixes, and this repo
already ships the machinery for it on the exceedance head
(`_fit_exceedance_calibrator` / `_apply_exceedance_calibrator`, PAV in pure
numpy). This harness asks whether the same layer earns 30d its disclosure.

## Arms

    global_rate    one pooled constant from the train fold. AUC 0.5 by
                   construction; it is the LOG LOSS bar, and the bar that
                   matters — a served probability that cannot beat a constant
                   is worse than decorative, it is misleading.
    item_rate      the item's own train-fold anomaly frequency.
    gbm            production's head trained on the whole train fold, raw
                   output. The status quo, and what the 2026-09-09 run scored.
    gbm_inner      the same head trained on the INNER train slice only (the
                   calibration slice held out). The honest control for
                   gbm_cal: it has seen strictly less data, so comparing
                   gbm_cal against `gbm` would credit calibration for a
                   difference in training rows.
    gbm_cal        gbm_inner's raw output mapped through an isotonic fit on the
                   held-out calibration slice.

`gbm_cal` vs `gbm_inner` isolates the calibration layer. `gbm_cal` vs
`global_rate` on held-out log loss is the SHIPPING TEST: 30d may be disclosed
only if that delta is negative with a CI clear of zero.

## The calibration slice

Fitted out-of-sample or not at all. The train fold is cut by date: the last
`--calib-frac` of its dates become the calibration slice, the head trains on
what precedes them, and `_purge_overlapping_train_rows` applies the H+13
embargo at that inner boundary exactly as the outer split does — without it the
inner-train rows whose labels resolve inside the calibration slice would leak
the very outcomes the map is fitted on.

Usage:
    python -m scripts.anomaly_calibration_ab --horizon 30 \
        --frame-cache /tmp/anom_frame.parquet --out /tmp/anom_cal_h30.json
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
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
from scripts.anomaly_gbm_ab import clean_anomaly_label, item_rate_predictions
from scripts.exceedance_meta_ab import TREE_PARAMS, _score, paired_fold_deltas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("anomaly_calibration_ab")

ARMS = ("gbm", "gbm_inner", "gbm_cal", "item_rate", "global_rate")

#: Production clips served anomaly probabilities into this range
#: (`anomaly_probability`), so every arm is clipped the same way before
#: scoring — otherwise the arms are compared on different supports and a 0.0
#: from a featureless rate scores an infinite log loss.
P_CLIP = (1e-3, 1.0)


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), *P_CLIP)


def inner_calibration_split(train_df, horizon, calib_frac):
    """(inner_train, calib) by date, with the H+13 embargo at the boundary.

    Returns (None, None) when the fold is too short to give the calibration
    slice its own dates — the caller then scores no gbm_cal for that fold
    rather than fitting a map on rows the head already saw.
    """
    dates = sorted(train_df["date"].unique())
    if len(dates) < 10:
        return None, None
    cut = int(len(dates) * (1.0 - calib_frac))
    if cut < 1 or cut >= len(dates):
        return None, None
    calib_start = dates[cut]
    inner = train_df[train_df["date"] < calib_start]
    calib = train_df[train_df["date"] >= calib_start]
    if inner.empty or calib.empty:
        return None, None
    inner = ItemForecaster._purge_overlapping_train_rows(inner, calib_start, horizon)
    if inner.empty:
        return None, None
    return inner, calib


def run(df, pruned, horizon_filter=30, n_jobs=None, clean_label=False, calib_frac=0.25):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    eval_items, train_items, trained_eval = assign_items(df)

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    if not forecaster.anomaly_gbm_enabled():
        raise SystemExit("ANOMALY_GBM did not take effect — no labels to score.")
    try:
        horizons = [h for h in ItemForecaster.HORIZONS if horizon_filter is None or h == horizon_filter]
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Anomaly calibration {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            target_col = f"target_anomaly_{horizon}d"
            if target_col not in tdf.columns:
                logger.warning(f"    {target_col} absent — skipping")
                continue
            tdf = tdf.sort_values(["item_id", "date"])
            if clean_label:
                tdf[target_col] = clean_anomaly_label(tdf, horizon)
            tdf = tdf.dropna(subset=[target_col])
            if tdf.empty:
                logger.warning(f"    No valid anomaly labels for {horizon}d")
                continue
            logger.info(
                f"    label base rate: {tdf[target_col].mean():.4f} "
                f"over {len(tdf):,} rows "
                f"({'STRICTLY-PRIOR' if clean_label else 'production'} "
                f"threshold)"
            )

            base_cols = [c for c in pruned if c in tdf.columns]
            sub = tdf[["item_id", "date", "price", target_col] + base_cols].copy()

            dates = sorted(sub["date"].unique())
            split_idx = len(dates) * 2 // 3
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            is_heldout = sub["item_id"].isin(set(eval_items)).to_numpy()
            is_trained_eval = sub["item_id"].isin(set(trained_eval)).to_numpy()
            is_train_item = sub["item_id"].isin(set(train_items)).to_numpy()

            per_fold = {a: [] for a in ARMS}
            cal_meta = []
            for fold_idx, window_end in enumerate(range(split_idx + 1, len(dates), STEP_DAYS)):
                val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue
                in_train = sub_days <= dates_dt[window_end - 1]
                in_val = (sub_days >= dates_dt[window_end]) & (sub_days <= dates_dt[window_end + len(val_dates) - 1])
                train_df = ItemForecaster._purge_overlapping_train_rows(
                    sub[in_train & is_train_item], val_dates[0], horizon
                )
                val_df = sub[in_val & (is_heldout | is_trained_eval)]
                if len(val_df) < 50 or train_df.empty:
                    continue
                train_df = _stratified_sample(train_df, train_items, ROW_BUDGET, fold_idx)

                med = train_df[base_cols].median()
                X_train = train_df[base_cols].fillna(med)
                X_val = val_df[base_cols].fillna(med)

                def _fit(Xtr, ytr):
                    return forecaster._fit_anomaly_classifier(
                        Xtr,
                        ytr,
                        "gbdt",
                        dict(TREE_PARAMS, n_jobs=n_jobs),
                        horizon=horizon,
                        tier_train=None,
                        num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                    )

                head = _fit(X_train, train_df[target_col].to_numpy())
                if head is None:
                    continue

                inner, calib = inner_calibration_split(train_df, horizon, calib_frac)
                p_inner = p_cal = None
                if inner is not None:
                    inner_head = _fit(inner[base_cols].fillna(med), inner[target_col].to_numpy())
                    if inner_head is not None:
                        p_inner = inner_head.predict(X_val)
                        p_fit = _clip(inner_head.predict(calib[base_cols].fillna(med)))
                        y_fit = calib[target_col].to_numpy(dtype=float)
                        if len(p_fit) >= forecaster.MIN_EXCEEDANCE_CALIBRATION_ROWS:
                            xs, ys = forecaster._isotonic_fit(p_fit, y_fit)
                            p_cal = np.interp(_clip(p_inner), xs, ys, left=float(ys[0]), right=float(ys[-1]))
                            cal_meta.append(
                                {
                                    "fold": fold_idx,
                                    "n_calib": len(p_fit),
                                    "n_steps": len(xs),
                                    "calib_base_rate": round(float(y_fit.mean()), 4),
                                }
                            )

                p_item, pooled = item_rate_predictions(train_df, val_df, target_col)
                preds = {
                    "gbm": head.predict(X_val),
                    "gbm_inner": p_inner,
                    "gbm_cal": p_cal,
                    "item_rate": p_item,
                    "global_rate": np.full(len(val_df), pooled),
                }

                ids = val_df["item_id"].to_numpy()
                held = np.isin(ids, eval_items)
                y = val_df[target_col].to_numpy(dtype=float)
                price = val_df["price"].to_numpy(dtype=float)
                for arm, p in preds.items():
                    if p is None:
                        continue
                    p = _clip(p)
                    row = {"fold": fold_idx, "val_start": str(val_dates[0]), "n_train": len(train_df)}
                    for cohort, mask in (("heldout", held), ("trained", ~held)):
                        sel = mask & (price >= 1.0)
                        auc, ll, n = _score(y[sel], p[sel])
                        row[f"{cohort}_auc"] = auc
                        row[f"{cohort}_logloss"] = ll
                        row[f"{cohort}_n"] = n
                    per_fold[arm].append(row)

            if not per_fold["gbm_cal"]:
                logger.warning(f"    no calibrated folds at {horizon}d")
                continue
            results[horizon] = {arm: {"per_fold": rows} for arm, rows in per_fold.items()}
            results[horizon]["_calibrators"] = cal_meta
            for arm in ARMS:
                rows = per_fold[arm]
                for cohort in ("heldout", "trained"):
                    aucs = [r[f"{cohort}_auc"] for r in rows if r[f"{cohort}_auc"] is not None]
                    lls = [r[f"{cohort}_logloss"] for r in rows if r[f"{cohort}_logloss"] is not None]
                    if aucs:
                        logger.info(
                            f"      {arm:12s} {cohort:8s} AUC={np.mean(aucs):.4f} "
                            f"logloss={np.mean(lls):.5f} ({len(aucs)} folds)"
                        )

            # Negative logloss delta = the treatment states better numbers.
            results[horizon]["_paired"] = {
                f"{treat}_vs_{null}": {
                    f"{cohort}_{metric}": paired_fold_deltas(per_fold[null], per_fold[treat], f"{cohort}_{metric}")
                    for cohort in ("heldout", "trained")
                    for metric in ("auc", "logloss")
                }
                for treat, null in (
                    ("gbm_cal", "global_rate"),
                    ("gbm_cal", "item_rate"),
                    ("gbm_cal", "gbm_inner"),
                    ("gbm", "global_rate"),
                )
            }
        return results
    finally:
        db.close()


def print_summary(results):
    for horizon, res in results.items():
        print(f"\n=== {horizon}d ===")
        for name, cells in res.get("_paired", {}).items():
            print(f"  {name}")
            for cell, d in cells.items():
                if not isinstance(d, dict) or d.get("mean") is None:
                    continue
                lo, hi = d.get("ci_low"), d.get("ci_high")
                sig = "" if lo is None or hi is None else ("  SIG" if (lo > 0 or hi < 0) else "  ns")
                print(f"    {cell:18s} {d['mean']:+.5f} [{lo:+.5f}, {hi:+.5f}] n={d.get('n_folds')}{sig}")


def main():
    # Set here, NOT at import — see anomaly_band_modulator_ab.main.
    os.environ["ANOMALY_GBM"] = "1"
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--metadata-parquet", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--calib-frac", type=float, default=0.25, help="fraction of train DATES held out to fit the map"
    )
    parser.add_argument("--clean-label", action="store_true", help="strictly-prior threshold (see anomaly_gbm_ab)")
    args = parser.parse_args()

    df, pruned, _ = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    results = run(
        df,
        pruned,
        horizon_filter=args.horizon,
        n_jobs=args.n_jobs,
        clean_label=args.clean_label,
        calib_frac=args.calib_frac,
    )
    print_summary(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
