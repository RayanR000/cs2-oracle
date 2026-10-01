#!/usr/bin/env python3
"""Isotonic calibration of the EXCEEDANCE head, scored on held-out dates only.

Implements `docs/research/2026-09-13-exceedance-calibration-preregistration.md`
(written before any number below was computed). The exceedance head is the one
live magnitude signal here; unlike the band its question is calibration, and
production already ships an isotonic layer for it
(`_fit_exceedance_calibrator`, OOF pairs). This asks whether that layer states
better probabilities than the raw head on rows neither saw, and whether either
beats the featureless constant that is the shipping floor for a served
probability.

Two fixes applied in advance (see the prereg):

1. PINNED FOLD GRID. Real-OOF per-fold q_hat spread is 3.4-3.6x, so between-fold
   regime movement dwarfs between-arm movement. All five arms share one fold
   enumeration, one `_stratified_sample` per fold (seed depends on the fold,
   never the arm), and identical val rows. Deltas pair on fold index only.
2. HELD-OUT-ONLY SCORING. The anomaly modulator fitted its map on in-sample
   predicted p and applied it out-of-sample, so the bins misaligned. Here the
   map is fitted on a calibration slice date-disjoint from the inner train
   slice with the H+13 embargo at the inner boundary, applied to the val
   window, and scored ONLY there. Scoring a map on its fit rows voids the run.

Arms:

    gbm          full train fold, raw head (status quo ante).
    gbm_inner    inner train slice only; raw. Honest control for gbm_cal.
    gbm_cal      inner head mapped through the isotonic fit on the calib slice.
    item_rate    train-fold per-item exceedance frequency (min_obs=10, else
                 pooled). Expected to collapse onto global_rate on held-out.
    global_rate  train-fold pooled constant. The SHIPPING bar for log loss.

Primary cohort is held-out items at h=3/7/14; trained-eval is reported.
Horizons are never pooled.

Usage:
    python -m scripts.archive.exceedance_calibration_ab --horizon 7 \\
        --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \\
        --frame-cache /tmp/exc_cal_frame.parquet --out /tmp/exc_cal_h7.json
    python -m scripts.archive.exceedance_calibration_ab --served-only  # maturity + raw served reliability, no fit
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from scripts.archive.anomaly_calibration_ab import inner_calibration_split
from scripts.archive.anomaly_gbm_ab import item_rate_predictions
from scripts.archive.exceedance_meta_ab import TREE_PARAMS, _score, paired_fold_deltas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("exceedance_calibration_ab")

ARMS = ("gbm", "gbm_inner", "gbm_cal", "item_rate", "global_rate")

#: Production clips served exceedance probabilities into this range
#: (`exceedance_probability`), so every arm is clipped the same way before
#: scoring — otherwise a 0.0 from a featureless rate scores an infinite log
#: loss the GBM never risks.
P_CLIP = (1e-3, 1.0)


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), *P_CLIP)


def _brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return None
    return float(np.mean((p[ok] - y[ok]) ** 2))


def run(df, pruned, horizon_filter=7, n_jobs=None, calib_frac=0.25, max_folds=None):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    # One item split for every arm: the grid is pinned, never re-derived.
    eval_items, train_items, trained_eval = assign_items(df)

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        horizons = [h for h in ItemForecaster.HORIZONS if horizon_filter is None or h == horizon_filter]
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Exceedance calibration {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            target_col = f"target_exceed_{horizon}d"
            if target_col not in tdf.columns:
                logger.warning(f"    {target_col} absent — skipping")
                continue
            tdf = tdf.dropna(subset=[target_col]).sort_values(["item_id", "date"])
            if tdf.empty:
                logger.warning(f"    No valid exceedance labels for {horizon}d")
                continue
            logger.info(f"    label base rate: {tdf[target_col].mean():.4f} over {len(tdf):,} rows")

            base_cols = [c for c in pruned if c in tdf.columns]
            if not base_cols:
                logger.warning("    no allowlisted feature columns — skipping")
                continue
            sub = tdf[["item_id", "date", "price", target_col, *base_cols]].copy()

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

            per_fold = {a: [] for a in ARMS}
            cal_meta = []
            pooled = {a: {"p": [], "y": []} for a in ARMS}
            for fold_idx, window_end in enumerate(fold_list):
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
                # Pinned: one sample per fold, shared by every arm.
                train_df = _stratified_sample(train_df, train_items, ROW_BUDGET, fold_idx)

                med = train_df[base_cols].median()
                X_train = train_df[base_cols].fillna(med)
                X_val = val_df[base_cols].fillna(med)

                def _fit(Xtr, ytr, horizon=horizon):
                    return forecaster._fit_exceedance_classifier(
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

                p_item, pooled_rate = item_rate_predictions(train_df, val_df, target_col)
                preds = {
                    "gbm": head.predict(X_val),
                    "gbm_inner": p_inner,
                    "gbm_cal": p_cal,
                    "item_rate": p_item,
                    "global_rate": np.full(len(val_df), pooled_rate),
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
                        row[f"{cohort}_brier"] = _brier(y[sel], p[sel])
                        row[f"{cohort}_n"] = n
                        if cohort == "heldout" and bool(sel.any()):
                            pooled[arm]["p"].append(p[sel])
                            pooled[arm]["y"].append(y[sel])
                    per_fold[arm].append(row)

            if not per_fold["gbm_cal"]:
                logger.warning(f"    no calibrated folds at {horizon}d")
                continue
            entry = {arm: {"per_fold": rows} for arm, rows in per_fold.items()}
            entry["_calibrators"] = cal_meta
            # Pooled held-out reliability tables: diagnostic, never the verdict
            # (folds are regime-heterogeneous; the verdict is the paired mean).
            entry["_reliability"] = {}
            for arm in ARMS:
                if not pooled[arm]["p"]:
                    continue
                p_all = np.concatenate(pooled[arm]["p"])
                y_all = np.concatenate(pooled[arm]["y"])
                table = ItemForecaster.exceedance_reliability_table(p_all, y_all)
                entry["_reliability"][arm] = {
                    "n": len(p_all),
                    "brier": round(float(np.mean((p_all - y_all) ** 2)), 5),
                    "ece_pp": round(100.0 * ItemForecaster.exceedance_ece(table), 3),
                    "table": table,
                }
            results[horizon] = entry
            for arm in ARMS:
                rows = per_fold[arm]
                for cohort in ("heldout", "trained"):
                    lls = [r[f"{cohort}_logloss"] for r in rows if r[f"{cohort}_logloss"] is not None]
                    if lls:
                        aucs = [r[f"{cohort}_auc"] for r in rows if r[f"{cohort}_auc"] is not None]
                        auc_txt = f"AUC={np.mean(aucs):.4f} " if aucs else "AUC=n/a "
                        logger.info(
                            f"      {arm:12s} {cohort:8s} {auc_txt}logloss={np.mean(lls):.5f} ({len(lls)} folds)"
                        )

            results[horizon]["_paired"] = {
                f"{treat}_vs_{null}": {
                    f"{cohort}_{metric}": paired_fold_deltas(per_fold[null], per_fold[treat], f"{cohort}_{metric}")
                    for cohort in ("heldout", "trained")
                    for metric in ("auc", "logloss", "brier")
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


def served_panel():
    """Maturity + RAW served reliability at h=3/7/14 (/30). Read-only, no fit.

    5-6 dates per horizon cannot support an honest isotonic fit, so this fits
    nothing: it is the baseline the next retrain's
    `cv_results[h]["exceedance_calibration"]` will be checked against once the
    panel matures past MIN_HEADLINE_DATES=20.
    """
    from backtest.friction import actionable_threshold
    from backtest.scoring import MIN_HEADLINE_DATES, excluded_forecast_date, price_tier
    from sqlalchemy import text

    db = SessionLocal()
    try:
        rows = db.execute(
            text("""
            SELECT o.horizon_days, o.forecast_date, o.base_price,
                   o.actual_price, o.current_price, f.exceed_p
            FROM forecast_outcomes o
            JOIN item_forecasts f ON f.id = o.forecast_id
            WHERE o.actual_price IS NOT NULL AND o.base_price >= 1.0
        """)
        ).fetchall()
    finally:
        db.close()
    df = pd.DataFrame(rows, columns=["h", "forecast_date", "base_price", "actual_price", "current_price", "exceed_p"])
    if df.empty:
        return {"mature": False, "note": "empty served panel"}
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    reasons = df["forecast_date"].map(excluded_forecast_date)
    df = df[reasons.isna()].reset_index(drop=True)
    out = {"min_forecast_dates": MIN_HEADLINE_DATES, "horizons": {}}
    for h, g in sorted(df.groupby("h")):
        g = g.copy()
        b = pd.to_numeric(g["base_price"], errors="coerce").to_numpy(float)
        a = pd.to_numeric(g["actual_price"], errors="coerce").to_numpy(float)
        cur = pd.to_numeric(g["current_price"], errors="coerce").to_numpy(float)
        q = np.where(np.isfinite(cur) & (cur > 0), cur, b)
        p = pd.to_numeric(g["exceed_p"], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            actual_ret = a / q - 1.0
        have = np.isfinite(p) & np.isfinite(actual_ret) & np.isfinite(q) & (q > 0)
        n_dates = int(g["forecast_date"].nunique())
        n_dates_p = int(g.loc[have, "forecast_date"].nunique())
        entry = {
            "rows": len(g),
            "dates": n_dates,
            "rows_with_exceed_p": int(have.sum()),
            "dates_with_exceed_p": n_dates_p,
            "mature": bool(n_dates_p >= MIN_HEADLINE_DATES),
        }
        if have.any():
            thr = np.array([actionable_threshold(price_tier(float(c)), "csfloat") for c in q[have]])
            y = (actual_ret[have] > thr).astype(float)
            pc = _clip(p[have])
            table = ItemForecaster.exceedance_reliability_table(pc, y)
            entry["base_rate"] = round(float(y.mean()), 4)
            entry["brier_raw"] = round(float(np.mean((pc - y) ** 2)), 5)
            entry["ece_raw_pp"] = round(100.0 * ItemForecaster.exceedance_ece(table), 3)
            entry["reliability_raw"] = table
        out["horizons"][int(h)] = entry
    return out


def print_summary(results):
    for horizon, res in results.items():
        if horizon == "served":
            continue
        print(f"\n=== {horizon}d ===")
        for name, cells in res.get("_paired", {}).items():
            print(f"  {name}")
            for cell, d in cells.items():
                if not isinstance(d, dict) or d.get("mean") is None:
                    continue
                lo, hi = d.get("ci_low"), d.get("ci_high")
                sig = "" if lo is None or hi is None else ("  SIG" if (lo > 0 or hi < 0) else "  ns")
                print(f"    {cell:18s} {d['mean']:+.5f} [{lo:+.5f}, {hi:+.5f}] n={d.get('n_folds')}{sig}")
        rel = res.get("_reliability", {})
        for arm, r in rel.items():
            print(f"    [pooled held-out] {arm:12s} n={r['n']:,} brier={r['brier']:.5f} ECE={r['ece_pp']:.2f}pp")


def print_served(served):
    print("\n=== served exceed_p panel (read-only, no fit) ===")
    for h, e in sorted(served.get("horizons", {}).items()):
        tag = "MATURE" if e["mature"] else "immature"
        print(
            f"  h={h}: {e['rows_with_exceed_p']:,} rows / "
            f"{e['dates_with_exceed_p']} dates [{tag}] "
            f"(gate={served['min_forecast_dates']})",
            end="",
        )
        if "ece_raw_pp" in e:
            print(f" base={e['base_rate']:.3f} brier={e['brier_raw']:.5f} ECE={e['ece_raw_pp']:.2f}pp")
            for r in e["reliability_raw"]:
                print(
                    f"      [{r['lo']:.1f}-{r['hi']:.1f}] n={r['n']:>5,} "
                    f"pred={r['pred']:.3f} realized={r['realized']:.3f}"
                )
        else:
            print(" (no exceed_p)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--horizons", type=str, default=None, help="comma list, e.g. 3,7,14 (overrides --horizon)")
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--metadata-parquet", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--calib-frac", type=float, default=0.25, help="fraction of train DATES held out to fit the map"
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="score only the K most recent folds (quick read; the verdict needs the full run)",
    )
    parser.add_argument("--served-only", action="store_true", help="report served maturity + raw reliability only")
    args = parser.parse_args()

    if args.served_only:
        served = served_panel()
        print_served(served)
        if args.out:
            Path(args.out).write_text(json.dumps(served, indent=2, default=str))
            print(f"\nWrote {args.out}")
        return

    if args.horizons is not None:
        wanted = {int(x) for x in args.horizons.split(",") if x.strip()}
    elif args.horizon is not None:
        wanted = {args.horizon}
    else:
        wanted = None

    df, pruned, _ = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    # One run() per horizon: the item split is re-seeded identically each
    # call (fixed SPLIT_SEED), so the grid stays pinned across arms within
    # every horizon; cross-horizon pinning is meaningless (different labels
    # and embargo per h). Looping also avoids paying for h=30 when the
    # prereg scopes the verdict to 3/7/14.
    from models.forecaster import ItemForecaster as _F

    wanted_list = [h for h in _F.HORIZONS if wanted is None or h in wanted]
    results = {}
    for h in wanted_list:
        results.update(
            run(df, pruned, horizon_filter=h, n_jobs=args.n_jobs, calib_frac=args.calib_frac, max_folds=args.max_folds)
        )
    served = served_panel()
    results["served"] = served
    print_summary(results)
    print_served(served)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
