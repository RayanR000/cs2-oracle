#!/usr/bin/env python3
"""Does training on the 2025 clean-label era produce narrower bands?

The 2025 label ceiling is R²=0.92 at h=30 (vs 0.64 for 2026), because the
2025 era has 0.7-1.5% frozen quotes vs 18-33% in 2026. The hypothesis: a
climatology table built on cleaner labels should produce narrower bands at
matched coverage — even with less data.

All arms are evaluated on the SAME 2026+ rows (the only window where no arm's
training data overlaps eval). The arms differ only in which subset of the fit
split they use to build the climatology table:

  control:       full fit (all data up to the fold boundary, ~4 years)
  clean_2025:    fit rows from 2025-01-01..2025-12-31 only
  recent_1yr:    fit rows from the last 365 days before the fold boundary
                 (a recency control — same depth as clean_2025 but more recent)

Metric: paired per-fold delta on log-width at matched 80% coverage, same as
shrink_k_vol_rank_ab.py. Negative = narrower = better.

Because the 2026 eval window is only ~250 days, we use a 14-day step (not the
family's default 60) to get enough folds for a paired interval. Eval windows
overlap at long horizons — accepted as a first read; the paired interval is
still valid under correlation (conservative), not liberal.

Usage:
    venv/bin/python -m scripts.archive.clean_label_training_ab \\
        --frame-cache /tmp/skvr_frame.parquet \\
        --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \\
        --out /tmp/clean_label_ab.json
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("CLIMATOLOGY_SCALE", "1")

import numpy as np
import pandas as pd
from api.serving_policy import MIN_SERVED_PRICE_USD
from database import SessionLocal
from models.forecaster import ItemForecaster
from scripts.archive.ab_test_item_metadata import (
    assign_items,
    build_frame,
)
from scripts.archive.exceedance_meta_ab import paired_fold_deltas
from scripts.archive.shrink_k_vol_rank_ab import _lookup, matched_width

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("clean_label_training_ab")

TARGET_COVERAGE = 0.80
MIN_EVAL_ROWS = 200
STEP_DAYS = 14
VAL_WINDOW_DAYS = 21

CLEAN_2025_START = "2025-01-01"
CLEAN_2025_END = "2025-12-31"
EVAL_START = "2026-01-01"

ARMS = ("control", "clean_2025", "recent_1yr")


def _filter_fit(fit_df, arm, fold_boundary_date):
    """Restrict fit rows to the arm's training window."""
    if arm == "control":
        return fit_df
    dates = pd.to_datetime(fit_df["date"])
    if arm == "clean_2025":
        mask = (dates >= pd.Timestamp(CLEAN_2025_START)) & (dates <= pd.Timestamp(CLEAN_2025_END))
        return fit_df[mask]
    if arm == "recent_1yr":
        cutoff = pd.Timestamp(fold_boundary_date) - pd.to_timedelta(365, unit="D")
        return fit_df[dates >= cutoff]
    raise ValueError(f"unknown arm {arm}")


def run(df, horizon_filter=None, max_folds=None):
    eval_items, train_items, trained_eval = assign_items(df)

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        horizons = [h for h in ItemForecaster.HORIZONS if horizon_filter is None or h == horizon_filter]
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Clean-label training {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            tcol = f"target_return_{horizon}d"
            if tcol not in tdf.columns:
                logger.warning(f"    {tcol} absent — skipping")
                continue
            tdf = tdf.dropna(subset=[tcol]).sort_values(["item_id", "date"])
            if tdf.empty:
                continue
            sub = tdf[["item_id", "date", "price", tcol]].copy()

            dates = sorted(sub["date"].unique())
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            is_heldout = sub["item_id"].isin(set(eval_items)).to_numpy()
            is_trained_eval = sub["item_id"].isin(set(trained_eval)).to_numpy()
            is_train_item = sub["item_id"].isin(set(train_items)).to_numpy()

            # Only eval on 2026+ dates so no arm's training overlaps eval.
            _eval_ts = pd.Timestamp(EVAL_START)
            eval_start_idx = next((i for i, d in enumerate(dates) if pd.Timestamp(d) >= _eval_ts), None)
            if eval_start_idx is None:
                logger.warning("    no 2026+ dates — skipping")
                continue

            fold_list = list(range(eval_start_idx, len(dates), STEP_DAYS))
            if max_folds is not None:
                fold_list = fold_list[-max_folds:]

            per_fold = {arm: [] for arm in ARMS}
            for fold_idx, window_end in enumerate(fold_list):
                val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue
                in_fit = sub_days <= dates_dt[window_end - 1]
                in_val = (sub_days >= dates_dt[window_end]) & (sub_days <= dates_dt[window_end + len(val_dates) - 1])
                fit_df_full = ItemForecaster._purge_overlapping_train_rows(
                    sub[in_fit & is_train_item], val_dates[0], horizon
                )
                val_df = sub[in_val & (is_heldout | is_trained_eval)]
                val_df = val_df[val_df["price"] >= MIN_SERVED_PRICE_USD]
                if len(val_df) < MIN_EVAL_ROWS or fit_df_full.empty:
                    continue

                val_ids = val_df["item_id"].to_numpy()
                val_px = val_df["price"].to_numpy(dtype=float)
                abs_r = val_df[tcol].abs().to_numpy(dtype=float)
                fold_boundary = pd.Timestamp(dates[window_end - 1])

                row = {"fold": fold_idx, "val_start": str(val_dates[0]), "n_eval": len(val_df)}

                for arm in ARMS:
                    arm_fit = _filter_fit(fit_df_full, arm, fold_boundary)
                    if arm_fit.empty or len(arm_fit) < 100:
                        per_fold[arm].append(dict(row, width=None, log_width=None, n_fit=0))
                        logger.warning(f"    fold {fold_idx} {arm}: only {len(arm_fit)} fit rows — skipped")
                        continue
                    fit_min = arm_fit[["item_id", "price", tcol]]
                    try:
                        table, pool, g = ItemForecaster._build_climatology_table(fit_min, tcol)
                    except Exception as exc:
                        per_fold[arm].append(dict(row, width=None, log_width=None, n_fit=len(arm_fit)))
                        logger.warning(f"    fold {fold_idx} {arm}: climatology failed ({exc!r})")
                        continue
                    if not table:
                        per_fold[arm].append(dict(row, width=None, log_width=None, n_fit=len(arm_fit)))
                        continue
                    scale = _lookup(
                        forecaster, horizon, {"table": table, "tier_pool": pool, "global": g}, val_ids, val_px
                    )
                    forecaster.climatology_scale.pop(horizon, None)

                    _, w = matched_width(abs_r, scale)
                    per_fold[arm].append(
                        dict(
                            row,
                            n_fit=len(arm_fit),
                            width=(None if not np.isfinite(w) else float(w)),
                            log_width=(None if not np.isfinite(w) or w <= 0 else float(np.log(w))),
                        )
                    )

                for arm in ARMS:
                    r = per_fold[arm][-1]
                    w_str = f"{r['width']:.3f}%" if r.get("width") else "n/a"
                    logger.info(
                        f"    fold {fold_idx} ({val_dates[0]}): {arm:12s} n_fit={r.get('n_fit', '?'):>7} width={w_str}"
                    )

            if not [r for r in per_fold["control"] if r["width"] is not None]:
                logger.warning(f"    no usable folds at {horizon}d")
                continue
            results[horizon] = {arm: {"per_fold": rows} for arm, rows in per_fold.items()}
            results[horizon]["_paired"] = {
                arm: paired_fold_deltas(per_fold["control"], per_fold[arm], "log_width")
                for arm in ARMS
                if arm != "control"
            }
            for arm in ARMS:
                ws = [r["width"] for r in per_fold[arm] if r["width"] is not None]
                fits = [r.get("n_fit", 0) for r in per_fold[arm] if r["width"] is not None]
                if ws:
                    logger.info(
                        f"      {arm:12s} mean width={np.mean(ws):.3f}% mean fit={np.mean(fits):.0f} ({len(ws)} folds)"
                    )
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 80)
    print("CLEAN-LABEL TRAINING — paired per-fold deltas on log-width")
    print("negative delta = NARROWER at matched 80% coverage (good)")
    print("=" * 80)
    for horizon, entry in sorted(results.items()):
        n_ctl = sum(1 for r in entry["control"]["per_fold"] if r["width"] is not None)
        ctl_fits = [r.get("n_fit", 0) for r in entry["control"]["per_fold"] if r["width"] is not None]
        print(f"\nh={horizon}d   control folds={n_ctl} mean_fit={np.mean(ctl_fits):.0f}")
        for arm in ARMS:
            if arm == "control":
                continue
            d = entry["_paired"].get(arm)
            rows = entry[arm]["per_fold"]
            arm_fits = [r.get("n_fit", 0) for r in rows if r["width"] is not None]
            base = {r["fold"]: r["width"] for r in entry["control"]["per_fold"]}
            ratios = [
                r["width"] / base[r["fold"]]
                for r in rows
                if r["width"] is not None and base.get(r["fold"]) is not None and base[r["fold"]] > 0
            ]
            rinfo = (
                f"mean ratio {np.mean(ratios):.4f}, "
                f"worst {np.max(ratios):.4f}, "
                f"best {np.min(ratios):.4f}, "
                f"narrower {sum(x < 1.0 for x in ratios)}/{len(ratios)}"
                if ratios
                else "no paired folds"
            )
            fit_info = f"mean_fit={np.mean(arm_fits):.0f}" if arm_fits else "no fits"
            if d is None:
                print(f"  {arm:14s} — too few paired folds; {fit_info}; {rinfo}")
                continue
            flag = "*" if d["excludes_zero"] else " "
            better = d["n_folds"] - d["wins"]
            print(
                f"  {arm:14s} d={d['mean']:+.5f} "
                f"[{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]{flag} "
                f"narrower {better}/{d['n_folds']}; {fit_info}; {rinfo}"
            )
    print("\n* = 95% interval excludes zero.")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--frame-cache", default=None)
    ap.add_argument("--metadata-parquet", default=None)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df, _, _ = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    results = run(df, horizon_filter=args.horizon, max_folds=args.max_folds)
    print_summary(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
