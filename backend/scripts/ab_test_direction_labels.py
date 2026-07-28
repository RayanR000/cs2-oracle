#!/usr/bin/env python3
"""Sweep per-horizon vol multiplier k and mover-weight for the directional
classifier over purge-gap CV, scored against a FIXED ±0.5% yardstick.

Data loading mirrors ab_test_hp_search.py's `load_features` (duckdb over the
price-archive parquet files + ItemForecaster.engineer_features), NOT a
`build_training_frame()` method (that does not exist on ItemForecaster).

Only the directional classifier is trained here (no quantile models). The
evaluation yardstick is always the fixed ±0.5% realized-sign labels
(DIRECTION_FLAT_TOLERANCE_PCT), independent of whichever k is being swept, so
a bigger k can't just make its own yardstick easier.

Prints a per-horizon table; adopt winners by editing
ItemForecaster.DIRECTION_VOL_MULTIPLIER_MAP / DIRECTION_MOVER_WEIGHT_MAP in
models/forecaster.py. Nothing here writes model constants.

Usage:
    python -m scripts.ab_test_direction_labels [--max-items 200] [--horizon 14]
                                                [--step 60] [--purge-days N]
"""
import sys
import itertools
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from database import SessionLocal
from models.forecaster import (
    ItemForecaster,
    DIRECTION_LABEL_VOL_COL,
    DIRECTION_FLAT_TOLERANCE_PCT,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_direction_labels")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

K_GRID = [0.5, 1.0, 1.5, 2.0]
MOVER_WEIGHT_GRID = [1.5, 3.0, 5.0]
HORIZONS = [3, 7, 14, 30]

VAL_WINDOW_DAYS = 21
STEP = 60
MIN_TRAIN_ROWS = 2000
MIN_VAL_ROWS = 200


def score_fixed_yardstick(pred_cls, actual_returns):
    """Accuracy vs the fixed ±0.5% realized-sign labels, and accuracy over
    movers only (|actual| > 0.5%). Returns (overall_acc, movers_acc).

    This is the anti-gaming yardstick: it is FIXED regardless of whichever
    k / sigma the classifier was trained with, so a combo can't inflate its
    own score by widening its training band.
    """
    actual = np.asarray(actual_returns, dtype=float)
    actual_cls = ItemForecaster._direction_classes(actual, DIRECTION_FLAT_TOLERANCE_PCT)
    pred = np.asarray(pred_cls, dtype=int)
    overall = float((pred == actual_cls).mean()) if len(pred) else float("nan")
    mover_mask = np.abs(actual) > DIRECTION_FLAT_TOLERANCE_PCT
    movers = (float((pred[mover_mask] == actual_cls[mover_mask]).mean())
              if mover_mask.any() else float("nan"))
    return overall, movers


def load_features(con, forecaster, events_df, max_items):
    """Load price history for the top `max_items` items (by row count) from
    the price-archive parquet files, then run ItemForecaster's feature
    pipeline. Mirrors ab_test_hp_search.py's load_features exactly so this
    sweep trains on the same data path as the real HP-search A/B."""
    pq_files = sorted(str(p) for p in ARCHIVE_DIR.glob("prices-*.parquet"))
    pq_queries = []
    for pqf in pq_files:
        cols = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()}
        if "source" in cols:
            pq_queries.append(f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') WHERE source = 'STEAMCOMMUNITY'")
        else:
            pq_queries.append(f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}')")
    union_sql = " UNION ALL BY NAME ".join(pq_queries)

    items = con.sql(f"""
        SELECT item_slug, COUNT(*) AS row_count
        FROM ({union_sql})
        GROUP BY item_slug HAVING row_count >= 90
        ORDER BY row_count DESC LIMIT {max_items}
    """).fetchall()
    logger.info(f"  {len(items)} items for evaluation")

    all_rows = []
    for item_slug, _ in items:
        rows = con.sql(f"""
            SELECT item_slug AS item_id, day AS timestamp, mean_price AS price, volume
            FROM ({union_sql}) WHERE item_slug = ? ORDER BY day
        """, params=[item_slug]).fetchall()
        idf = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
        idf["timestamp"] = pd.to_datetime(idf["timestamp"])
        idf["date"] = idf["timestamp"].dt.date
        all_rows.append(idf)

    all_prices = pd.concat(all_rows, ignore_index=True)
    df = forecaster.engineer_features(all_prices, events_df)
    df = forecaster._add_cross_sectional_features(df)

    EXCLUDE = {"item_id", "date", "timestamp", "price", "volume", "name", "release_date"}
    feat_cols = [c for c in df.columns if c not in EXCLUDE
                 and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]
    if len(feat_cols) > 2:
        corr = df[feat_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            to_drop.update(upper[col][upper[col] > 0.95].index)
        feat_cols = [c for c in feat_cols if c not in to_drop]
    logger.info(f"  {len(feat_cols)} features after corr prune")
    return df, feat_cols


def _build_folds(dates, step=STEP):
    """Fixed walk-forward folds: (train_dates, val_dates). Same windowing as
    ab_test_hp_search.py's `_build_folds` (2/3 split, step-day stride,
    VAL_WINDOW_DAYS validation window)."""
    split_idx = len(dates) * 2 // 3
    folds = []
    for window_end in range(split_idx + 1, len(dates), step):
        train_dates = dates[:window_end]
        val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]
        if len(val_dates) < 7:
            continue
        folds.append((train_dates, val_dates))
    return folds


def _apply_purge_gap(folds, dates, purge_days):
    """Apply a purge/embargo gap of `purge_days` between the end of train and
    the start of val in each fold, dropping any val dates that fall within
    `purge_days` of the last train date. Prevents label leakage: a training
    row's `horizon`-day-forward target can otherwise reach into the
    validation window."""
    dates = list(dates)
    purged = []
    for train_dates, val_dates in folds:
        if len(train_dates) == 0:
            continue
        last_train = pd.Timestamp(train_dates[-1])
        gapped_val = [d for d in val_dates if (pd.Timestamp(d) - last_train).days > purge_days]
        if len(gapped_val) < 7:
            continue
        purged.append((train_dates, np.array(gapped_val)))
    return purged


def _run_horizon(fc, tdf, feat_cols, horizon, purge_days):
    """Yield result dicts for every (k, mover_weight) combo at this horizon."""
    target_col = f"target_return_{horizon}d"
    dates = np.array(sorted(tdf["date"].unique()))
    folds = _build_folds(dates, STEP)
    folds = _apply_purge_gap(folds, dates, purge_days)
    boosting_type = fc.BOOSTING_TYPE_MAP.get(horizon, "gbdt")

    if not folds:
        logger.warning(f"  no folds for {horizon}d after purge gap")
        return

    for k, mw in itertools.product(K_GRID, MOVER_WEIGHT_GRID):
        fc.DIRECTION_VOL_MULTIPLIER_MAP[horizon] = k
        fc.DIRECTION_MOVER_WEIGHT_MAP[horizon] = mw
        fold_overall, fold_movers = [], []
        for train_dates, val_dates in folds:
            tr = tdf[tdf["date"].isin(train_dates)]
            va = tdf[tdf["date"].isin(val_dates)]
            if len(tr) < MIN_TRAIN_ROWS or len(va) < MIN_VAL_ROWS:
                continue
            X_tr, X_va = tr[feat_cols].fillna(0.0), va[feat_cols].fillna(0.0)
            clf = fc._fit_direction_classifier(
                X_tr, tr[target_col].to_numpy(dtype=float),
                X_va, va[target_col].to_numpy(dtype=float),
                boosting_type,
                fc._direction_tree_params({}),
                horizon,
                sigma_train=tr[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float),
                sigma_val=va[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float))
            pred_cls = clf.predict(X_va).argmax(axis=1)
            ov, mv = score_fixed_yardstick(pred_cls, va[target_col].to_numpy(dtype=float))
            fold_overall.append(ov)
            fold_movers.append(mv)
        yield {
            "horizon": horizon, "k": k, "mover_weight": mw,
            "overall_acc": round(100 * np.nanmean(fold_overall), 2) if fold_overall else None,
            "movers_acc": round(100 * np.nanmean(fold_movers), 2) if fold_movers else None,
            "n_folds": len(fold_overall),
        }


def run(max_items, horizon_filter, purge_days_arg):
    import duckdb
    con = duckdb.connect()
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()
        df, feat_cols = load_features(con, forecaster, events_df, max_items)
    finally:
        con.close()

    # Restrict to the production feature allowlist (price_technicals only) so
    # the sweep matches what production actually trains the classifier on.
    feat_cols = forecaster._apply_feature_allowlist(feat_cols, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
    feat_cols = [c for c in feat_cols if c in df.columns]

    horizons = [h for h in HORIZONS if horizon_filter is None or h == horizon_filter]

    print(f"{'h':>3} {'k':>4} {'mw':>4} {'overall%':>9} {'movers%':>8} {'folds':>6}")
    results = []
    for horizon in horizons:
        target_col = f"target_return_{horizon}d"
        tdf = forecaster.prepare_targets(df, horizon)
        tdf = tdf.dropna(subset=[target_col]).sort_values(["item_id", "date"]).copy()
        if tdf.empty:
            logger.warning(f"  no targets for {horizon}d")
            continue
        purge_days = purge_days_arg if purge_days_arg is not None else horizon

        best = None
        for row in _run_horizon(forecaster, tdf, feat_cols, horizon, purge_days):
            results.append(row)
            print(f"{row['horizon']:>3} {row['k']:>4} {row['mover_weight']:>4} "
                  f"{str(row['overall_acc']):>9} {str(row['movers_acc']):>8} {row['n_folds']:>6}")
            if row["overall_acc"] is not None and (best is None or row["overall_acc"] > best["overall_acc"]):
                best = row
        if best:
            print(f"  -> best {horizon}d: k={best['k']} mover_weight={best['mover_weight']} "
                  f"overall={best['overall_acc']}% movers={best['movers_acc']}%")

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Sweep vol multiplier k / mover weight for the directional "
                    "classifier, scored against the fixed ±0.5% yardstick")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=None, choices=HORIZONS)
    ap.add_argument("--purge-days", type=int, default=None,
                    help="CV purge/embargo gap in days; default = horizon per horizon")
    args = ap.parse_args()

    logger.info("=" * 70)
    logger.info("SWEEP: directional classifier vol-multiplier k / mover-weight")
    logger.info(f"  max_items={args.max_items} horizon={args.horizon} "
                f"purge_days={args.purge_days}")
    logger.info("=" * 70)

    run(args.max_items, args.horizon, args.purge_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
