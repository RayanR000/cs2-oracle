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
                                                [--purge-days N]
"""
import sys
import itertools
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import paired_records
from database import SessionLocal
from models.forecaster import (
    ItemForecaster,
    DIRECTION_LABEL_VOL_COL,
    DIRECTION_FLAT_TOLERANCE_PCT,
    embargo_days,
    phase_collapsed_sql_filter,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_direction_labels")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Production's item universe, spelled into every archive read this harness
# makes. Before 2026-08-08 the `ab_test_*` family globbed the Parquet privately
# and saw a universe production does not train on, so an A/B advised a model it
# had not measured. The bid sources need no clause here: `STEAMCOMMUNITY` is a
# single ask feed and already excludes them. See `models/item_parser.py`.
_UNIVERSE = phase_collapsed_sql_filter()


K_GRID = [0.25, 0.5, 1.0]
MOVER_WEIGHT_GRID = [3.0, 5.0, 8.0]
HORIZONS = [3, 7, 14, 30]

MIN_TRAIN_ROWS = 2000
MIN_VAL_ROWS = 200

# Recent folds are more live-relevant than 2013-era data; capping folds is
# the main speedup lever (a prior 29-fold run over the full 2013-2026
# history was too slow to iterate on).
MAX_FOLDS = 8


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
            # `source = 'STEAMCOMMUNITY'` matches 0 rows post archive-rebuild, so
            # this degenerated to the NULL (pre-2026) branch; the dead disjunct is
            # dropped, keeping the intended pre-2026 cohort. See 2026-08-13 repin.
            pq_queries.append(f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') WHERE source IS NULL AND {_UNIVERSE}")
        else:
            pq_queries.append(f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') WHERE {_UNIVERSE}")
    union_sql = " UNION ALL BY NAME ".join(pq_queries)

    items = con.sql(f"""
        SELECT item_slug, COUNT(*) AS row_count
        FROM ({union_sql})
        GROUP BY item_slug HAVING row_count >= 90
        ORDER BY row_count DESC, item_slug LIMIT {max_items}
    """).fetchall()
    if not items:
        raise RuntimeError(
            "direction_labels universe query selected 0 items — the source pin "
            "matched no rows (see 2026-08-13 harness repin).")
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


def _run_horizon(fc, tdf, feat_cols, horizon, purge_days):
    """Yield result dicts for every (k, mover_weight) combo at this horizon.

    CV folds come from the production method `ItemForecaster._compute_cv_splits`
    (models/forecaster.py), the same purge-gap CV the design cites as the
    acceptance gate. It purges the TRAIN side (drops train dates whose
    horizon-day-forward target would land inside the val window), so the val
    window width is unaffected by purge_days — unlike purging val dates,
    which degenerates (or empties entirely) once purge_days exceeds the val
    window width, as happens for horizon=30 > VAL_WINDOW_DAYS=21.
    """
    target_col = f"target_return_{horizon}d"
    dates = np.array(sorted(tdf["date"].unique()))
    folds = fc._compute_cv_splits(dates, purge_days=purge_days)
    boosting_type = fc.BOOSTING_TYPE

    if not folds:
        logger.warning(f"  no folds for {horizon}d after purge gap")
        return

    # Cap to the most recent MAX_FOLDS folds: recent data is more
    # live-relevant, and this is the main sweep-speedup lever.
    folds = folds[-MAX_FOLDS:]

    # Control cell first: k=None is the sentinel for "current production
    # config" -- fixed ±0.5% band (sigma_train=sigma_val=None) with
    # mover_weight=3.0. This is the apples-to-apples in-harness baseline
    # that vol-scaled cells must beat.
    combos = [(None, 3.0)] + list(itertools.product(K_GRID, MOVER_WEIGHT_GRID))

    for k, mw in combos:
        is_control = k is None
        # Process-local mutation of the class-level maps; fine for this
        # single-process CLI sweep (never persisted back to forecaster.py).
        fc.DIRECTION_VOL_MULTIPLIER_MAP[horizon] = k if not is_control else 1.0
        fc.DIRECTION_MOVER_WEIGHT_MAP[horizon] = mw
        fold_overall, fold_movers = [], []
        records = []
        for fold_idx, (train_dates, val_dates) in enumerate(folds):
            tr = tdf[tdf["date"].isin(train_dates)]
            va = tdf[tdf["date"].isin(val_dates)]
            if len(tr) < MIN_TRAIN_ROWS or len(va) < MIN_VAL_ROWS:
                continue
            X_tr, X_va = tr[feat_cols].fillna(0.0), va[feat_cols].fillna(0.0)
            if is_control:
                sigma_train = None
                sigma_val = None
            else:
                sigma_train = tr[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float)
                sigma_val = va[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float)
            clf = fc._fit_direction_classifier(
                X_tr, tr[target_col].to_numpy(dtype=float),
                X_va, va[target_col].to_numpy(dtype=float),
                boosting_type,
                fc._direction_tree_params({}),
                horizon,
                sigma_train=sigma_train,
                sigma_val=sigma_val)
            pred_cls = clf.predict(X_va).argmax(axis=1)
            actual = va[target_col].to_numpy(dtype=float)
            ov, mv = score_fixed_yardstick(pred_cls, actual)
            fold_overall.append(ov)
            fold_movers.append(mv)

            # Paired records against the control cell, for the fold-clustered
            # interval `run` computes below. Scored on the same FIXED +/-0.5%
            # yardstick the sweep reports, over movers only -- a flat actual is
            # not a directional call and the control cell answers "flat" for
            # free on it. Every cell sees identical folds and identical rows,
            # so the pairing is exact.
            actual_cls = ItemForecaster._direction_classes(
                actual, DIRECTION_FLAT_TOLERANCE_PCT)
            records.extend(paired_records(
                item_ids=va["item_id"].to_numpy(),
                forecast_dates=va["date"].to_numpy(),
                fold_id=fold_idx,
                keep=np.abs(actual) > DIRECTION_FLAT_TOLERANCE_PCT,
                direction_correct=(np.asarray(pred_cls, dtype=int) == actual_cls),
            ))
        yield {
            "horizon": horizon, "k": "ctrl" if is_control else k, "mover_weight": mw,
            "overall_acc": round(100 * np.nanmean(fold_overall), 2) if fold_overall else None,
            "movers_acc": round(100 * np.nanmean(fold_movers), 2) if fold_movers else None,
            "n_folds": len(fold_overall),
            "records": records,
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
        purge_days = (purge_days_arg if purge_days_arg is not None
                      else embargo_days(horizon))

        best, control = None, None
        for row in _run_horizon(forecaster, tdf, feat_cols, horizon, purge_days):
            results.append(row)
            print(f"{row['horizon']:>3} {str(row['k']):>4} {row['mover_weight']:>4} "
                  f"{str(row['overall_acc']):>9} {str(row['movers_acc']):>8} {row['n_folds']:>6}")
            if row["k"] == "ctrl":
                control = row
                continue
            if row["overall_acc"] is not None and (best is None or row["overall_acc"] > best["overall_acc"]):
                best = row
        if control:
            print(f"  -> control {horizon}d: overall={control['overall_acc']}% "
                  f"movers={control['movers_acc']}%")
        if best:
            print(f"  -> best {horizon}d: k={best['k']} mover_weight={best['mover_weight']} "
                  f"overall={best['overall_acc']}% movers={best['movers_acc']}%")

        # Fold-clustered paired intervals against the control cell. Until
        # 2026-08-08 this sweep reported a raw mean per cell and picked the
        # highest, with no interval anywhere -- so "best" meant "won the draw",
        # and with ten cells against an item-level MDE of 2.21-3.69pp a winner
        # was guaranteed whether or not any cell differed from control.
        if control and control.get("records"):
            cells = {
                f"k={r['k']},mw={r['mover_weight']}": r["records"]
                for r in results if r["horizon"] == horizon and r.get("records")
            }
            ctrl_key = f"k=ctrl,mw={control['mover_weight']}"
            if ctrl_key in cells and len(cells) > 1:
                contrasts = paired_arm_contrasts(cells, base=ctrl_key)
                for cell, paired in sorted(contrasts.items()):
                    print(f"     paired {cell:<20} vs control (movers): "
                          f"{format_paired(paired)}")
                control["paired_vs_control"] = contrasts

        # The records are the pairing input, not a result. Dropping them keeps
        # the returned rows printable -- one cell at one horizon carries tens
        # of thousands.
        for r in results:
            r.pop("records", None)

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Sweep vol multiplier k / mover weight for the directional "
                    "classifier, scored against the fixed ±0.5% yardstick")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=None, choices=HORIZONS)
    ap.add_argument("--purge-days", type=int, default=None,
                    help="CV purge/embargo gap in days; default = "
                         "embargo_days(horizon), i.e. horizon + 13")
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
