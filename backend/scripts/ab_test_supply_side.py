#!/usr/bin/env python3
"""
A/B test: compare prediction accuracy WITH vs WITHOUT supply-side features
(rarity, weapon_type, weapon-type cross-sectional group features).

Trains two models on the same expanding-window walk-forward evaluation:
  - Model A (control):  no supply-side features
  - Model B (treatment): with supply-side features

Usage:
    python scripts/ab_test_supply_side.py [--max-items 200]

Embargo (added 2026-08-08):
    This harness had **no purge gap at all** before that date: the fold split
    was `train = every date <= window_end - 1`, so every training row within
    `horizon` days of the boundary carried a label resolved from inside the
    validation window. The train side now goes through production's own
    `ItemForecaster._purge_overlapping_train_rows`, which embargoes
    `embargo_days(horizon)` = `horizon + 13` days -- the label's resolved-anchor
    support, not its nominal date. The validation window is untouched; purging
    it would empty the 21-day window at h=30.

    **Every delta this harness printed before 2026-08-08 is un-embargoed.**
    How much that inflated them is not known here. The often-quoted
    "+12.1pp unpurged -> +6.1pp purged" at h=30 is from the external review
    (docs/research/2026-08-07-cs2-forecasting-research.md), describes an
    event-calendar arm, and has never been replicated in this repo.
"""

import sys
import json
import math
import logging
import copy
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from database import SessionLocal
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import (
    paired_records,
    without_records,
)
from db.archive import prices_relation
from models.forecaster import ItemForecaster, archive_universe_sql_filter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_supply_side")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Production's item universe and price consensus, spelled into every archive
# read this harness makes. Before 2026-08-08 this was a raw
# `read_parquet('prices-*.parquet')` glob, which DuckDB narrows to the first
# file's schema so `source` is not even a column there, and it carried no
# universe rule at all — so the BUFF bid voted straight into its price series.
# `prices_relation` is the reader that projects `source` correctly whether or
# not the archive has been migrated. See `backend/AGENTS.md` and `db/archive.py`.
#
# Still outstanding: this reads EVERY ask source, where most harnesses filter
# to one, so a 2026 item-day reaches it several times over and
# `engineer_features` collapses the copies with a plain mean rather than
# production's outlier-voted median. `backend/AGENTS.md` puts the archive's
# duplication at 1.37x item-days.
_PRICE_COLUMNS = ["item_slug", "day", "mean_price", "volume", "source"]
_UNIVERSE = archive_universe_sql_filter()



def _noop_supply_side(df: pd.DataFrame) -> pd.DataFrame:
    return df


def run_evaluation(max_items=200, use_supply_side=True):
    """Walk-forward evaluation, optionally including supply-side features."""
    import duckdb
    con = duckdb.connect()
    db = SessionLocal()

    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        # ── Load items ──────────────────────────────────────────────
        relation = prices_relation(con, ARCHIVE_DIR, columns=_PRICE_COLUMNS)
        rows = con.sql(f"""
            SELECT item_slug,
                   MIN(day) AS first_day,
                   MAX(day) AS last_day,
                   COUNT(*) AS row_count
            FROM {relation}
            WHERE {_UNIVERSE}
              AND item_slug IN (
                SELECT DISTINCT item_slug FROM {relation}
                WHERE source = 'STEAMCOMMUNITY'
              )
            GROUP BY item_slug
            HAVING row_count >= 90
            ORDER BY row_count DESC
            LIMIT {max_items}
        """).fetchall()

        items = rows
        logger.info(f"  {len(items)} items for evaluation")

        # ── Load all price data ─────────────────────────────────────
        all_rows = []
        for item_slug, _, _, _ in items:
            item_rows = con.sql(f"""
                SELECT item_slug AS item_id, day AS timestamp,
                       mean_price AS price, volume
                FROM {relation}
                WHERE item_slug = ? AND {_UNIVERSE}
                ORDER BY day
            """, params=[item_slug]).fetchall()
            item_df = pd.DataFrame(
                item_rows, columns=["item_id", "timestamp", "price", "volume"]
            )
            item_df["timestamp"] = pd.to_datetime(item_df["timestamp"])
            item_df["date"] = item_df["timestamp"].dt.date
            all_rows.append(item_df)

        all_prices = pd.concat(all_rows, ignore_index=True)

        # ── Monkey-patch to control supply-side features ─────────────
        if not use_supply_side:
            forecaster._add_supply_side_features = _noop_supply_side
            # Also prevent weapon-type cross-sectional features by
            # clearing the metadata cache so _fetch_supply_metadata
        else:
            # Restore original if needed (clean instance, so not needed)
            pass

        # ── Build features ──────────────────────────────────────────
        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)

        exclude = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date"}
        feature_cols = [c for c in df.columns if c not in exclude
                        and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]

        if len(feature_cols) > 2:
            corr = df[feature_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                if col in to_drop:
                    continue
                highly_corr = upper[col][upper[col] > 0.95].index
                to_drop.update(highly_corr)
            feature_cols = [c for c in feature_cols if c not in to_drop]

        logger.info(f"  Feature count: {len(feature_cols)}")

        results_by_horizon = {}
        for horizon in ItemForecaster.HORIZONS:
            logger.info(f"\n  Evaluating {horizon}d horizon...")

            tdf = forecaster.prepare_targets(df, horizon)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values(["item_id", "date"])

            if tdf.empty:
                logger.warning(f"    No valid targets for {horizon}d")
                continue

            dates = sorted(tdf["date"].unique())
            split_idx = len(dates) * 2 // 3

            directional_hits = 0
            directional_total = 0
            mae_total = 0.0
            mae_count = 0
            interval_hits = 0
            interval_total = 0
            per_fold = []
            records = []

            VAL_WINDOW_DAYS = 21
            step = 60
            for window_end in range(split_idx + 1, len(dates), step):
                train_dates = dates[:window_end]
                val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue

                train_df = ItemForecaster._purge_overlapping_train_rows(
                    tdf[tdf["date"].isin(train_dates)], val_dates[0], horizon)
                val_df = tdf[tdf["date"].isin(val_dates)]

                if len(val_df) < 50:
                    continue

                if len(train_df) > 200000:
                    train_df = train_df.sort_values("date").tail(200000)

                fc = [c for c in feature_cols if c in tdf.columns]
                if not fc:
                    continue

                X_train = train_df[fc].fillna(train_df[fc].median())
                y_train = train_df[f"target_return_{horizon}d"]
                X_val = val_df[fc].fillna(train_df[fc].median())
                y_val = val_df[f"target_return_{horizon}d"]

                models = {}
                for q in [0.1, 0.5, 0.9]:
                    params = {
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "boosting_type": "gbdt",
                        "num_leaves": 31,
                        "max_depth": 5,
                        "min_data_in_leaf": 15,
                        "min_gain_to_split": 0.1,
                        "learning_rate": 0.03,
                        "feature_fraction": 0.7,
                        "bagging_fraction": 0.7,
                        "bagging_freq": 5,
                        "lambda_l1": 0.5,
                        "lambda_l2": 0.5,
                        "verbosity": -1,
                        "random_state": 42,
                        "n_jobs": -1,
                    }
                    dtrain = lgb.Dataset(X_train.values, y_train.values)
                    dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain)
                    # Production's trainer and round table. The old call
                    # early-stopped on `dval` and scored `X_val` — the same
                    # rows. `dval` is ignored unless EARLY_STOPPING=1.
                    model = ItemForecaster._train_ensemble_member(
                        params, dtrain, dval,
                        num_boost_round=ItemForecaster._boost_rounds(
                            horizon, cv=True),
                        early_stopping=ItemForecaster._early_stopping_enabled(),
                    )
                    models[q] = model.predict(X_val.values)

                p10_ret = models[0.1]
                p50_ret = models[0.5]
                p90_ret = models[0.9]

                crossing_mask = (p10_ret > p50_ret) | (p50_ret > p90_ret)
                non_crossing = ~crossing_mask
                low_ret = np.minimum(p10_ret, p50_ret)
                high_ret = np.maximum(p50_ret, p90_ret)
                if non_crossing.any():
                    avg_hw = np.mean([
                        np.mean(p50_ret[non_crossing] - p10_ret[non_crossing]),
                        np.mean(p90_ret[non_crossing] - p50_ret[non_crossing]),
                    ])
                    if avg_hw > 0:
                        low_ret[crossing_mask] = p50_ret[crossing_mask] - avg_hw
                        high_ret[crossing_mask] = p50_ret[crossing_mask] + avg_hw
                low_ret = np.minimum(low_ret, p50_ret)
                high_ret = np.maximum(high_ret, p50_ret)

                current_prices = val_df["price"].values
                actual_returns = y_val.values

                fold_hits = 0
                fold_total = 0
                fold_mae = 0.0
                fold_int_hits = 0
                fold_int_total = 0
                for i in range(len(val_df)):
                    low_r, mid_r, high_r = low_ret[i], p50_ret[i], high_ret[i]
                    cp = float(current_prices[i])
                    ar = float(actual_returns[i])

                    actual_dir = "up" if ar > 0 else "down" if ar < 0 else "flat"
                    pred_dir = "up" if mid_r > 0 else "down" if mid_r < 0 else "flat"

                    if pred_dir == actual_dir:
                        directional_hits += 1
                        fold_hits += 1
                    directional_total += 1
                    fold_total += 1

                    abs_err = abs(cp * (1 + mid_r / 100) - cp * (1 + ar / 100))
                    mae_total += abs_err
                    fold_mae += abs_err
                    mae_count += 1

                    price_low = cp * (1 + low_r / 100)
                    price_high = cp * (1 + high_r / 100)
                    actual_future = cp * (1 + ar / 100)
                    fold_int_total += 1
                    interval_total += 1
                    if price_low <= actual_future <= price_high:
                        interval_hits += 1
                        fold_int_hits += 1

                # Paired records for the fold-clustered interval below.
                # Scored on non-flat actuals at >=$1, which is the population
                # production serves and is arm-independent, so the two runs
                # pair row for row. `window_end` rather than a running counter:
                # a counter drifts the moment one arm skips a fold the other
                # kept.
                _match = (np.sign(np.nan_to_num(actual_returns))
                          == np.sign(np.nan_to_num(p50_ret)))
                _scored = ((np.asarray(actual_returns) != 0)
                           & (np.asarray(current_prices, dtype=float) >= 1.0))
                records.extend(paired_records(
                    item_ids=val_df["item_id"].to_numpy(),
                    forecast_dates=val_df["date"].to_numpy(),
                    fold_id=window_end,
                    keep=_scored,
                    direction_correct=_match,
                ))

                per_fold.append({
                    "fold": len(per_fold) + 1,
                    "val_start": str(val_dates[0]),
                    "val_end": str(val_dates[-1]),
                    "dir_acc": round(fold_hits / fold_total * 100, 1) if fold_total > 0 else 0,
                    "mae": round(fold_mae / fold_total, 4) if fold_total > 0 else 0,
                    "int_cov": round(fold_int_hits / fold_int_total * 100, 1) if fold_int_total > 0 else 0,
                    "n": fold_total,
                })

            if directional_total > 0:
                dir_acc = directional_hits / directional_total * 100
                mae = mae_total / mae_count if mae_count > 0 else 0
                int_cov = interval_hits / interval_total * 100 if interval_total > 0 else 0

                fold_accs = [f["dir_acc"] for f in per_fold]
                baseline_2class = 50.0
                result = {
                    "directional_accuracy": round(dir_acc, 2),
                    "mae": round(mae, 4),
                    "interval_coverage": round(int_cov, 2),
                    "sample_count": directional_total,
                    "effective_baseline": baseline_2class,
                    "fold_count": len(per_fold),
                    # The pairing input for the fold-clustered interval that
                    # `main` computes across the two runs.
                    "records": records,
                    "fold_mean_dir_acc": round(np.mean(fold_accs), 1) if fold_accs else 0,
                    "fold_std_dir_acc": round(np.std(fold_accs), 1) if len(fold_accs) > 1 else 0,
                    "fold_min_dir_acc": round(min(fold_accs), 1) if fold_accs else 0,
                    "fold_max_dir_acc": round(max(fold_accs), 1) if fold_accs else 0,
                }
                result["improvement_over_baseline_pp"] = round(dir_acc - baseline_2class, 1)
                results_by_horizon[horizon] = result

                logger.info(f"  === {horizon}d: DirAcc={dir_acc:.1f}% "
                            f"({directional_total:,} samples, "
                            f"{result['improvement_over_baseline_pp']:.1f}pp above baseline)")

        return results_by_horizon

    finally:
        con.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=200,
                        help="Number of items to evaluate (default: 200)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: Supply-Side Features (rarity, weapon_type, weapon-type cross-sectional)")
    logger.info("=" * 70)

    # ── Run WITHOUT supply-side features ──────────────────────────────
    logger.info("\n>>> MODEL A (CONTROL): Without supply-side features <<<")
    results_without = run_evaluation(max_items=args.max_items, use_supply_side=False)

    # ── Run WITH supply-side features ─────────────────────────────────
    logger.info("\n>>> MODEL B (TREATMENT): With supply-side features <<<")
    results_with = run_evaluation(max_items=args.max_items, use_supply_side=True)

    # ── Compare ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("A/B TEST RESULTS — Supply-Side Features")
    print("=" * 70)
    print(f"{'Horizon':>8} | {'Control (w/o)':<20} | {'Treatment (w/)':<20} | {'Delta':>10}")
    print("-" * 8 + " | " + "-" * 20 + " | " + "-" * 20 + " | " + "-" * 10)

    total_delta = 0
    horizon_count = 0

    for h in ItemForecaster.HORIZONS:
        wo = results_without.get(h, {})
        w = results_with.get(h, {})

        wo_acc = wo.get("directional_accuracy", 0)
        w_acc = w.get("directional_accuracy", 0)
        wo_imp = wo.get("improvement_over_baseline_pp", 0)
        w_imp = w.get("improvement_over_baseline_pp", 0)
        wo_samples = wo.get("sample_count", 0)
        w_samples = w.get("sample_count", 0)
        wo_mae = wo.get("mae", 0)
        w_mae = w.get("mae", 0)
        wo_int = wo.get("interval_coverage", 0)
        w_int = w.get("interval_coverage", 0)

        delta = round(w_acc - wo_acc, 2)
        delta_imp = round(w_imp - wo_imp, 1)
        total_delta += delta
        horizon_count += 1

        label_wo = f"{wo_acc:.1f}% (baseline+{wo_imp:.1f}pp)"
        label_w = f"{w_acc:.1f}% (baseline+{w_imp:.1f}pp)"
        delta_str = f"{delta:+.2f}pp"
        if delta > 0:
            delta_str += " ✅"
        elif delta < 0:
            delta_str += " ❌"

        print(f"  {h:>2}d     | {label_wo:<20} | {label_w:<20} | {delta_str:>10}")

    print("-" * 8 + " | " + "-" * 20 + " | " + "-" * 20 + " | " + "-" * 10)

    avg_delta = total_delta / horizon_count if horizon_count > 0 else 0
    summary = "IMPROVEMENT" if avg_delta > 0 else "DEGRADATION" if avg_delta < 0 else "NO CHANGE"
    print(f"\n  Pooled delta (NOT a verdict): {summary} ({avg_delta:+.2f}pp avg)")

    # The verdict. Until 2026-08-08 the line above was it: a mean of per-horizon
    # deltas, with the sign alone deciding "IMPROVEMENT" or "DEGRADATION" and no
    # interval anywhere. The item-level MDE here is 2.21-3.69pp, so that rule
    # called a coin flip either way about half the time.
    print("\n  Paired, fold-clustered (treatment - control, >=$1 non-flat):")
    for h in ItemForecaster.HORIZONS:
        base = results_without.get(h, {}).get("records")
        arm = results_with.get(h, {}).get("records")
        if not base or not arm:
            print(f"    {h:>2}d:  unresolved — one arm produced no scored rows")
            continue
        contrasts = paired_arm_contrasts(
            {"control": base, "treatment": arm}, base="control")
        print(f"    {h:>2}d:  {format_paired(contrasts['treatment'])}")

    print("\n  Detail:")
    for h in ItemForecaster.HORIZONS:
        wo = results_without.get(h, {})
        w = results_with.get(h, {})
        wo_mae = wo.get("mae", 0)
        w_mae = w.get("mae", 0)
        wo_int = wo.get("interval_coverage", 0)
        w_int = w.get("interval_coverage", 0)
        wo_n = wo.get("sample_count", 0)
        w_n = w.get("sample_count", 0)
        print(f"    {h:>2}d:  Control: MAE=${wo_mae:.2f}  IntCov={wo_int:.1f}%  n={wo_n}")
        print(f"           Treat:  MAE=${w_mae:.2f}  IntCov={w_int:.1f}%  n={w_n}")

    print(f"\n  JSON: {json.dumps(without_records({
        'control': results_without,
        'treatment': results_with,
    }), indent=2)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
