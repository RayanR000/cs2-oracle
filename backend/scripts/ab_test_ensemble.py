#!/usr/bin/env python3
"""
A/B test: 3-member ensemble vs 6-member ensemble.

Compares directional accuracy, MAE, and interval coverage across both
ensemble sizes using walk-forward evaluation on historical Parquet data.
Stores results to prediction_accuracy table and prints a JSON report.

Usage:
    python scripts/ab_test_ensemble.py                          # full A/B test
    python scripts/ab_test_ensemble.py --max-items 200         # limit items
    python scripts/ab_test_ensemble.py --skip-db                # don't write to DB
    python scripts/ab_test_ensemble.py --horizons 7 30          # specific horizons

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
import time
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from database import SessionLocal, PredictionAccuracy
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import paired_records
from db.archive import prices_relation
from models.forecaster import ItemForecaster, archive_universe_sql_filter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_ensemble")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Production's item universe and price consensus, spelled into every archive
# read this harness makes. These three harnesses (`regime`, `ensemble`,
# `supply_side`) were the worst case before 2026-08-08: a raw
# `read_parquet('prices-*.parquet')` glob, which DuckDB narrows to the first
# file's schema so `source` is not even a column, and no universe rule at all —
# so the BUFF bid voted straight into their price series. `prices_relation`
# is the reader that projects `source` correctly whether or not the archive has
# been migrated. See `backend/AGENTS.md` and `db/archive.py`.
#
# Still outstanding here: these three read EVERY ask source, where the other
# ten filter to one, so a 2026 item-day reaches them several times over and
# `engineer_features` collapses the copies with a plain mean rather than
# production's outlier-voted median. `backend/AGENTS.md` puts the archive's
# duplication at 1.37x item-days. Narrowing the cohort would change what
# these harnesses measure and is not part of this change.
_PRICE_COLUMNS = ["item_slug", "day", "mean_price", "volume", "source"]
_UNIVERSE = archive_universe_sql_filter()


N_ENSEMBLES_3 = 3
ENSEMBLE_SEEDS_3 = [42, 73, 91]
ENSEMBLE_FEATURE_FRACTIONS_3 = [0.6, 0.65, 0.7]

N_ENSEMBLES_6 = 6
ENSEMBLE_SEEDS_6 = [42, 73, 91, 13, 57, 128]
ENSEMBLE_FEATURE_FRACTIONS_6 = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def _load_parquet_items(con, min_rows=90, backfilled_only=False):
    relation = prices_relation(con, ARCHIVE_DIR, columns=_PRICE_COLUMNS)
    conds = [_UNIVERSE]
    if backfilled_only:
        conds.append(f"""item_slug IN (
                SELECT DISTINCT item_slug FROM {relation}
                WHERE source = 'STEAMCOMMUNITY'
            )""")
    rows = con.sql(f"""
        SELECT item_slug,
               MIN(day) AS first_day,
               MAX(day) AS last_day,
               COUNT(*) AS row_count
        FROM {relation}
        WHERE {" AND ".join(conds)}
        GROUP BY item_slug
        HAVING row_count >= 90
        ORDER BY row_count DESC
    """).fetchall()
    return rows


def _load_all_prices(con, items):
    relation = prices_relation(con, ARCHIVE_DIR, columns=_PRICE_COLUMNS)
    all_rows = []
    for item_slug, first_day, last_day, row_count in items:
        rows = con.sql(f"""
            SELECT item_slug AS item_id, CAST(day AS DATE) AS timestamp,
                   mean_price AS price, volume
            FROM {relation}
            WHERE item_slug = ? AND {_UNIVERSE}
            ORDER BY day
        """, params=[item_slug]).fetchall()
        item_df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
        item_df["timestamp"] = pd.to_datetime(item_df["timestamp"])
        item_df["date"] = item_df["timestamp"].dt.date
        all_rows.append(item_df)

    return pd.concat(all_rows, ignore_index=True)


def _compute_metrics(y_true, y_pred_low, y_pred_mid, y_pred_high, current_prices):
    errors = []
    dir_hits = 0
    dir_total = 0
    int_hits = 0
    int_total = 0

    for i in range(len(y_true)):
        actual = float(y_true[i])
        mid = float(y_pred_mid[i])
        low = float(y_pred_low[i])
        high = float(y_pred_high[i])
        current = float(current_prices[i])

        abs_err = abs(mid - actual)
        pct_err = abs(abs_err / actual) * 100 if actual > 0 else 0
        errors.append({"abs": abs_err, "pct": pct_err, "sq": (mid - actual) ** 2})

        actual_dir = "up" if actual > current else "down" if actual < current else "flat"
        pred_dir = "up" if mid > current else "down" if mid < current else "flat"
        if pred_dir == actual_dir:
            dir_hits += 1
        dir_total += 1

        if low is not None and high is not None:
            int_total += 1
            if low <= actual <= high:
                int_hits += 1

    n = len(errors)
    if n == 0:
        return None
    mae = sum(e["abs"] for e in errors) / n
    rmse = math.sqrt(sum(e["sq"] for e in errors) / n)
    mape = sum(e["pct"] for e in errors) / n
    dir_acc = (dir_hits / dir_total * 100) if dir_total > 0 else 0
    int_cov = (int_hits / int_total * 100) if int_total > 0 else 0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "directional_accuracy": round(dir_acc, 2),
        "interval_coverage": round(int_cov, 2),
        "sample_count": n,
    }


def _train_ensemble(params, dtrain, dval, n_ensembles, seeds, feature_fractions,
                    horizon):
    """Train an ensemble of LightGBM quantile models and return averaged predictions on val.

    `horizon` is only used to look up production's round count; it carries no
    other meaning here. It is a required argument rather than a default because
    the old call early-stopped on `dval` and then predicted `dval.data` — the
    same rows — so a caller that forgets to say which horizon it is measuring
    should fail loudly rather than silently pick one.
    """
    all_preds = []
    for ei in range(n_ensembles):
        p = params.copy()
        p["random_state"] = seeds[ei]
        p["feature_fraction"] = feature_fractions[ei]
        model = ItemForecaster._train_ensemble_member(
            p, dtrain, dval,
            num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
            early_stopping=ItemForecaster._early_stopping_enabled(),
        )
        all_preds.append(model.predict(dval.data))
    return np.mean(all_preds, axis=0)


def run_ab_test(max_items=500, horizons=None, skip_db=False):
    """Run walk-forward A/B test comparing 3-member vs 6-member ensembles."""
    logger.info("=" * 60)
    logger.info("A/B TEST: 3-Member Ensemble vs 6-Member Ensemble")
    logger.info("=" * 60)

    if not ARCHIVE_DIR.exists():
        logger.error(f"Parquet archive not found at {ARCHIVE_DIR}")
        return {"status": "error", "message": "Archive not found"}

    import duckdb
    con = duckdb.connect()

    try:
        db = SessionLocal()
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        items = _load_parquet_items(con, backfilled_only=True)
        items = items[:max_items]
        logger.info(f"  {len(items)} items for A/B evaluation")

        all_prices = _load_all_prices(con, items)

        results_by_horizon = {}
        total_train_start = time.time()
        total_ens3_elapsed = 0.0
        total_ens6_elapsed = 0.0

        for horizon in (horizons or ItemForecaster.HORIZONS):
            logger.info(f"\n  === Evaluating {horizon}d horizon ===")

            df = forecaster.engineer_features(all_prices, events_df)
            df = forecaster._add_cross_sectional_features(df)

            tdf = forecaster.prepare_targets(df, horizon)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values(["item_id", "date"])

            if tdf.empty:
                logger.warning(f"    No valid targets for {horizon}d")
                continue

            dates = sorted(tdf["date"].unique())
            split_idx = len(dates) * 2 // 3
            VAL_WINDOW_DAYS = 21
            step = 60

            ens3_fold_results = []
            ens6_fold_results = []
            ens3_records = []
            ens6_records = []

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

                feature_cols = [c for c in forecaster.feature_cols if c in tdf.columns]
                if not feature_cols:
                    exclude = {"item_id", "date", "timestamp", "price", "volume",
                               "name", "release_date"}
                    exclude |= {f"target_{h}d" for h in forecaster.HORIZONS}
                    exclude |= {f"target_return_{h}d" for h in forecaster.HORIZONS}
                    feature_cols = [c for c in tdf.columns if c not in exclude
                                    and tdf[c].dtype in (np.float64, np.float32, np.int64, int, float)]

                if len(feature_cols) > 2:
                    corr = train_df[feature_cols].corr().abs()
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    to_drop = set()
                    for col in upper.columns:
                        if col in to_drop:
                            continue
                        highly_corr = upper[col][upper[col] > 0.95].index
                        to_drop.update(highly_corr)
                    feature_cols = [c for c in feature_cols if c not in to_drop]

                X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
                y_train = train_df[f"target_return_{horizon}d"]
                X_val = val_df[feature_cols].fillna(train_df[feature_cols].median())
                y_val = val_df[f"target_return_{horizon}d"]

                # Shared base params
                base_params = {
                    "objective": "quantile",
                    "metric": "quantile",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "max_depth": 5,
                    "min_data_in_leaf": 15,
                    "min_gain_to_split": 0.1,
                    "learning_rate": 0.03,
                    "bagging_fraction": 0.7,
                    "bagging_freq": 5,
                    "lambda_l1": 0.5,
                    "lambda_l2": 0.5,
                    "verbosity": -1,
                    "n_jobs": -1,
                }

                # Train both ensemble sizes for this fold
                ens3_preds = {}
                ens6_preds = {}
                fold_ens3_time = 0.0
                fold_ens6_time = 0.0

                for q in [0.1, 0.5, 0.9]:
                    params = dict(base_params)
                    params["alpha"] = q

                    dtrain = lgb.Dataset(X_train.values, y_train.values)
                    dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain)

                    # 3-member ensemble
                    t0 = time.time()
                    ens3_preds[q] = _train_ensemble(
                        params, dtrain, dval,
                        N_ENSEMBLES_3, ENSEMBLE_SEEDS_3,
                        ENSEMBLE_FEATURE_FRACTIONS_3, horizon
                    )
                    fold_ens3_time += time.time() - t0

                    # 6-member ensemble
                    t0 = time.time()
                    ens6_preds[q] = _train_ensemble(
                        params, dtrain, dval,
                        N_ENSEMBLES_6, ENSEMBLE_SEEDS_6,
                        ENSEMBLE_FEATURE_FRACTIONS_6, horizon
                    )
                    fold_ens6_time += time.time() - t0

                total_ens3_elapsed += fold_ens3_time
                total_ens6_elapsed += fold_ens6_time

                if len(ens3_preds) == 3 and len(ens6_preds) == 3:
                    current_prices = val_df["price"].values
                    actual_returns = y_val.values

                    e3_low, e3_high = ItemForecaster._fix_quantile_crossing(
                        ens3_preds[0.1], ens3_preds[0.5], ens3_preds[0.9])
                    e6_low, e6_high = ItemForecaster._fix_quantile_crossing(
                        ens6_preds[0.1], ens6_preds[0.5], ens6_preds[0.9])

                    actual_prices = current_prices * (1 + actual_returns / 100)
                    e3_mid_prices = current_prices * (1 + ens3_preds[0.5] / 100)
                    e3_low_prices = current_prices * (1 + e3_low / 100)
                    e3_high_prices = current_prices * (1 + e3_high / 100)
                    e6_mid_prices = current_prices * (1 + ens6_preds[0.5] / 100)
                    e6_low_prices = current_prices * (1 + e6_low / 100)
                    e6_high_prices = current_prices * (1 + e6_high / 100)

                    e3_metrics = _compute_metrics(actual_prices, e3_low_prices, e3_mid_prices, e3_high_prices, current_prices)
                    e6_metrics = _compute_metrics(actual_prices, e6_low_prices, e6_mid_prices, e6_high_prices, current_prices)

                    if e3_metrics:
                        ens3_fold_results.append(e3_metrics)
                    if e6_metrics:
                        ens6_fold_results.append(e6_metrics)

                    # Paired records for the fold-clustered interval below.
                    # Scored on non-flat actuals at >=$1, which is the
                    # population production serves and does not depend on the
                    # arm, so the two ensembles pair row for row. `window_end`
                    # rather than a running counter: a counter drifts the
                    # moment one arm skips a fold the other kept.
                    _scored = ((np.asarray(actual_returns) != 0)
                               & (np.asarray(current_prices, dtype=float) >= 1.0))
                    _sign = np.sign(np.nan_to_num(actual_returns))
                    for bucket, preds in ((ens3_records, ens3_preds[0.5]),
                                          (ens6_records, ens6_preds[0.5])):
                        bucket.extend(paired_records(
                            item_ids=val_df["item_id"].to_numpy(),
                            forecast_dates=val_df["date"].to_numpy(),
                            fold_id=window_end,
                            keep=_scored,
                            direction_correct=(
                                _sign == np.sign(np.nan_to_num(preds))),
                        ))

            # Aggregate across folds
            if ens3_fold_results and ens6_fold_results:
                e3_agg = _aggregate_folds(ens3_fold_results)
                e6_agg = _aggregate_folds(ens6_fold_results)
                results_by_horizon[horizon] = {
                    "ens3": e3_agg,
                    "ens6": e6_agg,
                    "delta": {
                        "directional_accuracy_pp": round(e3_agg["directional_accuracy"] - e6_agg["directional_accuracy"], 2),
                        "mae_delta": round(e3_agg["mae"] - e6_agg["mae"], 4),
                        "mape_delta": round(e3_agg["mape"] - e6_agg["mape"], 2),
                        "interval_coverage_pp": round(e3_agg["interval_coverage"] - e6_agg["interval_coverage"], 2),
                    },
                    # Kept for continuity; it is not the verdict and never
                    # was one. A bare `a > b` fires half the time on two arms
                    # that differ only by ensemble size against an item-level
                    # MDE of 2.21-3.69pp.
                    "ens3_wins": e3_agg["directional_accuracy"] > e6_agg["directional_accuracy"],
                }
                if ens3_records and ens6_records:
                    contrasts = paired_arm_contrasts(
                        {"ens6": ens6_records, "ens3": ens3_records},
                        base="ens6")
                    results_by_horizon[horizon]["paired_ens3_vs_ens6"] = (
                        contrasts["ens3"])

                logger.info(f"\n  === {horizon}d A/B Results ===")
                logger.info(f"  Ens3:       DirAcc={e3_agg['directional_accuracy']:.1f}% "
                            f"MAE=${e3_agg['mae']:.2f} MAPE={e3_agg['mape']:.1f}% "
                            f"IntCov={e3_agg['interval_coverage']:.1f}%")
                logger.info(f"  Ens6:       DirAcc={e6_agg['directional_accuracy']:.1f}% "
                            f"MAE=${e6_agg['mae']:.2f} MAPE={e6_agg['mape']:.1f}% "
                            f"IntCov={e6_agg['interval_coverage']:.1f}%")
                delta = results_by_horizon[horizon]["delta"]
                logger.info(f"  Delta:      DirAcc={delta['directional_accuracy_pp']:+.2f}pp "
                            f"MAE=${delta['mae_delta']:+.4f} "
                            f"IntCov={delta['interval_coverage_pp']:+.2f}pp")
                paired = results_by_horizon[horizon].get("paired_ens3_vs_ens6")
                if paired:
                    logger.info(f"  Paired (ens3 - ens6, >=$1 non-flat): "
                                f"{format_paired(paired)}")
                else:
                    logger.info("  Paired: unresolved — no scored rows")

        total_elapsed = time.time() - total_train_start
        logger.info(f"\n{'='*60}")
        logger.info(f"A/B TEST COMPLETE in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        logger.info(f"  Ens3 training: {total_ens3_elapsed:.0f}s ({total_ens3_elapsed/60:.1f}min)")
        logger.info(f"  Ens6 training: {total_ens6_elapsed:.0f}s ({total_ens6_elapsed/60:.1f}min)")
        logger.info(f"  Speedup: {total_ens6_elapsed/total_ens3_elapsed:.2f}x")

        report = {
            "test_date": str(date.today()),
            "total_items": len(items),
            "total_elapsed_seconds": round(total_elapsed, 1),
            "total_elapsed_minutes": round(total_elapsed / 60, 1),
            "ens3_training_seconds": round(total_ens3_elapsed, 1),
            "ens6_training_seconds": round(total_ens6_elapsed, 1),
            "speedup_x": round(total_ens6_elapsed / total_ens3_elapsed, 2) if total_ens3_elapsed > 0 else 0,
            "horizons": {},
        }

        for horizon, h_results in results_by_horizon.items():
            report["horizons"][str(horizon)] = h_results

        if not skip_db:
            _store_ab_results(db, report)

        logger.info(f"\n{'='*60}")
        logger.info("A/B TEST REPORT")
        logger.info("=" * 60)
        for h, hr in sorted(results_by_horizon.items()):
            e3, e6 = hr["ens3"], hr["ens6"]
            d = hr["delta"]
            logger.info(f"\n  {h}d:")
            logger.info(f"    Ens3:  DirAcc={e3['directional_accuracy']:.1f}%  MAE=${e3['mae']:.2f}  MAPE={e3['mape']:.1f}%  IntCov={e3['interval_coverage']:.1f}%")
            logger.info(f"    Ens6:  DirAcc={e6['directional_accuracy']:.1f}%  MAE=${e6['mae']:.2f}  MAPE={e6['mape']:.1f}%  IntCov={e6['interval_coverage']:.1f}%")
            logger.info(f"    Delta: DirAcc={d['directional_accuracy_pp']:+.2f}pp  MAE=${d['mae_delta']:+.4f}  IntCov={d['interval_coverage_pp']:+.2f}pp")
            # The verdict, and the last line an operator reads. `ens3_wins` is
            # a bare `a > b` and is reported beside it as context only.
            paired = hr.get("paired_ens3_vs_ens6")
            logger.info(f"    Verdict: "
                        f"{format_paired(paired) if paired else 'unresolved'}")
            logger.info(f"    (ens3_wins={hr['ens3_wins']} — a bare a>b, "
                        f"not a test)")

        con.close()
        db.close()

        return report

    except Exception as e:
        logger.error(f"A/B test failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def _aggregate_folds(fold_results):
    n = sum(f["sample_count"] for f in fold_results)
    if n == 0:
        return None
    weighted = {
        "mae": sum(f["mae"] * f["sample_count"] for f in fold_results) / n,
        "rmse": sum(f["rmse"] * f["sample_count"] for f in fold_results) / n,
        "mape": sum(f["mape"] * f["sample_count"] for f in fold_results) / n,
        "directional_accuracy": sum(f["directional_accuracy"] * f["sample_count"] for f in fold_results) / n,
        "interval_coverage": sum(f["interval_coverage"] * f["sample_count"] for f in fold_results) / n,
        "sample_count": n,
        "fold_count": len(fold_results),
    }
    return weighted


def _store_ab_results(db, report):
    """Store A/B test results to prediction_accuracy table."""
    today = date.today()
    for horizon_str, h_results in report.get("horizons", {}).items():
        horizon = int(horizon_str)
        for version_key, version_label in [("ens3", "lgbm-v3-ens3"), ("ens6", "lgbm-v3-ens6")]:
            metrics = h_results.get(version_key)
            if not metrics:
                continue
            row = {
                "prediction_type": "ab_test_ensemble",
                "evaluation_date": today,
                "horizon_days": horizon,
                "model_version": version_label,
                "evaluation_window_days": None,
                "sample_count": metrics["sample_count"],
                "metrics": {
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "directional_accuracy": metrics["directional_accuracy"],
                    "interval_coverage": metrics["interval_coverage"],
                    "fold_count": metrics.get("fold_count", 0),
                },
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
            _upsert_accuracy(db, [row])

        delta = h_results["delta"]
        drow = {
            "prediction_type": "ab_test_ensemble_delta",
            "evaluation_date": today,
            "horizon_days": horizon,
            "model_version": "ens3_vs_ens6",
            "evaluation_window_days": None,
            "sample_count": h_results["ens3"]["sample_count"],
            "metrics": {
                "directional_accuracy_delta_pp": delta["directional_accuracy_pp"],
                "mae_delta": delta["mae_delta"],
                "mape_delta": delta["mape_delta"],
                "interval_coverage_delta_pp": delta["interval_coverage_pp"],
                "ens3_wins": h_results["ens3_wins"],
                # The paired, fold-clustered verdict — the only one of these
                # two that is a test. Stored flat: `db/parquet.py` serialises
                # nested values, but a reader should not have to parse JSON to
                # find out whether the effect was real.
                "paired_verdict": (h_results.get("paired_ens3_vs_ens6") or {})
                    .get("verdict", "unresolved"),
                "paired_mean_diff_pp": (
                    h_results.get("paired_ens3_vs_ens6") or {}).get("mean_diff"),
                "paired_ci_lower_pp": (
                    h_results.get("paired_ens3_vs_ens6") or {}).get("ci_lower"),
                "paired_ci_upper_pp": (
                    h_results.get("paired_ens3_vs_ens6") or {}).get("ci_upper"),
                "paired_n_clusters": (
                    h_results.get("paired_ens3_vs_ens6") or {}).get("n_clusters"),
                "ens3_dir_acc": h_results["ens3"]["directional_accuracy"],
                "ens6_dir_acc": h_results["ens6"]["directional_accuracy"],
                "ens3_mae": h_results["ens3"]["mae"],
                "ens6_mae": h_results["ens6"]["mae"],
                "ens3_mape": h_results["ens3"]["mape"],
                "ens6_mape": h_results["ens6"]["mape"],
                "ens3_int_cov": h_results["ens3"]["interval_coverage"],
                "ens6_int_cov": h_results["ens6"]["interval_coverage"],
                "ens3_sample_count": h_results["ens3"]["sample_count"],
                "ens6_sample_count": h_results["ens6"]["sample_count"],
                "ens3_fold_count": h_results["ens3"].get("fold_count", 0),
                "ens6_fold_count": h_results["ens6"].get("fold_count", 0),
            },
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        _upsert_accuracy(db, [drow])

    return report


def _upsert_accuracy(db, rows):
    from database import PredictionAccuracy
    for row in rows:
        filters = {
            "prediction_type": row["prediction_type"],
            "evaluation_date": row["evaluation_date"],
            "horizon_days": row.get("horizon_days"),
            "model_version": row.get("model_version"),
        }
        existing = db.query(PredictionAccuracy).filter_by(**filters).first()
        if existing:
            existing.sample_count = row["sample_count"]
            existing.metrics = row["metrics"]
            existing.evaluation_window_days = row.get("evaluation_window_days")
            existing.created_at = row["created_at"]
        else:
            db.add(PredictionAccuracy(**row))
    db.commit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="A/B test 3-member vs 6-member ensembles")
    parser.add_argument("--max-items", type=int, default=500,
                        help="Number of items to evaluate (default: 500)")
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                        help="Horizons to test (default: all)")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip writing results to database")
    args = parser.parse_args()

    report = run_ab_test(
        max_items=args.max_items,
        horizons=args.horizons,
        skip_db=args.skip_db,
    )
    print(f"\nRESULT: {json.dumps(report, indent=2, default=str)}")
    return 0 if report.get("status") != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
