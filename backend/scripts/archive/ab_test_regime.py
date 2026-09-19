#!/usr/bin/env python3
"""
A/B test: regime-switching models vs global-only.

Compares directional accuracy, MAE, and interval coverage across both
model versions using walk-forward evaluation on historical Parquet data.
Stores results to prediction_accuracy table and prints a JSON report.

Usage:
    python scripts/ab_test_regime.py                          # full A/B test
    python scripts/ab_test_regime.py --max-items 200         # limit items
    python scripts/ab_test_regime.py --skip-db                # don't write to DB
    python scripts/ab_test_regime.py --horizons 7 30          # specific horizons

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

import json
import logging
import math
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import paired_records
from database import PredictionAccuracy, SessionLocal
from db.archive import prices_relation
from models.forecaster import ItemForecaster, archive_universe_sql_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ab_test_regime")

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


def _load_parquet_items(con, min_rows=90, backfilled_only=False):
    relation = prices_relation(con, ARCHIVE_DIR, columns=_PRICE_COLUMNS)
    conds = [_UNIVERSE]
    if backfilled_only:
        # `source = 'STEAMCOMMUNITY'` matched 0 rows post archive-rebuild
        # (values are NULL pre-2026 / aggregator_* in 2026), so this subquery
        # made the whole query return 0 items. `source IS NULL` is the
        # backfilled (pre-2026) series this filter means. See 2026-08-13 repin.
        conds.append(f"""item_slug IN (
                SELECT DISTINCT item_slug FROM {relation}
                WHERE source IS NULL
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
        ORDER BY row_count DESC, item_slug
    """).fetchall()
    if not rows:
        raise RuntimeError(
            "regime universe query selected 0 items — the source pin matched no rows (see 2026-08-13 harness repin)."
        )
    return rows


def _load_all_prices(con, items):
    relation = prices_relation(con, ARCHIVE_DIR, columns=_PRICE_COLUMNS)
    all_rows = []
    for item_slug, _first_day, _last_day, _row_count in items:
        rows = con.sql(
            f"""
            SELECT item_slug AS item_id, CAST(day AS DATE) AS timestamp,
                   mean_price AS price, volume
            FROM {relation}
            WHERE item_slug = ? AND {_UNIVERSE}
            ORDER BY day
        """,
            params=[item_slug],
        ).fetchall()
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


def run_ab_test(max_items=500, horizons=None, skip_db=False):
    """Run walk-forward A/B test comparing regime vs global-only models."""
    logger.info("=" * 60)
    logger.info("A/B TEST: Regime-Switching vs Global-Only")
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

        all_rows = []
        for item_slug, _first_day, _last_day, _row_count in items:
            rows = con.sql(
                f"""
                SELECT item_slug AS item_id, CAST(day AS DATE) AS timestamp,
                       mean_price AS price, volume
                FROM {prices_relation(con, ARCHIVE_DIR, columns=_PRICE_COLUMNS)}
                WHERE item_slug = ? AND {_UNIVERSE}
                ORDER BY day
            """,
                params=[item_slug],
            ).fetchall()
            item_df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
            item_df["timestamp"] = pd.to_datetime(item_df["timestamp"])
            item_df["date"] = item_df["timestamp"].dt.date
            all_rows.append(item_df)

        all_prices = pd.concat(all_rows, ignore_index=True)

        results_by_horizon = {}
        total_train_start = time.time()

        for horizon in horizons or ItemForecaster.HORIZONS:
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

            regime_fold_results = []
            global_fold_results = []
            regime_records = []
            global_records = []

            for window_end in range(split_idx + 1, len(dates), step):
                train_dates = dates[:window_end]
                val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue

                train_df = ItemForecaster._purge_overlapping_train_rows(
                    tdf[tdf["date"].isin(train_dates)], val_dates[0], horizon
                )
                val_df = tdf[tdf["date"].isin(val_dates)]

                if len(val_df) < 50:
                    continue

                if len(train_df) > 200000:
                    train_df = train_df.sort_values("date").tail(200000)

                feature_cols = [c for c in forecaster.feature_cols if c in tdf.columns]
                if not feature_cols:
                    exclude = {"item_id", "date", "timestamp", "price", "volume", "name", "release_date"}
                    exclude |= {f"target_{h}d" for h in forecaster.HORIZONS}
                    exclude |= {f"target_return_{h}d" for h in forecaster.HORIZONS}
                    feature_cols = [
                        c
                        for c in tdf.columns
                        if c not in exclude and tdf[c].dtype in (np.float64, np.float32, np.int64, int, float)
                    ]

                # Measure the model production serves: shelve + allowlist. This
                # harness's arms differ only in training config, not features, so
                # the fixed production allowlist applies to both. Without it the
                # harness measured a 138+-column model production does not serve.
                # See 2026-08-13 repin.
                feature_cols = [c for c in feature_cols if c not in ItemForecaster.SHELVED_FEATURES]
                feature_cols = ItemForecaster._apply_feature_allowlist(
                    feature_cols, ItemForecaster.FEATURE_GROUP_ALLOWLIST
                )

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

                # Train global models
                global_models = {}
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
                        params,
                        dtrain,
                        dval,
                        num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                        early_stopping=ItemForecaster._early_stopping_enabled(),
                    )
                    global_models[q] = model.predict(X_val.values)

                # Train regime models
                train_df["_regime"] = forecaster._assign_regime_labels(train_df)
                val_df["_regime"] = forecaster._assign_regime_labels(val_df)
                current_regime = forecaster._detect_current_regime(val_df)

                regime_models = {}
                for regime in forecaster.REGIMES:
                    r_train = train_df[train_df["_regime"] == regime]
                    r_val = val_df[val_df["_regime"] == regime]
                    if len(r_train) < 500 or len(r_val) < 50:
                        continue
                    r_X_train = r_train[feature_cols].fillna(train_df[feature_cols].median())
                    r_y_train = r_train[f"target_return_{horizon}d"]
                    r_X_val = r_val[feature_cols].fillna(train_df[feature_cols].median())
                    r_y_val = r_val[f"target_return_{horizon}d"]
                    r_dtrain = lgb.Dataset(r_X_train.values, r_y_train.values)
                    r_dval = lgb.Dataset(r_X_val.values, r_y_val.values, reference=r_dtrain)
                    r_models = {}
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
                        r_model = ItemForecaster._train_ensemble_member(
                            params,
                            r_dtrain,
                            r_dval,
                            num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                            early_stopping=ItemForecaster._early_stopping_enabled(),
                        )
                        r_models[q] = r_model.predict(r_X_val.values)

                # Predict with regime models (prefer regime, fallback to global)
                regime_preds = {}
                for q in [0.1, 0.5, 0.9]:
                    if (
                        current_regime in forecaster.REGIMES
                        and current_regime in regime_models
                        and q in regime_models[current_regime]
                    ):
                        regime_preds[q] = regime_models[current_regime][q]
                    elif q in global_models:
                        regime_preds[q] = global_models[q]

                # Predict with global-only models
                global_preds = {q: global_models[q] for q in [0.1, 0.5, 0.9] if q in global_models}

                if len(regime_preds) == 3 and len(global_preds) == 3:
                    current_prices = val_df["price"].values
                    actual_returns = y_val.values

                    # Fix quantile crossing for both
                    r_low, r_high = ItemForecaster._fix_quantile_crossing(
                        regime_preds[0.1], regime_preds[0.5], regime_preds[0.9]
                    )
                    g_low, g_high = ItemForecaster._fix_quantile_crossing(
                        global_preds[0.1], global_preds[0.5], global_preds[0.9]
                    )

                    # Convert returns to prices
                    actual_prices = current_prices * (1 + actual_returns / 100)
                    r_mid_prices = current_prices * (1 + regime_preds[0.5] / 100)
                    r_low_prices = current_prices * (1 + r_low / 100)
                    r_high_prices = current_prices * (1 + r_high / 100)
                    g_mid_prices = current_prices * (1 + global_preds[0.5] / 100)
                    g_low_prices = current_prices * (1 + g_low / 100)
                    g_high_prices = current_prices * (1 + g_high / 100)

                    r_metrics = _compute_metrics(
                        actual_prices, r_low_prices, r_mid_prices, r_high_prices, current_prices
                    )
                    g_metrics = _compute_metrics(
                        actual_prices, g_low_prices, g_mid_prices, g_high_prices, current_prices
                    )

                    if r_metrics:
                        regime_fold_results.append(r_metrics)
                    if g_metrics:
                        global_fold_results.append(g_metrics)

                    # Paired records for the fold-clustered interval below.
                    # Scored on non-flat actuals at >=$1, which is the
                    # population production serves and does not depend on the
                    # arm, so the two pair row for row. `window_end` rather
                    # than a running counter: a counter drifts the moment one
                    # arm skips a fold the other kept.
                    _scored = (np.asarray(actual_returns) != 0) & (np.asarray(current_prices, dtype=float) >= 1.0)
                    _sign = np.sign(np.nan_to_num(actual_returns))
                    for bucket, preds in ((regime_records, regime_preds[0.5]), (global_records, global_preds[0.5])):
                        bucket.extend(
                            paired_records(
                                item_ids=val_df["item_id"].to_numpy(),
                                forecast_dates=val_df["date"].to_numpy(),
                                fold_id=window_end,
                                keep=_scored,
                                direction_correct=(_sign == np.sign(np.nan_to_num(preds))),
                            )
                        )

            # Aggregate across folds
            if regime_fold_results and global_fold_results:
                r_agg = _aggregate_folds(regime_fold_results)
                g_agg = _aggregate_folds(global_fold_results)
                results_by_horizon[horizon] = {
                    "regime": r_agg,
                    "global_only": g_agg,
                    "delta": {
                        "directional_accuracy_pp": round(
                            r_agg["directional_accuracy"] - g_agg["directional_accuracy"], 2
                        ),
                        "mae_delta": round(r_agg["mae"] - g_agg["mae"], 4),
                        "mape_delta": round(r_agg["mape"] - g_agg["mape"], 2),
                        "interval_coverage_pp": round(r_agg["interval_coverage"] - g_agg["interval_coverage"], 2),
                    },
                    # Kept for continuity; it is not the verdict and never
                    # was one. A bare `a > b` fires half the time against an
                    # item-level MDE of 2.21-3.69pp.
                    "regime_wins": r_agg["directional_accuracy"] > g_agg["directional_accuracy"],
                }
                if regime_records and global_records:
                    contrasts = paired_arm_contrasts(
                        {"global_only": global_records, "regime": regime_records}, base="global_only"
                    )
                    results_by_horizon[horizon]["paired_regime_vs_global"] = contrasts["regime"]

                logger.info(f"\n  === {horizon}d A/B Results ===")
                logger.info(
                    f"  Regime:     DirAcc={r_agg['directional_accuracy']:.1f}% "
                    f"MAE=${r_agg['mae']:.2f} MAPE={r_agg['mape']:.1f}% "
                    f"IntCov={r_agg['interval_coverage']:.1f}%"
                )
                logger.info(
                    f"  Global:     DirAcc={g_agg['directional_accuracy']:.1f}% "
                    f"MAE=${g_agg['mae']:.2f} MAPE={g_agg['mape']:.1f}% "
                    f"IntCov={g_agg['interval_coverage']:.1f}%"
                )
                delta = results_by_horizon[horizon]["delta"]
                logger.info(
                    f"  Delta:      DirAcc={delta['directional_accuracy_pp']:+.2f}pp "
                    f"MAE=${delta['mae_delta']:+.4f} "
                    f"IntCov={delta['interval_coverage_pp']:+.2f}pp"
                )
                paired = results_by_horizon[horizon].get("paired_regime_vs_global")
                if paired:
                    logger.info(f"  Paired (regime - global, >=$1 non-flat): {format_paired(paired)}")
                else:
                    logger.info("  Paired: unresolved — no scored rows")

        total_elapsed = time.time() - total_train_start
        logger.info(f"\n{'=' * 60}")
        logger.info(f"A/B TEST COMPLETE in {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")

        # Build report
        report = {
            "test_date": str(date.today()),
            "total_items": len(items),
            "total_elapsed_seconds": round(total_elapsed, 1),
            "total_elapsed_minutes": round(total_elapsed / 60, 1),
            "horizons": {},
        }

        for horizon, h_results in results_by_horizon.items():
            report["horizons"][str(horizon)] = h_results

        # Store to DB
        if not skip_db:
            _store_ab_results(db, report)

        logger.info(f"\n{'=' * 60}")
        logger.info("A/B TEST REPORT")
        logger.info("=" * 60)
        for h, hr in sorted(results_by_horizon.items()):
            r, g = hr["regime"], hr["global_only"]
            d = hr["delta"]
            logger.info(f"\n  {h}d:")
            logger.info(
                f"    Regime:  DirAcc={r['directional_accuracy']:.1f}%  MAE=${r['mae']:.2f}  MAPE={r['mape']:.1f}%  IntCov={r['interval_coverage']:.1f}%"
            )
            logger.info(
                f"    Global:  DirAcc={g['directional_accuracy']:.1f}%  MAE=${g['mae']:.2f}  MAPE={g['mape']:.1f}%  IntCov={g['interval_coverage']:.1f}%"
            )
            logger.info(
                f"    Delta:   DirAcc={d['directional_accuracy_pp']:+.2f}pp  MAE=${d['mae_delta']:+.4f}  IntCov={d['interval_coverage_pp']:+.2f}pp"
            )
            # The verdict, and the last line an operator reads. `regime_wins`
            # is a bare `a > b` and is reported beside it as context only.
            paired = hr.get("paired_regime_vs_global")
            logger.info(f"    Verdict: {format_paired(paired) if paired else 'unresolved'}")
            logger.info(f"    (regime_wins={hr['regime_wins']} — a bare a>b, not a test)")

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
        for version_key, version_label in [("regime", "lgbm-v3-regime"), ("global_only", "lgbm-v3-global-only")]:
            metrics = h_results.get(version_key)
            if not metrics:
                continue
            row = {
                "prediction_type": "ab_test_regime",
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
                "created_at": datetime.now(UTC).replace(tzinfo=None),
            }
            _upsert_accuracy(db, [row])

        delta = h_results["delta"]
        drow = {
            "prediction_type": "ab_test_regime_delta",
            "evaluation_date": today,
            "horizon_days": horizon,
            "model_version": "regime_vs_global",
            "evaluation_window_days": None,
            "sample_count": h_results["regime"]["sample_count"],
            "metrics": {
                "directional_accuracy_delta_pp": delta["directional_accuracy_pp"],
                "mae_delta": delta["mae_delta"],
                "mape_delta": delta["mape_delta"],
                "interval_coverage_delta_pp": delta["interval_coverage_pp"],
                "regime_wins": h_results["regime_wins"],
                # The paired, fold-clustered verdict — the only one of these
                # two that is a test. Stored flat: `db/parquet.py` serialises
                # nested values, but a reader should not have to parse JSON to
                # find out whether the effect was real.
                "paired_verdict": (h_results.get("paired_regime_vs_global") or {}).get("verdict", "unresolved"),
                "paired_mean_diff_pp": (h_results.get("paired_regime_vs_global") or {}).get("mean_diff"),
                "paired_ci_lower_pp": (h_results.get("paired_regime_vs_global") or {}).get("ci_lower"),
                "paired_ci_upper_pp": (h_results.get("paired_regime_vs_global") or {}).get("ci_upper"),
                "paired_n_clusters": (h_results.get("paired_regime_vs_global") or {}).get("n_clusters"),
                "regime_dir_acc": h_results["regime"]["directional_accuracy"],
                "global_dir_acc": h_results["global_only"]["directional_accuracy"],
                "regime_mae": h_results["regime"]["mae"],
                "global_mae": h_results["global_only"]["mae"],
                "regime_mape": h_results["regime"]["mape"],
                "global_mape": h_results["global_only"]["mape"],
                "regime_int_cov": h_results["regime"]["interval_coverage"],
                "global_int_cov": h_results["global_only"]["interval_coverage"],
                "regime_sample_count": h_results["regime"]["sample_count"],
                "global_sample_count": h_results["global_only"]["sample_count"],
                "regime_fold_count": h_results["regime"].get("fold_count", 0),
                "global_fold_count": h_results["global_only"].get("fold_count", 0),
            },
            "created_at": datetime.now(UTC).replace(tzinfo=None),
        }
        _upsert_accuracy(db, [drow])

    return report


def _upsert_accuracy(db, rows):
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

    parser = argparse.ArgumentParser(description="A/B test regime-switching vs global-only models")
    parser.add_argument("--max-items", type=int, default=500, help="Number of items to evaluate (default: 500)")
    parser.add_argument("--horizons", type=int, nargs="+", default=None, help="Horizons to test (default: all)")
    parser.add_argument("--skip-db", action="store_true", help="Skip writing results to database")
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
