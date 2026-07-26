#!/usr/bin/env python3
"""
A/B test: do the pure-price technical primitives (volatility asymmetry +
oscillator divergence) add real directional accuracy?

Three arms on the same walk-forward folds:
  - baseline:  drop the 6 new primitive columns
  - treatment: all features (including the 6 new)
  - placebo:   the 6 new columns column-shuffled (capacity-inflation guard)

Ship only if treatment > baseline (meaningful, non-flat) AND treatment > placebo
AND no horizon regresses beyond the 0.5-1.5pp budget.

Usage:
    python scripts/ab_test_price_primitives.py [--max-items 200] [--horizon 14]
"""

NEW_PRIMITIVES = (
    "vol_semidev_down_30d", "vol_semidev_up_30d", "vol_skew_30d",
    "rsi_divergence_7d", "rsi_price_divergence_7d", "macd_hist_slope_7d",
)

import sys
import json
import math
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from database import SessionLocal
from models.forecaster import ItemForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_price_primitives")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"


def run_evaluation(max_items=200, horizon_filter=None):
    """Walk-forward evaluation, returns per-horizon results for all three configurations.

    Builds features once, then evaluates with three feature subsets:
      full, no_cross_sectional, no_events.
    """
    import duckdb
    con = duckdb.connect()
    db = SessionLocal()

    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        # ── Load items ──────────────────────────────────────────────
        pq_files = sorted([str(p) for p in ARCHIVE_DIR.glob("prices-*.parquet")])
        pq_queries = []
        for pqf in pq_files:
            cols = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()
            col_names = {r[0] for r in cols}
            if "source" in col_names:
                pq_queries.append(
                    f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') WHERE source = 'aggregator_sync'"
                )
            else:
                pq_queries.append(
                    f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}')"
                )
        union_sql = " UNION ALL BY NAME ".join(pq_queries)

        rows = con.sql(f"""
            SELECT item_slug,
                   MIN(day) AS first_day,
                   MAX(day) AS last_day,
                   COUNT(*) AS row_count
            FROM ({union_sql})
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
                FROM ({union_sql})
                WHERE item_slug = ?
                ORDER BY day
            """, params=[item_slug]).fetchall()
            item_df = pd.DataFrame(
                item_rows, columns=["item_id", "timestamp", "price", "volume"]
            )
            item_df["timestamp"] = pd.to_datetime(item_df["timestamp"])
            item_df["date"] = item_df["timestamp"].dt.date
            all_rows.append(item_df)

        all_prices = pd.concat(all_rows, ignore_index=True)

        # ── Build features once ─────────────────────────────────────
        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)

        EXCLUDE = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date"}
        all_feature_cols = [c for c in df.columns if c not in EXCLUDE
                            and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]

        # Prune highly correlated
        if len(all_feature_cols) > 2:
            corr = df[all_feature_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                if col in to_drop:
                    continue
                highly_corr = upper[col][upper[col] > 0.95].index
                to_drop.update(highly_corr)
            pruned = [c for c in all_feature_cols if c not in to_drop]
        else:
            pruned = all_feature_cols

        logger.info(f"  Full feature count: {len(all_feature_cols)} → {len(pruned)} (after corr prune)")

        # Three arms. baseline/treatment differ only by the 6 new columns.
        # placebo keeps the columns but shuffles their values (handled per-fold).
        subsets = {
            "baseline": [c for c in pruned if c not in NEW_PRIMITIVES],
            "treatment": pruned,
            "placebo": pruned,
        }
        present_new = [c for c in NEW_PRIMITIVES if c in pruned]
        logger.info(f"  New primitives present after prune: {present_new}")
        for name, cols in subsets.items():
            logger.info(f"    {name:20s}: {len(cols):>3d} features")

        # ── Horizons to evaluate ────────────────────────────────────
        horizons = ItemForecaster.HORIZONS
        if horizon_filter is not None:
            horizons = [h for h in horizons if h == horizon_filter]

        # Store results: results[horizon][config_name] = metrics dict
        results = {}

        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}")
            logger.info(f"  Evaluating {horizon}d horizon...")
            logger.info(f"  {'=' * 60}")

            tdf = forecaster.prepare_targets(df, horizon)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values(["item_id", "date"])

            if tdf.empty:
                logger.warning(f"    No valid targets for {horizon}d")
                continue

            dates = sorted(tdf["date"].unique())
            split_idx = len(dates) * 2 // 3

            results[horizon] = {}

            for config_name, fc in subsets.items():
                # Skip if too few features
                if len(fc) < 3:
                    logger.info(f"    [{config_name}] Skipping — only {len(fc)} features")
                    continue

                logger.info(f"\n    --- Config: {config_name} ({len(fc)} features) ---")

                directional_hits = 0
                directional_total = 0
                mae_total = 0.0
                mae_count = 0
                interval_hits = 0
                interval_total = 0
                per_fold = []

                VAL_WINDOW_DAYS = 21
                step = 60
                for window_end in range(split_idx + 1, len(dates), step):
                    train_dates = dates[:window_end]
                    val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]
                    if len(val_dates) < 7:
                        continue

                    train_df = tdf[tdf["date"].isin(train_dates)]
                    val_df = tdf[tdf["date"].isin(val_dates)]

                    if config_name == "placebo" and present_new:
                        rng = np.random.default_rng(42)
                        train_df = train_df.copy()
                        val_df = val_df.copy()
                        for col in present_new:
                            train_df[col] = rng.permutation(train_df[col].values)
                            val_df[col] = rng.permutation(val_df[col].values)

                    if len(val_df) < 50:
                        continue

                    if len(train_df) > 200000:
                        train_df = train_df.sort_values("date").tail(200000)

                    available = [c for c in fc if c in tdf.columns]
                    if not available:
                        continue

                    X_train = train_df[available].fillna(train_df[available].median())
                    y_train = train_df[f"target_return_{horizon}d"]
                    X_val = val_df[available].fillna(train_df[available].median())
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
                        model = lgb.train(
                            params, dtrain,
                            num_boost_round=100,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(15, verbose=False),
                                       lgb.log_evaluation(0)]
                        )
                        models[q] = model.predict(X_val.values)

                    p10_ret = models[0.1]
                    p50_ret = models[0.5]
                    p90_ret = models[0.9]

                    # Fix quantile crossing
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
                        "dir_acc": round(dir_acc, 2),
                        "mae": round(mae, 4),
                        "interval_coverage": round(int_cov, 2),
                        "sample_count": directional_total,
                        "effective_baseline": baseline_2class,
                        "fold_count": len(per_fold),
                        "fold_mean_dir_acc": round(np.mean(fold_accs), 1) if fold_accs else 0,
                        "fold_std_dir_acc": round(np.std(fold_accs), 1) if len(fold_accs) > 1 else 0,
                        "fold_min_dir_acc": round(min(fold_accs), 1) if fold_accs else 0,
                        "fold_max_dir_acc": round(max(fold_accs), 1) if fold_accs else 0,
                    }
                    result["improvement_over_baseline_pp"] = round(dir_acc - baseline_2class, 1)
                    results[horizon][config_name] = result

                    logger.info(f"      DirAcc={dir_acc:.1f}% ({directional_total:,} samples, "
                                f"{result['improvement_over_baseline_pp']:.1f}pp above baseline)")

        logger.info("\n" + "=" * 64)
        logger.info("SHIP DECISION SUMMARY (dir-acc %, treatment vs baseline vs placebo)")
        logger.info("=" * 64)
        for h in sorted(results):
            r = results[h]
            if not all(k in r for k in ("baseline", "treatment", "placebo")):
                continue
            b = r["baseline"]["dir_acc"]
            t = r["treatment"]["dir_acc"]
            p = r["placebo"]["dir_acc"]
            logger.info(
                f"  {h:>2}d  baseline={b:5.2f}  treatment={t:5.2f}  "
                f"placebo={p:5.2f}  (t-b={t-b:+.2f}, t-p={t-p:+.2f})"
            )

        return results

    finally:
        con.close()


def print_comparison(results):
    """Print a comparison table across arms (baseline/treatment/placebo) and horizons."""
    config_order = ["baseline", "treatment", "placebo"]
    config_labels = {
        "baseline": "Baseline (no primitives)",
        "treatment": "Treatment (+6 primitives)",
        "placebo": "Placebo (shuffled)",
    }

    print("\n" + "=" * 100)
    print("A/B TEST — Price Technical Primitives (volatility asymmetry + oscillator divergence)")
    print("=" * 100)

    for horizon in sorted(results.keys()):
        h_results = results[horizon]
        print(f"\n  ┌─ {horizon}d Horizon {'─' * 60}┐")

        header = f"  │ {'Arm':<26} {'DirAcc':>8} {'vs Base':>9} {'MAE':>8} {'IntCov':>8} {'Folds':>6} {'Samples':>9}"
        sep = f"  │ {'─' * 26} {'─' * 8} {'─' * 9} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 9}"
        base_dir_acc = h_results.get("baseline", {}).get("directional_accuracy", 0)

        print(header)
        print(sep)

        for cfg in config_order:
            r = h_results.get(cfg)
            if r is None:
                continue
            dir_acc = r["directional_accuracy"]
            delta = dir_acc - base_dir_acc
            delta_str = f"{delta:+.2f}pp" + (" ✅" if delta > 0.5 else " ❌" if delta < -0.5 else "  ")
            label = config_labels.get(cfg, cfg)
            print(f"  │ {label:<26} {dir_acc:>7.1f}% {delta_str:>9} ${r['mae']:>5.2f} "
                  f"{r['interval_coverage']:>6.1f}% {r['fold_count']:>5}  {r['sample_count']:>8,}")

        print(f"  └{'─' * 78}┘")

    # ── Ship decision guidance ───────────────────────────────────────
    print(f"\n  {'=' * 100}")
    print(f"  INTERPRETATION")
    print(f"  {'=' * 100}")
    print(f"")
    print(f"  'vs Base' compares each arm to baseline (the 6 new primitives dropped).")
    print(f"  Ship the primitives only if treatment beats baseline by a meaningful,")
    print(f"  non-flat margin AND treatment beats placebo (rules out capacity inflation)")
    print(f"  AND no horizon regresses beyond the 0.5-1.5pp budget.")

    # ── Verdict: new primitives, short vs long horizons ─────────────
    short_horizons, long_horizons = [3, 7], [14, 30]
    short_deltas, long_deltas = [], []
    for h in results:
        h_res = results[h]
        base = h_res.get("baseline", {}).get("directional_accuracy")
        treat = h_res.get("treatment", {}).get("directional_accuracy")
        if base is not None and treat is not None:
            delta = treat - base
            if h in short_horizons:
                short_deltas.append(delta)
            if h in long_horizons:
                long_deltas.append(delta)

    print(f"\n  New Primitives (treatment vs baseline):")
    if short_deltas:
        avg_short = np.mean(short_deltas)
        print(f"    Short horizons (3d/7d):    avg Δ = {avg_short:+.2f}pp "
              f"{'📈 helpful' if avg_short > 0.3 else '📉 harmful' if avg_short < -0.3 else '➡️ neutral'}")
    if long_deltas:
        avg_long = np.mean(long_deltas)
        print(f"    Long horizons (14d/30d):   avg Δ = {avg_long:+.2f}pp "
              f"{'📈 helpful' if avg_long > 0.3 else '📉 harmful' if avg_long < -0.3 else '➡️ neutral'}")

    print("")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="A/B test: price technical primitives contribution per horizon"
    )
    parser.add_argument("--max-items", type=int, default=200,
                        help="Number of items to evaluate (default: 200)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Only evaluate this horizon (default: all)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: Price Technical Primitives (baseline/treatment/placebo)")
    logger.info("=" * 70)

    results = run_evaluation(
        max_items=args.max_items,
        horizon_filter=args.horizon,
    )

    print_comparison(results)

    print(f"\n  JSON: {json.dumps(results, indent=2, default=str)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
