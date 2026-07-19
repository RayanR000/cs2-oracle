#!/usr/bin/env python3
"""
A/B test: does cross-sectional + event relevance weighting actually help
longer horizons (14d/30d)?

Ablation study with three configurations on the same walk-forward eval:
  - Full:         All features (baseline)
  - NoCrossSec:   Remove market-level cross-sectional features
  - NoEvents:     Remove event features (decay, density, relevance)

The hypothesis is that skin-market prices past ~2 weeks are dominated by
unforecastable one-off events, so cross-sectional and event features add
less signal for 14d/30d than for 3d/7d.

Usage:
    python scripts/ab_test_feature_contribution.py [--max-items 200] [--horizon 14]
"""

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
logger = logging.getLogger("ab_test_feature_contribution")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

CROSS_SECTIONAL_PREFIXES = ("market_", "market_regime_", "item_return_vs_market_", "item_volume_vs_market_")
EVENT_PREFIXES = ("event_decay_", "events_next_", "event_density_")


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
                    f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') WHERE source = 'STEAMCOMMUNITY'"
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

        # Define feature subsets
        subsets = {
            "full": pruned,
            "no_cross_sectional": [c for c in pruned if not c.startswith(CROSS_SECTIONAL_PREFIXES)],
            "no_events": [c for c in pruned if not c.startswith(EVENT_PREFIXES)],
        }
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

        return results

    finally:
        con.close()


def print_comparison(results):
    """Print a comparison table across configurations and horizons."""
    config_order = ["full", "no_cross_sectional", "no_events"]
    config_labels = {
        "full": "Full (baseline)",
        "no_cross_sectional": "No Cross-Sec",
        "no_events": "No Events",
    }

    print("\n" + "=" * 100)
    print("FEATURE CONTRIBUTION A/B TEST — Cross-Sectional & Event Features")
    print("=" * 100)

    for horizon in sorted(results.keys()):
        h_results = results[horizon]
        print(f"\n  ┌─ {horizon}d Horizon {'─' * 60}┐")

        header = f"  │ {'Config':<22} {'DirAcc':>8} {'vs Base':>9} {'MAE':>8} {'IntCov':>8} {'Folds':>6} {'Samples':>9}"
        sep = f"  │ {'─' * 22} {'─' * 8} {'─' * 9} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 9}"
        base_dir_acc = h_results.get("full", {}).get("directional_accuracy", 0)

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
            print(f"  │ {label:<22} {dir_acc:>7.1f}% {delta_str:>9} ${r['mae']:>5.2f} "
                  f"{r['interval_coverage']:>6.1f}% {r['fold_count']:>5}  {r['sample_count']:>8,}")

        print(f"  └{'─' * 78}┘")

    # ── Per-feature-group summary ──────────────────────────────────
    print(f"\n  {'=' * 100}")
    print(f"  INTERPRETATION")
    print(f"  {'=' * 100}")
    print(f"")
    print(f"  A positive delta for 'No Cross-Sec' or 'No Events' means removing")
    print(f"  those features IMPROVED accuracy (the features added noise).")
    print(f"  A negative delta means the features were genuinely helpful.")
    print(f"")
    print(f"  Check whether the delta for 14d/30d is materially different from 3d/7d:")
    print(f"  - If cross-sectional helps 3d/7d but NOT 14d/30d → shorter-horizon signal")
    print(f"  - If events helps short horizons but not long → supports the hypothesis")
    print(f"    that longer horizons are dominated by unforecastable events")

    # ── Verdict per feature group ──────────────────────────────────
    for feat_name, feat_key, short_horizons, long_horizons in [
        ("Cross-Sectional Features", "no_cross_sectional", [3, 7], [14, 30]),
        ("Event Features", "no_events", [3, 7], [14, 30]),
    ]:
        short_deltas = []
        long_deltas = []
        for h in results:
            h_res = results[h]
            base = h_res.get("full", {}).get("directional_accuracy")
            ablated = h_res.get(feat_key, {}).get("directional_accuracy")
            if base is not None and ablated is not None:
                delta = ablated - base
                if h in short_horizons:
                    short_deltas.append(delta)
                if h in long_horizons:
                    long_deltas.append(delta)

        print(f"\n  {feat_name}:")
        if short_deltas:
            avg_short = np.mean(short_deltas)
            print(f"    Short horizons (3d/7d):    avg Δ = {avg_short:+.2f}pp "
                  f"{'📈 harmful' if avg_short > 0.3 else '📉 helpful' if avg_short < -0.3 else '➡️ neutral'}")
        if long_deltas:
            avg_long = np.mean(long_deltas)
            print(f"    Long horizons (14d/30d):   avg Δ = {avg_long:+.2f}pp "
                  f"{'📈 harmful' if avg_long > 0.3 else '📉 helpful' if avg_long < -0.3 else '➡️ neutral'}")

    print("")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="A/B test: cross-sectional & event feature contribution per horizon"
    )
    parser.add_argument("--max-items", type=int, default=200,
                        help="Number of items to evaluate (default: 200)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Only evaluate this horizon (default: all)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: Cross-Sectional & Event Feature Contribution")
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
