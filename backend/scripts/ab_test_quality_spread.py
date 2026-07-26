#!/usr/bin/env python3
"""
A/B test: do the quality-spread / cross-wear features add directional accuracy?

Ablation on a single walk-forward eval (mirrors ab_test_feature_contribution.py):
  - full:              all features INCLUDING the quality_spread group
  - no_quality_spread: quality_spread features removed

quality_spread = wear-ladder ratio/z/change, StatTrak & Souvenir premiums, and
the has_* indicator flags (see forecaster._add_quality_spread_features).

Because these features are only non-neutral for items that have sibling variants
(same skin in other wears; StatTrak/Souvenir pairs), the item universe is chosen
from actual multi-variant groups rather than the top-N by volume — otherwise the
features would be ~all-zero on the subset and the effect unmeasurable. Coverage
(fraction of rows where each axis is active) is logged.

Usage:
    python scripts/ab_test_quality_spread.py [--max-items 1500] [--horizon 7]
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from database import SessionLocal
from models.forecaster import ItemForecaster
from models.item_parser import parse_item_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_quality_spread")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Prefixes that make up the quality_spread feature group (mirror _feature_group).
QS_PREFIXES = ("wear_", "stattrak_", "souvenir_",
               "has_wear", "has_stattrak", "has_souvenir")


def _variant_group_items(all_slugs):
    """Return the set of item slugs that belong to a multi-member variant group
    on at least one axis (wear ladder, StatTrak pair, or Souvenir pair)."""
    wear = defaultdict(set)
    st = defaultdict(set)
    sv = defaultdict(set)
    parsed = {}
    for slug in all_slugs:
        p = parse_item_name(slug)
        parsed[slug] = p
        w, s, q = p.get("weapon"), p.get("skin_name"), p.get("quality")
        if not (w and s and q):
            continue
        stf, svf = int(p.get("is_stattrak", False)), int(p.get("is_souvenir", False))
        wear[(w, s, stf, svf)].add(slug)
        st[(w, s, q, svf)].add(slug)
        sv[(w, s, q, stf)].add(slug)

    keep = set()
    for groups in (wear, st, sv):
        for members in groups.values():
            if len(members) >= 2:
                keep.update(members)
    return keep


def run_evaluation(max_items=1500, horizon_filter=None):
    import duckdb
    con = duckdb.connect()
    db = SessionLocal()

    try:
        forecaster = ItemForecaster(db_session=db)
        forecaster.ENABLE_QUALITY_SPREAD = True  # compute the features for this run
        events_df = forecaster.fetch_events()
        db.close()

        # ── Enumerate items with enough history ─────────────────────
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
            SELECT item_slug, COUNT(*) AS row_count
            FROM ({union_sql})
            GROUP BY item_slug
            HAVING row_count >= 90
            ORDER BY row_count DESC
        """).fetchall()

        eligible = [r[0] for r in rows]
        group_items = _variant_group_items(eligible)
        # Keep variant-group items (ordered by history depth), capped at max_items.
        selected = [s for s in eligible if s in group_items][:max_items]
        logger.info(f"  {len(eligible)} items with >=90d history; "
                    f"{len(group_items)} in multi-variant groups; "
                    f"{len(selected)} selected for eval")

        if len(selected) < 20:
            logger.error("  Too few variant-group items to run a meaningful A/B.")
            return {}

        # ── Load price data for selected items ──────────────────────
        sel_set = set(selected)
        placeholders = ",".join(["?"] * len(selected))
        all_item_rows = con.sql(f"""
            SELECT item_slug AS item_id, day AS timestamp,
                   mean_price AS price, volume
            FROM ({union_sql})
            WHERE item_slug IN ({placeholders})
            ORDER BY item_slug, day
        """, params=selected).fetchall()
        all_prices = pd.DataFrame(
            all_item_rows, columns=["item_id", "timestamp", "price", "volume"])
        all_prices["timestamp"] = pd.to_datetime(all_prices["timestamp"])
        all_prices["date"] = all_prices["timestamp"].dt.date

        # Seed the metadata cache so _add_quality_spread_features resolves names
        # without a DB round-trip (item_slug IS the full item name).
        meta = pd.DataFrame({"item_id": selected, "name": selected, "type": "skin"})
        forecaster._item_meta_cache = meta

        # ── Build features once (incl. quality_spread) ──────────────
        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)
        df = forecaster._add_quality_spread_features(df)

        # Coverage: fraction of rows where each axis is active
        for flag in ("has_wear_siblings", "has_stattrak_pair", "has_souvenir_pair"):
            if flag in df.columns:
                frac = float((df[flag] > 0).mean()) * 100
                logger.info(f"  coverage {flag}: {frac:.1f}% of rows")

        EXCLUDE = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date"}
        all_feature_cols = [c for c in df.columns if c not in EXCLUDE
                            and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]

        # Prune highly correlated (same threshold as production)
        if len(all_feature_cols) > 2:
            corr = df[all_feature_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                if col in to_drop:
                    continue
                to_drop.update(upper[col][upper[col] > 0.95].index)
            pruned = [c for c in all_feature_cols if c not in to_drop]
        else:
            pruned = all_feature_cols

        qs_in_pruned = [c for c in pruned if c.startswith(QS_PREFIXES)]
        logger.info(f"  Full feature count: {len(all_feature_cols)} → {len(pruned)} "
                    f"(after corr prune); {len(qs_in_pruned)} quality_spread features retained")
        logger.info(f"  quality_spread features: {qs_in_pruned}")

        subsets = {
            "full": pruned,
            "no_quality_spread": [c for c in pruned if not c.startswith(QS_PREFIXES)],
        }

        horizons = ItemForecaster.HORIZONS
        if horizon_filter is not None:
            horizons = [h for h in horizons if h == horizon_filter]

        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Evaluating {horizon}d horizon...\n  {'=' * 60}")
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
                if len(fc) < 3:
                    continue
                logger.info(f"\n    --- Config: {config_name} ({len(fc)} features) ---")
                directional_hits = 0
                directional_total = 0
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

                    med = train_df[available].median()
                    X_train = train_df[available].fillna(med)
                    y_train = train_df[f"target_return_{horizon}d"]
                    X_val = val_df[available].fillna(med)
                    y_val = val_df[f"target_return_{horizon}d"]

                    preds = {}
                    for q in [0.1, 0.5, 0.9]:
                        params = {
                            "objective": "quantile", "alpha": q, "metric": "quantile",
                            "boosting_type": "gbdt", "num_leaves": 31, "max_depth": 5,
                            "min_data_in_leaf": 15, "min_gain_to_split": 0.1,
                            "learning_rate": 0.03, "feature_fraction": 0.7,
                            "bagging_fraction": 0.7, "bagging_freq": 5,
                            "lambda_l1": 0.5, "lambda_l2": 0.5, "verbosity": -1,
                            "random_state": 42, "n_jobs": -1,
                        }
                        dtrain = lgb.Dataset(X_train.values, y_train.values)
                        dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain)
                        model = lgb.train(
                            params, dtrain, num_boost_round=100,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(15, verbose=False),
                                       lgb.log_evaluation(0)])
                        preds[q] = model.predict(X_val.values)

                    p50_ret = preds[0.5]
                    actual_returns = y_val.values
                    fold_hits = 0
                    fold_total = 0
                    for i in range(len(val_df)):
                        ar = float(actual_returns[i])
                        mid_r = float(p50_ret[i])
                        actual_dir = "up" if ar > 0 else "down" if ar < 0 else "flat"
                        pred_dir = "up" if mid_r > 0 else "down" if mid_r < 0 else "flat"
                        if pred_dir == actual_dir:
                            directional_hits += 1
                            fold_hits += 1
                        directional_total += 1
                        fold_total += 1

                    per_fold.append({
                        "val_start": str(val_dates[0]), "val_end": str(val_dates[-1]),
                        "dir_acc": round(fold_hits / fold_total * 100, 1) if fold_total else 0,
                        "n": fold_total,
                    })

                if directional_total > 0:
                    dir_acc = directional_hits / directional_total * 100
                    fold_accs = [f["dir_acc"] for f in per_fold]
                    results[horizon][config_name] = {
                        "directional_accuracy": round(dir_acc, 2),
                        "sample_count": directional_total,
                        "fold_count": len(per_fold),
                        "fold_mean_dir_acc": round(np.mean(fold_accs), 1) if fold_accs else 0,
                        "fold_std_dir_acc": round(np.std(fold_accs), 1) if len(fold_accs) > 1 else 0,
                    }
                    logger.info(f"      DirAcc={dir_acc:.2f}% ({directional_total:,} samples, "
                                f"{len(per_fold)} folds)")

        return results

    finally:
        con.close()


def print_comparison(results):
    print("\n" + "=" * 92)
    print("QUALITY-SPREAD A/B TEST — contribution of the quality_spread feature group")
    print("=" * 92)
    print(f"\n  Positive delta = quality_spread IMPROVES directional accuracy "
          f"(full minus no_quality_spread).\n")
    header = f"  {'Horizon':<9} {'full':>9} {'no_qs':>9} {'delta':>10} {'folds':>6} {'samples':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    verdict_rows = []
    for horizon in sorted(results.keys()):
        h = results[horizon]
        full = h.get("full", {}).get("directional_accuracy")
        noqs = h.get("no_quality_spread", {}).get("directional_accuracy")
        if full is None or noqs is None:
            continue
        delta = full - noqs
        mark = " ✅" if delta > 0.5 else " ❌" if delta < -0.5 else "  "
        folds = h.get("full", {}).get("fold_count", 0)
        samples = h.get("full", {}).get("sample_count", 0)
        print(f"  {str(horizon)+'d':<9} {full:>8.2f}% {noqs:>8.2f}% {delta:>+8.2f}pp{mark} "
              f"{folds:>6} {samples:>9,}")
        verdict_rows.append((horizon, delta))

    print("\n  INTERPRETATION")
    print("  Gate (per spec): SHIP only if quality_spread is retained AND short-horizon")
    print("  (3d/7d) delta >= +0.5pp AND no horizon regresses beyond -1.5pp.")
    short = [d for hz, d in verdict_rows if hz in (3, 7)]
    if short:
        best_short = max(short)
        print(f"  Best 3d/7d delta: {best_short:+.2f}pp — "
              f"{'meets' if best_short >= 0.5 else 'BELOW'} the +0.5pp ship threshold.")
    print("  NOTE: walk-forward on a data-rich variant-group subset typically reads")
    print("  higher than production backtest; treat deltas as directional evidence.")


def main():
    args = list(sys.argv[1:])
    max_items = 1500
    horizon = None
    i = 0
    while i < len(args):
        if args[i] == "--max-items" and i + 1 < len(args):
            max_items = int(args[i + 1]); i += 2
        elif args[i] == "--horizon" and i + 1 < len(args):
            horizon = int(args[i + 1]); i += 2
        else:
            i += 1
    logger.info(f"Quality-spread A/B: max_items={max_items}, "
                f"horizon={horizon or 'all'}")
    results = run_evaluation(max_items=max_items, horizon_filter=horizon)
    print_comparison(results)
    print(f"\nRESULT_JSON: {json.dumps(results, default=str)}")


if __name__ == "__main__":
    main()
