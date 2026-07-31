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

import os
import sys
import json
import math
import hashlib
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

# Correlation-prune threshold. Part of the frame fingerprint: it decides the
# cached frame's column set, so changing it must invalidate the cache.
CORR_PRUNE_THRESHOLD = 0.95

# Dataset-construction params. Mirrors production (`ItemForecaster.MAX_BIN`)
# so the A/B measures the model prod actually ships, and bins 4x coarser than
# LightGBM's default 255 — cheaper to build and to train.
DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}


def _frame_fingerprint():
    """Identify the feature-producing code, so a cached frame can't outlive it.

    The frame is only as valid as `_compute_price_features`; a cache key over
    the data alone cannot see a code change. Same failure mode the voted-frame
    cache guards with VOTED_CACHE_VERSION.

    Hashes content rather than size+mtime: git checkouts rewrite mtime without
    changing bytes, and a stale-looking-but-valid cache hard-exits every
    worker. Covers this script's own frame-shaping constants too — they are
    baked into the cached column set just as much as forecaster.py is.
    """
    src = Path(__file__).parent.parent / "models" / "forecaster.py"
    h = hashlib.sha256(src.read_bytes())
    h.update(repr((NEW_PRIMITIVES, CORR_PRUNE_THRESHOLD)).encode())
    return h.hexdigest()[:16]


def build_frame(max_items=200, cache_path=None):
    """Load prices, engineer features, correlation-prune.

    Returns (df, pruned, present_new). If `cache_path` exists, reads it instead
    of rebuilding; otherwise builds and writes it.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != _frame_fingerprint():
                raise SystemExit(
                    f"Frame cache {cache_path} was built from different "
                    f"feature code. Rebuild it with --build-cache-only."
                )
            if meta.get("max_items") != max_items:
                raise SystemExit(
                    f"Frame cache {cache_path} was built with max_items="
                    f"{meta.get('max_items')}, not {max_items}."
                )
            df = pd.read_parquet(cache_path)
            logger.info(f"  Loaded cached frame {cache_path} ({len(df):,} rows)")
            return df, meta["pruned"], meta["present_new"]

    df, pruned, present_new = _build_frame_uncached(max_items)

    if cache_path is not None:
        # Write-then-rename, and the parquet lands before the meta that
        # vouches for it. Without this, N workers that all miss the cache
        # interleave writes into one path and a reader can pick up a meta
        # that points at a half-written frame.
        tmp_frame = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
        tmp_meta = cache_path.with_suffix(f".{os.getpid()}.tmp.json")
        df.to_parquet(tmp_frame, index=False)
        tmp_meta.write_text(json.dumps({
            "fingerprint": _frame_fingerprint(),
            "max_items": max_items,
            "pruned": pruned,
            "present_new": present_new,
            "rows": len(df),
        }))
        os.replace(tmp_frame, cache_path)
        os.replace(tmp_meta, cache_path.with_suffix(".meta.json"))
        logger.info(f"  Wrote frame cache {cache_path} ({len(df):,} rows)")

    return df, pruned, present_new


def _build_frame_uncached(max_items):
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
            -- item_slug breaks row_count ties. Without it DuckDB's parallel
            -- top-N picks a different item universe (and a different item
            -- order, which feeds LightGBM's row sampling) on every run —
            -- measured at ~1pp of dir-acc swing between two identical runs,
            -- the same size as the effect this A/B is trying to detect.
            ORDER BY row_count DESC, item_slug
            LIMIT {max_items}
        """).fetchall()

        items = rows
        logger.info(f"  {len(items)} items for evaluation")

        # ── Load all price data ─────────────────────────────────────
        # One scan of the archive for every item, not one scan per item.
        # Row order still matches the per-item loop this replaced: items in
        # `items` order (row_count DESC), day-ascending within each item.
        slugs = [r[0] for r in items]
        placeholders = ", ".join("?" for _ in slugs)
        all_prices = con.sql(f"""
            SELECT item_slug AS item_id, day AS timestamp,
                   mean_price AS price, volume
            FROM ({union_sql})
            WHERE item_slug IN ({placeholders})
        """, params=slugs).df()

        all_prices["timestamp"] = pd.to_datetime(all_prices["timestamp"])
        all_prices["date"] = all_prices["timestamp"].dt.date
        all_prices["item_id"] = pd.Categorical(
            all_prices["item_id"], categories=slugs, ordered=True
        )
        all_prices = all_prices.sort_values(
            ["item_id", "timestamp"], kind="stable"
        ).reset_index(drop=True)
        all_prices["item_id"] = all_prices["item_id"].astype(str)

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
                highly_corr = upper[col][upper[col] > CORR_PRUNE_THRESHOLD].index
                to_drop.update(highly_corr)
            pruned = [c for c in all_feature_cols if c not in to_drop]
        else:
            pruned = all_feature_cols

        logger.info(f"  Full feature count: {len(all_feature_cols)} → {len(pruned)} (after corr prune)")

        present_new = [c for c in NEW_PRIMITIVES if c in pruned]
        logger.info(f"  New primitives present after prune: {present_new}")

        return df, pruned, present_new

    finally:
        con.close()


def run_evaluation(df, pruned, present_new, horizon_filter=None, arm_filter=None,
                   n_jobs=None, q50_only=False):
    """Walk-forward evaluation over the prebuilt frame.

    Returns results[horizon][arm]. `horizon_filter` restricts to one horizon so
    the 4 horizons can run as independent processes; `n_jobs` must then be
    sized to cores/workers, or the shards oversubscribe the machine and each
    one runs slower than it would alone.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    quantiles = [0.5] if q50_only else [0.1, 0.5, 0.9]
    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)

    try:
        # Three arms. baseline/treatment differ only by the 6 new columns.
        # placebo keeps the columns but shuffles their values (handled per-fold).
        subsets = {
            "baseline": [c for c in pruned if c not in NEW_PRIMITIVES],
            "treatment": pruned,
            "placebo": pruned,
        }
        if arm_filter is not None:
            subsets = {k: v for k, v in subsets.items() if k == arm_filter}
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

            # Fold membership by range comparison on datetime64, not
            # `date.isin(train_dates)`. train_dates is always a prefix of
            # `dates`, so the isin it replaces was hashing a list that grew to
            # thousands of entries against the whole frame, once per fold per
            # arm, to select a contiguous block.
            tdf_days = pd.to_datetime(tdf["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()

            results[horizon] = {}

            for config_name, fc in subsets.items():
                # Skip if too few features
                if len(fc) < 3:
                    logger.info(f"    [{config_name}] Skipping — only {len(fc)} features")
                    continue

                logger.info(f"\n    --- Config: {config_name} ({len(fc)} features) ---")

                # Both invariant across folds: hoisted out of the loop, and the
                # per-fold slice then copies ~len(available) columns instead of
                # every column in the engineered frame.
                available = [c for c in fc if c in tdf.columns]
                if not available:
                    logger.info(f"    [{config_name}] Skipping — no features present")
                    continue
                target_col = f"target_return_{horizon}d"
                # "date" rides along for the 200K row cap's sort below.
                sub = tdf[available + [target_col, "price", "date"]]

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
                    val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]
                    if len(val_dates) < 7:
                        continue

                    train_df = sub[tdf_days <= dates_dt[window_end - 1]]
                    val_df = sub[
                        (tdf_days >= dates_dt[window_end])
                        & (tdf_days <= dates_dt[window_end + len(val_dates) - 1])
                    ]

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

                    # One median pass, used to impute both matrices (it was
                    # recomputed over the full training block for each).
                    train_median = train_df[available].median()
                    X_train = train_df[available].fillna(train_median)
                    y_train = train_df[target_col]
                    X_val = val_df[available].fillna(train_median)
                    y_val = val_df[target_col]

                    # Bin once for all quantiles: only `alpha` differs between
                    # them, and alpha plays no part in Dataset construction, so
                    # rebuilding per quantile re-binned identical data 3x.
                    Xtr_v, Xv_v = X_train.values, X_val.values
                    dtrain = lgb.Dataset(Xtr_v, y_train.values, params=DS_PARAMS,
                                         free_raw_data=False)
                    dval = lgb.Dataset(Xv_v, y_val.values, reference=dtrain,
                                       params=DS_PARAMS, free_raw_data=False)

                    models = {}
                    for q in quantiles:
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
                            "n_jobs": n_jobs,
                            # LightGBM's auto row/col-wise choice picks
                            # col-wise below ~5 threads on this shape and falls
                            # off a cliff: measured 386s vs 64s for the same
                            # 7d/3-arm unit at n_jobs=3, bit-identical results.
                            # Sharding by horizon means low per-process thread
                            # counts, so this must be pinned.
                            "force_row_wise": True,
                            **DS_PARAMS,
                        }
                        model = lgb.train(
                            params, dtrain,
                            num_boost_round=100,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(15, verbose=False),
                                       lgb.log_evaluation(0)]
                        )
                        models[q] = model.predict(Xv_v)

                    p50_ret = models[0.5]
                    # q50-only mode trains no interval bounds. Fall back to a
                    # zero-width band so the arithmetic below stays defined,
                    # and report coverage as unavailable rather than as a
                    # number that looks real (see `int_cov` below).
                    p10_ret = models.get(0.1, p50_ret)
                    p90_ret = models.get(0.9, p50_ret)

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

                    cp = val_df["price"].to_numpy(dtype=float)
                    ar = y_val.to_numpy(dtype=float)

                    # Vectorized restatement of the former per-row loop. NaN
                    # returns map to sign 0, matching the old string-compare
                    # path where a NaN failed both > and < and fell to "flat".
                    fold_total = len(val_df)
                    fold_hits = int(np.count_nonzero(
                        np.sign(np.nan_to_num(ar)) == np.sign(np.nan_to_num(p50_ret))
                    ))

                    abs_err = np.abs(cp * (1 + p50_ret / 100) - cp * (1 + ar / 100))
                    fold_mae = float(abs_err.sum())

                    actual_future = cp * (1 + ar / 100)
                    fold_int_hits = int(np.count_nonzero(
                        (cp * (1 + low_ret / 100) <= actual_future)
                        & (actual_future <= cp * (1 + high_ret / 100))
                    ))
                    fold_int_total = fold_total

                    directional_hits += fold_hits
                    directional_total += fold_total
                    mae_total += fold_mae
                    mae_count += fold_total
                    interval_hits += fold_int_hits
                    interval_total += fold_total

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
                        # None, not a number, when there was no interval to
                        # score — a zero-width band would otherwise report a
                        # plausible-looking coverage figure that means nothing.
                        "interval_coverage": None if q50_only else round(int_cov, 2),
                        "sample_count": directional_total,
                        "effective_baseline": baseline_2class,
                        "fold_count": len(per_fold),
                        "fold_mean_dir_acc": round(np.mean(fold_accs), 1) if fold_accs else 0,
                        "fold_std_dir_acc": round(np.std(fold_accs), 1) if len(fold_accs) > 1 else 0,
                        "fold_min_dir_acc": round(min(fold_accs), 1) if fold_accs else 0,
                        "fold_max_dir_acc": round(max(fold_accs), 1) if fold_accs else 0,
                        # Retained so the merge step can compute per-fold win
                        # counts between arms, not just the pooled delta.
                        "per_fold": per_fold,
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
        db.close()


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
            cov = r.get("interval_coverage")
            cov_str = f"{cov:>6.1f}%" if cov is not None else f"{'n/a':>7}"
            print(f"  │ {label:<26} {dir_acc:>7.1f}% {delta_str:>9} ${r['mae']:>5.2f} "
                  f"{cov_str} {r['fold_count']:>5}  {r['sample_count']:>8,}")

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
    parser.add_argument("--arm", choices=["baseline", "treatment", "placebo"],
                        default=None,
                        help="Only evaluate this arm (default: all three)")
    parser.add_argument("--frame-cache", default=None,
                        help="Read/write the engineered frame at this path")
    parser.add_argument("--build-cache-only", action="store_true",
                        help="Build the frame cache and exit (run once before workers)")
    parser.add_argument("--out", default=None,
                        help="Write results JSON here instead of stdout")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="LightGBM threads per process. When sharding by "
                             "horizon, set this to cores/shards (default: "
                             "cores/2, matching production training)")
    parser.add_argument("--q50-only", action="store_true",
                        help="Train only the median quantile — 3x faster, and "
                             "the ship rule only reads dir-acc. Interval "
                             "coverage is not meaningful in this mode")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: Price Technical Primitives (baseline/treatment/placebo)")
    logger.info("=" * 70)

    df, pruned, present_new = build_frame(
        max_items=args.max_items, cache_path=args.frame_cache
    )

    if args.build_cache_only:
        logger.info("Frame cache built; exiting before evaluation.")
        return 0

    results = run_evaluation(
        df, pruned, present_new,
        horizon_filter=args.horizon,
        arm_filter=args.arm,
        n_jobs=args.n_jobs,
        q50_only=args.q50_only,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"Wrote {args.out}")
        return 0

    print_comparison(results)
    print(f"\n  JSON: {json.dumps(results, indent=2, default=str)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
