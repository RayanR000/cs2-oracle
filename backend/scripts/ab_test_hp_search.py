#!/usr/bin/env python3
"""
A/B test: does enabling Optuna HP search on 14d/30d DART actually improve
calibration vs. the DART_PARAMS defaults?

Measured on IDENTICAL walk-forward folds so the effect is attributable — unlike
the meta.json before/after, which compared a 4-fold run to an 8-9-fold run and
so couldn't isolate the HP change from the fold change.

Two arms per horizon, sharing the same items / window / folds / seed:
  - defaults: DART with DART_PARAMS defaults + the old hardcoded fallback tree
              params (num_leaves=47, lr=0.01, ...). This is what
              SKIP_HP_HORIZONS=[3,14,30] produced before the change.
  - tuned:    DART params from ItemForecaster._optuna_search_params (15 trials),
              searched ONCE on the largest training window (mirrors production:
              HP searched once, applied across the rolling retrains), including
              the tuned dropout params (drop_rate/max_drop/skip_drop).

Primary metric — conformal q̂: the (1-α)(1+1/n) quantile of CQR nonconformity
scores (α=0.10), i.e. the interval widening (in return pp) needed to hit 90%
coverage. Lower = better-calibrated raw intervals. Same quantity meta.json's
`conformal_calibration` tracks. Because the folds are identical across arms, the
per-fold q̂ delta is attributable to the HP change alone.

Usage:
    python scripts/ab_test_hp_search.py [--max-items 200] [--horizon 14]
                                        [--trials 15] [--boost-rounds 500]
"""

import sys
import json
import logging
from pathlib import Path

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
logger = logging.getLogger("ab_test_hp_search")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

ALPHA = 0.10          # target 90% conformal coverage
VAL_WINDOW_DAYS = 21
STEP = 60
# Old hardcoded fallback tree params (forecaster.py `else` branch of the HP merge).
FALLBACK_TREE_PARAMS = {
    "num_leaves": 47, "learning_rate": 0.01, "lambda_l1": 0.0,
    "lambda_l2": 1.5, "max_depth": 5, "min_data_in_leaf": 15,
}


def _base_params(q: float) -> dict:
    """Base DART params for quantile q, mirroring _train_horizon_inline."""
    p = {
        "feature_pre_filter": False,
        "objective": "quantile",
        "alpha": q,
        "metric": "quantile",
        "boosting_type": "dart",
        "min_gain_to_split": 0.1,
        "feature_fraction": 0.7,
        "max_bin": ItemForecaster.MAX_BIN,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    if q == 0.5:
        p["data_sample_strategy"] = "goss"
        p["top_rate"] = 0.2
        p["other_rate"] = 0.1
    else:
        p["data_sample_strategy"] = "bagging"
        p["subsample"] = 0.8
    return p


def _apply_hp(base: dict, best: dict | None) -> dict:
    """Merge searched HP into base params (mirrors forecaster merge_keys), then
    fill DART dropout defaults for anything not tuned. `best=None` = defaults arm."""
    p = dict(base)
    merge_keys = ["num_leaves", "learning_rate", "lambda_l1", "lambda_l2",
                  "max_depth", "min_data_in_leaf", "subsample",
                  "drop_rate", "max_drop", "skip_drop"]
    src = best if best is not None else FALLBACK_TREE_PARAMS
    for k in merge_keys:
        if k in src:
            p[k] = src[k]
    for k, v in ItemForecaster.DART_PARAMS.items():
        p.setdefault(k, v)
    return p


def _conformal_qhat(low_ret, high_ret, y) -> float:
    """CQR nonconformity q̂: (1-α)(1+1/n) quantile of max(low-y, y-high)."""
    nc = np.maximum(low_ret - y, y - high_ret)
    n = len(nc)
    if n == 0:
        return float("nan")
    q_level = min(1.0, (1 - ALPHA) * (1 + 1 / n))
    return float(np.quantile(nc, q_level))


def _build_folds(dates):
    """Fixed walk-forward folds: (train_dates, val_dates), shared across arms."""
    split_idx = len(dates) * 2 // 3
    folds = []
    for window_end in range(split_idx + 1, len(dates), STEP):
        train_dates = dates[:window_end]
        val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]
        if len(val_dates) < 7:
            continue
        folds.append((train_dates, val_dates))
    return folds


def _train_predict(params, X_train, y_train, X_val, boost_rounds):
    dtrain = lgb.Dataset(X_train.values, y_train.values,
                         params={"max_bin": ItemForecaster.MAX_BIN,
                                 "feature_pre_filter": False})
    model = lgb.train(params, dtrain, num_boost_round=boost_rounds,
                      callbacks=[lgb.log_evaluation(0)])
    return model.predict(X_val.values)


def load_features(con, forecaster, events_df, max_items):
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


def run(max_items, horizon_filter, trials, boost_rounds):
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

    horizons = [h for h in (14, 30) if horizon_filter is None or h == horizon_filter]
    results = {}

    for horizon in horizons:
        logger.info(f"\n{'=' * 60}\n  {horizon}d horizon\n{'=' * 60}")
        tdf = forecaster.prepare_targets(df, horizon)
        tcol = f"target_return_{horizon}d"
        tdf = tdf.dropna(subset=[tcol]).sort_values(["item_id", "date"]).copy()
        if tdf.empty:
            logger.warning(f"  no targets for {horizon}d")
            continue
        available = [c for c in feat_cols if c in tdf.columns]
        dates = sorted(tdf["date"].unique())
        folds = _build_folds(dates)
        if not folds:
            logger.warning(f"  no folds for {horizon}d")
            continue

        # ── Tuned arm: search HP once on the pre-fold train window ──
        split_idx = len(dates) * 2 // 3
        pre = tdf[tdf["date"].isin(dates[:split_idx])]
        cut = dates[split_idx - VAL_WINDOW_DAYS]
        hp_tr = pre[pre["date"] < cut]
        hp_va = pre[pre["date"] >= cut]
        med = hp_tr[available].median()
        best_by_q = {}
        for q in ItemForecaster.QUANTILES:
            logger.info(f"  Optuna search {horizon}d p{int(q*100)} ({trials} trials, dart)...")
            best_by_q[q] = forecaster._optuna_search_params(
                hp_tr[available].fillna(med), hp_tr[tcol],
                hp_va[available].fillna(med), hp_va[tcol],
                quantile=q, boosting_type="dart", n_trials=trials, horizon=horizon,
            )

        arms = {
            "defaults": {q: _apply_hp(_base_params(q), None) for q in ItemForecaster.QUANTILES},
            "tuned":    {q: _apply_hp(_base_params(q), best_by_q[q]) for q in ItemForecaster.QUANTILES},
        }

        results[horizon] = {"folds": len(folds), "arms": {}, "per_fold_qhat": {}}
        for arm_name, params_by_q in arms.items():
            per_fold = []
            for fi, (train_dates, val_dates) in enumerate(folds):
                train_df = tdf[tdf["date"].isin(train_dates)]
                val_df = tdf[tdf["date"].isin(val_dates)]
                if len(val_df) < 50:
                    continue
                if len(train_df) > 200000:
                    train_df = train_df.sort_values("date").tail(200000)
                med = train_df[available].median()
                X_train = train_df[available].fillna(med)
                X_val = val_df[available].fillna(med)
                y_val = val_df[tcol].values

                preds = {q: _train_predict(params_by_q[q], X_train, train_df[tcol],
                                           X_val, boost_rounds)
                         for q in ItemForecaster.QUANTILES}
                p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
                low = np.minimum(p10, p50)
                high = np.maximum(p90, p50)

                qhat = _conformal_qhat(low, high, y_val)
                cov = float(np.mean((y_val >= low) & (y_val <= high))) * 100
                dir_acc = float(np.mean(np.sign(p50) == np.sign(y_val))) * 100
                mae = float(np.mean(np.abs(p50 - y_val)))
                per_fold.append({"fold": fi, "val_start": str(val_dates[0]),
                                 "qhat": round(qhat, 4), "int_cov": round(cov, 1),
                                 "dir_acc": round(dir_acc, 1), "mae": round(mae, 4),
                                 "n": len(val_df)})

            qhats = [f["qhat"] for f in per_fold]
            results[horizon]["arms"][arm_name] = {
                "mean_qhat": round(float(np.mean(qhats)), 4) if qhats else None,
                "mean_int_cov": round(float(np.mean([f["int_cov"] for f in per_fold])), 1) if per_fold else None,
                "mean_dir_acc": round(float(np.mean([f["dir_acc"] for f in per_fold])), 1) if per_fold else None,
                "mean_mae": round(float(np.mean([f["mae"] for f in per_fold])), 4) if per_fold else None,
                "n_folds": len(per_fold),
                "per_fold": per_fold,
            }
            results[horizon]["per_fold_qhat"][arm_name] = qhats

    return results


def print_report(results):
    print("\n" + "=" * 78)
    print("HP-SEARCH A/B — DART defaults vs. Optuna-tuned (identical folds)")
    print("=" * 78)
    for horizon in sorted(results):
        r = results[horizon]
        print(f"\n  ┌─ {horizon}d ({r['folds']} folds) {'─' * 45}┐")
        print(f"  │ {'Arm':<10} {'q̂ (lower=better)':>18} {'IntCov':>8} {'DirAcc':>8} {'MAE':>9}")
        print(f"  │ {'─'*10} {'─'*18} {'─'*8} {'─'*8} {'─'*9}")
        for arm in ("defaults", "tuned"):
            a = r["arms"].get(arm)
            if not a:
                continue
            print(f"  │ {arm:<10} {a['mean_qhat']:>18} {a['mean_int_cov']:>7}% "
                  f"{a['mean_dir_acc']:>7}% {a['mean_mae']:>9}")
        # Paired same-fold q̂ delta (the attributable number).
        d = results[horizon]["per_fold_qhat"].get("defaults", [])
        t = results[horizon]["per_fold_qhat"].get("tuned", [])
        if d and t and len(d) == len(t):
            deltas = np.array(t) - np.array(d)
            better = int(np.sum(deltas < 0))
            print(f"  │")
            print(f"  │ Paired q̂ delta (tuned − defaults): mean {deltas.mean():+.4f}"
                  f"  ({better}/{len(deltas)} folds better)")
            verdict = ("tuned better-calibrated" if deltas.mean() < -0.05
                       else "defaults better-calibrated" if deltas.mean() > 0.05
                       else "no material difference")
            print(f"  │ Verdict: {verdict}")
        print(f"  └{'─' * 66}┘")


def main():
    import argparse
    p = argparse.ArgumentParser(description="A/B: 14d/30d DART HP search vs defaults")
    p.add_argument("--max-items", type=int, default=200)
    p.add_argument("--horizon", type=int, default=None, choices=[14, 30])
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--boost-rounds", type=int, default=ItemForecaster.DART_NUM_BOOST_ROUND)
    args = p.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: 14d/30d DART HP search vs DART_PARAMS defaults")
    logger.info(f"  max_items={args.max_items} trials={args.trials} boost_rounds={args.boost_rounds}")
    logger.info("=" * 70)

    results = run(args.max_items, args.horizon, args.trials, args.boost_rounds)
    print_report(results)
    print(f"\n  JSON: {json.dumps(results, indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
