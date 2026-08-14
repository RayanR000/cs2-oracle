#!/usr/bin/env python3
"""A/B the q50 row-sampling strategy (GOSS vs bagging) over purge-gap CV.

Motivation
----------
`data_sample_strategy="goss"` is applied to the q50 quantile only
(`forecaster.py` param sites ~1736 / ~2343 / ~2372); q10/q90 use bagging.
GOSS ranks rows by |gradient| to decide which to subsample, but the quantile
objective emits constant ±alpha gradients, so that ranking is degenerate and
the small-gradient amplification injects bias. Observed effect in the shipped
artifacts: 7d q50 saves 1-2 trees per ensemble member (best_iteration ~= 1),
i.e. the served median model is effectively an intercept.

Arms (identical data, identical folds, identical tree params — only sampling
differs):
  - **goss**    : production config (control)
  - **bagging** : data_sample_strategy=bagging, subsample=0.8, bagging_freq=1

Pre-registered gate (decided before running)
--------------------------------------------
SHIP bagging for a horizon iff ALL of:
  1. mean paired pinball-loss improvement >= 0.5% (relative), AND
  2. bagging wins on pinball in >= half the paired folds, AND
  3. directional accuracy (sign of predicted return) does not regress by
     more than 0.5pp.
Otherwise keep GOSS for that horizon.

Pinball loss is primary because it is the loss q50 actually optimizes. DA is a
no-regression guard only: since 2026-07-24 the served up/flat/down signal
comes from the directional classifier, not q50, so this change is not expected
to move DA. MAE is reported as the served point-forecast metric.

Tree params come from the persisted production `meta.json:tuned_params` for
each horizon's q50, so both arms use the exact params production trains with.
Nothing here writes model constants or artifacts.

Usage:
    python -m scripts.ab_test_q50_sampling [--max-items 200] [--horizon 7]
                                            [--max-folds 8] [--single-fold]
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from database import SessionLocal
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import fold_level_records
from models.forecaster import (
    ItemForecaster,
    embargo_days,
    phase_collapsed_sql_filter,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_q50_sampling")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Production's item universe, spelled into every archive read this harness
# makes. Before 2026-08-08 the `ab_test_*` family globbed the Parquet privately
# and saw a universe production does not train on, so an A/B advised a model it
# had not measured. The bid sources need no clause here: `STEAMCOMMUNITY` is a
# single ask feed and already excludes them. See `models/item_parser.py`.
_UNIVERSE = phase_collapsed_sql_filter()

META_PATH = Path(__file__).parent.parent / "models" / "saved_models" / "meta.json"

HORIZONS = [3, 7, 14, 30]
QUANTILE = 0.5
ARMS = ["goss", "bagging"]

MIN_TRAIN_ROWS = 2000
MIN_VAL_ROWS = 200
MAX_FOLDS = 8

# Gate thresholds (pre-registered)
GATE_MIN_REL_PINBALL_GAIN = 0.005   # 0.5% relative
GATE_MAX_DA_REGRESSION_PP = 0.5     # percentage points

# Production ensemble uses feature_fraction 0.6/0.7/0.8 across members; the
# A/B trains one member at the middle value so the arms differ only in
# sampling strategy.
SINGLE_MEMBER_FEATURE_FRACTION = 0.7

# Boosting rounds per model. Was `DART_NUM_BOOST_ROUND if dart else 1000`;
# DART is gone from the forecaster, so only the GBDT arm of that branch
# survives.
NUM_BOOST_ROUND = 1000


def pinball_loss(y_true, y_pred, alpha=QUANTILE):
    """Mean pinball (quantile) loss — the objective q50 is trained on."""
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def directional_accuracy(y_true, y_pred):
    """Fraction of rows where predicted return sign matches realized sign.
    Zero-return rows are excluded (no sign to predict)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float((np.sign(yp[mask]) == np.sign(yt[mask])).mean())


def load_features(con, forecaster, events_df, max_items):
    """Mirrors ab_test_direction_labels.load_features / ab_test_hp_search so
    this A/B trains on the same data path as the other harnesses."""
    pq_files = sorted(str(p) for p in ARCHIVE_DIR.glob("prices-*.parquet"))
    pq_queries = []
    for pqf in pq_files:
        cols = {r[0] for r in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()}
        if "source" in cols:
            # `source = 'STEAMCOMMUNITY'` matches 0 rows post archive-rebuild, so
            # this degenerated to the NULL (pre-2026) branch; the dead disjunct is
            # dropped, keeping the intended pre-2026 cohort. See 2026-08-13 repin.
            pq_queries.append(
                f"SELECT item_slug, day, mean_price, volume FROM "
                f"read_parquet('{pqf}') WHERE source IS NULL "
                f"AND {_UNIVERSE}")
        else:
            pq_queries.append(
                f"SELECT item_slug, day, mean_price, volume FROM "
                f"read_parquet('{pqf}') WHERE {_UNIVERSE}")
    union_sql = " UNION ALL BY NAME ".join(pq_queries)

    items = con.sql(f"""
        SELECT item_slug, COUNT(*) AS row_count
        FROM ({union_sql})
        GROUP BY item_slug HAVING row_count >= 90
        ORDER BY row_count DESC, item_slug LIMIT {max_items}
    """).fetchall()
    if not items:
        raise RuntimeError(
            "q50_sampling universe query selected 0 items — the source pin "
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


def load_production_q50_params(horizon):
    """Production tuned params for this horizon's q50, from the persisted
    meta.json — the same dict the warm-retrain path feeds to lgb.train."""
    with open(META_PATH) as fh:
        meta = json.load(fh)
    tp = meta.get("tuned_params", {})
    src = tp.get(str(horizon)) or tp.get(horizon) or {}
    q = src.get("0.5") or src.get(0.5)
    if not q:
        raise RuntimeError(
            f"no persisted q50 params for horizon {horizon} in {META_PATH}")
    return dict(q)


def build_arm_params(base, arm, boosting_type, forecaster):
    """Production params with only the row-sampling strategy swapped."""
    p = dict(base)
    p["objective"] = "quantile"
    p["alpha"] = QUANTILE
    p["metric"] = "quantile"
    p["boosting_type"] = boosting_type
    p["max_bin"] = forecaster.MAX_BIN
    p["feature_pre_filter"] = False
    p["feature_fraction"] = SINGLE_MEMBER_FEATURE_FRACTION
    p["verbosity"] = -1
    p["device"] = "cpu"
    # Pin per-process thread count when sharding across processes, so N
    # concurrent shards don't each grab every core and thrash. Separate OS
    # processes (not a shared ThreadPoolExecutor) deliberately avoid the
    # nested-pool + libomp hang documented at forecaster.py:2384.
    if os.environ.get("AB_N_JOBS"):
        p["n_jobs"] = int(os.environ["AB_N_JOBS"])
    for k in ("data_sample_strategy", "top_rate", "other_rate",
              "subsample", "bagging_fraction", "bagging_freq"):
        p.pop(k, None)
    if arm == "goss":
        p["data_sample_strategy"] = "goss"
        p["top_rate"] = 0.2
        p["other_rate"] = 0.1
    else:
        p["data_sample_strategy"] = "bagging"
        p["subsample"] = 0.8
        p["bagging_freq"] = 1
    return p


def _run_horizon(fc, tdf, feat_cols, horizon, max_folds,
                 fold_start=0, fold_count=None):
    """Train both arms on identical folds; yield one record per (arm, fold).

    `fold_start`/`fold_count` shard the fold list across processes. Fold
    indices stay GLOBAL (offset by fold_start) so per-shard CSVs merge without
    key collisions, and both arms of a given fold always run in the same shard
    so the paired comparison is never split across processes.
    """
    target_col = f"target_return_{horizon}d"
    dates = np.array(sorted(tdf["date"].unique()))
    folds = fc._compute_cv_splits(dates, purge_days=embargo_days(horizon))
    boosting_type = fc.BOOSTING_TYPE
    if not folds:
        logger.warning(f"  no folds for {horizon}d after purge gap")
        return
    folds = folds[-max_folds:]
    n_total = len(folds)
    end = n_total if fold_count is None else fold_start + fold_count
    sharded = list(enumerate(folds))[fold_start:end]
    logger.info(f"  {horizon}d: {n_total} folds total, running "
                f"{len(sharded)} (global idx {fold_start}..{end - 1}), "
                f"boosting={boosting_type}")

    base = load_production_q50_params(horizon)
    # Production's per-horizon table, not a 1000-round cap for early
    # stopping to cut down. `best_iter` below is therefore the fixed count
    # unless EARLY_STOPPING=1, so the artifact-size diagnostic in the module
    # docstring ('7d q50 saves 1-2 trees') only reproduces under that flag.
    nbr = ItemForecaster._boost_rounds(horizon, cv=True)

    for fold_idx, (train_dates, val_dates) in sharded:
        tr = tdf[tdf["date"].isin(train_dates)]
        va = tdf[tdf["date"].isin(val_dates)]
        if len(tr) < MIN_TRAIN_ROWS or len(va) < MIN_VAL_ROWS:
            logger.info(f"    fold {fold_idx}: skipped "
                        f"({len(tr)} train, {len(va)} val)")
            continue

        X_tr = tr[feat_cols].replace([np.inf, -np.inf], np.nan)
        med = X_tr.median()
        X_tr = X_tr.fillna(med)
        X_va = va[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(med)
        y_tr = tr[target_col].to_numpy(dtype=float)
        y_va = va[target_col].to_numpy(dtype=float)

        # Production weighting. Under EARLY_STOPPING=1 the weighted val
        # metric is what stopping reads; by default nothing reads it.
        w_tr = fc._compute_sample_weights(tr, horizon)
        w_va = fc._compute_sample_weights(va, horizon)

        for arm in ARMS:
            params = build_arm_params(base, arm, boosting_type, fc)
            ds_params = {"max_bin": fc.MAX_BIN, "feature_pre_filter": False}
            dtrain = lgb.Dataset(X_tr, y_tr, params=ds_params,
                                 **({"weight": w_tr} if w_tr is not None else {}))
            dval = lgb.Dataset(X_va, y_va, reference=dtrain, params=ds_params,
                               **({"weight": w_va} if w_va is not None else {}))
            t0 = time.time()
            model = ItemForecaster._train_ensemble_member(
                params, dtrain, dval, num_boost_round=nbr,
                early_stopping=ItemForecaster._early_stopping_enabled())
            fit_s = time.time() - t0

            pred = model.predict(X_va, num_iteration=model.best_iteration or None)
            rec = {
                "horizon": horizon,
                "arm": arm,
                "fold": fold_idx,
                "pinball": pinball_loss(y_va, pred),
                "mae": float(np.mean(np.abs(y_va - pred))),
                "da": directional_accuracy(y_va, pred),
                "trees": int(model.num_trees()),
                "best_iter": int(model.best_iteration or model.num_trees()),
                "fit_s": round(fit_s, 1),
                "n_train": len(tr),
                "n_val": len(va),
            }
            logger.info(
                f"    fold {fold_idx} {arm:>7}: pinball={rec['pinball']:.5f} "
                f"mae={rec['mae']:.4f} da={100*rec['da']:.2f}% "
                f"trees={rec['trees']} ({fit_s:.1f}s)")
            yield rec


def summarize(records):
    """Paired per-fold comparison + pre-registered gate verdict per horizon."""
    df = pd.DataFrame(records)
    if df.empty:
        print("\nno results")
        return df, {}

    print("\n" + "=" * 78)
    print("PAIRED RESULTS (bagging vs goss, same folds)")
    print("=" * 78)
    print(f"{'h':>3} {'arm':>8} {'pinball':>9} {'mae':>8} {'da%':>7} "
          f"{'trees':>7} {'folds':>6} {'fit_s':>7}")

    verdicts = {}
    for horizon in sorted(df["horizon"].unique()):
        hd = df[df["horizon"] == horizon]
        for arm in ARMS:
            a = hd[hd["arm"] == arm]
            if a.empty:
                continue
            print(f"{horizon:>3} {arm:>8} {a['pinball'].mean():>9.5f} "
                  f"{a['mae'].mean():>8.4f} {100*a['da'].mean():>7.2f} "
                  f"{a['trees'].mean():>7.1f} {len(a):>6} "
                  f"{a['fit_s'].sum():>7.1f}")

        g = hd[hd["arm"] == "goss"].set_index("fold")
        b = hd[hd["arm"] == "bagging"].set_index("fold")
        common = sorted(set(g.index) & set(b.index))
        if not common:
            continue
        g, b = g.loc[common], b.loc[common]

        rel_gain = ((g["pinball"] - b["pinball"]) / g["pinball"]).mean()
        folds_won = int((b["pinball"] < g["pinball"]).sum())
        da_delta_pp = 100 * (b["da"].mean() - g["da"].mean())
        mae_delta = b["mae"].mean() - g["mae"].mean()

        c1 = rel_gain >= GATE_MIN_REL_PINBALL_GAIN
        # Fold-clustered paired interval on the pinball difference, replacing
        # the "wins on at least half the folds" condition this gate used until
        # 2026-08-08. A win count is not a test: two arms that differ only by
        # seed clear it half the time, and at 8 folds the binomial makes even
        # 6/8 unremarkable. Lower pinball is better, so a SHIP needs the
        # interval strictly below zero.
        paired = paired_arm_contrasts(
            {"goss": fold_level_records(common, g["pinball"], metric="pinball"),
             "bagging": fold_level_records(common, b["pinball"], metric="pinball")},
            base="goss", value_key="pinball", scale=1.0,
            higher_is_better=False)["bagging"]

        c2 = paired["verdict"] == "positive"
        c3 = da_delta_pp >= -GATE_MAX_DA_REGRESSION_PP
        ship = bool(c1 and c2 and c3)
        verdicts[horizon] = {
            "ship": ship, "rel_pinball_gain": rel_gain,
            "folds_won": folds_won, "n_folds": len(common),
            "da_delta_pp": da_delta_pp, "mae_delta": mae_delta,
            "paired_pinball": paired,
            "gate": {"pinball_gain": c1, "paired_interval": c2, "da_no_regress": c3},
        }
        print(f"  -> {horizon}d: pinball {rel_gain*100:+.2f}% "
              f"(gate >= +{GATE_MIN_REL_PINBALL_GAIN*100:.1f}%) [{'PASS' if c1 else 'FAIL'}] | "
              f"paired {format_paired(paired, unit='')} [{'PASS' if c2 else 'FAIL'}] | "
              f"DA {da_delta_pp:+.2f}pp (gate >= -{GATE_MAX_DA_REGRESSION_PP}pp) "
              f"[{'PASS' if c3 else 'FAIL'}] | MAE {mae_delta:+.4f} "
              f"| folds won {folds_won}/{len(common)} (context, not a gate)")
        print(f"     VERDICT {horizon}d: {'SHIP bagging' if ship else 'KEEP goss'}")

    return df, verdicts


def run(max_items, horizon_filter, max_folds, fold_start=0, fold_count=None,
        feature_cache=None):
    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    events_df = forecaster.fetch_events()
    db.close()

    # The engineered matrix is a pure function of (archive contents, max_items),
    # so cache it: reruns and sibling shards skip the ~30s rebuild. Cache key
    # includes max_items; delete the file to force a rebuild.
    cache_ok = False
    if feature_cache:
        cpath = Path(f"{feature_cache}.items{max_items}.parquet")
        ccols = Path(f"{feature_cache}.items{max_items}.cols.json")
        if cpath.exists() and ccols.exists():
            t0 = time.time()
            df = pd.read_parquet(cpath)
            feat_cols = json.loads(ccols.read_text())
            logger.info(f"  loaded cached features {cpath.name} "
                        f"({time.time()-t0:.0f}s, {df.shape})")
            cache_ok = True

    if not cache_ok:
        import duckdb
        con = duckdb.connect()
        try:
            t0 = time.time()
            df, feat_cols = load_features(con, forecaster, events_df, max_items)
            logger.info(f"  feature build took {time.time()-t0:.0f}s")
        finally:
            con.close()
        if feature_cache:
            df.to_parquet(cpath)
            ccols.write_text(json.dumps(list(feat_cols)))
            logger.info(f"  wrote feature cache {cpath.name}")

    # Match production: price_technicals only.
    feat_cols = forecaster._apply_feature_allowlist(
        feat_cols, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
    feat_cols = [c for c in feat_cols if c in df.columns]
    logger.info(f"  {len(feat_cols)} features after production allowlist")

    horizons = [h for h in HORIZONS
                if horizon_filter is None or h == horizon_filter]

    records = []
    for horizon in horizons:
        target_col = f"target_return_{horizon}d"
        tdf = forecaster.prepare_targets(df, horizon)
        tdf = tdf.dropna(subset=[target_col]).sort_values(
            ["item_id", "date"]).copy()
        if tdf.empty:
            logger.warning(f"  no targets for {horizon}d")
            continue
        for rec in _run_horizon(forecaster, tdf, feat_cols, horizon, max_folds,
                                fold_start=fold_start, fold_count=fold_count):
            records.append(rec)

    return summarize(records)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="A/B q50 row sampling: GOSS (production) vs bagging")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=None, choices=HORIZONS)
    ap.add_argument("--max-folds", type=int, default=MAX_FOLDS)
    ap.add_argument("--single-fold", action="store_true",
                    help="1 fold only — for timing calibration")
    ap.add_argument("--out", type=str, default=None,
                    help="write per-fold records to this CSV")
    ap.add_argument("--fold-start", type=int, default=0,
                    help="shard: first (global) fold index to run")
    ap.add_argument("--fold-count", type=int, default=None,
                    help="shard: number of folds to run from --fold-start")
    ap.add_argument("--feature-cache", type=str, default=None,
                    help="path prefix for caching the engineered feature "
                         "matrix across runs/shards (skips the ~30s rebuild)")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge per-shard CSVs and print the combined gate "
                         "verdict; skips all training")
    args = ap.parse_args()

    # Merge mode: recombine shard CSVs and apply the gate over all folds.
    if args.merge:
        frames = [pd.read_csv(p) for p in args.merge]
        allrecs = pd.concat(frames, ignore_index=True)
        dupes = allrecs.duplicated(subset=["horizon", "arm", "fold"]).sum()
        if dupes:
            logger.warning(f"{dupes} duplicate (horizon,arm,fold) rows — "
                           f"check shard fold ranges for overlap")
        logger.info(f"merged {len(allrecs)} records from {len(args.merge)} files")
        summarize(allrecs.to_dict("records"))
        return 0

    max_folds = 1 if args.single_fold else args.max_folds

    logger.info("=" * 70)
    logger.info("A/B: q50 row-sampling strategy (goss vs bagging)")
    logger.info(f"  max_items={args.max_items} horizon={args.horizon} "
                f"max_folds={max_folds}")
    logger.info("=" * 70)

    t0 = time.time()
    df, verdicts = run(args.max_items, args.horizon, max_folds,
                       fold_start=args.fold_start, fold_count=args.fold_count,
                       feature_cache=args.feature_cache)
    logger.info(f"total wall clock: {time.time()-t0:.0f}s")

    if args.out and not df.empty:
        df.to_csv(args.out, index=False)
        logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
