#!/usr/bin/env python3
"""A/B test: does ByMykel/CSGO-API item metadata add directional accuracy?

`docs/changelog/2026-08-06-data-acquisition-ranking.md` puts `ByMykel/CSGO-API`
in Tier 2 on the argument that **item age is not inferable from a truncated
price history**, and that rarity "is the one metadata family that ever measured
causal here (+10-12pp within the model)". Two facts make that worth re-testing
rather than believing:

1. The +10-12pp rarity figure (`docs/research/accuracy-opportunities.md:141`)
   predates `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`. `_feature_group`
   routes `rarity_*` to `item_identity`, `type_*` to `item_metadata` and
   `item_age*` to `temporal` -- all three are discarded before training today.
   So nothing in this family currently reaches the model at all.
2. The allowlist exists because 85 non-price features measured within fold
   noise. But that test used `item-metadata.parquet`, which carries rarity NULL
   on 4,296 of 8,691 rows and has no item age, crate or collection at all. This
   arm supplies rarity on 99.5% of the training cohort and item age on 88.5%.

So the question is whether *better-populated* metadata clears the bar the old
metadata failed. Four arms:

    baseline:   production's 33 price_technicals columns, nothing added
    treatment:  + every metadata column
    age_only:   + item_age_days and rarity_meta_rank only. Item age is the one
                field with a mechanism argument, and the volume A/B measured
                that uninformative columns actively DILUTE (-1.27/-1.56pp at
                7d/14d from 11 noise columns at feature_fraction=0.7). Without
                this arm a null on `treatment` cannot separate "age is useless"
                from "age helped and the other eight columns ate the gain".
    placebo:    + every metadata column, permuted per fold in train and val.
                Guards capacity inflation, which is a live risk here: crate and
                collection are categorical with ~110-300 levels over 870 items,
                so a tree can memorise per-item baselines through them.

Scored on TWO cohorts, because production does both:
  - held-out items, in no arm's training set -- does metadata generalise to an
    item the model has never seen?
  - trained items, time-separated -- production trains on most of what it
    serves, so this is the deployed situation.

## Embargo (fixed 2026-08-07)

Until 2026-08-07 the fold split here was `train = date < val_start` with **no purge
gap**, so every training row inside the last `horizon` days before the boundary carried
a `target_return_{h}d` resolved from inside the validation window. Production never had
that bug (`ItemForecaster._compute_cv_splits(..., purge_days=horizon)`), and this harness
family had simply diverged from it. The overlap is symmetric across arms, but only an arm
that can *locate* the overlapping rows exploits it, so date-level and date-proxy columns
banked leakage that per-item columns could not — measured on an event-calendar arm at
h=30 (25 folds, held-out >=$1, fold-clustered): **+12.1pp unpurged -> +6.1pp purged**.
`item_age_days` is exactly such a column here: it is (date - first sale date), a per-row
date proxy, so `age_only` and `treatment` were the exposed arms.

The train side is now embargoed by calling production's own
`ItemForecaster._purge_overlapping_train_rows(train, val_start, horizon)`; a second
implementation of the same rule is what caused the drift in the first place. Purging is
unconditional -- this is a research tool with no results to keep back-compatible. The
**val** side is never purged: at h=30 that would empty the 21-day window.

Metadata is joined from a table built off the ByMykel dumps; see
`--metadata-parquet`. Item age is computed per ROW as (observation date - first
sale date), never as a static "days since sale as of today" -- the latter is a
different feature and would leak the calendar into a per-item constant.

Usage:
    python scripts/ab_test_item_metadata.py --build-cache-only \
        --frame-cache /tmp/meta_frame.parquet --metadata-parquet /tmp/item_metadata_features.parquet
    python scripts/ab_test_item_metadata.py --frame-cache /tmp/meta_frame.parquet \
        --horizon 7 --out /tmp/meta_h7.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from backtest.paired_mde import paired_da_difference
from database import SessionLocal
from models.forecaster import ItemForecaster, phase_collapsed_sql_filter
from models.market_factor import (
    build_market_index,
    market_factor_for_horizon,
)

MIN_MEDIAN_PRICE = 1.0
MIN_ITEM_DAYS = 180
N_UNIVERSE = 870
N_EVAL_ITEMS = 150

# Training items. Held between narrow and wide on the breadth axis, which the
# companion breadth test measured as flat-to-positive there, so this arm is not
# sitting at a knowingly bad point on that axis while a different one is tested.
N_TRAIN_ITEMS = 500
ROW_BUDGET = 200_000

# Trained items also scored, to report the deployed situation alongside the
# held-out generalisation test. Disjoint from the eval hold-out by construction.
N_TRAINED_EVAL_ITEMS = 150

CORR_PRUNE_THRESHOLD = 0.95
VAL_WINDOW_DAYS = 21
STEP_DAYS = 60

# Must match ab_test_training_breadth.py so the held-out cohort is the same 150
# items in both experiments and the two results are directly comparable.
SPLIT_SEED = 20260806
SAMPLE_SEED = 4242
PLACEBO_SEED = 42

# The metadata columns under test. Named explicitly rather than discovered by
# prefix: the >0.95 correlation prune is data-dependent, and a column silently
# dropping out of the treatment arm would make the contrast unreadable.
META_ALL = (
    "item_age_days",
    "item_age_ambiguous",
    "rarity_meta_rank",
    "is_meta_stattrak",
    "is_meta_souvenir",
    "float_meta_min",
    "float_meta_max",
    "type_meta_crate_id",
    "type_meta_collection_id",
)
META_AGE_ONLY = ("item_age_days", "rarity_meta_rank")
# Every metadata column EXCEPT the age pair. These are static per item: they
# carry no date component at all, so an arm built from them cannot act as a
# calendar proxy.
META_STATIC = (
    "rarity_meta_rank",
    "is_meta_stattrak",
    "is_meta_souvenir",
    "float_meta_min",
    "float_meta_max",
    "type_meta_crate_id",
    "type_meta_collection_id",
)
# The leakage control. `item_age_days` is (observation date - first sale date):
# for a fixed item that is the calendar date plus a constant, and the folds are
# ordered in time, so it is a date proxy as much as it is an item property.
# `date_ordinal` is the date and NOTHING else -- no metadata, no item identity.
# If it reproduces the treatment arm's gain, the gain is the calendar, and the
# route by which the calendar pays is early stopping, which scores on the
# validation set. Computed at evaluation time, not stored in the frame.
DATE_PROXY_COL = "date_ordinal"
# Passed to LightGBM as categorical. An arbitrary integer code for a crate is
# not an ordinal, and letting trees split it as one would test a meaningless
# encoding rather than the identity.
META_CATEGORICAL = ("type_meta_crate_id", "type_meta_collection_id")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ab_test_item_metadata")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Production's item universe, spelled into every archive read this harness
# makes. Before 2026-08-08 the `ab_test_*` family globbed the Parquet privately
# and saw a universe production does not train on, so an A/B advised a model it
# had not measured. The bid sources need no clause here: `aggregator_sync` is a
# single ask feed and already excludes them. See `models/item_parser.py`.
_UNIVERSE = phase_collapsed_sql_filter()

DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}


def _frame_fingerprint(metadata_parquet):
    src = Path(__file__).parent.parent / "models" / "forecaster.py"
    h = hashlib.sha256(src.read_bytes())
    h.update(repr((MIN_MEDIAN_PRICE, MIN_ITEM_DAYS, N_UNIVERSE, CORR_PRUNE_THRESHOLD, META_ALL, _UNIVERSE)).encode())
    if metadata_parquet:
        h.update(hashlib.sha256(Path(metadata_parquet).read_bytes()).digest())
    return h.hexdigest()[:16]


def _archive_union_sql(con):
    """One UNION over the archive, restricted to the single continuous series.

    Identical to ab_test_training_breadth.py: 2026 files filtered to
    `aggregator_sync`, pre-2026 files taken whole. `volume` is selected only
    because engineer_features requires the column; every feature derived from
    it is shelved.
    """
    pq_files = sorted(str(p) for p in ARCHIVE_DIR.glob("prices-*.parquet"))
    queries = []
    for pqf in pq_files:
        cols = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()
        if "source" in {r[0] for r in cols}:
            queries.append(
                f"SELECT item_slug, day, mean_price, volume FROM "
                f"read_parquet('{pqf}') WHERE (source IS NULL OR source = 'aggregator_sync') AND {_UNIVERSE}"
            )
        else:
            queries.append(f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') WHERE {_UNIVERSE}")
    return " UNION ALL BY NAME ".join(queries)


def build_frame(metadata_parquet, cache_path=None):
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != _frame_fingerprint(metadata_parquet):
                raise SystemExit(
                    f"Frame cache {cache_path} was built from different feature "
                    f"code, constants or metadata. Rebuild with --build-cache-only."
                )
            df = pd.read_parquet(cache_path)
            logger.info(f"  Loaded cached frame {cache_path} ({len(df):,} rows)")
            return df, meta["pruned"], meta["meta_present"]

    df, pruned, meta_present = _build_frame_uncached(metadata_parquet)

    if cache_path is not None:
        tmp_frame = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
        tmp_meta = cache_path.with_suffix(f".{os.getpid()}.tmp.json")
        df.to_parquet(tmp_frame, index=False)
        tmp_meta.write_text(
            json.dumps(
                {
                    "fingerprint": _frame_fingerprint(metadata_parquet),
                    "pruned": pruned,
                    "meta_present": meta_present,
                    "rows": len(df),
                }
            )
        )
        os.replace(tmp_frame, cache_path)
        os.replace(tmp_meta, cache_path.with_suffix(".meta.json"))
        logger.info(f"  Wrote frame cache {cache_path} ({len(df):,} rows)")
    return df, pruned, meta_present


def _build_frame_uncached(metadata_parquet):
    import duckdb

    con = duckdb.connect()
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        union_sql = _archive_union_sql(con)
        rows = con.sql(f"""
            SELECT item_slug, COUNT(DISTINCT day) AS n_days
            FROM ({union_sql})
            GROUP BY item_slug
            HAVING COUNT(DISTINCT day) >= {MIN_ITEM_DAYS}
               AND MEDIAN(mean_price) >= {MIN_MEDIAN_PRICE}
               AND MIN(day) < DATE '2026-01-01'
            ORDER BY n_days DESC, item_slug
            LIMIT {N_UNIVERSE}
        """).fetchall()
        slugs = [r[0] for r in rows]
        logger.info(f"  Universe: {len(slugs)} deep >=$1 items")

        placeholders = ", ".join("?" for _ in slugs)
        all_prices = con.sql(
            f"""
            SELECT item_slug AS item_id, day AS timestamp,
                   mean_price AS price, volume
            FROM ({union_sql}) WHERE item_slug IN ({placeholders})
        """,
            params=slugs,
        ).df()

        all_prices["timestamp"] = pd.to_datetime(all_prices["timestamp"])
        all_prices["date"] = all_prices["timestamp"].dt.date
        all_prices["item_id"] = pd.Categorical(all_prices["item_id"], categories=slugs, ordered=True)
        all_prices = all_prices.sort_values(["item_id", "timestamp"], kind="stable").reset_index(drop=True)
        all_prices["item_id"] = all_prices["item_id"].astype(str)
        logger.info(f"  Loaded {len(all_prices):,} price rows")

        # Market index for the market-relative label arm. Built on the FULL
        # price frame before feature engineering, as build_training_data does:
        # the index is a per-date cross-sectional median over the >=$1 cohort
        # and thinning the item set would thin the median.
        market_index = build_market_index(all_prices)
        n_valid = int(market_index["valid"].sum()) if not market_index.empty else 0
        logger.info(f"  market index: {len(market_index):,} dates, {n_valid:,} valid")

        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)

        # One realized-factor column per horizon, joined on date. These are
        # LABEL components, never features -- `market_factor_*` is not in
        # FEATURE_GROUP_ALLOWLIST, and the explicit column list below keeps
        # them out of `pruned` regardless.
        mf_cols = []
        join_dates = pd.to_datetime(df["date"])
        for h in ItemForecaster.HORIZONS:
            col = f"market_factor_{h}d"
            factor = market_factor_for_horizon(market_index, h)
            df[col] = join_dates.map(factor).astype(float)
            cov = float(df[col].notna().mean()) * 100
            logger.info(f"  market factor {h}d coverage: {cov:.1f}% of rows")
            mf_cols.append(col)

        EXCLUDE = {"item_id", "date", "timestamp", "price", "volume", "name", "release_date"}
        numeric = (np.float64, np.float32, np.int64, int, float)
        all_cols = [c for c in df.columns if c not in EXCLUDE and df[c].dtype in numeric]
        kept = [c for c in all_cols if c not in ItemForecaster.SHELVED_FEATURES]
        kept = ItemForecaster._apply_feature_allowlist(kept, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
        logger.info(f"  Features: {len(all_cols)} -> {len(kept)} after shelving + allowlist")

        if len(kept) > 2:
            corr = df[kept].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                if col in to_drop:
                    continue
                to_drop.update(upper[col][upper[col] > CORR_PRUNE_THRESHOLD].index)
            pruned = [c for c in kept if c not in to_drop]
        else:
            pruned = kept
        logger.info(f"  {len(kept)} -> {len(pruned)} after corr prune")

        df = df[["item_id", "date", "price", *pruned, *mf_cols]].copy()
        meta_present = _join_metadata(df, metadata_parquet)
        return df, pruned, meta_present
    finally:
        con.close()


def _join_metadata(df, metadata_parquet):
    """Left-join the metadata table and turn first-sale date into per-row age.

    Mutates `df` in place and returns the metadata columns that actually landed
    with some non-null content. A column that is entirely null after the join
    is dropped rather than carried: LightGBM would treat it as a constant, and
    it would pad the treatment arm's width without carrying information, which
    is precisely the dilution the placebo arm is meant to isolate.
    """
    if not metadata_parquet:
        raise SystemExit("--metadata-parquet is required to build the frame")
    meta = pd.read_parquet(metadata_parquet)
    if "item_slug" not in meta.columns:
        raise SystemExit(f"{metadata_parquet} has no item_slug column")
    logger.info(f"  Metadata table: {len(meta):,} rows, {len(meta.columns)} columns")

    meta = meta.drop_duplicates(subset=["item_slug"])
    date_col = "item_age_first_sale_date"
    first_sale = None
    if date_col in meta.columns:
        first_sale = pd.to_datetime(meta[date_col], errors="coerce")
        meta = meta.drop(columns=[date_col])

    meta = meta.set_index("item_slug")
    if first_sale is not None:
        meta["_first_sale"] = first_sale.values

    matched = df["item_id"].isin(meta.index)
    logger.info(f"  Joined {matched.sum():,}/{len(df):,} rows ({df.loc[matched, 'item_id'].nunique()} items matched)")

    aligned = meta.reindex(df["item_id"].to_numpy())
    if "_first_sale" in aligned.columns:
        obs = pd.to_datetime(pd.Series(df["date"].to_numpy()))
        fs = pd.Series(aligned["_first_sale"].to_numpy())
        age = (obs - fs).dt.days.to_numpy(dtype=float)
        # A negative age means the item was trading before the date the dump
        # claims it first went on sale -- the metadata is wrong for that item,
        # not that the item is "very new". Null it rather than clip to 0, which
        # would fabricate a real-looking value at a meaningful boundary.
        n_neg = int(np.nansum(age < 0))
        age[age < 0] = np.nan
        df["item_age_days"] = age
        logger.info(
            f"  item_age_days: {np.isfinite(age).sum():,}/{len(age):,} rows non-null ({n_neg:,} negative -> null)"
        )
        aligned = aligned.drop(columns=["_first_sale"])

    for col in aligned.columns:
        if col in df.columns:
            continue
        df[col] = pd.to_numeric(aligned[col].to_numpy(), errors="coerce")

    present = []
    for col in META_ALL:
        if col not in df.columns:
            logger.warning(f"  metadata column {col} absent from the table")
            continue
        n = int(df[col].notna().sum())
        if n == 0:
            logger.warning(f"  metadata column {col} is entirely null — dropped")
            df.drop(columns=[col], inplace=True)
            continue
        logger.info(f"  {col:26s} {n:>10,}/{len(df):,} non-null ({100.0 * n / len(df):.1f}%)")
        present.append(col)
    return present


def assign_items(df):
    """Split into held-out eval items, training items, and a trained-eval slice.

    The permutation is seeded identically to ab_test_training_breadth.py, so
    the 150 held-out items are the same in both experiments.
    """
    slugs = list(pd.unique(df["item_id"]))
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = [slugs[i] for i in rng.permutation(len(slugs))]

    eval_items = sorted(shuffled[:N_EVAL_ITEMS])
    pool = shuffled[N_EVAL_ITEMS:]
    if len(pool) < N_TRAIN_ITEMS:
        raise SystemExit(f"need {N_TRAIN_ITEMS} training items, pool has {len(pool)}")
    train_items = sorted(pool[:N_TRAIN_ITEMS])
    trained_eval = sorted(train_items[:N_TRAINED_EVAL_ITEMS])

    logger.info(f"  Held-out eval items: {len(eval_items)}")
    logger.info(f"  Training items:      {len(train_items)}")
    logger.info(f"  Trained-eval slice:  {len(trained_eval)} (subset of training)")
    assert not (set(eval_items) & set(train_items))
    return eval_items, train_items, trained_eval


def _stratified_sample(train_df, items, budget, fold_idx):
    """Up to `budget` rows spread evenly over `items`, uniform within each.

    Seed depends on the fold, not the arm, so every arm draws the same rows.
    The arms must differ by feature set alone.
    """
    sub = train_df[train_df["item_id"].isin(items)]
    if budget is None or len(sub) <= budget:
        return sub
    per_item = max(1, budget // len(items))
    rng = np.random.default_rng(SAMPLE_SEED + fold_idx)
    keep = []
    for _, idx in sub.groupby("item_id", sort=True).indices.items():
        keep.append(idx if len(idx) <= per_item else rng.choice(idx, size=per_item, replace=False))
    picked = np.concatenate(keep)
    picked.sort()
    return sub.iloc[picked]


def run_evaluation(df, pruned, meta_present, horizon_filter=None, n_jobs=None, market_relative=False):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    eval_items, train_items, trained_eval = assign_items(df)

    meta_all = [c for c in META_ALL if c in meta_present]
    meta_age = [c for c in META_AGE_ONLY if c in meta_present]
    arms = {
        "baseline": [],
        "treatment": meta_all,
        "age_only": meta_age,
        "placebo": meta_all,
        "static_only": [c for c in META_STATIC if c in meta_present],
        "date_proxy": [DATE_PROXY_COL],
    }
    for name, extra in arms.items():
        logger.info(
            f"  Arm {name:12s}: {len(pruned) + len(extra)} features ({len(pruned)} price + {len(extra)} metadata)"
        )

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        horizons = ItemForecaster.HORIZONS
        if horizon_filter is not None:
            horizons = [h for h in horizons if h == horizon_filter]

        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Evaluating {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            target_col = f"target_return_{horizon}d"
            tdf = tdf.dropna(subset=[target_col]).copy()
            tdf = tdf.sort_values(["item_id", "date"])
            if tdf.empty:
                logger.warning(f"    No valid targets for {horizon}d")
                continue

            base_cols = [c for c in pruned if c in tdf.columns]
            keep_cols = ["item_id", "date", "price", target_col, *base_cols, *meta_all]
            mf_col = f"market_factor_{horizon}d"
            if market_relative and mf_col not in tdf.columns:
                raise SystemExit(
                    f"--market-relative needs {mf_col} in the frame; rebuild the cache with --build-cache-only."
                )
            if market_relative:
                keep_cols.append(mf_col)
            sub = tdf[[c for c in keep_cols if c in tdf.columns]].copy()
            sub[DATE_PROXY_COL] = pd.to_datetime(sub["date"]).map(lambda d: d.toordinal()).astype(float)

            dates = sorted(sub["date"].unique())
            split_idx = len(dates) * 2 // 3
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            is_heldout = sub["item_id"].isin(set(eval_items)).to_numpy()
            is_trained_eval = sub["item_id"].isin(set(trained_eval)).to_numpy()
            is_train_item = sub["item_id"].isin(set(train_items)).to_numpy()

            results[horizon] = {}
            for arm, extra in arms.items():
                features = base_cols + extra
                cats = [c for c in extra if c in META_CATEGORICAL]
                logger.info(f"\n    --- {arm} ({len(features)} features, {len(cats)} categorical) ---")

                rec = {"heldout": [], "trained": []}
                per_fold = []
                for fold_idx, window_end in enumerate(range(split_idx + 1, len(dates), STEP_DAYS)):
                    val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                    if len(val_dates) < 7:
                        continue
                    in_train = sub_days <= dates_dt[window_end - 1]
                    in_val = (sub_days >= dates_dt[window_end]) & (
                        sub_days <= dates_dt[window_end + len(val_dates) - 1]
                    )

                    # Embargo the TRAIN side only, via production's own purge
                    # (see the "Embargo" section of the module docstring).
                    # The val mask above is untouched: purging it would shrink
                    # the 21-day window and empty it outright at 30d.
                    train_df = ItemForecaster._purge_overlapping_train_rows(
                        sub[in_train & is_train_item], val_dates[0], horizon
                    )
                    # Both scoring cohorts come out of one val slice, so the
                    # two reads share a model and a fold and stay comparable.
                    val_df = sub[in_val & (is_heldout | is_trained_eval)]
                    if len(val_df) < 50 or train_df.empty:
                        continue
                    train_df = _stratified_sample(train_df, train_items, ROW_BUDGET, fold_idx)

                    if arm == "placebo" and extra:
                        rng = np.random.default_rng(PLACEBO_SEED)
                        train_df = train_df.copy()
                        val_df = val_df.copy()
                        for col in extra:
                            train_df[col] = rng.permutation(train_df[col].values)
                            val_df[col] = rng.permutation(val_df[col].values)

                    train_median = train_df[features].median()
                    X_train = train_df[features].fillna(train_median)
                    X_val = val_df[features].fillna(train_median)
                    # LightGBM requires non-negative integer codes for a
                    # categorical column; the fill above can leave NaN when a
                    # column is all-null inside a fold.
                    for c in cats:
                        X_train[c] = X_train[c].fillna(-1).astype(int).clip(lower=-1)
                        X_val[c] = X_val[c].fillna(-1).astype(int).clip(lower=-1)

                    y_train = train_df[target_col].values
                    y_val = val_df[target_col].values
                    if market_relative:
                        # e = r - m, exactly as the production directional
                        # classifier does it (forecaster.py::_demean_returns).
                        # A missing factor demeans by zero rather than dropping
                        # the row: the arms must see identical rows for the
                        # paired comparison to hold.
                        y_train = ItemForecaster._demean_returns(y_train, train_df[mf_col].to_numpy())
                        y_val = ItemForecaster._demean_returns(y_val, val_df[mf_col].to_numpy())
                    ds_kw = {"params": DS_PARAMS, "free_raw_data": False}
                    if cats:
                        ds_kw["categorical_feature"] = cats
                    dtrain = lgb.Dataset(X_train, y_train, **ds_kw)
                    dval = lgb.Dataset(X_val, y_val, reference=dtrain, **ds_kw)
                    params = {
                        "objective": "quantile",
                        "alpha": 0.5,
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
                        "force_row_wise": True,
                        **DS_PARAMS,
                    }
                    # Early stopping scored on `dval`, i.e. on the rows being
                    # measured. That is a leak every arm shares -- except that
                    # an arm holding a date proxy can split the validation
                    # window out on its own, so the leak is worth MORE to it
                    # than to baseline. It is now off by default: production's
                    # trainer and round table, with `dval` ignored unless
                    # EARLY_STOPPING=1 reinstates the old arm for a paired read.
                    model = ItemForecaster._train_ensemble_member(
                        params,
                        dtrain,
                        dval,
                        num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                        early_stopping=ItemForecaster._early_stopping_enabled(),
                    )
                    pred = model.predict(X_val)

                    price = val_df["price"].to_numpy(dtype=float)
                    actual = np.asarray(y_val, dtype=float)
                    match = np.sign(np.nan_to_num(actual)) == np.sign(np.nan_to_num(pred))
                    # Strict >=$1: flat-actual rows excluded, because sign(0)
                    # matched by a q50 emitting 0.0 is a free hit, not a call.
                    scored = (actual != 0) & (price >= 1.0)
                    ids = val_df["item_id"].to_numpy()
                    dts = val_df["date"].to_numpy()
                    held = np.isin(ids, eval_items)

                    fold_row = {
                        "fold": len(per_fold) + 1,
                        "val_start": str(val_dates[0]),
                        "val_end": str(val_dates[-1]),
                        "n_train": len(train_df),
                    }
                    for cohort, mask in (("heldout", held), ("trained", ~held)):
                        sel = scored & mask
                        for i in np.flatnonzero(sel):
                            rec[cohort].append(
                                {
                                    "item_id": ids[i],
                                    "forecast_date": str(dts[i]),
                                    # The resampling cluster. Dates inside one
                                    # validation window share a fitted model, so
                                    # clustering on them understates the variance
                                    # -- see paired_mde's module docstring, which
                                    # names this experiment as one to re-derive.
                                    "fold_id": fold_idx,
                                    "direction_correct": bool(match[i]),
                                }
                            )
                        tot = int(sel.sum())
                        hits = int(np.count_nonzero(match & sel))
                        fold_row[f"{cohort}_n"] = tot
                        fold_row[f"{cohort}_da"] = round(hits / tot * 100, 2) if tot else None
                    per_fold.append(fold_row)

                if not per_fold:
                    logger.warning(f"    {arm}: no usable folds")
                    continue

                entry = {
                    "n_features": len(features),
                    "fold_count": len(per_fold),
                    "mean_rows_per_fold": int(np.mean([f["n_train"] for f in per_fold])),
                    "per_fold": per_fold,
                    "records": rec,
                }
                for cohort in ("heldout", "trained"):
                    n = len(rec[cohort])
                    hits = sum(int(r["direction_correct"]) for r in rec[cohort])
                    entry[f"da_{cohort}"] = round(hits / n * 100, 2) if n else None
                    entry[f"n_{cohort}"] = n
                results[horizon][arm] = entry
                logger.info(
                    f"      held-out DA={entry['da_heldout']}% "
                    f"(n={entry['n_heldout']:,})   "
                    f"trained DA={entry['da_trained']}% "
                    f"(n={entry['n_trained']:,})   folds={len(per_fold)}"
                )

            base = results[horizon].get("baseline")
            if base:
                results[horizon]["_paired"] = {}
                for arm in ("treatment", "age_only", "placebo", "static_only", "date_proxy"):
                    if arm not in results[horizon]:
                        continue
                    results[horizon]["_paired"][arm] = {
                        cohort: paired_da_difference(
                            base["records"][cohort], results[horizon][arm]["records"][cohort], cluster_key="fold_id"
                        )
                        for cohort in ("heldout", "trained")
                    }
                # treatment vs placebo is the arm that rules out capacity
                # inflation: both carry the same 9 extra columns, so a
                # difference cannot be model width.
                if "treatment" in results[horizon] and "placebo" in results[horizon]:
                    results[horizon]["_paired"]["treatment_vs_placebo"] = {
                        cohort: paired_da_difference(
                            results[horizon]["placebo"]["records"][cohort],
                            results[horizon]["treatment"]["records"][cohort],
                            cluster_key="fold_id",
                        )
                        for cohort in ("heldout", "trained")
                    }
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 100)
    print("ITEM METADATA A/B — DA(strict, >=$1), held-out and trained cohorts")
    print("=" * 100)
    for h in sorted(k for k in results if isinstance(k, int)):
        r = results[h]
        print(f"\n  {h}d horizon")
        print(f"    {'arm':<12} {'feats':>6} {'held-out DA':>12} {'n':>9} {'trained DA':>12} {'n':>9} {'folds':>6}")
        print(f"    {'-' * 72}")
        for arm in ("baseline", "treatment", "age_only", "placebo", "static_only", "date_proxy"):
            a = r.get(arm)
            if not a:
                continue
            print(
                f"    {arm:<12} {a['n_features']:>6} "
                f"{a['da_heldout']:>11.2f}% {a['n_heldout']:>9,} "
                f"{a['da_trained']:>11.2f}% {a['n_trained']:>9,} "
                f"{a['fold_count']:>6}"
            )
        for label, block in r.get("_paired", {}).items():
            print(f"\n    {label} (paired, dates clustered):")
            for cohort, p in block.items():
                ci = (
                    f"[{p['ci_lower_pp']:+.2f}, {p['ci_upper_pp']:+.2f}]" if p.get("ci_lower_pp") is not None else "n/a"
                )
                mde = f"{p['mde_pp']:.2f}" if p.get("mde_pp") is not None else "n/a"
                print(
                    f"      {cohort:<10} {p['mean_diff_pp']:+.2f}pp  95% CI {ci}"
                    f"  MDE {mde}pp  n={p['n_paired']:,}/{p['n_dates']} dates"
                )
    print("")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="A/B: ByMykel item metadata (age, rarity, crate, float)")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--metadata-parquet", default=None)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--market-relative",
        action="store_true",
        help="Demean the target by the realized market factor, "
        "as the production directional classifier does "
        "when market_relative_labels is on. DA then "
        "measures idiosyncratic direction, so absolute "
        "numbers are NOT comparable across this flag -- "
        "only the arm contrasts are.",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: ByMykel item metadata")
    logger.info("=" * 70)

    df, pruned, meta_present = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    if args.build_cache_only:
        logger.info("Frame cache built; exiting before evaluation.")
        return 0

    results = run_evaluation(
        df, pruned, meta_present, horizon_filter=args.horizon, n_jobs=args.n_jobs, market_relative=args.market_relative
    )
    if args.out:
        slim = {}
        for h, r in results.items():
            slim[h] = {}
            for arm, a in r.items():
                slim[h][arm] = (
                    {k: v for k, v in a.items() if k != "records"} if isinstance(a, dict) and "records" in a else a
                )
        Path(args.out).write_text(json.dumps(slim, indent=2, default=str))
        logger.info(f"Wrote {args.out}")
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
