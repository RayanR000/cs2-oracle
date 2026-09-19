#!/usr/bin/env python3
"""A/B test: does a CSFloat-vs-archive price basis add directional accuracy?

The motivating argument is in the 2026-08-06 acquisition ranking: every price
feature this project has is a repackaging of "what the market did that day", which
is why demeaning the label by the market factor drops accuracy below a constant call
(`2026-08-06-market-relative-labels-refuted.md`). A **cross-market basis** is the one
feature form that is structurally immune to that -- it is a difference between two
prices for ONE item at ONE instant, so the market term cancels by construction.

`backend/scripts/probe_csfloat_history.py` supplied the data and one warning. The
CSFloat daily `avg_price` has a within-item CV of 1.35 against the archive, falling
monotonically to 0.098 once you require >=50 sales that day (median is 4). The reading
is that `avg_price` is the mean of whichever specific float/pattern instances sold, so
on a typical day the composition term dominates the market term. The arms below are
built around that: they separate the raw daily basis from smoothed forms, so a null on
`basis_raw` with a positive on `basis_smooth` is interpretable rather than confusing.

## Arms

    baseline      production's 33 price_technicals features
    basis_raw     + cf_basis                 -- the naive daily spread, the noisy one
    basis_smooth  + cf_basis_7d, cf_basis_dev_30d, cf_basis_chg_7d
    basis_all     + every basis column and cf_log_sales_7d
    count_only    + cf_log_sales_7d only     -- separates "the spread helped" from
                                                "a liquidity proxy helped"
    placebo       + basis_all's columns, permuted per fold in train and val

`placebo` is not optional here. A bare date-ordinal column bought +11.21pp at 30d in
`2026-08-06-breadth-beats-depth-item-age-does-not.md` purely as a market proxy, so any
arm that widens the model needs a same-width control.

## Embargo (fixed 2026-08-07)

Until 2026-08-07 the fold split here was `train = date < val_start` with **no purge
gap**, so every training row inside the last `horizon` days before the boundary carried
a `target_return_{h}d` resolved from inside the validation window. Production never had
that bug (`ItemForecaster._compute_cv_splits(..., purge_days=horizon)`), and this harness
family had simply diverged from it. The overlap is symmetric across arms, but only an arm
that can *locate* the overlapping rows exploits it, so date-level and date-proxy columns
banked leakage that per-item columns could not — measured on an event-calendar arm at
h=30 (25 folds, held-out >=$1, fold-clustered): **+12.1pp unpurged -> +6.1pp purged**.

The train side is now embargoed by calling production's own
`ItemForecaster._purge_overlapping_train_rows(train, val_start, horizon)`; a second
implementation of the same rule is what caused the drift in the first place. Purging is
unconditional -- this is a research tool with no results to keep back-compatible. The
**val** side is never purged: at h=30 that would empty the 21-day window.

## No lookahead

Every CSFloat column is computed from data at t-1 and shifted forward, including the
archive leg of the basis: `cf_basis` at date t is `log(cf[t-1] / archive[t-1])`.
CSFloat's `avg_price` for day t averages sales *during* day t and is not complete when
production predicts, so using it at t would be a leak worth ruling out by construction
rather than by argument.

## Method, held identical to the sibling harnesses

Universe, fold geometry, row budget, metric, seeds and the significance test all match
`ab_test_item_metadata.py` and `ab_test_training_breadth.py`, so the numbers are
directly comparable to the metadata and breadth results. The one forced difference is
size: the universe is the deep >=$1 items that the probe actually pulled (271, against
those scripts' 870), because 500 requests/day is CSFloat's budget. That costs power,
and the MDE reported alongside every comparison is the thing to read before believing
any of it.

Usage:
    backend/venv/bin/python backend/scripts/ab_test_csfloat_basis.py \
        --probe-dir /tmp/csfloat_probe --frame-cache /tmp/cf_frame.parquet \
        --horizon 7 --out /tmp/cf_h7.json
    # all four horizons:
    backend/venv/bin/python backend/scripts/ab_test_csfloat_basis.py \
        --probe-dir /tmp/csfloat_probe --frame-cache /tmp/cf_frame.parquet \
        --out /tmp/cf_all.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from backtest.paired_mde import paired_da_difference
from database import SessionLocal
from models.forecaster import ItemForecaster, phase_collapsed_sql_filter
from models.market_factor import build_market_index, market_factor_for_horizon

MIN_MEDIAN_PRICE = 1.0
MIN_ITEM_DAYS = 180

# Held-out / training split. Scaled down from the sibling harnesses' 150/500 because
# the probe could only pull 271 qualifying items inside one day's request budget.
N_EVAL_ITEMS = 80
N_TRAIN_ITEMS = 190
N_TRAINED_EVAL_ITEMS = 80
ROW_BUDGET = 200_000

CORR_PRUNE_THRESHOLD = 0.95
VAL_WINDOW_DAYS = 21
STEP_DAYS = 60

# Identical to ab_test_item_metadata.py / ab_test_training_breadth.py so the held-out
# split is drawn the same way; the item pool differs, so the items themselves differ.
SPLIT_SEED = 20260806
SAMPLE_SEED = 4242
PLACEBO_SEED = 42

# CSFloat's earliest observation across the probed sample is 2020-03-17. Rows before
# it carry no basis at all, and including them puts the fold split in a region the
# feature does not exist in -- see the note in run_evaluation.
DEFAULT_MIN_DATE = "2020-04-01"

CF_RAW = ("cf_basis",)
CF_SMOOTH = ("cf_basis_7d", "cf_basis_dev_30d", "cf_basis_chg_7d")
CF_COUNT = ("cf_log_sales_7d",)
CF_ALL = CF_RAW + CF_SMOOTH + CF_COUNT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ab_test_csfloat_basis")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Production's item universe, spelled into every archive read this harness
# makes. Before 2026-08-08 the `ab_test_*` family globbed the Parquet privately
# and saw a universe production does not train on, so an A/B advised a model it
# had not measured. The bid sources need no clause here: `aggregator_sync` is a
# single ask feed and already excludes them. See `models/item_parser.py`.
_UNIVERSE = phase_collapsed_sql_filter()

DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #


def _archive_union_sql(con):
    """One UNION over the archive, restricted to the single continuous series.

    Identical to ab_test_item_metadata.py: 2026 files filtered to `aggregator_sync`,
    pre-2026 files taken whole. `volume` is selected only because engineer_features
    requires the column; every feature derived from it is shelved.
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


def load_csfloat(probe_dir: Path) -> pd.DataFrame:
    """Per item-day CSFloat sale average and count, in dollars."""
    path = probe_dir / "csfloat_probe_series.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run probe_csfloat_history.py first")
    cf = pd.read_parquet(path)
    cf = cf.rename(columns={"price": "cf_cents", "count": "cf_sales"})
    cf["day"] = pd.to_datetime(cf["day"])
    # The probe measured the median ratio at 0.955 after /100, i.e. parity with the
    # archive's net price and no fee offset to correct -- unlike Steam's 1.1607.
    cf["cf_usd"] = cf["cf_cents"] / 100.0
    cf = (
        cf[["item_slug", "day", "cf_usd", "cf_sales"]]
        .dropna(subset=["cf_usd"])
        .drop_duplicates(subset=["item_slug", "day"])
        .sort_values(["item_slug", "day"])
    )
    logger.info(
        "  CSFloat: %s rows, %d items, %s -> %s",
        f"{len(cf):,}",
        cf.item_slug.nunique(),
        cf.day.min().date(),
        cf.day.max().date(),
    )
    return cf


def build_basis_features(px: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    """Basis columns per (item, date), every one lagged a full day.

    `px` is the archive daily series. Rolling windows are positional rather than
    calendar-indexed; CSFloat coverage is near-daily inside an item's span, so the two
    coincide in the overwhelming majority of rows, and a positional window cannot
    silently reach across a gap the way a calendar one can.
    """
    m = px.merge(cf, left_on=["item_id", "date"], right_on=["item_slug", "day"], how="left").sort_values(
        ["item_id", "date"]
    )
    ok = (m.cf_usd > 0) & (m.price > 0)
    m["cf_basis"] = np.where(ok, np.log(m.cf_usd.where(ok) / m.price.where(ok)), np.nan)

    g = m.groupby("item_id", sort=False)
    cf_7 = g.cf_usd.transform(lambda s: s.rolling(7, min_periods=3).mean())
    px_7 = g.price.transform(lambda s: s.rolling(7, min_periods=3).mean())
    ok7 = (cf_7 > 0) & (px_7 > 0)
    m["cf_basis_7d"] = np.where(ok7, np.log(cf_7.where(ok7) / px_7.where(ok7)), np.nan)

    # Deviation from the item's own recent basis: removes the per-item constant
    # offset (float mix, market preference) and asks whether CSFloat is unusually
    # expensive relative to the archive RIGHT NOW. This is the mispricing form.
    m["cf_basis_dev_30d"] = m.cf_basis - g.cf_basis.transform(lambda s: s.rolling(30, min_periods=10).mean())
    m["cf_basis_chg_7d"] = g.cf_basis.transform(lambda s: s.diff(7))
    m["cf_log_sales_7d"] = np.log1p(g.cf_sales.transform(lambda s: s.rolling(7, min_periods=3).sum()))

    # Shift every column one day forward inside each item. Nothing here is known
    # until the day it describes has closed.
    for col in CF_ALL:
        m[col] = m.groupby("item_id", sort=False)[col].shift(1)

    out = m[["item_id", "date", *CF_ALL]].copy()
    for col in CF_ALL:
        pct = 100.0 * out[col].notna().mean()
        logger.info("  %-18s %9s/%s non-null (%.1f%%)", col, f"{out[col].notna().sum():,}", f"{len(out):,}", pct)
    return out


def _fingerprint(probe_dir: Path) -> str:
    src = Path(__file__).parent.parent / "models" / "forecaster.py"
    h = hashlib.sha256(src.read_bytes())
    h.update(repr((MIN_MEDIAN_PRICE, MIN_ITEM_DAYS, CORR_PRUNE_THRESHOLD, CF_ALL, _UNIVERSE)).encode())
    series = probe_dir / "csfloat_probe_series.parquet"
    h.update(hashlib.sha256(series.read_bytes()).digest())
    return h.hexdigest()[:16]


def build_frame(probe_dir: Path, cache_path=None):
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != _fingerprint(probe_dir):
                raise SystemExit(
                    f"Frame cache {cache_path} was built from different feature code, "
                    f"constants or probe data. Rebuild with --build-cache-only."
                )
            df = pd.read_parquet(cache_path)
            logger.info("  Loaded cached frame %s (%s rows)", cache_path, f"{len(df):,}")
            return df, meta["pruned"]

    df, pruned = _build_frame_uncached(probe_dir)

    if cache_path is not None:
        tmp_frame = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
        tmp_meta = cache_path.with_suffix(f".{os.getpid()}.tmp.json")
        df.to_parquet(tmp_frame, index=False)
        tmp_meta.write_text(json.dumps({"fingerprint": _fingerprint(probe_dir), "pruned": pruned, "rows": len(df)}))
        os.replace(tmp_frame, cache_path)
        os.replace(tmp_meta, cache_path.with_suffix(".meta.json"))
        logger.info("  Wrote frame cache %s (%s rows)", cache_path, f"{len(df):,}")
    return df, pruned


def _build_frame_uncached(probe_dir: Path):
    con = duckdb.connect()
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        cf = load_csfloat(probe_dir)
        union_sql = _archive_union_sql(con)
        con.register("cf_items", pd.DataFrame({"item_slug": cf.item_slug.unique()}))

        # The universe is the sibling harnesses' definition, intersected with the
        # items the probe actually pulled. Depth-rank is not re-applied: an item at
        # depth rank 400 is as valid a member of "deep >=$1 items" as one at rank 4.
        slugs = [
            r[0]
            for r in con.sql(f"""
            SELECT item_slug FROM ({union_sql})
            WHERE item_slug IN (SELECT item_slug FROM cf_items)
            GROUP BY item_slug
            HAVING COUNT(DISTINCT day) >= {MIN_ITEM_DAYS}
               AND MEDIAN(mean_price) >= {MIN_MEDIAN_PRICE}
               AND MIN(day) < DATE '2026-01-01'
            ORDER BY COUNT(DISTINCT day) DESC, item_slug
        """).fetchall()
        ]
        logger.info("  Universe: %d deep >=$1 items with CSFloat history", len(slugs))
        if len(slugs) < N_EVAL_ITEMS + N_TRAIN_ITEMS:
            raise SystemExit(
                f"Universe is {len(slugs)}; need {N_EVAL_ITEMS + N_TRAIN_ITEMS}. "
                f"Pull more items with probe_csfloat_history.py."
            )

        placeholders = ", ".join("?" for _ in slugs)
        all_prices = con.sql(
            f"""
            SELECT item_slug AS item_id, day AS timestamp, mean_price AS price, volume
            FROM ({union_sql}) WHERE item_slug IN ({placeholders})
        """,
            params=slugs,
        ).df()

        all_prices["timestamp"] = pd.to_datetime(all_prices["timestamp"])
        all_prices["date"] = all_prices["timestamp"].dt.date
        all_prices["item_id"] = pd.Categorical(all_prices["item_id"], categories=slugs, ordered=True)
        all_prices = all_prices.sort_values(["item_id", "timestamp"], kind="stable").reset_index(drop=True)
        all_prices["item_id"] = all_prices["item_id"].astype(str)
        logger.info("  Loaded %s price rows", f"{len(all_prices):,}")

        market_index = build_market_index(all_prices)
        logger.info(
            "  market index: %s dates, %s valid",
            f"{len(market_index):,}",
            f"{int(market_index['valid'].sum()) if not market_index.empty else 0:,}",
        )

        # Basis is computed on the raw daily series, before feature engineering
        # collapses anything, so both legs of the spread come from the same day.
        px = all_prices[["item_id", "timestamp", "price"]].rename(columns={"timestamp": "date"})
        basis = build_basis_features(px, cf)

        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)

        mf_cols = []
        join_dates = pd.to_datetime(df["date"])
        for h in ItemForecaster.HORIZONS:
            col = f"market_factor_{h}d"
            df[col] = join_dates.map(market_factor_for_horizon(market_index, h)).astype(float)
            mf_cols.append(col)

        EXCLUDE = {"item_id", "date", "timestamp", "price", "volume", "name", "release_date"}
        numeric = (np.float64, np.float32, np.int64, int, float)
        all_cols = [c for c in df.columns if c not in EXCLUDE and df[c].dtype in numeric]
        kept = [c for c in all_cols if c not in ItemForecaster.SHELVED_FEATURES]
        kept = ItemForecaster._apply_feature_allowlist(kept, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
        logger.info("  Features: %d -> %d after shelving + allowlist", len(all_cols), len(kept))

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
        logger.info("  %d -> %d after corr prune", len(kept), len(pruned))

        df = df[["item_id", "date", "price", *pruned, *mf_cols]].copy()
        df["date"] = pd.to_datetime(df["date"])
        basis["date"] = pd.to_datetime(basis["date"])
        before = len(df)
        df = df.merge(basis, on=["item_id", "date"], how="left")
        assert len(df) == before, "basis join changed the row count"
        for col in CF_ALL:
            logger.info("  joined %-18s %6.1f%% of frame rows non-null", col, 100.0 * df[col].notna().mean())
        return df, pruned
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def assign_items(df):
    slugs = list(pd.unique(df["item_id"]))
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = [slugs[i] for i in rng.permutation(len(slugs))]
    eval_items = sorted(shuffled[:N_EVAL_ITEMS])
    pool = shuffled[N_EVAL_ITEMS:]
    if len(pool) < N_TRAIN_ITEMS:
        raise SystemExit(f"need {N_TRAIN_ITEMS} training items, pool has {len(pool)}")
    train_items = sorted(pool[:N_TRAIN_ITEMS])
    trained_eval = sorted(train_items[:N_TRAINED_EVAL_ITEMS])
    logger.info("  Held-out eval items: %d", len(eval_items))
    logger.info("  Training items:      %d", len(train_items))
    logger.info("  Trained-eval slice:  %d (subset of training)", len(trained_eval))
    assert not (set(eval_items) & set(train_items))
    return eval_items, train_items, trained_eval


def _stratified_sample(train_df, items, budget, fold_idx):
    """Up to `budget` rows spread evenly over `items`. Seeded by fold, not by arm."""
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


def run_evaluation(df, pruned, horizon_filter=None, n_jobs=None, market_relative=False, min_date=DEFAULT_MIN_DATE):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)

    # The frame carries each item's full archive history back to 2013, but CSFloat
    # starts 2020-04. The fold split is taken over the DATE INDEX, so on the full
    # frame it lands at 2022-01-27 and leaves the training region with 0.8% basis
    # coverage against 59% in validation -- the model would be fitted almost
    # entirely on the median fill and then scored on real values, which tests
    # nothing. Restricting to the CSFloat era puts both regions on the same
    # footing. Features are engineered on the full series first, so lags and
    # rolling windows still see the pre-cutoff history.
    if min_date:
        before, items_before = len(df), df["item_id"].nunique()
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(min_date)].copy()
        logger.info(
            "  min-date %s: %s -> %s rows, %d -> %d items",
            min_date,
            f"{before:,}",
            f"{len(df):,}",
            items_before,
            df["item_id"].nunique(),
        )
        if df.empty:
            raise SystemExit(f"No rows at or after {min_date}")

    # `date` is carried as datetime64 through the basis join (a date-object key
    # cannot be merged against one), but prepare_targets merges on plain dates.
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = df["date"].dt.date

    eval_items, train_items, trained_eval = assign_items(df)

    arms = {
        "baseline": [],
        "basis_raw": list(CF_RAW),
        "basis_smooth": list(CF_SMOOTH),
        "basis_all": list(CF_ALL),
        "count_only": list(CF_COUNT),
        "placebo": list(CF_ALL),
    }
    for name, extra in arms.items():
        logger.info(
            "  Arm %-13s %d features (%d price + %d csfloat)", name, len(pruned) + len(extra), len(pruned), len(extra)
        )

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        horizons = ItemForecaster.HORIZONS
        if horizon_filter is not None:
            horizons = [h for h in horizons if h == horizon_filter]

        results = {}
        for horizon in horizons:
            logger.info("\n  %s\n  Evaluating %dd\n  %s", "=" * 60, horizon, "=" * 60)
            tdf = forecaster.prepare_targets(df, horizon)
            target_col = f"target_return_{horizon}d"
            tdf = tdf.dropna(subset=[target_col]).copy()
            tdf = tdf.sort_values(["item_id", "date"])
            if tdf.empty:
                logger.warning("    No valid targets for %dd", horizon)
                continue

            base_cols = [c for c in pruned if c in tdf.columns]
            keep_cols = ["item_id", "date", "price", target_col, *base_cols, *list(CF_ALL)]
            mf_col = f"market_factor_{horizon}d"
            if market_relative:
                if mf_col not in tdf.columns:
                    raise SystemExit(f"--market-relative needs {mf_col}; rebuild the cache.")
                keep_cols.append(mf_col)
            sub = tdf[[c for c in keep_cols if c in tdf.columns]].copy()

            dates = sorted(sub["date"].unique())
            split_idx = len(dates) * 2 // 3
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            is_heldout = sub["item_id"].isin(set(eval_items)).to_numpy()
            is_trained_eval = sub["item_id"].isin(set(trained_eval)).to_numpy()
            is_train_item = sub["item_id"].isin(set(train_items)).to_numpy()
            logger.info("    %s rows, %d dates, split at %s", f"{len(sub):,}", len(dates), dates[split_idx])

            results[horizon] = {}
            for arm, extra in arms.items():
                features = base_cols + extra
                logger.info("\n    --- %s (%d features) ---", arm, len(features))

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

                    y_train = train_df[target_col].values
                    y_val = val_df[target_col].values
                    if market_relative:
                        y_train = ItemForecaster._demean_returns(y_train, train_df[mf_col].to_numpy())
                        y_val = ItemForecaster._demean_returns(y_val, val_df[mf_col].to_numpy())

                    ds_kw = {"params": DS_PARAMS, "free_raw_data": False}
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
                    # Production's trainer and round table. The old call
                    # early-stopped on `dval` and scored `X_val` — the same
                    # rows — unless an opt-in flag was passed, which
                    # defaulted off. Fixed rounds are the default now and
                    # EARLY_STOPPING=1 is the only opt-out.
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
                    # Strict >=$1: flat-actual rows excluded, because sign(0) matched
                    # by a q50 emitting 0.0 is a free hit, not a call.
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
                            # fold_id is the resampling cluster: dates inside
                            # one validation window share a fitted model, so
                            # clustering on them understates the variance --
                            # see paired_mde's module docstring, which names
                            # this experiment as one to re-derive.
                            rec[cohort].append(
                                {
                                    "item_id": ids[i],
                                    "forecast_date": str(dts[i]),
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
                    logger.warning("    %s: no usable folds", arm)
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
                    "      held-out DA=%s%% (n=%s)   trained DA=%s%% (n=%s)   folds=%d",
                    entry["da_heldout"],
                    f"{entry['n_heldout']:,}",
                    entry["da_trained"],
                    f"{entry['n_trained']:,}",
                    len(per_fold),
                )

            base = results[horizon].get("baseline")
            if base:
                results[horizon]["_paired"] = {}
                for arm in ("basis_raw", "basis_smooth", "basis_all", "count_only", "placebo"):
                    if arm not in results[horizon]:
                        continue
                    results[horizon]["_paired"][arm] = {
                        cohort: paired_da_difference(
                            base["records"][cohort], results[horizon][arm]["records"][cohort], cluster_key="fold_id"
                        )
                        for cohort in ("heldout", "trained")
                    }
                # basis_all vs placebo is the comparison that rules out capacity
                # inflation: same width, only the values differ.
                if "basis_all" in results[horizon] and "placebo" in results[horizon]:
                    results[horizon]["_paired"]["basis_all_vs_placebo"] = {
                        cohort: paired_da_difference(
                            results[horizon]["placebo"]["records"][cohort],
                            results[horizon]["basis_all"]["records"][cohort],
                            cluster_key="fold_id",
                        )
                        for cohort in ("heldout", "trained")
                    }
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 100)
    print("CSFLOAT BASIS A/B — DA(strict, >=$1), held-out and trained cohorts")
    print("=" * 100)
    for h in sorted(k for k in results if isinstance(k, int)):
        r = results[h]
        print(f"\n  {h}d horizon")
        print(f"    {'arm':<14} {'feats':>6} {'held-out DA':>12} {'n':>9} {'trained DA':>12} {'n':>9} {'folds':>6}")
        print(f"    {'-' * 74}")
        for arm in ("baseline", "basis_raw", "basis_smooth", "basis_all", "count_only", "placebo"):
            a = r.get(arm)
            if not a:
                continue
            print(
                f"    {arm:<14} {a['n_features']:>6} "
                f"{a['da_heldout']:>11.2f}% {a['n_heldout']:>9,} "
                f"{a['da_trained']:>11.2f}% {a['n_trained']:>9,} {a['fold_count']:>6}"
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
    ap = argparse.ArgumentParser(description="A/B: CSFloat cross-market price basis")
    ap.add_argument("--probe-dir", default="/tmp/csfloat_probe")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--frame-cache", default=None)
    ap.add_argument("--build-cache-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument(
        "--market-relative",
        action="store_true",
        help="score idiosyncratic direction (label demeaned by the market "
        "factor); NOT comparable to raw-label or production numbers",
    )
    ap.add_argument(
        "--min-date",
        default=DEFAULT_MIN_DATE,
        help="drop rows before this date so the fold split lands inside the CSFloat era; '' to disable",
    )
    args = ap.parse_args()

    logger.info("A/B TEST: CSFloat cross-market basis")
    df, pruned = build_frame(Path(args.probe_dir), args.frame_cache)
    if args.build_cache_only:
        logger.info("Cache built; exiting.")
        return 0

    results = run_evaluation(
        df,
        pruned,
        horizon_filter=args.horizon,
        n_jobs=args.n_jobs,
        market_relative=args.market_relative,
        min_date=args.min_date,
    )
    print_summary(results)

    if args.out:
        slim = {}
        for h, r in results.items():
            slim[str(h)] = {
                k: ({kk: vv for kk, vv in v.items() if kk != "records"} if k != "_paired" else v) for k, v in r.items()
            }
        Path(args.out).write_text(
            json.dumps(
                {
                    "market_relative": args.market_relative,
                    "min_date": args.min_date,
                    "n_eval_items": N_EVAL_ITEMS,
                    "n_train_items": N_TRAIN_ITEMS,
                    "results": slim,
                },
                indent=2,
                default=str,
            )
        )
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
