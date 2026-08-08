#!/usr/bin/env python3
"""Re-derive `TRAIN_MIN_MEDIAN_PRICE`'s +3.50pp at 30d without the look-ahead.

`2026-08-07-training-item-universe.md` credits a $1 median-price floor with
**+3.50pp at 30d [+1.56, +5.98]** on held-out >=$1 items, and next-steps step 7
proposes shipping it. Its own caveat: *"Re-derive the result first with a
per-fold price filter — the current version selects on a full-sample median."*
This is that re-derivation. Nothing here changes production.

## The defect

`models/forecaster.py::_filter_by_median_price` takes the median over the whole
2013-2026 frame, once, before any split exists. A fold whose window ends in
2021 therefore trains on *"items whose median price through 2026 is >= $1"* — a
set nobody could name in 2021. Measured on this archive at h=30's 26 folds:

    fold 1  (cutoff 2021-12-18):  319 items knowable,  876 used  (594 leaked)
    fold 26 (cutoff 2026-01-26):  817 items knowable,  876 used

and **141 items enter some per-fold universe but never the full-sample one** —
they were expensive and fell to pennies, so the full-sample filter rejects them
on prices that postdate every fold that wanted them.

**The signature fits survivorship.** A genuine liquidity effect has no reason to
be horizon-selective; the measured effect is h=30 only, null at 3/7/14. And the
harness that produced +3.50pp carried the same leak, so it cannot adjudicate
itself.

## Arms — train side only

| arm | training universe |
|---|---|
| `prod_pool`            | 99 items at production's measured mix (17 >=$1 + 82 sub-$1). |
| `prod_pool_b`          | a disjoint second draw of the same shape. **PLACEBO 1.** |
| `full_sample`          | full-sample median >= $1 and >= 180 days. `ge1_full`, the config step 7 would ship. |
| `per_fold`             | median and day count over `date < val_start - embargo(h)`. |
| `full_sample_matched`  | `full_sample`, downsampled per fold to `per_fold`'s item count. |
| `full_sample_matched_b`| the same at a second seed. **PLACEBO 2.** |

Two contrasts, because one alone cannot answer the question:

* **`full_sample` and `per_fold`, each against `prod_pool`.** This is the +3.50pp itself
  (`ge1_full` vs `prod_a`) and its leak-free counterpart. If the floor's gain
  survives per-fold selection it is not survivorship.
* **`per_fold` against `full_sample_matched`.** The leak in isolation, at equal
  universe size.

`full_sample_matched` is not optional. Per-fold selection *shrinks* early folds
(319 vs 876), so without it "the effect vanished because the leak is gone" and
"the effect vanished because there is less data" are one observation. Note the
plan this executes had the matched arm downsampling `per_fold` to
`full_sample`'s size; the measurement above shows the inequality runs the other
way, so the full-sample arm is the one that gets thinned.

The placebo is read **first**: `ab-fold-count-floor` records a permuted placebo
reading significantly positive at ~7 folds, so a treatment number is worthless
until the instrument's own noise floor is on the page. It is the second seed of
the matching draw because that draw is the only randomness in the design —
`full_sample` and `per_fold` are both deterministic given the folds.

`val_set` is never filtered. The metric is already DA(strict, >=$1) on 150
held-out items, so filtering val would move the evaluation cohort with the
treatment and the two would be inseparable — which is the artifact that halved
the original unpaired 58.60-vs-50.88 headline.

## Three defects inherited from the scratchpad instrument, fixed here

The +3.50pp came from an `ab_universe_paired.py` that lived only in a
scratchpad. Promoted here so it is re-runnable, with:

1. **A NULL-safe source filter.** It spelled `WHERE source = 'aggregator_sync'`,
   which is the bug that emptied eight harnesses when the 2026-08-08 migration
   materialised `source` as a typed NULL on the pre-2026 files
   (`2026-08-08-migrated-archive-emptied-eight-harnesses.md`). It reads through
   `db/archive.py::prices_relation` now.
2. **The universe rules.** It applied neither the bid-source exclusion nor the
   phase-collapsed names, both of which every `ab_test_*` harness got on
   2026-08-08.
3. **A train-side embargo.** It had none. `ab_test_training_breadth.py` was
   fixed on 2026-08-07 and this one never was, so **the stored +3.50pp is an
   un-purged number** — and the un-purged split was later measured to inflate
   DA by +10.15pp at 30d (`2026-08-08-embargo-discontinuity-measured.md`).
   Arms differing in item set are asymmetrically exposed: a wider arm sees more
   distinct items inside the overlapping band.

## What is held fixed, and is wrong the same way in every arm

The market factor. `_add_cross_sectional_features` runs once on the pooled
frame, so `market_return_*d` and `item_return_vs_market_*d` describe the
1,017-item superset rather than any arm's own universe. Computing it per arm
per fold would be a second treatment riding inside a one-variable contrast. It
differences out of the paired comparison; it is not a claim that the pooled
market factor is correct.

Usage (from `backend/`, ~1-2h per horizon):

    venv/bin/python scripts/ab_test_train_universe.py --horizon 30 \\
        --out /tmp/universe_h30.json
"""
from __future__ import annotations

# ── Universe ────────────────────────────────────────────────────────────
MIN_MEDIAN_PRICE = 1.0
# Minimum distinct days. A full-sample clause in the original, and survivorship
# on the same axis as the median: selecting items with 180 days over 2013-2026
# pre-selects items that survived to 2026. It moves inside the fold cutoff in
# `per_fold` alongside the median.
MIN_ITEM_DAYS = 180

# Held out from every arm, drawn from the full-sample universe as the original
# did — so the re-derivation is read against the number it is re-deriving.
N_EVAL_ITEMS = 150

# `prod_pool`: production's measured draw, 99 items of which 20.1 on average are
# >= $1 across the 8-seed sweep. 17 is the low end of the measured 17-23, taken
# as the original did so the reproduced baseline is the same baseline.
N_PROD_ITEMS = 99
N_PROD_GE1 = 17
# Sub-$1 items pulled into the frame: enough to supply two disjoint draws of
# `N_PROD_ITEMS - N_PROD_GE1`, plus depth for the market factor.
N_SUB1_POOL = 600

# Fold schedule, inherited from `ab_test_training_breadth.py` so the numbers sit
# on the same instrument as the result being re-derived.
VAL_WINDOW_DAYS = 21
STEP_DAYS = 60
CORR_PRUNE_THRESHOLD = 0.95

SPLIT_SEED = 20260807     # as the original, so the eval draw is comparable
MATCH_SEED = 909_000      # the per-fold downsample
MATCH_SEED_B = 606_000    # ... and its placebo twin

import os
import sys
import json
import hashlib
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from database import SessionLocal
from db.archive import prices_relation
from models.forecaster import (ItemForecaster, archive_universe_sql_filter,
                               embargo_days)
from backtest.paired_mde import paired_arm_contrasts, format_paired

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ab_test_train_universe")

DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}

ARMS = ("prod_pool", "prod_pool_b", "full_sample", "per_fold",
        "full_sample_matched", "full_sample_matched_b")
BASE_ARM = "prod_pool"

# The universe predicate, spelled into the cache key. A frame cache
# fingerprints `forecaster.py`'s bytes, which cannot see a change to
# `item_parser.py` — five harnesses hash it explicitly and a new one must too.
_UNIVERSE = archive_universe_sql_filter(source_column="source")


def _frame_fingerprint():
    src = Path(__file__).parent.parent / "models" / "forecaster.py"
    h = hashlib.sha256(src.read_bytes())
    h.update(repr((MIN_MEDIAN_PRICE, MIN_ITEM_DAYS, CORR_PRUNE_THRESHOLD,
                   _UNIVERSE)).encode())
    return h.hexdigest()[:16]


def _prices_sql(con):
    """The single continuous series, universe-filtered, NULL-safe.

    `source IS NULL` is the pre-2026 series and `aggregator_sync` is its 2026
    continuation; mixing the other ten 2026 sources in would put a 1.37x
    duplicate item-day into the frame and move the price consensus partway
    through the span.
    """
    return prices_relation(
        con,
        columns=["item_slug", "day", "source", "mean_price", "volume"],
        where=f"(source IS NULL OR source = 'aggregator_sync') AND {_UNIVERSE}",
    )


def _fold_cutoffs(con, relation, horizon):
    """Every fold's selection cutoff, off the archive calendar.

    Used only to size the frame superset. The real cutoffs are recomputed
    inside `run()` from the horizon's own date grid; this is a superset of
    them because that grid is a subset of these days.
    """
    dates = [r[0] for r in con.sql(
        f"SELECT DISTINCT day FROM {relation} ORDER BY day").fetchall()]
    split_idx = len(dates) * 2 // 3
    out = []
    for we in range(split_idx + 1, len(dates), STEP_DAYS):
        val_dates = dates[we:we + VAL_WINDOW_DAYS]
        if len(val_dates) < 7:
            continue
        out.append(pd.Timestamp(val_dates[0])
                   - pd.Timedelta(days=embargo_days(horizon)))
    return dates, out


def build_frame(horizons, cache_path=None):
    """Load the superset, engineer features, prune. Cached.

    The frame must contain the union of every arm's universe over every fold,
    or `per_fold` would be silently clipped to the full-sample set and the
    harness would answer its own question with the leak still in. That union is
    computed here rather than assumed.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != _frame_fingerprint():
                raise SystemExit(
                    f"Frame cache {cache_path} was built from different feature "
                    f"code or constants. Rebuild with --build-cache-only.")
            df = pd.read_parquet(cache_path)
            logger.info(f"  Loaded cached frame {cache_path} ({len(df):,} rows)")
            return df, meta["pruned"]

    df, pruned = _build_frame_uncached(horizons)

    if cache_path is not None:
        # Frame lands before the meta that vouches for it, so a reader can
        # never pick up a meta pointing at a half-written parquet.
        tmp_frame = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
        tmp_meta = cache_path.with_suffix(f".{os.getpid()}.tmp.json")
        df.to_parquet(tmp_frame, index=False)
        tmp_meta.write_text(json.dumps({
            "fingerprint": _frame_fingerprint(),
            "pruned": pruned, "rows": len(df),
        }))
        os.replace(tmp_frame, cache_path)
        os.replace(tmp_meta, cache_path.with_suffix(".meta.json"))
        logger.info(f"  Wrote frame cache {cache_path} ({len(df):,} rows)")

    return df, pruned


def _build_frame_uncached(horizons):
    import duckdb
    con = duckdb.connect()
    db = SessionLocal()

    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        relation = _prices_sql(con)

        full = {r[0] for r in con.sql(f"""
            SELECT item_slug FROM {relation}
            GROUP BY item_slug
            HAVING COUNT(DISTINCT day) >= {MIN_ITEM_DAYS}
               AND MEDIAN(mean_price) >= {MIN_MEDIAN_PRICE}
               AND MIN(day) < DATE '2026-01-01'
        """).fetchall()}
        logger.info(f"  Full-sample >=${MIN_MEDIAN_PRICE:g} universe: "
                    f"{len(full)} items")

        union = set()
        for horizon in horizons:
            _, cutoffs = _fold_cutoffs(con, relation, horizon)
            for cutoff in cutoffs:
                union |= {r[0] for r in con.sql(f"""
                    SELECT item_slug FROM {relation}
                    WHERE day < DATE '{cutoff.date()}'
                    GROUP BY item_slug
                    HAVING COUNT(DISTINCT day) >= {MIN_ITEM_DAYS}
                       AND MEDIAN(mean_price) >= {MIN_MEDIAN_PRICE}
                """).fetchall()}
            logger.info(f"  h={horizon}: {len(cutoffs)} fold cutoffs, "
                        f"running per-fold union {len(union)} items")

        # The sub-$1 pool exists so `prod_pool` can be built: production applies
        # no price filter, so 82 of its 99 drawn items are sub-$1 and the
        # +3.50pp is measured against that mix. Without it there is no baseline
        # to reproduce and the harness can only say whether the leak is
        # load-bearing, not whether the floor is worth shipping.
        sub1 = [r[0] for r in con.sql(f"""
            SELECT item_slug FROM {relation}
            GROUP BY item_slug
            HAVING COUNT(DISTINCT day) >= {MIN_ITEM_DAYS}
               AND MEDIAN(mean_price) < {MIN_MEDIAN_PRICE}
               AND MIN(day) < DATE '2026-01-01'
            -- item_slug breaks n_days ties: without it DuckDB's parallel top-N
            -- picks a different pool, and a different item order, run to run.
            ORDER BY COUNT(DISTINCT day) DESC, item_slug
            LIMIT {N_SUB1_POOL}
        """).fetchall()]
        logger.info(f"  Sub-$1 pool: {len(sub1)} items")

        slugs = sorted((full | union | set(sub1)))
        logger.info(
            f"  Frame superset: {len(slugs)} items "
            f"({len(union - full)} reachable only per-fold, "
            f"{len(full - union)} only full-sample, {len(sub1)} sub-$1)")

        if not slugs:
            raise SystemExit(
                "Universe is empty. That is the 2026-08-08 archive-migration "
                "failure mode, not a data statement — check the source filter.")

        placeholders = ", ".join("?" for _ in slugs)
        # `volume` is selected only because engineer_features requires the
        # column to exist; every feature derived from it is in SHELVED_FEATURES.
        all_prices = con.sql(f"""
            SELECT item_slug AS item_id, day AS timestamp,
                   mean_price AS price, volume
            FROM {relation} WHERE item_slug IN ({placeholders})
        """, params=slugs).df()

        all_prices["timestamp"] = pd.to_datetime(all_prices["timestamp"])
        all_prices["date"] = all_prices["timestamp"].dt.date
        # Deterministic row order: LightGBM's bagging reads it.
        all_prices["item_id"] = pd.Categorical(
            all_prices["item_id"], categories=slugs, ordered=True)
        all_prices = all_prices.sort_values(
            ["item_id", "timestamp"], kind="stable").reset_index(drop=True)
        all_prices["item_id"] = all_prices["item_id"].astype(str)
        logger.info(f"  Loaded {len(all_prices):,} price rows")

        df = forecaster.engineer_features(all_prices, events_df)
        # Pooled, and therefore identically wrong in every arm — see the module
        # docstring. Do not move this inside the fold loop without saying so.
        df = forecaster._add_cross_sectional_features(df)

        EXCLUDE = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date"}
        numeric = (np.float64, np.float32, np.int64, int, float)
        all_cols = [c for c in df.columns
                    if c not in EXCLUDE and df[c].dtype in numeric]
        kept = [c for c in all_cols if c not in ItemForecaster.SHELVED_FEATURES]
        kept = ItemForecaster._apply_feature_allowlist(
            kept, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
        logger.info(
            f"  Features: {len(all_cols)} engineered -> {len(kept)} after "
            f"shelving + allowlist {ItemForecaster.FEATURE_GROUP_ALLOWLIST}")

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

        df = df[["item_id", "date", "price"] + pruned].copy()
        df["_full_sample"] = df["item_id"].isin(full)
        df["_sub1"] = df["item_id"].isin(set(sub1))
        return df, pruned

    finally:
        con.close()


def _full_sample_items(df):
    return sorted(df.loc[df["_full_sample"], "item_id"].unique())


def _per_fold_items(df, cutoff):
    """`per_fold`'s universe: median AND day count, both over `date < cutoff`.

    Fixing only the median would leave the day-count survivorship in and
    produce a confident null.
    """
    priced = ItemForecaster._fold_median_price_items(
        df, MIN_MEDIAN_PRICE, cutoff)
    past = df[pd.to_datetime(df["date"]) < pd.Timestamp(cutoff)]
    n_days = past.groupby("item_id")["date"].nunique()
    deep = set(n_days[n_days >= MIN_ITEM_DAYS].index)
    return priced & deep


def assign_eval(df):
    """150 held-out items, drawn from the full-sample universe.

    Drawn there rather than from a per-fold universe because the arms are read
    against a stored full-sample result; an eval cohort defined by the
    treatment would not be the population that number describes.
    """
    full = _full_sample_items(df)
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = [full[i] for i in rng.permutation(len(full))]
    eval_items = sorted(shuffled[:N_EVAL_ITEMS])
    logger.info(f"  Eval (held out, >=$1): {len(eval_items)} of {len(full)}")
    return set(eval_items)


def _match(items, k, seed, fold_idx):
    """Downsample `items` to `k`, with the fold's own seed."""
    if k >= len(items):
        return list(items)
    rng = np.random.default_rng(seed + fold_idx)
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in sorted(idx)]


def assign_prod_draws(df, eval_set, full_items):
    """Two disjoint 99-item draws at production's measured composition.

    Disjoint because the real draws are: eight production retrains varying only
    `_stratified_item_subsample`'s seed shared 2 to 5 items of 99 (Jaccard
    0.010-0.026). `prod_pool_b` is therefore a second universe rather than a
    perturbation of the first, which is what makes it a placebo.
    """
    sub1 = sorted(df.loc[df["_sub1"], "item_id"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    ge1 = [i for i in
           (full_items[j] for j in rng.permutation(len(full_items)))]
    sub1 = [sub1[j] for j in rng.permutation(len(sub1))]

    n_sub = N_PROD_ITEMS - N_PROD_GE1
    a = sorted(ge1[:N_PROD_GE1] + sub1[:n_sub])
    b = sorted(ge1[N_PROD_GE1:2 * N_PROD_GE1] + sub1[n_sub:2 * n_sub])
    assert not (set(a) & set(b)), "the placebo draw must share no item"
    assert not ((set(a) | set(b)) & eval_set)
    logger.info(f"  prod_pool: {len(a)} items ({N_PROD_GE1} >=$1), "
                f"prod_pool_b: {len(b)} disjoint")
    return a, b


def run(df, pruned, horizon, n_jobs):
    """Walk-forward at one horizon. Returns per-arm records and diagnostics."""
    eval_set = assign_eval(df)
    full_items = [i for i in _full_sample_items(df) if i not in eval_set]
    prod_a, prod_b = assign_prod_draws(df, eval_set, full_items)

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        # prepare_targets carries the 2026-08-08 frozen-price-run rule, so this
        # re-derivation runs on the post-step-6 label set. The stored +3.50pp
        # predates it and is not comparable on level, only on sign.
        tdf = forecaster.prepare_targets(df, horizon)
    finally:
        db.close()

    tcol = f"target_return_{horizon}d"
    tdf = tdf.dropna(subset=[tcol]).sort_values(["item_id", "date"])
    avail = [c for c in pruned if c in tdf.columns]
    sub = tdf[["item_id", "date", "price", tcol] + avail]

    dates = sorted(sub["date"].unique())
    split_idx = len(dates) * 2 // 3
    days = pd.to_datetime(sub["date"]).to_numpy()
    ddt = pd.to_datetime(pd.Series(dates)).to_numpy()
    is_eval = sub["item_id"].isin(eval_set).to_numpy()

    records = {arm: [] for arm in ARMS}
    per_fold = {arm: [] for arm in ARMS}

    for fold_idx, we in enumerate(range(split_idx + 1, len(dates), STEP_DAYS)):
        val_dates = dates[we:we + VAL_WINDOW_DAYS]
        if len(val_dates) < 7:
            continue
        val_start = val_dates[0]
        cutoff = pd.Timestamp(val_start) - pd.Timedelta(
            days=embargo_days(horizon))

        # The selection statistic is a function of prices, so it is computed on
        # the embargoed cutoff, never on val_start.
        fold_items = sorted(_per_fold_items(df, cutoff) - eval_set)
        if not fold_items:
            logger.info(f"    fold {fold_idx}: empty per-fold universe at "
                        f"{cutoff.date()}, skipped")
            continue

        arm_items = {
            "prod_pool": prod_a,
            "prod_pool_b": prod_b,
            "full_sample": full_items,
            "per_fold": fold_items,
            "full_sample_matched": _match(
                full_items, len(fold_items), MATCH_SEED, fold_idx),
            "full_sample_matched_b": _match(
                full_items, len(fold_items), MATCH_SEED_B, fold_idx),
        }

        in_val = ((days >= ddt[we])
                  & (days <= ddt[we + len(val_dates) - 1]))
        val_df = sub[in_val & is_eval]
        if len(val_df) < 50:
            continue
        in_train_window = days <= ddt[we - 1]

        logger.info(
            f"    fold {fold_idx} val {val_start} cutoff {cutoff.date()}: "
            f"full_sample {len(full_items)} items, per_fold {len(fold_items)} "
            f"({len(set(fold_items) - set(full_items))} unreachable "
            f"full-sample), val n={len(val_df):,}")

        for arm in ARMS:
            items = set(arm_items[arm])
            train_df = sub[in_train_window
                           & sub["item_id"].isin(items).to_numpy()]
            # Embargo the TRAIN side only, through production's own purge. The
            # val side is never purged: at h=30 that would empty the window.
            train_df = ItemForecaster._purge_overlapping_train_rows(
                train_df, val_start, horizon)
            if train_df.empty:
                continue

            train_median = train_df[avail].median()
            X_train = train_df[avail].fillna(train_median).values
            y_train = train_df[tcol].values
            X_val = val_df[avail].fillna(train_median).values
            y_val = val_df[tcol].values

            dtrain = lgb.Dataset(X_train, y_train, params=DS_PARAMS,
                                 free_raw_data=False)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain,
                               params=DS_PARAMS, free_raw_data=False)
            params = {
                "objective": "quantile", "alpha": 0.5, "metric": "quantile",
                "boosting_type": "gbdt", "num_leaves": 31, "max_depth": 5,
                "min_data_in_leaf": 15, "min_gain_to_split": 0.1,
                "learning_rate": 0.03, "feature_fraction": 0.7,
                "bagging_fraction": 0.7, "bagging_freq": 5,
                "lambda_l1": 0.5, "lambda_l2": 0.5, "verbosity": -1,
                "random_state": 42, "n_jobs": n_jobs,
                "force_row_wise": True, **DS_PARAMS,
            }
            model = lgb.train(
                params, dtrain, num_boost_round=100, valid_sets=[dval],
                callbacks=[lgb.early_stopping(15, verbose=False),
                           lgb.log_evaluation(0)])
            pred = model.predict(X_val)

            # Strict >=$1 scoring, flat-actual rows dropped: sign(0)==sign(0)
            # is what inflated the old penny-cohort DA by ~31pp.
            price = val_df["price"].to_numpy(dtype=float)
            actual = np.asarray(y_val, dtype=float)
            match = (np.sign(np.nan_to_num(actual))
                     == np.sign(np.nan_to_num(pred)))
            scored = (actual != 0) & (price >= 1.0)

            ids = val_df["item_id"].to_numpy()
            dts = val_df["date"].to_numpy()
            for i in np.flatnonzero(scored):
                records[arm].append({
                    "item_id": ids[i], "forecast_date": str(dts[i]),
                    # fold_id, not forecast_date, is the resampling cluster:
                    # dates inside one 21-day window share a fitted model.
                    "fold_id": fold_idx,
                    "direction_correct": bool(match[i]),
                })
            tot = int(scored.sum())
            per_fold[arm].append({
                "fold": fold_idx, "val_start": str(val_start),
                "cutoff": str(cutoff.date()),
                "n_items": len(items), "n_train": len(train_df),
                "n_scored": tot,
                "dir_acc": round(int((match & scored).sum()) / tot * 100, 2)
                           if tot else None,
            })

    summary = {}
    for arm in ARMS:
        folds = per_fold[arm]
        if not folds:
            logger.warning(f"    {arm}: no usable folds")
            continue
        n = sum(f["n_scored"] for f in folds)
        hits = sum(int(r["direction_correct"]) for r in records[arm])
        summary[arm] = {
            "DA": round(hits / n * 100, 2) if n else None,
            "n_scored": n, "folds": len(folds),
            "mean_items": round(float(np.mean([f["n_items"] for f in folds])), 1),
            "mean_rows_per_fold": int(np.mean([f["n_train"] for f in folds])),
            "per_fold": folds,
        }
        logger.info(
            f"    {arm:24s} DA={summary[arm]['DA']}%  n={n:,}  "
            f"folds={len(folds)}  items/fold={summary[arm]['mean_items']}  "
            f"rows/fold={summary[arm]['mean_rows_per_fold']:,}")

    def _against(base, arms):
        return paired_arm_contrasts(
            {k: v for k, v in records.items() if k in (base, *arms)},
            base, cluster_key="fold_id")

    return {
        "arms": summary,
        # The headline: `full_sample` here IS `ge1_full`, and its contrast with
        # `prod_pool` is the +3.50pp being re-derived. `per_fold` beside it is
        # the same claim with the look-ahead removed.
        "vs_prod_pool": paired_arm_contrasts(
            records, BASE_ARM, cluster_key="fold_id"),
        # The leak in isolation, at equal universe size. A contrast between two
        # non-base arms, so it is computed explicitly.
        "per_fold_vs_matched": _against("full_sample_matched", ["per_fold"]),
        # Two noise floors, read before anything above. `prod_pool_b` is the
        # item draw's; `full_sample_matched_b` is the matching draw's.
        "placebo_prod": _against("prod_pool", ["prod_pool_b"]),
        "placebo_matched": _against(
            "full_sample_matched", ["full_sample_matched_b"]),
    }


def print_summary(results):
    print("\n" + "=" * 86)
    print("PER-FOLD PRICE FILTER — DA(strict, >=$1) on 150 held-out items, "
          "train side only")
    print("=" * 86)
    for h in sorted(results):
        r = results[h]
        print(f"\n  {h}d horizon")
        print(f"    {'arm':<24} {'items/fold':>11} {'rows/fold':>11} "
              f"{'DA':>7} {'folds':>6} {'n':>9}")
        print(f"    {'-' * 74}")
        for arm in ARMS:
            a = r["arms"].get(arm)
            if not a:
                continue
            print(f"    {arm:<24} {a['mean_items']:>11.1f} "
                  f"{a['mean_rows_per_fold']:>11,} {a['DA']:>6.2f}% "
                  f"{a['folds']:>6} {a['n_scored']:>9,}")

        print("\n    PLACEBOS FIRST — neither may read as an effect:")
        for key in ("placebo_prod", "placebo_matched"):
            for arm, p in r[key].items():
                print(f"      {arm:<24} {format_paired(p)}")
        print("\n    THE STORED RESULT — vs prod_pool "
              "(`ge1_full` vs `prod_a` was +3.50pp at 30d):")
        for arm, p in r["vs_prod_pool"].items():
            if arm == "prod_pool_b":
                continue
            print(f"      {arm:<24} {format_paired(p)}")
        print("\n    THE LEAK ALONE — per_fold vs full_sample at equal size:")
        for arm, p in r["per_fold_vs_matched"].items():
            print(f"      {arm:<24} {format_paired(p)}")
    print("")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Re-derive the $1 training floor without the look-ahead")
    parser.add_argument("--horizon", type=int, action="append", default=None,
                        help="repeatable; defaults to 30 then 14")
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    horizons = args.horizon or [30, 14]
    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 4) // 2)

    logger.info("=" * 70)
    logger.info("A/B: per-fold vs full-sample >=$1 training universe")
    logger.info("=" * 70)

    df, pruned = build_frame(horizons, cache_path=args.frame_cache)
    if args.build_cache_only:
        logger.info("Frame cache built; exiting before evaluation.")
        return 0

    results = {}
    for horizon in horizons:
        logger.info(f"\n  {'=' * 60}\n  Evaluating {horizon}d\n  {'=' * 60}")
        results[horizon] = run(df, pruned, horizon, n_jobs)
        if args.out:
            Path(args.out).write_text(
                json.dumps(results, indent=2, default=str))
            logger.info(f"  Wrote {args.out}")

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
