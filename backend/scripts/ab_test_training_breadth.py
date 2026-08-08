#!/usr/bin/env python3
"""A/B test: at a FIXED training-row budget, do more items beat more history?

This measures the load-bearing assumption behind the whole data-acquisition
ranking (`docs/changelog/2026-08-06-data-acquisition-ranking.md`). That ranking
puts the Steam listing-page backfill in Tier 1 on breadth grounds -- it *lowers*
rows/item from 1,341 to 923 and so raises `target_items` at the 700K budget from
521 to 758 -- and rejects `atalantus/buff-price-history-archive` on the mirror
image of the same argument (rows/item 1,341 -> 1,641, `target_items` 521 -> 426).

Both claims rest on "more items at a fixed budget is better than more rows per
item", which has never been measured. This measures it.

WHY THIS SHAPE, AND NOT "TRAIN ON THE SHALLOW 2026 ITEMS". The obvious test --
add the ~36k items that the `is_backfilled` gate excludes -- cannot be run in a
walk-forward harness. Those items exist only from 2026-01, so every fold before
2026 has no treatment data and no treatment eval rows, leaving ~3 disjoint folds
inside 2026. At the fold counts this project has measured (8 folds -> a 3-4pp
floor, `2026-08-06-paired-retrain-measures-no-gain.md`) that resolves nothing.

So the axis is varied INSIDE the deep cohort instead: every arm draws from the
same pre-2026 >=$1 population over the full 2013-2026 span, at ~25 folds, and the
arms differ only in how the same row budget is spread over items. A
newly-backfilled item from the Steam route arrives with deep history, so an arm
built from deep items at reduced rows-per-item is a faithful model of it -- which
is exactly what "1,341 -> 923 rows/item" describes.

Arms are NESTED (wide contains mid contains narrow) so the contrast is purely
"add more items", and all three see the same total rows per fold:

    narrow:  N_NARROW items x (R / N_NARROW) rows
    mid:     N_MID    items x (R / N_MID)    rows
    wide:    N_WIDE   items x (R / N_WIDE)   rows

Plus one diagnostic arm outside the budget:

    wide_unbudgeted: N_WIDE items, no row cap -- separates "breadth helped" from
                     "more data helped", which the budgeted arms cannot.

Evaluation is on HELD-OUT items that appear in no arm's training set, over
identical folds and identical rows, so `paired_da_difference` pairs perfectly on
(item_id, forecast_date).

## Embargo (fixed 2026-08-07)

Until 2026-08-07 the fold split here was `train = date < val_start` with **no purge
gap**, so every training row inside the last `horizon` days before the boundary carried
a `target_return_{h}d` resolved from inside the validation window. Production never had
that bug (`ItemForecaster._compute_cv_splits(..., purge_days=horizon)`), and this harness
family had simply diverged from it. The overlap is symmetric across arms, but only an arm
that can *locate* the overlapping rows exploits it, so date-level and date-proxy columns
banked leakage that per-item columns could not — measured on an event-calendar arm at
h=30 (25 folds, held-out >=$1, fold-clustered): **+12.1pp unpurged -> +6.1pp purged**.
This harness's arms differ only in the training item set, not in features, so the
exposure here is second-order: a wider arm sees more distinct items inside the
overlapping band and can average the shared future move more precisely. The date-ordinal
diagnostic reported in `2026-08-06-breadth-beats-depth-item-age-does-not.md` was
directly exposed.

The train side is now embargoed by calling production's own
`ItemForecaster._purge_overlapping_train_rows(train, val_start, horizon)`; a second
implementation of the same rule is what caused the drift in the first place. Purging is
unconditional -- this is a research tool with no results to keep back-compatible. The
**val** side is never purged: at h=30 that would empty the 21-day window. Note the purge
lands BEFORE `_stratified_sample`, so each arm still spends its full row budget.

Usage:
    python scripts/ab_test_training_breadth.py --build-cache-only \
        --frame-cache /tmp/breadth_frame.parquet
    python scripts/ab_test_training_breadth.py --frame-cache /tmp/breadth_frame.parquet \
        --horizon 7 --out /tmp/breadth_h7.json
"""
from __future__ import annotations

# ── Universe ────────────────────────────────────────────────────────────
# Minimum median price. Same rationale as ab_test_volume_features.py: without
# it the universe is penny items, 41% of forward returns are exactly zero, and
# `sign(0)==sign(0)` hands the model ~31pp of free hits. Production reports on
# the >=$1 cohort.
MIN_MEDIAN_PRICE = 1.0

# Minimum distinct days for an item to enter the universe. Higher than the
# volume harness's 90: an arm only means something if there is enough history
# per item to *take away*, and the narrow arm needs ~R/N_NARROW rows from each
# of its items.
#
# 180 is the largest threshold that still leaves the design room. Measured on
# this series: 904 deep >=$1 items at >=90 days, 878 at >=180, 779 at >=365,
# 585 at >=730. At 365 the universe cannot supply N_EVAL_ITEMS + N_WIDE.
MIN_ITEM_DAYS = 180

# Items pulled into the frame. Ordered by day count DESC so the deep end of the
# >=$1 cohort is used -- the population the Steam backfill would extend.
N_UNIVERSE = 870

# Held out for evaluation, in no arm's training set.
N_EVAL_ITEMS = 150

# Nested training-item counts. wide ⊇ mid ⊇ narrow.
#
# 150 -> 700 is a 4.7x breadth increase, far larger than the 1.45x the Steam
# backfill is projected to buy (`target_items` 521 -> 758). Deliberately so: if
# 4.7x moves nothing, 1.45x cannot, and if it does move something the `mid` arm
# says whether the response is monotone or noise.
N_NARROW = 150
N_MID = 350
N_WIDE = 700

# Total training rows per fold, shared by the three budgeted arms. Matches the
# 200K per-fold cap ab_test_volume_features.py already used, so the per-fold
# training cost is comparable to the run this is being compared against.
ROW_BUDGET = 200_000

# Correlation-prune threshold, as production.
CORR_PRUNE_THRESHOLD = 0.95

# Fold schedule. Same as ab_test_volume_features.py so fold counts and the
# resulting measurement floor are comparable to the volume result.
VAL_WINDOW_DAYS = 21
STEP_DAYS = 60

# Seeds. Item assignment and per-fold row sampling must be reproducible, and
# the row-sampling seed must not depend on the arm -- two arms that share an
# item should draw the same rows for it wherever their per-item quota allows.
SPLIT_SEED = 20260806
SAMPLE_SEED = 4242

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
from models.forecaster import ItemForecaster, phase_collapsed_sql_filter
from backtest.paired_mde import paired_da_difference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ab_test_training_breadth")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Production's item universe, spelled into every archive read this harness
# makes. Before 2026-08-08 the `ab_test_*` family globbed the Parquet privately
# and saw a universe production does not train on, so an A/B advised a model it
# had not measured. The bid sources need no clause here: `aggregator_sync` is a
# single ask feed and already excludes them. See `models/item_parser.py`.
_UNIVERSE = phase_collapsed_sql_filter()


DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}

ARM_ITEM_COUNTS = {
    "narrow": N_NARROW,
    "mid": N_MID,
    "wide": N_WIDE,
    "wide_unbudgeted": N_WIDE,
}
BUDGETED_ARMS = ("narrow", "mid", "wide")


def _frame_fingerprint():
    """Identify the feature-producing code and the frame-shaping constants.

    Same reasoning as the volume harness: a cache key over the data alone
    cannot see a change to `_compute_price_features`, and a stale frame would
    silently answer a different question.
    """
    src = Path(__file__).parent.parent / "models" / "forecaster.py"
    h = hashlib.sha256(src.read_bytes())
    h.update(repr((
        MIN_MEDIAN_PRICE, MIN_ITEM_DAYS, N_UNIVERSE, CORR_PRUNE_THRESHOLD,
        _UNIVERSE,
    )).encode())
    return h.hexdigest()[:16]


def build_frame(cache_path=None):
    """Load prices for the universe, engineer features, prune. Cached."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != _frame_fingerprint():
                raise SystemExit(
                    f"Frame cache {cache_path} was built from different feature "
                    f"code or constants. Rebuild with --build-cache-only."
                )
            df = pd.read_parquet(cache_path)
            logger.info(f"  Loaded cached frame {cache_path} ({len(df):,} rows)")
            return df, meta["pruned"]

    df, pruned = _build_frame_uncached()

    if cache_path is not None:
        # Frame lands before the meta that vouches for it, so a reader can
        # never pick up a meta pointing at a half-written parquet.
        tmp_frame = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
        tmp_meta = cache_path.with_suffix(f".{os.getpid()}.tmp.json")
        df.to_parquet(tmp_frame, index=False)
        tmp_meta.write_text(json.dumps({
            "fingerprint": _frame_fingerprint(),
            "pruned": pruned,
            "rows": len(df),
        }))
        os.replace(tmp_frame, cache_path)
        os.replace(tmp_meta, cache_path.with_suffix(".meta.json"))
        logger.info(f"  Wrote frame cache {cache_path} ({len(df):,} rows)")

    return df, pruned


def _archive_union_sql(con):
    """One UNION over the archive, restricted to the single continuous series.

    2026 files are filtered to `aggregator_sync`; pre-2026 files have no
    `source` column and are taken whole. This mirrors
    ab_test_volume_features.py: mixing the eleven 2026 sources in would put a
    1.37x duplicate item-day into the frame and change the price consensus
    partway through the span.
    """
    pq_files = sorted(str(p) for p in ARCHIVE_DIR.glob("prices-*.parquet"))
    queries = []
    for pqf in pq_files:
        cols = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()
        if "source" in {r[0] for r in cols}:
            queries.append(
                f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') "
                f"WHERE (source IS NULL OR source = 'aggregator_sync') AND {_UNIVERSE}"
            )
        else:
            queries.append(
                f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') "
                f"WHERE {_UNIVERSE}"
            )
    return " UNION ALL BY NAME ".join(queries)


def _build_frame_uncached():
    import duckdb
    con = duckdb.connect()
    db = SessionLocal()

    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        union_sql = _archive_union_sql(con)

        # Deep >=$1 items only: first_day before 2026 is the archive-side proxy
        # for the `is_backfilled` gate (data-inventory.md §2 -- the pre-2026
        # cohort is exactly 5,542 items, matching the recorded prod count).
        rows = con.sql(f"""
            SELECT item_slug, COUNT(DISTINCT day) AS n_days
            FROM ({union_sql})
            GROUP BY item_slug
            HAVING COUNT(DISTINCT day) >= {MIN_ITEM_DAYS}
               AND MEDIAN(mean_price) >= {MIN_MEDIAN_PRICE}
               AND MIN(day) < DATE '2026-01-01'
            -- item_slug breaks n_days ties. Without it DuckDB's parallel top-N
            -- picks a different universe (and a different item order, which
            -- feeds LightGBM's row sampling) run to run.
            ORDER BY n_days DESC, item_slug
            LIMIT {N_UNIVERSE}
        """).fetchall()
        slugs = [r[0] for r in rows]
        # The guard runs BEFORE the log line. When the 2026-08-08 archive
        # migration emptied this query, `rows[0]` raised `IndexError: list
        # index out of range` from inside the *logging*, which buried the
        # explicit diagnostic three lines below it and cost a debugging pass.
        # An empty universe is the failure this function most needs to name.
        if len(slugs) < N_EVAL_ITEMS + N_WIDE:
            raise SystemExit(
                f"Universe has {len(slugs)} items but the design needs "
                f"{N_EVAL_ITEMS + N_WIDE} (N_EVAL_ITEMS + N_WIDE). Lower "
                f"MIN_ITEM_DAYS or N_WIDE."
            )
        logger.info(
            f"  Universe: {len(slugs)} deep >=${MIN_MEDIAN_PRICE:.0f} items "
            f"with >={MIN_ITEM_DAYS} days "
            f"(days/item: max {rows[0][1]}, min {rows[-1][1]})"
        )

        placeholders = ", ".join("?" for _ in slugs)
        # `volume` is selected only because engineer_features requires the
        # column to exist. Every feature derived from it is in SHELVED_FEATURES
        # (it has been identically 0 archive-wide since 2026-04-16), so none of
        # them survives the shelving step below.
        all_prices = con.sql(f"""
            SELECT item_slug AS item_id, day AS timestamp,
                   mean_price AS price, volume
            FROM ({union_sql})
            WHERE item_slug IN ({placeholders})
        """, params=slugs).df()

        all_prices["timestamp"] = pd.to_datetime(all_prices["timestamp"])
        all_prices["date"] = all_prices["timestamp"].dt.date
        # Deterministic row order: items in universe order, day-ascending
        # within each item. LightGBM's bagging reads row order.
        all_prices["item_id"] = pd.Categorical(
            all_prices["item_id"], categories=slugs, ordered=True
        )
        all_prices = all_prices.sort_values(
            ["item_id", "timestamp"], kind="stable"
        ).reset_index(drop=True)
        all_prices["item_id"] = all_prices["item_id"].astype(str)
        logger.info(f"  Loaded {len(all_prices):,} price rows")

        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)

        EXCLUDE = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date"}
        numeric = (np.float64, np.float32, np.int64, int, float)
        all_cols = [c for c in df.columns
                    if c not in EXCLUDE and df[c].dtype in numeric]

        # Match production's feature set exactly: shelved columns out, then the
        # group allowlist. Without this the frame carries the eleven dead
        # volume columns (identically 0 from 2026-04-16) and the dollar-scale
        # columns that left the feature set at MODEL_ARTIFACT_VERSION 5.
        kept = [c for c in all_cols if c not in ItemForecaster.SHELVED_FEATURES]
        kept = ItemForecaster._apply_feature_allowlist(
            kept, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
        logger.info(
            f"  Features: {len(all_cols)} engineered -> {len(kept)} after "
            f"shelving + allowlist {ItemForecaster.FEATURE_GROUP_ALLOWLIST}"
        )

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

        # Carry only what the evaluation needs. The engineered frame is ~10x
        # the volume harness's and the unused columns are pure memory.
        df = df[["item_id", "date", "price"] + pruned].copy()
        return df, pruned

    finally:
        con.close()


def assign_items(df):
    """Split the universe into held-out eval items and nested training sets.

    The eval set is drawn at random rather than taken off either end of the
    depth ordering: the deepest and shallowest items in a >=$1 universe are
    different kinds of item (long-lived cases vs. recent knives), and an eval
    cohort taken from one end would not be the population the arms train on.
    """
    slugs = list(pd.unique(df["item_id"]))
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(slugs))
    shuffled = [slugs[i] for i in order]

    eval_items = sorted(shuffled[:N_EVAL_ITEMS])
    pool = shuffled[N_EVAL_ITEMS:]

    arms = {}
    for name, n in ARM_ITEM_COUNTS.items():
        if n > len(pool):
            raise SystemExit(f"arm {name} needs {n} items, pool has {len(pool)}")
        arms[name] = sorted(pool[:n])   # nested by construction

    logger.info(f"  Eval items (held out): {len(eval_items)}")
    for name in ARM_ITEM_COUNTS:
        logger.info(f"  Arm {name:16s}: {len(arms[name]):>4d} training items")
    # Nesting is what makes the contrast "add items" rather than "swap items".
    assert set(arms["narrow"]) <= set(arms["mid"]) <= set(arms["wide"])
    assert not (set(arms["wide"]) & set(eval_items))
    return eval_items, arms


def _stratified_sample(train_df, items, budget, fold_idx):
    """Take up to `budget` rows, spread evenly over `items`.

    Sampling is uniform-at-random within each item rather than the most recent
    rows, so every arm keeps the full calendar window -- the property
    production's `max_feature_rows` subsample is documented to preserve, and
    the one a `tail()` cap silently destroyed once already
    (`2026-07-16-training-window-audit.md`).

    The seed depends on the fold, not the arm. Two arms that both hold an item
    therefore draw the same rows for it wherever their per-item quotas agree,
    so the arms differ by their item set and quota alone.
    """
    if budget is None:
        return train_df[train_df["item_id"].isin(items)]

    sub = train_df[train_df["item_id"].isin(items)]
    if len(sub) <= budget:
        return sub

    per_item = max(1, budget // len(items))
    rng = np.random.default_rng(SAMPLE_SEED + fold_idx)
    keep = []
    for _, idx in sub.groupby("item_id", sort=True).indices.items():
        if len(idx) <= per_item:
            keep.append(idx)
        else:
            keep.append(rng.choice(idx, size=per_item, replace=False))
    picked = np.concatenate(keep)
    picked.sort()
    return sub.iloc[picked]


def run_evaluation(df, pruned, horizon_filter=None, n_jobs=None):
    """Walk-forward over the prebuilt frame. Returns results[horizon][arm]."""
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)

    eval_items, arms = assign_items(df)
    eval_set = set(eval_items)

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

            available = [c for c in pruned if c in tdf.columns]
            sub = tdf[["item_id", "date", "price", target_col] + available]

            dates = sorted(sub["date"].unique())
            split_idx = len(dates) * 2 // 3
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            is_eval = sub["item_id"].isin(eval_set).to_numpy()

            results[horizon] = {}
            for arm, items in arms.items():
                budget = None if arm == "wide_unbudgeted" else ROW_BUDGET
                logger.info(
                    f"\n    --- {arm} ({len(items)} items, "
                    f"budget={budget if budget else 'none'}) ---"
                )
                item_set = set(items)
                is_train_item = sub["item_id"].isin(item_set).to_numpy()

                records = []
                per_fold = []
                n_flat = 0
                for fold_idx, window_end in enumerate(
                        range(split_idx + 1, len(dates), STEP_DAYS)):
                    val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]
                    if len(val_dates) < 7:
                        continue

                    in_train_window = sub_days <= dates_dt[window_end - 1]
                    in_val_window = (
                        (sub_days >= dates_dt[window_end])
                        & (sub_days <= dates_dt[window_end + len(val_dates) - 1])
                    )
                    # Embargo the TRAIN side only, via production's own purge
                    # (see the "Embargo" section of the module docstring). The
                    # val mask above is untouched: purging it would shrink the
                    # 21-day window and empty it outright at 30d.
                    train_df = ItemForecaster._purge_overlapping_train_rows(
                        sub[in_train_window & is_train_item], val_dates[0], horizon)
                    # Eval rows are the SAME in every arm -- held-out items
                    # only. That is what lets paired_da_difference pair on
                    # (item_id, forecast_date) rather than compare two pooled
                    # numbers with overlapping intervals.
                    val_df = sub[in_val_window & is_eval]
                    if len(val_df) < 50 or train_df.empty:
                        continue

                    train_df = _stratified_sample(
                        train_df, items, budget, fold_idx)

                    train_median = train_df[available].median()
                    X_train = train_df[available].fillna(train_median).values
                    y_train = train_df[target_col].values
                    X_val = val_df[available].fillna(train_median).values
                    y_val = val_df[target_col].values

                    dtrain = lgb.Dataset(X_train, y_train, params=DS_PARAMS,
                                         free_raw_data=False)
                    dval = lgb.Dataset(X_val, y_val, reference=dtrain,
                                       params=DS_PARAMS, free_raw_data=False)
                    params = {
                        "objective": "quantile", "alpha": 0.5,
                        "metric": "quantile", "boosting_type": "gbdt",
                        "num_leaves": 31, "max_depth": 5,
                        "min_data_in_leaf": 15, "min_gain_to_split": 0.1,
                        "learning_rate": 0.03, "feature_fraction": 0.7,
                        "bagging_fraction": 0.7, "bagging_freq": 5,
                        "lambda_l1": 0.5, "lambda_l2": 0.5,
                        "verbosity": -1, "random_state": 42, "n_jobs": n_jobs,
                        # Pinned for the same reason as the volume harness:
                        # LightGBM's auto row/col-wise choice picks col-wise at
                        # low thread counts on this shape and costs ~6x.
                        "force_row_wise": True,
                        **DS_PARAMS,
                    }
                    model = lgb.train(
                        params, dtrain, num_boost_round=100,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(15, verbose=False),
                                   lgb.log_evaluation(0)],
                    )
                    pred = model.predict(X_val)

                    # Strict >=$1 scoring. Flat-actual rows are excluded: an
                    # exactly-zero forward return makes sign(actual)=0, which a
                    # q50 matches by emitting 0.0, and counting that as a hit
                    # is what inflated the old penny-cohort DA by ~31pp.
                    price = val_df["price"].to_numpy(dtype=float)
                    actual = np.asarray(y_val, dtype=float)
                    match = (np.sign(np.nan_to_num(actual))
                             == np.sign(np.nan_to_num(pred)))
                    scored = (actual != 0) & (price >= 1.0)
                    n_flat += int((~(actual != 0)).sum())

                    ids = val_df["item_id"].to_numpy()
                    dts = val_df["date"].to_numpy()
                    for i in np.flatnonzero(scored):
                        records.append({
                            "item_id": ids[i],
                            "forecast_date": str(dts[i]),
                            # The resampling cluster. Dates inside one 21-day
                            # validation window come from a single fitted
                            # model, so clustering on them understates the
                            # variance -- see paired_mde's module docstring,
                            # which names this experiment as one to re-derive.
                            "fold_id": fold_idx,
                            "direction_correct": bool(match[i]),
                        })
                    hits = int(np.count_nonzero(match & scored))
                    tot = int(scored.sum())
                    per_fold.append({
                        "fold": len(per_fold) + 1,
                        "val_start": str(val_dates[0]),
                        "val_end": str(val_dates[-1]),
                        "n_train": len(train_df),
                        "n_train_items": int(train_df["item_id"].nunique()),
                        "n_scored": tot,
                        "dir_acc": round(hits / tot * 100, 2) if tot else None,
                    })

                if not per_fold:
                    logger.warning(f"    {arm}: no usable folds")
                    continue

                scored_n = sum(f["n_scored"] for f in per_fold)
                hits_n = sum(int(r["direction_correct"]) for r in records)
                accs = [f["dir_acc"] for f in per_fold if f["dir_acc"] is not None]
                results[horizon][arm] = {
                    "n_items": len(items),
                    "dir_acc_strict_ge1": round(hits_n / scored_n * 100, 2) if scored_n else None,
                    "n_scored": scored_n,
                    "pct_flat_dropped": round(100.0 * n_flat / max(1, n_flat + scored_n), 2),
                    "fold_count": len(per_fold),
                    "fold_mean": round(float(np.mean(accs)), 2) if accs else None,
                    "fold_std": round(float(np.std(accs)), 2) if len(accs) > 1 else None,
                    "mean_rows_per_fold": int(np.mean([f["n_train"] for f in per_fold])),
                    "mean_rows_per_item": round(
                        float(np.mean([f["n_train"] / max(1, f["n_train_items"])
                                       for f in per_fold])), 1),
                    "per_fold": per_fold,
                    "records": records,
                }
                logger.info(
                    f"      DA(strict,>=$1)={results[horizon][arm]['dir_acc_strict_ge1']}% "
                    f"n={scored_n:,} folds={len(per_fold)} "
                    f"rows/fold={results[horizon][arm]['mean_rows_per_fold']:,} "
                    f"rows/item={results[horizon][arm]['mean_rows_per_item']}"
                )

            # Paired contrasts against `narrow`, which is the arm that stands
            # in for production's current depth-heavy universe.
            base = results[horizon].get("narrow")
            if base:
                results[horizon]["_paired_vs_narrow"] = {}
                for arm in results[horizon]:
                    if arm in ("narrow",) or arm.startswith("_"):
                        continue
                    results[horizon]["_paired_vs_narrow"][arm] = (
                        paired_da_difference(base["records"],
                                             results[horizon][arm]["records"],
                                             cluster_key="fold_id")
                    )
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 92)
    print("BREADTH vs DEPTH AT A FIXED ROW BUDGET — DA(strict, >=$1) on held-out items")
    print("=" * 92)
    for h in sorted(k for k in results if isinstance(k, int)):
        r = results[h]
        print(f"\n  {h}d horizon")
        print(f"    {'arm':<18} {'items':>6} {'rows/fold':>10} {'rows/item':>10} "
              f"{'DA':>7} {'folds':>6} {'n':>9}")
        print(f"    {'-' * 74}")
        for arm in ("narrow", "mid", "wide", "wide_unbudgeted"):
            a = r.get(arm)
            if not a:
                continue
            print(f"    {arm:<18} {a['n_items']:>6} {a['mean_rows_per_fold']:>10,} "
                  f"{a['mean_rows_per_item']:>10.1f} "
                  f"{a['dir_acc_strict_ge1']:>6.2f}% {a['fold_count']:>6} "
                  f"{a['n_scored']:>9,}")
        paired = r.get("_paired_vs_narrow", {})
        if paired:
            print(f"\n    paired vs narrow (dates clustered, held-out items):")
            for arm, p in paired.items():
                ci = (f"[{p['ci_lower_pp']:+.2f}, {p['ci_upper_pp']:+.2f}]"
                      if p.get("ci_lower_pp") is not None else "n/a")
                mde = f"{p['mde_pp']:.2f}" if p.get("mde_pp") is not None else "n/a"
                print(f"      {arm:<18} {p['mean_diff_pp']:+.2f}pp  95% CI {ci}"
                      f"  MDE {mde}pp  n={p['n_paired']:,} over {p['n_dates']} dates")
    print("")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="A/B: more items vs more history at a fixed row budget")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: training breadth vs depth at a fixed row budget")
    logger.info("=" * 70)

    df, pruned = build_frame(cache_path=args.frame_cache)
    if args.build_cache_only:
        logger.info("Frame cache built; exiting before evaluation.")
        return 0

    results = run_evaluation(df, pruned, horizon_filter=args.horizon,
                            n_jobs=args.n_jobs)

    if args.out:
        # Per-row records are what make the pairing possible but they dominate
        # the file; keep them out of the summary artifact.
        slim = {}
        for h, r in results.items():
            slim[h] = {}
            for arm, a in r.items():
                slim[h][arm] = ({k: v for k, v in a.items() if k != "records"}
                                if isinstance(a, dict) and "records" in a else a)
        Path(args.out).write_text(json.dumps(slim, indent=2, default=str))
        logger.info(f"Wrote {args.out}")

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
