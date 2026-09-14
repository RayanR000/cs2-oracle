#!/usr/bin/env python3
"""
A/B test: compare prediction accuracy WITH vs WITHOUT supply-side features
(rarity, weapon_type, weapon-type cross-sectional group features).

Trains two models on the same expanding-window walk-forward evaluation:
  - Model A (control):  no supply-side features
  - Model B (treatment): with supply-side features

Usage:
    python scripts/ab_test_supply_side.py [--max-items 200]

Embargo (added 2026-08-08):
    This harness had **no purge gap at all** before that date: the fold split
    was `train = every date <= window_end - 1`, so every training row within
    `horizon` days of the boundary carried a label resolved from inside the
    validation window. The train side now goes through production's own
    `ItemForecaster._purge_overlapping_train_rows`, which embargoes
    `embargo_days(horizon)` = `horizon + 13` days -- the label's resolved-anchor
    support, not its nominal date. The validation window is untouched; purging
    it would empty the 21-day window at h=30.

    **Every delta this harness printed before 2026-08-08 is un-embargoed.**
    How much that inflated them is not known here. The often-quoted
    "+12.1pp unpurged -> +6.1pp purged" at h=30 is from the external review
    (docs/research/2026-08-07-cs2-forecasting-research.md), describes an
    event-calendar arm, and has never been replicated in this repo.
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import (
    paired_records,
    without_records,
)
from database import SessionLocal
from db.archive import prices_relation
from models.forecaster import ItemForecaster, archive_universe_sql_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ab_test_supply_side")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Production's item universe and price consensus, spelled into every archive
# read this harness makes. Before 2026-08-08 this was a raw
# `read_parquet('prices-*.parquet')` glob, which DuckDB narrows to the first
# file's schema so `source` is not even a column there, and it carried no
# universe rule at all — so the BUFF bid voted straight into its price series.
# `prices_relation` is the reader that projects `source` correctly whether or
# not the archive has been migrated. See `backend/AGENTS.md` and `db/archive.py`.
#
# Still outstanding: this reads EVERY ask source, where most harnesses filter
# to one, so a 2026 item-day reaches it several times over and
# `engineer_features` collapses the copies with a plain mean rather than
# production's outlier-voted median. `backend/AGENTS.md` puts the archive's
# duplication at 1.37x item-days.
_PRICE_COLUMNS = ["item_slug", "day", "mean_price", "volume", "source"]
_UNIVERSE = archive_universe_sql_filter()

# Dataset params production and every other harness train with. Notably
# `max_bin=63` — LightGBM defaults to 255, a finer-binned, different model.
# `feature_pre_filter=False` keeps columns LightGBM would otherwise drop before
# the split search. See ab_test_csfloat_basis.py / item_metadata.py.
DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}


def _frame_fingerprint():
    """Identify the feature-producing code so a cached frame can't outlive it.

    Hashes forecaster.py content plus this harness's frame-shaping constants,
    exactly as ab_test_price_primitives._frame_fingerprint does — a data-only
    key cannot see a code change.
    """
    import hashlib

    src = (Path(__file__).parent.parent / "models" / "forecaster.py").read_bytes()
    h = hashlib.sha256(src)
    h.update(repr((_UNIVERSE, DS_PARAMS, "supply-side-rarity-q50-v1")).encode())
    return h.hexdigest()


def _build_frame_uncached(max_items):
    """Build the shared feature frame ONCE, with rarity present.

    All three arms read this identical frame, so `prepare_targets` yields the
    same dates and the fold grids match across arms — a prerequisite for paired
    scoring. Building per-arm frames drifted the placebo grid by a day and gave
    zero shared folds. Returns (df, base_cols, rarity_cols): base_cols is the
    production set (shelve + allowlist, rarity stripped, corr-pruned among
    itself); rarity_cols is the tested addition, kept whole. The arm picks which.
    """
    import duckdb

    con = duckdb.connect()
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        relation = prices_relation(con, ARCHIVE_DIR, columns=_PRICE_COLUMNS)
        # Universe: backfilled (pre-2026, source IS NULL) items on the >=$1
        # served cohort; the >=$1 median floor is lifted here from scoring so the
        # model trains on the population it is scored on. See 2026-08-13 repin.
        rows = con.sql(f"""
            SELECT item_slug, MIN(day) AS first_day, MAX(day) AS last_day,
                   COUNT(*) AS row_count
            FROM {relation}
            WHERE {_UNIVERSE}
              AND item_slug IN (
                SELECT DISTINCT item_slug FROM {relation} WHERE source IS NULL)
            GROUP BY item_slug
            HAVING row_count >= 90 AND MEDIAN(mean_price) >= 1.0
            ORDER BY row_count DESC, item_slug
            LIMIT {max_items}
        """).fetchall()
        if not rows:
            raise RuntimeError(
                "supply_side universe query selected 0 items — the source pin or "
                "the >=$1 floor matched no rows (see 2026-08-13 harness repin)."
            )
        logger.info(f"  {len(rows)} items for evaluation")

        all_rows = []
        for item_slug, _, _, _ in rows:
            item_rows = con.sql(
                f"""
                SELECT item_slug AS item_id, day AS timestamp,
                       mean_price AS price, volume
                FROM {relation}
                WHERE item_slug = ? AND {_UNIVERSE}
                ORDER BY day
            """,
                params=[item_slug],
            ).fetchall()
            item_df = pd.DataFrame(item_rows, columns=["item_id", "timestamp", "price", "volume"])
            item_df["timestamp"] = pd.to_datetime(item_df["timestamp"])
            item_df["date"] = item_df["timestamp"].dt.date
            all_rows.append(item_df)
        all_prices = pd.concat(all_rows, ignore_index=True)

        # Always build WITH rarity so every arm shares one frame. The model
        # production serves is the 33-col production base (shelve + allowlist,
        # which strips the item_identity group engineer_features emits natively);
        # rarity_* is the tested addition, isolated so the contrast is rarity
        # alone rather than the whole item_identity group. See 2026-08-13 repin.
        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)

        exclude = {"item_id", "date", "timestamp", "price", "volume", "name", "release_date"}
        feature_cols = [
            c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.float32, np.int64, int, float)
        ]
        feature_cols = [c for c in feature_cols if c not in ItemForecaster.SHELVED_FEATURES]
        rarity_cols = [c for c in feature_cols if c.startswith("rarity_")]
        base_cols = ItemForecaster._apply_feature_allowlist(
            [c for c in feature_cols if not c.startswith("rarity_")], ItemForecaster.FEATURE_GROUP_ALLOWLIST
        )

        # Correlation-prune the production base among itself (control's set, and
        # treatment's base). Rarity is the tested addition and is kept whole, as
        # ab_test_csfloat_basis keeps its basis columns.
        if len(base_cols) > 2:
            corr = df[base_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                if col in to_drop:
                    continue
                to_drop.update(upper[col][upper[col] > 0.95].index)
            base_cols = [c for c in base_cols if c not in to_drop]
        logger.info("  %d production base + %d rarity columns", len(base_cols), len(rarity_cols))

        keep = ["item_id", "date", "price", "volume"] + base_cols + rarity_cols
        keep = [c for c in dict.fromkeys(keep) if c in df.columns]
        return df[keep].copy(), base_cols, rarity_cols
    finally:
        con.close()


def build_frame(max_items=200, cache_path=None):
    """Build (or load) the shared frame. Mirrors ab_test_price_primitives."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != _frame_fingerprint():
                raise SystemExit(
                    f"Frame cache {cache_path} was built from different feature code. Rebuild with --build-cache-only."
                )
            if meta.get("max_items") != max_items:
                raise SystemExit(
                    f"Frame cache {cache_path} was built with max_items={meta.get('max_items')}, not {max_items}."
                )
            df = pd.read_parquet(cache_path)
            logger.info(f"  Loaded cached frame {cache_path} ({len(df):,} rows)")
            return df, meta["base_cols"], meta["rarity_cols"]

    df, base_cols, rarity_cols = _build_frame_uncached(max_items)
    if cache_path is not None:
        # Write-then-rename, parquet before the meta that vouches for it.
        tmp_frame = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
        tmp_meta = cache_path.with_suffix(f".{os.getpid()}.tmp.json")
        df.to_parquet(tmp_frame, index=False)
        tmp_meta.write_text(
            json.dumps(
                {
                    "fingerprint": _frame_fingerprint(),
                    "max_items": max_items,
                    "base_cols": base_cols,
                    "rarity_cols": rarity_cols,
                    "rows": len(df),
                }
            )
        )
        os.replace(tmp_frame, cache_path)
        os.replace(tmp_meta, cache_path.with_suffix(".meta.json"))
        logger.info(f"  Wrote frame cache {cache_path} ({len(df):,} rows)")
    return df, base_cols, rarity_cols


def run_evaluation(df, base_cols, rarity_cols, arm, horizon_filter=None):
    """Walk-forward evaluation of one arm on the shared frame.

    arm ∈ {control, treatment, placebo}. control uses the production base only;
    treatment adds the rarity columns; placebo adds them but permutes each within
    train and val per fold — the capacity control the 2026-08-13 leak audit
    requires (a real rarity effect must beat shuffled rarity, not just control).
    """
    placebo = arm == "placebo"
    feature_cols = list(base_cols) + (list(rarity_cols) if arm != "control" else [])
    logger.info(
        "  arm=%s: %d features (%d base%s)",
        arm,
        len(feature_cols),
        len(base_cols),
        f" + {len(rarity_cols)} rarity" if arm != "control" else "",
    )

    try:
        forecaster = ItemForecaster(db_session=None)

        results_by_horizon = {}
        horizons = ItemForecaster.HORIZONS if horizon_filter is None else [horizon_filter]
        for horizon in horizons:
            logger.info(f"\n  Evaluating {horizon}d horizon...")

            tdf = forecaster.prepare_targets(df, horizon)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values(["item_id", "date"])

            if tdf.empty:
                logger.warning(f"    No valid targets for {horizon}d")
                continue

            dates = sorted(tdf["date"].unique())
            split_idx = len(dates) * 2 // 3

            directional_hits = 0
            directional_total = 0
            mae_total = 0.0
            mae_count = 0
            per_fold = []
            records = []

            VAL_WINDOW_DAYS = 21
            step = 60
            for window_end in range(split_idx + 1, len(dates), step):
                train_dates = dates[:window_end]
                val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue

                train_df = ItemForecaster._purge_overlapping_train_rows(
                    tdf[tdf["date"].isin(train_dates)], val_dates[0], horizon
                )
                val_df = tdf[tdf["date"].isin(val_dates)]

                if len(val_df) < 50:
                    continue

                if len(train_df) > 200000:
                    train_df = train_df.sort_values("date").tail(200000)

                fc = [c for c in feature_cols if c in tdf.columns]
                if not fc:
                    continue

                X_train = train_df[fc].fillna(train_df[fc].median())
                y_train = train_df[f"target_return_{horizon}d"]
                X_val = val_df[fc].fillna(train_df[fc].median())
                y_val = val_df[f"target_return_{horizon}d"]

                if placebo:
                    # Destroy the rarity signal while keeping the columns'
                    # capacity: permute each within train and val. Seeded per
                    # fold so all three arms score the same rows.
                    rng = np.random.default_rng(20260813 + window_end)
                    supply_cols = [c for c in fc if c.startswith("rarity_")]
                    for col in supply_cols:
                        X_train[col] = rng.permutation(X_train[col].to_numpy())
                        X_val[col] = rng.permutation(X_val[col].to_numpy())

                # q50 only + conformal is the served model, so the p10/p90
                # quantile GBMs (a scheme production abandoned) are not trained —
                # they cost 2x the compute and feed only an interval-coverage
                # metric that is not this A/B's verdict. Matches the trainer
                # convention in ab_test_csfloat_basis.py / item_metadata.py:
                # hardcoded HP + DS_PARAMS (max_bin=63) + force_row_wise.
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
                    "n_jobs": -1,
                    "force_row_wise": True,
                    **DS_PARAMS,
                }
                ds_kw = {"params": DS_PARAMS, "free_raw_data": False}
                dtrain = lgb.Dataset(X_train.values, y_train.values, **ds_kw)
                dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain, **ds_kw)
                # Production's trainer and round table. The old call early-stopped
                # on `dval` and scored `X_val` — the same rows. `dval` is ignored
                # unless EARLY_STOPPING=1.
                model = ItemForecaster._train_ensemble_member(
                    params,
                    dtrain,
                    dval,
                    num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                    early_stopping=ItemForecaster._early_stopping_enabled(),
                )
                p50_ret = model.predict(X_val.values)

                current_prices = val_df["price"].values
                actual_returns = y_val.values

                fold_hits = 0
                fold_total = 0
                fold_mae = 0.0
                for i in range(len(val_df)):
                    mid_r = p50_ret[i]
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

                # Paired records for the fold-clustered interval below.
                # Scored on non-flat actuals at >=$1, which is the population
                # production serves and is arm-independent, so the two runs
                # pair row for row. `window_end` rather than a running counter:
                # a counter drifts the moment one arm skips a fold the other
                # kept.
                _match = np.sign(np.nan_to_num(actual_returns)) == np.sign(np.nan_to_num(p50_ret))
                _scored = (np.asarray(actual_returns) != 0) & (np.asarray(current_prices, dtype=float) >= 1.0)
                records.extend(
                    paired_records(
                        item_ids=val_df["item_id"].to_numpy(),
                        forecast_dates=val_df["date"].to_numpy(),
                        fold_id=window_end,
                        keep=_scored,
                        direction_correct=_match,
                    )
                )

                per_fold.append(
                    {
                        "fold": len(per_fold) + 1,
                        "val_start": str(val_dates[0]),
                        "val_end": str(val_dates[-1]),
                        "dir_acc": round(fold_hits / fold_total * 100, 1) if fold_total > 0 else 0,
                        "mae": round(fold_mae / fold_total, 4) if fold_total > 0 else 0,
                        "n": fold_total,
                    }
                )

            if directional_total > 0:
                dir_acc = directional_hits / directional_total * 100
                mae = mae_total / mae_count if mae_count > 0 else 0

                fold_accs = [f["dir_acc"] for f in per_fold]
                baseline_2class = 50.0
                result = {
                    "directional_accuracy": round(dir_acc, 2),
                    "mae": round(mae, 4),
                    "sample_count": directional_total,
                    "effective_baseline": baseline_2class,
                    "fold_count": len(per_fold),
                    # Row-grain pairing input for the in-process `main` contrast.
                    "records": records,
                    # Fold-grain pairing input (val_start + dir_acc) that survives
                    # `without_records` into --out, so sharded arms can be paired
                    # by scripts/merge_supply_side_ab.py.
                    "per_fold": per_fold,
                    "fold_mean_dir_acc": round(np.mean(fold_accs), 1) if fold_accs else 0,
                    "fold_std_dir_acc": round(np.std(fold_accs), 1) if len(fold_accs) > 1 else 0,
                    "fold_min_dir_acc": round(min(fold_accs), 1) if fold_accs else 0,
                    "fold_max_dir_acc": round(max(fold_accs), 1) if fold_accs else 0,
                }
                result["improvement_over_baseline_pp"] = round(dir_acc - baseline_2class, 1)
                results_by_horizon[horizon] = result

                logger.info(
                    f"  === {horizon}d: DirAcc={dir_acc:.1f}% "
                    f"({directional_total:,} samples, "
                    f"{result['improvement_over_baseline_pp']:.1f}pp above baseline)"
                )

        return results_by_horizon

    finally:
        # The forecaster holds no external resource here (db_session=None); the
        # try/finally preserves the walk-forward body's indentation only.
        pass


_ARMS = ("control", "treatment", "placebo")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=200, help="Number of items to evaluate (default: 200)")
    parser.add_argument(
        "--arm",
        choices=_ARMS,
        default=None,
        help="Run one arm only and write its result to --out. "
        "Shards the run so each arm fits the 30-min cap; "
        "merge with scripts/merge_supply_side_ab.py.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        choices=ItemForecaster.HORIZONS,
        help="Restrict to one horizon (finer shard, if an arm still exceeds the cap).",
    )
    parser.add_argument(
        "--frame-cache",
        default=None,
        help="Path to the shared frame parquet. All arm shards "
        "MUST pass the same one so their fold grids match — "
        "per-arm frame builds drift by a day and never pair.",
    )
    parser.add_argument(
        "--build-cache-only", action="store_true", help="Build --frame-cache and exit (run once before the arm shards)."
    )
    parser.add_argument("--out", default=None, help="Write the (records-stripped) shard JSON here.")
    args = parser.parse_args()

    if args.build_cache_only:
        if not args.frame_cache:
            raise SystemExit("--build-cache-only requires --frame-cache")
        build_frame(max_items=args.max_items, cache_path=args.frame_cache)
        return 0

    df, base_cols, rarity_cols = build_frame(max_items=args.max_items, cache_path=args.frame_cache)

    # ── Sharded path: one arm (optionally one horizon), write JSON ────
    if args.arm is not None:
        logger.info("=" * 70)
        logger.info("A/B TEST (SHARD): arm=%s horizon=%s", args.arm, args.horizon or "all")
        logger.info("=" * 70)
        results = run_evaluation(df, base_cols, rarity_cols, args.arm, horizon_filter=args.horizon)
        payload = without_records({args.arm: results})
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
            logger.info("Wrote %s", args.out)
        else:
            print(f"\n  JSON: {json.dumps(payload, indent=2, default=str)}")
        return 0

    logger.info("=" * 70)
    logger.info("A/B TEST: Supply-Side Features (rarity)")
    logger.info("=" * 70)

    # All three arms share the one frame built above, so their fold grids match.
    logger.info("\n>>> MODEL A (CONTROL): production base only <<<")
    results_without = run_evaluation(df, base_cols, rarity_cols, "control")

    logger.info("\n>>> MODEL B (TREATMENT): base + rarity <<<")
    results_with = run_evaluation(df, base_cols, rarity_cols, "treatment")

    logger.info("\n>>> MODEL C (PLACEBO): base + shuffled rarity <<<")
    results_placebo = run_evaluation(df, base_cols, rarity_cols, "placebo")

    # ── Compare ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("A/B TEST RESULTS — Supply-Side Features")
    print("=" * 70)
    print(f"{'Horizon':>8} | {'Control (w/o)':<20} | {'Treatment (w/)':<20} | {'Delta':>10}")
    print("-" * 8 + " | " + "-" * 20 + " | " + "-" * 20 + " | " + "-" * 10)

    total_delta = 0
    horizon_count = 0

    for h in ItemForecaster.HORIZONS:
        wo = results_without.get(h, {})
        w = results_with.get(h, {})

        wo_acc = wo.get("directional_accuracy", 0)
        w_acc = w.get("directional_accuracy", 0)
        wo_imp = wo.get("improvement_over_baseline_pp", 0)
        w_imp = w.get("improvement_over_baseline_pp", 0)
        wo_samples = wo.get("sample_count", 0)
        w_samples = w.get("sample_count", 0)
        wo_mae = wo.get("mae", 0)
        w_mae = w.get("mae", 0)

        delta = round(w_acc - wo_acc, 2)
        delta_imp = round(w_imp - wo_imp, 1)
        total_delta += delta
        horizon_count += 1

        label_wo = f"{wo_acc:.1f}% (baseline+{wo_imp:.1f}pp)"
        label_w = f"{w_acc:.1f}% (baseline+{w_imp:.1f}pp)"
        delta_str = f"{delta:+.2f}pp"
        if delta > 0:
            delta_str += " ✅"
        elif delta < 0:
            delta_str += " ❌"

        print(f"  {h:>2}d     | {label_wo:<20} | {label_w:<20} | {delta_str:>10}")

    print("-" * 8 + " | " + "-" * 20 + " | " + "-" * 20 + " | " + "-" * 10)

    avg_delta = total_delta / horizon_count if horizon_count > 0 else 0
    summary = "IMPROVEMENT" if avg_delta > 0 else "DEGRADATION" if avg_delta < 0 else "NO CHANGE"
    print(f"\n  Pooled delta (NOT a verdict): {summary} ({avg_delta:+.2f}pp avg)")

    # The verdict. Until 2026-08-08 the line above was it: a mean of per-horizon
    # deltas, with the sign alone deciding "IMPROVEMENT" or "DEGRADATION" and no
    # interval anywhere. The item-level MDE here is 2.21-3.69pp, so that rule
    # called a coin flip either way about half the time.
    print("\n  Paired, fold-clustered (vs control, >=$1 non-flat). A real rarity")
    print("  effect must beat BOTH control and placebo (shuffled rarity):")
    for h in ItemForecaster.HORIZONS:
        base = results_without.get(h, {}).get("records")
        arm = results_with.get(h, {}).get("records")
        plc = results_placebo.get(h, {}).get("records")
        if not base or not arm:
            print(f"    {h:>2}d:  unresolved — one arm produced no scored rows")
            continue
        arms = {"control": base, "treatment": arm}
        if plc:
            arms["placebo"] = plc
        contrasts = paired_arm_contrasts(arms, base="control")
        print(f"    {h:>2}d:  treatment {format_paired(contrasts['treatment'])}")
        if "placebo" in contrasts:
            print(f"          placebo   {format_paired(contrasts['placebo'])}")

    print("\n  Detail:")
    for h in ItemForecaster.HORIZONS:
        wo = results_without.get(h, {})
        w = results_with.get(h, {})
        wo_mae = wo.get("mae", 0)
        w_mae = w.get("mae", 0)
        wo_n = wo.get("sample_count", 0)
        w_n = w.get("sample_count", 0)
        print(f"    {h:>2}d:  Control: MAE=${wo_mae:.2f}  n={wo_n}")
        print(f"           Treat:  MAE=${w_mae:.2f}  n={w_n}")

    print(
        f"\n  JSON: {
            json.dumps(
                without_records(
                    {
                        'control': results_without,
                        'treatment': results_with,
                        'placebo': results_placebo,
                    }
                ),
                indent=2,
            )
        }"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
