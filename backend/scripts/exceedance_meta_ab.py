#!/usr/bin/env python3
"""A/B test: does ByMykel item metadata help the EXCEEDANCE head?

`docs/changelog/2026-08-06-bymykel-metadata-refuted.md` measured the nine
ByMykel columns against the **centre** — directional accuracy of the q50 — and
found nothing, which is why `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`
still stands. `EXCEEDANCE_META=1` exists in `forecaster.py` on the argument that
the exceedance head asks a different question, and so a null on the centre does
not transfer:

  - the centre predicts a SIGNED return, and `centre-shrinkage-lambda-is-zero`
    plus `composite-centre-ranks-cannot-scale` say that target is essentially
    unpredictable here at any scale;
  - the exceedance head predicts P(|move| > round-trip cost), an UNSIGNED
    magnitude, and that is the one signal on this panel that has ever measured
    real and date-stable.

Rarity, StatTrak/souvenir, float range and crate/collection are plausible
magnitude priors (a covert knife is not as calm as a consumer-grade case) while
being useless for sign. So this harness scores the head production actually
fits, on the label production actually uses, and nothing else.

## What is held fixed

The fold machinery, item split, row budget and purge are imported wholesale
from `ab_test_item_metadata` rather than re-derived. That harness family has
already been wrong twice from a second implementation of a shared rule — once
on the train universe (2026-08-08) and once on the embargo (2026-08-07) — and
the point of this script is a *paired* read, which only holds if both arms see
identical rows.

The model is production's own `ItemForecaster._fit_exceedance_classifier`, so
the served-cohort reweighting and the NaN-label drop are the deployed ones. The
arms differ ONLY in which columns enter `X`.

## Arms

    baseline:   the pruned price_technicals columns, nothing added
    treatment:  + all nine metadata columns
    placebo:    + all nine, permuted per fold in train AND val
    static_only: + rarity, StatTrak and souvenir ONLY. These are static per item
                and low-cardinality, so they cannot address an individual item
                the way crate/collection can, and they are the three with a
                magnitude mechanism. Without this arm a null on `treatment`
                cannot separate "no prior here" from "a real prior, drowned by
                two ID columns the trees memorised through".

`placebo` is not optional here. Nine extra columns widen the model whatever
they contain, and `type_meta_crate_id` / `type_meta_collection_id` carry
~110-300 levels over ~870 items, so a tree can memorise a per-item base rate
through them. treatment-vs-placebo is the contrast that separates a real prior
from capacity, and treatment-vs-baseline alone cannot.

## Metrics

AUC and log loss on the validation slice, per fold, both cohorts. AUC is the
ranking read (does the head order calm items below volatile ones better?) and
log loss is the calibration read — the head is served as a probability and used
as a band scale, so an arm that ranks better while calibrating worse is not an
improvement. Reported as a PAIRED per-fold delta with a t-interval over folds,
because folds share a fitted model and the absolute level moves far more
between folds than between arms.

Usage:
    python -m scripts.exceedance_meta_ab --horizon 7 \
        --frame-cache /tmp/exc_meta_frame.parquet --out /tmp/exc_meta.json
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from database import SessionLocal
from models.forecaster import ItemForecaster
from scripts.ab_test_item_metadata import (
    META_ALL,
    META_STATIC,
    PLACEBO_SEED,
    ROW_BUDGET,
    STEP_DAYS,
    VAL_WINDOW_DAYS,
    _stratified_sample,
    assign_items,
    build_frame,
)
from sklearn.metrics import log_loss, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("exceedance_meta_ab")

# Same tree shape both arms, and the same shape the directional/exceedance
# heads get in production via `_direction_tree_params`. Spelled out rather than
# read from a fitted p50 config so a fold that fails to tune cannot silently
# hand the two arms different trees.
TREE_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_data_in_leaf": 15,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
}


def paired_fold_deltas(base_folds, arm_folds, key):
    """Mean per-fold (arm - baseline) delta on `key`, with a 95% t-interval.

    Pairs on fold index, drops any fold either arm could not score, and returns
    None when fewer than two paired folds survive — a single fold has no
    interval, and reporting its delta as a result is how an underpowered read
    gets mistaken for a verdict (see `ab-family-was-never-powered`).
    """
    base = {f["fold"]: f.get(key) for f in base_folds}
    deltas = [f[key] - base[f["fold"]] for f in arm_folds if f.get(key) is not None and base.get(f["fold"]) is not None]
    if len(deltas) < 2:
        return None
    d = np.asarray(deltas, dtype=float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    # Normal critical value: with <30 folds this interval is optimistic, which
    # is stated rather than corrected because the decision rule below is
    # "CI must exclude zero", and an optimistic interval makes that rule
    # harder to satisfy honestly, not easier.
    half = 1.96 * se
    return {
        "n_folds": len(d),
        "mean": round(mean, 5),
        "ci_low": round(mean - half, 5),
        "ci_high": round(mean + half, 5),
        "wins": int((d > 0).sum()),
        "excludes_zero": bool((mean - half) * (mean + half) > 0),
    }


def fold_tally(metric_name, d):
    """Fold count in the metric's OWN good direction.

    `wins` counts folds where the delta is positive, which is the good
    direction for AUC and the BAD one for log loss. Printing one label for both
    reads a 22-of-26 calibration improvement as "wins 4/26".
    """
    better = (d["n_folds"] - d["wins"]) if "logloss" in metric_name else d["wins"]
    return f"better {better}/{d['n_folds']}"


def _score(y, p):
    """AUC and log loss, or None when the slice is single-class.

    A single-class validation slice has no AUC. Returning None rather than 0.5
    keeps it out of the paired mean instead of dragging the mean toward no
    effect.
    """
    y = np.asarray(y, dtype=float)
    keep = ~np.isnan(y)
    y, p = y[keep].astype(int), np.asarray(p, dtype=float)[keep]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None, None, len(y)
    return (float(roc_auc_score(y, p)), float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), len(y))


def run(df, pruned, meta_present, horizon_filter=None, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    eval_items, train_items, trained_eval = assign_items(df)
    meta_all = [c for c in META_ALL if c in meta_present]
    if not meta_all:
        raise SystemExit(
            "No metadata columns present in the frame — the ByMykel join came "
            "back empty, so there is nothing to test. Check "
            "price-archive/item-metadata-bymykel.parquet."
        )
    meta_static = [
        c for c in META_STATIC if c in meta_present and c not in ("type_meta_crate_id", "type_meta_collection_id")
    ]
    arms = {"baseline": [], "treatment": meta_all, "placebo": meta_all, "static_only": meta_static}

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        horizons = [h for h in ItemForecaster.HORIZONS if horizon_filter is None or h == horizon_filter]
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Exceedance {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            target_col = f"target_exceed_{horizon}d"
            if target_col not in tdf.columns:
                logger.warning(f"    {target_col} absent — skipping")
                continue
            tdf = tdf.dropna(subset=[target_col]).sort_values(["item_id", "date"])
            if tdf.empty:
                logger.warning(f"    No valid exceedance labels for {horizon}d")
                continue

            base_cols = [c for c in pruned if c in tdf.columns]
            keep = ["item_id", "date", "price", target_col] + base_cols + meta_all
            sub = tdf[[c for c in keep if c in tdf.columns]].copy()

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
                logger.info(f"\n    --- {arm} ({len(features)} features) ---")
                per_fold = []
                for fold_idx, window_end in enumerate(range(split_idx + 1, len(dates), STEP_DAYS)):
                    val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                    if len(val_dates) < 7:
                        continue
                    in_train = sub_days <= dates_dt[window_end - 1]
                    in_val = (sub_days >= dates_dt[window_end]) & (
                        sub_days <= dates_dt[window_end + len(val_dates) - 1]
                    )
                    train_df = ItemForecaster._purge_overlapping_train_rows(
                        sub[in_train & is_train_item], val_dates[0], horizon
                    )
                    val_df = sub[in_val & (is_heldout | is_trained_eval)]
                    if len(val_df) < 50 or train_df.empty:
                        continue
                    train_df = _stratified_sample(train_df, train_items, ROW_BUDGET, fold_idx)

                    if arm == "placebo" and extra:
                        rng = np.random.default_rng(PLACEBO_SEED)
                        train_df, val_df = train_df.copy(), val_df.copy()
                        for col in extra:
                            train_df[col] = rng.permutation(train_df[col].values)
                            val_df[col] = rng.permutation(val_df[col].values)

                    med = train_df[features].median()
                    X_train = train_df[features].fillna(med)
                    X_val = val_df[features].fillna(med)
                    params = dict(TREE_PARAMS, n_jobs=n_jobs)
                    head = forecaster._fit_exceedance_classifier(
                        X_train,
                        train_df[target_col].to_numpy(),
                        "gbdt",
                        params,
                        horizon=horizon,
                        tier_train=None,
                        num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                    )
                    if head is None:
                        continue
                    p = head.predict(X_val)
                    ids = val_df["item_id"].to_numpy()
                    held = np.isin(ids, eval_items)
                    y = val_df[target_col].to_numpy(dtype=float)
                    price = val_df["price"].to_numpy(dtype=float)

                    row = {"fold": fold_idx, "val_start": str(val_dates[0]), "n_train": len(train_df)}
                    # Served cohort only (>=$1), matching every other read on
                    # this panel; a sub-$1 tail has its own exceedance regime.
                    for cohort, mask in (("heldout", held), ("trained", ~held)):
                        sel = mask & (price >= 1.0)
                        auc, ll, n = _score(y[sel], p[sel])
                        row[f"{cohort}_auc"] = auc
                        row[f"{cohort}_logloss"] = ll
                        row[f"{cohort}_n"] = n
                    per_fold.append(row)

                if not per_fold:
                    logger.warning(f"    {arm}: no usable folds")
                    continue
                results[horizon][arm] = {"n_features": len(features), "per_fold": per_fold}
                for cohort in ("heldout", "trained"):
                    aucs = [f[f"{cohort}_auc"] for f in per_fold if f[f"{cohort}_auc"] is not None]
                    logger.info(
                        f"      {cohort}: mean AUC={np.mean(aucs):.4f} over {len(aucs)} folds"
                        if aucs
                        else f"      {cohort}: no scorable fold"
                    )

            base = results[horizon].get("baseline")
            if not base:
                continue
            paired = {}
            for arm in ("treatment", "placebo", "static_only"):
                if arm not in results[horizon]:
                    continue
                paired[arm] = {
                    f"{cohort}_{metric}": paired_fold_deltas(
                        base["per_fold"], results[horizon][arm]["per_fold"], f"{cohort}_{metric}"
                    )
                    for cohort in ("heldout", "trained")
                    for metric in ("auc", "logloss")
                }
            if "treatment" in results[horizon] and "placebo" in results[horizon]:
                paired["treatment_vs_placebo"] = {
                    f"{cohort}_{metric}": paired_fold_deltas(
                        results[horizon]["placebo"]["per_fold"],
                        results[horizon]["treatment"]["per_fold"],
                        f"{cohort}_{metric}",
                    )
                    for cohort in ("heldout", "trained")
                    for metric in ("auc", "logloss")
                }
            results[horizon]["_paired"] = paired
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 78)
    print("EXCEEDANCE HEAD x ByMykel METADATA — paired per-fold deltas")
    print("higher AUC is better; LOWER log loss is better")
    print("=" * 78)
    for horizon, arms in sorted(results.items()):
        print(f"\nh={horizon}d")
        for arm, entry in sorted(arms.items()):
            if arm == "_paired":
                continue
            print(f"  {arm:22s} {entry['n_features']} features, {len(entry['per_fold'])} folds")
        for arm, metrics in (arms.get("_paired") or {}).items():
            print(f"  vs {arm}:")
            for name, d in sorted(metrics.items()):
                if d is None:
                    print(f"    {name:20s} — too few paired folds")
                    continue
                flag = "*" if d["excludes_zero"] else " "
                print(
                    f"    {name:20s} {d['mean']:+.5f} "
                    f"[{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]{flag} "
                    f"{fold_tally(name, d)}"
                )
    print(
        "\n* = 95% interval excludes zero. A verdict needs treatment to beat "
        "\n  BOTH baseline and placebo, on AUC and log loss, in the same "
        "direction."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--metadata-parquet", default=None)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    df, pruned, meta_present = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    if args.build_cache_only:
        logger.info("  Frame cache built; exiting before evaluation.")
        return
    results = run(df, pruned, meta_present, horizon_filter=args.horizon, n_jobs=args.n_jobs)
    print_summary(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
