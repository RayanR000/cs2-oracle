#!/usr/bin/env python3
"""Is the ANOMALY_GBM head better than knowing the item's own anomaly rate?

`ANOMALY_GBM=1` trains a binary head per horizon on P(|return_h| > 2sigma of the
item's trailing 60-day return std) and serves it as `anomaly_p`. The head is
built, persisted, reloaded and already in the public API schema
(`api/schemas.py:113`), but it has never been measured and has no tests, so the
field ships as a permanent null.

The question is not "does the head beat chance" — a 2sigma label is roughly 5%
positive, and almost anything beats chance on AUC when the base rate varies
across items. The question this project has learned to ask instead
(`climatology-beats-gbm-band`, `gbm-decorative-on-clean-label`, and the
magnitude booster refuted in `magnitude-target-beats-climatology`) is whether
the features add anything over a **featureless per-item rate**. Three times now
the answer has been no, and the pooled constant has twice beaten the per-item
version. So the arms are:

    gbm:         production's `_fit_anomaly_classifier` on the allowlisted
                 price_technicals columns
    item_rate:   the item's own anomaly frequency, measured on the TRAIN fold
                 only and applied unchanged to val. No features, no fitting.
                 Items unseen in train fall back to the pooled rate.
    global_rate: one pooled constant from the train fold. Ranks nothing, so its
                 AUC is 0.5 by construction; it is here for LOG LOSS, which is
                 the metric that matters for a served probability.

A GBM that cannot beat `item_rate` on held-out items is decorative. One that
cannot beat `global_rate` on log loss is worse than decorative, because
`anomaly_p` is served as a calibrated probability and a mis-calibrated alert
flag is actively misleading.

## A caveat on the label itself

`prepare_targets` builds the threshold as `2 * rolling(60).std()` of
`target_return_{h}d.shift(1)`. Every one of those trailing targets resolves h
days AFTER its own row, so for h>1 the threshold is normalised by returns that
overlap the prediction window. That does not corrupt the label as a *target*
— it is a statement about the future either way — but it does mean neighbouring
rows share threshold information, so a date-level feature can bank some of it.
The arms all see the same label, so the comparison holds; the absolute AUC
should not be read as a serving expectation. Flagged rather than fixed: fixing
it is a change to a production label definition and belongs in its own change.

Usage:
    python -m scripts.anomaly_gbm_ab --horizon 7 \
        --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \
        --frame-cache /tmp/exc_meta_frame.parquet --out /tmp/anom_h7.json
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

from scripts.archive.ab_test_item_metadata import (
    ROW_BUDGET,
    STEP_DAYS,
    VAL_WINDOW_DAYS,
    _stratified_sample,
    assign_items,
    build_frame,
)
from scripts.exceedance_meta_ab import TREE_PARAMS, _score, fold_tally, paired_fold_deltas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("anomaly_gbm_ab")


def clean_anomaly_label(tdf, horizon, k=2.0, window=60, min_periods=10):
    """The anomaly label with a threshold that is knowable at the row's date.

    Production normalises by `target_return_{h}d.shift(1).rolling(60).std()`
    (forecaster.py:5369-5371). Every trailing target in that window resolves h
    days AFTER its own row, so at h>1 the threshold is built from returns that
    overlap the prediction window and neighbouring rows share it.

    This rebuilds it from `return_{h}d` — the BACKWARD h-day return, already
    observed at the row's own date — shifted one row and rolled the same way.
    Same units (both are percent, winsorized at ±500%), same k, same window, so
    the only thing that changes is what the threshold is allowed to see.

    Voids are inherited: a NaN target stays NaN, and a row without enough
    trailing history to form a threshold is NaN rather than silently False.
    `tdf` must already be sorted by (item_id, date).
    """
    ret_col = f"return_{horizon}d"
    if ret_col not in tdf.columns:
        raise SystemExit(
            f"{ret_col} absent from the frame — a strictly-prior threshold "
            f"cannot be built at h={horizon}. Rebuild the frame cache."
        )
    prior_std = tdf.groupby("item_id")[ret_col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).std()
    )
    ret = tdf[f"target_return_{horizon}d"]
    anom = (ret.abs() > k * prior_std).astype(float)
    anom[ret.isna().to_numpy()] = np.nan
    anom[prior_std.isna().to_numpy()] = np.nan
    return anom


def item_rate_predictions(train_df, val_df, target_col, min_obs=10):
    """Per-item anomaly rate from train, mapped onto val rows.

    `min_obs` guards the rate of an item with three training rows, which is
    otherwise 0.0 or 1.0 and would dominate the log loss on noise. Items below
    the floor, and items absent from train entirely, take the pooled rate —
    which is the honest thing a featureless serving rule would do.
    """
    g = train_df.groupby("item_id")[target_col]
    rate, n = g.mean(), g.count()
    pooled = float(train_df[target_col].mean())
    usable = rate[n >= min_obs]
    return (val_df["item_id"].map(usable).fillna(pooled).to_numpy(dtype=float), pooled)


def run(df, pruned, horizon_filter=None, n_jobs=None, clean_label=False):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    eval_items, train_items, trained_eval = assign_items(df)

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    if not forecaster.anomaly_gbm_enabled():
        raise SystemExit("ANOMALY_GBM did not take effect — no labels to score.")
    try:
        horizons = [h for h in ItemForecaster.HORIZONS if horizon_filter is None or h == horizon_filter]
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Anomaly {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            target_col = f"target_anomaly_{horizon}d"
            if target_col not in tdf.columns:
                logger.warning(f"    {target_col} absent — skipping")
                continue
            # Sort BEFORE the label rebuild: the rolling threshold is a
            # per-item time series, and dropping rows first would roll over
            # gaps that the production label does not have.
            tdf = tdf.sort_values(["item_id", "date"])
            if clean_label:
                tdf[target_col] = clean_anomaly_label(tdf, horizon)
            tdf = tdf.dropna(subset=[target_col])
            if tdf.empty:
                logger.warning(f"    No valid anomaly labels for {horizon}d")
                continue
            logger.info(
                f"    label base rate: {tdf[target_col].mean():.4f} "
                f"over {len(tdf):,} rows "
                f"({'STRICTLY-PRIOR' if clean_label else 'production'} "
                f"threshold)"
            )

            base_cols = [c for c in pruned if c in tdf.columns]
            sub = tdf[["item_id", "date", "price", target_col, *base_cols]].copy()

            dates = sorted(sub["date"].unique())
            split_idx = len(dates) * 2 // 3
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            is_heldout = sub["item_id"].isin(set(eval_items)).to_numpy()
            is_trained_eval = sub["item_id"].isin(set(trained_eval)).to_numpy()
            is_train_item = sub["item_id"].isin(set(train_items)).to_numpy()

            per_fold = {"gbm": [], "item_rate": [], "global_rate": []}
            for fold_idx, window_end in enumerate(range(split_idx + 1, len(dates), STEP_DAYS)):
                val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue
                in_train = sub_days <= dates_dt[window_end - 1]
                in_val = (sub_days >= dates_dt[window_end]) & (sub_days <= dates_dt[window_end + len(val_dates) - 1])
                train_df = ItemForecaster._purge_overlapping_train_rows(
                    sub[in_train & is_train_item], val_dates[0], horizon
                )
                val_df = sub[in_val & (is_heldout | is_trained_eval)]
                if len(val_df) < 50 or train_df.empty:
                    continue
                train_df = _stratified_sample(train_df, train_items, ROW_BUDGET, fold_idx)

                med = train_df[base_cols].median()
                X_train = train_df[base_cols].fillna(med)
                X_val = val_df[base_cols].fillna(med)
                head = forecaster._fit_anomaly_classifier(
                    X_train,
                    train_df[target_col].to_numpy(),
                    "gbdt",
                    dict(TREE_PARAMS, n_jobs=n_jobs),
                    horizon=horizon,
                    tier_train=None,
                    num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                )
                if head is None:
                    continue

                p_item, pooled = item_rate_predictions(train_df, val_df, target_col)
                preds = {"gbm": head.predict(X_val), "item_rate": p_item, "global_rate": np.full(len(val_df), pooled)}

                ids = val_df["item_id"].to_numpy()
                held = np.isin(ids, eval_items)
                y = val_df[target_col].to_numpy(dtype=float)
                price = val_df["price"].to_numpy(dtype=float)
                for arm, p in preds.items():
                    row = {"fold": fold_idx, "val_start": str(val_dates[0]), "n_train": len(train_df)}
                    for cohort, mask in (("heldout", held), ("trained", ~held)):
                        sel = mask & (price >= 1.0)
                        auc, ll, n = _score(y[sel], p[sel])
                        row[f"{cohort}_auc"] = auc
                        row[f"{cohort}_logloss"] = ll
                        row[f"{cohort}_n"] = n
                    per_fold[arm].append(row)

            if not per_fold["gbm"]:
                logger.warning(f"    no usable folds at {horizon}d")
                continue
            results[horizon] = {arm: {"per_fold": rows} for arm, rows in per_fold.items()}
            for arm, rows in per_fold.items():
                for cohort in ("heldout", "trained"):
                    aucs = [r[f"{cohort}_auc"] for r in rows if r[f"{cohort}_auc"] is not None]
                    lls = [r[f"{cohort}_logloss"] for r in rows if r[f"{cohort}_logloss"] is not None]
                    if aucs:
                        logger.info(
                            f"      {arm:12s} {cohort:8s} AUC={np.mean(aucs):.4f} "
                            f"logloss={np.mean(lls):.5f} ({len(aucs)} folds)"
                        )

            # Every delta is GBM minus a featureless null: positive AUC and
            # NEGATIVE log loss mean the features earned their place.
            results[horizon]["_paired"] = {
                f"gbm_vs_{null}": {
                    f"{cohort}_{metric}": paired_fold_deltas(per_fold[null], per_fold["gbm"], f"{cohort}_{metric}")
                    for cohort in ("heldout", "trained")
                    for metric in ("auc", "logloss")
                }
                for null in ("item_rate", "global_rate")
            }
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 78)
    print("ANOMALY HEAD vs FEATURELESS NULLS — paired per-fold deltas (GBM - null)")
    print("higher AUC is better; LOWER log loss is better")
    print("=" * 78)
    for horizon, entry in sorted(results.items()):
        print(f"\nh={horizon}d   folds={len(entry['gbm']['per_fold'])}")
        for contrast, metrics in sorted(entry["_paired"].items()):
            print(f"  {contrast}:")
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
        "\n* = 95% interval excludes zero. The head earns its place only by "
        "beating\n  item_rate on held-out AUC AND not degrading log loss."
    )


def main():
    # Set here, NOT at import — see anomaly_band_modulator_ab.main.
    os.environ["ANOMALY_GBM"] = "1"
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--metadata-parquet", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--clean-label",
        action="store_true",
        help="rebuild the 2-sigma threshold from the strictly-prior "
        "return_{h}d instead of production's forward-overlapping one",
    )
    args = parser.parse_args()

    df, pruned, _ = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    results = run(df, pruned, horizon_filter=args.horizon, n_jobs=args.n_jobs, clean_label=args.clean_label)
    print_summary(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
