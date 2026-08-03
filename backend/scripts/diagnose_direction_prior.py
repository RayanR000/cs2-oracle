#!/usr/bin/env python3
"""Step 0 gate for the directional prior correction.

Tests one falsifiable claim: that the classifier's unconditional "down" bias
is inherited from its training class prior.

  training prior up/down SKEWED   -> prior correction is the right fix
  training prior up/down BALANCED -> the bias is serve-time, not the prior.
                                     STOP. Do not build the correction.

Read-only. Derives forward labels from the persisted engineered frame by
reproducing prepare_targets's date-shifted merge, so it needs no feature
rebuild.

Row universe measured (read this before trusting the numbers): the frame is
`engineered_data.parquet`, the predict-path feature cache. It is NOT the
frame the classifiers actually trained on. Training additionally applies
dead-item filtering, corrupt-item flagging, stratified subsampling, a
time-based train/val split, and drops rows with a missing target -- none of
that is reproduced here. What this script reports is the class prior over
the persisted predict-path frame, after the same forward-return derivation
and finite-value filter `_direction_class_prior` applies (see the printed
finite-row coverage per horizon) -- not necessarily the exact distribution
LightGBM's objective saw during the historical training run that produced
the saved models.

The "bulk_frame_{down,flat,up}" columns are NOT the production served
distribution and must never be read as such. Production (ItemForecaster.predict)
predicts one row per item -- the latest date, ~8.7k rows -- after reindexing
to the full feature set, replacing +/-inf with NaN, and filling NaN with
training medians. This script instead predicts all ~6.1M historical rows in
this frame, in one bulk in-sample pass, passing raw feature values (including
any NaN/+-inf) straight to the booster with none of that cleanup. These
figures show what the classifier outputs over the whole historical frame;
they are a different measurement from the production per-forecast served
rate and cannot confirm or refute it.

Usage:
    python scripts/diagnose_direction_prior.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from unittest.mock import MagicMock

from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT, ItemForecaster

FRAME = Path(__file__).parent.parent / "models" / "saved_models" / "engineered_data.parquet"
# Skew large enough to be worth correcting. Below this the prior is not the
# explanation for a 57-87% down rate and the hypothesis is refuted.
SKEW_RATIO_THRESHOLD = 1.20


def forward_return(df, horizon):
    """target_return_{h}d, reproducing prepare_targets exactly.

    Date-based merge, not a row shift: row-based shifts give wrong horizons
    wherever an item's series has gaps.
    """
    d = df[["item_id", "date", "price"]].copy()
    d["_dt"] = pd.to_datetime(d["date"])
    future = d[["item_id", "_dt", "price"]].copy()
    future.columns = ["item_id", "date", "target"]
    future["date"] = (future["date"] - pd.Timedelta(days=horizon)).dt.date
    merged = d.merge(future, on=["item_id", "date"], how="left")
    ret = (merged["target"] - merged["price"]) / merged["price"].replace(0, np.nan) * 100
    return ret.clip(-500.0, 500.0).to_numpy(dtype=float)


def main():
    if not FRAME.exists():
        print(f"FAIL: {FRAME} not found. Run a training pass first.")
        return 1

    df = pd.read_parquet(FRAME, columns=["item_id", "date", "price"])
    n_rows = len(df)
    print(f"Engineered frame: {n_rows:,} rows, "
          f"{df['item_id'].nunique():,} items, "
          f"{df['date'].min()} .. {df['date'].max()}\n")
    print("NOTE: this is the predict-path feature cache, not the exact frame "
          "training saw (training adds dead-item filtering, corrupt-item "
          "flagging, stratified subsampling, a time-based split, and target "
          "dropna). See module docstring.")
    print("NOTE: the 'bulk_frame_*' columns below are a bulk in-sample pass "
          "over all historical rows with no inf/NaN cleanup -- they are NOT "
          "the production served distribution (which predicts one "
          "reindexed/cleaned row per item). Do not compare them to the "
          "production 57-87% down-rate observation.\n")

    f = ItemForecaster(db_session=MagicMock())
    f.load_models()

    # feature_cols (from meta.json) is horizon-invariant, so the schema check
    # and the column-restricted data read happen once, outside the horizon
    # loop. Reading it inside the loop -- once per horizon, with the previous
    # iteration's frame still referenced while the next read allocates --
    # roughly doubles peak memory, which is exactly the risk the
    # schema-only/column-restricted read was meant to remove.
    feats = list(f.feature_cols)
    available = pq.ParquetFile(FRAME).schema_arrow.names
    missing = [c for c in feats if c not in available]
    bulk_reason = None
    bulk_full = None
    if not feats:
        bulk_reason = "(no feature_cols in meta.json)"
    elif missing:
        bulk_reason = f"(frame missing {len(missing)} feature cols)"
    else:
        bulk_full = pd.read_parquet(FRAME, columns=feats)

    print(f"{'h':>3} {'pi_down':>8} {'pi_flat':>8} {'pi_up':>8} {'down/up':>8} "
          f"{'finite_n':>10} {'finite_%':>8} | "
          f"{'bulk_down':>10} {'bulk_flat':>10} {'bulk_up':>9}")
    print("-" * 100)

    verdicts = {}
    not_estimable = []
    for h in ItemForecaster.HORIZONS:
        y = forward_return(df, h)
        n_finite = int(np.isfinite(y).sum())
        pct_finite = (n_finite / n_rows * 100.0) if n_rows else 0.0

        prior = f._direction_class_prior(
            y, DIRECTION_FLAT_TOLERANCE_PCT,
            ItemForecaster.DIRECTION_MOVER_WEIGHT_MAP.get(h, 3.0))
        if not prior:
            not_estimable.append(h)
            print(f"{h:>3} prior not estimable (finite_n={n_finite:,}, "
                  f"finite_%={pct_finite:.1f})")
            continue

        ratio = prior[0] / prior[2] if prior[2] > 0 else float("inf")
        verdicts[h] = ratio

        bulk = "  (no classifier)"
        clf = f.direction_models.get(h)
        if clf is not None:
            if bulk_reason is not None:
                bulk = f"  {bulk_reason}"
            else:
                cls = clf.predict(bulk_full[feats]).argmax(axis=1)
                n = len(cls)
                bulk = (f"{(cls == 0).mean() * 100:9.1f}% "
                        f"{(cls == 1).mean() * 100:9.1f}% "
                        f"{(cls == 2).mean() * 100:8.1f}%  (n={n:,})")

        print(f"{h:>3} {prior[0]:8.4f} {prior[1]:8.4f} {prior[2]:8.4f} "
              f"{ratio:8.3f} {n_finite:10,} {pct_finite:7.1f}% | {bulk}")

    print()
    if not_estimable:
        print(f"COVERAGE WARNING: horizon(s) {not_estimable} had no estimable "
              f"prior (see 'prior not estimable' rows above). The verdict "
              f"below covers only horizons {sorted(verdicts)} out of "
              f"{list(ItemForecaster.HORIZONS)} and must NOT be read as a "
              f"full four-horizon conclusion.")

    skewed = {h: r for h, r in verdicts.items()
              if r > SKEW_RATIO_THRESHOLD or r < 1.0 / SKEW_RATIO_THRESHOLD}
    coverage_note = ("" if not not_estimable else
                      f" [PARTIAL COVERAGE: horizons {not_estimable} not "
                      f"estimable, not included in this verdict]")
    if skewed:
        print(f"VERDICT: PROCEED. Prior is up/down skewed beyond "
              f"{SKEW_RATIO_THRESHOLD:.2f}x at horizons "
              f"{sorted(skewed)} (down/up ratios "
              f"{ {h: round(r, 3) for h, r in sorted(skewed.items())} })."
              f"{coverage_note}")
        print("         Build the correction (Tasks 5-7).")
    else:
        print(f"VERDICT: STOP. No horizon's prior is skewed beyond "
              f"{SKEW_RATIO_THRESHOLD:.2f}x, so the training prior does not "
              f"explain the observed down-bias.{coverage_note}")
        print("         Do NOT build the correction. Ship Tasks 1-2, write the "
              "diagnostic up as a changelog entry, and stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
