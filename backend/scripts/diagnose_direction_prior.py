#!/usr/bin/env python3
"""Step 0 gate for the directional prior correction.

Tests one falsifiable claim: that the classifier's unconditional "down" bias
is inherited from its training class prior.

  training prior up/down SKEWED   -> prior correction is the right fix
  training prior up/down BALANCED -> the bias is serve-time, not the prior.
                                     STOP. Do not build the correction.

Read-only. Derives forward labels from the persisted engineered frame by
reproducing build_training_data's date-shifted merge, so it needs no feature
rebuild.

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
    print(f"Engineered frame: {len(df):,} rows, "
          f"{df['item_id'].nunique():,} items, "
          f"{df['date'].min()} .. {df['date'].max()}\n")

    f = ItemForecaster(db_session=MagicMock())
    f.load_models()

    print(f"{'h':>3} {'pi_down':>8} {'pi_flat':>8} {'pi_up':>8} "
          f"{'down/up':>8} | {'served_down':>11} {'served_flat':>11} {'served_up':>10}")
    print("-" * 84)

    verdicts = {}
    for h in ItemForecaster.HORIZONS:
        y = forward_return(df, h)
        prior = f._direction_class_prior(
            y, DIRECTION_FLAT_TOLERANCE_PCT,
            ItemForecaster.DIRECTION_MOVER_WEIGHT_MAP.get(h, 3.0))
        if not prior:
            print(f"{h:>3} prior not estimable")
            continue

        ratio = prior[0] / prior[2] if prior[2] > 0 else float("inf")
        verdicts[h] = ratio

        served = "  (no classifier)"
        clf = f.direction_models.get(h)
        if clf is not None:
            # load_models restores feature_cols from meta.json. Require ALL of
            # them: predicting on a silently-truncated subset would produce a
            # served distribution that is not the served one. Read the
            # parquet schema only (no data) to check availability, since the
            # frame is 1.8GB/184 columns and this machine has 24GB RAM with a
            # documented history of OOM kills in the prediction path.
            feats = list(f.feature_cols)
            available = pq.ParquetFile(FRAME).schema_arrow.names
            missing = [c for c in feats if c not in available]
            if not feats:
                served = "  (no feature_cols in meta.json)"
            elif missing:
                served = f"  (frame missing {len(missing)} feature cols)"
            else:
                full = pd.read_parquet(FRAME, columns=feats)
                cls = clf.predict(full[feats]).argmax(axis=1)
                n = len(cls)
                served = (f"{(cls == 0).mean() * 100:10.1f}% "
                          f"{(cls == 1).mean() * 100:10.1f}% "
                          f"{(cls == 2).mean() * 100:9.1f}%  (n={n:,})")

        print(f"{h:>3} {prior[0]:8.4f} {prior[1]:8.4f} {prior[2]:8.4f} "
              f"{ratio:8.3f} | {served}")

    print()
    skewed = {h: r for h, r in verdicts.items()
              if r > SKEW_RATIO_THRESHOLD or r < 1.0 / SKEW_RATIO_THRESHOLD}
    if skewed:
        print(f"VERDICT: PROCEED. Prior is up/down skewed beyond "
              f"{SKEW_RATIO_THRESHOLD:.2f}x at horizons "
              f"{sorted(skewed)} (down/up ratios "
              f"{ {h: round(r, 3) for h, r in sorted(skewed.items())} }).")
        print("         Build the correction (Tasks 5-7).")
    else:
        print(f"VERDICT: STOP. No horizon's prior is skewed beyond "
              f"{SKEW_RATIO_THRESHOLD:.2f}x, so the training prior does not "
              f"explain the observed down-bias.")
        print("         Do NOT build the correction. Ship Tasks 1-2, write the "
              "diagnostic up as a changelog entry, and stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
