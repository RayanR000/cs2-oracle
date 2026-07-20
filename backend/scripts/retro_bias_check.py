#!/usr/bin/env python3
"""
Retroactive bias-correction check — threshold-based version.

Fits per-tier/horizon classification THRESHOLDS (not additive shifts) that make
the predicted up/down/flat split match the actual observed base rate, then
evaluates the directional accuracy improvement.

Usage:
    python scripts/retro_bias_check.py              # use saved bias_corrections.json
    python scripts/retro_bias_check.py --split       # fit on old half, test on new half (OOS)
    python scripts/retro_bias_check.py --additive    # compare old additive method
    python scripts/retro_bias_check.py --split --additive  # OOS comparison of both
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from database import SessionLocal
from sqlalchemy import text

DIRECTION_FLAT_TOLERANCE_PCT = 0.5
PRICE_TIER_BOUNDARIES = [(0, 1, "<$1"), (1, 5, "$1-5"), (5, 20, "$5-20"),
                         (20, 100, "$20-100")]


def get_price_tier(price):
    for lo, hi, label in PRICE_TIER_BOUNDARIES:
        if lo <= price < hi:
            return label
    return ">$100"


def classify(ret, t_down=-DIRECTION_FLAT_TOLERANCE_PCT, t_up=DIRECTION_FLAT_TOLERANCE_PCT):
    if ret > t_up:
        return "up"
    if ret < t_down:
        return "down"
    return "flat"


def fit_thresholds(df):
    """Fit threshold-based corrections per tier/horizon.

    For each (horizon, tier):
      1. Compute the actual up/down base rate from direction_actual.
      2. Find thresholds (t_down, t_up) on the predicted mid_ret distribution
         that make the predicted split match the actual split.
         t_up = (1 - actual_up_pct) percentile of predicted mid_rets
         t_down = actual_down_pct percentile of predicted mid_rets
    Returns {horizon: {tier: {"t_down": x, "t_up": y}}}.
    """
    results = {}
    for (horizon, tier), g in df.groupby(["horizon_days", "tier"]):
        n = len(g)
        if n < 10:
            continue

        actual_up = (g["direction_actual"] == "up").mean()
        actual_down = (g["direction_actual"] == "down").mean()

        mid_rets = g["approx_mid_ret"].values
        if len(mid_rets) == 0:
            continue

        pct_up_target = max(0, min(100, (1 - actual_up) * 100))
        pct_down_target = max(0, min(100, actual_down * 100))
        t_up = float(np.percentile(mid_rets, pct_up_target))
        t_down = float(np.percentile(mid_rets, pct_down_target))



        if t_down > t_up:
            t_down = -DIRECTION_FLAT_TOLERANCE_PCT
            t_up = DIRECTION_FLAT_TOLERANCE_PCT

        # Clamp to [-3, 3] range. Both thresholds can be negative if the
        # model has a systematic directional bias (e.g., predicting too
        # negative while actual outcomes are mostly up).
        t_up = max(-3.0, min(3.0, t_up))
        t_down = max(-3.0, min(3.0, t_down))

        if horizon not in results:
            results[horizon] = {}
        results[horizon][tier] = {"t_down": round(t_down, 2), "t_up": round(t_up, 2)}

    return results


def fit_additive_correction(df):
    """Original additive correction method (baseline for comparison)."""
    corr = defaultdict(dict)
    for (horizon, tier), g in df.groupby(["horizon_days", "tier"]):
        n = len(g)
        if n < 10:
            continue
        up_pct = (g["direction_predicted"] == "up").mean() * 100
        imbalance_pp = up_pct - 50.0
        estimated = -imbalance_pp / 30.0
        estimated = max(-2.0, min(2.0, estimated))
        corr[horizon][tier] = round(estimated, 1)
    return corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true",
                     help="Fit correction on older half of forecast_dates, test on newer half (out-of-sample)")
    ap.add_argument("--fit", action="store_true",
                     help="Fit thresholds on all data (in-sample estimate)")
    ap.add_argument("--additive", action="store_true",
                     help="Use old additive-shift method instead of threshold-based")
    args = ap.parse_args()

    db = SessionLocal()
    rows = db.execute(text("""
        SELECT horizon_days, forecast_date, current_price, predicted_price_mid,
               actual_price, direction_predicted, direction_actual, direction_correct
        FROM forecast_outcomes
        WHERE current_price > 0
          AND direction_actual IS NOT NULL
    """)).fetchall()
    db.close()

    df = pd.DataFrame(rows, columns=[
        "horizon_days", "forecast_date", "current_price", "predicted_price_mid",
        "actual_price", "direction_predicted", "direction_actual", "direction_correct",
    ])
    if df.empty:
        print("No forecast_outcomes rows found.")
        return

    df["tier"] = df["current_price"].apply(get_price_tier)
    df["approx_mid_ret"] = (df["predicted_price_mid"] / df["current_price"] - 1) * 100

    if args.fit or args.split:
        if args.split:
            unique_dates = sorted(df["forecast_date"].unique())
            if len(unique_dates) == 1:
                # Single date — use random 50/50 split instead
                n = len(df)
                half = n // 2
                idx = np.random.RandomState(42).permutation(n)
                fit_df = df.iloc[idx[:half]]
                test_df = df.iloc[idx[half:]]
                print(f"Single date ({unique_dates[0]}): random 50/50 split, "
                      f"fit on {len(fit_df)} rows, test on {len(test_df)} rows")
            else:
                mid = unique_dates[len(unique_dates) // 2]
                fit_df = df[df["forecast_date"] <= mid]
                test_df = df[df["forecast_date"] > mid]
                print(f"Split: fit on {len(fit_df)} rows (<= {mid}), test on {len(test_df)} rows (> {mid})")
        else:
            fit_df = df
            test_df = df
            print(f"Fit on all {len(fit_df)} rows, evaluate in-sample")

        if args.additive:
            corrections = fit_additive_correction(fit_df)
            thresholds = {}
            method = "additive"
        else:
            thresholds = fit_thresholds(fit_df)
            corrections = {}
            method = "threshold"
    else:
        with open("models/saved_models/bias_corrections.json") as f:
            saved = json.load(f)
        corrections = {int(k): v for k, v in saved.get("corrections", {}).items()}
        raw_th = saved.get("thresholds", {})
        thresholds = {int(k): v for k, v in raw_th.items()} if raw_th else {}
        test_df = df
        method = "saved"
        print(f"Using saved bias_corrections.json, evaluating all {len(test_df)} rows "
              f"(IN-SAMPLE — corrections were fit on this data)")

    def apply_correction(row):
        horizon = row["horizon_days"]
        tier = row["tier"]
        mid_ret = row["approx_mid_ret"]

        # Threshold-based correction takes priority
        th = thresholds.get(horizon, {}).get(tier) if thresholds else None
        if th:
            return classify(mid_ret, th["t_down"], th["t_up"])

        # Fall back to additive
        corr = corrections.get(horizon, {}).get(tier, 0.0)
        return classify(mid_ret + corr)

    test_df = test_df.copy()
    test_df["direction_corrected"] = test_df.apply(apply_correction, axis=1)
    test_df["correct_original"] = test_df["direction_correct"]
    test_df["correct_corrected"] = (test_df["direction_corrected"] == test_df["direction_actual"]).astype(int)

    print(f"\n{'Method':>8} {'Horizon':>7} {'Tier':>8} {'n':>6} {'Orig DirAcc':>12} {'Corr DirAcc':>17} {'Delta':>8}   {'PredDir orig(u/d/f)':>22}  {'PredDir corr(u/d/f)':>22}  {'Actual(u/d/f)':>18}")
    for (h, t), g in test_df.groupby(["horizon_days", "tier"]):
        n = len(g)
        orig = g["correct_original"].mean() * 100
        corrd = g["correct_corrected"].mean() * 100
        po = g["direction_predicted"].value_counts()
        pc = g["direction_corrected"].value_counts()
        pa = g["direction_actual"].value_counts()

        def fmt(vc):
            return f"{vc.get('up',0)}/{vc.get('down',0)}/{vc.get('flat',0)}"

        print(f"{method:>8} {h:>7} {t:>8} {n:>6} {orig:>11.1f}% {corrd:>16.1f}% {corrd - orig:>+7.1f}pp   {fmt(po):>22}  {fmt(pc):>22}  {fmt(pa):>18}")

    print("\nInvestment tier ($5-100 combined):")
    inv = test_df[test_df["tier"].isin(["$5-20", "$20-100"])]
    for h, g in inv.groupby("horizon_days"):
        n = len(g)
        orig = g["correct_original"].mean() * 100
        corrd = g["correct_corrected"].mean() * 100
        print(f"  {h}d: n={n:<5} orig={orig:.1f}% corrected={corrd:.1f}% ({corrd - orig:+.1f}pp)")

    # Show threshold values if threshold method was used
    if thresholds:
        print("\nFitted thresholds (t_down / t_up):")
        for h in sorted(thresholds.keys()):
            for t in sorted(thresholds[h].keys()):
                th = thresholds[h][t]
                print(f"  {h:2d}d  {t:>8}:  t_down={th['t_down']:+.2f}  t_up={th['t_up']:+.2f}")


if __name__ == "__main__":
    main()
