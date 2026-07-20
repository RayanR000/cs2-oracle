#!/usr/bin/env python3
"""
Retroactive bias-correction check — threshold-based version.

Fits per-tier/horizon classification THRESHOLDS (not additive shifts) that make
the predicted up/down/flat split match the actual observed base rate, then
evaluates the directional accuracy improvement.

Usage:
    python scripts/retro_bias_check.py                    # use saved bias_corrections.json
    python scripts/retro_bias_check.py --split             # single 50/50 random split (OOS)
    python scripts/retro_bias_check.py --rolling-splits 10 # N random splits (OOS, reports mean±std)
    python scripts/retro_bias_check.py --fit               # in-sample estimate
    python scripts/retro_bias_check.py --additive          # compare old additive method
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
MIN_FLAT_MARGIN = 0.3
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


def enforce_min_margin(t_down, t_up, margin=MIN_FLAT_MARGIN):
    """Enforce minimum flat-zone width by widening thresholds symmetrically."""
    if t_up - t_down >= margin:
        return t_down, t_up
    center = (t_up + t_down) / 2.0
    half = margin / 2.0
    return center - half, center + half


def fit_thresholds(df):
    """Fit threshold-based corrections per tier/horizon with safety guards.

    For each (horizon, tier):
      1. Compute the actual up/down base rate from direction_actual.
      2. Find thresholds (t_down, t_up) on the predicted mid_ret distribution
         that make the predicted split match the actual split.
         t_up = (1 - actual_up_pct) percentile of predicted mid_rets
         t_down = actual_down_pct percentile of predicted mid_rets
      3. Safety guards:
         - n < 20: skip (keep default ±0.5)
         - n < 100: shrink thresholds toward default ±0.5 by (n/100) factor
         - enforce minimum flat-zone width of MIN_FLAT_MARGIN pp
         - if t_down > t_up after enforcement, fall back to defaults
    Returns {horizon: {tier: {"t_down": x, "t_up": y}}}.
    """
    DEFAULTS = {"t_down": -DIRECTION_FLAT_TOLERANCE_PCT, "t_up": DIRECTION_FLAT_TOLERANCE_PCT}
    MIN_MIDRET_IQR = 0.5  # skip if mid_ret spread is too narrow for usable signal
    results = {}
    for (horizon, tier), g in df.groupby(["horizon_days", "tier"]):
        n = len(g)
        if n < 20:
            continue

        mid_rets = g["approx_mid_ret"].values
        if len(mid_rets) == 0:
            continue

        # Skip if predicted mid_ret distribution is too narrow — the
        # model produces no usable directional signal for this tier/horizon.
        iqr = float(np.percentile(mid_rets, 75) - np.percentile(mid_rets, 25))
        if iqr < MIN_MIDRET_IQR:
            continue

        actual_up = (g["direction_actual"] == "up").mean()
        actual_down = (g["direction_actual"] == "down").mean()

        pct_up_target = max(0, min(100, (1 - actual_up) * 100))
        pct_down_target = max(0, min(100, actual_down * 100))
        t_up = float(np.percentile(mid_rets, pct_up_target))
        t_down = float(np.percentile(mid_rets, pct_down_target))

        # Clamp to [-3, 3]
        t_up = max(-3.0, min(3.0, t_up))
        t_down = max(-3.0, min(3.0, t_down))

        # Enforce minimum flat-zone width
        t_down, t_up = enforce_min_margin(t_down, t_up)

        # If after enforcement the order flips, fall back to defaults
        if t_down >= t_up:
            t_down, t_up = DEFAULTS["t_down"], DEFAULTS["t_up"]

        # Conservative shrinkage toward defaults for small samples
        if n < 100:
            shrink = n / 100.0
            t_down = shrink * t_down + (1 - shrink) * DEFAULTS["t_down"]
            t_up = shrink * t_up + (1 - shrink) * DEFAULTS["t_up"]

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


def evaluate(test_df, thresholds, corrections):
    """Apply correction and return test_df with corrected columns."""
    def apply_correction(row):
        horizon = row["horizon_days"]
        tier = row["tier"]
        mid_ret = row["approx_mid_ret"]
        th = thresholds.get(horizon, {}).get(tier) if thresholds else None
        if th:
            return classify(mid_ret, th["t_down"], th["t_up"])
        corr = corrections.get(horizon, {}).get(tier, 0.0)
        return classify(mid_ret + corr)

    out = test_df.copy()
    out["direction_corrected"] = out.apply(apply_correction, axis=1)
    out["correct_original"] = out["direction_correct"]
    out["correct_corrected"] = (out["direction_corrected"] == out["direction_actual"]).astype(int)
    return out


def print_tier_table(out, method):
    print(f"\n{'Method':>8} {'Horizon':>7} {'Tier':>8} {'n':>6} {'Orig DirAcc':>12} {'Corr DirAcc':>17} {'Delta':>8}   {'PredDir orig(u/d/f)':>22}  {'PredDir corr(u/d/f)':>22}  {'Actual(u/d/f)':>18}")
    for (h, t), g in out.groupby(["horizon_days", "tier"]):
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
    inv = out[out["tier"].isin(["$5-20", "$20-100"])]
    for h, g in inv.groupby("horizon_days"):
        n = len(g)
        orig = g["correct_original"].mean() * 100
        corrd = g["correct_corrected"].mean() * 100
        print(f"  {h}d: n={n:<5} orig={orig:.1f}% corrected={corrd:.1f}% ({corrd - orig:+.1f}pp)")


def print_rolling_summary(results_by_horizon):
    """Print mean ± std across rolling splits for $5-100 investment tier."""
    print("\nRolling split summary ($5-100 combined, mean ± std across splits):")
    print(f"  {'Horizon':>7}  {'n':>6}  {'Orig DirAcc':>14}  {'Corr DirAcc':>18}  {'Delta':>12}")
    for h in sorted(results_by_horizon.keys()):
        entries = results_by_horizon[h]
        n = int(np.mean([e["n"] for e in entries]))
        orig_mean = np.mean([e["orig"] for e in entries])
        orig_std = np.std([e["orig"] for e in entries])
        corr_mean = np.mean([e["corr"] for e in entries])
        corr_std = np.std([e["corr"] for e in entries])
        delta_mean = corr_mean - orig_mean
        delta_std = np.std([e["corr"] - e["orig"] for e in entries])
        print(f"  {h:>7d}  {n:>6}  {orig_mean:>6.1f}% ± {orig_std:<4.1f}%  "
              f"{corr_mean:>6.1f}% ± {corr_std:<4.1f}%  "
              f"{delta_mean:>+5.1f}pp ± {delta_std:.1f}pp")


def run_single_split(df, seed, args_additive):
    """Run a single 50/50 random split with the given seed."""
    n = len(df)
    half = n // 2
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    fit_df = df.iloc[idx[:half]]
    test_df = df.iloc[idx[half:]]

    if args_additive:
        corrections = fit_additive_correction(fit_df)
        thresholds = {}
        method = "additive"
    else:
        thresholds = fit_thresholds(fit_df)
        corrections = {}
        method = "threshold"

    out = evaluate(test_df, thresholds, corrections)

    results = {}
    inv = out[out["tier"].isin(["$5-20", "$20-100"])]
    for h, g in inv.groupby("horizon_days"):
        orig = g["correct_original"].mean() * 100
        corr = g["correct_corrected"].mean() * 100
        results[h] = {"n": len(g), "orig": orig, "corr": corr}
    return out, method, results, thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true",
                     help="Single 50/50 random split (out-of-sample)")
    ap.add_argument("--rolling-splits", type=int, default=0,
                     help="Number of random 50/50 splits for OOS validation (reports mean±std)")
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

    # Rolling splits mode
    if args.rolling_splits > 0:
        print(f"Rolling splits: {args.rolling_splits} x 50/50 random splits\n")
        all_results_by_horizon = defaultdict(list)
        final_out = None
        for i in range(args.rolling_splits):
            seed = 42 + i * 13
            out, method, results, thresholds = run_single_split(df, seed, args.additive)
            for h, r in results.items():
                all_results_by_horizon[h].append(r)
            if i == 0:
                final_out = out
                final_thresholds = thresholds
        print_rolling_summary(dict(all_results_by_horizon))

        # Show thresholds from the first split
        if final_thresholds:
            print(f"\nThresholds from first split ({method}):")
            for h in sorted(final_thresholds.keys()):
                for t in sorted(final_thresholds[h].keys()):
                    th = final_thresholds[h][t]
                    print(f"  {h:2d}d  {t:>8}:  t_down={th['t_down']:+.2f}  t_up={th['t_up']:+.2f}")
        return

    # Single split mode
    if args.split:
        n = len(df)
        half = n // 2
        idx = np.random.RandomState(42).permutation(n)
        fit_df = df.iloc[idx[:half]]
        test_df = df.iloc[idx[half:]]
        print(f"Single split: fit on {len(fit_df)} rows, test on {len(test_df)} rows")

        if args.additive:
            corrections = fit_additive_correction(fit_df)
            thresholds = {}
            method = "additive"
        else:
            thresholds = fit_thresholds(fit_df)
            corrections = {}
            method = "threshold"
    elif args.fit:
        if args.additive:
            corrections = fit_additive_correction(df)
            thresholds = {}
            method = "additive"
        else:
            thresholds = fit_thresholds(df)
            corrections = {}
            method = "threshold"
        test_df = df
        print(f"Fit on all {len(test_df)} rows, evaluate in-sample")
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

    out = evaluate(test_df, thresholds, corrections)
    print_tier_table(out, method)

    if thresholds:
        print("\nFitted thresholds (t_down / t_up):")
        for h in sorted(thresholds.keys()):
            for t in sorted(thresholds[h].keys()):
                th = thresholds[h][t]
                print(f"  {h:2d}d  {t:>8}:  t_down={th['t_down']:+.2f}  t_up={th['t_up']:+.2f}")


if __name__ == "__main__":
    main()
