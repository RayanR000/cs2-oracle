#!/usr/bin/env python3
"""A/B time-decayed sample weights (SAMPLE_WEIGHT_HALFLIFE_DAYS) on the
PRODUCTION training frame, over production purge-gap CV.

Motivation (diagnosed 2026-07-29)
---------------------------------
Production trains on a 1460-day window in which only ~14.5% of rows fall in the
last 180 days, while the early-stopping validation window (most recent 30 days)
has ~2x the return spread of the training set overall (std 35.1 vs 21.9,
p10/p90 -14.4/+20.1 vs -9.1/+10.0). The 7d q50 validation curve is therefore
nearly flat (0.28% total improvement) and its early-stopping round is
noise-determined: 32/16/94 across the three ensemble seeds, sd 41 on mean 47,
reproducing the shipped 41/11/89.

Ruled out first: target leakage at the un-purged train/val split (only 0.62% of
train rows leak; purging changed nothing) and insufficient data (400K rows /
540 items made stopping *worse*, 66/185/1).

Arms (identical data, folds, and tree params — only the weight vector differs):
  - **flat**  : SAMPLE_WEIGHT_HALFLIFE_DAYS = 0 (control, pre-2026-07-29)
  - **decay** : SAMPLE_WEIGHT_HALFLIFE_DAYS = 365

Pre-registered gate (decided before running)
--------------------------------------------
SHIP decay for a horizon iff ALL of:
  1. mean paired pinball improvement >= 0.5% relative, AND
  2. decay wins on pinball in >= half the paired folds, AND
  3. directional accuracy does not regress by more than 0.5pp.
Same gate as the q50 A/B, for comparability.

Runs on the frame built by build_training_data(days_back=1460,
backfilled_only=True) — 141 items / ~115K rows — NOT a 200-item proxy. The q50
A/B used 876K-row full-history frames, a data regime production never sees;
this harness deliberately does not.

Usage:
    python -m scripts.ab_test_recency_weights [--horizon 7] [--max-folds 8]
        --frame /path/prod_frame.parquet --cols /path/prod_frame.cols.json
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import models.forecaster as fmod
import numpy as np
import pandas as pd
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import fold_level_records
from models.forecaster import ItemForecaster, embargo_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ab_test_recency_weights")

META_PATH = Path(__file__).parent.parent / "models" / "saved_models" / "meta.json"
HORIZONS = [3, 7, 14, 30]
QUANTILES = (0.1, 0.5, 0.9)
ARMS = {"flat": 0.0, "decay": 365.0}
# Third arm computed specially in run_horizon: the decay recency-MULTIPLIER
# permuted across rows per fold, on top of flat's vol/direction weighting. It
# holds decay's weight DISTRIBUTION but decouples it from recency, so a real
# recency effect must beat it. Guards the fold-count floor (this harness runs
# ~8 folds; see .claude/rules/ab-statistics.md — read a placebo before shipping).
ARM_ORDER = ["flat", "decay", "placebo"]
PLACEBO_SEED = 20260814
CONTROL = "flat"
MIN_TRAIN_ROWS, MIN_VAL_ROWS, MAX_FOLDS = 2000, 200, 8
GATE_MIN_REL_PINBALL_GAIN = 0.005
GATE_MAX_DA_REGRESSION_PP = 0.5
SINGLE_MEMBER_FEATURE_FRACTION = 0.7

# Boosting rounds per model. Was `DART_NUM_BOOST_ROUND if dart else 1000`;
# DART is gone from the forecaster, so only the GBDT arm of that branch
# survives.
NUM_BOOST_ROUND = 1000


def pinball(y, p, a):
    d = np.asarray(y, float) - np.asarray(p, float)
    return float(np.mean(np.maximum(a * d, (a - 1.0) * d)))


def directional_accuracy(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = y != 0
    return float((np.sign(p[m]) == np.sign(y[m])).mean()) if m.any() else float("nan")


def load_params(horizon, q):
    tp = json.load(open(META_PATH))["tuned_params"]
    src = tp.get(str(horizon), {})
    p = src.get(str(q)) or src.get(q)
    if not p:
        # Production's minimal model tunes only q0.5 and serves the band from a
        # conformal q_hat*sigma, not separately-tuned q10/q90 boosters. Reuse
        # the q0.5 tree params for the band quantiles so the harness runs; the
        # ship gate is q50 pinball + DA either way.
        p = src.get("0.5") or src.get(0.5)
    if not p:
        raise RuntimeError(f"no persisted params for {horizon}d q{q}")
    return dict(p)


def run_horizon(fc, tdf, feat_cols, horizon, max_folds):
    target = f"target_return_{horizon}d"
    dates = np.array(sorted(tdf["date"].unique()))
    folds = fc._compute_cv_splits(dates, purge_days=embargo_days(horizon))
    boosting = fc.BOOSTING_TYPE
    if not folds:
        logger.warning(f"  no folds for {horizon}d")
        return
    folds = folds[-max_folds:]
    # Production's per-horizon table. This harness's motivation above is that
    # the early-stopping round was noise-determined (32/16/94 across seeds);
    # production removed the mechanism on 2026-08-08, so that premise now
    # describes history. The weight question it tests is unaffected.
    nbr = ItemForecaster._boost_rounds(horizon, cv=True)
    base = {q: load_params(horizon, q) for q in QUANTILES}
    logger.info(f"  {horizon}d: {len(folds)} folds, boosting={boosting}")

    for fi, (tr_dates, va_dates) in enumerate(folds):
        tr = tdf[tdf["date"].isin(tr_dates)]
        va = tdf[tdf["date"].isin(va_dates)]
        if len(tr) < MIN_TRAIN_ROWS or len(va) < MIN_VAL_ROWS:
            logger.info(f"    fold {fi}: skipped ({len(tr)}/{len(va)})")
            continue
        X_tr = tr[feat_cols].replace([np.inf, -np.inf], np.nan)
        med = X_tr.median()
        X_tr = X_tr.fillna(med)
        X_va = va[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(med)
        y_tr = tr[target].to_numpy(float)
        y_va = va[target].to_numpy(float)

        # Weights per arm. flat/decay differ only by the half-life the weight
        # helper reads (patched around weight computation only). placebo keeps
        # flat's vol/direction weighting but scrambles the recency multiplier.
        def _weights(hl, frame):
            prev = fmod.SAMPLE_WEIGHT_HALFLIFE_DAYS
            fmod.SAMPLE_WEIGHT_HALFLIFE_DAYS = hl
            try:
                return fc._compute_sample_weights(frame, horizon)
            finally:
                fmod.SAMPLE_WEIGHT_HALFLIFE_DAYS = prev

        w_flat_tr, w_flat_va = _weights(0.0, tr), _weights(0.0, va)
        w_decay_tr, w_decay_va = _weights(365.0, tr), _weights(365.0, va)
        if w_flat_tr is None or w_decay_tr is None:
            logger.info(f"    fold {fi}: skipped (no weights)")
            continue

        # Permuted-weight placebo. ratio = decay/flat is the renormalised
        # recency multiplier 0.5^(age/hl); permuting it row-wise (fixed seed per
        # fold) preserves its distribution while breaking its alignment to age.
        rng = np.random.default_rng(PLACEBO_SEED + fi)

        def _placebo(w_flat, w_decay):
            ratio = w_decay / np.maximum(w_flat, 1e-12)
            w = w_flat * rng.permutation(ratio)
            return (w / max(float(np.mean(w)), 1e-8)).astype(np.float32)

        weights = {
            "flat": (w_flat_tr, w_flat_va),
            "decay": (w_decay_tr, w_decay_va),
            "placebo": (_placebo(w_flat_tr, w_decay_tr), _placebo(w_flat_va, w_decay_va)),
        }

        for arm in ARM_ORDER:
            w_tr, w_va = weights[arm]
            t0 = time.time()
            preds, iters = {}, {}
            for q in QUANTILES:
                p = dict(base[q])
                p.update(
                    objective="quantile",
                    alpha=q,
                    metric="quantile",
                    boosting_type=boosting,
                    max_bin=fc.MAX_BIN,
                    feature_pre_filter=False,
                    verbosity=-1,
                    device="cpu",
                    feature_fraction=SINGLE_MEMBER_FEATURE_FRACTION,
                )
                ds = {"max_bin": fc.MAX_BIN, "feature_pre_filter": False}
                dtr = lgb.Dataset(X_tr, y_tr, params=ds, weight=w_tr)
                dva = lgb.Dataset(X_va, y_va, reference=dtr, params=ds, weight=w_va)
                m = ItemForecaster._train_ensemble_member(
                    p, dtr, dva, num_boost_round=nbr, early_stopping=ItemForecaster._early_stopping_enabled()
                )
                bi = m.best_iteration or m.num_trees()
                preds[q] = m.predict(X_va, num_iteration=bi)
                iters[q] = bi
            fit_s = time.time() - t0

            rec = {
                "horizon": horizon,
                "arm": arm,
                "fold": fi,
                "pinball_q50": pinball(y_va, preds[0.5], 0.5),
                "pinball_mean": float(np.mean([pinball(y_va, preds[q], q) for q in QUANTILES])),
                "da": directional_accuracy(y_va, preds[0.5]),
                "iter_q10": iters[0.1],
                "iter_q50": iters[0.5],
                "iter_q90": iters[0.9],
                "fit_s": round(fit_s, 1),
                "n_train": len(tr),
                "n_val": len(va),
            }
            logger.info(
                f"    fold {fi} {arm:>5}: q50_pinball={rec['pinball_q50']:.5f} "
                f"mean={rec['pinball_mean']:.5f} da={100 * rec['da']:.2f}% "
                f"iters={iters[0.1]}/{iters[0.5]}/{iters[0.9]} ({fit_s:.0f}s)"
            )
            yield rec


def summarize(records):
    df = pd.DataFrame(records)
    if df.empty:
        print("\nno results")
        return df, {}
    print("\n" + "=" * 96)
    print("PAIRED RESULTS — recency decay vs flat weights (same folds)")
    print("=" * 96)
    print(
        f"{'h':>3} {'arm':>6} {'q50_pinball':>12} {'mean_pinball':>13} {'da%':>7} {'iter q10/q50/q90':>20} {'folds':>6}"
    )
    verdicts = {}
    for h in sorted(df.horizon.unique()):
        hd = df[df.horizon == h]
        for arm in ARM_ORDER:
            a = hd[hd.arm == arm]
            if a.empty:
                continue
            print(
                f"{h:>3} {arm:>8} {a.pinball_q50.mean():>12.5f} "
                f"{a.pinball_mean.mean():>13.5f} {100 * a.da.mean():>7.2f} "
                f"{a.iter_q10.mean():>6.0f}/{a.iter_q50.mean():.0f}/"
                f"{a.iter_q90.mean():<.0f}{'':>4} {len(a):>6}"
            )
        c = hd[hd.arm == CONTROL].set_index("fold")

        def _contrast(arm_name):
            """Fold-clustered paired interval on the q50 pinball difference vs
            flat. A win count is not a test; lower pinball is better, so a
            positive verdict needs the interval strictly below zero."""
            t = hd[hd.arm == arm_name].set_index("fold")
            common = sorted(set(c.index) & set(t.index))
            if not common:
                return None
            cc, tt = c.loc[common], t.loc[common]
            gain = ((cc.pinball_q50 - tt.pinball_q50) / cc.pinball_q50).mean()
            won = int((tt.pinball_q50 < cc.pinball_q50).sum())
            da_pp = 100 * (tt.da.mean() - cc.da.mean())
            paired = paired_arm_contrasts(
                {
                    "flat": fold_level_records(common, cc.pinball_q50, metric="pinball"),
                    arm_name: fold_level_records(common, tt.pinball_q50, metric="pinball"),
                },
                base="flat",
                value_key="pinball",
                scale=1.0,
                higher_is_better=False,
            )[arm_name]
            return {
                "gain": gain,
                "won": won,
                "n": len(common),
                "da_pp": da_pp,
                "paired": paired,
                "iter_sd": tt.iter_q50.std(),
            }

        decay = _contrast("decay")
        placebo = _contrast("placebo")
        if decay is None:
            continue

        g1 = decay["gain"] >= GATE_MIN_REL_PINBALL_GAIN
        g2 = decay["paired"]["verdict"] == "positive"
        g3 = decay["da_pp"] >= -GATE_MAX_DA_REGRESSION_PP
        # The placebo must NOT itself clear the paired bar; if it does, the
        # "gain" is weight-variance/capacity, not recency structure.
        g4 = (placebo is None) or (placebo["paired"]["verdict"] != "positive")
        ship = bool(g1 and g2 and g3 and g4)
        verdicts[h] = {
            "ship": ship,
            "rel_gain": decay["gain"],
            "folds_won": decay["won"],
            "n_folds": decay["n"],
            "da_delta_pp": decay["da_pp"],
            "paired_pinball": decay["paired"],
            "placebo_paired": placebo["paired"] if placebo else None,
            "placebo_gain": placebo["gain"] if placebo else None,
            "iter_q50_sd_flat": c.iter_q50.std(),
            "iter_q50_sd_decay": decay["iter_sd"],
        }
        print(
            f"  -> {h}d: q50 pinball {decay['gain'] * 100:+.2f}% [{'PASS' if g1 else 'FAIL'}] | "
            f"paired {format_paired(decay['paired'], unit='')} [{'PASS' if g2 else 'FAIL'}] | "
            f"DA {decay['da_pp']:+.2f}pp [{'PASS' if g3 else 'FAIL'}] | "
            f"folds {decay['won']}/{decay['n']} (context, not a gate)"
        )
        if placebo is not None:
            print(
                f"     PLACEBO {placebo['gain'] * 100:+.2f}% "
                f"paired {format_paired(placebo['paired'], unit='')} "
                f"[{'null=PASS' if g4 else 'POSITIVE=FAIL'}]"
            )
        print(f"     q50 stopping-round sd: flat={c.iter_q50.std():.0f} decay={decay['iter_sd']:.0f}")
        print(f"     VERDICT {h}d: {'SHIP decay' if ship else 'KEEP flat'}")
    return df, verdicts


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--cols", required=True)
    ap.add_argument("--horizon", type=int, default=None, choices=HORIZONS)
    ap.add_argument("--max-folds", type=int, default=MAX_FOLDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fc = ItemForecaster.__new__(ItemForecaster)
    # `__new__` skips `__init__`; `prepare_targets` now records into
    # `self.label_voiding` (added after this harness's bypass was written), so
    # seed the same empty dict `__init__` would have.
    fc.label_voiding = {}
    df = pd.read_parquet(args.frame)
    feat_cols = [c for c in json.loads(Path(args.cols).read_text()) if c in df.columns]
    logger.info(f"frame {df.shape}, {len(feat_cols)} features, items={df['item_id'].nunique()}")

    records = []
    t0 = time.time()
    for h in [args.horizon] if args.horizon else HORIZONS:
        tdf = ItemForecaster.prepare_targets(fc, df, h)
        tdf = tdf.dropna(subset=[f"target_return_{h}d"]).sort_values(["item_id", "date"]).copy()
        if tdf.empty:
            continue
        records += list(run_horizon(fc, tdf, feat_cols, h, args.max_folds))
    out, _ = summarize(records)
    logger.info(f"total wall clock: {time.time() - t0:.0f}s")
    if args.out and not out.empty:
        out.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
