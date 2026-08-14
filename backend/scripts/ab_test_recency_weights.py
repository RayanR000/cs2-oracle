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
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

import models.forecaster as fmod
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import fold_level_records
from models.forecaster import ItemForecaster, embargo_days

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ab_test_recency_weights")

META_PATH = Path(__file__).parent.parent / "models" / "saved_models" / "meta.json"
HORIZONS = [3, 7, 14, 30]
QUANTILES = (0.1, 0.5, 0.9)
ARMS = {"flat": 0.0, "decay": 365.0}
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
    y = np.asarray(y, float); p = np.asarray(p, float)
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
        med = X_tr.median(); X_tr = X_tr.fillna(med)
        X_va = va[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(med)
        y_tr = tr[target].to_numpy(float); y_va = va[target].to_numpy(float)

        for arm, hl in ARMS.items():
            # The ONLY difference between arms: the half-life the weight
            # helper reads. Patched around weight computation only.
            prev = fmod.SAMPLE_WEIGHT_HALFLIFE_DAYS
            fmod.SAMPLE_WEIGHT_HALFLIFE_DAYS = hl
            try:
                w_tr = fc._compute_sample_weights(tr, horizon)
                w_va = fc._compute_sample_weights(va, horizon)
            finally:
                fmod.SAMPLE_WEIGHT_HALFLIFE_DAYS = prev

            t0 = time.time()
            preds, iters = {}, {}
            for q in QUANTILES:
                p = dict(base[q])
                p.update(objective="quantile", alpha=q, metric="quantile",
                         boosting_type=boosting, max_bin=fc.MAX_BIN,
                         feature_pre_filter=False, verbosity=-1, device="cpu",
                         feature_fraction=SINGLE_MEMBER_FEATURE_FRACTION)
                ds = {"max_bin": fc.MAX_BIN, "feature_pre_filter": False}
                dtr = lgb.Dataset(X_tr, y_tr, params=ds, weight=w_tr)
                dva = lgb.Dataset(X_va, y_va, reference=dtr, params=ds, weight=w_va)
                m = ItemForecaster._train_ensemble_member(
                    p, dtr, dva, num_boost_round=nbr,
                    early_stopping=ItemForecaster._early_stopping_enabled())
                bi = m.best_iteration or m.num_trees()
                preds[q] = m.predict(X_va, num_iteration=bi)
                iters[q] = bi
            fit_s = time.time() - t0

            rec = {"horizon": horizon, "arm": arm, "fold": fi,
                   "pinball_q50": pinball(y_va, preds[0.5], 0.5),
                   "pinball_mean": float(np.mean([pinball(y_va, preds[q], q)
                                                  for q in QUANTILES])),
                   "da": directional_accuracy(y_va, preds[0.5]),
                   "iter_q10": iters[0.1], "iter_q50": iters[0.5],
                   "iter_q90": iters[0.9], "fit_s": round(fit_s, 1),
                   "n_train": len(tr), "n_val": len(va)}
            logger.info(f"    fold {fi} {arm:>5}: q50_pinball={rec['pinball_q50']:.5f} "
                        f"mean={rec['pinball_mean']:.5f} da={100*rec['da']:.2f}% "
                        f"iters={iters[0.1]}/{iters[0.5]}/{iters[0.9]} ({fit_s:.0f}s)")
            yield rec


def summarize(records):
    df = pd.DataFrame(records)
    if df.empty:
        print("\nno results"); return df, {}
    print("\n" + "=" * 96)
    print("PAIRED RESULTS — recency decay vs flat weights (same folds)")
    print("=" * 96)
    print(f"{'h':>3} {'arm':>6} {'q50_pinball':>12} {'mean_pinball':>13} {'da%':>7} "
          f"{'iter q10/q50/q90':>20} {'folds':>6}")
    verdicts = {}
    for h in sorted(df.horizon.unique()):
        hd = df[df.horizon == h]
        for arm in ARMS:
            a = hd[hd.arm == arm]
            if a.empty:
                continue
            print(f"{h:>3} {arm:>6} {a.pinball_q50.mean():>12.5f} "
                  f"{a.pinball_mean.mean():>13.5f} {100*a.da.mean():>7.2f} "
                  f"{a.iter_q10.mean():>6.0f}/{a.iter_q50.mean():.0f}/"
                  f"{a.iter_q90.mean():<.0f}{'':>4} {len(a):>6}")
        c = hd[hd.arm == CONTROL].set_index("fold")
        t = hd[hd.arm == "decay"].set_index("fold")
        common = sorted(set(c.index) & set(t.index))
        if not common:
            continue
        c, t = c.loc[common], t.loc[common]
        gain = ((c.pinball_q50 - t.pinball_q50) / c.pinball_q50).mean()
        won = int((t.pinball_q50 < c.pinball_q50).sum())
        da_pp = 100 * (t.da.mean() - c.da.mean())
        # Fold-clustered paired interval on the pinball difference, replacing
        # the "wins on at least half the folds" condition this gate used until
        # 2026-08-08. A win count is not a test: two arms differing only by
        # seed clear it half the time. Lower pinball is better, so a SHIP needs
        # the interval strictly below zero.
        paired = paired_arm_contrasts(
            {"flat": fold_level_records(common, c.pinball_q50, metric="pinball"),
             "decay": fold_level_records(common, t.pinball_q50, metric="pinball")},
            base="flat", value_key="pinball", scale=1.0,
            higher_is_better=False)["decay"]

        g1 = gain >= GATE_MIN_REL_PINBALL_GAIN
        g2 = paired["verdict"] == "positive"
        g3 = da_pp >= -GATE_MAX_DA_REGRESSION_PP
        ship = bool(g1 and g2 and g3)
        # Stopping-round stability: the defect that motivated this change.
        sd_c, sd_t = c.iter_q50.std(), t.iter_q50.std()
        verdicts[h] = {"ship": ship, "rel_gain": gain, "folds_won": won,
                       "n_folds": len(common), "da_delta_pp": da_pp,
                       "paired_pinball": paired,
                       "iter_q50_sd_flat": sd_c, "iter_q50_sd_decay": sd_t}
        print(f"  -> {h}d: q50 pinball {gain*100:+.2f}% [{'PASS' if g1 else 'FAIL'}] | "
              f"paired {format_paired(paired, unit='')} [{'PASS' if g2 else 'FAIL'}] | "
              f"DA {da_pp:+.2f}pp [{'PASS' if g3 else 'FAIL'}] | "
              f"folds {won}/{len(common)} (context, not a gate)")
        print(f"     q50 stopping-round sd: flat={sd_c:.0f} decay={sd_t:.0f}")
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
    logger.info(f"frame {df.shape}, {len(feat_cols)} features, "
                f"items={df['item_id'].nunique()}")

    records = []
    t0 = time.time()
    for h in ([args.horizon] if args.horizon else HORIZONS):
        tdf = ItemForecaster.prepare_targets(fc, df, h)
        tdf = tdf.dropna(subset=[f"target_return_{h}d"]).sort_values(
            ["item_id", "date"]).copy()
        if tdf.empty:
            continue
        records += list(run_horizon(fc, tdf, feat_cols, h, args.max_folds))
    out, _ = summarize(records)
    logger.info(f"total wall clock: {time.time()-t0:.0f}s")
    if args.out and not out.empty:
        out.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
