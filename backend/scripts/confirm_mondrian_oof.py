#!/usr/bin/env python3
"""Confirm Mondrian-vs-placebo on REAL out-of-fold residuals (h=7/14).

Pre-registration. The model-free probe
(`scripts/measure_qhat_bagging_mondrian.py`, 2026-09-13) found Mondrian
-20–26% narrower at matched coverage with the guardrail passing 4/4 — but
the shuffled placebo matched it on width, voiding the width bar for the
third time. The unconfounded contrast (treatment-vs-placebo LM
sigma-strata err) favoured Mondrian 4/4. This re-runs EXACTLY that
contrast with real residuals: q50 OOF predictions from per-fold LightGBM
models instead of r_hat = 0.

Two phases (run together or separately; scoring re-reads saved OOF):

  Phase A (--phase train): production CV, minus everything that does not
  touch q50 OOF residuals. `build_training_data` with train()'s exact
  arguments, sigma clip exactly as train(), then `_cv_evaluate_horizon`
  per horizon with default q50 params (see below). No Optuna, no
  production fit, no `save_models`, no artifact, no DB writes. OOF rows +
  fold windows are written to --workdir (default /tmp/mondrian_oof) so
  scoring never retrains.

  Phase B (--phase score): the probe's own machinery
  (`CalibPanel`/`fit_s0`/`fit_mondrian`/`evaluate` — same bins, same
  finite-sample levels, same level-match) over OOF rows. Test dates are
  the OOF dates in the last --test-folds folds; calibration is earlier
  OOF rows under the H+13 embargo. Arms: S0 pooled, M vol-decile,
  P shuffled-sigma. No bagged arm: bagging was refuted by the probe.

Production footing, and the two deliberate deltas:
  - Mirrored from price-forecast.yml: FEATURE_NATIVE_NAN=1 (changes fold
    models — must match), EXCEEDANCE_HEAD=1 (residual-neutral; kept so the
    CV is production's), CV_DIAGNOSTIC_CLASSIFIER=0 (residual-relevant:
    OOF residuals are vs the q50 mid, exactly like production's).
    Set via setdefault and logged; explicit env wins.
  - DELTA 1 (unavoidable): no tuned params exist locally (no meta.json),
    so folds train with the SKIP_HP defaults (num_leaves 47, lr 0.01,
    l1 0, l2 1.5, depth 5, min_data 15). Common to all arms. Absolute
    q_hat levels are NOT comparable to production; only the
    treatment-vs-placebo contrast transfers. The probe's validation
    ratio (r_hat=0 vs shipped q_hat: 0.958/0.887 at h=7/14) bounds how
    much the centre can matter to a quantile contrast.
  - DELTA 2 (declared): CONFORMAL_SERVED_BASIS is unset, as in
    production, so residuals are on the raw-anchor training label. The
    geometry stays sigma-symmetric like the probe; transfer to the
    climatology-signed served band remains open after this.

PRE-REGISTERED BAR (confirm): PASS iff ALL of, per horizon,
  (a) LM sigma-strata err: M < P (the primary contrast),
  (b) M1_M inside 80+-3pp raw,
  (c) min LM decile of M >= 70%.
  Width is reported, never gated (voided twice). PASS at BOTH h=7 and
  h=14 -> wire Mondrian behind a flag for a served confirm. Any FAIL ->
  Mondrian is closed (third strike: WACI, probe width, OOF).

DB: one read-only session for events metadata only (the frame builder
requires it). The session is fenced `SET TRANSACTION READ ONLY`, so an
accidental write raises instead of landing in prod.

    venv/bin/python -m scripts.confirm_mondrian_oof --horizons 7,14
    venv/bin/python -m scripts.confirm_mondrian_oof --phase score --horizons 7,14
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Production footing first (call-time reads, but set before any use).
os.environ.setdefault("FEATURE_NATIVE_NAN", "1")
os.environ.setdefault("EXCEEDANCE_HEAD", "1")
os.environ.setdefault("CV_DIAGNOSTIC_CLASSIFIER", "0")

from models import conformal  # noqa: E402
from models.forecaster import ItemForecaster, embargo_days  # noqa: E402
from scripts.measure_qhat_bagging_mondrian import (  # noqa: E402
    TARGET,
    CalibPanel,
    evaluate,
    fit_mondrian,
    fit_s0,
)

logger = logging.getLogger("confirm_mondrian_oof")

# The SKIP_HP fallback the trainer uses when a horizon skips Optuna
# (forecaster.py, merge block): the documented untuned centre (DELTA 1).
DEFAULT_Q50 = {"num_leaves": 47, "learning_rate": 0.01, "lambda_l1": 0.0,
               "lambda_l2": 1.5, "max_depth": 5, "min_data_in_leaf": 15}

MIN_OOF_TEST_FOLD_ROWS = 1000
PLACEBO_SEED = 20260913


def default_q50_params(fc: ItemForecaster) -> dict:
    """Untuned q50 LightGBM params, mirroring the trainer's SKIP_HP block."""
    from models.forecaster import _gpu_available  # noqa: E402
    p = {"device": "cuda" if _gpu_available() else "cpu",
         "feature_pre_filter": False, "objective": "quantile", "alpha": 0.5,
         "metric": "quantile", "boosting_type": fc.BOOSTING_TYPE,
         "min_gain_to_split": 0.1, "feature_fraction": 0.7,
         "max_bin": fc.MAX_BIN, "verbosity": -1, "n_jobs": -1}
    p.update(DEFAULT_Q50)
    fc._apply_row_sampling(p, 0.5)
    return p


def readonly_session():
    """Prod session fenced read-only: events metadata in, nothing out."""
    from database import SessionLocal  # noqa: E402
    from sqlalchemy import text  # noqa: E402
    db = SessionLocal()
    db.execute(text("SET TRANSACTION READ ONLY"))
    return db


def train_oof(fc: ItemForecaster, df: pd.DataFrame, horizon: int,
              workdir: str) -> tuple[str, str]:
    """Run production CV for one horizon; persist OOF rows + fold windows."""
    tdf = fc.prepare_targets(df, horizon)
    params = {0.5: default_q50_params(fc)}
    oof_records, fold_metrics, _, _ = fc._cv_evaluate_horizon(
        tdf, horizon, params)
    if not oof_records:
        raise RuntimeError(f"CV produced zero OOF records at h={horizon}")
    oof = pd.DataFrame(oof_records)
    # `row_index` names the row in tdf (a RangeIndex frame), so the date
    # joins exactly; without it the fold is the only partition available.
    if "row_index" not in oof.columns:
        raise RuntimeError("OOF records carry no row_index: cannot date rows")
    tdf_dates = pd.to_datetime(tdf["date"])
    oof["date"] = tdf_dates.loc[oof["row_index"].to_numpy()].to_numpy()
    oof_path = os.path.join(workdir, f"oof_h{horizon}.parquet")
    folds_path = os.path.join(workdir, f"folds_h{horizon}.json")
    oof.to_parquet(oof_path, index=False)
    with open(folds_path, "w") as f:
        json.dump([{"fold": m["fold"], "val_start": m["val_start"],
                    "val_end": m["val_end"], "n_val": m["n_val"],
                    "fold_q_hat": m.get("fold_q_hat"),
                    "n_train": m["n_train"]} for m in fold_metrics], f)
    # Fold-composition variance, real-residual edition: per-fold q_hats are
    # fitted at beta=1.0 for comparability (see _cv_evaluate_horizon).
    qs = [m["q_hat"] for m in
          [{"q_hat": m.get("fold_q_hat")} for m in fold_metrics]
          if m["q_hat"] is not None]
    if len(qs) >= 2:
        logger.info("  real-OOF fold_q_hat: %s (max/min=%.3f)",
                    " -> ".join(f"{v:.4f}" for v in qs), max(qs) / min(qs))
    logger.info("  h=%d: %s OOF rows, %d folds -> %s",
                horizon, f"{len(oof):,}", len(fold_metrics), oof_path)
    return oof_path, folds_path


def last_test_folds(oof: pd.DataFrame, n_folds: int,
                    min_rows: int = MIN_OOF_TEST_FOLD_ROWS) -> list:
    """The last `n_folds` folds with enough OOF rows, oldest first."""
    counts = oof.groupby("fold").size()
    eligible = [f for f in sorted(counts.index) if counts[f] >= min_rows]
    if len(eligible) < 2:
        raise RuntimeError(
            f"only {len(eligible)} folds clear {min_rows} rows: no held-out "
            f"read possible")
    return eligible[-n_folds:]


def oof_panel(oof: pd.DataFrame, horizon: int) -> CalibPanel:
    """The probe's panel surface over real OOF rows (no refit, no relearn)."""
    frame = pd.DataFrame({"resid": oof["residual_pct"].to_numpy(dtype=float),
                          "sigma": oof["sigma"].to_numpy(dtype=float),
                          "date": pd.to_datetime(oof["date"])})
    frame = frame[np.isfinite(frame["resid"]) & np.isfinite(frame["sigma"])
                  & (frame["sigma"] > 0)]
    return CalibPanel(frame.sort_values("date").reset_index(drop=True),
                      horizon)


def score_oof(horizon: int, workdir: str, test_folds_n: int) -> dict:
    """Phase B: probe arms on real OOF, test = OOF dates in late folds."""
    oof = pd.read_parquet(os.path.join(workdir, f"oof_h{horizon}.parquet"))
    test_folds = last_test_folds(oof, test_folds_n)
    p = oof_panel(oof, horizon)
    test_dates = sorted(pd.to_datetime(
        oof.loc[oof["fold"].isin(test_folds), "date"]).unique())
    # Embargo separation, asserted not assumed: every test date's
    # calibration set ends H+13 before it (enforced by the masks), and the
    # earliest test fold must start after the latest calibration fold's
    # window — otherwise "held-out" shares a regime edge with calibration.
    emb = embargo_days(horizon)
    calib_end = pd.to_datetime(
        oof.loc[~oof["fold"].isin(test_folds), "date"]).max()
    test_start = pd.to_datetime(test_dates).min()
    gap = (test_start - calib_end).days
    logger.info("  h=%d: %d test folds %s, %d test dates, calib->test gap %dd "
                "(embargo %dd)", horizon, len(test_folds), test_folds,
                len(test_dates), gap, emb)
    # Production pools every fold, so the grid is every OOF date: the mask
    # below then keeps all OOF rows at or before day - embargo.
    grid = frozenset(pd.to_datetime(p.dates))
    arms = {
        "S0_pooled": lambda pp, d: fit_s0(pp, d, grid),
        "M_vol_dec": lambda pp, d: fit_mondrian(pp, d, grid),
        "P_shuffled": lambda pp, d: fit_mondrian(pp, d, grid, shuffle=True,
                                                seed=PLACEBO_SEED),
    }
    out = {}
    for name, fn in arms.items():
        m = evaluate(p, fn, [np.datetime64(d) for d in test_dates])
        if not m:
            logger.warning("  %-10s no fit", name)
            continue
        m.update(horizon=horizon, scheme=name, test_folds=",".join(
            str(f) for f in test_folds))
        out[name] = m
        logger.info(
            "  %-10s M1=%.1f%% M2=%5.2fpp w=%.4f | LM c=%.3f M2*=%5.2fpp "
            "sig*=%5.2fpp w*=%.4f mindec*=%.0f%% [%s]",
            name, m["M1_marginal"] * 100, m["M2_cond_err_pp"],
            m["mean_width"], m["level_match_c"], m["M2lm_cond_err_pp"],
            m["M2lm_sigma_err_pp"], m["mean_width_lm"],
            m["min_decile_cov_lm"] * 100, m["sigma_decile_cov_lm"])
    return out


def verdict(rows: list) -> int:
    """The pre-registered confirm bar, applied where it cannot be misread."""
    out = pd.DataFrame(rows)
    if out.empty:
        logger.error("no results")
        return 1
    base = out[out["scheme"] == "S0_pooled"].set_index("horizon")
    logger.info("")
    logger.info("CONFIRM BAR — M sig* below P sig* AND M1_M in 80+-3pp AND "
                "M min decile >=70%%, at BOTH h=7 and h=14.")
    ok = True
    for h in sorted(base.index):
        sub = out[out["horizon"] == h].set_index("scheme")
        if not {"M_vol_dec", "P_shuffled"} <= set(sub.index):
            logger.info("  h=%d: missing arm -> fail", h)
            ok = False
            continue
        m, p = sub.loc["M_vol_dec"], sub.loc["P_shuffled"]
        cond = m["M2lm_sigma_err_pp"] < p["M2lm_sigma_err_pp"]
        band = abs(m["M1_marginal"] - 0.80) <= 0.03
        guard = m["min_decile_cov_lm"] >= 0.70
        passed = bool(cond and band and guard)
        ok &= passed
        logger.info("  h=%d: sig* M=%.2f P=%.2f (%s), M1_M=%.1f%% (%s), "
                    "mindec=%.0f%% (%s) -> %s",
                    h, m["M2lm_sigma_err_pp"], p["M2lm_sigma_err_pp"],
                    cond, m["M1_marginal"] * 100, band,
                    m["min_decile_cov_lm"] * 100, guard,
                    "PASS" if passed else "fail")
    logger.info("A P win on sig* closes Mondrian (third strike). "
                "M PASS at both horizons wires it behind a flag.")
    return 0 if ok else 2


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", default="7,14")
    ap.add_argument("--phase", default="both",
                    choices=["train", "score", "both"])
    ap.add_argument("--workdir", default="/tmp/mondrian_oof")
    ap.add_argument("--test-folds", type=int, default=3)
    ap.add_argument("--out", default="/tmp/mondrian_oof_confirm.csv")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",")]

    logger.info("env footing: FEATURE_NATIVE_NAN=%s EXCEEDANCE_HEAD=%s "
                "CV_DIAGNOSTIC_CLASSIFIER=%s",
                os.environ.get("FEATURE_NATIVE_NAN"),
                os.environ.get("EXCEEDANCE_HEAD"),
                os.environ.get("CV_DIAGNOSTIC_CLASSIFIER"))

    if args.phase in ("train", "both"):
        db = readonly_session()
        try:
            fc = ItemForecaster(db_session=db)
            df = fc.build_training_data(
                days_back=1460, backfilled_only=True,
                max_feature_rows=1_200_000, min_median_price=1.0,
                universe="train")
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma_raw = (df["price_std_60d"].to_numpy(dtype=float)
                             / df["price"].to_numpy(dtype=float))
            floor, cap = conformal.sigma_bounds(sigma_raw)
            finite = sigma_raw[np.isfinite(sigma_raw) & (sigma_raw > 0)]
            fc.sigma_clip = {"floor": floor, "cap": cap,
                             "fallback": float(np.median(finite))}
            logger.info("sigma clip: floor=%.5f cap=%.5f fallback=%.5f",
                        floor, cap, fc.sigma_clip["fallback"])
            for h in horizons:
                train_oof(fc, df, h, args.workdir)
        finally:
            db.close()

    if args.phase in ("score", "both"):
        rows = []
        for h in horizons:
            for name, m in score_oof(h, args.workdir,
                                     args.test_folds).items():
                rows.append(m)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        logger.info("wrote %s", args.out)
        return verdict(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
