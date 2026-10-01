#!/usr/bin/env python3
"""Per-fold q_hat dispersion under varying CV fold designs (Item 3 probe).

Pre-registration: `docs/research/2026-09-14-qhat-dispersion-fold-design-preregistration.md`
(write first, read numbers second — the bars below are quoted from it, not
fitted to any output of this script). Orders:
`docs/specs/2026-09-14-wait-window-workplan.md` Item 3.

DEFECT (measured, not hypothesised): per-fold q_hat swings +-26% on fold
composition alone (h=3, folds 4-9: 68.8 -> 126.9 at identical 300K training
rows), and spearman(n_train, q_hat) flips sign by horizon, so training size
explains approximately none of it. The lever here is the fold DESIGN
(stride, count, overlap) — never the band formula, never per-item
conditioning (that family is 0-for-7+).

ARMS (each a production-CV pass over the same archive panel, horizons never
pooled):
  control — CV_STEP_DAYS=150, CV_ROW_SEED=42 (status quo)
  placebo — CV_STEP_DAYS=150, CV_ROW_SEED=<seed-b> (identical grid; the gap
            to control is the sampling-noise floor)
  dense   — CV_STEP_DAYS=75  (more folds, more overlap)
  sparse  — CV_STEP_DAYS=300 (fewer folds, less overlap)

METHOD (mirrors `scripts/confirm_mondrian_oof.py`, whose `train_oof` is
reused verbatim): one shared training frame (`build_training_data` with
train()'s exact arguments, sigma clip exactly as train()), then
`_cv_evaluate_horizon` per horizon with the SKIP_HP default q50 params.
Deliberate deltas, same as the confirm's: no Optuna (fixing params isolates
the grid effect — re-tuning per arm would add HP noise to a grid read),
no production fit, no artifact, no DB writes. Per-fold `fold_q_hat` is read
out of the persisted folds JSON exactly as the confirm does. The LightGBM
booster seed stays 42 on every arm: the placebo varies ONLY the 300K-cap
row draw (`CV_ROW_SEED`), so control-vs-placebo is pure sampling noise.

FALSIFIABLE BAR (fixed before reading numbers — a design wins iff ALL hold):
  1. Its per-horizon q_hat CV is >=30% below control at >=3 of 4 horizons,
     with no horizon worse than +10% vs control.
  2. The control-vs-placebo CV gap is <15% at every horizon (else the grid
     itself is seed-unstable: the probe is VOID, not null).
  3. Matched-width coverage on the prod-basis replay (`replay_serving.py`)
     is within +-1pp of control at every horizon.
If (1) fails everywhere: estimator variance is irreducible by grid choice;
close Item 3. Bars 1-2 are scored here (Phase A). Bar 3 needs served
artifacts per arm (full train + save + replay) and runs ONLY for an arm
that passes 1-2 — this script prints the exact Phase-B commands then.

DB: one read-only session for events metadata only (the frame builder
requires it), fenced `SET TRANSACTION READ ONLY` like the confirm, so an
accidental write raises instead of landing in prod. Run from `backend/` so
config reads `backend/.env`:

    venv/bin/python -m scripts.archive.measure_qhat_dispersion --horizons 3,7,14,30
    venv/bin/python -m scripts.archive.measure_qhat_dispersion --arms control  # pacing: one arm at a time

Exit codes: 0 = measured (null or candidate — read the verdict table);
2 = VOID (placebo-unstable grid, or <2 measurable folds at a horizon).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import conformal
from models.forecaster import ItemForecaster
from scripts.archive.confirm_mondrian_oof import readonly_session, train_oof

logger = logging.getLogger("qhat_dispersion")

ARMS = ("control", "placebo", "dense", "sparse")
HORIZONS = (3, 7, 14, 30)

# No further arms without a second pre-registration — that is how
# single-split tuning happens.
ARM_GRID = {
    "control": {"CV_STEP_DAYS": "150", "CV_ROW_SEED": "42"},
    "placebo": {"CV_STEP_DAYS": "150", "CV_ROW_SEED": None},  # filled from --seed-b
    "dense": {"CV_STEP_DAYS": "75", "CV_ROW_SEED": "42"},
    "sparse": {"CV_STEP_DAYS": "300", "CV_ROW_SEED": "42"},
}

MIN_FOLDS_PER_HORIZON = 2
BAR1_CV_RATIO_WIN = 0.70  # >=30% below control
BAR1_CV_RATIO_WORSE = 1.10  # no horizon worse than +10%
BAR1_HORIZONS_NEEDED = 3
BAR2_PLACEBO_GAP = 0.15  # control-vs-placebo CV gap <15%


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested; no DB, no training)
# --------------------------------------------------------------------------- #


def dispersion_stats(qs: list[float]) -> dict:
    """Per-fold q_hat dispersion for one arm x horizon. Horizons never pooled."""
    qs = [float(q) for q in qs if q is not None and np.isfinite(q)]
    out: dict = {"n_folds": len(qs), "q_hats": qs}
    if len(qs) < MIN_FOLDS_PER_HORIZON:
        out.update(mean=None, stdev=None, cv=None, max_min_ratio=None)
        return out
    arr = np.asarray(qs, dtype=float)
    mean = float(arr.mean())
    stdev = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
    out["mean"] = mean
    out["stdev"] = stdev
    out["cv"] = (stdev / mean) if mean > 0 else None
    out["max_min_ratio"] = float(arr.max() / arr.min()) if arr.min() > 0 else None
    return out


def _rel_gap(a: float | None, b: float | None) -> float | None:
    """|a-b|/a with the degenerate-CV guard: two ~zero CVs are stable."""
    if a is None or b is None:
        return None
    if abs(a) < 1e-9:
        return 0.0 if abs(b) < 1e-9 else float("inf")
    return abs(a - b) / abs(a)


def evaluate_bars(stats: dict) -> dict:
    """Apply the pre-registered bars where they cannot be misread.

    `stats[h][arm]` -> dispersion_stats dict. Returns per-horizon placebo
    stability plus a per-arm PASS/FAIL over bar 1, and the whole-probe VOID
    flag for bar 2. Bar 3 (replay coverage) is Phase B, not scored here.
    """
    horizons = sorted(stats)
    placebo_gap: dict = {}
    void = False
    for h in horizons:
        ctrl = stats[h].get("control", {}).get("cv")
        pbo = stats[h].get("placebo", {}).get("cv")
        gap = _rel_gap(ctrl, pbo)
        ok = gap is not None and gap < BAR2_PLACEBO_GAP
        placebo_gap[h] = {"gap": gap, "stable": bool(ok)}
        if not ok:
            void = True
    arms: dict = {}
    for arm in ARMS:
        if arm in ("control", "placebo"):
            continue
        wins = 0
        worse = 0
        scored = 0
        for h in horizons:
            ctrl = stats[h].get("control", {}).get("cv")
            mine = stats[h].get(arm, {}).get("cv")
            if ctrl is None or mine is None:
                continue  # void horizon for this contrast; never a win
            scored += 1
            ratio = (mine / ctrl) if ctrl > 0 else (0.0 if mine <= 0 else float("inf"))
            if ratio <= BAR1_CV_RATIO_WIN:
                wins += 1
            if ratio > BAR1_CV_RATIO_WORSE:
                worse += 1
        arms[arm] = {
            "wins": wins,
            "worse": worse,
            "scored": scored,
            "pass": bool(wins >= BAR1_HORIZONS_NEEDED and worse == 0),
        }
    return {"placebo_gap": placebo_gap, "void": void, "arms": arms}


def read_arm_folds(arm_dir: str, horizon: int) -> list[dict]:
    """fold_q_hat + n_train per fold, exactly as confirm_mondrian_oof reads."""
    path = os.path.join(arm_dir, f"folds_h{horizon}.json")
    with open(path) as f:
        folds = json.load(f)
    return [{"fold_q_hat": m.get("fold_q_hat"), "n_train": m.get("n_train")} for m in folds]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run_arm(fc: ItemForecaster, df: pd.DataFrame, arm: str, horizons: list[int], workdir: str) -> str:
    """One arm's production-CV pass; persists OOF + folds JSON per horizon."""
    grid = ARM_GRID[arm]
    os.environ["CV_STEP_DAYS"] = grid["CV_STEP_DAYS"]
    os.environ["CV_ROW_SEED"] = grid["CV_ROW_SEED"]
    arm_dir = os.path.join(workdir, arm)
    os.makedirs(arm_dir, exist_ok=True)
    logger.info(
        "arm %-8s CV_STEP_DAYS=%s CV_ROW_SEED=%s",
        arm,
        grid["CV_STEP_DAYS"],
        grid["CV_ROW_SEED"],
    )
    for h in horizons:
        t0 = time.time()
        _oof_path, folds_path = train_oof(fc, df, h, arm_dir)
        logger.info("  arm %-8s h=%2d done in %.0fs -> %s", arm, h, time.time() - t0, folds_path)
    return arm_dir


def summarize(workdir: str, arms: list[str], horizons: list[int]) -> tuple[pd.DataFrame, dict]:
    """Per-horizon dispersion per arm + the bar verdicts. No numbers read twice."""
    stats: dict = {}
    rows = []
    for h in horizons:
        stats[h] = {}
        for arm in arms:
            folds = read_arm_folds(os.path.join(workdir, arm), h)
            s = dispersion_stats([m["fold_q_hat"] for m in folds])
            s["n_trains"] = sorted({m["n_train"] for m in folds if m["n_train"] is not None})
            stats[h][arm] = s
            rows.append(
                {
                    "arm": arm,
                    "horizon": h,
                    "n_folds": s["n_folds"],
                    "q_hats": " ".join(f"{q:.2f}" for q in s["q_hats"]),
                    "max_min_ratio": s["max_min_ratio"],
                    "cv": s["cv"],
                    "mean": s["mean"],
                    "n_trains": " ".join(str(n) for n in s["n_trains"]),
                }
            )
    return pd.DataFrame(rows), stats


def report(summary: pd.DataFrame, verdict: dict) -> None:
    logger.info("")
    logger.info("PER-FOLD q_hat DISPERSION (horizons never pooled)")
    for _, r in summary.iterrows():
        logger.info(
            "  %-8s h=%2d folds=%d q=[%s] max/min=%s cv=%s n_train=[%s]",
            r["arm"],
            r["horizon"],
            r["n_folds"],
            r["q_hats"],
            f'{r["max_min_ratio"]:.3f}' if r["max_min_ratio"] is not None else "n/a",
            f'{r["cv"]:.3f}' if r["cv"] is not None else "n/a",
            r["n_trains"],
        )
    logger.info("")
    logger.info(
        "BAR 2 (placebo stability, gap<15%% every horizon): %s",
        "VOID — re-run control/placebo once" if verdict["void"] else "holds",
    )
    for h, g in sorted(verdict["placebo_gap"].items()):
        logger.info(
            "  h=%2d control-vs-placebo CV gap=%s -> %s",
            h,
            f'{g["gap"]:.3f}' if g["gap"] not in (None, float("inf")) else str(g["gap"]),
            "stable" if g["stable"] else "UNSTABLE",
        )
    logger.info("BAR 1 (>=30%% below control at >=3/4 horizons, none +10%% worse):")
    for arm, v in verdict["arms"].items():
        logger.info(
            "  %-8s wins %d/%d, worse %d -> %s",
            arm,
            v["wins"],
            v["scored"],
            v["worse"],
            "PASS (Phase B: replay)" if v["pass"] else "fail",
        )


def phase_b_instructions(arm: str) -> None:
    logger.info("")
    logger.info("PHASE B for %s (bar 3: matched-width replay coverage within +-1pp):", arm)
    logger.info("  1. Full train + save per arm (control and %s), same commit,", arm)
    logger.info("     CV_STEP_DAYS/CV_ROW_SEED set as in Phase A, distinct model dirs.")
    logger.info("  2. FORECAST_MODEL_DIR=<dir> venv/bin/python -m scripts.replay_serving")
    logger.info("     per arm; compare interval_coverage at matched width per horizon.")
    logger.info("  3. Ship NOTHING unless bars 1+2+3 all hold — a 1-2 pass buys one")
    logger.info("     replay, not a grid change.")


def main() -> int:
    # Production footing, set here — NOT at import (see confirm_mondrian_oof
    # for why import-time setdefault poisons pytest). Explicit env wins.
    os.environ.setdefault("FEATURE_NATIVE_NAN", "1")
    os.environ.setdefault("EXCEEDANCE_HEAD", "1")
    os.environ.setdefault("CV_DIAGNOSTIC_CLASSIFIER", "0")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seed-b", type=int, default=7,
                    help="placebo CV_ROW_SEED. Arbitrary but fixed; the only "
                    "requirement is != 42. Fixed here, not tuned.")
    ap.add_argument("--workdir", default="/tmp/qhat_dispersion")
    ap.add_argument("--out", default="/tmp/qhat_dispersion.csv")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a in ARMS]
    if not arms:
        logger.error("no valid arms in %r", args.arms)
        return 1
    if int(args.seed_b) == 42:
        logger.error("--seed-b must differ from the production seed 42")
        return 1
    ARM_GRID["placebo"]["CV_ROW_SEED"] = str(int(args.seed_b))
    horizons = [int(x) for x in args.horizons.split(",")]
    os.makedirs(args.workdir, exist_ok=True)

    logger.info(
        "env footing: FEATURE_NATIVE_NAN=%s EXCEEDANCE_HEAD=%s "
        "CV_DIAGNOSTIC_CLASSIFIER=%s CV_ROW_SEED(placebo)=%s",
        os.environ.get("FEATURE_NATIVE_NAN"),
        os.environ.get("EXCEEDANCE_HEAD"),
        os.environ.get("CV_DIAGNOSTIC_CLASSIFIER"),
        args.seed_b,
    )

    db = readonly_session()
    try:
        fc = ItemForecaster(db_session=db)
        df = fc.build_training_data(
            days_back=1460, backfilled_only=True, max_feature_rows=1_200_000, min_median_price=1.0, universe="train"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_raw = df["price_std_60d"].to_numpy(dtype=float) / df["price"].to_numpy(dtype=float)
        floor, cap = conformal.sigma_bounds(sigma_raw)
        finite = sigma_raw[np.isfinite(sigma_raw) & (sigma_raw > 0)]
        fc.sigma_clip = {"floor": floor, "cap": cap, "fallback": float(np.median(finite))}
        logger.info("frame: %s rows; sigma clip floor=%.5f cap=%.5f", f"{len(df):,}", floor, cap)
        for arm in arms:
            run_arm(fc, df, arm, horizons, args.workdir)
    finally:
        db.close()

    summary, stats = summarize(args.workdir, arms, horizons)
    summary.to_csv(args.out, index=False)
    logger.info("wrote %s", args.out)
    verdict = evaluate_bars(stats)
    report(summary, verdict)
    for arm, v in verdict["arms"].items():
        if v["pass"] and not verdict["void"]:
            phase_b_instructions(arm)
    if verdict["void"]:
        logger.error("VOID: control-vs-placebo unstable — re-run control/placebo once; "
                     "if still unstable, void and report.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
