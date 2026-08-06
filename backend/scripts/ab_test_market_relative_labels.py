#!/usr/bin/env python3
"""Paired A/B: does training the directional classifier on market-demeaned
returns beat training it on raw returns?

This is a DRIVER, not a harness. It sets env vars and invokes two real cold
retrains via scripts/forecast_prices.py --train-only, then diffs their
meta.json. Both arms therefore run through the live _cv_evaluate_horizon.

That distinction is the whole point. scripts/ab_test_*.py mostly build their
own walk-forward loops, and that pattern produced
docs/changelog/2026-08-06-volume-ab-and-harness-defects.md -- that harness scored
a cohort 92% of which production never serves, so ~31pp of its reported
directional accuracy was free hits.

Both arms write into scratch directories via FORECAST_MODEL_DIR, so the
deployed artifact in backend/models/saved_models/ is never touched.

PRE-REGISTERED DECISION RULE (spec, set before either arm ran):
  1. KILL if relative_accuracy_ge1 <= 51% at EVERY horizon. No market model
     rescues an absent idiosyncratic signal. Do not tune m_hat to save it.
  2. RECOMMEND ADOPTION only on a paired mean diff in classifier_accuracy_ge1
     of > +2pp at two or more horizons AND not worse than -1pp at any.
  3. WITHIN +/-1pp: market-date domination is not addressable by relabelling.
     Record and close.
Adoption is a recommendation for a follow-up change. Nothing ships from here.

Usage:
    python scripts/ab_test_market_relative_labels.py --out /tmp/mrl
    python scripts/ab_test_market_relative_labels.py --out /tmp/mrl --report-only
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [3, 7, 14, 30]
ARMS = {"control": "0", "treatment": "1"}

# Minimum market-factor coverage before a run is readable. Below this the
# treatment arm is diluted toward the control by the zero-fill fallback and
# the contrast measures nothing.
MIN_COVERAGE_PCT = 95.0


def run_arm(name: str, flag: str, out_root: Path) -> Path:
    """Run one cold --train-only retrain into its own scratch model dir."""
    model_dir = out_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["FORECAST_MODEL_DIR"] = str(model_dir)
    env["TRAIN_MARKET_RELATIVE_LABELS"] = flag
    logger.info(f"\n=== arm: {name} (TRAIN_MARKET_RELATIVE_LABELS={flag}) ===")
    logger.info(f"    artifacts -> {model_dir}")
    proc = subprocess.run(
        [sys.executable, "scripts/forecast_prices.py", "--train-only"],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"arm {name!r} failed with exit code {proc.returncode}")
    return model_dir / "meta.json"


def _folds(meta: dict, horizon: int) -> list:
    return meta["cv_results"][str(horizon)]["per_fold"]


def verify_pairing(control: dict, treatment: dict) -> None:
    """Confirm the arms are comparable BEFORE any metric is read.

    Identical fold ids, row counts, date bounds and tuned params. If these
    differ, the two arms scored different data and the contrast is void --
    reading the accuracy numbers first is how a broken run gets believed.
    """
    problems = []
    for h in HORIZONS:
        c, t = _folds(control, h), _folds(treatment, h)
        if len(c) != len(t):
            problems.append(f"{h}d: fold count {len(c)} vs {len(t)}")
            continue
        for cf, tf in zip(c, t):
            for key in ("fold", "n_train", "n_val",
                        "train_start", "train_end", "val_start", "val_end"):
                if cf.get(key) != tf.get(key):
                    problems.append(
                        f"{h}d fold {cf.get('fold')}: {key} "
                        f"{cf.get(key)!r} vs {tf.get(key)!r}")
        if control.get("tuned_params", {}).get(str(h)) != \
                treatment.get("tuned_params", {}).get(str(h)):
            problems.append(f"{h}d: tuned_params differ")
    if problems:
        for p in problems:
            logger.error(f"  PAIRING BROKEN: {p}")
        raise SystemExit("arms are not paired; the contrast is void")
    logger.info("  pairing verified: identical folds, row counts, dates, params")


def check_coverage(treatment: dict) -> None:
    for h in HORIZONS:
        covs = [f.get("market_factor_coverage") for f in _folds(treatment, h)]
        covs = [c for c in covs if c is not None]
        if covs and min(covs) < MIN_COVERAGE_PCT:
            logger.error(
                f"  COVERAGE TOO LOW at {h}d: min fold coverage {min(covs):.1f}% "
                f"< {MIN_COVERAGE_PCT}%. The treatment arm is diluted toward "
                f"the control; this run is void.")
            raise SystemExit("insufficient market-factor coverage")
    logger.info("  market-factor coverage OK on every fold")


def paired_table(control: dict, treatment: dict, metric: str) -> list:
    rows = []
    for h in HORIZONS:
        c = [f.get(metric) for f in _folds(control, h)]
        t = [f.get(metric) for f in _folds(treatment, h)]
        pairs = [(a, b) for a, b in zip(c, t) if a is not None and b is not None]
        if not pairs:
            rows.append((h, 0, None, None, None, None, None))
            continue
        ca = np.array([p[0] for p in pairs], dtype=float)
        ta = np.array([p[1] for p in pairs], dtype=float)
        diff = ta - ca
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        tstat = (float(diff.mean()) / (sd / np.sqrt(len(diff)))
                 if sd > 0 else None)
        rows.append((h, len(pairs), float(ca.mean()), float(ta.mean()),
                     float(diff.mean()), sd, tstat))
    return rows


def _fmt(rows, title):
    logger.info(f"\n{title}")
    logger.info("| horizon | folds | control | treatment | paired diff | sd | t |")
    logger.info("|---|---|---|---|---|---|---|")
    for h, n, c, t, d, sd, ts in rows:
        if c is None:
            logger.info(f"| {h}d | 0 | - | - | - | - | - |")
            continue
        logger.info(f"| {h}d | {n} | {c:.2f} | {t:.2f} | **{d:+.2f}** | "
                    f"{sd:.2f} | {'-' if ts is None else f'{ts:.2f}'} |")


def verdict(ge1_rows, treatment: dict) -> None:
    rel = {}
    for h in HORIZONS:
        vals = [f.get("relative_accuracy_ge1") for f in _folds(treatment, h)]
        vals = [v for v in vals if v is not None]
        rel[h] = float(np.mean(vals)) if vals else None

    logger.info("\nStage 1 — relative_accuracy_ge1 (the kill switch):")
    for h in HORIZONS:
        v = rel[h]
        logger.info(f"  {h}d: {'n/a' if v is None else f'{v:.2f}%'}")

    live = [v for v in rel.values() if v is not None]
    if live and all(v <= 51.0 for v in live):
        logger.info(
            "\nVERDICT: KILL (rule 1). The classifier cannot call the "
            "idiosyncratic direction at any horizon, so there is no signal for "
            "a market forecast to recover. Do not interpret stage 2 and do not "
            "tune the m_hat estimator to rescue it.")
        return

    diffs = {h: d for h, _n, _c, _t, d, _sd, _ts in ge1_rows if d is not None}
    if not diffs:
        logger.info("\nVERDICT: no comparable folds.")
        return
    wins = [h for h, d in diffs.items() if d > 2.0]
    worst = min(diffs.values())
    if len(wins) >= 2 and worst >= -1.0:
        logger.info(
            f"\nVERDICT: RECOMMEND ADOPTION (rule 2). >+2pp at {wins}, worst "
            f"horizon {worst:+.2f}pp. This is a recommendation for a follow-up "
            f"change — nothing ships from this script.")
    elif all(abs(d) <= 1.0 for d in diffs.values()):
        logger.info(
            "\nVERDICT: NO EFFECT (rule 3). Every horizon within ±1pp. "
            "Market-date domination is not addressable by relabelling. Leave "
            "the default off and record the result.")
    else:
        logger.info(
            f"\nVERDICT: DOES NOT CLEAR THE BAR (rule 2 not met). "
            f"{len(wins)} horizon(s) above +2pp, worst {worst:+.2f}pp. "
            f"Leave the default off.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="scratch root; each arm gets a subdirectory")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the retrains and read existing meta.json files")
    args = ap.parse_args()

    out_root = Path(args.out).expanduser()
    metas = {}
    for name, flag in ARMS.items():
        path = out_root / name / "meta.json"
        if not args.report_only:
            path = run_arm(name, flag, out_root)
        if not path.exists():
            raise SystemExit(f"missing {path}")
        metas[name] = json.loads(path.read_text())

    logger.info("\n=== validity checks (before any metric is read) ===")
    verify_pairing(metas["control"], metas["treatment"])
    check_coverage(metas["treatment"])

    ge1 = paired_table(metas["control"], metas["treatment"],
                       "classifier_accuracy_ge1")
    _fmt(ge1, "classifier_accuracy_ge1 — the headline (>= $1 served cohort)")
    _fmt(paired_table(metas["control"], metas["treatment"],
                      "classifier_accuracy"),
         "classifier_accuracy — pooled, all tiers (~83% penny rows)")
    verdict(ge1, metas["treatment"])


if __name__ == "__main__":
    main()
