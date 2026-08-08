#!/usr/bin/env python3
"""Does voiding frozen-price-run labels narrow the measurement instrument?

Step 6's claim is NOT that accuracy rises. It is that dropping labels measured
to or from a bit-identical price run **lowers the MDE** — that it makes every
future experiment easier to resolve. That is a claim about interval WIDTH, and
it needs a different design from every other `ab_test_*` in this directory.

## Why this is not an on-vs-off A/B

Pairing the rule-on arm against the rule-off arm answers "did accuracy move",
which is not the claim, and it answers it badly: the rule voids labels, so the
two arms do not score the same rows and `paired_metric_difference` would pair
only on the intersection — precisely the rows the rule did not touch.

Instead this runs `compute_mde.py`'s design once per label regime. Within a
regime, two arms differ **only in the LightGBM seed**, so the true difference
is zero by construction and the half-width of the paired fold-clustered
interval is that regime's noise floor. The comparison is then floor-off vs
floor-on:

    MDE_off  = half-width of (seed 42 vs seed 7), labels unfiltered
    MDE_on   = half-width of (seed 42 vs seed 7), frozen-run labels voided
    narrowing_pct = (MDE_off - MDE_on) / MDE_off * 100

A positive `narrowing_pct` is the step's claim. Zero or negative is a real and
publishable result: the rule takes 13.7-15.9% of NON-zero >=$1 labels with it
(measured 2026-08-08), so buying less noise at the price of less signal is a
live outcome, not a hypothetical.

## Read the point estimates too, and separately

Each regime also reports its own arms' absolute DA. If the point estimate moves
a lot while the width barely does, that is a **different finding** from the one
this harness is built to test and must be reported as one rather than folded in
— see `docs/superpowers/specs/2026-08-08-frozen-price-runs-design.md`.

## Caveats a citation must carry

- The seed-pair floor is the floor for a SEED-sized perturbation. A treatment
  that also changes the item draw has a wider floor; see
  `docs/changelog/2026-08-07-training-item-universe.md` (item-level floor
  2.21-3.69pp).
- Bootstrap resamples are seeded (`BOOTSTRAP_RNG_SEED`), so the two half-widths
  are comparable draw-for-draw, but the underlying folds are not identical
  across regimes: voiding labels can empty a fold. `n_clusters` is reported per
  regime for exactly that reason and a difference in it invalidates the
  comparison.

Usage:
    python -m scripts.ab_test_frozen_runs --max-items 300
    python -m scripts.ab_test_frozen_runs --max-items 300 --horizons 7 30
    python -m scripts.ab_test_frozen_runs --thresholds none 0 1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.paired_mde import NoPairedRows, paired_da_difference
from models import forecaster as fc
from scripts import walkforward_backtest as wf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_frozen_runs")

# The two seeds compute_mde.py uses. Kept identical so a floor measured here is
# directly comparable to one measured there.
SEEDS = (42, 7)


def _parse_threshold(raw: str):
    """`none` -> None (rule off), an integer -> that many days tolerated."""
    if raw.lower() in {"none", "off", "null"}:
        return None
    return int(raw)


def _label(threshold) -> str:
    return "off" if threshold is None else f"max_run_{threshold}"


def _measure_floor(threshold, horizon, max_items, step_days):
    """One regime's noise floor: seed 42 vs seed 7, labels at `threshold`."""
    # Rebound on the module rather than passed through run_walkforward: the
    # label rule lives in prepare_targets, which walkforward_backtest calls
    # through the forecaster, and threading a parameter down that path would
    # mean touching the production signature for a harness. Restored by the
    # caller's try/finally so a crash cannot leave the module rebound.
    fc.LABEL_MAX_STALE_RUN_DAYS = threshold

    records = {}
    for seed in SEEDS:
        report = wf.run_walkforward(
            max_items=max_items,
            horizons=[horizon],
            skip_db=True,
            return_records=True,
            step_days=step_days,
            fold_seed=seed,
        )
        records[seed] = (report.get("horizons", {})
                         .get(str(horizon), {})
                         .get("records"))

    a, b = records[SEEDS[0]], records[SEEDS[1]]
    if not a or not b:
        return {"error": "no records returned",
                "n_a": len(a or []), "n_b": len(b or [])}
    try:
        paired = paired_da_difference(a, b)
    except NoPairedRows as exc:
        return {"error": str(exc)}

    # The absolute DA of each arm, so a moved point estimate is visible rather
    # than hidden behind a width that did not move.
    paired["arm_da"] = {
        str(seed): round(
            sum(r["direction_correct"] for r in recs) / len(recs) * 100, 3)
        for seed, recs in records.items() if recs
    }
    paired["n_rows"] = {str(seed): len(recs or []) for seed, recs in records.items()}
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--step-days", type=int, default=wf.STEP_DAYS)
    parser.add_argument(
        "--thresholds", nargs="+", default=["none", "0"],
        help="Label regimes to measure. 'none' disables the rule "
             "(default: none 0)")
    args = parser.parse_args()

    thresholds = [_parse_threshold(t) for t in args.thresholds]
    if None not in thresholds:
        logger.warning(
            "No 'none' regime requested — without the unfiltered control there "
            "is no baseline width to narrow, and narrowing_pct cannot be computed."
        )

    horizons = args.horizons or wf.ItemForecaster.HORIZONS
    original = fc.LABEL_MAX_STALE_RUN_DAYS
    out: dict = {}

    try:
        for horizon in horizons:
            per_regime = {}
            for threshold in thresholds:
                label = _label(threshold)
                logger.info(
                    f"h={horizon} regime={label}: measuring the seed-pair floor")
                per_regime[label] = _measure_floor(
                    threshold, horizon, args.max_items, args.step_days)

            baseline = per_regime.get("off", {})
            base_mde = baseline.get("mde_pp")
            for label, res in per_regime.items():
                if label == "off" or base_mde in (None, 0) or res.get("mde_pp") is None:
                    continue
                res["narrowing_pct"] = round(
                    (base_mde - res["mde_pp"]) / base_mde * 100, 2)
                # A fold-count difference means the regimes did not measure the
                # same design, so the widths are not comparable. Reported on the
                # row rather than logged, so a stored result carries it.
                res["comparable_folds"] = (
                    res.get("n_clusters") == baseline.get("n_clusters"))

            out[str(horizon)] = per_regime
            # Flushed per horizon: a run killed partway still leaves usable output.
            print(f"FROZEN-RUN MDE h={horizon}: "
                  f"{json.dumps(per_regime, default=str)}", flush=True)
    finally:
        fc.LABEL_MAX_STALE_RUN_DAYS = original

    print(json.dumps(out, indent=2, default=str))

    for horizon, regimes in out.items():
        base = regimes.get("off", {})
        for label, res in regimes.items():
            if label == "off":
                continue
            narrowing = res.get("narrowing_pct")
            if narrowing is None:
                logger.info(f"  h={horizon} {label}: no comparison available")
                continue
            note = "" if res.get("comparable_folds") else \
                "  [NOT COMPARABLE: fold counts differ]"
            logger.info(
                f"  h={horizon} {label}: MDE {base.get('mde_pp')}pp -> "
                f"{res.get('mde_pp')}pp ({narrowing:+.1f}% narrowing){note}"
            )


if __name__ == "__main__":
    main()
