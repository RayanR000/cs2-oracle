#!/usr/bin/env python3
"""Estimate the gate's minimum detectable effect, per horizon.

Runs the walkforward gate twice on the same folds, changing only the LightGBM
seed. Both runs are the same design, so the paired difference is pure noise
and the width of its date-clustered interval is the smallest real effect the
gate could distinguish from noise.

`fold_seed` and `step_days` are passed as call arguments to `run_walkforward`
on each iteration rather than mutated on the module — both seed runs execute
in this one process, and a global rebind would leak between them and make
each run's actual configuration unclear from its own call site.

Usage:
    python3 scripts/compute_mde.py --max-items 500
    python3 scripts/compute_mde.py --max-items 60 --step-days 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.paired_mde import paired_da_difference
from scripts import walkforward_backtest as wf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--step-days", type=int, default=wf.STEP_DAYS,
                         help=f"Fold stride in days (default: {wf.STEP_DAYS})")
    args = parser.parse_args()

    # One horizon at a time, both seeds, emitting each result the moment it is
    # paired. The previous shape ran every horizon for seed 42, then every
    # horizon for seed 7, and printed only at the very end -- so the run that
    # was killed at 16 minutes yielded nothing at all. The cost of this shape
    # is that the archive load and feature engineering repeat per horizon
    # instead of per seed; that is a couple of minutes against losing the lot.
    out = {}
    horizons = args.horizons or wf.ItemForecaster.HORIZONS
    for horizon in horizons:
        records = {}
        for seed in (42, 7):
            report = wf.run_walkforward(
                max_items=args.max_items,
                horizons=[horizon],
                skip_db=True,
                return_records=True,
                step_days=args.step_days,
                fold_seed=seed,
            )
            records[seed] = (report.get("horizons", {})
                             .get(str(horizon), {})
                             .get("records"))
        a, b = records[42], records[7]
        if not a or not b:
            out[str(horizon)] = {"error": "no records returned"}
        else:
            out[str(horizon)] = paired_da_difference(a, b)
        # Flushed per horizon so an interrupted run still leaves usable output.
        print(f"MDE h={horizon}: "
              f"{json.dumps(out[str(horizon)], default=str)}", flush=True)

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
