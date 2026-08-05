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

    out = {}
    runs = []
    for seed in (42, 7):
        runs.append(wf.run_walkforward(
            max_items=args.max_items,
            horizons=args.horizons,
            skip_db=True,
            return_records=True,
            step_days=args.step_days,
            fold_seed=seed,
        ))

    for horizon_str in runs[0]["horizons"]:
        a = runs[0]["horizons"][horizon_str].get("records")
        b = runs[1]["horizons"][horizon_str].get("records")
        if not a or not b:
            out[horizon_str] = {"error": "no records returned"}
            continue
        out[horizon_str] = paired_da_difference(a, b)

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
