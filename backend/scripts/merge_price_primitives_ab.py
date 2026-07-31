#!/usr/bin/env python3
"""Merge sharded `ab_test_price_primitives.py --out` results into one verdict.

Each shard covers one horizon (all three arms), so merging is a union over
horizons. Adds a per-fold win count between arms, which the pooled dir-acc
delta alone can hide: one lucky fold can carry a pooled gain that 8 of 10
folds contradict.

Usage:
    python scripts/merge_price_primitives_ab.py /tmp/ab_*.json
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts.ab_test_price_primitives import print_comparison

ARMS = ("baseline", "treatment", "placebo")


def merge(paths):
    merged = {}
    for p in paths:
        shard = json.loads(Path(p).read_text())
        for horizon, arms in shard.items():
            h = int(horizon)
            if h in merged:
                overlap = set(merged[h]) & set(arms)
                if overlap:
                    raise SystemExit(
                        f"Two shards both report {sorted(overlap)} for {h}d — "
                        f"refusing to pick one. Check the shard set."
                    )
                merged[h].update(arms)
            else:
                merged[h] = dict(arms)
    return merged


def fold_win_counts(merged):
    """Per-horizon (treatment>baseline, treatment>placebo) fold win counts."""
    out = {}
    for h, arms in merged.items():
        if not all(a in arms and arms[a].get("per_fold") for a in ARMS):
            continue
        by_arm = {
            a: {f["val_start"]: f["dir_acc"] for f in arms[a]["per_fold"]}
            for a in ARMS
        }
        shared = set.intersection(*(set(v) for v in by_arm.values()))
        t_b = sum(1 for k in shared if by_arm["treatment"][k] > by_arm["baseline"][k])
        t_p = sum(1 for k in shared if by_arm["treatment"][k] > by_arm["placebo"][k])
        out[h] = (t_b, t_p, len(shared))
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)

    merged = merge(paths)

    missing = {
        h: [a for a in ARMS if a not in arms] for h, arms in merged.items()
    }
    missing = {h: m for h, m in missing.items() if m}

    print_comparison(merged)

    wins = fold_win_counts(merged)
    if wins:
        print(f"  {'=' * 100}")
        print("  PER-FOLD WIN COUNTS (a pooled delta can rest on one fold)")
        print(f"  {'=' * 100}")
        for h in sorted(wins):
            t_b, t_p, n = wins[h]
            print(f"    {h:>2}d  treatment>baseline {t_b}/{n} folds   "
                  f"treatment>placebo {t_p}/{n} folds")

    deltas = []
    for h, arms in merged.items():
        if "baseline" in arms and "treatment" in arms:
            deltas.append(arms["treatment"]["dir_acc"] - arms["baseline"]["dir_acc"])
    if deltas:
        print(f"\n  Mean treatment-baseline across {len(deltas)} horizons: "
              f"{np.mean(deltas):+.2f}pp")

    if missing:
        print("\n  ⚠️  Incomplete — these horizons are missing arms, so the ship "
              "rule cannot be applied to them:")
        for h in sorted(missing):
            print(f"    {h}d: missing {', '.join(missing[h])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
