#!/usr/bin/env python3
"""Merge sharded `ab_test_price_primitives.py --out` results into one verdict.

Each shard covers one horizon (all three arms), so merging is a union over
horizons. `--arm` also shards by arm, and a single-arm shard has nothing to
contrast in-process — which is the case this script exists for.

The verdict is a **paired, fold-clustered interval** on the dir-acc difference,
at FOLD grain. The harness itself pairs at row grain, which is strictly better;
the rows do not survive into `--out` (they are hundreds of thousands per arm and
`without_records` strips them), so what a merge can reach is the per-fold series
`per_fold` already carries, keyed on `val_start`. That is still the independent
unit, and it is what makes this a test.

The per-fold win count is still printed, as context. Until 2026-08-08 it was the
only thing here, and a win count is not a test: two arms differing by a seed
alone clear "wins on more than half the folds" half the time.

Usage:
    python scripts/merge_price_primitives_ab.py /tmp/ab_*.json
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import fold_level_records
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


def paired_verdicts(merged):
    """Per-horizon paired, fold-clustered intervals against `baseline`.

    Folds are keyed on `val_start`, which is the window's own first date and so
    identifies the same fold in every arm — a positional index would not, since
    an arm that skipped a starved fold shifts every index after it.

    A horizon missing an arm is skipped rather than partially reported: the
    ship rule is treatment-vs-baseline *and* treatment-vs-placebo, and half of
    it is not a weaker version of it.
    """
    out = {}
    for h, arms in merged.items():
        usable = {a: arms[a] for a in ARMS
                  if a in arms and arms[a].get("per_fold")}
        if "baseline" not in usable or len(usable) < 2:
            continue
        shared = set.intersection(*(
            {f["val_start"] for f in v["per_fold"]} for v in usable.values()))
        if len(shared) < 2:
            # One shared fold is one cluster, and one cluster carries no
            # between-cluster variance to resample. `unresolved`, never `null`.
            out[h] = {a: {"verdict": "unresolved", "n_clusters": len(shared)}
                      for a in usable if a != "baseline"}
            continue
        folds = sorted(shared)
        records = {
            a: fold_level_records(
                folds,
                [next(f["dir_acc"] for f in v["per_fold"]
                      if f["val_start"] == k) for k in folds],
                metric="dir_acc")
            for a, v in usable.items()
        }
        out[h] = paired_arm_contrasts(
            records, base="baseline", value_key="dir_acc", scale=1.0)
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

    verdicts = paired_verdicts(merged)
    if verdicts:
        print(f"  {'=' * 100}")
        print("  PAIRED, FOLD-CLUSTERED (dir-acc difference vs baseline)")
        print(f"  {'=' * 100}")
        for h in sorted(verdicts):
            for arm in sorted(verdicts[h]):
                print(f"    {h:>2}d  {arm:<10} "
                      f"{format_paired(verdicts[h][arm])}")

    wins = fold_win_counts(merged)
    if wins:
        print(f"  {'=' * 100}")
        print("  PER-FOLD WIN COUNTS — context, not a gate")
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
