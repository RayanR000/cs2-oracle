#!/usr/bin/env python3
"""Merge sharded `ab_test_supply_side.py --arm ... --out` results into one verdict.

Each shard is one arm (`control`/`treatment`/`placebo`), optionally one horizon,
so merging is a union over (horizon, arm). A single-arm shard has nothing to
contrast in-process — which is the case this script exists for.

The verdict is a **paired, fold-clustered interval** on the dir-acc difference,
at FOLD grain. The harness pairs at row grain, which is strictly better, but the
rows do not survive `--out` (`without_records` strips them), so what a merge can
reach is the per-fold series `per_fold` carries, keyed on `val_start`. That is
the independent unit, and it is what makes this a test, not a win count. See
`.claude/rules/ab-statistics.md` and `merge_price_primitives_ab.py`.

Ship rule for the rarity feature: treatment must beat BOTH control AND placebo
(shuffled rarity). The placebo is the capacity control the 2026-08-13 leak audit
requires — the early-stopping leak pays capacity, so a rarity effect must beat
shuffled-rarity capacity, not merely the no-rarity control.

Usage:
    python scripts/merge_supply_side_ab.py /tmp/supply_*.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import fold_level_records

ARMS = ("control", "treatment", "placebo")


def merge(paths):
    merged = {}
    for p in paths:
        shard = json.loads(Path(p).read_text())
        for arm, horizons in shard.items():
            for horizon, result in horizons.items():
                h = int(horizon)
                merged.setdefault(h, {})
                if arm in merged[h]:
                    raise SystemExit(
                        f"Two shards both report arm '{arm}' for {h}d — refusing to pick one. Check the shard set."
                    )
                merged[h][arm] = result
    return merged


def paired_verdicts(merged):
    """Per-horizon paired, fold-clustered intervals against `control`.

    Folds keyed on `val_start` (the window's own first date), which identifies
    the same fold in every arm — a positional index would not, since an arm that
    skipped a starved fold shifts every later index.
    """
    out = {}
    for h, arms in merged.items():
        usable = {a: arms[a] for a in ARMS if a in arms and arms[a].get("per_fold")}
        if "control" not in usable or len(usable) < 2:
            continue
        shared = set.intersection(*({f["val_start"] for f in v["per_fold"]} for v in usable.values()))
        if len(shared) < 2:
            out[h] = {a: {"verdict": "unresolved", "n_clusters": len(shared)} for a in usable if a != "control"}
            continue
        folds = sorted(shared)
        records = {
            a: fold_level_records(
                folds,
                [next(f["dir_acc"] for f in v["per_fold"] if f["val_start"] == k) for k in folds],
                metric="dir_acc",
            )
            for a, v in usable.items()
        }
        out[h] = paired_arm_contrasts(records, base="control", value_key="dir_acc", scale=1.0)
    return out


def fold_win_counts(merged):
    """Per-horizon (treatment>control, treatment>placebo) fold win counts."""
    out = {}
    for h, arms in merged.items():
        if not all(a in arms and arms[a].get("per_fold") for a in ARMS):
            continue
        by_arm = {a: {f["val_start"]: f["dir_acc"] for f in arms[a]["per_fold"]} for a in ARMS}
        shared = set.intersection(*(set(v) for v in by_arm.values()))
        t_c = sum(1 for k in shared if by_arm["treatment"][k] > by_arm["control"][k])
        t_p = sum(1 for k in shared if by_arm["treatment"][k] > by_arm["placebo"][k])
        out[h] = (t_c, t_p, len(shared))
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)

    merged = merge(paths)
    missing = {h: [a for a in ARMS if a not in arms] for h, arms in merged.items()}
    missing = {h: m for h, m in missing.items() if m}

    print("\n  Per-arm dir-acc (NOT a verdict; quotable only beside the paired")
    print("  interval and realised down-rate):")
    for h in sorted(merged):
        arms = merged[h]
        cells = "  ".join(f"{a}={arms[a].get('directional_accuracy', float('nan')):.1f}%" for a in ARMS if a in arms)
        print(f"    {h:>2}d  {cells}")

    verdicts = paired_verdicts(merged)
    if verdicts:
        print(f"\n  {'=' * 80}")
        print("  PAIRED, FOLD-CLUSTERED (dir-acc difference vs control)")
        print("  Ship rarity only if treatment beats BOTH control and placebo.")
        print(f"  {'=' * 80}")
        for h in sorted(verdicts):
            for arm in sorted(verdicts[h]):
                print(f"    {h:>2}d  {arm:<10} {format_paired(verdicts[h][arm])}")

    wins = fold_win_counts(merged)
    if wins:
        print(f"\n  {'=' * 80}")
        print("  PER-FOLD WIN COUNTS — context, not a gate")
        print(f"  {'=' * 80}")
        for h in sorted(wins):
            t_c, t_p, n = wins[h]
            print(f"    {h:>2}d  treatment>control {t_c}/{n}   treatment>placebo {t_p}/{n}")

    deltas = [
        merged[h]["treatment"]["directional_accuracy"] - merged[h]["control"]["directional_accuracy"]
        for h in merged
        if "control" in merged[h] and "treatment" in merged[h]
    ]
    if deltas:
        print(
            f"\n  Mean treatment-control across {len(deltas)} horizons: "
            f"{np.mean(deltas):+.2f}pp (context, not a verdict)"
        )

    if missing:
        print("\n  ⚠️  Incomplete — these horizons are missing arms, so the ship rule cannot be applied:")
        for h in sorted(missing):
            print(f"    {h}d: missing {', '.join(missing[h])}")


if __name__ == "__main__":
    main()
