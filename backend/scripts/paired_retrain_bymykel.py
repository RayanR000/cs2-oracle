#!/usr/bin/env python3
"""Paired production-path retrain: does the ByMykel metadata bundle move CV accuracy?

`2026-08-06-bymykel-metadata-ingest.md` measures the bundle at +1.26pp (7d) and
+3.72pp (30d) on `ab_test_item_metadata.py` -- a held-out-item CV instrument over
an 870-item deep >=$1 universe, 33 baseline columns, its own fold scheme. That is
not the production path, and this project's record is that the two disagree:
market-relative labels cleared essentially that same harness and were refuted the
same day on the live retrain path.

This runs the REAL training path twice and pairs the folds:

    control    BYMYKEL_METADATA unset -- byte-identical to today's production
    treatment  BYMYKEL_METADATA=1     -- the nine columns admitted to the allowlist

Folds are paired on identical `(train_start, train_end, val_start, val_end)`, the
same rule `2026-08-06-paired-retrain-measures-no-gain.md` used, and the metric is
`classifier_accuracy_ge1` -- the >=$1 cohort the production headline reports, not
the all-tiers figure that reads close to the penny-item score.

Pre-registered bar, fixed before the run (see the changelog):

    > +1pp paired at two or more horizons, and not worse than -1pp at any.

Safety:
  * Each arm trains into its OWN scratch model_dir, so `models/saved_models/` is
    never touched and the two arms cannot share an engineered-feature cache.
  * Read-only against the database. Nothing is written to price-archive/.
  * The per-arm feature count is logged and asserted to differ. A silently
    identical treatment is the failure mode that makes a null unreadable.

Usage:
    python scripts/paired_retrain_bymykel.py --out /tmp/paired.json
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import logging
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from database import SessionLocal
from models.forecaster import ItemForecaster
from scripts.forecast_prices import (
    TRAIN_HORIZON_MAX_ROWS,
    DEFAULT_TRAIN_FEATURE_ROWS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("paired_retrain_bymykel")

METRIC = "classifier_accuracy_ge1"
WINDOW_KEYS = ("train_start", "train_end", "val_start", "val_end")
# Two-sided alpha=0.05 at 80% power: z(0.975) + z(0.80).
MDE_Z = 1.96 + 0.8416


def build_placebo(source: Path, dest: Path, seed: int = 20260806) -> Path:
    """Permute the metadata across items, keeping each column's distribution.

    Whole rows are shuffled against `item_slug` rather than each column being
    permuted independently, so the correlation structure between the nine
    columns is preserved and only the item->metadata assignment is destroyed.
    A treatment effect that survives this is not the columns carrying signal.

    This arm is not optional. `ab-fold-count-floor`: at ~7 folds a permuted
    placebo has already read significantly positive on this project's designs,
    and the paired retrain runs 8-9 folds.
    """
    import pandas as pd
    df = pd.read_parquet(source)
    values = df.drop(columns=["item_slug"])
    shuffled = values.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = pd.concat([df[["item_slug"]].reset_index(drop=True), shuffled], axis=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    logger.info(f"  placebo table: {len(out):,} rows permuted -> {dest.name}")
    return dest


def run_arm(name: str, enabled: bool, model_dir: Path,
            feature_rows: int, horizon_rows: int,
            metadata_path: Path | None = None) -> dict:
    """Train one arm and return {horizon: [fold metric dicts]} plus its features."""
    logger.info("=" * 70)
    logger.info(f"ARM: {name}  (BYMYKEL_METADATA={'1' if enabled else 'unset'})")
    logger.info("=" * 70)

    if enabled:
        os.environ["BYMYKEL_METADATA"] = "1"
    else:
        os.environ.pop("BYMYKEL_METADATA", None)
    if metadata_path is not None:
        os.environ["BYMYKEL_METADATA_PATH"] = str(metadata_path)
    else:
        os.environ.pop("BYMYKEL_METADATA_PATH", None)

    model_dir.mkdir(parents=True, exist_ok=True)
    db = SessionLocal()
    t0 = time.time()
    try:
        fc = ItemForecaster(db_session=db, model_dir=str(model_dir))
        fc.train(max_rows=horizon_rows, max_feature_rows=feature_rows)
        folds = {int(h): cv.get("per_fold", [])
                 for h, cv in fc.cv_results.items()}
        features = list(fc.feature_cols)
    finally:
        # The prod pooler drops the connection during a long train -- at the
        # 700K budget a single arm is ~14.5 min -- so close() itself raises
        # `SSL SYSCALL error: EOF detected` and would discard a completed arm's
        # results on the way out. scripts/forecast_prices.py guards the same
        # call for the same reason.
        try:
            db.close()
        except Exception as exc:
            logger.warning(f"  db.close() failed (connection already dropped): {exc}")

    elapsed = time.time() - t0
    meta_cols = sorted(set(features) & ItemForecaster.BYMYKEL_META_FEATURES)
    logger.info(f"  {name}: {len(features)} features "
                f"({len(meta_cols)} from the bundle), {elapsed:.0f}s")
    return {"folds": folds, "features": features,
            "meta_features": meta_cols, "elapsed_s": elapsed}


def pair_folds(control: dict, treatment: dict) -> dict:
    """Pair on the window tuple and difference the metric, per horizon.

    Folds are matched rather than zipped: voiding labels or changing the frame
    can reschedule the expanding-window folds at the tail, so arm i's fold 7 is
    not necessarily arm j's fold 7. An unmatched fold is dropped and reported,
    never silently aligned by position.
    """
    out = {}
    for horizon in sorted(set(control) | set(treatment)):
        c_by_window = {tuple(f[k] for k in WINDOW_KEYS): f
                       for f in control.get(horizon, [])}
        t_by_window = {tuple(f[k] for k in WINDOW_KEYS): f
                       for f in treatment.get(horizon, [])}
        shared = sorted(set(c_by_window) & set(t_by_window))

        diffs, pairs = [], []
        for window in shared:
            c_val = c_by_window[window].get(METRIC)
            t_val = t_by_window[window].get(METRIC)
            # None, not 0.0, when a fold held no >=$1 rows. An empty partition
            # has no accuracy; scoring it as zero would drag the mean down.
            if c_val is None or t_val is None:
                continue
            diffs.append(t_val - c_val)
            pairs.append({"window": window, "control": c_val,
                          "treatment": t_val, "diff": t_val - c_val})

        n = len(diffs)
        arr = np.array(diffs, dtype=float)
        sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
        mean = float(arr.mean()) if n else float("nan")
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        out[horizon] = {
            "n_folds_paired": n,
            "n_folds_control": len(c_by_window),
            "n_folds_treatment": len(t_by_window),
            "n_unmatched": len(set(c_by_window) ^ set(t_by_window)),
            "mean_diff_pp": mean,
            "sd_pp": sd,
            "t": mean / se if n > 1 and se > 0 else float("nan"),
            "ci_low": mean - 1.96 * se if n > 1 else float("nan"),
            "ci_high": mean + 1.96 * se if n > 1 else float("nan"),
            "mde_80_pp": MDE_Z * se if n > 1 else float("nan"),
            "control_mean": float(np.mean([p["control"] for p in pairs])) if pairs else float("nan"),
            "treatment_mean": float(np.mean([p["treatment"] for p in pairs])) if pairs else float("nan"),
            "pairs": pairs,
        }
    return out


def verdict(paired: dict) -> tuple[bool, str]:
    """The pre-registered rule: >+1pp at >=2 horizons, and never worse than -1pp."""
    wins = [h for h, r in paired.items() if r["mean_diff_pp"] > 1.0]
    losses = [h for h, r in paired.items() if r["mean_diff_pp"] < -1.0]
    if losses:
        return False, (f"FAIL — worse than -1pp at {sorted(losses)}d "
                       f"(rule 2), regardless of the {len(wins)} horizon(s) above +1pp")
    if len(wins) >= 2:
        return True, f"PASS — above +1pp at {sorted(wins)}d and never below -1pp"
    return False, (f"FAIL — above +1pp at {len(wins)} horizon(s) "
                   f"{sorted(wins)}d, rule 1 needs 2")


def print_summary(paired: dict, arms: dict, placebo: dict | None = None) -> None:
    print("\n" + "=" * 78)
    print("PAIRED PRODUCTION RETRAIN — ByMykel metadata bundle")
    print("=" * 78)
    print(f"metric: {METRIC} (>=$1 cohort, the production headline population)")
    print(f"control  : {len(arms['control']['features'])} features, "
          f"{arms['control']['elapsed_s']:.0f}s")
    print(f"treatment: {len(arms['treatment']['features'])} features "
          f"(+{len(arms['treatment']['meta_features'])} bundle), "
          f"{arms['treatment']['elapsed_s']:.0f}s")
    print()
    print(f"{'h':>4} {'folds':>6} {'control':>9} {'treatment':>10} "
          f"{'diff':>8} {'95% CI':>20} {'MDE80':>7}")
    print("-" * 78)
    for h in sorted(paired):
        r = paired[h]
        ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
        print(f"{h:>3}d {r['n_folds_paired']:>6} {r['control_mean']:>8.2f}% "
              f"{r['treatment_mean']:>9.2f}% {r['mean_diff_pp']:>+7.2f}pp "
              f"{ci:>20} {r['mde_80_pp']:>6.2f}pp")
        if r["n_unmatched"]:
            print(f"      ({r['n_unmatched']} unmatched fold window(s) dropped)")

    if placebo:
        print("\nplacebo (metadata permuted across items) — the null band:")
        for h in sorted(placebo):
            p = placebo[h]
            ci = f"[{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]"
            flag = "" if p["ci_low"] <= 0 <= p["ci_high"] else "   <-- NOT ZERO"
            print(f"{h:>3}d {p['n_folds_paired']:>6} "
                  f"{p['mean_diff_pp']:>+29.2f}pp {ci:>20}{flag}")
        print("Read every treatment number against this row, not against 0.")

    ok, why = verdict(paired)
    print("\n" + why)
    print("Note: the rule is on point estimates. A horizon whose diff is below "
          "its own MDE80\nis not resolvable by this instrument, however the rule "
          "scores it.")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired retrain: ByMykel metadata")
    parser.add_argument("--out", default=None)
    parser.add_argument("--feature-rows", type=int,
                        default=DEFAULT_TRAIN_FEATURE_ROWS)
    parser.add_argument("--horizon-rows", type=int, default=TRAIN_HORIZON_MAX_ROWS)
    parser.add_argument("--work-dir", default=None,
                        help="Scratch root for the two arms' model dirs")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the scratch model dirs after the run")
    parser.add_argument("--skip-placebo", action="store_true",
                        help="Skip the permuted arm. Do not use for a result "
                             "you intend to act on — see build_placebo.")
    args = parser.parse_args()

    work = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="paired_bymykel_"))
    logger.info(f"Scratch model dirs under {work} "
                f"(models/saved_models/ is not touched)")

    arms = {}
    try:
        arms["control"] = run_arm("control", False, work / "control",
                                  args.feature_rows, args.horizon_rows)
        # Placebo BEFORE treatment, deliberately: reading it second invites
        # rationalising it against a number already seen.
        if not args.skip_placebo:
            source = Path(os.environ.get(
                "BYMYKEL_METADATA_PATH",
                Path(__file__).parent.parent.parent / "price-archive"
                / "item-metadata-bymykel.parquet"))
            arms["placebo"] = run_arm(
                "placebo", True, work / "placebo",
                args.feature_rows, args.horizon_rows,
                metadata_path=build_placebo(source, work / "placebo-meta.parquet"))
        arms["treatment"] = run_arm("treatment", True, work / "treatment",
                                    args.feature_rows, args.horizon_rows)

        # A treatment that silently equals control makes the null unreadable.
        # This is the single most likely way this measurement lies, so it is an
        # assertion and not a log line.
        if not arms["treatment"]["meta_features"]:
            raise SystemExit(
                "treatment arm trained with ZERO ByMykel columns in "
                "feature_cols. Check that "
                "price-archive/item-metadata-bymykel.parquet exists (run "
                "scripts/ingest_bymykel_metadata.py) — the measurement is "
                "meaningless without it.")

        paired = pair_folds(arms["control"]["folds"], arms["treatment"]["folds"])
        placebo = (pair_folds(arms["control"]["folds"], arms["placebo"]["folds"])
                   if "placebo" in arms else None)
        print_summary(paired, arms, placebo)

        if args.out:
            ok, why = verdict(paired)
            Path(args.out).write_text(json.dumps({
                "metric": METRIC,
                "verdict": why,
                "passed": ok,
                "paired": paired,
                "placebo": placebo,
                "arms": {k: {kk: vv for kk, vv in v.items() if kk != "folds"}
                         for k, v in arms.items()},
            }, indent=2, default=str))
            logger.info(f"Wrote {args.out}")
    finally:
        if not args.keep and args.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
