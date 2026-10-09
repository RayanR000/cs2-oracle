#!/usr/bin/env python3
"""Date-overlap sensitivity rows for the two October reads. Reported only.

The preregistrations freeze iid date bootstraps, and neither frozen instrument
is edited (the h=7 prereg diffs `outside_baseline.py` and `backtest/scoring.py`
against 997e31c before reading). This script imports them, rebuilds the exact
per-date series each one resamples, reproduces its iid interval, and prints a
moving-block and a HAC interval beside it. How the rows are read is fixed in
`docs/research/2026-10-09-date-overlap-sensitivity.md`, written before both reads.

    # PID read (h=3, served panel, prod read-only), not before 2026-10-23:
    venv/bin/python scripts/date_dependence_sensitivity.py pid

    # outside-baseline reads; same --archive-dir/--since/--until as the read itself:
    venv/bin/python scripts/date_dependence_sensitivity.py outside-baseline \\
        --archive-dir <dir> --horizon 7 --since 2026-09-27 --until <D20>

Exit codes: 0 = reported; 2 = refused (too early, or the iid row did not
reproduce the frozen instrument's interval, which means the series is wrong).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.date_dependence import sensitivity

# A preregistered window opens here per horizon; reading inside it needs the
# read's own condition (20 scoreable dates).
OUTSIDE_BASELINE_WINDOWS = {7: date(2026, 9, 27), 3: date(2026, 9, 28)}
MIN_READ_DATES = 20


def _print(name: str, s: dict, frozen: tuple[float, float]) -> None:
    pct = round(s["level"] * 100)
    print(f"\n{name}: n_dates={s['n_dates']} mean={s['mean']:+.4f}")
    print("  acf " + " ".join(f"{k}:{v:+.2f}" if v is not None else f"{k}:na" for k, v in s["acf"].items()))
    print(f"  frozen iid {pct}% CI        [{frozen[0]:+.4f}, {frozen[1]:+.4f}]")
    for k in ("iid", "moving_block", "hac"):
        lo, hi = s[k]
        ratio = "" if k == "iid" else f"  width x{s['width_vs_iid'][k]:.2f}"
        print(f"  {k:<12} {pct}% CI      [{lo:+.4f}, {hi:+.4f}]  {s['sign'][k]}{ratio}")
    print(f"  FRAGILE TO DATE OVERLAP: {'YES' if s['fragile'] else 'no'}")


def _reproduced(s: dict, frozen: tuple[float, float]) -> bool:
    ok = bool(np.allclose(s["iid"], frozen, rtol=0, atol=1e-9))
    if not ok:
        print(f"REFUSED: iid row {s['iid']} does not reproduce the frozen interval {frozen}")
    return ok


def run_pid(args, today: date) -> int:
    import measure_conformal_pid as pid

    if today < pid.READ_NOT_BEFORE:
        print(f"REFUSED: the PID panel is not read before {pid.READ_NOT_BEFORE} (its prereg's single read).")
        return 2
    from database import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        db.execute(text("SET TRANSACTION READ ONLY"))
        panel = pid.build_panel(pid.load_prod_panel(db))
    finally:
        db.close()
    scores = pid.evaluate(panel)["scores"]
    a, b = scores["QI"]["cov_by_date"], scores["B"]["cov_by_date"]
    diff = (np.abs(a - (1 - pid.ALPHA)) - np.abs(b - (1 - pid.ALPHA))) * 100
    _, lo, hi = pid.paired_bootstrap(a, b)
    s = sensitivity(panel.dates, diff, pid.HORIZON, 0.95, pid.N_BOOT, pid.BOOT_SEED)
    if not _reproduced(s, (lo, hi)):
        return 2
    _print("PID h=3: |cov_QI - .8| - |cov_B - .8| (pp; below zero favours QI)", s, (lo, hi))
    _dump(args, s)
    return 0


def run_outside_baseline(args) -> int:
    import outside_baseline as ob

    panel = ob.load_panel(args.archive_dir)
    panel = panel[(panel["horizon_days"] == args.horizon) & panel["item_slug"].notna()]
    if args.since:
        panel = panel[panel["forecast_date"] >= args.since]
    if args.until:
        panel = panel[panel["forecast_date"] <= args.until]
    panel = panel.reset_index(drop=True)
    opens = OUTSIDE_BASELINE_WINDOWS.get(args.horizon)
    n_dates = panel["forecast_date"].nunique()
    if opens and panel["forecast_date"].max() >= opens and n_dates < MIN_READ_DATES:
        print(
            f"REFUSED: h={args.horizon} dates from {opens} are a preregistered read; "
            f"{n_dates} dates < {MIN_READ_DATES}, so this would be a peek."
        )
        return 2
    slugs = sorted(panel["item_slug"].unique())
    base = ob.build_baselines(args.archive_dir, panel, slugs, with_ets=False, workers=args.workers)
    df = panel.merge(base, on=["item_slug", "forecast_date", "horizon_days"], how="inner")
    df = df.dropna(subset=["naive_lo", "naive_hi"]).reset_index(drop=True)
    df["naive_lo"], df["naive_hi"] = np.exp(df["naive_lo"]), np.exp(df["naive_hi"])
    y = df["y"].to_numpy()
    df["is_served"] = ob.interval_score(df["served_lo"].to_numpy(), df["served_hi"].to_numpy(), y, alpha=ob.ALPHA)
    df["is_naive"] = ob.interval_score(df["naive_lo"].to_numpy(), df["naive_hi"].to_numpy(), y, alpha=ob.ALPHA)
    _, lo, hi = ob.paired_date_diff(df, "is_served", "is_naive")
    per = df.assign(_d=df["is_served"] - df["is_naive"]).groupby("forecast_date")["_d"].mean()
    s = sensitivity(list(per.index), per.to_numpy(), args.horizon, 0.90, 2000, ob.RNG_SEED)
    if not _reproduced(s, (lo, hi)):
        return 2
    _print(f"outside-baseline h={args.horizon}: IS_served - IS_naive (above zero favours naive)", s, (lo, hi))
    _dump(args, s)
    return 0


def _dump(args, s: dict) -> None:
    if args.json_out:
        args.json_out.write_text(json.dumps(s, indent=2, default=str))
        print(f"\nwrote {args.json_out}")


def main(argv=None, today: date | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="read", required=True)
    p = sub.add_parser("pid")
    p.add_argument("--json-out", type=Path)
    o = sub.add_parser("outside-baseline")
    o.add_argument("--archive-dir", required=True, type=Path)
    o.add_argument("--horizon", required=True, type=int)
    o.add_argument("--since", type=date.fromisoformat)
    o.add_argument("--until", type=date.fromisoformat)
    o.add_argument("--workers", type=int, default=4)
    o.add_argument("--json-out", type=Path)
    args = ap.parse_args(argv)
    if args.read == "pid":
        return run_pid(args, today or date.today())
    return run_outside_baseline(args)


if __name__ == "__main__":
    raise SystemExit(main())
