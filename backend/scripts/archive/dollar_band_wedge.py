#!/usr/bin/env python3
"""How much of the published dollar band's under-coverage is the anchor wedge?

The product serves a DOLLAR range and a consumer reads it in dollars. That band
covers far less than the 80% it is calibrated to — the 2026-08-25 backtest run
reported `IntCov=82.8% ($-basis 54.0%)` at h=14 — and the gap has a known
suspect: `predict()` quotes the triple off `current_price` (the SMOOTHED anchor,
a number no venue quoted, substituted unconditionally at forecaster.py:6443)
while the outcome resolves off `base_price` from `resolve_anchors`. The two
disagree on ~85% of rows.

`_derive_verdict` already reports both predicates — `in_interval` rebased by
`base/quote`, and `in_interval_dollar` raw — but nothing has attributed the gap
BETWEEN them, which is what decides whether re-anchoring the served quote (the
`2026-08-11-serving-anchor-freshness.md` arms) is worth shipping. Arm A passed
its own gate on dollar ERROR; its band-coverage effect was never measured. This
measures the ceiling on that effect.

METHOD (read-only, prod Postgres by default, no replay):

* Per horizon, on the >=$1 served cohort with `excluded_forecast_date` applied:
      calibrated coverage  low*rebase <= actual <= high*rebase   (rebase = base/quote)
      dollar coverage      low        <= actual <= high
* ATTRIBUTION. Every dollar miss is exactly one of:
      recoverable — inside the rebased band, so the miss is the wedge alone and
                    re-anchoring the quote would have covered it;
      genuine     — outside both, so the band was simply too narrow or
                    mis-centred and no anchor fix touches it.
  The recoverable share is the CEILING on what any anchor arm can buy, because
  it assumes the fix removes the wedge entirely.
* The wedge itself is reported as `base/quote - 1`, and dollar coverage is cut by
  its decile — if the story is right, coverage should fall off a cliff in the top
  deciles and sit near the calibrated level in the bottom ones.

LIMIT, stated up front: the archive holds the SMOOTHED quote, not the raw one, so
this measures quote-vs-resolved-anchor. That bundles the smoothing substitution
(what Arm A gates) with ordinary drift between the two resolution times. It
bounds the arm from above; it does not simulate it. Simulating Arm A needs a
replay with the raw price frame.

Run: backend/venv/bin/python scripts/dollar_band_wedge.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.archive.centre_vs_lastprice import SERVED_MIN_PRICE, _load

N_BOOTSTRAP = 1000
RNG_SEED = 42
WEDGE_DECILES = 10


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Both coverage predicates, per row, in dollars.

    Deliberately NOT in return space: the dollar predicate is the one a consumer
    evaluates, and moving it to returns would quietly re-anchor the thing under
    test.
    """
    b = df["base_price"].to_numpy()
    q = df["current_price"].to_numpy(dtype=float)
    q = np.where(np.isnan(q) | (q <= 0), b, q)  # `_quote_basis` fallback
    lo = df["predicted_price_low"].to_numpy()
    hi = df["predicted_price_high"].to_numpy()
    actual = df["actual_price"].to_numpy()

    out = df.copy()
    rebase = b / q
    out["wedge"] = b / q - 1.0
    out["cov_dollar"] = (lo <= actual) & (actual <= hi)
    out["cov_calibrated"] = (lo * rebase <= actual) & (actual <= hi * rebase)
    # A miss the wedge alone explains: outside the published band, inside the
    # calibrated one.
    out["recoverable"] = (~out["cov_dollar"]) & out["cov_calibrated"]
    out["genuine_miss"] = (~out["cov_dollar"]) & (~out["cov_calibrated"])
    return out.dropna(subset=["predicted_price_low", "predicted_price_high"])


def _bootstrap_gap(g: pd.DataFrame, rng: np.random.Generator) -> tuple:
    """90% CI on the coverage gap, resampling forecast DATES (market factor)."""
    dates = g["forecast_date"].unique()
    by_date = {d: sub for d, sub in g.groupby("forecast_date")}
    out = []
    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(dates, size=len(dates), replace=True)
        s = pd.concat([by_date[d] for d in draw], ignore_index=True)
        out.append(float(s["cov_calibrated"].mean() - s["cov_dollar"].mean()))
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-dir", default=None, help="read a Parquet copy instead of prod Postgres")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    df = _prepare(_load(Path(args.archive_dir) if args.archive_dir else None))
    rng = np.random.default_rng(RNG_SEED)

    print(
        f"\nserved cohort (>=${SERVED_MIN_PRICE:.0f}): {len(df):,} rows, "
        f"{df['forecast_date'].nunique()} forecast dates\n"
    )
    hdr = (
        f"{'h':>4} {'rows':>7} {'cov_$':>7} {'cov_cal':>8} {'gap':>7} "
        f"{'90% CI':>16} {'recoverable':>12} {'genuine':>8} "
        f"{'|wedge| med':>12} {'p90':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for h, g in df.groupby("horizon_days"):
        cov_d = float(g["cov_dollar"].mean())
        cov_c = float(g["cov_calibrated"].mean())
        lo, hi = _bootstrap_gap(g, rng)
        n_miss = int((~g["cov_dollar"]).sum())
        rec = float(g["recoverable"].sum() / n_miss) if n_miss else float("nan")
        gen = float(g["genuine_miss"].sum() / n_miss) if n_miss else float("nan")
        aw = g["wedge"].abs()
        print(
            f"{h:>4} {len(g):>7,} {cov_d:>7.3f} {cov_c:>8.3f} "
            f"{cov_c - cov_d:>+7.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>16} "
            f"{rec:>12.1%} {gen:>8.1%} {aw.median():>12.4f} "
            f"{aw.quantile(0.90):>7.4f}"
        )
        results.append(
            dict(
                horizon=int(h),
                rows=len(g),
                coverage_dollar=cov_d,
                coverage_calibrated=cov_c,
                gap=cov_c - cov_d,
                gap_ci90=[lo, hi],
                dollar_misses=n_miss,
                recoverable_share=rec,
                genuine_share=gen,
                wedge_abs_median=float(aw.median()),
                wedge_abs_p90=float(aw.quantile(0.90)),
            )
        )

    print(
        "\nrecoverable = share of DOLLAR misses that the rebased band covers, "
        "i.e. the ceiling on\n              what removing the wedge could buy. "
        "genuine = missed on both bases.\n"
    )

    print("=== dollar coverage by |wedge| decile (h pooled within each cut) ===")
    print(
        "If the wedge is the cause, coverage falls as the wedge grows while "
        "the calibrated\nband stays flat — the calibrated column is the "
        "control.\n"
    )
    hdr2 = f"{'decile':>7} {'|wedge| <=':>11} {'rows':>7} {'cov_$':>7} {'cov_cal':>8} {'gap':>7}"
    print(hdr2)
    print("-" * len(hdr2))
    aw = df["wedge"].abs()
    # `duplicates="drop"`: a heavily tied wedge distribution (many rows where the
    # two anchors agree exactly) collapses the low deciles into one edge.
    cuts = pd.qcut(aw, WEDGE_DECILES, labels=False, duplicates="drop")
    decile_rows = []
    for d in sorted(pd.Series(cuts).dropna().unique()):
        sel = df[cuts == d]
        edge = float(aw[cuts == d].max())
        cov_d = float(sel["cov_dollar"].mean())
        cov_c = float(sel["cov_calibrated"].mean())
        print(f"{int(d) + 1:>7} {edge:>11.4f} {len(sel):>7,} {cov_d:>7.3f} {cov_c:>8.3f} {cov_c - cov_d:>+7.3f}")
        decile_rows.append(
            dict(decile=int(d) + 1, wedge_upper=edge, rows=len(sel), coverage_dollar=cov_d, coverage_calibrated=cov_c)
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"by_horizon": results, "by_wedge_decile": decile_rows}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
