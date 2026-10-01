#!/usr/bin/env python3
"""WHY do the published band and the score disagree? Split the wedge by cause.

`dollar_band_wedge.py` established the size of the problem: published dollar
coverage runs 54-72% against a band calibrated to 83-90%, and 67-73% of the
misses are recoverable — inside the rebased band, so the wedge alone explains
them. It could not establish the CAUSE. The served `current_price` could not be
rebuilt from the archive (six candidate reconstructions matched at most 11.7% of
rows), so "the serving anchor is computed wrong" and "the serving anchor is
computed right on data that later moved" were indistinguishable, and three
anchor arms were being chosen against a quote nobody could reproduce.

`2026-08-25-centre-shrinks-to-zero-and-the-dollar-band-is-the-wedge.md` §9
settled the first half: the serving computation matches `resolve_anchors`
EXACTLY (5,536 of 5,536 items, p90 diff 0.00000) when both read the same
archive. That leaves the data moving between serve time and scoring time — a
hypothesis that could only be tested going forward, because the archive is
published as an orphan force-push and keeps no previous version of any day.

`ANCHOR_AUDIT=1` (live in price-forecast.yml since 2026-08-25) is that forward
record: raw quote, smoothed value, and the resolved served base, per item per
anchor, published into `price-archive/ops/anchor_audit/`. This reads it.

METHOD — three multiplicative legs, telescoping exactly onto the total:

    total     = base_price / current_price - 1     (the wedge as scored)

    capture   = served_base / current_price - 1    GATE, not a finding
                What the audit recorded vs what the forecast row stored. Must be
                ~0. Anything else means the audit is not observing the serving
                path and every figure below is void.

    revision  = base_now / served_base - 1         THE HYPOTHESIS
                Same anchor date, same definition, TODAY's archive. The
                computation is held fixed by construction — `resolve_anchors` is
                imported, not reimplemented — so this leg is the input data
                moving and nothing else.

    residual  = base_price / base_now - 1
                Scoring-time resolution vs now. Further drift after the score
                was taken; also the catch-all for anything the two legs above do
                not name.

Multiplicative on purpose: an additive split leaves a cross term that grows with
the wedge, and the wedge's entire interest is in its tail.

READ IT AS: if `revision` carries the mass and `residual` is small, the
hypothesis holds and no anchor arm can help — the fix is to score against the
frozen served basis. If `residual` carries it instead, the two resolution times
differ for a reason this does not name and the question is still open.

REQUIRES at least one published audit file. The chain has been paused since
2026-08-20 (GH Actions billing), and the workflow flag landed 08-25, so as of
writing `ops/anchor_audit/` is EMPTY and this reports nothing. That is the
expected state, not a failure.

Run: backend/venv/bin/python scripts/anchor_wedge_attribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

# The audit must reproduce the stored quote to floating-point noise, not to a
# tolerance. Both are the same float that `predict` computed; a real difference
# means the audit is reading a different object, and there is no "small" version
# of that failure.
CAPTURE_TOL = 1e-9
CAPTURE_MIN_SHARE = 0.999

_LEGS = ("capture_gap", "revision", "residual")


def attribute(df: pd.DataFrame) -> pd.DataFrame:
    """The three legs plus the total, per row. Pure: no archive, no DB.

    Rows with a non-positive or missing price on ANY leg are dropped rather than
    carried: each leg is a ratio, and one zero anchor divides to inf and
    dominates every median it lands in.
    """
    cols = ["current_price", "served_base", "base_now", "base_price"]
    out = df.copy()
    vals = {c: pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float) for c in cols}
    usable = np.ones(len(out), dtype=bool)
    for c in cols:
        usable &= np.isfinite(vals[c]) & (vals[c] > 0)
    out = out[usable].copy()
    q, s = vals["current_price"][usable], vals["served_base"][usable]
    n, b = vals["base_now"][usable], vals["base_price"][usable]

    out["capture_gap"] = s / q - 1.0
    out["revision"] = n / s - 1.0
    out["residual"] = b / n - 1.0
    out["total_wedge"] = b / q - 1.0
    return out.reset_index(drop=True)


def capture_verdict(df: pd.DataFrame) -> tuple[bool, str]:
    """Did the audit record what was actually served? A precondition, so it is
    reported as a pass/fail rather than as one number among the findings."""
    if df.empty:
        return False, "capture gate: no usable rows"
    share = float((df["capture_gap"].abs() <= CAPTURE_TOL).mean())
    ok = share >= CAPTURE_MIN_SHARE
    return ok, (
        f"capture gate {'PASS' if ok else 'FAIL'}: "
        f"{share:.4%} of {len(df):,} rows reproduce the served quote to "
        f"{CAPTURE_TOL:g} (need {CAPTURE_MIN_SHARE:.1%}); "
        f"median |gap| {df['capture_gap'].abs().median():.6%}"
    )


def _leg_shares(df: pd.DataFrame) -> dict[str, float]:
    """How much of the wedge's MAGNITUDE each leg carries.

    In log space, because that is the only scale on which multiplicative legs
    add. Shares are of summed absolute log-moves, so a leg that cancels another
    still shows the work it did rather than netting to zero.
    """
    mag = {leg: float(np.abs(np.log1p(df[leg].to_numpy())).sum()) for leg in _LEGS}
    total = sum(mag.values())
    return {leg: (v / total if total else float("nan")) for leg, v in mag.items()}


def _report(df: pd.DataFrame) -> dict:
    """Pooled and per-date. Medians and p90 — the tail is the point."""

    def stats(g):
        row = {"n": len(g)}
        for leg in (*_LEGS, "total_wedge"):
            row[f"{leg}_median"] = float(g[leg].median())
            row[f"{leg}_p90"] = float(g[leg].abs().quantile(0.90))
        row["shares"] = _leg_shares(g)
        return row

    out = {"pooled": stats(df), "by_date": {}}
    for d, g in df.groupby("anchor_date"):
        out["by_date"][str(d)] = stats(g)
    return out


def _print(rep: dict) -> None:
    def line(label, r):
        s = r["shares"]
        print(
            f"  {label:<14} n={r['n']:>7,}  "
            f"total {r['total_wedge_median']:+.3%} (p90 |{r['total_wedge_p90']:.3%}|)  "
            f"| revision {r['revision_median']:+.3%} "
            f"residual {r['residual_median']:+.3%}  "
            f"| share rev {s['revision']:.1%} res {s['residual']:.1%} "
            f"cap {s['capture_gap']:.1%}"
        )

    print("\nWedge attribution (median per row; share = of summed |log move|)")
    line("POOLED", rep["pooled"])
    if len(rep["by_date"]) > 1:
        print("  --- by anchor date ---")
        for d, r in sorted(rep["by_date"].items()):
            line(d, r)


# --------------------------------------------------------------------------
# Loading. Everything below touches the archive; the maths above does not.
# --------------------------------------------------------------------------


def _load_audit(audit_dir: Path) -> pd.DataFrame:
    """Every published audit file, concatenated.

    `served_base` is read, never recomputed. Files written before it was
    recorded are SKIPPED rather than back-filled from the shipped arm: under
    `SERVE_OUTLIER_GATED_ANCHOR` the base is a per-item choice, and guessing it
    is exactly the ambiguity the column was added to remove.
    """
    files = sorted(audit_dir.glob("anchor_audit_*.parquet"))
    if not files:
        raise SystemExit(
            f"no audit files in {audit_dir}. ANCHOR_AUDIT=1 is live in "
            f"price-forecast.yml, but the forecast chain has been paused since "
            f"2026-08-20 — there is nothing to read yet."
        )
    frames, skipped = [], []
    for f in files:
        d = pd.read_parquet(f)
        if "served_base" not in d.columns:
            skipped.append(f.name)
            continue
        frames.append(d)
    for name in skipped:
        print(f"skipped {name}: predates the served_base column")
    if not frames:
        raise SystemExit("every audit file predates the served_base column")
    out = pd.concat(frames, ignore_index=True)
    out["anchor_date"] = pd.to_datetime(out["anchor_date"]).dt.date
    print(f"audit: {len(out):,} rows over {out['anchor_date'].nunique()} anchor date(s)")
    return out


def _slug_map(archive_dir: Path | None) -> dict:
    """item_id -> slug, from the outcomes panel when there is one, else prod."""
    if archive_dir is not None:
        import duckdb

        p = archive_dir / "ops" / "forecast_outcomes.parquet"
        d = (
            duckdb.connect()
            .sql(f"SELECT DISTINCT item_id, item_slug FROM read_parquet('{p}') WHERE item_slug IS NOT NULL")
            .fetchdf()
        )
        return dict(zip(d["item_id"], d["item_slug"]))
    from backtest_accuracy import _id_to_slug
    from database import SessionLocal

    db = SessionLocal()
    try:
        return _id_to_slug(db)
    finally:
        db.close()


def _base_now(pairs: set, archive: Path) -> dict:
    """Re-resolve each (slug, anchor_date) against TODAY's archive.

    Imported from `backtest.price_resolution`, never reimplemented: the leg only
    means "the data moved" if the computation is provably held fixed.
    """
    from backtest.price_resolution import load_voted_prices, resolve_anchors

    slugs = sorted({s for s, _ in pairs})
    dates = [d for _, d in pairs]
    voted = load_voted_prices(archive, slugs, min(dates), max(dates))
    resolved = resolve_anchors(voted, pairs)
    print(f"re-resolved {len(resolved):,} of {len(pairs):,} (slug, anchor) pairs")
    return {k: v.price for k, v in resolved.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--archive-dir", default=None, type=Path, help="read outcomes from this archive's Parquet instead of prod"
    )
    ap.add_argument("--audit-dir", default=None, type=Path, help="defaults to <archive>/ops/anchor_audit")
    ap.add_argument(
        "--price-archive",
        default=None,
        type=Path,
        help="archive the re-resolution reads; defaults to the canonical clone",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(BACKEND / ".env")

    from centre_vs_lastprice import _load
    from resolve_readonly import DEFAULT_ARCHIVE

    price_archive = args.price_archive or DEFAULT_ARCHIVE
    audit_dir = args.audit_dir or (price_archive / "ops" / "anchor_audit")

    audit = _load_audit(audit_dir)
    out = _load(args.archive_dir)

    # The audit's `item_id` column is the serving path's key, which is the
    # market_hash_name — NOT the integer `forecast_outcomes.item_id`. Both sides
    # must therefore be expressed as slugs BEFORE the join, so the map is
    # applied here rather than after it.
    slugs = _slug_map(args.archive_dir)
    out["item_slug"] = out["item_id"].map(slugs)
    out = out.dropna(subset=["item_slug"])
    merged = out.merge(
        audit[["item_id", "anchor_date", "price", "_smoothed_price", "served_base"]].rename(
            columns={"item_id": "item_slug"}
        ),
        left_on=["item_slug", "forecast_date"],
        right_on=["item_slug", "anchor_date"],
        how="inner",
    )
    if merged.empty:
        raise SystemExit(
            "no outcome row shares an anchor date with an audit file. The audit "
            "starts when the chain resumes; outcomes for those dates only exist "
            "once their horizon has elapsed."
        )
    print(f"joined {len(merged):,} outcome rows to the audit")

    pairs = set(zip(merged["item_slug"], merged["forecast_date"]))
    now = _base_now(pairs, price_archive)
    merged["base_now"] = [now.get((s, d)) for s, d in zip(merged["item_slug"], merged["forecast_date"])]

    df = attribute(merged)
    ok, msg = capture_verdict(df)
    print(msg)
    if not ok:
        # Loud and terminal. A capture failure does not degrade the finding, it
        # voids it: the legs are defined off a quote the audit did not observe.
        print("\nABORT: the decomposition below would be meaningless.")
        return 1

    rep = _report(df)
    _print(rep)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
