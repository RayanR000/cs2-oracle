#!/usr/bin/env python3
"""Served confirmation: does h=14 wedge-free ranking survive on realised outcomes?

Implements `docs/research/2026-09-13-served-centre-rank-preregistration.md` (+ same-day
amendment). Rank overlay on the `centre_vs_lastprice` served panel — no retrain,
read-only throughout: production r_hat vs quote-basis realised return, naive
`-return_1d` paired on identical rows, date-block bootstrap CIs.

Cohorts (stored legs + serve-time mask only, never a reconstruction):
  clean:  item_forecasts.anchor_clean == True (serve-time tied mask; primary iff
          non-null on >=70% of panel rows — mechanical rule in the prereg amendment)
  wedge:  |current_price/base_price - 1| <= 0.001 (fallback primary)
  exact:  current_price == base_price (sensitivity only)

Usage:
    venv/bin/python -m scripts.served_centre_rank --out /tmp/scr_h14.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from backtest.scoring import excluded_forecast_date
from database import SessionLocal
from models.forecaster import ItemForecaster

from scripts.clean_era_centre_ab import bootstrap_ci, paired_delta_ci, per_date_ic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("served_centre_rank")

HORIZON = 14
MIN_PRICE = 1.0
WEDGE_TOL = 0.001
MIN_ROWS_PER_DATE = 10
MIN_DATES = 20
MIN_COHORT_SHARE = 0.15
CLEAN_COVERAGE_GATE = 0.70

DURABLE_ARCHIVE = Path(__file__).parent.parent.parent.parent / "cs2-oracle-data" / "price-archive"


def derive_panel(df):
    """Served legs -> centre/outcome/cohorts. Pure function (tested).

    r_hat and the outcome share the quote basis (current_price, falling back
    to base exactly as centre_vs_lastprice does); rows on fallback carry no
    wedge information and sit outside every wedge cohort.
    """
    df = df.copy()
    for c in ("base_price", "actual_price", "current_price", "predicted_price_mid"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    b = df["base_price"].to_numpy(dtype=float)
    cur = df["current_price"].to_numpy(dtype=float)
    valid_cur = np.isfinite(cur) & (cur > 0)
    q = np.where(valid_cur, cur, b)
    mid = df["predicted_price_mid"].to_numpy(dtype=float)
    df["r_hat"] = mid / q - 1.0
    df["realized"] = df["actual_price"].to_numpy(dtype=float) / q - 1.0
    df["wedge"] = np.where(valid_cur, np.abs(cur / b - 1.0), np.nan)
    df["coh_wedge"] = valid_cur & (df["wedge"].to_numpy() <= WEDGE_TOL)
    df["coh_exact"] = valid_cur & (cur == b)
    clean = df["anchor_clean"].to_numpy()
    df["coh_clean"] = np.array([c is True for c in clean])
    # clean_known: non-null mask, NaN-safe (DB NULL -> None).
    df["clean_known"] = np.array([c is not None and not (isinstance(c, float) and np.isnan(c)) for c in clean])
    df["n_fallback"] = (~valid_cur).astype(int)
    return df


def select_primary(coverage):
    """Mechanical primary-cohort rule from the prereg amendment (tested)."""
    return "coh_clean" if coverage >= CLEAN_COVERAGE_GATE else "coh_wedge"


def attach_naive(df, naive_map):
    """Join the trailing-return naive signal; rows missing it leave BOTH arms
    (paired comparison on identical rows — pre-registered). Pure (tested)."""
    df = df.copy()
    keys = list(zip(df["slug"].astype(str), pd.to_datetime(df["forecast_date"]).dt.date))
    df["naive"] = np.array([-naive_map[k] if k in naive_map else np.nan for k in keys], dtype=float)
    keep = np.isfinite(df["naive"].to_numpy())
    return df[keep].reset_index(drop=True), int((~keep).sum())


def qualify(df, cohort_col, min_rows=MIN_ROWS_PER_DATE):
    """Forecast dates with >= min_rows cohort rows. Pure (tested)."""
    sub = df[df[cohort_col]]
    counts = sub.groupby("forecast_date").size()
    return sorted(d for d, n in counts.items() if n >= min_rows)


def summarize(df, dates, pred_col="r_hat", actual_col="realized"):
    """Per-date rank ICs on the given dates. Pure (tested)."""
    sub = df[df["forecast_date"].isin(set(dates))].copy()
    return per_date_ic(
        sub["forecast_date"].to_numpy(), sub[pred_col].to_numpy(dtype=float), sub[actual_col].to_numpy(dtype=float)
    )


def evaluate_confirmation(model_ci, paired_ci):
    """CONFIRMED iff model IC and paired edge both clear zero (tested)."""
    if model_ci is None or paired_ci is None:
        return "VOID"
    if model_ci["ci_low"] > 0 and paired_ci["ci_low"] > 0:
        return "CONFIRMED"
    return "NOT CONFIRMED"


def load_served():
    """h=14 served panel with serve-time mask + slug. Read-only."""
    from sqlalchemy import text

    db = SessionLocal()
    try:
        rows = db.execute(
            text("""
            SELECT o.forecast_date, o.base_price, o.actual_price,
                   o.current_price, o.predicted_price_mid,
                   i.item_id AS slug, f.anchor_clean
            FROM forecast_outcomes o
            JOIN item_forecasts f ON f.id = o.forecast_id
            JOIN items i ON i.id = o.item_id
            WHERE o.horizon_days = :h
              AND o.actual_price IS NOT NULL
              AND o.base_price > 0 AND o.base_price >= :min_price
              AND o.predicted_price_mid IS NOT NULL
        """),
            {"h": HORIZON, "min_price": MIN_PRICE},
        ).fetchall()
    finally:
        db.close()
    df = pd.DataFrame(
        rows,
        columns=[
            "forecast_date",
            "base_price",
            "actual_price",
            "current_price",
            "predicted_price_mid",
            "slug",
            "anchor_clean",
        ],
    )
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    reasons = df["forecast_date"].map(excluded_forecast_date)
    for why, n in reasons.dropna().value_counts().items():
        logger.info(f"  excluded {n:,} row(s): {why}")
    return df[reasons.isna()].reset_index(drop=True)


def load_naive_returns(slugs, day_min, day_max, archive_dir):
    """Calendar-exact trailing-1d log return on the voted composite, keyed
    (slug, date). Read-only DuckDB over the durable archive."""
    import duckdb
    from db.archive import prices_relation
    from models.item_parser import archive_universe_sql_filter

    con = duckdb.connect()
    try:
        rel = prices_relation(con, archive_dir=archive_dir)
        uni = archive_universe_sql_filter()
        lo = (pd.Timestamp(day_min) - pd.to_timedelta(3, unit="D")).date()
        placeholders = ", ".join("?" for _ in slugs)
        frame = con.sql(
            f"""
            SELECT item_slug, CAST(day AS DATE) AS day, source,
                   mean_price AS price, volume
            FROM {rel}
            WHERE CAST(day AS DATE) BETWEEN DATE '{lo}' AND DATE '{day_max}'
              AND item_slug IN ({placeholders})
              AND ({uni}) AND mean_price IS NOT NULL AND mean_price > 0
        """,
            params=list(slugs),
        ).fetchdf()
    finally:
        con.close()
    frame["day"] = pd.to_datetime(frame["day"])
    frame = frame.rename(columns={"item_slug": "item_id", "day": "date"})
    voted = ItemForecaster._apply_multi_source_voting(frame[["item_id", "date", "source", "price", "volume"]])
    voted = voted.sort_values(["item_id", "date"])
    out = {}
    for slug, g in voted.groupby("item_id"):
        g = g.sort_values("date")
        d = pd.to_datetime(g["date"])
        p = g["price"].to_numpy(dtype=float)
        gap = d.diff().dt.days.to_numpy() == 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.log(p[1:] / p[:-1])
        for ok, r, day in zip(gap[1:], ret, pd.to_datetime(g["date"]).dt.date.to_numpy()[1:]):
            if ok and np.isfinite(r):
                out[(str(slug), day)] = float(r)
    logger.info(f"  naive map: {len(out):,} (slug, date) returns over {voted['item_id'].nunique():,} items")
    return out


def run(archive_dir=None):
    archive_dir = Path(archive_dir) if archive_dir else DURABLE_ARCHIVE
    if not archive_dir.exists():
        raise SystemExit(f"durable archive not found at {archive_dir}")
    served = load_served()
    logger.info(f"  served h={HORIZON}: {len(served):,} rows, {served['forecast_date'].nunique()} dates")
    df = derive_panel(served)
    coverage = float(df["clean_known"].mean()) if len(df) else 0.0
    primary = select_primary(coverage)
    logger.info(f"  anchor_clean coverage: {coverage:.3f} -> primary {primary}")
    slugs = sorted(df["slug"].astype(str).unique())
    naive_map = load_naive_returns(slugs, df["forecast_date"].min(), df["forecast_date"].max(), archive_dir)
    df, n_dropped = attach_naive(df, naive_map)
    logger.info(f"  paired frame: {len(df):,} rows ({n_dropped:,} dropped without naive signal)")

    result = {
        "n_rows": len(df),
        "n_dropped_naive": n_dropped,
        "clean_coverage": round(coverage, 4),
        "primary": primary,
        "cohorts": {},
    }
    for cohort in ("coh_clean", "coh_wedge", "coh_exact"):
        sub = df[df[cohort]]
        share = len(sub) / len(df) if len(df) else 0.0
        dates = qualify(df, cohort)
        entry = {"rows": len(sub), "share": round(share, 4), "qualifying_dates": len(dates)}
        mic = summarize(df[df[cohort]], dates)
        nic = summarize(df[df[cohort]], dates, pred_col="naive")
        entry["model"] = bootstrap_ci(mic)
        entry["naive"] = bootstrap_ci(nic)
        entry["paired"] = paired_delta_ci(mic, nic)
        entry["per_date"] = {"model": mic, "naive": nic}
        result["cohorts"][cohort] = entry

    prim = result["cohorts"][primary]
    if prim["qualifying_dates"] < MIN_DATES or prim["share"] < MIN_COHORT_SHARE:
        result["verdict"] = f"VOID (primary {primary}: {prim['qualifying_dates']} dates, share {prim['share']:.3f})"
    else:
        result["verdict"] = evaluate_confirmation(prim["model"], prim["paired"])
    return result


def print_summary(result):
    print("\n" + "=" * 78)
    print(f"SERVED CENTRE RANK — h={HORIZON}, production r_hat vs realised (primary: {result['primary']})")
    print("=" * 78)
    for cohort, e in result["cohorts"].items():
        tag = "  [PRIMARY]" if cohort == result["primary"] else ""
        print(f"\n  {cohort}{tag} rows={e['rows']:,} share={e['share']:.3f} dates={e['qualifying_dates']}")
        for key in ("model", "naive", "paired"):
            c = e.get(key)
            if c is None:
                print(f"    {key:8s} n/a (<2 dates)")
                continue
            flag = "*" if c["ci_low"] > 0 else ("-" if c["ci_high"] < 0 else " ")
            print(
                f"    {key:8s} mean={c['mean']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]{flag} n={c['n_dates']}"
            )
    print(f"\n  verdict: {result['verdict']}")
    print("  * = CI entirely positive. CONFIRMED needs model>0 AND paired>0 on the primary cohort.")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = run(args.archive_dir)
    print_summary(result)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        logger.info(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
