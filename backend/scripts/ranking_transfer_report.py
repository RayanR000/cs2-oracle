#!/usr/bin/env python3
"""Read-only ranking transfer report.

Asks whether the shadow LambdaRank score transfers to future served dates:
per-date Spearman rank correlation of `lambdarank_v1.score` against the
canonical realised return, paired against the production q50 centre return
on the same dates. Shadow-only throughout: even SUPPORTED leaves the score
undisclosed — public exposure is a separate future design.

Usage:
    venv/bin/python -m scripts.ranking_transfer_report --horizon {3,7,14,30,all} \
        --json-out PATH --markdown-out PATH

Exit codes: 0 SUPPORTED, 1 UNRESOLVED or INSUFFICIENT_EVIDENCE,
2 data-integrity or execution failure, 3 REJECTED.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib

from backtest.candidate_scoring import paired_daily_interval
from backtest.promotion import MIN_BATCH_COMPLETENESS, MIN_SHARED_DATES
from backtest.scoring import HEADLINE_MIN_TIER, excluded_forecast_date, price_tier
from database import SessionLocal
from models.centre_policy import HORIZONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ranking_transfer_report")

EXIT_SUPPORTED = 0
EXIT_UNRESOLVED = 1
EXIT_INTEGRITY = 2
EXIT_REJECTED = 3

SUPPORTED = "SUPPORTED"
REJECTED = "REJECTED"
UNRESOLVED = "UNRESOLVED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
DATA_INTEGRITY_FAILURE = "DATA_INTEGRITY_FAILURE"

MIN_ITEMS_PER_DATE = 100


def load_horizon_frame(db, horizon: int) -> dict:
    """Load shadow scores with canonical realised returns for one horizon.

    Joins ranking candidates to their frozen outcomes and the shared
    production outcome legs. Applies the >=$1 cohort and the excluded
    forecast dates here.
    """
    from sqlalchemy import text

    rows = db.execute(
        text("""
        SELECT c.item_id, c.forecast_date, c.score,
               c.candidate_version, c.config_fingerprint,
               co.base_price AS c_base, co.actual_price AS c_actual,
               co.resolved_at AS c_resolved,
               o.base_price AS p_base, o.actual_price AS p_actual,
               o.resolved_at AS p_resolved,
               o.predicted_price_mid AS p_mid
        FROM forecast_candidates c
        JOIN forecast_candidate_outcomes co ON co.candidate_id = c.id
        JOIN forecast_outcomes o
          ON o.item_id = c.item_id
         AND o.forecast_date = c.forecast_date
         AND o.horizon_days = c.horizon_days
        WHERE c.horizon_days = :h AND c.component = 'ranking'
        """),
        {"h": horizon},
    ).mappings().all()

    out = []
    for r in rows:
        fdate = r["forecast_date"]
        if excluded_forecast_date(fdate):
            continue
        if price_tier(float(r["p_base"])) < HEADLINE_MIN_TIER:
            continue
        out.append(
            {
                "item_id": r["item_id"],
                "forecast_date": fdate,
                "base_price": float(r["c_base"]),
                "actual_price": float(r["c_actual"]),
                "resolved_at": r["c_resolved"],
                "legs_match": (
                    r["c_base"] == r["p_base"]
                    and r["c_actual"] == r["p_actual"]
                    and str(r["c_resolved"]) == str(r["p_resolved"])
                ),
                "rank_score": float(r["score"]),
                "q50_ret": float(r["p_mid"]) / float(r["p_base"]) - 1.0,
                "actual_ret": float(r["c_actual"]) / float(r["c_base"]) - 1.0,
                "version": r["candidate_version"],
                "fingerprint": r["config_fingerprint"],
            }
        )

    first = db.execute(
        text("SELECT MIN(forecast_date) FROM forecast_candidates WHERE horizon_days = :h"),
        {"h": horizon},
    ).scalar()
    if first is None:
        expected, observed = 0, 0
    else:
        expected = db.execute(
            text("""
            SELECT COUNT(DISTINCT forecast_date) FROM item_forecasts
            WHERE horizon_days = :h AND forecast_date >= :first
            """),
            {"h": horizon, "first": first},
        ).scalar() or 0
        observed = db.execute(
            text("""
            SELECT COUNT(DISTINCT forecast_date) FROM forecast_candidates
            WHERE horizon_days = :h AND component = 'ranking'
            """),
            {"h": horizon},
        ).scalar() or 0
    return {"rows": out, "expected_batches": int(expected), "observed_batches": int(observed)}


def _spearman(x, y) -> float | None:
    from scipy.stats import spearmanr

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 2 or np.unique(xa).size < 2 or np.unique(ya).size < 2:
        return None
    ic = spearmanr(xa, ya).statistic
    return float(ic) if np.isfinite(ic) else None


def build_ranking_report(horizon: int, frame: dict) -> dict:
    """Pure report builder over a loaded frame. Performs no I/O."""
    rows = frame.get("rows", [])
    by_date: dict[str, list] = defaultdict(list)
    for r in rows:
        by_date[str(r["forecast_date"])].append(r)

    per_date, dropped = [], 0
    for key in sorted(by_date):
        group = by_date[key]
        if len(group) < MIN_ITEMS_PER_DATE:
            dropped += 1
            continue
        ic_rank = _spearman([g["rank_score"] for g in group], [g["actual_ret"] for g in group])
        ic_q50 = _spearman([g["q50_ret"] for g in group], [g["actual_ret"] for g in group])
        if ic_rank is None or ic_q50 is None:
            dropped += 1
            continue
        per_date.append({"date": key, "n": len(group), "ic_rank": ic_rank, "ic_q50": ic_q50})

    versions = {r["version"] for r in rows}
    fingerprints = {r["fingerprint"] for r in rows}
    completeness = (frame.get("observed_batches", 0) / frame["expected_batches"]) if frame.get("expected_batches") else 0.0
    resolution_ok = all(r.get("legs_match", True) for r in rows)

    if len(per_date) < MIN_SHARED_DATES:
        return _report(horizon, INSUFFICIENT_EVIDENCE, per_date, dropped, versions, fingerprints, completeness, frame,
                       paired=None, reasons=(f"only {len(per_date)} qualifying dates < {MIN_SHARED_DATES}",))
    integrity = []
    if len(versions) != 1:
        integrity.append(f"mixed candidate versions: {sorted(versions)}")
    if len(fingerprints) != 1:
        integrity.append(f"mixed config fingerprints: {len(fingerprints)} distinct")
    if completeness < MIN_BATCH_COMPLETENESS:
        integrity.append(f"batch completeness {completeness:.2f} < {MIN_BATCH_COMPLETENESS}")
    if not resolution_ok:
        integrity.append("candidate/production resolution mismatch")
    if integrity:
        return _report(horizon, DATA_INTEGRITY_FAILURE, per_date, dropped, versions, fingerprints, completeness, frame,
                       paired=None, reasons=tuple(integrity))

    paired = paired_daily_interval(
        [d["ic_rank"] for d in per_date], [d["ic_q50"] for d in per_date], [d["date"] for d in per_date]
    )
    if paired.lower is not None and paired.lower > 0:
        verdict, reasons = SUPPORTED, ()
    elif paired.upper is not None and paired.upper < 0:
        verdict, reasons = REJECTED, ("shadow rank underperforms the q50 ordering",)
    else:
        verdict, reasons = UNRESOLVED, ("transfer evidence does not establish either direction",)
    return _report(horizon, verdict, per_date, dropped, versions, fingerprints, completeness, frame,
                   paired=paired, reasons=reasons)


def _report(horizon, verdict, per_date, dropped, versions, fingerprints, completeness, frame, paired, reasons):
    return {
        "horizon": horizon,
        "verdict": verdict,
        "reasons": list(reasons),
        "paired_ic": (
            {"point": paired.point, "lower": paired.lower, "upper": paired.upper, "n_dates": paired.n_dates}
            if paired is not None
            else None
        ),
        "mean_ic_rank": (sum(d["ic_rank"] for d in per_date) / len(per_date)) if per_date else None,
        "mean_ic_q50": (sum(d["ic_q50"] for d in per_date) / len(per_date)) if per_date else None,
        "shared_dates": len(per_date),
        "min_shared_dates": MIN_SHARED_DATES,
        "dropped_dates": dropped,
        "completeness": completeness,
        "expected_batches": int(frame.get("expected_batches", 0)),
        "observed_batches": int(frame.get("observed_batches", 0)),
        "candidate_versions": sorted(versions),
        "candidate_fingerprints": sorted(fingerprints),
        "shadow_only": True,
        "per_date": per_date,
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Ranking transfer report — h={report['horizon']}",
        "",
        f"Verdict: {report['verdict']} (shadow-only; no API exposure)",
        f"Qualifying dates: {report['shared_dates']} (need {report['min_shared_dates']}); "
        f"dropped: {report['dropped_dates']}",
        f"Completeness: {report['completeness']:.2f}",
    ]
    if report["paired_ic"]:
        p = report["paired_ic"]
        lines.append(f"paired rank IC (lambdarank - q50): {p['point']:.4f} [90% CI {p['lower']}, {p['upper']}]")
    for reason in report["reasons"]:
        lines.append(f"- {reason}")
    lines += ["", "## Per-date IC", "", "| date | n | ic_rank | ic_q50 |", "|---|---|---|---|"]
    for row in report["per_date"]:
        lines.append(f"| {row['date']} | {row['n']} | {row['ic_rank']:.4f} | {row['ic_q50']:.4f} |")
    return "\n".join(lines) + "\n"


def _exit_code(verdict: str) -> int:
    if verdict == SUPPORTED:
        return EXIT_SUPPORTED
    if verdict == REJECTED:
        return EXIT_REJECTED
    if verdict == DATA_INTEGRITY_FAILURE:
        return EXIT_INTEGRITY
    return EXIT_UNRESOLVED


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only ranking transfer report")
    parser.add_argument("--horizon", required=True, choices=["3", "7", "14", "30", "all"])
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args(argv)

    horizons = list(HORIZONS) if args.horizon == "all" else [int(args.horizon)]
    db = SessionLocal()
    try:
        reports = {}
        for horizon in horizons:
            try:
                frame = load_horizon_frame(db, horizon)
                reports[str(horizon)] = build_ranking_report(horizon, frame)
            except Exception as e:
                logger.error("  h=%s report failed: %s", horizon, e)
                reports[str(horizon)] = {"horizon": horizon, "verdict": DATA_INTEGRITY_FAILURE, "reasons": [str(e)]}
    finally:
        with contextlib.suppress(Exception):
            db.close()

    payload = reports[str(horizons[0])] if len(horizons) == 1 else {"horizons": reports}
    with open(args.json_out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    if len(horizons) == 1:
        markdown = render_markdown(payload)
    else:
        markdown = "\n".join(
            render_markdown(r) if "per_date" in r else f"# h={h}\n\nError: {r.get('reasons')}" for h, r in reports.items()
        )
    with open(args.markdown_out, "w") as f:
        f.write(markdown)
    return max(_exit_code(r.get("verdict", DATA_INTEGRITY_FAILURE)) for r in reports.values())


if __name__ == "__main__":
    sys.exit(main())
