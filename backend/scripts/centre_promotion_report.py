#!/usr/bin/env python3
"""Read-only centre promotion report.

Compares the last-price centre against the GBM q50 centre on shared served
dates with identical frozen outcome legs, and renders the manual-promotion
evidence (or its absence). Read-only: it never writes predictions, outcomes,
or champion configuration — promotion is a separate reviewed code change.

Usage:
    venv/bin/python -m scripts.centre_promotion_report --horizon {3,7,14,30,all} \
        --json-out PATH --markdown-out PATH

Exit codes: 0 every requested horizon PASS_FOR_MANUAL_PROMOTION,
1 UNRESOLVED or INSUFFICIENT_EVIDENCE, 2 data-integrity or execution
failure, 3 any requested horizon REJECTED.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib

from backtest.promotion import (
    DATA_INTEGRITY_FAILURE,
    INSUFFICIENT_EVIDENCE,
    MIN_SHARED_DATES,
    PASS_FOR_MANUAL_PROMOTION,
    REJECTED,
    UNRESOLVED,
    evaluate_centre_promotion,
)
from backtest.scoring import HEADLINE_MIN_TIER, excluded_forecast_date, price_tier
from database import SessionLocal
from models.centre_policy import HORIZONS, centre_champion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("centre_promotion_report")

EXIT_PASS = 0
EXIT_UNRESOLVED = 1
EXIT_INTEGRITY = 2
EXIT_REJECTED = 3


def load_horizon_frame(db, horizon: int) -> dict:
    """Load production and challenger rows for one horizon.

    Joins candidates/outcomes/item_forecasts/forecast_outcomes on exact
    item/date/horizon identity. Applies the >=$1 cohort and the excluded
    forecast dates here so the report never scores an unquotable row.
    """
    from sqlalchemy import text

    rows = db.execute(
        text("""
        SELECT c.item_id, c.forecast_date,
               c.candidate_name, c.candidate_version, c.config_fingerprint,
               c.centre_price AS c_mid, c.predicted_price_low AS c_low,
               c.predicted_price_high AS c_high,
               co.base_price AS c_base, co.actual_price AS c_actual,
               co.resolved_at AS c_resolved,
               o.base_price AS p_base, o.actual_price AS p_actual,
               o.resolved_at AS p_resolved,
               o.predicted_price_mid AS p_mid, o.predicted_price_low AS p_low,
               o.predicted_price_high AS p_high, o.in_interval AS p_in
        FROM forecast_candidates c
        JOIN forecast_candidate_outcomes co ON co.candidate_id = c.id
        JOIN forecast_outcomes o
          ON o.item_id = c.item_id
         AND o.forecast_date = c.forecast_date
         AND o.horizon_days = c.horizon_days
        WHERE c.horizon_days = :h AND c.component = 'centre'
        """),
        {"h": horizon},
    ).mappings().all()

    production, candidate = [], []
    seen_prod = set()
    for r in rows:
        fdate = r["forecast_date"]
        if excluded_forecast_date(fdate):
            continue
        if price_tier(float(r["p_base"])) < HEADLINE_MIN_TIER:
            continue
        key = (r["item_id"], str(fdate))
        if key not in seen_prod:
            seen_prod.add(key)
            production.append(
                {
                    "item_id": r["item_id"],
                    "forecast_date": fdate,
                    "base_price": float(r["p_base"]),
                    "actual_price": float(r["p_actual"]),
                    "resolved_at": r["p_resolved"],
                    "mid": float(r["p_mid"]),
                    "low": float(r["p_low"]) if r["p_low"] is not None else None,
                    "high": float(r["p_high"]) if r["p_high"] is not None else None,
                    "in_interval": r["p_in"],
                }
            )
        candidate.append(
            {
                "item_id": r["item_id"],
                "forecast_date": fdate,
                "candidate_name": r["candidate_name"],
                "base_price": float(r["c_base"]),
                "actual_price": float(r["c_actual"]),
                "resolved_at": r["c_resolved"],
                "centre_price": float(r["c_mid"]),
                "low": float(r["c_low"]),
                "high": float(r["c_high"]),
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
            WHERE horizon_days = :h AND component = 'centre'
            """),
            {"h": horizon},
        ).scalar() or 0
    return {
        "production": production,
        "candidate": candidate,
        "expected_batches": int(expected),
        "observed_batches": int(observed),
    }


def _centre_rows(rows, *, mid_key, low_key, high_key, cover_key=None):
    """Promotion row shape from absolute legs. Drops non-positive centres.

    When `cover_key` names a stored coverage verdict, it is preferred over
    recomputing from the legs: the stored predicate is what production
    scored, and recomputation must never disagree with it.
    """
    out = []
    for r in rows:
        mid = float(r[mid_key])
        if mid <= 0:
            continue
        low, high = float(r[low_key]), float(r[high_key])
        actual = float(r["actual_price"])
        stored = r.get(cover_key) if cover_key else None
        out.append(
            {
                "forecast_date": r["forecast_date"],
                "item_id": r["item_id"],
                "abs_error": abs(mid - actual),
                "in_interval": (bool(stored) if stored is not None else bool(low <= actual <= high)),
                "width_pct": (high - low) / mid * 100.0,
            }
        )
    return out


def build_centre_report(horizon: int, frame: dict) -> dict:
    """Pure report builder over a loaded frame. Performs no I/O."""
    champion = centre_champion(horizon)
    challenger = "last_price" if champion == "gbm_q50" else "gbm_q50"

    cand_rows = [r for r in frame.get("candidate", []) if r.get("candidate_name", challenger) == challenger]
    prod_by_key = {(str(r["forecast_date"]), r["item_id"]): r for r in frame.get("production", [])}

    resolution_ok = True
    for r in cand_rows:
        mate = prod_by_key.get((str(r["forecast_date"]), r["item_id"]))
        if mate is None:
            continue
        if not (
            mate["base_price"] == r["base_price"]
            and mate["actual_price"] == r["actual_price"]
            and str(mate["resolved_at"]) == str(r["resolved_at"])
        ):
            resolution_ok = False
            break

    if champion == "gbm_q50":
        rows_gbm = _centre_rows(
            frame.get("production", []), mid_key="mid", low_key="low", high_key="high", cover_key="in_interval"
        )
        rows_last = _centre_rows(cand_rows, mid_key="centre_price", low_key="low", high_key="high")
    else:
        rows_last = _centre_rows(
            frame.get("production", []), mid_key="mid", low_key="low", high_key="high", cover_key="in_interval"
        )
        rows_gbm = _centre_rows(cand_rows, mid_key="centre_price", low_key="low", high_key="high")

    result = evaluate_centre_promotion(
        horizon=horizon,
        rows_last=rows_last,
        rows_gbm=rows_gbm,
        candidate_versions={r["version"] for r in cand_rows},
        candidate_fingerprints={r["fingerprint"] for r in cand_rows},
        expected_batches=int(frame.get("expected_batches", 0)),
        observed_batches=int(frame.get("observed_batches", 0)),
        resolution_ok=resolution_ok,
    )

    per_date = _per_date_deltas(rows_last, rows_gbm)
    return {
        "horizon": horizon,
        "champion": champion,
        "challenger": challenger,
        "verdict": result.verdict,
        "reasons": list(result.reasons),
        "delta_mae": _interval_json(result.delta_mae),
        "delta_coverage": _interval_json(result.delta_coverage),
        "width_delta_pp": result.width_delta_pp,
        "median_width_last_pp": result.median_width_last_pp,
        "median_width_gbm_pp": result.median_width_gbm_pp,
        "shared_dates": result.shared_dates,
        "min_shared_dates": MIN_SHARED_DATES,
        "n_rows_last": len(rows_last),
        "n_rows_gbm": len(rows_gbm),
        "completeness": result.completeness,
        "expected_batches": int(frame.get("expected_batches", 0)),
        "observed_batches": int(frame.get("observed_batches", 0)),
        "candidate_versions": sorted({r["version"] for r in cand_rows}),
        "candidate_fingerprints": sorted({r["fingerprint"] for r in cand_rows}),
        "resolution_ok": resolution_ok,
        "per_date": per_date,
    }


def _interval_json(interval) -> dict | None:
    if interval is None:
        return None
    return {"point": interval.point, "lower": interval.lower, "upper": interval.upper, "n_dates": interval.n_dates}


def _per_date_deltas(rows_last, rows_gbm) -> list[dict]:
    from collections import defaultdict

    last_by_date: dict[str, list[float]] = defaultdict(list)
    gbm_by_date: dict[str, list[float]] = defaultdict(list)
    for r in rows_last:
        last_by_date[str(r["forecast_date"])].append(float(r["abs_error"]))
    for r in rows_gbm:
        gbm_by_date[str(r["forecast_date"])].append(float(r["abs_error"]))
    out = []
    for key in sorted(set(last_by_date) & set(gbm_by_date)):
        mae_last = sum(last_by_date[key]) / len(last_by_date[key])
        mae_gbm = sum(gbm_by_date[key]) / len(gbm_by_date[key])
        out.append({"date": key, "mae_last": mae_last, "mae_gbm": mae_gbm, "delta": mae_last - mae_gbm})
    return out


def render_markdown(report: dict) -> str:
    lines = [
        f"# Centre promotion report — h={report['horizon']}",
        "",
        f"Verdict: {report['verdict']}",
        f"Champion: {report['champion']} · Challenger: {report['challenger']}",
        f"Shared dates: {report['shared_dates']} (need {report['min_shared_dates']}); "
        f"rows last/gbm: {report['n_rows_last']}/{report['n_rows_gbm']}",
        f"Completeness: {report['completeness']:.2f} "
        f"({report['observed_batches']}/{report['expected_batches']} batches)",
    ]
    if report["delta_mae"]:
        m = report["delta_mae"]
        lines.append(f"delta_mae (last - gbm): {m['point']:.4f} [90% CI {m['lower']}, {m['upper']}] over {m['n_dates']} dates")
    if report["delta_coverage"]:
        c = report["delta_coverage"]
        lines.append(f"delta_coverage (last - gbm): {c['point']:.4f} [90% CI {c['lower']}, {c['upper']}]")
    lines.append(f"width_delta_pp: {report['width_delta_pp']}")
    for reason in report["reasons"]:
        lines.append(f"- {reason}")
    lines += ["", "## Per-date deltas", "", "| date | mae_last | mae_gbm | delta |", "|---|---|---|---|"]
    for row in report["per_date"]:
        lines.append(f"| {row['date']} | {row['mae_last']:.4f} | {row['mae_gbm']:.4f} | {row['delta']:+.4f} |")
    return "\n".join(lines) + "\n"


def _exit_code(verdict: str) -> int:
    if verdict == PASS_FOR_MANUAL_PROMOTION:
        return EXIT_PASS
    if verdict == REJECTED:
        return EXIT_REJECTED
    if verdict in (DATA_INTEGRITY_FAILURE,):
        return EXIT_INTEGRITY
    if verdict in (INSUFFICIENT_EVIDENCE, UNRESOLVED):
        return EXIT_UNRESOLVED
    return EXIT_INTEGRITY


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only centre promotion report")
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
                reports[str(horizon)] = build_centre_report(horizon, frame)
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
        markdown = "\n".join(render_markdown(r) if "challenger" in r else f"# h={h}\n\nError: {r.get('reasons')}" for h, r in reports.items())
    with open(args.markdown_out, "w") as f:
        f.write(markdown)
    return max(_exit_code(r.get("verdict", DATA_INTEGRITY_FAILURE)) for r in reports.values())


if __name__ == "__main__":
    sys.exit(main())
