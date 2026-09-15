#!/usr/bin/env python3
"""
Walk-forward backtest report per horizon.

Evaluates forecast accuracy over multiple sliding time windows from the
forecast_outcomes table, producing per-horizon reports with accuracy
trends over time — not just a single recent validation window.

This makes accuracy claims trustworthy by showing whether performance
is stable, improving, or degrading across the full history of forecasts.

Usage:
    python scripts/backtest_walkforward_report.py
    python scripts/backtest_walkforward_report.py --window-days 30
    python scripts/backtest_walkforward_report.py --horizon 7 --window-days 60
"""

import json
import logging
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Archived 2026-09-15 (scripts/ -> scripts/archive/): one extra parent.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from database import PredictionAccuracy, SessionLocal
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_walkforward_report")


def _load_outcomes(db, horizon_days=None, min_date=None, max_date=None):
    """Load forecast outcomes, optionally filtered by horizon and date range."""
    q = """
        SELECT fo.id, fo.forecast_date, fo.horizon_days,
               fo.direction_correct, fo.in_interval,
               fo.abs_error, fo.pct_error,
               fo.model_version, fo.evaluated_at,
               fo.direction_predicted, fo.direction_actual
        FROM forecast_outcomes fo
        WHERE 1=1
    """
    params = {}
    if horizon_days is not None:
        q += " AND fo.horizon_days = :horizon_days"
        params["horizon_days"] = horizon_days
    if min_date is not None:
        q += " AND fo.forecast_date >= :min_date"
        params["min_date"] = min_date
    if max_date is not None:
        q += " AND fo.forecast_date <= :max_date"
        params["max_date"] = max_date
    q += " ORDER BY fo.forecast_date"

    return db.execute(text(q), params).fetchall()


def _compute_window_metrics(rows_in_window):
    """Compute accuracy metrics for a set of outcome rows."""
    n = len(rows_in_window)
    if n < 5:
        return None

    directional_hits = sum(1 for r in rows_in_window if r.direction_correct)
    interval_hits = sum(1 for r in rows_in_window if r.in_interval)
    interval_total = sum(1 for r in rows_in_window if r.in_interval is not None)

    mae = sum(r.abs_error for r in rows_in_window) / n
    sq_errors = sum(r.abs_error**2 for r in rows_in_window)
    rmse = math.sqrt(sq_errors / n)
    pct_errors = [r.pct_error for r in rows_in_window if r.pct_error is not None]
    mape = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0

    dir_acc = directional_hits / n * 100
    int_cov = interval_hits / interval_total * 100 if interval_total > 0 else 0.0

    return {
        "sample_count": n,
        "directional_accuracy": round(dir_acc, 2),
        "directional_hits": directional_hits,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "interval_coverage": round(int_cov, 2),
        "interval_hits": interval_hits,
        "interval_total": interval_total,
    }


def _upsert_accuracy(db, row):
    """Insert or replace a PredictionAccuracy record."""
    filters = {
        "prediction_type": row["prediction_type"],
        "evaluation_date": row["evaluation_date"],
    }
    if row.get("horizon_days") is not None:
        filters["horizon_days"] = row["horizon_days"]
    else:
        filters["horizon_days"] = None
    if row.get("model_version") is not None:
        filters["model_version"] = row["model_version"]
    else:
        filters["model_version"] = None

    existing = db.query(PredictionAccuracy).filter_by(**filters).first()
    if existing:
        existing.sample_count = row["sample_count"]
        existing.metrics = row["metrics"]
        existing.evaluation_window_days = row.get("evaluation_window_days")
        existing.created_at = row["created_at"]
    else:
        db.add(PredictionAccuracy(**row))
    db.commit()


def _detect_trend(values):
    """Classify a sequence as improving, degrading, or stable."""
    if len(values) < 2:
        return "stable"
    if len(values) == 2:
        diff = values[-1] - values[0]
        return "improving" if diff > 1 else ("degrading" if diff < -1 else "stable")
    # Linear regression slope approximation
    n = len(values)
    xs = np.arange(n)
    slope = (n * np.sum(xs * values) - np.sum(xs) * np.sum(values)) / (n * np.sum(xs**2) - np.sum(xs) ** 2)
    # Average absolute change per window step
    mean_abs_change = np.mean(np.abs(np.diff(values)))
    if mean_abs_change < 1.0:
        return "stable"
    return "improving" if slope > 0 else "degrading"


def build_sliding_windows(rows, window_days, overlap_pct=0.5):
    """Partition outcomes into overlapping sliding windows by forecast_date.

    Adapts window size to available date range: if data span is smaller than
    the requested window_days, uses a single window covering all data.
    Returns list of dicts with window_start, window_end, and the rows.
    """
    if not rows:
        return []

    dates = sorted(set(r.forecast_date for r in rows))
    min_date, max_date = dates[0], dates[-1]
    total_span = (max_date - min_date).days

    # Single-day data: one window with all rows
    if total_span == 0:
        return [
            {
                "window_start": min_date,
                "window_end": min_date + timedelta(days=1),
                "rows": rows,
            }
        ]

    # Shrink window if data span is too short for requested size
    effective_window = min(window_days, total_span + 1)

    step_days = max(1, int(effective_window * (1 - overlap_pct)))
    windows = []
    current_start = min_date

    while current_start < max_date:
        window_end = current_start + timedelta(days=effective_window)
        if window_end > max_date + timedelta(days=1):
            # Extend last window to cover remaining data
            window_end = max_date + timedelta(days=1)
        window_rows = [r for r in rows if current_start <= r.forecast_date < window_end]
        if window_rows:
            windows.append(
                {
                    "window_start": current_start,
                    "window_end": window_end,
                    "rows": window_rows,
                }
            )
        if window_end >= max_date + timedelta(days=1):
            break
        current_start += timedelta(days=step_days)

    return windows


def run_walkforward_report(window_days=30, horizon_filter=None):
    """Main entry: load outcomes, compute per-horizon walk-forward report.

    Args:
        window_days: Size of each evaluation window in days.
        horizon_filter: If set, only report for this horizon (e.g. 7).
    """
    db = SessionLocal()
    try:
        # Discover available horizons
        all_rows = _load_outcomes(db)
        if not all_rows:
            logger.warning("No forecast outcomes found. Run backtest_accuracy.py first.")
            return {"status": "empty", "message": "No forecast outcomes found in DB"}

        horizons = sorted(set(r.horizon_days for r in all_rows))
        if horizon_filter is not None:
            horizons = [h for h in horizons if h == horizon_filter]

        reports = {}
        for horizon in horizons:
            rows = _load_outcomes(db, horizon_days=horizon)
            if not rows or len(rows) < 10:
                logger.info(f"  [{horizon}d] Skipping — only {len(rows)} outcomes available")
                continue

            # Build sliding windows
            windows_data = build_sliding_windows(rows, window_days)
            if not windows_data:
                logger.info(f"  [{horizon}d] No complete windows (need >{window_days}d of data)")
                continue

            # Compute metrics for each window
            windows = []
            for w in windows_data:
                metrics = _compute_window_metrics(w["rows"])
                if metrics:
                    windows.append(
                        {
                            "window_start": w["window_start"].isoformat(),
                            "window_end": w["window_end"].isoformat(),
                            **metrics,
                        }
                    )

            if not windows:
                logger.info(f"  [{horizon}d] Skipping — no valid windows")
                continue

            # Summary across windows
            dir_accs = [w["directional_accuracy"] for w in windows]
            maes = [w["mae"] for w in windows]
            mapes = [w["mape"] for w in windows]
            int_covs = [w["interval_coverage"] for w in windows]
            counts = [w["sample_count"] for w in windows]

            baseline = 50.0
            mean_dir = float(np.mean(dir_accs))
            summary = {
                "directional_accuracy_mean": round(mean_dir, 2),
                "directional_accuracy_std": round(float(np.std(dir_accs)), 2),
                "directional_accuracy_min": round(float(min(dir_accs)), 2),
                "directional_accuracy_max": round(float(max(dir_accs)), 2),
                "directional_accuracy_trend": _detect_trend(np.array(dir_accs)),
                "improvement_over_baseline_mean": round(mean_dir - baseline, 1),
                "mae_mean": round(float(np.mean(maes)), 4),
                "mae_std": round(float(np.std(maes)), 4),
                "mape_mean": round(float(np.mean(mapes)), 2),
                "mape_std": round(float(np.std(mapes)), 2),
                "interval_coverage_mean": round(float(np.mean(int_covs)), 2),
                "interval_coverage_std": round(float(np.std(int_covs)), 2),
                "window_count": len(windows),
                "window_days": window_days,
                "total_outcomes": sum(counts),
                "improving_windows": sum(1 for i in range(1, len(dir_accs)) if dir_accs[i] > dir_accs[i - 1]),
                "degrading_windows": sum(1 for i in range(1, len(dir_accs)) if dir_accs[i] < dir_accs[i - 1]),
            }

            report = {
                "horizon_days": horizon,
                "date_range": {
                    "start": rows[0].forecast_date.isoformat(),
                    "end": rows[-1].forecast_date.isoformat(),
                },
                "total_outcomes": len(rows),
                "window_count": len(windows),
                "window_days": window_days,
                "per_window": windows,
                "summary": summary,
            }

            reports[horizon] = report

            # ── Console report ──────────────────────────────────────────
            logger.info(f"\n{'=' * 65}")
            logger.info(f"WALK-FORWARD BACKTEST REPORT — {horizon}d Horizon")
            logger.info(f"{'=' * 65}")
            logger.info(f"  Date range:        {report['date_range']['start']} → {report['date_range']['end']}")
            logger.info(f"  Total outcomes:    {report['total_outcomes']:,}")
            logger.info(f"  Windows:           {len(windows)} ({window_days}-day windows, {window_days // 2}-day step)")
            logger.info(f"  Effective baseline: {baseline:.0f}% (2-class)")
            logger.info("")
            logger.info("  ┌─────────────────────────── Summary ───────────────────────────┐")
            s = summary
            logger.info(
                f"  │ Directional Accuracy: {s['directional_accuracy_mean']:>6.1f}%  ± {s['directional_accuracy_std']:.1f}%  "
                f"[{s['directional_accuracy_min']:.1f}%, {s['directional_accuracy_max']:.1f}%] │"
            )
            logger.info(
                f"  │ Improvement vs 50%:  {s['improvement_over_baseline_mean']:>+6.1f}pp                              │"
            )
            logger.info(f"  │ Trend:               {s['directional_accuracy_trend']:>14}                          │")
            logger.info(
                f"  │ MAE:                 ${s['mae_mean']:>6.2f}  ± ${s['mae_std']:.2f}                         │"
            )
            logger.info(
                f"  │ MAPE:                {s['mape_mean']:>6.2f}%  ± {s['mape_std']:.2f}%                        │"
            )
            logger.info(
                f"  │ Interval Coverage:   {s['interval_coverage_mean']:>6.1f}%  ± {s['interval_coverage_std']:.1f}%                      │"
            )
            logger.info(
                f"  │ Improving windows:   {s['improving_windows']:>3} / {s['window_count']}                              │"
            )
            logger.info("  └───────────────────────────────────────────────────────────────┘")
            logger.info("")
            logger.info("  Per-Window Breakdown:")
            logger.info(f"    {'Window Range':<28} {'N':>7} {'DirAcc':>8} {'MAE':>9} {'MAPE':>7} {'IntCov':>8}")
            logger.info(f"    {'─' * 28} {'─' * 7} {'─' * 8} {'─' * 9} {'─' * 7} {'─' * 8}")
            for w in windows:
                logger.info(
                    f"    {w['window_start'][:10]}..{w['window_end'][:10]:<17}"
                    f" {w['sample_count']:>7,}"
                    f" {w['directional_accuracy']:>7.1f}%"
                    f" ${w['mae']:>6.2f}"
                    f" {w['mape']:>5.1f}%"
                    f" {w['interval_coverage']:>6.1f}%"
                )
            logger.info("")

        # ── Store to PredictionAccuracy ────────────────────────────────
        today = date.today()
        stored_count = 0
        for horizon, report in reports.items():
            # Per-window records for trend visualization
            for w in report["per_window"]:
                record = {
                    "prediction_type": "walkforward_report",
                    "evaluation_date": date.fromisoformat(w["window_end"]),
                    "horizon_days": horizon,
                    "model_version": None,
                    "evaluation_window_days": window_days,
                    "sample_count": w["sample_count"],
                    "metrics": {
                        "directional_accuracy": w["directional_accuracy"],
                        "mae": w["mae"],
                        "rmse": w.get("rmse", 0),
                        "mape": w["mape"],
                        "interval_coverage": w["interval_coverage"],
                        "window_start": w["window_start"],
                        "window_end": w["window_end"],
                        "directional_hits": w["directional_hits"],
                        "interval_hits": w.get("interval_hits", 0),
                        "interval_total": w.get("interval_total", 0),
                    },
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                }
                _upsert_accuracy(db, record)
                stored_count += 1

            # Summary record for quick API queries
            summary_record = {
                "prediction_type": "walkforward_summary",
                "evaluation_date": today,
                "horizon_days": horizon,
                "model_version": None,
                "evaluation_window_days": window_days,
                "sample_count": report["total_outcomes"],
                "metrics": report["summary"],
                "created_at": datetime.now(UTC).replace(tzinfo=None),
            }
            _upsert_accuracy(db, summary_record)
            stored_count += 1

        logger.info(f"Walk-forward reports stored: {stored_count} records across {len(reports)} horizons")

        return {
            "status": "success",
            "horizons": list(reports.keys()),
            "window_days": window_days,
            "records_stored": stored_count,
        }

    except Exception as e:
        logger.error(f"Walk-forward report failed: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "message": str(e)}

    finally:
        db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward backtest report per horizon")
    parser.add_argument(
        "--window-days", type=int, default=30, help="Size of each evaluation window in days (default: 30)"
    )
    parser.add_argument("--horizon", type=int, default=None, help="Only evaluate this horizon (default: all)")
    args = parser.parse_args()

    result = run_walkforward_report(
        window_days=args.window_days,
        horizon_filter=args.horizon,
    )
    print(f"RESULT: {json.dumps(result, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
