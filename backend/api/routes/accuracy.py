"""
Accuracy metrics API — serves backtest results for all prediction/analysis types.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import date

from database import get_db, PredictionAccuracy

router = APIRouter(prefix="/accuracy", tags=["accuracy"])


def _row_to_dict(row: PredictionAccuracy) -> dict:
    return {
        "id": row.id,
        "prediction_type": row.prediction_type,
        "evaluation_date": row.evaluation_date.isoformat() if row.evaluation_date else None,
        "horizon_days": row.horizon_days,
        "model_version": row.model_version,
        "evaluation_window_days": row.evaluation_window_days,
        "sample_count": row.sample_count,
        "metrics": row.metrics,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/")
def list_accuracy(
    prediction_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    q = db.query(PredictionAccuracy).order_by(desc(PredictionAccuracy.evaluation_date))
    if prediction_type:
        q = q.filter(PredictionAccuracy.prediction_type == prediction_type)
    rows = q.limit(limit).all()
    return [_row_to_dict(r) for r in rows]


@router.get("/latest")
def get_latest_accuracy(
    prediction_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Get the most recent accuracy record for each prediction type."""
    rows = db.query(PredictionAccuracy).order_by(
        PredictionAccuracy.prediction_type,
        desc(PredictionAccuracy.evaluation_date),
    ).all()

    latest = {}
    for r in rows:
        key = r.prediction_type
        if key not in latest:
            latest[key] = _row_to_dict(r)

    if prediction_type:
        return latest.get(prediction_type)

    return latest


@router.get("/summary")
def get_accuracy_summary(
    prediction_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Returns aggregated summary across all available accuracy records."""
    q = db.query(PredictionAccuracy)
    if prediction_type:
        q = q.filter(PredictionAccuracy.prediction_type == prediction_type)

    rows = q.order_by(PredictionAccuracy.evaluation_date).all()

    summary = {}
    for r in rows:
        key = f"{r.prediction_type}"

        # Group forecast by horizon also
        if r.prediction_type == "forecast" and r.horizon_days:
            gh_key = f"{key}_{r.horizon_days}d"
        elif r.prediction_type == "trend_direction" and r.evaluation_window_days:
            gh_key = f"{key}_{r.evaluation_window_days}d"
        elif r.prediction_type == "opportunity" and r.evaluation_window_days:
            gh_key = f"{key}_{r.evaluation_window_days}d"
        else:
            gh_key = key

        if gh_key not in summary:
            summary[gh_key] = {
                "prediction_type": r.prediction_type,
                "horizon_days": r.horizon_days,
                "evaluation_window_days": r.evaluation_window_days,
                "records": [],
            }
        summary[gh_key]["records"].append(_row_to_dict(r))

    # Return as list sorted by type
    result = sorted(summary.values(), key=lambda x: x["prediction_type"])
    return result
