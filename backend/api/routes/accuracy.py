"""
Accuracy metrics API — serves backtest results and per-forecast outcomes.

Reads from Parquet (ops/prediction_accuracy.parquet, ops/forecast_outcomes.parquet)
with SQLAlchemy fallback.
"""

import json
from typing import Optional
import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, text
from datetime import date

from database import get_db, PredictionAccuracy, ForecastOutcome
from api.serving_policy import MIN_SERVED_PRICE_USD
from backtest.directional_test import PT_T_HURDLE
from backtest.scoring import HEADLINE_TIER, MIN_FORECAST_DATES

router = APIRouter(prefix="/accuracy", tags=["accuracy"])


def _json_safe(value):
    """Recursively replace non-JSON floats (NaN/Inf) so responses serialize."""
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _row_to_dict(row: PredictionAccuracy) -> dict:
    return _json_safe({
        "id": row.id,
        "prediction_type": row.prediction_type,
        "evaluation_date": row.evaluation_date.isoformat() if row.evaluation_date else None,
        "horizon_days": row.horizon_days,
        "model_version": row.model_version,
        "price_tier": row.price_tier,
        "evaluation_window_days": row.evaluation_window_days,
        "sample_count": row.sample_count,
        "metrics": row.metrics,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    })


def _dict_to_row(d: dict) -> dict:
    return {
        "id": d.get("id"),
        "prediction_type": d.get("prediction_type"),
        "evaluation_date": str(d.get("evaluation_date")) if d.get("evaluation_date") else None,
        "horizon_days": d.get("horizon_days"),
        "model_version": d.get("model_version"),
        "price_tier": d.get("price_tier"),
        "evaluation_window_days": d.get("evaluation_window_days"),
        "sample_count": d.get("sample_count"),
        "metrics": d.get("metrics"),
        "created_at": str(d.get("created_at")) if d.get("created_at") else None,
    }


def _metrics_from_mirror(value):
    """The mirror's ``metrics`` cell as a dict.

    ``db.parquet`` stores nested values as JSON text — DuckDB's type inference
    over a column of Python dicts is data-dependent and could not be appended
    to, see the _jsonify_nested comment there. The HTTP contract is unchanged:
    callers have always received an object, and the DB fallback path returns one
    from its JSON column, so the two legs must not disagree.

    Rows written before that change come back as a dict (or, from a file whose
    struct has not been rewritten yet, something dict-like), so both shapes are
    accepted rather than assuming the migration has run.
    """
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    # A DuckDB STRUCT read through pandas arrives as a mapping-like object.
    try:
        return dict(value)
    except (TypeError, ValueError):
        return None


def _tier_clause(price_tier: Optional[int], column: str = "price_tier") -> str:
    """SQL restricting to one price cohort.

    ``price_tier`` is a discriminator, not a filterable attribute: the table
    holds a row per price band (0..5), one per floor in ``FLOOR_SWEEP`` (-1 is
    the >=$1 headline, -2 is >=$5, -3 is >=$20), and one all-tiers aggregate
    (NULL). They overlap, so a query that does not pick exactly one is summing
    the same forecasts several times over. ``None`` means the all-tiers
    aggregate, which is the row the endpoints served before price_tier existed —
    it is pooled across liquidity populations and is not a quotable number.
    """
    if price_tier is None:
        return f"{column} IS NULL"
    return f"{column} = {int(price_tier)}"


def _query_prediction_accuracy(
    prediction_type: Optional[str] = None,
    limit: int = 200,
    price_tier: Optional[int] = None,
) -> Optional[list[dict]]:
    from db.parquet import ParquetQuery
    try:
        with ParquetQuery("prediction_accuracy") as q:
            cols = set(
                q.query("DESCRIBE SELECT * FROM prediction_accuracy")
                 .iloc[:, 0].tolist()
            )
            clauses = []
            if prediction_type:
                pt = prediction_type.replace("'", "''")
                clauses.append(f"prediction_type = '{pt}'")
            # A mirror written before migration 0019 has no price_tier column.
            # Such a file holds only all-tiers rows, so an unqualified request
            # is still correct; a request for a specific cohort is not, and
            # returning None falls the caller back to the DB.
            if "price_tier" in cols:
                clauses.append(_tier_clause(price_tier))
            elif price_tier is not None:
                return None
            where = " AND ".join(clauses) or "1=1"
            df = q.query(
                f"SELECT * FROM prediction_accuracy WHERE {where} ORDER BY evaluation_date DESC LIMIT {limit}"
            )
            if df.empty:
                return []
            result = []
            for r in df.itertuples():
                tier = getattr(r, "price_tier", None)
                d = {
                    "id": int(getattr(r, "id", 0)),
                    "prediction_type": str(getattr(r, "prediction_type", "")),
                    "evaluation_date": str(getattr(r, "evaluation_date", "")),
                    "horizon_days": getattr(r, "horizon_days", None),
                    "model_version": str(getattr(r, "model_version", "")) if getattr(r, "model_version", None) else None,
                    "price_tier": None if pd.isna(tier) else int(tier),
                    "evaluation_window_days": getattr(r, "evaluation_window_days", None),
                    "sample_count": int(getattr(r, "sample_count", 0)),
                    "metrics": _metrics_from_mirror(getattr(r, "metrics", None)) or {},
                    "created_at": str(getattr(r, "created_at", "")),
                }
                result.append(d)
            return [_json_safe(d) for d in result]
    except Exception:
        return None


PRICE_TIER_QUERY = Query(
    None,
    ge=-3,
    le=5,
    description=(
        "Price cohort: 0-5 for a single price band (5 is >=$1000, split out "
        "from tier 4 because its bid-ask spread is 5.2% against 10.8%), or a "
        "floor sentinel -1/-2/-3 for the >=$1 / >=$5 / >=$20 headline sweep. "
        "Omit for the all-tiers aggregate, which is pooled across liquidity "
        "populations and is not quotable. The cohorts overlap, so exactly one "
        "is served."
    ),
)


@router.get("/")
def list_accuracy(
    prediction_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    price_tier: Optional[int] = PRICE_TIER_QUERY,
    db: Session = Depends(get_db),
):
    try:
        rows = _query_prediction_accuracy(prediction_type, limit, price_tier)
        if rows is not None:
            return rows
    except Exception:
        pass

    q = db.query(PredictionAccuracy).order_by(desc(PredictionAccuracy.evaluation_date))
    if prediction_type:
        q = q.filter(PredictionAccuracy.prediction_type == prediction_type)
    q = q.filter(
        PredictionAccuracy.price_tier.is_(None) if price_tier is None
        else PredictionAccuracy.price_tier == price_tier
    )
    rows_st = q.limit(limit).all()
    return [_row_to_dict(r) for r in rows_st]


@router.get("/latest")
def get_latest_accuracy(
    prediction_type: Optional[str] = Query(None),
    price_tier: Optional[int] = PRICE_TIER_QUERY,
    db: Session = Depends(get_db),
):
    """Get the most recent accuracy record for each prediction type."""
    try:
        rows = _query_prediction_accuracy(prediction_type, 500, price_tier)
        if rows is not None:
            latest = {}
            for r in rows:
                key = r["prediction_type"]
                if key not in latest:
                    latest[key] = r
            if prediction_type:
                return latest.get(prediction_type)
            return latest
    except Exception:
        pass

    rows_st = db.query(PredictionAccuracy).filter(
        PredictionAccuracy.price_tier.is_(None) if price_tier is None
        else PredictionAccuracy.price_tier == price_tier
    ).order_by(
        PredictionAccuracy.prediction_type,
        desc(PredictionAccuracy.evaluation_date),
    ).all()

    latest = {}
    for r in rows_st:
        key = r.prediction_type
        if key not in latest:
            latest[key] = _row_to_dict(r)

    if prediction_type:
        return latest.get(prediction_type)

    return latest


# A row written before the Pesaran-Timmermann headline landed has no pt_* keys
# at all. Such a row is not "no skill" — it is untested, and the two must not
# render the same, so the verdict is explicit rather than defaulted.
_UNTESTED_VERDICT = "untested"


def _headline_entry(row: dict) -> dict:
    """One horizon's honest headline: the test first, the accuracy as context."""
    m = row.get("metrics") or {}
    verdict = m.get("pt_verdict", _UNTESTED_VERDICT)
    return {
        "horizon_days": row.get("horizon_days"),
        "model_version": row.get("model_version"),
        "evaluation_date": row.get("evaluation_date"),
        "sample_count": row.get("sample_count"),
        # The test.
        "verdict": verdict,
        "pt_excess_pp": m.get("pt_excess_pp"),
        "pt_t_stat": m.get("pt_t_stat"),
        "pt_p_value": m.get("pt_p_value"),
        "pt_nw_lag": m.get("pt_nw_lag"),
        "pt_n_dates": m.get("pt_n_dates"),
        "pt_n_dates_dropped": m.get("pt_n_dates_dropped"),
        # Context. `directional_accuracy` is deliberately shipped only inside
        # this triple: the realised down-rate swings 29.4% -> 76.9% between
        # stored forecast dates, so the same DA is skill on one date and
        # incompetence on another, and the constant call is what it must beat.
        "directional_accuracy": m.get("directional_accuracy"),
        "constant_call_accuracy": m.get("constant_call_accuracy"),
        "constant_call_direction": m.get("constant_call_direction"),
        "realised_down_rate": m.get("realised_down_rate"),
        "directional_accuracy_ci_clustered_lower": m.get(
            "directional_accuracy_ci_clustered_lower"
        ),
        "directional_accuracy_ci_clustered_upper": m.get(
            "directional_accuracy_ci_clustered_upper"
        ),
        # Everything else the headline is not allowed to be read without.
        "distinct_forecast_dates": m.get("distinct_forecast_dates"),
        "date_coverage_sufficient": m.get("date_coverage_sufficient"),
        "unchanged_pct": m.get("unchanged_pct"),
        "interval_coverage": m.get("interval_coverage"),
    }


@router.get("/headline")
def get_headline(db: Session = Depends(get_db)):
    """The published accuracy claim, per horizon, as a significance test.

    This is the endpoint the product surfaces are meant to render. It exists
    because ``/accuracy/latest`` hands back a metrics blob from which a caller
    can pull ``directional_accuracy`` alone, and a bare DA on this data is not
    an accuracy claim — it is a report of which way the market moved on the
    dates the cohort happens to contain.

    Always the >=$1 cohort (HEADLINE_TIER), because that is the population the
    product actually serves; see MIN_SERVED_PRICE_USD in api/serving_policy.py.
    """
    rows = None
    try:
        rows = _query_prediction_accuracy("forecast", 2000, HEADLINE_TIER)
    except Exception:
        rows = None

    if not rows:
        q = db.query(PredictionAccuracy).filter(
            PredictionAccuracy.prediction_type == "forecast",
            PredictionAccuracy.price_tier == HEADLINE_TIER,
        ).order_by(desc(PredictionAccuracy.evaluation_date))
        rows = [_row_to_dict(r) for r in q.limit(2000).all()]

    # One entry per horizon, from that horizon's most recent evaluation. Rows
    # arrive newest-first from both legs, so the first sighting wins.
    latest: dict = {}
    for r in rows:
        key = r.get("horizon_days")
        if key is not None and key not in latest:
            latest[key] = r

    return _json_safe({
        # Named from the serving floor, not from HEADLINE_TIER's index — the
        # two are held equal on purpose (see MIN_SERVED_PRICE_USD) and a label
        # derived from the tier number would read right while meaning nothing.
        "cohort": f">=${MIN_SERVED_PRICE_USD:.0f}",
        "price_tier": HEADLINE_TIER,
        "hurdle_t": PT_T_HURDLE,
        "min_forecast_dates": MIN_FORECAST_DATES,
        "test": "Pesaran-Timmermann, per forecast date, Newey-West t over dates",
        "horizons": [
            _headline_entry(latest[h]) for h in sorted(latest)
        ],
    })


@router.get("/summary")
def get_accuracy_summary(
    prediction_type: Optional[str] = Query(None),
    price_tier: Optional[int] = PRICE_TIER_QUERY,
    db: Session = Depends(get_db),
):
    """Returns aggregated summary across all available accuracy records."""
    try:
        rows = _query_prediction_accuracy(prediction_type, 2000, price_tier)
        if rows is None:
            rows = []
    except Exception:
        rows = []

    if not rows:
        q = db.query(PredictionAccuracy)
        if prediction_type:
            q = q.filter(PredictionAccuracy.prediction_type == prediction_type)
        q = q.filter(
            PredictionAccuracy.price_tier.is_(None) if price_tier is None
            else PredictionAccuracy.price_tier == price_tier
        )
        rows_st = q.order_by(PredictionAccuracy.evaluation_date).all()
        rows = [_row_to_dict(r) for r in rows_st]

    summary = {}
    for r in rows:
        key = f"{r['prediction_type']}"
        if r["prediction_type"] == "forecast" and r.get("horizon_days"):
            gh_key = f"{key}_{r['horizon_days']}d"
        elif r["prediction_type"] in ("trend_direction", "opportunity") and r.get("evaluation_window_days"):
            gh_key = f"{key}_{r['evaluation_window_days']}d"
        else:
            gh_key = key

        if gh_key not in summary:
            summary[gh_key] = {
                "prediction_type": r["prediction_type"],
                "horizon_days": r["horizon_days"],
                "evaluation_window_days": r.get("evaluation_window_days"),
                "records": [],
            }
        summary[gh_key]["records"].append(r)

    result = sorted(summary.values(), key=lambda x: x["prediction_type"])
    return result


# ---------------------------------------------------------------------------
# Per-forecast outcomes
# ---------------------------------------------------------------------------


def _outcome_to_dict(o: ForecastOutcome) -> dict:
    return _json_safe({
        "id": o.id,
        "forecast_id": o.forecast_id,
        "item_id": o.item_id,
        "forecast_date": o.forecast_date.isoformat() if o.forecast_date else None,
        "horizon_days": o.horizon_days,
        "target_date": o.target_date.isoformat() if o.target_date else None,
        "current_price": o.current_price,
        "predicted_price_mid": o.predicted_price_mid,
        "actual_price": o.actual_price,
        "direction_predicted": o.direction_predicted,
        "direction_actual": o.direction_actual,
        "direction_correct": bool(o.direction_correct),
        "in_interval": bool(o.in_interval) if o.in_interval is not None else None,
        "abs_error": o.abs_error,
        "pct_error": o.pct_error,
        "model_version": o.model_version,
        "evaluated_at": o.evaluated_at.isoformat() if o.evaluated_at else None,
    })


def _outcome_dict_from_row(r) -> dict:
    return _json_safe({
        "id": int(getattr(r, "id", 0)),
        "forecast_id": int(getattr(r, "forecast_id", 0)),
        "item_id": int(getattr(r, "item_id", 0)),
        "forecast_date": str(getattr(r, "forecast_date", "")),
        "horizon_days": getattr(r, "horizon_days", None),
        "target_date": str(getattr(r, "target_date", "")),
        "current_price": float(getattr(r, "current_price", 0)),
        "predicted_price_mid": float(getattr(r, "predicted_price_mid", 0)),
        "actual_price": float(getattr(r, "actual_price", 0)),
        "direction_predicted": str(getattr(r, "direction_predicted", "")) if getattr(r, "direction_predicted", None) else None,
        "direction_actual": str(getattr(r, "direction_actual", "")) if getattr(r, "direction_actual", None) else None,
        "direction_correct": bool(getattr(r, "direction_correct", False)),
        "in_interval": bool(getattr(r, "in_interval", False)) if getattr(r, "in_interval", None) is not None else None,
        "abs_error": float(getattr(r, "abs_error", 0)),
        "pct_error": float(getattr(r, "pct_error", 0)),
        "model_version": str(getattr(r, "model_version", "")) if getattr(r, "model_version", None) else None,
        "evaluated_at": str(getattr(r, "evaluated_at", "")),
    })


def _query_outcomes(
    item_id: Optional[int] = None,
    horizon_days: Optional[int] = None,
    correct: Optional[bool] = None,
    limit: int = 500,
) -> Optional[list[dict]]:
    from db.parquet import ParquetQuery
    try:
        with ParquetQuery("forecast_outcomes") as q:
            clauses = []
            if item_id is not None:
                clauses.append(f"item_id = {item_id}")
            if horizon_days is not None:
                clauses.append(f"horizon_days = {horizon_days}")
            if correct is not None:
                clauses.append(f"direction_correct = {1 if correct else 0}")
            where = " AND ".join(clauses) if clauses else "1=1"
            df = q.query(
                f"SELECT * FROM forecast_outcomes WHERE {where} ORDER BY evaluated_at DESC LIMIT {limit}"
            )
            if df.empty:
                return []
            return [_json_safe(_outcome_dict_from_row(r)) for r in df.itertuples()]
    except Exception:
        return None


@router.get("/outcomes")
def list_outcomes(
    item_id: Optional[int] = Query(None),
    horizon_days: Optional[int] = Query(None),
    correct: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Query individual forecast outcomes — was each prediction right or wrong?"""
    try:
        rows = _query_outcomes(item_id, horizon_days, correct, limit)
        if rows is not None:
            return rows
    except Exception:
        pass

    q = db.query(ForecastOutcome).order_by(desc(ForecastOutcome.evaluated_at))
    if item_id is not None:
        q = q.filter(ForecastOutcome.item_id == item_id)
    if horizon_days is not None:
        q = q.filter(ForecastOutcome.horizon_days == horizon_days)
    if correct is not None:
        q = q.filter(ForecastOutcome.direction_correct == (1 if correct else 0))
    rows_st = q.limit(limit).all()
    return [_outcome_to_dict(r) for r in rows_st]


@router.get("/outcomes/stats")
def outcome_stats(
    db: Session = Depends(get_db),
):
    """Aggregated stats from forecast outcomes — accuracy, error distribution."""
    try:
        from db.parquet import ParquetQuery
        with ParquetQuery("forecast_outcomes") as q:
            total = q.scalar("SELECT COUNT(*) FROM forecast_outcomes") or 0
            if total == 0:
                return {"total_outcomes": 0}
            correct = q.scalar("SELECT COUNT(*) FROM forecast_outcomes WHERE direction_correct = 1") or 0
            avg_error = q.scalar("SELECT AVG(abs_error) FROM forecast_outcomes") or 0
            avg_pct = q.scalar("SELECT AVG(pct_error) FROM forecast_outcomes") or 0
            ph_df = q.query("""
                SELECT horizon_days, COUNT(*) AS total,
                       SUM(direction_correct) AS correct,
                       ROUND(AVG(abs_error), 4) AS avg_abs_error,
                       ROUND(AVG(pct_error), 2) AS avg_pct_error
                FROM forecast_outcomes
                GROUP BY horizon_days
                ORDER BY horizon_days
            """)
            return {
                "total_outcomes": int(total),
                "overall_accuracy": round(correct / total * 100, 2) if total > 0 else 0,
                "mean_abs_error": round(float(avg_error), 4),
                "mean_pct_error": round(float(avg_pct), 2),
                "per_horizon": [
                    {
                        "horizon_days": int(r.horizon_days),
                        "total": int(r.total),
                        "correct": int(r.correct),
                        "accuracy": round(r.correct / r.total * 100, 2) if r.total > 0 else 0,
                        "avg_abs_error": float(r.avg_abs_error),
                        "avg_pct_error": float(r.avg_pct_error),
                    }
                    for r in ph_df.itertuples()
                ],
            }
    except Exception:
        pass

    total_st = db.query(func.count(ForecastOutcome.id)).scalar() or 0
    if total_st == 0:
        return {"total_outcomes": 0}

    correct_st = db.query(func.count(ForecastOutcome.id)).filter(
        ForecastOutcome.direction_correct == 1
    ).scalar() or 0

    avg_error_st = db.query(func.avg(ForecastOutcome.abs_error)).scalar() or 0
    avg_pct_st = db.query(func.avg(ForecastOutcome.pct_error)).scalar() or 0

    per_horizon_st = db.execute(text("""
        SELECT horizon_days,
               COUNT(*) AS total,
               SUM(direction_correct) AS correct,
               ROUND(AVG(abs_error), 4) AS avg_abs_error,
               ROUND(AVG(pct_error), 2) AS avg_pct_error
        FROM forecast_outcomes
        GROUP BY horizon_days
        ORDER BY horizon_days
    """)).fetchall()

    return {
        "total_outcomes": total_st,
        "overall_accuracy": round(correct_st / total_st * 100, 2) if total_st > 0 else 0,
        "mean_abs_error": round(avg_error_st, 4),
        "mean_pct_error": round(avg_pct_st, 2),
        "per_horizon": [
            {
                "horizon_days": r.horizon_days,
                "total": r.total,
                "correct": r.correct,
                "accuracy": round(r.correct / r.total * 100, 2) if r.total > 0 else 0,
                "avg_abs_error": r.avg_abs_error,
                "avg_pct_error": r.avg_pct_error,
            }
            for r in per_horizon_st
        ],
    }
