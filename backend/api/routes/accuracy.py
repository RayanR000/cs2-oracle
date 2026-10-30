"""
Accuracy metrics API — serves backtest results and per-forecast outcomes.

Reads the ops store only (Phase 2 PR 6): ``ops/prediction_accuracy.parquet``, which the
backtest writes before Postgres, and the monthly ``forecast_outcomes`` partitions
(``db/ops_store.py``), not the old single-file mirror. No route falls back to Postgres: a
missing store raises.
"""

import json
import logging

import pandas as pd
from backtest.directional_test import PT_T_HURDLE
from backtest.scoring import HEADLINE_TIER, MIN_HEADLINE_DATES
from fastapi import APIRouter, Query

from api.serving_policy import MIN_SERVED_PRICE_USD

router = APIRouter(prefix="/accuracy", tags=["accuracy"])

logger = logging.getLogger(__name__)


def _json_safe(value):
    """Recursively replace non-JSON floats (NaN/Inf) so responses serialize."""
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _metrics_from_mirror(value):
    """The mirror's ``metrics`` cell as a dict.

    ``db.parquet`` stores nested values as JSON text — DuckDB's type inference
    over a column of Python dicts is data-dependent and could not be appended
    to, see the _jsonify_nested comment there. The HTTP contract is unchanged:
    callers have always received an object.

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


def _tier_clause(price_tier: int | None, column: str = "price_tier") -> str:
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


_PA_HAS_PRICE_TIER: bool | None = None


def _has_price_tier(q) -> bool:
    """Cached mirror-schema check — the DESCRIBE runs once per process, not per request."""
    global _PA_HAS_PRICE_TIER
    if _PA_HAS_PRICE_TIER is None:
        cols = set(q.query("DESCRIBE SELECT * FROM prediction_accuracy").iloc[:, 0].tolist())
        _PA_HAS_PRICE_TIER = "price_tier" in cols
    return _PA_HAS_PRICE_TIER


def _query_prediction_accuracy(
    prediction_type: str | None = None,
    limit: int = 200,
    price_tier: int | None = None,
) -> list[dict]:
    from db.parquet import ParquetQuery

    with ParquetQuery("prediction_accuracy") as q:
        _ = q.con  # raises FileNotFoundError when the store holds no file, rather than serving nothing
        has_tier = _has_price_tier(q)
        clauses = []
        if prediction_type:
            pt = prediction_type.replace("'", "''")
            clauses.append(f"prediction_type = '{pt}'")
        # A mirror written before migration 0019 has no price_tier column.
        # Such a file holds only all-tiers rows, so an unqualified request
        # is still correct; a request for a specific cohort has no rows.
        if has_tier:
            clauses.append(_tier_clause(price_tier))
        elif price_tier is not None:
            return []
        where = " AND ".join(clauses) or "1=1"
        df = q.query(f"SELECT * FROM prediction_accuracy WHERE {where} ORDER BY evaluation_date DESC LIMIT {limit}")
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
    prediction_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    price_tier: int | None = PRICE_TIER_QUERY,
):
    return _query_prediction_accuracy(prediction_type, limit, price_tier)


@router.get("/latest")
def get_latest_accuracy(
    prediction_type: str | None = Query(None),
    price_tier: int | None = PRICE_TIER_QUERY,
):
    """Get the most recent accuracy record for each prediction type."""
    latest = {}
    for r in _query_prediction_accuracy(prediction_type, 500, price_tier):
        key = r["prediction_type"]
        if key not in latest:
            latest[key] = r
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
        "directional_accuracy_ci_clustered_lower": m.get("directional_accuracy_ci_clustered_lower"),
        "directional_accuracy_ci_clustered_upper": m.get("directional_accuracy_ci_clustered_upper"),
        # Everything else the headline is not allowed to be read without.
        "distinct_forecast_dates": m.get("distinct_forecast_dates"),
        "date_coverage_sufficient": m.get("date_coverage_sufficient"),
        "unchanged_pct": m.get("unchanged_pct"),
        "interval_coverage": m.get("interval_coverage"),
    }


@router.get("/headline")
def get_headline():
    """The published accuracy claim, per horizon, as a significance test.

    This is the endpoint the product surfaces are meant to render. It exists
    because ``/accuracy/latest`` hands back a metrics blob from which a caller
    can pull ``directional_accuracy`` alone, and a bare DA on this data is not
    an accuracy claim — it is a report of which way the market moved on the
    dates the cohort happens to contain.

    Always the >=$1 cohort (HEADLINE_TIER), because that is the population the
    product actually serves; see MIN_SERVED_PRICE_USD in api/serving_policy.py.
    """
    rows = _query_prediction_accuracy("forecast", 2000, HEADLINE_TIER)

    # One entry per horizon, from that horizon's most recent evaluation. Rows
    # arrive newest-first, so the first sighting wins.
    latest: dict = {}
    for r in rows:
        key = r.get("horizon_days")
        if key is not None and key not in latest:
            latest[key] = r

    return _json_safe(
        {
            # Named from the serving floor, not from HEADLINE_TIER's index — the
            # two are held equal on purpose (see MIN_SERVED_PRICE_USD) and a label
            # derived from the tier number would read right while meaning nothing.
            "cohort": f">=${MIN_SERVED_PRICE_USD:.0f}",
            "price_tier": HEADLINE_TIER,
            "hurdle_t": PT_T_HURDLE,
            "min_forecast_dates": MIN_HEADLINE_DATES,
            "test": "Pesaran-Timmermann, per forecast date, Newey-West t over dates",
            "horizons": [_headline_entry(latest[h]) for h in sorted(latest)],
        }
    )


@router.get("/summary")
def get_accuracy_summary(
    prediction_type: str | None = Query(None),
    price_tier: int | None = PRICE_TIER_QUERY,
):
    """Returns aggregated summary across all available accuracy records."""
    rows = _query_prediction_accuracy(prediction_type, 2000, price_tier)

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


def _outcome_dict_from_row(r) -> dict:
    return _json_safe(
        {
            "id": int(getattr(r, "id", 0)),
            "forecast_id": int(getattr(r, "forecast_id", 0)),
            "item_id": int(getattr(r, "item_id", 0)),
            "forecast_date": str(getattr(r, "forecast_date", "")),
            "horizon_days": getattr(r, "horizon_days", None),
            "target_date": str(getattr(r, "target_date", "")),
            "current_price": float(getattr(r, "current_price", 0)),
            "predicted_price_mid": float(getattr(r, "predicted_price_mid", 0)),
            "actual_price": float(getattr(r, "actual_price", 0)),
            "direction_predicted": str(getattr(r, "direction_predicted", ""))
            if getattr(r, "direction_predicted", None)
            else None,
            "direction_actual": str(getattr(r, "direction_actual", ""))
            if getattr(r, "direction_actual", None)
            else None,
            "direction_correct": bool(getattr(r, "direction_correct", False)),
            "in_interval": bool(getattr(r, "in_interval", False))
            if getattr(r, "in_interval", None) is not None
            else None,
            "abs_error": float(getattr(r, "abs_error", 0)),
            "pct_error": float(getattr(r, "pct_error", 0)),
            "model_version": str(getattr(r, "model_version", "")) if getattr(r, "model_version", None) else None,
            "evaluated_at": str(getattr(r, "evaluated_at", "")),
        }
    )


def _outcomes_frame(sql: str) -> pd.DataFrame:
    """*sql* over the ``forecast_outcomes`` partitions as a DataFrame (NULLs as NaN, as the
    row shaping expects). Raises ``ops_store.MissingTable`` when no partition exists."""
    from db import ops_store

    con = ops_store.connect(tables=("forecast_outcomes",))
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def _query_outcomes(
    item_id: int | None = None,
    horizon_days: int | None = None,
    correct: bool | None = None,
    limit: int = 500,
) -> list[dict]:
    clauses = []
    if item_id is not None:
        clauses.append(f"item_id = {int(item_id)}")
    if horizon_days is not None:
        clauses.append(f"horizon_days = {int(horizon_days)}")
    if correct is not None:
        clauses.append(f"direction_correct = {1 if correct else 0}")
    where = " AND ".join(clauses) if clauses else "1=1"
    df = _outcomes_frame(f"SELECT * FROM forecast_outcomes WHERE {where} ORDER BY evaluated_at DESC LIMIT {int(limit)}")
    return [_json_safe(_outcome_dict_from_row(r)) for r in df.itertuples()]


@router.get("/outcomes")
def list_outcomes(
    item_id: int | None = Query(None),
    horizon_days: int | None = Query(None),
    correct: bool | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """Query individual forecast outcomes — was each prediction right or wrong?"""
    return _query_outcomes(item_id, horizon_days, correct, limit)


@router.get("/outcomes/stats")
def outcome_stats():
    """Aggregated stats from forecast outcomes — accuracy, error distribution."""
    totals = _outcomes_frame(
        "SELECT COUNT(*) AS total, SUM(direction_correct) AS correct, "
        "AVG(abs_error) AS avg_error, AVG(pct_error) AS avg_pct FROM forecast_outcomes"
    ).iloc[0]
    total = int(totals["total"] or 0)
    if total == 0:
        return {"total_outcomes": 0}
    correct = float(totals["correct"] or 0)
    ph_df = _outcomes_frame("""
        SELECT horizon_days, COUNT(*) AS total,
               SUM(direction_correct) AS correct,
               ROUND(AVG(abs_error), 4) AS avg_abs_error,
               ROUND(AVG(pct_error), 2) AS avg_pct_error
        FROM forecast_outcomes
        GROUP BY horizon_days
        ORDER BY horizon_days
    """)
    return {
        "total_outcomes": total,
        "overall_accuracy": round(correct / total * 100, 2),
        "mean_abs_error": round(float(totals["avg_error"] or 0), 4),
        "mean_pct_error": round(float(totals["avg_pct"] or 0), 2),
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
