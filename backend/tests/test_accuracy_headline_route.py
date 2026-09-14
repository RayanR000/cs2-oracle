"""Tests for GET /accuracy/headline.

The endpoint exists so a product surface cannot render a bare hit rate. What is
worth testing is therefore the shaping, not the plumbing: that the triple always
travels with the verdict, that a row written before the test existed is not
reported as a null result, and that each horizon is served from its own latest
evaluation rather than whichever row happened to sort first.
"""

from __future__ import annotations

import pytest
from api.routes import accuracy as route
from backtest.directional_test import PT_T_HURDLE
from backtest.scoring import HEADLINE_TIER, MIN_FORECAST_DATES


def _row(horizon, evaluation_date, **metric_overrides):
    metrics = {
        "pt_verdict": "no_skill",
        "pt_excess_pp": 0.12,
        "pt_t_stat": 0.9,
        "pt_p_value": 0.3727,
        "pt_nw_lag": 2,
        "pt_n_dates": 24,
        "pt_n_dates_dropped": 1,
        "directional_accuracy": 49.6,
        "constant_call_accuracy": 51.7,
        "constant_call_direction": "down",
        "realised_down_rate": 51.7,
        "directional_accuracy_ci_clustered_lower": 46.1,
        "directional_accuracy_ci_clustered_upper": 53.0,
        "distinct_forecast_dates": 24,
        "date_coverage_sufficient": True,
        "unchanged_pct": 31.2,
        "interval_coverage": 72.4,
    }
    metrics.update(metric_overrides)
    return {
        "horizon_days": horizon,
        "model_version": "v9",
        "evaluation_date": evaluation_date,
        "sample_count": 12345,
        "metrics": metrics,
    }


def _patch_rows(monkeypatch, rows):
    def fake(prediction_type=None, limit=200, price_tier=None):
        assert prediction_type == "forecast"
        assert price_tier == HEADLINE_TIER, "the headline is always the served cohort"
        return rows

    monkeypatch.setattr(route, "_query_prediction_accuracy", fake)


def test_the_verdict_and_the_triple_are_served_together(monkeypatch):
    """DA must never be reachable without the two numbers that make it readable."""
    _patch_rows(monkeypatch, [_row(7, "2026-08-07")])
    payload = route.get_headline(db=None)

    assert payload["price_tier"] == HEADLINE_TIER
    assert payload["hurdle_t"] == PT_T_HURDLE
    assert payload["min_forecast_dates"] == MIN_FORECAST_DATES

    entry = payload["horizons"][0]
    assert entry["verdict"] == "no_skill"
    assert entry["directional_accuracy"] == 49.6
    assert entry["constant_call_accuracy"] == 51.7
    assert entry["constant_call_direction"] == "down"
    assert entry["realised_down_rate"] == 51.7
    assert entry["pt_t_stat"] == 0.9
    assert entry["pt_n_dates_dropped"] == 1


def test_a_row_predating_the_test_reads_untested_not_no_skill(monkeypatch):
    """ "We never ran the test" and "the test came back null" are different claims.

    Defaulting the first to the second would let the stored series quietly
    report a measurement nobody made.
    """
    _patch_rows(monkeypatch, [{"horizon_days": 7, "metrics": {"directional_accuracy": 61.8}}])
    entry = route.get_headline(db=None)["horizons"][0]

    assert entry["verdict"] == "untested"
    assert entry["pt_t_stat"] is None
    assert entry["directional_accuracy"] == 61.8


def test_each_horizon_is_served_from_its_own_latest_evaluation(monkeypatch):
    """Rows arrive newest-first, so the first sighting of a horizon wins."""
    _patch_rows(
        monkeypatch,
        [
            _row(7, "2026-08-07", pt_t_stat=4.1, pt_verdict="skill"),
            _row(30, "2026-08-07", pt_t_stat=-0.2),
            _row(7, "2026-07-01", pt_t_stat=9.9, pt_verdict="skill"),
        ],
    )
    horizons = route.get_headline(db=None)["horizons"]

    assert [h["horizon_days"] for h in horizons] == [7, 30]
    assert horizons[0]["pt_t_stat"] == 4.1
    assert horizons[0]["evaluation_date"] == "2026-08-07"


def test_the_cohort_label_names_the_serving_floor(monkeypatch):
    from api.serving_policy import MIN_SERVED_PRICE_USD

    _patch_rows(monkeypatch, [_row(7, "2026-08-07")])
    assert route.get_headline(db=None)["cohort"] == f">=${MIN_SERVED_PRICE_USD:.0f}"


def test_non_json_floats_do_not_break_the_response(monkeypatch):
    """NaN reaches the mirror through DuckDB and is not valid JSON."""
    _patch_rows(monkeypatch, [_row(7, "2026-08-07", pt_t_stat=float("nan"))])
    assert route.get_headline(db=None)["horizons"][0]["pt_t_stat"] is None


@pytest.mark.parametrize("rows", [[], None])
def test_no_parquet_rows_falls_through_to_the_database(monkeypatch, rows):
    """Both "empty" and "could not read" must reach the SQLAlchemy leg.

    An empty list from the mirror is not evidence that there is no headline —
    a mirror written before price_tier existed holds no HEADLINE_TIER row at
    all while the DB still does.
    """
    monkeypatch.setattr(route, "_query_prediction_accuracy", lambda *a, **k: rows)
    calls = []

    class _Q:
        def filter(self, *a):
            calls.append("filter")
            return self

        def order_by(self, *a):
            return self

        def limit(self, *a):
            return self

        def all(self):
            return []

    class _DB:
        def query(self, *a):
            calls.append("query")
            return _Q()

    assert route.get_headline(db=_DB())["horizons"] == []
    assert "query" in calls


def test_price_tier_query_admits_every_cohort_score_by_tier_emits():
    """A bound narrower than the emitted sentinels makes a stored row
    unreachable through the API — a silent 422 on a row that exists.

    FastAPI keeps the bounds as annotated-types constraints in `metadata`, not
    as `.ge` / `.le` attributes, so they are read out by type.
    """
    from annotated_types import Ge, Le
    from api.routes.accuracy import PRICE_TIER_QUERY
    from backtest.scoring import FLOOR_SWEEP, price_tier

    lower = next(c.ge for c in PRICE_TIER_QUERY.metadata if isinstance(c, Ge))
    upper = next(c.le for c in PRICE_TIER_QUERY.metadata if isinstance(c, Le))
    assert lower == min(FLOOR_SWEEP)  # -3
    assert upper == price_tier(50_000)  # 5
