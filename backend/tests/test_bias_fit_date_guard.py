"""The outcome-fitted bias loop must see date clustering, not just row counts.

update_bias_corrections_from_outcomes matches the predicted up/down/flat split
to the observed base rate. Every item sharing a forecast_date is exposed to the
same market-wide move, so 11,000 rows on two opposite-direction days describe
those two days, not the market. Row-count guards cannot detect this: the real
cohort is 11,000 rows and 2 dates.
"""
from __future__ import annotations

from datetime import date

from backtest.scoring import MIN_FORECAST_DATES
from models.forecaster import ItemForecaster


def _dates(n_distinct, rows_each=500):
    """rows_each rows on each of n_distinct consecutive days."""
    out = []
    for i in range(n_distinct):
        out.extend([date(2026, 1, 1 + i)] * rows_each)
    return out


def test_two_dates_lack_coverage_regardless_of_row_count():
    # The real cohort shape: five-figure row count, two market days.
    assert ItemForecaster._has_date_coverage(_dates(2, rows_each=5500)) is False


def test_coverage_is_met_at_the_minimum():
    assert ItemForecaster._has_date_coverage(_dates(MIN_FORECAST_DATES, 1)) is True


def test_one_below_the_minimum_lacks_coverage():
    assert ItemForecaster._has_date_coverage(_dates(MIN_FORECAST_DATES - 1, 1)) is False


def test_null_dates_never_count_toward_coverage():
    # Rows predating the forecast_date backfill must not manufacture coverage.
    dates = _dates(MIN_FORECAST_DATES, 1) + [None] * 5000
    assert ItemForecaster._has_date_coverage(dates) is True
    assert ItemForecaster._has_date_coverage([None] * 5000) is False


def test_empty_input_lacks_coverage():
    assert ItemForecaster._has_date_coverage([]) is False


def test_the_fit_and_the_headline_share_one_constant():
    """A second, drifting threshold is the failure this guards against.

    scoring.MIN_FORECAST_DATES gates what gets *reported*; the guard gates
    what gets *fitted*. If they ever diverge, production could fit
    thresholds on a cohort the same codebase refuses to quote.
    """
    from backtest import scoring
    from models import forecaster as fc

    assert fc.MIN_FORECAST_DATES is scoring.MIN_FORECAST_DATES
