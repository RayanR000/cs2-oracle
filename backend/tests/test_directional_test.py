"""Tests for the Pesaran-Timmermann headline.

The property under test throughout is the one that made DA unquotable: a model
whose calls track nothing but the market's own direction must NOT score as
skilful, however high its raw hit rate is.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest
from backtest.directional_test import (
    PT_MIN_ROWS_PER_DATE,
    PT_T_HURDLE,
    constant_call_baseline,
    hac_long_run_variance,
    newey_west_lag,
    pesaran_timmermann,
    realised_down_rate,
)
from backtest.scoring import MIN_FORECAST_DATES, score_cohort

BASE_DATE = date(2026, 1, 1)


def _rec(predicted, actual, forecast_date):
    return {
        "predicted_direction": predicted,
        "actual_direction": actual,
        "direction_correct": 1 if predicted == actual else 0,
        "forecast_date": forecast_date,
    }


def _table(forecast_date, a, b, c, d):
    """One date's 2x2 contingency table as records.

    a = (pred down, act down), b = (pred down, act up),
    c = (pred up,   act down), d = (pred up,   act up).
    Independence — the PT null — is exactly ``a * d == b * c``.
    """
    rows = [_rec("down", "down", forecast_date) for _ in range(a)]
    rows += [_rec("down", "up", forecast_date) for _ in range(b)]
    rows += [_rec("up", "down", forecast_date) for _ in range(c)]
    rows += [_rec("up", "up", forecast_date) for _ in range(d)]
    return rows


def _panel(table_for_date, n_dates=40):
    rows = []
    for i in range(n_dates):
        rows.extend(table_for_date(BASE_DATE + timedelta(days=i), i))
    return rows


# ---------------------------------------------------------------------------
# The reason this module exists
# ---------------------------------------------------------------------------


def test_a_constant_call_on_a_swinging_market_is_not_skill():
    """The headline case, from 2026-08-03-accuracy-is-clustered-by-forecast-date.

    An always-down model scored 29.4% on 2025-12-01 and 76.9% on 2026-07-17 at
    7d. Pooled over dates like those it can post any DA at all, and the number
    says nothing about the model. Under PT its excess is *identically* zero on
    every date, because when the call never varies the independence null equals
    the realised down-rate exactly.
    """
    rng = random.Random(3)
    down_rates = [rng.uniform(0.29, 0.77) for _ in range(40)]

    def table(forecast_date, i):
        n_down = round(200 * down_rates[i])
        return _table(forecast_date, a=n_down, b=200 - n_down, c=0, d=0)

    records = _panel(table)
    result = pesaran_timmermann(records, MIN_FORECAST_DATES)

    raw_da = sum(r["direction_correct"] for r in records) / len(records) * 100
    assert 45 < raw_da < 65, "the raw hit rate looks like a real result"

    assert result["pt_excess_pp"] == 0.0
    assert result["pt_verdict"] != "skill"
    assert result["pt_verdict"] == "degenerate"


def test_calls_independent_of_outcomes_do_not_clear_the_hurdle():
    """Marginals that look predictive, information that is not there.

    The model says "down" half the time into a market that falls 80% of the
    time, so it is right on plenty of rows — but its calls are drawn
    independently of the outcome, and PT sees through it.
    """
    rng = random.Random(7)

    def table(forecast_date, i):
        j = rng.randint(-4, 4)
        return _table(forecast_date, a=80 + j, b=20 - j, c=80 - j, d=20 + j)

    result = pesaran_timmermann(_panel(table), MIN_FORECAST_DATES)
    assert abs(result["pt_t_stat"]) < PT_T_HURDLE
    assert result["pt_verdict"] == "no_skill"


def test_genuine_within_date_skill_clears_the_hurdle():
    """Same marginals as the null case above, real association inside them."""
    rng = random.Random(11)

    def table(forecast_date, i):
        j = rng.randint(-4, 4)
        return _table(forecast_date, a=95 + j, b=5 - j, c=65 - j, d=35 + j)

    result = pesaran_timmermann(_panel(table), MIN_FORECAST_DATES)
    assert result["pt_excess_pp"] > 0
    assert result["pt_t_stat"] > PT_T_HURDLE
    assert result["pt_verdict"] == "skill"
    assert result["pt_p_value"] < 0.01


def test_anti_correlated_calls_are_reported_as_a_finding_not_a_null():
    """A significantly negative statistic gets its own verdict.

    Folding it into "no skill" would discard the strongest signal the test can
    produce: calls that are reliably wrong once the market effect is removed.
    """
    rng = random.Random(13)

    def table(forecast_date, i):
        j = rng.randint(-4, 4)
        return _table(forecast_date, a=65 + j, b=35 - j, c=95 - j, d=5 + j)

    result = pesaran_timmermann(_panel(table), MIN_FORECAST_DATES)
    assert result["pt_excess_pp"] < 0
    assert result["pt_t_stat"] < -PT_T_HURDLE
    assert result["pt_verdict"] == "perverse"


# ---------------------------------------------------------------------------
# Coverage gating
# ---------------------------------------------------------------------------


def test_a_clearing_t_stat_over_too_few_dates_is_still_not_a_headline():
    """The date count is the evidence, not the sample count."""
    rng = random.Random(11)

    def table(forecast_date, i):
        j = rng.randint(-4, 4)
        return _table(forecast_date, a=95 + j, b=5 - j, c=65 - j, d=35 + j)

    n_dates = MIN_FORECAST_DATES - 1
    result = pesaran_timmermann(_panel(table, n_dates=n_dates), MIN_FORECAST_DATES)

    assert result["pt_t_stat"] > PT_T_HURDLE
    assert result["pt_verdict"] == "insufficient_dates"
    assert result["pt_n_dates"] == n_dates


def test_dates_too_thin_to_estimate_the_null_are_dropped_and_counted():
    """At tiny n_d the per-date null is degenerate, so those dates are excluded.

    They are counted rather than silently skipped: a cohort whose dates are
    mostly too thin to test is a different situation from one with few dates,
    and the two must be distinguishable in the stored metrics.
    """
    rng = random.Random(11)

    def table(forecast_date, i):
        j = rng.randint(-4, 4)
        return _table(forecast_date, a=95 + j, b=5 - j, c=65 - j, d=35 + j)

    records = _panel(table, n_dates=25)
    thin = PT_MIN_ROWS_PER_DATE - 1
    records += _table(BASE_DATE + timedelta(days=900), a=thin, b=0, c=0, d=0)

    result = pesaran_timmermann(records, MIN_FORECAST_DATES)
    assert result["pt_n_dates"] == 25
    assert result["pt_n_dates_dropped"] == 1


def test_records_with_no_forecast_date_are_excluded_not_pooled():
    """Pooling undated rows into one pseudo-date would invent a market day."""
    rng = random.Random(11)

    def table(forecast_date, i):
        j = rng.randint(-4, 4)
        return _table(forecast_date, a=95 + j, b=5 - j, c=65 - j, d=35 + j)

    records = _panel(table, n_dates=22)
    undated = [_rec("down", "down", None) for _ in range(500)]

    with_undated = pesaran_timmermann(records + undated, MIN_FORECAST_DATES)
    without = pesaran_timmermann(records, MIN_FORECAST_DATES)
    assert with_undated == without


def test_a_single_date_yields_no_statistic():
    result = pesaran_timmermann(_table(BASE_DATE, a=95, b=5, c=65, d=35), MIN_FORECAST_DATES)
    assert result["pt_n_dates"] == 1
    assert result["pt_t_stat"] is None
    assert result["pt_excess_pp"] is None
    assert result["pt_verdict"] == "insufficient_dates"


def test_the_result_shape_is_constant_so_an_absent_statistic_is_visible():
    computed = pesaran_timmermann(_panel(lambda fd, i: _table(fd, 95, 5, 65, 35)), MIN_FORECAST_DATES)
    absent = pesaran_timmermann([], MIN_FORECAST_DATES)
    assert set(computed) == set(absent)
    assert absent["pt_verdict"] == "insufficient_dates"


def test_pesaran_timmermann_does_not_mutate_its_input():
    records = _panel(lambda fd, i: _table(fd, 95, 5, 65, 35))
    snapshot = [dict(r) for r in records]
    pesaran_timmermann(records, MIN_FORECAST_DATES)
    assert records == snapshot


# ---------------------------------------------------------------------------
# The estimator's pieces
# ---------------------------------------------------------------------------


def test_newey_west_bandwidth_matches_the_published_rule():
    assert newey_west_lag(MIN_FORECAST_DATES) == 2  # floor(4 * 0.2^(2/9))
    assert newey_west_lag(100) == 4
    assert newey_west_lag(1) == 0
    assert newey_west_lag(2) <= 1  # never exceeds T - 1


def test_zero_lag_variance_is_the_plain_sample_variance():
    values = [0.1, -0.2, 0.3, 0.05, -0.15]
    mean = sum(values) / len(values)
    expected = sum((v - mean) ** 2 for v in values) / len(values)
    assert hac_long_run_variance(values, 0) == pytest.approx(expected)


def test_positive_serial_correlation_widens_the_variance():
    """The whole reason for the HAC leg.

    Carry-forward prices make adjacent forecast dates near-duplicates. Under an
    i.i.d. variance that duplication reads as extra evidence; the Bartlett
    kernel prices it as the redundancy it is.
    """
    trending = [0.01 * i for i in range(30)]  # strongly autocorrelated
    assert hac_long_run_variance(trending, 3) > hac_long_run_variance(trending, 0)


# ---------------------------------------------------------------------------
# The triple that travels with DA
# ---------------------------------------------------------------------------


def test_constant_call_baseline_is_the_majority_class_not_flat():
    records = (
        [_rec("up", "down", BASE_DATE) for _ in range(60)]
        + [_rec("up", "up", BASE_DATE) for _ in range(30)]
        + [_rec("up", "flat", BASE_DATE) for _ in range(10)]
    )
    direction, accuracy = constant_call_baseline(records)
    assert direction == "down"
    assert accuracy == pytest.approx(60.0)
    assert realised_down_rate(records) == pytest.approx(60.0)


def test_constant_call_baseline_breaks_ties_deterministically():
    records = [_rec("up", "down", BASE_DATE) for _ in range(50)] + [_rec("up", "up", BASE_DATE) for _ in range(50)]
    first = constant_call_baseline(records)
    assert first == constant_call_baseline(list(reversed(records)))
    assert first[0] == "up"  # max() on (count, label): "up" > "down"


def test_constant_call_baseline_is_empty_rather_than_zero_on_no_records():
    assert constant_call_baseline([]) == (None, None)
    assert realised_down_rate([]) is None


# ---------------------------------------------------------------------------
# Wiring into the stored metrics
# ---------------------------------------------------------------------------


def _full_record(predicted, actual, forecast_date, price_tier=1):
    return {
        "abs_error": 0.10,
        "pct_error": 10.0,
        "sq_error": 0.01,
        "direction_correct": 1 if predicted == actual else 0,
        "predicted_direction": predicted,
        "actual_direction": actual,
        "in_interval": 1,
        "confidence": "high",
        "base_price": 1.00,
        "actual_price": 1.10,
        "price_tier": price_tier,
        "item_id": 1,
        "forecast_date": forecast_date,
    }


def test_score_cohort_publishes_the_test_and_the_triple_together():
    rng = random.Random(11)
    records = []
    for i in range(40):
        forecast_date = BASE_DATE + timedelta(days=i)
        j = rng.randint(-4, 4)
        for _ in range(95 + j):
            records.append(_full_record("down", "down", forecast_date))
        for _ in range(5 - j):
            records.append(_full_record("down", "up", forecast_date))
        for _ in range(65 - j):
            records.append(_full_record("up", "down", forecast_date))
        for _ in range(35 + j):
            records.append(_full_record("up", "up", forecast_date))

    metrics, n = score_cohort(records)

    assert metrics["pt_verdict"] == "skill"
    assert metrics["pt_t_stat"] > PT_T_HURDLE
    assert metrics["pt_hurdle_t"] == PT_T_HURDLE
    assert metrics["pt_n_dates"] == 40

    # DA never ships alone.
    assert metrics["constant_call_direction"] == "down"
    assert metrics["constant_call_accuracy"] == pytest.approx(80.0, abs=0.5)
    assert metrics["realised_down_rate"] == pytest.approx(80.0, abs=0.5)
    assert n == len(records)


def test_score_cohort_reports_untestable_rather_than_omitting_the_keys():
    """A cohort with no dates must still carry the pt_* fields.

    A stored row missing them is indistinguishable from one written before the
    test existed, and the two mean different things.
    """
    records = [_full_record("down", "down", None) for _ in range(50)]
    metrics, _ = score_cohort(records)
    assert metrics["pt_verdict"] == "insufficient_dates"
    assert metrics["pt_t_stat"] is None
    assert metrics["pt_n_dates"] == 0
