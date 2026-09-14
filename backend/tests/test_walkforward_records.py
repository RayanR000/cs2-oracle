"""Record construction for the walkforward gate.

score_cohort consumes per-record dicts, not per-fold aggregates. This module
owns the two conversions that are easy to get silently wrong: percent returns
into direction_from_return's fractions, and the classifier's integer classes
into direction labels.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from backtest.walkforward_records import CLASS_TO_DIRECTION, fold_records


def _kwargs(**overrides):
    base = dict(
        item_ids=np.array(["ak47"]),
        forecast_dates=np.array([date(2026, 1, 5)], dtype=object),
        base_prices=np.array([100.0]),
        actual_returns_pct=np.array([2.0]),
        mid_returns_pct=np.array([1.0]),
        low_returns_pct=np.array([-5.0]),
        high_returns_pct=np.array([5.0]),
        predicted_classes=np.array([2]),
    )
    base.update(overrides)
    return base


def test_class_to_direction_mapping_is_down_flat_up():
    # forecaster._direction_classes: 0=down, 1=flat, 2=up.
    assert CLASS_TO_DIRECTION == {0: "down", 1: "flat", 2: "up"}


def test_returns_are_converted_from_percent_to_fraction():
    # +0.3% is inside the 0.5% flat band. If the percent value were passed
    # straight to direction_from_return (fraction, tolerance 0.005) it would
    # read as +30% and label "up".
    rec = fold_records(**_kwargs(actual_returns_pct=np.array([0.3])))[0]
    assert rec["actual_direction"] == "flat"


def test_two_percent_return_is_up():
    rec = fold_records(**_kwargs(actual_returns_pct=np.array([2.0])))[0]
    assert rec["actual_direction"] == "up"


def test_direction_correct_compares_classifier_call_to_actual():
    up = fold_records(**_kwargs(predicted_classes=np.array([2])))[0]
    down = fold_records(**_kwargs(predicted_classes=np.array([0])))[0]
    assert up["direction_correct"] == 1
    assert down["direction_correct"] == 0


def test_prices_are_reconstructed_from_base_and_return():
    rec = fold_records(**_kwargs())[0]
    assert rec["base_price"] == pytest.approx(100.0)
    assert rec["actual_price"] == pytest.approx(102.0)
    assert rec["abs_error"] == pytest.approx(1.0)  # mid 101 vs actual 102
    assert rec["sq_error"] == pytest.approx(1.0)
    assert rec["pct_error"] == pytest.approx(1.0)  # divided by BASE, not actual


def test_in_interval_uses_price_space_band():
    inside = fold_records(**_kwargs())[0]
    assert inside["in_interval"] == 1
    outside = fold_records(**_kwargs(high_returns_pct=np.array([1.5])))[0]
    assert outside["in_interval"] == 0


def test_forecast_date_is_carried_through_as_the_cluster_key():
    rec = fold_records(**_kwargs())[0]
    assert rec["forecast_date"] == date(2026, 1, 5)


def test_confidence_is_uniform_low():
    # The harness has no confidence estimator; production's comes from
    # _calibrate_confidence. Uniform "low" makes score_cohort's conf_gap_pp
    # and conf_high_interval_cov structurally zero, which is why the results
    # write-up must not read them for harness arms.
    rec = fold_records(**_kwargs())[0]
    assert rec["confidence"] == "low"


def test_fold_records_carry_the_prediction_leg_and_an_optional_horizon():
    """The walkforward gate scores through score_cohort, so its records need the
    same two fields — otherwise the gate silently reports out_of_scope forever."""
    rec = fold_records(**_kwargs(), horizon_days=14)[0]
    assert rec["predicted_mid"] == pytest.approx(101.0)  # 100 * (1 + 1.0/100)
    assert rec["horizon_days"] == 14


def test_fold_records_horizon_defaults_to_none_for_existing_callers():
    """A caller that has not said which horizon it is measuring must score as
    out_of_scope, not have one guessed for it."""
    rec = fold_records(**_kwargs())[0]
    assert rec["horizon_days"] is None


def test_rejects_mismatched_array_lengths():
    with pytest.raises(ValueError, match="equal length"):
        fold_records(**_kwargs(mid_returns_pct=np.array([1.0, 2.0])))
