import inspect

from backtest import scoring
from models import served_recalibration


def test_threshold_values_are_separate():
    assert scoring.MIN_HEADLINE_DATES == 20
    assert scoring.MIN_FEEDBACK_DATES == 8
    assert not hasattr(scoring, "MIN_FORECAST_DATES")


def test_served_feedback_defaults_to_feedback_floor():
    sig = inspect.signature(served_recalibration.served_coverage_factors)
    assert sig.parameters["min_dates"].default == scoring.MIN_FEEDBACK_DATES


def test_scoring_uses_only_the_headline_floor():
    source = inspect.getsource(scoring.score_cohort)
    assert "MIN_HEADLINE_DATES" in source
    assert "MIN_FEEDBACK_DATES" not in source
