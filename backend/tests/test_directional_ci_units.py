"""The directional-accuracy CI must be in the same units as the accuracy.

``score_cohort`` reported ``directional_accuracy`` as a PERCENT (0-100) while
the four interval bounds derived from the same 0/1 indicators came back as
FRACTIONS (0-1), because ``bootstrap_ci`` / ``block_bootstrap_ci`` average the
raw indicators and nothing rescaled them. The 2026-08-05 21:30 failure dump
showed both in one dict: ``'directional_accuracy': 49.57`` next to
``'directional_accuracy_ci_lower': 0.4854``.

Stored side by side under names that differ only by suffix, that is a reading
trap — a 48.5-51.6 interval around 49.57 looks like a 0.49-0.52 interval that
excludes the point estimate by 49 points. ``_score_groups`` compensated with a
``* 100`` in the log line only, so the console was right and the stored row was
not, and the stored row is what gets audited.
"""

from __future__ import annotations

from datetime import date, timedelta

from backtest.scoring import score_cohort

CI_FIELDS = (
    "directional_accuracy_ci_lower",
    "directional_accuracy_ci_upper",
    "directional_accuracy_ci_clustered_lower",
    "directional_accuracy_ci_clustered_upper",
)


def _record(correct, forecast_date, base=5.0):
    return {
        "abs_error": 0.1,
        "pct_error": 2.0,
        "sq_error": 0.01,
        "direction_correct": correct,
        "predicted_direction": "up",
        "actual_direction": "up" if correct else "down",
        "in_interval": 1,
        "confidence": "high",
        "base_price": base,
        "actual_price": base + 0.1,
        "price_tier": 1,
        "item_id": 1,
        "forecast_date": forecast_date,
    }


def _mixed_cohort(n_dates=6, per_date=10):
    d0 = date(2026, 7, 1)
    records = []
    for i in range(n_dates):
        d = d0 + timedelta(days=i)
        # Alternate so accuracy is near 50% and the interval is wide.
        records += [_record(j % 2, d) for j in range(per_date)]
    return records


def test_the_interval_brackets_the_point_estimate():
    """The one property that makes the units bug impossible to reintroduce."""
    metrics, n = score_cohort(_mixed_cohort())
    assert n > 0
    da = metrics["directional_accuracy"]

    for lo_key, hi_key in (
        ("directional_accuracy_ci_lower", "directional_accuracy_ci_upper"),
        ("directional_accuracy_ci_clustered_lower", "directional_accuracy_ci_clustered_upper"),
    ):
        lo, hi = metrics[lo_key], metrics[hi_key]
        assert lo is not None and hi is not None
        assert lo <= da <= hi, (
            f"{lo_key}..{hi_key} = {lo}..{hi} does not bracket directional_accuracy = {da} — units disagree"
        )


def test_the_bounds_are_on_a_percent_scale():
    metrics, _ = score_cohort(_mixed_cohort())
    for field in CI_FIELDS:
        value = metrics[field]
        assert value is not None
        assert 1.0 < value <= 100.0, f"{field} = {value} looks like a fraction, not a percent"


def test_an_all_correct_cohort_reports_a_hundred_not_one():
    """The sharpest discriminator: a fraction bound would be exactly 1.0."""
    d0 = date(2026, 7, 1)
    records = [_record(1, d0 + timedelta(days=i)) for i in range(6) for _ in range(10)]
    metrics, _ = score_cohort(records)
    assert metrics["directional_accuracy"] == 100.0
    for field in CI_FIELDS:
        assert metrics[field] == 100.0, f"{field} = {metrics[field]}"


def test_mae_bounds_are_left_in_dollars():
    """mae_ci_* were always in the same units as mae; do not rescale them."""
    metrics, _ = score_cohort(_mixed_cohort())
    assert metrics["mae_ci_lower"] <= metrics["mae"] <= metrics["mae_ci_upper"]
    assert metrics["mae_ci_upper"] < 1.0, "mae bounds were scaled like a percent"


def test_an_empty_interval_is_still_none():
    """Too few records to bootstrap must stay None, not become 0.0."""
    metrics, _ = score_cohort([_record(1, date(2026, 7, 1))])
    for field in CI_FIELDS:
        assert metrics[field] is None, f"{field} = {metrics[field]}"
