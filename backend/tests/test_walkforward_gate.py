"""The walkforward gate must score what production serves, with the clustered CI.

Two defects this locks down:
  - it aggregated per-fold metrics weighted by sample_count, treating
    correlated item-rows as independent observations;
  - it derived the predicted direction from the sign of the p50 regression,
    while production serves the directional classifier's argmax
    (forecaster.py:2981-2982).
"""
from __future__ import annotations

import inspect
from datetime import date

import pytest

import scripts.walkforward_backtest as wf
from backtest.scoring import HEADLINE_TIER


def test_sample_count_weighted_aggregation_is_gone():
    src = inspect.getsource(wf)
    assert 'f["directional_accuracy"] * f["sample_count"]' not in src, (
        "per-fold sample_count weighting treats correlated item-rows as "
        "independent; aggregate over pooled records via score_cohort instead"
    )


def test_compute_metrics_helper_is_removed():
    assert not hasattr(wf, "_compute_metrics"), (
        "_compute_metrics had no flat band (2-label at 50% chance) and took "
        "the median's sign as the prediction; both are replaced by "
        "fold_records + score_cohort"
    )


def test_aggregate_reports_clustered_ci_and_coverage_flag():
    records = [
        {
            "abs_error": 1.0, "sq_error": 1.0, "pct_error": 1.0,
            "direction_correct": i % 2, "predicted_direction": "up",
            "actual_direction": "up" if i % 2 else "down",
            "in_interval": 1, "confidence": "low",
            "base_price": 10.0, "actual_price": 11.0, "price_tier": 2,
            "item_id": f"i{i}", "forecast_date": date(2026, 1, 1 + (i % 25)),
        }
        for i in range(200)
    ]
    agg = wf._aggregate_records(records)
    assert agg["directional_accuracy_ci_clustered_lower"] is not None
    assert agg["distinct_forecast_dates"] == 25
    assert agg["date_coverage_sufficient"] is True


def test_aggregate_uses_the_dollar_headline_tier():
    # Tier 0 (<$1) is 72% of the universe and its labels are tick-quantised.
    # The headline must be the >=$1 aggregate, matching production.
    records = []
    for i in range(60):
        records.append({
            "abs_error": 1.0, "sq_error": 1.0, "pct_error": 1.0,
            "direction_correct": 1, "predicted_direction": "up",
            "actual_direction": "up", "in_interval": 1, "confidence": "low",
            "base_price": 50.0, "actual_price": 51.0, "price_tier": 3,
            "item_id": f"rich{i}", "forecast_date": date(2026, 1, 1 + i % 25),
        })
    for i in range(60):
        records.append({
            "abs_error": 1.0, "sq_error": 1.0, "pct_error": 1.0,
            "direction_correct": 0, "predicted_direction": "up",
            "actual_direction": "down", "in_interval": 0, "confidence": "low",
            "base_price": 0.10, "actual_price": 0.09, "price_tier": 0,
            "item_id": f"penny{i}", "forecast_date": date(2026, 1, 1 + i % 25),
        })
    agg = wf._aggregate_records(records)
    # 100% on the >=$1 subset, not the 50% all-tiers figure.
    assert agg["directional_accuracy"] == pytest.approx(100.0)


def test_fold_scores_both_estimators():
    sig = inspect.signature(wf._score_fold)
    assert "predicted_classes" in sig.parameters
    assert "mid_returns_pct" in sig.parameters


def test_classifier_is_fitted_per_fold():
    src = inspect.getsource(wf)
    assert "_fit_direction_classifier" in src, (
        "the gate must fit the directional classifier production serves, not "
        "score the median's sign"
    )


def _fake_results_by_horizon():
    return {
        3: {
            "classifier": {"directional_accuracy": 55.0},
            "median_sign": {"directional_accuracy": 50.0},
            "sample_count": 10,
            "records": [{"item_id": "a"}, {"item_id": "b"}],
        }
    }


def test_return_records_true_keeps_records_in_the_report():
    horizons = wf._build_horizons_report(_fake_results_by_horizon(), return_records=True)
    assert horizons["3"]["records"] == [{"item_id": "a"}, {"item_id": "b"}]


def test_return_records_false_omits_records_from_the_report():
    horizons = wf._build_horizons_report(_fake_results_by_horizon(), return_records=False)
    assert "records" not in horizons["3"]
