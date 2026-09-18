"""Directional accuracy is clustered by forecast_date, and must be reported so.

Every item sharing a forecast_date is exposed to the same market-wide move, so
N forecasts on one date are nowhere near N independent observations. The stored
2026-08-02 cohorts span ONE or TWO distinct forecast dates (2025-12-01, rising;
2026-07-17, falling) while carrying five-figure sample_counts, and the model's
persistent "down" bias makes it score 33% on the up date and 64% on the down
one. Resampling items as if independent turns two market days into a tight
confidence interval around a number that is really one coin flip per date.
"""

from __future__ import annotations

from datetime import date

import pytest
from backtest.scoring import (
    MIN_HEADLINE_DATES,
    block_bootstrap_ci,
    bootstrap_ci,
    score_cohort,
)


def _record(correct, forecast_date, tier=1, item_id=1):
    return {
        "abs_error": 0.1,
        "pct_error": 2.0,
        "sq_error": 0.01,
        "direction_correct": correct,
        "predicted_direction": "down",
        "actual_direction": "down" if correct else "up",
        "in_interval": 1,
        "confidence": "high",
        "base_price": 5.0,
        "actual_price": 5.1,
        "price_tier": tier,
        "item_id": item_id,
        "forecast_date": forecast_date,
    }


D1 = date(2025, 12, 1)
D2 = date(2026, 7, 17)


class TestDateCoverageIsReported:
    def test_distinct_forecast_dates_is_in_the_metrics(self):
        records = [_record(1, D1, item_id=i) for i in range(50)] + [_record(0, D2, item_id=100 + i) for i in range(50)]
        metrics, n = score_cohort(records)
        assert n == 100
        assert metrics["distinct_forecast_dates"] == 2

    def test_a_single_date_cohort_is_flagged_insufficient(self):
        records = [_record(1, D1, item_id=i) for i in range(5000)]
        metrics, _ = score_cohort(records)
        assert metrics["distinct_forecast_dates"] == 1
        assert metrics["date_coverage_sufficient"] is False, (
            "5,000 forecasts on one market day must not read as a sufficient cohort"
        )

    def test_sufficiency_threshold_is_met_at_the_minimum(self):
        records = [_record(1, date(2026, 1, d + 1), item_id=d) for d in range(MIN_HEADLINE_DATES)]
        metrics, _ = score_cohort(records)
        assert metrics["distinct_forecast_dates"] == MIN_HEADLINE_DATES
        assert metrics["date_coverage_sufficient"] is True

    def test_minimum_is_greater_than_the_two_dates_currently_stored(self):
        """The stored cohorts span at most 2 dates; the gate must catch them."""
        assert MIN_HEADLINE_DATES > 2


class TestBlockBootstrapWidensWithClustering:
    def test_clustered_ci_is_wider_than_the_naive_one(self):
        """The whole point: items are not independent draws.

        Two dates that disagree completely. Resampling items gives a tight
        interval around 50%; resampling dates must admit that the truth could
        be either date's answer.
        """
        records = [_record(1, D1, item_id=i) for i in range(500)] + [
            _record(0, D2, item_id=1000 + i) for i in range(500)
        ]
        values = [r["direction_correct"] for r in records]
        clusters = [r["forecast_date"] for r in records]

        naive_lo, naive_hi = bootstrap_ci(values)
        block_lo, block_hi = block_bootstrap_ci(values, clusters)

        assert (block_hi - block_lo) > (naive_hi - naive_lo), "clustered CI must be wider than the item-resampled one"
        # Resampling 2 all-or-nothing dates must reach both extremes.
        assert block_lo == pytest.approx(0.0, abs=0.01)
        assert block_hi == pytest.approx(1.0, abs=0.01)

    def test_ci_is_deterministic(self):
        records = [_record(i % 2, D1 if i % 3 else D2, item_id=i) for i in range(300)]
        values = [r["direction_correct"] for r in records]
        clusters = [r["forecast_date"] for r in records]
        assert block_bootstrap_ci(values, clusters) == block_bootstrap_ci(values, clusters)

    def test_single_cluster_yields_no_interval(self):
        """One date carries no information about between-date variation."""
        records = [_record(i % 2, D1, item_id=i) for i in range(200)]
        lo, hi = block_bootstrap_ci(
            [r["direction_correct"] for r in records],
            [r["forecast_date"] for r in records],
        )
        assert lo is None and hi is None

    def test_score_cohort_reports_the_clustered_interval(self):
        records = [_record(1, D1, item_id=i) for i in range(500)] + [
            _record(0, D2, item_id=1000 + i) for i in range(500)
        ]
        metrics, _ = score_cohort(records)
        naive_width = metrics["directional_accuracy_ci_upper"] - metrics["directional_accuracy_ci_lower"]
        clustered_width = (
            metrics["directional_accuracy_ci_clustered_upper"] - metrics["directional_accuracy_ci_clustered_lower"]
        )
        assert clustered_width > naive_width
