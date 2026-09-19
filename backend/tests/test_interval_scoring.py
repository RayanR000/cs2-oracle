"""Tests for evaluation.interval_scoring — standardized interval metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from evaluation.interval_scoring import (
    adaptivity,
    average_width,
    calibration_curve,
    compare_methods,
    coverage,
    coverage_by_tier,
    interval_score,
    relative_width,
    score_intervals,
)

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_perfect_coverage(self):
        low = np.array([0.0, 1.0, 2.0])
        high = np.array([2.0, 3.0, 4.0])
        actual = np.array([1.0, 2.0, 3.0])
        assert coverage(low, high, actual) == 1.0

    def test_zero_coverage(self):
        low = np.array([5.0, 5.0, 5.0])
        high = np.array([6.0, 6.0, 6.0])
        actual = np.array([0.0, 0.0, 0.0])
        assert coverage(low, high, actual) == 0.0

    def test_partial_coverage(self):
        low = np.array([0.0, 0.0])
        high = np.array([1.0, 1.0])
        actual = np.array([0.5, 2.0])
        assert coverage(low, high, actual) == 0.5

    def test_boundary_included(self):
        """Actuals exactly on the boundary count as covered."""
        low = np.array([1.0, 1.0])
        high = np.array([2.0, 2.0])
        actual = np.array([1.0, 2.0])
        assert coverage(low, high, actual) == 1.0

    def test_single_observation(self):
        assert coverage(np.array([0.0]), np.array([1.0]), np.array([0.5])) == 1.0
        assert coverage(np.array([0.0]), np.array([1.0]), np.array([2.0])) == 0.0


# ---------------------------------------------------------------------------
# Width metrics
# ---------------------------------------------------------------------------


class TestWidth:
    def test_average_width(self):
        low = np.array([0.0, 1.0])
        high = np.array([2.0, 4.0])
        assert average_width(low, high) == 2.5

    def test_relative_width(self):
        low = np.array([8.0, 18.0])
        mid = np.array([10.0, 20.0])
        high = np.array([12.0, 22.0])
        rw = relative_width(low, mid, high)
        # (4/10 + 4/20) / 2 = (0.4 + 0.2) / 2 = 0.3
        assert abs(rw - 0.3) < 1e-9

    def test_zero_width(self):
        low = np.array([5.0, 5.0])
        high = np.array([5.0, 5.0])
        assert average_width(low, high) == 0.0


# ---------------------------------------------------------------------------
# Interval (Winkler) score
# ---------------------------------------------------------------------------


class TestIntervalScore:
    def test_inside_interval(self):
        """When actual is inside, score equals width."""
        low = np.array([0.0])
        high = np.array([10.0])
        actual = np.array([5.0])
        score = interval_score(low, high, actual, alpha=0.20)
        assert abs(score - 10.0) < 1e-9

    def test_below_interval(self):
        """Penalty for actual below low."""
        low = np.array([5.0])
        high = np.array([10.0])
        actual = np.array([3.0])
        alpha = 0.20
        # width=5, penalty = (2/0.20)*(5-3) = 10*2 = 20, total = 25
        score = interval_score(low, high, actual, alpha=alpha)
        assert abs(score - 25.0) < 1e-9

    def test_above_interval(self):
        """Penalty for actual above high."""
        low = np.array([5.0])
        high = np.array([10.0])
        actual = np.array([14.0])
        alpha = 0.20
        # width=5, penalty = (2/0.20)*(14-10) = 10*4 = 40, total = 45
        score = interval_score(low, high, actual, alpha=alpha)
        assert abs(score - 45.0) < 1e-9

    def test_tighter_is_better_at_same_coverage(self):
        """Tighter intervals score lower (better) when both cover."""
        actual = np.array([5.0, 5.0, 5.0])
        # Wide intervals
        wide_score = interval_score(
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            actual,
            alpha=0.20,
        )
        # Tight intervals
        tight_score = interval_score(
            np.array([4.0, 4.0, 4.0]),
            np.array([6.0, 6.0, 6.0]),
            actual,
            alpha=0.20,
        )
        assert tight_score < wide_score

    def test_multiple_observations(self):
        """Average across observations."""
        low = np.array([0.0, 5.0])
        high = np.array([10.0, 10.0])
        actual = np.array([5.0, 3.0])  # first inside, second below
        alpha = 0.20
        # obs1: width=10
        # obs2: width=5, penalty=(2/0.2)*(5-3)=20, total=25
        # mean = (10+25)/2 = 17.5
        score = interval_score(low, high, actual, alpha=alpha)
        assert abs(score - 17.5) < 1e-9

    def test_zero_width_interval_miss(self):
        """Zero-width interval that misses gets a penalty."""
        low = np.array([5.0])
        high = np.array([5.0])
        actual = np.array([7.0])
        alpha = 0.20
        # width=0, penalty=(2/0.2)*(7-5)=20
        score = interval_score(low, high, actual, alpha=alpha)
        assert abs(score - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# Adaptivity
# ---------------------------------------------------------------------------


class TestAdaptivity:
    def test_positive_correlation(self):
        """Wider intervals for harder predictions -> positive correlation."""
        rng = np.random.default_rng(42)
        n = 200
        difficulty = rng.uniform(0.5, 5.0, n)
        mid = np.full(n, 10.0)
        actual = mid + rng.normal(0, 1, n) * difficulty
        # Intervals that track difficulty
        low = mid - difficulty * 2
        high = mid + difficulty * 2
        result = adaptivity(low, mid, high, actual)
        assert result["width_vs_abs_error"] > 0.3
        assert result["width_vs_sq_error"] > 0.3

    def test_constant_width_returns_nan(self):
        """Constant-width intervals -> NaN (Spearman is undefined on constant input)."""
        rng = np.random.default_rng(42)
        n = 200
        mid = np.full(n, 10.0)
        actual = mid + rng.normal(0, 1, n)
        low = mid - 2.0
        high = mid + 2.0
        result = adaptivity(low, mid, high, actual)
        assert np.isnan(result["width_vs_abs_error"])
        assert np.isnan(result["width_vs_sq_error"])

    def test_single_observation_returns_nan(self):
        result = adaptivity(np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([1.0]))
        assert np.isnan(result["width_vs_abs_error"])


# ---------------------------------------------------------------------------
# Coverage by tier
# ---------------------------------------------------------------------------


class TestCoverageByTier:
    def test_basic_tiers(self):
        prices = np.array([0.5, 2.0, 10.0, 50.0, 200.0, 1500.0])
        low = prices * 0.5
        high = prices * 1.5
        actual = prices * 1.0  # all inside
        result = coverage_by_tier(low, high, actual, prices)
        # All inside, so every tier should be 1.0
        for _tier, cov in result.items():
            assert cov == 1.0

    def test_tier_miss(self):
        prices = np.array([0.5, 2.0])
        low = np.array([0.0, 0.0])
        high = np.array([1.0, 1.0])
        actual = np.array([0.5, 5.0])  # tier 0 hit, tier 1 miss
        result = coverage_by_tier(low, high, actual, prices)
        assert result[0] == 1.0
        assert result[1] == 0.0


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------


class TestCalibrationCurve:
    def test_perfect_calibration(self):
        """When intervals are correctly sized, empirical ~ nominal."""
        rng = np.random.default_rng(42)
        n = 5000
        mid = np.full(n, 100.0)
        actual = mid + rng.normal(0, 10.0, n)
        # Build intervals at each nominal level from the known distribution

        results = calibration_curve(
            mid=mid,
            actual=actual,
            std_estimate=np.full(n, 10.0),
        )
        # Check that nominal and empirical are close (within 3pp)
        for nominal, empirical in results:
            assert abs(nominal - empirical) < 0.03, (
                f"nominal={nominal}, empirical={empirical}"
            )

    def test_overconfident(self):
        """Narrow intervals -> empirical < nominal at high levels."""
        rng = np.random.default_rng(42)
        n = 5000
        mid = np.full(n, 100.0)
        actual = mid + rng.normal(0, 20.0, n)  # actual spread is 20
        # But we claim std is 5 (overconfident)
        results = calibration_curve(
            mid=mid,
            actual=actual,
            std_estimate=np.full(n, 5.0),
        )
        # At 90% nominal, empirical should be much lower
        for nominal, empirical in results:
            if nominal >= 0.90:
                assert empirical < nominal


# ---------------------------------------------------------------------------
# score_intervals (integration)
# ---------------------------------------------------------------------------


class TestScoreIntervals:
    def test_returns_all_keys(self):
        low = np.array([0.0, 1.0])
        mid = np.array([5.0, 5.0])
        high = np.array([10.0, 9.0])
        actual = np.array([5.0, 5.0])
        result = score_intervals(low, mid, high, actual, alpha=0.20)
        expected_keys = {
            "coverage",
            "average_width",
            "relative_width",
            "interval_score",
            "adaptivity",
        }
        assert expected_keys.issubset(result.keys())

    def test_all_identical_predictions(self):
        """All predictions and actuals identical -> perfect coverage, zero width."""
        v = np.array([5.0, 5.0, 5.0])
        result = score_intervals(v, v, v, v, alpha=0.20)
        assert result["coverage"] == 1.0
        assert result["average_width"] == 0.0
        assert result["interval_score"] == 0.0


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------


class TestCompareMethods:
    def test_structure(self):
        actual = np.array([5.0, 5.0, 5.0])
        methods = {
            "wide": (np.array([0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), np.array([10.0, 10.0, 10.0])),
            "tight": (np.array([4.0, 4.0, 4.0]), np.array([5.0, 5.0, 5.0]), np.array([6.0, 6.0, 6.0])),
        }
        df = compare_methods(methods, actual, alpha=0.20)
        assert isinstance(df, pd.DataFrame)
        assert set(df.index) == {"wide", "tight"}
        assert "coverage" in df.columns
        assert "interval_score" in df.columns

    def test_tight_beats_wide_on_score(self):
        actual = np.array([5.0, 5.0, 5.0])
        methods = {
            "wide": (np.array([0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), np.array([10.0, 10.0, 10.0])),
            "tight": (np.array([4.0, 4.0, 4.0]), np.array([5.0, 5.0, 5.0]), np.array([6.0, 6.0, 6.0])),
        }
        df = compare_methods(methods, actual, alpha=0.20)
        assert df.loc["tight", "interval_score"] < df.loc["wide", "interval_score"]
