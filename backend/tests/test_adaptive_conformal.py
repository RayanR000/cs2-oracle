"""Tests for Adaptive Conformal Inference (ACI) — calibrate_adaptive and update_adaptive."""

import numpy as np
import pytest

from models.conformal import calibrate_adaptive, update_adaptive


class TestCalibrateAdaptive:
    def test_returns_per_stratum_qhats(self):
        # 1000 residuals with varying sigma -> should get 10 strata
        rng = np.random.default_rng(42)
        sigmas = rng.exponential(1.0, 1000)
        residuals = rng.normal(0, sigmas)
        q_hats, bin_edges = calibrate_adaptive(residuals, sigmas)
        assert len(q_hats) == 10
        assert len(bin_edges) == 11
        # Higher sigma strata should have larger q_hats
        assert q_hats[9] > q_hats[0]

    def test_per_stratum_coverage_near_target(self):
        # Large sample: each stratum's own coverage should be ~80%
        rng = np.random.default_rng(42)
        sigmas = rng.exponential(1.0, 10000)
        residuals = rng.normal(0, sigmas)
        q_hats, bin_edges = calibrate_adaptive(residuals, sigmas)
        strata = np.digitize(sigmas, bin_edges[1:-1])
        for s in range(10):
            mask = strata == s
            covered = np.abs(residuals[mask]) <= q_hats[s]
            assert 0.75 <= covered.mean() <= 0.85, f"Stratum {s}: {covered.mean():.3f}"

    def test_single_stratum(self):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        sigmas = np.ones(100)
        q_hats, _ = calibrate_adaptive(residuals, sigmas, n_strata=1)
        assert len(q_hats) == 1

    def test_custom_n_strata(self):
        rng = np.random.default_rng(42)
        sigmas = rng.exponential(1.0, 500)
        residuals = rng.normal(0, sigmas)
        q_hats, bin_edges = calibrate_adaptive(residuals, sigmas, n_strata=5)
        assert len(q_hats) == 5
        assert len(bin_edges) == 6

    def test_custom_alpha(self):
        rng = np.random.default_rng(42)
        sigmas = rng.exponential(1.0, 5000)
        residuals = rng.normal(0, sigmas)
        # alpha=0.10 -> 90% coverage target -> wider bands
        q_90, _ = calibrate_adaptive(residuals, sigmas, alpha=0.10)
        q_80, _ = calibrate_adaptive(residuals, sigmas, alpha=0.20)
        # 90% coverage needs wider q_hats than 80%
        assert q_90[0] > q_80[0]


class TestUpdateAdaptive:
    def test_reduces_qhat_on_overcoverage(self):
        q_hats = {0: 1.0, 1: 2.0}
        # miscoverage = 0.10 < alpha=0.20 -> overcovering -> q_hat should decrease
        miscoverage = {0: 0.10, 1: 0.10}
        counts = {0: 100, 1: 100}
        updated = update_adaptive(q_hats, miscoverage, counts)
        assert updated[0] < 1.0
        assert updated[1] < 2.0

    def test_increases_qhat_on_undercoverage(self):
        q_hats = {0: 1.0}
        # miscoverage = 0.30 > alpha=0.20 -> undercovering -> q_hat should increase
        updated = update_adaptive(q_hats, {0: 0.30}, {0: 100})
        assert updated[0] > 1.0

    def test_skips_low_count_strata(self):
        q_hats = {0: 1.0, 1: 2.0}
        updated = update_adaptive(q_hats, {0: 0.10, 1: 0.10}, {0: 100, 1: 3}, min_obs=5)
        assert updated[0] < 1.0  # updated
        assert updated[1] == 2.0  # unchanged -- too few obs

    def test_clamp_bounds(self):
        q_hats = {0: 1.0}
        # Extreme overcoverage for many rounds
        current = dict(q_hats)
        for _ in range(1000):
            current = update_adaptive(current, {0: 0.0}, {0: 100}, eta=0.1)
        assert current[0] >= 0.5  # clamped

    def test_clamp_upper_bound(self):
        q_hats = {0: 1.0}
        # Extreme undercoverage for many rounds
        current = dict(q_hats)
        for _ in range(1000):
            current = update_adaptive(current, {0: 1.0}, {0: 100}, eta=0.1)
        assert current[0] <= 2.0  # clamped

    def test_convergence(self):
        # Simulate 200 rounds with true miscoverage = alpha -> should stabilize
        q_hat = {0: 1.5}
        alpha = 0.20
        for _ in range(200):
            q_hat = update_adaptive(q_hat, {0: alpha}, {0: 50})
        # Should be very close to 1.5 (no net movement when at target)
        assert abs(q_hat[0] - 1.5) < 0.01

    def test_preserves_all_strata(self):
        q_hats = {0: 1.0, 1: 1.5, 2: 2.0}
        miscoverage = {0: 0.15, 1: 0.20, 2: 0.25}
        counts = {0: 50, 1: 50, 2: 50}
        updated = update_adaptive(q_hats, miscoverage, counts)
        assert set(updated.keys()) == {0, 1, 2}

    def test_missing_stratum_in_miscoverage_unchanged(self):
        q_hats = {0: 1.0, 1: 2.0}
        # Stratum 1 not in miscoverage dict
        updated = update_adaptive(q_hats, {0: 0.10}, {0: 100, 1: 100})
        assert updated[1] == 2.0  # unchanged since not in miscoverage
