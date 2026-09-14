"""The anomaly modulator's pure helpers.

`fit_anomaly_multiplier` is the shape the paired read stands on: a bug that
lets it fit a non-monotone curve, memorise the fit level, or vote on noise
turns a conditional-coverage claim into an artifact. Each failure mode gets a
test. `apply_anomaly_multiplier`, `anomaly_decile_error` and
`isotonic_increasing` are pinned alongside, since a drift in binning or in
the level-matched error would move the verdict without touching the fit.
"""

import numpy as np
import pytest
from scripts.anomaly_band_modulator_ab import (
    F_HI,
    F_LO,
    anomaly_decile_error,
    apply_anomaly_multiplier,
    fit_anomaly_multiplier,
    isotonic_increasing,
)


def _signal_frame(n=6000, seed=0):
    """|r|/scale grows along anomaly_p: the mechanism, stated as data."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=n)
    # Decile k of p carries dispersion 1 + 3*k/9 — top-decile rows move ~4x
    # the bottom-decile ones at the same climatology scale.
    k = np.clip((p * 10).astype(int), 0, 9)
    scale = np.full(n, 5.0)
    abs_r = (1.0 + 3.0 * k / 9.0) * rng.uniform(0.2, 1.8, size=n)
    return p, abs_r, scale


class TestIsotonicIncreasing:
    def test_increasing_input_passes_through(self):
        y = np.array([0.5, 0.8, 1.0, 1.5, 3.0])
        np.testing.assert_allclose(isotonic_increasing(y), y)

    def test_decreasing_input_comes_out_flat_and_preserves_the_mean(self):
        # PAVA merges preserve the (weighted) sum, so the mean survives the
        # monotonisation exactly — the fit re-shapes, it never re-levels.
        y = np.array([3.0, 1.5, 1.0, 0.8, 0.5])
        out = isotonic_increasing(y)
        assert np.all(np.diff(out) >= 0)
        assert out.mean() == pytest.approx(y.mean())

    def test_noisy_input_comes_out_non_decreasing(self):
        rng = np.random.default_rng(11)
        y = np.linspace(0.5, 2.0, 10) + rng.normal(0, 0.4, 10)
        out = isotonic_increasing(y)
        assert np.all(np.diff(out) >= -1e-12)

    def test_empty_input_returns_empty(self):
        assert isotonic_increasing(np.array([])).size == 0


class TestFitAnomalyMultiplier:
    def test_shape_is_monotone_and_spread(self):
        p, abs_r, scale = _signal_frame()
        m = fit_anomaly_multiplier(p, abs_r, scale)
        assert m["edges"].size == 9
        assert m["values"].size == 10
        assert np.all(np.diff(m["values"]) >= -1e-12)
        # The mechanism must survive the fit: top wider than bottom.
        assert m["values"][-1] > m["values"][0] + 0.2

    def test_mean_one_on_the_fit_split(self):
        # Exact only where no clip binds; the [F_LO, F_HI] tails move it
        # slightly — same guarantee as the vol-rank production normalisation.
        p, abs_r, scale = _signal_frame()
        m = fit_anomaly_multiplier(p, abs_r, scale)
        got = apply_anomaly_multiplier(p, m)
        assert abs(float(np.mean(got)) - 1.0) < 0.05

    def test_values_stay_inside_the_clip(self):
        p, abs_r, scale = _signal_frame()
        m = fit_anomaly_multiplier(p, abs_r, scale)
        assert np.all(m["values"] >= F_LO) and np.all(m["values"] <= F_HI)

    def test_deterministic(self):
        p, abs_r, scale = _signal_frame()
        a = fit_anomaly_multiplier(p, abs_r, scale)
        b = fit_anomaly_multiplier(p, abs_r, scale)
        np.testing.assert_array_equal(a["edges"], b["edges"])
        np.testing.assert_array_equal(a["values"], b["values"])

    def test_constant_anomaly_p_votes_flat(self):
        rng = np.random.default_rng(3)
        n = 2000
        m = fit_anomaly_multiplier(np.full(n, 0.05), rng.uniform(0.5, 5.0, size=n), np.full(n, 5.0))
        assert m["edges"].size == 0
        np.testing.assert_array_equal(m["values"], np.array([1.0]))

    def test_too_few_rows_vote_flat(self):
        rng = np.random.default_rng(4)
        m = fit_anomaly_multiplier(rng.uniform(0, 1, 100), rng.uniform(0.5, 5.0, 100), np.full(100, 5.0))
        assert m["edges"].size == 0

    def test_non_finite_rows_are_dropped_not_fatal(self):
        p, abs_r, scale = _signal_frame()
        p = p.copy()
        p[:100] = np.nan
        scale = scale.copy()
        scale[100:200] = 0.0
        m = fit_anomaly_multiplier(p, abs_r, scale)
        assert m["values"].size == 10
        assert np.all(np.isfinite(m["values"]))

    def test_unrelated_axis_still_returns_a_monotone_finite_curve(self):
        # Garbage in: monotonicity holds by construction, so the verdict can
        # only come from the eval folds, never from a fit-time shape violation.
        rng = np.random.default_rng(5)
        n = 3000
        m = fit_anomaly_multiplier(rng.uniform(0, 1, n), rng.uniform(0.5, 5.0, n), np.full(n, 5.0))
        assert np.all(np.isfinite(m["values"]))
        assert np.all(np.diff(m["values"]) >= -1e-12)


class TestApplyAnomalyMultiplier:
    def test_bin_mapping_matches_fit_edges(self):
        p, abs_r, scale = _signal_frame()
        m = fit_anomaly_multiplier(p, abs_r, scale)
        lo = apply_anomaly_multiplier(np.array([0.0]), m)[0]
        hi = apply_anomaly_multiplier(np.array([1.0]), m)[0]
        assert lo == m["values"][0]
        assert hi == m["values"][-1]
        assert hi > lo  # high anomaly_p never means a narrower band

    def test_flat_model_returns_ones(self):
        out = apply_anomaly_multiplier(np.array([0.0, 0.5, 1.0]), {"edges": np.array([]), "values": np.array([1.0])})
        np.testing.assert_array_equal(out, np.ones(3))

    def test_mismatched_model_returns_ones(self):
        out = apply_anomaly_multiplier(np.array([0.5]), {"edges": np.array([0.3]), "values": np.array([0.8, 1.2, 9.9])})
        np.testing.assert_array_equal(out, np.ones(1))


class TestAnomalyDecileError:
    def test_modulator_equalizes_coverage_in_sample(self):
        # By construction each decile gets its own q80, so the modulated arm
        # must score a smaller level-matched decile error than the constant
        # scale on the rows the multiplier was fitted on.
        from scripts.shrink_k_vol_rank_ab import matched_width

        p, abs_r, scale = _signal_frame()
        m = fit_anomaly_multiplier(p, abs_r, scale)
        mod = scale * apply_anomaly_multiplier(p, m)
        q_ctl, _ = matched_width(abs_r, scale)
        q_mod, _ = matched_width(abs_r, mod)
        e_ctl = anomaly_decile_error(abs_r, scale, p, q_ctl, m["edges"])
        e_mod = anomaly_decile_error(abs_r, mod, p, q_mod, m["edges"])
        assert np.isfinite(e_ctl) and np.isfinite(e_mod)
        assert e_mod < e_ctl
        assert e_mod < 3.0

    def test_uniform_rescale_leaves_the_error_unchanged(self):
        # Same property the width read has: a pure re-levelling must not read
        # as a conditional fix.
        from scripts.shrink_k_vol_rank_ab import matched_width

        p, abs_r, scale = _signal_frame()
        m = fit_anomaly_multiplier(p, abs_r, scale)
        q1, _ = matched_width(abs_r, scale)
        q2, _ = matched_width(abs_r, 2.0 * scale)
        e1 = anomaly_decile_error(abs_r, scale, p, q1, m["edges"])
        e2 = anomaly_decile_error(abs_r, 2.0 * scale, p, q2, m["edges"])
        assert e1 == e2 == abs(e1)

    def test_degenerate_input_votes_nan_not_zero(self):
        assert np.isnan(anomaly_decile_error(np.array([]), np.array([]), np.array([]), 1.0, np.full(9, np.nan)))
        p, abs_r, scale = _signal_frame(n=500)
        m = fit_anomaly_multiplier(p, abs_r, scale)
        assert np.isnan(anomaly_decile_error(abs_r, scale, p, float("nan"), m["edges"]))
