"""The learned band scale.

The tests that matter here are the two correctness properties, not the fit
quality: that the scale is out-of-sample where `q_hat` is calibrated on it, and
that a degenerate prediction cannot collapse a band.
"""

import numpy as np
import pandas as pd
import pytest
from models import conformal, scale_model

pytestmark = pytest.mark.slow


def _heteroscedastic(n=24_000, seed=0):
    """A cohort whose error size depends on a feature the model can see, and on
    a second feature it cannot use because it is pure noise.

    `sigma` is deliberately a BAD proxy here -- it is related to the true error
    scale only through a square root, which is the same over-reaction shape the
    real archive shows (sigma spans 11x, |residual| 2.3-2.6x). A learned scale
    should beat it; that is the whole hypothesis.
    """
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.5, 4.0, n)
    noise = rng.normal(size=n)
    true_scale = driver
    resid = rng.normal(scale=true_scale)
    X = pd.DataFrame(
        {
            "driver": driver,
            "noise": noise,
            "sigma": driver**2.0,  # over-reacts, exactly like the real one
        }
    )
    return X, resid, X["sigma"].to_numpy()


def _coverage_by_stratum(resid, scale, q_hat, n_strata=5):
    """Share covered within strata of the scale -- the served-tilt statistic."""
    order = np.argsort(np.argsort(scale))
    edges = (order * n_strata) // len(scale)
    covered = np.abs(resid) <= q_hat * scale
    return np.array([covered[edges == s].mean() for s in range(n_strata)])


def test_the_learned_scale_flattens_a_tilt_that_sigma_leaves():
    """The hypothesis, stated as a test: when the error size is driven by
    something `sigma` only distorts, dividing by a learned scale gives coverage
    that holds across the range where dividing by `sigma` does not."""
    X, resid, sigma = _heteroscedastic()
    folds = np.arange(len(X)) % 6

    q_sigma = conformal.calibrate(resid, sigma, conformal.ALPHA)
    prof_sigma = _coverage_by_stratum(resid, sigma, q_sigma)

    learned, n_models = scale_model.cross_fit(X, resid, folds, fallback=sigma)
    q_learned = conformal.calibrate(resid, learned, conformal.ALPHA)
    prof_learned = _coverage_by_stratum(resid, learned, q_learned)

    assert n_models == 6
    tilt_sigma = np.mean(np.abs(prof_sigma - conformal.NOMINAL_COVERAGE))
    tilt_learned = np.mean(np.abs(prof_learned - conformal.NOMINAL_COVERAGE))
    assert tilt_learned < tilt_sigma / 2, (
        f"learned scale did not flatten the profile: sigma {prof_sigma.round(3)} -> learned {prof_learned.round(3)}"
    )


def test_a_constant_factor_on_the_scale_leaves_the_band_where_it_was():
    """`q_hat` is a quantile of `|residual| / scale`, so scaling every row by a
    constant divides `q_hat` by it and the interval is unchanged. This is why
    the model only has to get RELATIVE variation right, and why a constant
    offset in log space is free."""
    X, resid, _sigma = _heteroscedastic(n=8_000)
    s = X["driver"].to_numpy()

    q1 = conformal.calibrate(resid, s, conformal.ALPHA)
    q2 = conformal.calibrate(resid, s * 37.0, conformal.ALPHA)

    lo1, hi1 = conformal.band(np.zeros_like(s), s, q1)
    lo2, hi2 = conformal.band(np.zeros_like(s), s * 37.0, q2)
    assert np.allclose(lo1, lo2)
    assert np.allclose(hi1, hi2)


def test_cross_fit_never_scores_a_row_with_a_model_that_saw_its_fold():
    """The honesty property. If this breaks, `q_hat` is fitted against a scale
    that already knows the answer, and the band under-covers in production while
    looking perfect offline."""
    n = 12_000
    rng = np.random.default_rng(3)
    fold = np.arange(n) % 4
    # A feature that is CONSTANT within a fold and differs between them. A model
    # that saw the row's own fold can read the fold's residual level off it; one
    # that did not has never seen this value and cannot.
    X = pd.DataFrame({"fold_marker": fold.astype(float), "noise": rng.normal(size=n)})
    # Each fold has a wildly different error scale, keyed to the marker.
    resid = rng.normal(scale=np.array([0.1, 1.0, 10.0, 100.0])[fold])

    learned, _ = scale_model.cross_fit(X, resid, fold, fallback=np.ones(n))
    # An in-sample fit would track the per-fold scale across four orders of
    # magnitude. An out-of-sample one cannot: it is extrapolating to a marker
    # value it never saw, so the spread between fold means must stay small
    # relative to the 1000x spread in the truth.
    means = np.array([learned[fold == f].mean() for f in range(4)])
    assert means.max() / means.min() < 50.0, (
        f"cross_fit leaked its own fold: per-fold scale means {means.round(3)} track the 1000x truth too closely"
    )


def test_a_fold_whose_complement_is_too_small_falls_back_rather_than_leaking():
    """Better no model than one that saw the row."""
    n = 6_000
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"a": rng.normal(size=n)})
    resid = rng.normal(size=n)
    fold = np.zeros(n, dtype=int)
    fold[: n // 2] = 1  # two folds, each complement is 3,000 < MIN_FIT_ROWS

    learned, n_models = scale_model.cross_fit(X, resid, fold, fallback=np.full(n, 7.0))
    assert n_models == 0
    assert np.allclose(learned, 7.0)


def test_the_clip_bounds_exclude_the_tails_they_are_there_to_catch():
    """1,000 ordinary rows and one degenerate value at each end. The bounds must
    come from the body of the distribution -- with a cohort this size the
    outliers sit outside the 1st/99th percentiles and cannot set them.

    (With only 100 rows a single outlier IS the 99th percentile and legitimately
    moves the bound. The conformal pool carries ~155-176K rows, so the regime
    that matters is this one.)
    """
    s = np.concatenate([np.full(1_000, 1.0), [1e-9], [1e9]])
    floor, cap = scale_model.clip_bounds(s)
    assert floor == pytest.approx(1.0)
    assert cap == pytest.approx(1.0)
    assert floor <= cap


def test_a_degenerate_prediction_cannot_collapse_the_band():
    """A near-zero scale would give an item a zero-width interval and zero
    coverage. The clip is what stands between a thin corner of feature space and
    a band that is a point."""
    X, resid, sigma = _heteroscedastic(n=8_000)
    booster = scale_model.fit(X, resid)
    assert booster is not None

    s = scale_model.predict_scale(booster, X, clip=(0.5, 2.0), fallback=sigma)
    assert s.min() >= 0.5
    assert s.max() <= 2.0
    assert np.isfinite(s).all()


def test_a_missing_model_serves_sigma_rather_than_nan():
    """None is a supported outcome of `fit`, so it must be a supported input
    here: the fallback is the pre-existing behaviour, and a NaN half-width
    surfaces in the API as a silently missing interval."""
    X = pd.DataFrame({"a": np.arange(5.0)})
    s = scale_model.predict_scale(None, X, fallback=np.full(5, 3.0))
    assert np.allclose(s, 3.0)


def test_fit_declines_a_tiny_cohort_instead_of_returning_a_junk_model():
    X = pd.DataFrame({"a": np.arange(100.0)})
    assert scale_model.fit(X, np.ones(100)) is None


def test_a_zero_residual_is_floored_not_dropped():
    """An unchanged price is a real outcome and it belongs to the calmest items
    -- the half of the range the band currently gets wrong. Dropping those rows
    would remove the evidence for the defect being fixed."""
    t = scale_model._target([0.0, 0.5])
    assert np.isfinite(t).all()
    assert t[0] == pytest.approx(np.log(scale_model.LOG_RESID_FLOOR_PCT))


def test_the_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LEARNED_SCALE", raising=False)
    assert scale_model.enabled() is False
    monkeypatch.setenv("LEARNED_SCALE", "1")
    assert scale_model.enabled() is True
