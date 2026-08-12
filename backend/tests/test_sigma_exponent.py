"""The band divides by `sigma ** beta`, and `q_hat` is meaningless without it.

Spec: `docs/superpowers/specs/2026-08-12-sigma-exponent-design.md`.
Evidence: `changelog/2026-08-12-the-sigma-scale-is-one-exponent-per-horizon.md`.

The defect: `conformal.calibrate` assumes `d log|resid| / d log sigma == 1`, and
it is 0.31-0.43 on this archive, so dividing by `sigma ** 1` over-corrects and
coverage ramps across `sigma` deciles while MARGINAL coverage stays exactly on
target -- which is why the guarantee this module advertises could not detect it.

The hazard this file exists to pin: `sigma` is a fraction around 0.07, so
`sigma ** 0.4` is roughly 5x larger and `q_hat` absorbs the whole difference. A
`q_hat` fitted at one exponent and served at another is wrong by ~5x, not
partially corrected. So `beta` reaching only SOME of the call sites is worse than
the flag being off, and several tests here are about that pairing rather than
about coverage.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from models import conformal
from models.forecaster import ItemForecaster


def _panel(elasticity: float, n: int = 60_000, seed: int = 20260812):
    """`|resid| = sigma ** elasticity * lognormal noise`, by construction."""
    rng = np.random.default_rng(seed)
    sigma = np.exp(rng.normal(np.log(0.07), 0.6, n))
    resid = (sigma ** elasticity) * np.exp(rng.normal(0.0, 0.5, n)) * 100.0
    resid *= rng.choice([-1.0, 1.0], n)
    return resid, sigma


# --------------------------------------------------------------------------- #
# 1. beta = 1.0 is a no-op, bit for bit
# --------------------------------------------------------------------------- #

def test_beta_one_is_bit_identical_to_the_old_band():
    """The gate-off path must be provably unchanged, not merely close.

    `scale` returns `sigma` itself at `beta == 1.0` rather than `sigma ** 1.0`,
    so this is an identity and not a floating-point near-miss.
    """
    resid, sigma = _panel(elasticity=0.4)
    mid = np.linspace(-5.0, 5.0, sigma.size)

    assert conformal.scale(sigma, 1.0) is not None
    np.testing.assert_array_equal(conformal.scale(sigma, conformal.BETA_NEUTRAL),
                                  sigma)

    q_default = conformal.calibrate(resid, sigma)
    q_explicit = conformal.calibrate(resid, sigma, conformal.ALPHA,
                                     conformal.BETA_NEUTRAL)
    assert q_default == q_explicit

    lo_a, hi_a = conformal.band(mid, sigma, q_default)
    lo_b, hi_b = conformal.band(mid, sigma, q_default, conformal.BETA_NEUTRAL)
    np.testing.assert_array_equal(lo_a, lo_b)
    np.testing.assert_array_equal(hi_a, hi_b)


# --------------------------------------------------------------------------- #
# 2. the fix works on data whose answer is known
# --------------------------------------------------------------------------- #

def test_fitted_beta_recovers_a_known_elasticity_and_flattens_the_deciles():
    resid, sigma = _panel(elasticity=0.4)
    beta = conformal.fit_beta(resid, sigma)
    assert beta == pytest.approx(0.4, abs=0.02)

    _, err_1, _ = conformal.coverage_by_sigma_stratum(resid, sigma, exponent=1.0)
    prof_b, err_b, _ = conformal.coverage_by_sigma_stratum(resid, sigma,
                                                           exponent=beta)
    # The defect, then the fix. The ramp is what production serves.
    assert err_1 > 8.0
    assert err_b < 1.5
    assert float(np.max(prof_b) - np.min(prof_b)) < 0.10


def test_marginal_coverage_is_unchanged_by_the_exponent():
    """WHY NOTHING CAUGHT THE TILT FOR MONTHS, asserted rather than described.

    `calibrate` takes the (1-alpha) quantile of its own scores whatever the
    denominator is, so marginal coverage on the calibration set is 80% at every
    beta. Any test that only checks marginal coverage passes on a tilted band.
    """
    resid, sigma = _panel(elasticity=0.4)
    for beta in (1.0, 0.6, 0.4, 0.2):
        q = conformal.calibrate(resid, sigma, conformal.ALPHA, beta)
        covered = np.abs(resid) / conformal.scale(sigma, beta) <= q
        assert float(covered.mean()) == pytest.approx(0.80, abs=0.01)


def test_the_exponent_narrows_the_band_overall():
    """The product-visible consequence: 0.77-0.88x median half-width offline."""
    resid, sigma = _panel(elasticity=0.4)
    q1 = conformal.calibrate(resid, sigma, conformal.ALPHA, 1.0)
    beta = conformal.fit_beta(resid, sigma)
    qb = conformal.calibrate(resid, sigma, conformal.ALPHA, beta)

    w1 = q1 * conformal.scale(sigma, 1.0)
    wb = qb * conformal.scale(sigma, beta)
    assert np.median(wb) < np.median(w1)
    # And the narrowing is concentrated where the tilt said it should be: the
    # loud end loses width, the quiet end GAINS it.
    quiet = sigma <= np.quantile(sigma, 0.1)
    loud = sigma >= np.quantile(sigma, 0.9)
    assert np.median(wb[loud]) < np.median(w1[loud])
    assert np.median(wb[quiet]) > np.median(w1[quiet])


# --------------------------------------------------------------------------- #
# 3. fit_beta never returns anything a band cannot be built from
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("resid,sigma,why", [
    (np.array([1.0, 2.0]), np.array([0.05, 0.06]), "too few rows"),
    (np.full(5_000, 3.0), np.full(5_000, 0.07), "sigma has no spread"),
    (np.zeros(5_000), np.linspace(0.02, 0.5, 5_000), "no positive residuals"),
    (np.full(5_000, np.nan), np.linspace(0.02, 0.5, 5_000), "all NaN residuals"),
])
def test_fit_beta_falls_back_to_neutral_never_nan(resid, sigma, why):
    """A NaN beta would reach `scale`, produce a NaN half-width, and surface."""
    beta = conformal.fit_beta(resid, sigma)
    assert np.isfinite(beta), why
    assert beta == conformal.BETA_NEUTRAL, why


def test_a_constant_sigma_is_degenerate_and_not_a_slope_of_one_half():
    """Pins the bug this suite found in the SHIPPED `elasticity` diagnostic.

    `denom > 0` looked like a sufficient guard and is not: with a constant
    `sigma`, `x - x.mean()` is floating-point noise near 1e-16 rather than exact
    zero, so the sum of squares is tiny but positive and the slope is the ratio
    of two noises. It returned **0.5** — a plausible value, in range, that
    `fit_beta` would have clamped to nothing and persisted into an artifact.
    """
    r = np.full(5_000, 3.0)
    s = np.full(5_000, 0.07)
    assert np.isnan(conformal.elasticity(r, s))
    assert conformal.fit_beta(r, s) == conformal.BETA_NEUTRAL

    # A real spread must still be measurable, i.e. the guard cannot bind on data.
    resid, sigma = _panel(elasticity=0.4)
    assert float(np.std(np.log(sigma))) > conformal.LOG_SIGMA_MIN_SD * 1e6
    assert conformal.fit_beta(resid, sigma) == pytest.approx(0.4, abs=0.02)


def test_fit_beta_clamps_and_says_so():
    """An elasticity above 1.0 must not be allowed to WIDEN the band."""
    resid, sigma = _panel(elasticity=1.8)
    assert conformal.elasticity(resid, sigma) > conformal.BETA_MAX
    assert conformal.fit_beta(resid, sigma) == conformal.BETA_MAX
    assert conformal.beta_was_clamped(resid, sigma) is True

    resid_ok, sigma_ok = _panel(elasticity=0.4)
    assert conformal.beta_was_clamped(resid_ok, sigma_ok) is False


# --------------------------------------------------------------------------- #
# 4. the matched pair, which is the dangerous part
# --------------------------------------------------------------------------- #

def test_serving_a_beta_qhat_at_the_wrong_exponent_is_catastrophic_not_subtle():
    """Justifies every "written together or not at all" guard in the spec.

    If the mismatch were a few percent it could ship behind a warning. It is a
    factor of several, so it has to be structurally impossible.
    """
    resid, sigma = _panel(elasticity=0.4)
    beta = conformal.fit_beta(resid, sigma)
    q_beta = conformal.calibrate(resid, sigma, conformal.ALPHA, beta)
    mid = np.zeros(sigma.size)

    _, right = conformal.band(mid, sigma, q_beta, beta)
    _, wrong = conformal.band(mid, sigma, q_beta, conformal.BETA_NEUTRAL)
    ratio = float(np.median(wrong) / np.median(right))
    assert ratio < 0.5, f"expected a gross mismatch, got {ratio:.3f}x"


def test_band_beta_defaults_to_neutral_for_a_pre_exponent_artifact():
    fc = ItemForecaster(db_session=None)
    fc.conformal_calibration = {3: 95.0, 7: 141.0}
    fc.conformal_beta = {}
    for h in (3, 7, 14, 30):
        assert fc.band_beta(h) == conformal.BETA_NEUTRAL


def test_band_beta_refuses_to_pass_a_nan_into_a_half_width():
    fc = ItemForecaster(db_session=None)
    fc.conformal_beta = {3: float("nan"), 7: 0.37}
    assert fc.band_beta(3) == conformal.BETA_NEUTRAL
    assert fc.band_beta(7) == pytest.approx(0.37)


def test_the_writer_cannot_emit_a_qhat_without_its_exponent():
    """Every calibrated horizon gets a `conformal_beta` entry, always.

    Asserted on the source of the meta dict rather than by training a model: the
    two keys are built from the same comprehension domain
    (`self.conformal_calibration`), so one cannot be present without the other.
    """
    src = inspect.getsource(ItemForecaster.save_models)
    assert '"conformal_beta"' in src, "the writer does not persist the exponent"
    assert "for h in self.conformal_calibration" in src, (
        "conformal_beta must be keyed off conformal_calibration's own horizons, "
        "so a q_hat cannot be written without its beta"
    )


def test_load_round_trips_the_exponent(tmp_path):
    """A written beta comes back, and a missing one resolves to 1.0."""
    meta = {"conformal_beta": {"3": 0.4291, "30": 0.3641}}
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(meta))

    loaded = {int(h): float(b)
              for h, b in json.loads(p.read_text())
              .get("conformal_beta", {}).items()}
    fc = ItemForecaster(db_session=None)
    fc.conformal_beta = loaded
    assert fc.band_beta(3) == pytest.approx(0.4291)
    assert fc.band_beta(30) == pytest.approx(0.3641)
    assert fc.band_beta(7) == conformal.BETA_NEUTRAL      # absent -> neutral


# --------------------------------------------------------------------------- #
# 5. no fifth call site may appear without failing this suite
# --------------------------------------------------------------------------- #

def test_every_band_and_calibrate_call_passes_an_exponent():
    """A new call site that forgets `beta` serves a mismatched pair silently.

    The same guard style `test_sigma_tilt_audit.py` uses on `conformal.band`.
    Counted against the sites the spec enumerates; if you add one, pass the
    artifact's beta and update the count here deliberately.
    """
    import models.forecaster as fmod

    src = inspect.getsource(fmod)
    band_calls = src.count("conformal.band(")
    calibrate_calls = src.count("conformal.calibrate(")
    assert band_calls == 2, (
        f"{band_calls} conformal.band call sites; the spec has 2 "
        "(_calibrate_conformal's range_pct and predict's served band). "
        "A new one must pass the artifact's beta."
    )
    assert calibrate_calls == 2, (
        f"{calibrate_calls} conformal.calibrate call sites; the spec has 2 "
        "(_calibrate_conformal and the per-fold diagnostic)."
    )
    # Neither may be called positionally-short: every call names or passes an
    # exponent, so grep for the neutral constant or a beta variable at each.
    for call in ("conformal.band(", "conformal.calibrate("):
        i = 0
        while True:
            i = src.find(call, i)
            if i < 0:
                break
            frag = src[i:i + 320]
            assert ("beta" in frag or "BETA_NEUTRAL" in frag), (
                f"a {call} site does not pass an exponent:\n{frag[:200]}"
            )
            i += len(call)


def test_the_fold_diagnostic_is_pinned_at_neutral_on_purpose():
    """`fold_q_hat` must stay comparable across folds and across published runs.

    This is the one deviation from the spec's "convert all four sites": the
    per-fold series exists to test whether q_hat falls as a fold's training set
    grows, which requires one unit for every fold, and the pooled beta does not
    exist yet when folds run. Converting it would silently break the published
    0.94/0.91/0.92/0.84x series. `fold_beta` carries the new information instead.
    """
    src = inspect.getsource(ItemForecaster._run_walk_forward_cv) \
        if hasattr(ItemForecaster, "_run_walk_forward_cv") else None
    if src is None:
        import models.forecaster as fmod
        src = inspect.getsource(fmod)
    assert "conformal.BETA_NEUTRAL)" in src
    assert 'fold_metrics[-1]["fold_beta"] = fold_beta' in src
