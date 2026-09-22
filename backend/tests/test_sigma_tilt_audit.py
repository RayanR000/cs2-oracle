"""The confirm read for the `sigma` tilt, and the guards that keep it honest.

Why it exists. `docs/changelog/2026-08-12-the-band-is-tilted-in-sigma.md`
measured `d log|residual| / d log sigma` at **0.408 / 0.401 / 0.363 / 0.327**
(3/7/14/30d) over 731 dates offline, against the **1.000** split conformal's
normalisation assumes, with level-matched coverage ramping **62 -> 95%** at h=3
and **58 -> 98%** at h=30 across `sigma` deciles. Every date-conditional remedy
was refuted in the same read, so this axis is the only live one.

But that read is model-free (`r_hat = 0`) and it CONTRADICTS the
**0.798 / 0.692 / 1.034 / 1.150** in
`2026-08-12-conformal-basis-follows-serving.md`, which reconstructed `sigma` as
`half_pct / q_hat` from ~20K prod rows and called the tilt second-order. The
sign agrees at every horizon; the magnitude differs by 3x. `_sigma_tilt_audit`
settles it on the real OOF residuals, which is the population `q_hat` is fitted
on.

**Reported only, and it must stay that way.** The exponent is not implemented and
must not be until this lands: three of the five causes already excluded in this
investigation were refuted by a sign rather than a size.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from models import conformal
from models.forecaster import ItemForecaster

from tests._source import method_closure_source


def _planted(n=40_000, beta=0.4, seed=0):
    """Residuals whose elasticity to sigma is `beta` by construction."""
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.02, 0.5, size=n)
    resid = rng.normal(scale=(sigma**beta) * 3.0, size=n)
    return resid, sigma


@pytest.mark.parametrize("beta", [0.35, 0.4, 0.7, 1.0, 1.15])
def test_elasticity_recovers_a_planted_exponent(beta):
    resid, sigma = _planted(beta=beta)
    assert conformal.elasticity(resid, sigma) == pytest.approx(beta, abs=0.03)


def test_the_stratum_read_is_level_matched_by_construction():
    """Marginal coverage must be exactly nominal for EVERY exponent, or the
    statistic is measuring band width instead of conditional coverage.

    This is the defect that voided the pre-registered read on 2026-08-12: a
    shuffled-state placebo passed the bar because `mean|cov - target|` falls
    whenever the marginal level drifts toward target.
    """
    resid, sigma = _planted()
    for exponent in (0.2, 1.0, 1.5):
        per, err, threshold = conformal.coverage_by_sigma_stratum(resid, sigma, exponent=exponent)
        scores = np.abs(resid) / (sigma**exponent)
        assert float((scores <= threshold).mean()) == pytest.approx(conformal.NOMINAL_COVERAGE, abs=0.005)
        # A per-stratum mean cannot be outside [0, 1] and the error follows it.
        assert per.min() >= 0.0 and per.max() <= 1.0
        assert err >= 0.0


def test_the_strata_are_the_same_rows_whatever_the_exponent_is():
    """Stratifying on the exponentiated denominator would re-cut the deciles per
    arm, and two arms cut differently are not comparable."""
    resid, sigma = _planted()
    _, err_1, _ = conformal.coverage_by_sigma_stratum(resid, sigma, 1.0)
    _, err_b, _ = conformal.coverage_by_sigma_stratum(resid, sigma, 0.4)
    # The planted exponent flattens it; production's does not.
    assert err_1 > 10.0
    assert err_b < 1.5

    src = inspect.getsource(conformal.coverage_by_sigma_stratum)
    assert "np.quantile(s, np.linspace" in src


def test_a_sub_unit_elasticity_under_covers_the_LOW_sigma_rows():
    """The direction, which is easy to get backwards. The score is
    `|r| / sigma ** 1 ∝ sigma ** (beta - 1)`, so with `beta < 1` it FALLS as
    sigma rises: low-sigma rows score high, breach the threshold more often, and
    are the under-covered ones."""
    resid, sigma = _planted(beta=0.4)
    per, _, _ = conformal.coverage_by_sigma_stratum(resid, sigma, exponent=1.0)
    assert per[0] < conformal.NOMINAL_COVERAGE
    assert per[-1] > conformal.NOMINAL_COVERAGE
    assert np.all(np.diff(per) > -0.02)  # monotone up, allowing sampling noise

    # And the mirror case, so the assertion above is not just true of any data.
    resid_hi, sigma_hi = _planted(beta=1.6)
    per_hi, _, _ = conformal.coverage_by_sigma_stratum(resid_hi, sigma_hi, exponent=1.0)
    assert per_hi[0] > conformal.NOMINAL_COVERAGE
    assert per_hi[-1] < conformal.NOMINAL_COVERAGE


def test_marginal_coverage_cannot_detect_the_tilt():
    """Why nothing caught this. `calibrate` + `band` deliver nominal coverage on
    average at any elasticity, so the guarantee the module advertises is intact
    while conditional coverage is 40pp apart across the sigma range."""
    resid, sigma = _planted(beta=0.4)
    q_hat = conformal.calibrate(resid, sigma, conformal.ALPHA)
    low, high = conformal.band(np.zeros(resid.size), sigma, q_hat)
    marginal = float(((resid >= low) & (resid <= high)).mean())
    assert marginal == pytest.approx(conformal.NOMINAL_COVERAGE, abs=0.01)

    per, err, _ = conformal.coverage_by_sigma_stratum(resid, sigma, 1.0)
    assert per.max() - per.min() > 0.30
    assert err > 10.0


def test_the_audit_holds_out_the_last_fold():
    """The pooled legs fit and score the exponent on the same rows, and an
    in-sample elasticity confirms nothing. The held-out leg fits on every fold
    but the last and scores on the last, which is the only leg that can fail."""
    src = inspect.getsource(ItemForecaster._sigma_tilt_audit)
    assert "elasticity_heldout" in src
    assert "folds < last" in src
    assert "folds == last" in src

    # It needs the fold tag, which the CV path has to attach.
    cv = inspect.getsource(ItemForecaster._cv_evaluate_horizon)
    assert '_rec["fold"] = fold_id' in cv


def test_the_audit_excludes_the_clip_as_the_cause():
    """Rows pinned at the sigma floor or cap carry a sigma detached from the
    item's volatility. If they are what produces the tilt, the remedy is the clip
    and not the exponent — so the audit reports both, and the share clipped."""
    src = inspect.getsource(ItemForecaster._sigma_tilt_audit)
    assert "elasticity_unclipped" in src
    assert "pct_rows_clipped" in src
    assert 'self.sigma_clip.get("floor"' in src
    assert 'self.sigma_clip.get("cap"' in src


def test_the_audit_reaches_meta_json_and_never_the_band():
    """One `meta.json` key and one log line, or a dispatch gets read by eye.

    ⚠️ UPDATED 2026-08-12. This test used to assert that `conformal.band` still
    divided by `sigma ** 1` literally, because at the time the exponent was
    measured but deliberately not implemented. It IS implemented now
    (`SIGMA_EXPONENT`, `docs/superpowers/specs/2026-08-12-sigma-exponent-design.md`),
    so the assertion moves to the invariant that actually matters and that the old
    one was standing in for: the AUDIT still cannot move the band, and the band's
    default exponent is still the neutral one.
    """
    train = method_closure_source(ItemForecaster, "_train_horizon_inline")
    assert '"sigma_tilt": sigma_tilt' in train

    predict = method_closure_source(ItemForecaster, "predict")
    assert "sigma_tilt" not in predict
    assert "elasticity" not in predict

    # The audit is diagnostic: it reports an exponent and never assigns one.
    audit = inspect.getsource(ItemForecaster._sigma_tilt_audit)
    assert "conformal_beta" not in audit
    assert "self.conformal_calibration" not in audit

    # The band takes its exponent from the artifact and defaults to neutral, so
    # an artifact that predates the field serves exactly the band it was
    # calibrated for.
    #
    # ⚠️ UPDATED AGAIN 2026-08-12, and for the same reason as the note above.
    # This asserted the literal line `half = float(q_hat) * scale(sigma, beta)`.
    # `band` now routes through `resolve_scale`, which is what lets a LEARNED
    # scale replace `sigma` entirely (`LEARNED_SCALE`, `models/scale_model.py`).
    # Pinning the old text would forbid that without protecting anything: what
    # the line was standing in for is that the band's DEFAULT is unchanged, so
    # every artifact written before either flag serves exactly what it was
    # calibrated for. That is asserted behaviourally below instead of textually,
    # which is stronger — it would catch a wrong default that still matched the
    # old string.
    band_src = inspect.getsource(conformal.band)
    assert "beta: float = BETA_NEUTRAL" in band_src
    assert "learned_scale=None" in band_src
    assert conformal.BETA_NEUTRAL == 1.0
    sigma = np.array([0.02, 0.07, 0.3])
    np.testing.assert_array_equal(conformal.scale(sigma), sigma)

    # The default path, end to end: no exponent and no learned scale must give
    # back exactly `q_hat * sigma`.
    lo, hi = conformal.band(np.zeros(3), sigma, q_hat=2.0)
    np.testing.assert_array_almost_equal(hi - lo, 2.0 * 2.0 * sigma)
    np.testing.assert_array_equal(conformal.resolve_scale(sigma), sigma)


def test_the_audit_honours_the_calibration_floor():
    """`MIN_CALIBRATION_ROWS` guards the pooled fit; a diagnostic that ignored it
    would publish a decile profile decided by a handful of tail draws."""
    src = inspect.getsource(ItemForecaster._sigma_tilt_audit)
    assert "resid.size < self.MIN_CALIBRATION_ROWS" in src
    assert src.count("self.MIN_CALIBRATION_ROWS") >= 2  # pooled and held-out

    # And the pure helpers degrade rather than raise on a thin input.
    per, err, thr = conformal.coverage_by_sigma_stratum(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    assert per.size == 0 and np.isnan(err) and np.isnan(thr)
    assert np.isnan(conformal.elasticity(np.array([]), np.array([])))
