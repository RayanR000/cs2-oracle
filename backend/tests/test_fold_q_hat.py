"""Per-fold `q_hat` is reported so the pooled one can be audited.

Why it exists. `q_hat` is fitted on residuals pooled across expanding-window CV
folds whose models saw **87,224 to 300,000** training rows, while the shipped
model trains on the full 1.2M-row budget. Split conformal's coverage guarantee
assumes the calibration residuals are exchangeable with the served ones; they
are not, because they come from systematically weaker models. A pooled `q_hat`
is then conservative *by construction*.

That predicts uniform over-coverage, which is what the 2026-08-12 panel shows:
the served band covers 87.2 / 91.8 / 90.6 / 89.0% against 80%, and the 80th
percentile of the served nonconformity score is below 1 in **19 of 20** sigma
strata — the whole range, not the tail. Four other causes are already excluded
(the calibration centre, quiet forecast dates, the calibration denominator, and
sigma's level).

`fold_q_hat` screens it: if it falls as `n_train` grows, the pooled fit is
inheriting the early folds' weakness.

It is only a screen, and the reason is `CV_MAX_TRAIN_ROWS` (300,000, shipped
2026-08-09 as `6b6fc81` to cut the conformal-CV phase). It binds on most folds,
so they share one `n_train` and the rho has barely any x-axis — and the range it
does span, 87K→300K, is smaller than the 300K→1.2M gap that actually separates a
fold model from the served one. **A rho near zero is therefore not evidence
against the hypothesis.** The decisive test is a `CV_MAX_TRAIN_ROWS` sweep, which
holds the fold geometry fixed and moves only the training size; that knob is a
`model-diagnostics.yml` input.

Worth noting where the hypothesis came from: the cap's own comment predicted this
outcome — "the fold model then fits on less data than the served one, so q_hat
comes out LARGER and the band wider — over-coverage, which is the safe direction.
Verify against NOMINAL_COVERAGE before lowering it." The band does now over-cover.

**Reported only.** Nothing builds a band from it — the remedy is a
calibration-set change and needs its own decision.

See docs/changelog/2026-08-12-conformal-basis-follows-serving.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from models import conformal
from models.conformal import ALPHA, calibrate


def test_a_weaker_model_yields_a_larger_q_hat():
    """The mechanism, isolated. Two models over the same outcomes, one noisier;
    the noisier one's residuals calibrate to a wider band. Pooling them and
    serving the stronger model is what over-covers."""
    rng = np.random.default_rng(0)
    n = 40_000
    sigma = rng.uniform(0.05, 0.5, size=n)
    signal = rng.normal(scale=sigma * 10.0, size=n)

    # Strong model: explains most of the signal. Weak: explains little.
    strong_resid = signal - 0.9 * signal
    weak_resid = signal - 0.3 * signal

    q_strong = calibrate(strong_resid, sigma, ALPHA)
    q_weak = calibrate(weak_resid, sigma, ALPHA)
    assert q_weak > q_strong

    # Pooling the two and serving the strong model over-covers.
    q_pooled = calibrate(np.concatenate([strong_resid, weak_resid]), np.concatenate([sigma, sigma]), ALPHA)
    low, high = conformal.band(np.zeros(n), sigma, q_pooled)
    covered = float(((strong_resid >= low) & (strong_resid <= high)).mean())
    assert covered > conformal.NOMINAL_COVERAGE + 0.05


def _cv_source() -> str:
    """The CV method's own source. Read by member, not by splitting the file:
    `predict` is defined ~900 lines ABOVE the CV loop, so a
    `source.split("def predict(")` tail contains the CV code and any assertion
    over it is vacuous."""
    import inspect

    from models.forecaster import ItemForecaster

    return inspect.getsource(ItemForecaster._cv_evaluate_horizon)


def test_fold_q_hat_is_reported_per_fold():
    """A structural guard: the CV path must attach `fold_q_hat` to every fold
    record it emits, or the audit above cannot be run from an artifact."""
    source = _cv_source()
    assert 'fold_metrics[-1]["fold_q_hat"]' in source
    assert "fold_q_hat={fold_q_hat" in source  # logged, not only stored


def test_fold_q_hat_never_builds_a_band():
    """It is a diagnostic. `predict` reads `conformal_calibration`, and a fold
    number reaching the served band would be a silent scheme change."""
    import inspect

    from models.forecaster import ItemForecaster

    assert "fold_q_hat" not in inspect.getsource(ItemForecaster.predict)


def test_the_audit_is_reported_as_a_summary_not_only_per_fold():
    """Nine folds x four horizons is 36 log lines. The verdict has to be one
    grep and one `meta.json` key, or the dispatch gets read by eye."""
    import inspect

    from models.forecaster import ItemForecaster

    source = inspect.getsource(ItemForecaster._train_horizon_inline)
    assert '"q_hat_trend": q_hat_trend' in source
    assert "spearman_n_train_vs_q_hat" in source
    assert "pooled_over_last_fold" in source
    # A rho over two points is not a measurement.
    assert "len(fold_q_hats) >= 3" in source
    # Without these the rho is unreadable: see the test below.
    assert "n_train_distinct" in source
    assert "cv_max_train_rows" in source


def test_the_screens_own_limitation_is_recorded_beside_it():
    """`CV_MAX_TRAIN_ROWS` (300,000) binds on most folds, so the folds share an
    `n_train` and the rho has almost no x-axis. A rho near zero is then NOT
    evidence against the expanding-window hypothesis — it is a measurement with
    nothing to measure over. Anything that reports the rho must report the range
    too, or the next reader takes a null at face value."""
    import inspect

    from models.forecaster import ItemForecaster

    source = inspect.getsource(ItemForecaster._train_horizon_inline)
    assert "NOT a null" in source
    assert "CV_MAX_TRAIN_ROWS" in source

    # And the cap really does sit far below the served budget — which is what
    # makes the fold residuals non-exchangeable with the served ones in the
    # first place. If these ever converge, the hypothesis dissolves.
    from scripts.forecast_prices import DEFAULT_TRAIN_FEATURE_ROWS

    assert ItemForecaster.CV_MAX_TRAIN_ROWS * 3 < DEFAULT_TRAIN_FEATURE_ROWS


def test_the_trend_screen_recovers_a_planted_decline():
    """The screen's arithmetic, on a series with the hypothesised shape. Spearman
    over (n_train, q_hat) must be -1 when q_hat declines monotonically, and the
    pooled/last ratio must exceed 1 when the early folds are the wide ones."""
    fold_q_hats = [(87_224, 130.0), (150_000, 120.0), (220_000, 105.0), (300_000, 95.0)]
    n = pd.Series([a for a, _ in fold_q_hats], dtype=float)
    q = pd.Series([b for _, b in fold_q_hats], dtype=float)
    assert float(n.corr(q, method="spearman")) == -1.0

    pooled = float(np.quantile(np.concatenate([np.full(1000, v) for _, v in fold_q_hats]), 0.8))
    assert pooled / fold_q_hats[-1][1] > 1.0

    # And the null shape: no relation between training size and width leaves
    # rho near zero, which is the reading that kills the hypothesis.
    flat = pd.Series([110.0, 108.0, 112.0, 109.0])
    assert abs(float(n.corr(flat, method="spearman"))) < 1.0


def test_a_short_fold_reports_none_rather_than_a_fabricated_q_hat():
    """`MIN_CALIBRATION_ROWS` guards the pooled fit; the per-fold diagnostic has
    to honour the same floor or it publishes a quantile decided by a handful of
    tail draws — and a fold with no finite score must not take out the CV that
    produces the pooled calibration."""
    source = _cv_source()
    assert "len(fold_conformal) >= self.MIN_CALIBRATION_ROWS" in source
    assert "except ValueError" in source

    # The floor exists because `calibrate` is willing to be asked for a
    # quantile of nothing, and raises rather than returning a number.
    with pytest.raises(ValueError):
        calibrate(np.array([]), np.array([]), ALPHA)
