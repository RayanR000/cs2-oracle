"""The interval (Winkler) score: one proper score for the served 80% band.

Coverage and width, reported apart, let an arm "win" by trading one for the
other. The interval score prices the trade: width, plus 2/alpha times the
distance by which the outcome fell outside the band (Gneiting & Raftery 2007).
It is proper, so the true alpha/2 and 1-alpha/2 quantiles minimise it in
expectation, and lower is better.
"""

from __future__ import annotations

import numpy as np
import pytest
from backtest.scoring import interval_score, score_cohort

ALPHA = 0.2


def test_inside_the_band_scores_the_width_only():
    assert interval_score(-0.1, 0.1, 0.0, ALPHA) == pytest.approx(0.2)


def test_a_miss_below_adds_two_over_alpha_times_the_shortfall():
    # width 0.2 + (2 / 0.2) * (-0.1 - (-0.2)) = 0.2 + 1.0
    assert interval_score(-0.1, 0.1, -0.2, ALPHA) == pytest.approx(1.2)


def test_a_miss_above_is_penalised_symmetrically():
    assert interval_score(-0.1, 0.1, 0.2, ALPHA) == pytest.approx(1.2)


def test_the_band_edges_count_as_inside():
    assert interval_score(-0.1, 0.1, 0.1, ALPHA) == pytest.approx(0.2)
    assert interval_score(-0.1, 0.1, -0.1, ALPHA) == pytest.approx(0.2)


def test_vectorised_inputs_score_elementwise():
    got = interval_score(np.array([-0.1, -0.1]), np.array([0.1, 0.1]), np.array([0.0, 0.2]), ALPHA)
    np.testing.assert_allclose(got, [0.2, 1.2])


def test_the_true_quantile_band_beats_narrower_and_wider_bands():
    """Propriety, the reason to use it: an arm cannot lower its expected score by
    being uniformly narrower or wider than the true 10/90 quantiles."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    z = 1.2815515655446004  # N(0,1) 90th percentile
    true = interval_score(-z, z, y, ALPHA).mean()
    for k in (0.7, 0.85, 1.15, 1.3):
        assert interval_score(-k * z, k * z, y, ALPHA).mean() > true


def test_agrees_with_scoringrules():
    sr = pytest.importorskip("scoringrules")
    rng = np.random.default_rng(1)
    lo = rng.uniform(-0.3, 0.0, 500)
    hi = lo + rng.uniform(0.0, 0.5, 500)
    y = rng.normal(0.0, 0.2, 500)
    np.testing.assert_allclose(interval_score(lo, hi, y, ALPHA), np.asarray(sr.interval_score(y, lo, hi, ALPHA)))


def test_verdict_scores_the_band_in_return_space_off_the_quote():
    """Same basis as `in_interval`: the band is rebased onto the resolved base.
    quote 12 -> band returns [-10%, +15%]; base 10 -> actual 10.5 is +5%, inside,
    so the score is the 25pp width."""
    from scripts.backtest_accuracy import _derive_verdict

    v = _derive_verdict(10.0, 10.5, 12.6, 10.8, 13.8, "up", quote=12.0)
    assert v["interval_score_pct"] == pytest.approx(25.0)


def test_verdict_penalises_a_rebased_miss():
    """actual 13.0 is +30%, 15pp above the +15% edge: 25 + (2/0.2) * 15 = 175."""
    from scripts.backtest_accuracy import _derive_verdict

    v = _derive_verdict(10.0, 13.0, 12.6, 10.8, 13.8, "up", quote=12.0)
    assert v["interval_score_pct"] == pytest.approx(175.0)


def test_verdict_without_a_band_has_no_score():
    from scripts.backtest_accuracy import _derive_verdict

    v = _derive_verdict(10.0, 10.5, 10.6, None, None, "up", quote=12.0)
    assert v["interval_score_pct"] is None


def test_the_score_is_reporting_only_and_never_stored():
    from scripts.backtest_accuracy import _derive_verdict, _verdict_for_storage

    v = _derive_verdict(10.0, 10.5, 12.6, 10.8, 13.8, "up", quote=12.0)
    assert "interval_score_pct" not in _verdict_for_storage(v)


def _record(**overrides):
    base = {
        "abs_error": 0.10,
        "pct_error": 10.0,
        "sq_error": 0.01,
        "direction_correct": 1,
        "predicted_direction": "up",
        "actual_direction": "up",
        "in_interval": 1,
        "confidence": "high",
        "base_price": 1.00,
        "actual_price": 1.10,
        "price_tier": 1,
        "item_id": 1,
        "predicted_mid": 1.10,
        "horizon_days": 14,
    }
    base.update(overrides)
    return base


def test_cohort_reports_the_mean_score_over_scored_rows():
    records = [
        _record(interval_score_pct=20.0),
        _record(interval_score_pct=40.0),
        _record(interval_score_pct=None),
        _record(),
    ]
    metrics, _ = score_cohort(records)
    assert metrics["interval_score_pct"] == pytest.approx(30.0)


def test_cohort_without_any_scored_row_reports_none():
    metrics, _ = score_cohort([_record(), _record()])
    assert metrics["interval_score_pct"] is None
