"""The band-level decomposition's arithmetic, pinned on constructed panels.

`scripts/attribute_band_level.py` asks why the served band is 1.52-1.55x as wide as
the same `q_hat`'s calibration band. Its whole answer is one split — WHICH items are
served versus WHEN they are served — plus the observation that the band's coverage
responds to `|resid| / sigma` rather than to `sigma` alone. Both are asserted here
against panels whose answer is known by construction, never against the archive:
the archive read is the *input* to that arithmetic and moves week to week.

See `docs/changelog/2026-08-13-the-band-is-sized-on-trailing-volatility.md`.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest
from scripts.archive.attribute_band_level import (
    SIGMA_TRAILING_WINDOW_DAYS,
    anchor_coverage,
    date_levels,
    decompose,
    fit_level_elasticity,
    mean_abs_miss,
    overshoot_breaches,
    pooled_anchor_coverage,
    select_low_level_anchors,
    shuffled_levels,
)

ANCHOR = "2026-06-16"
OTHER = ["2026-06-09", "2026-06-10", "2026-06-11"]


def _panel(rows):
    """rows: (item_id, 'YYYY-MM-DD', sigma, |resid|)."""
    return pd.DataFrame([{"item_id": i, "date": pd.Timestamp(d).date(), "sigma": s, "absr": r} for i, d, s, r in rows])


def _uniform(items, dates, sigma, absr):
    return [(i, d, sigma, absr) for i in items for d in dates]


def test_a_date_effect_is_reported_on_both_bases():
    """Every item's sigma is 1.5x on the anchor and its residual unchanged. The
    cohort is identical on every date, so WHICH cannot contribute and both bases
    must report the same thing -- a sigma ratio of 1.5 and a score ratio of 1/1.5.
    """
    items = [f"item-{n}" for n in range(20)]
    panel = _panel(_uniform(items, OTHER, 0.08, 4.0) + _uniform(items, [ANCHOR], 0.12, 4.0))
    d = decompose(panel, [pd.Timestamp(ANCHOR).date()])
    assert d["same_sigma"] == pytest.approx(1.5)
    assert d["same_absr"] == pytest.approx(1.0)
    assert d["score_same"] == pytest.approx(1 / 1.5)
    assert d["pooled_sigma"] == pytest.approx(1.5)


def test_a_pure_composition_effect_vanishes_on_the_same_items_basis():
    """The distinguishing case, and the reason the second basis exists. Sigma is
    constant in time per item; the anchor simply serves the volatile half of the
    cohort. The pooled ratio must rise and the same-items ratio must be exactly 1
    -- otherwise 'which items' and 'when' are not separated and the instrument
    cannot tell a cohort artifact from a regime shift.
    """
    # The calm group is the majority on purpose: at a 50/50 split the pooled
    # median itself sits in the volatile half and the composition effect the test
    # is constructing would not show up in a median ratio at all.
    calm = [f"calm-{n}" for n in range(30)]
    wild = [f"wild-{n}" for n in range(10)]
    panel = _panel(_uniform(calm, [*OTHER, ANCHOR], 0.05, 2.0) + _uniform(wild, [*OTHER, ANCHOR], 0.20, 8.0))
    # Only the volatile items are quoted on the anchor.
    panel = panel[(panel["date"] != pd.Timestamp(ANCHOR).date()) | (panel["item_id"].str.startswith("wild"))]
    d = decompose(panel, [pd.Timestamp(ANCHOR).date()])
    assert d["same_sigma"] == pytest.approx(1.0)
    assert d["same_absr"] == pytest.approx(1.0)
    assert d["score_same"] == pytest.approx(1.0)
    assert d["pooled_sigma"] > 1.5


def test_a_matched_residual_leaves_the_score_alone():
    """The null the whole entry turns on. An anchor twice as volatile in BOTH legs
    is not a calibration defect at all -- the band is wider because the item is
    genuinely wilder, and coverage is unchanged. If `score_*` moved here, the
    measured 0.65-0.76 could not be read as over-coverage.
    """
    items = [f"item-{n}" for n in range(20)]
    panel = _panel(_uniform(items, OTHER, 0.08, 4.0) + _uniform(items, [ANCHOR], 0.16, 8.0))
    d = decompose(panel, [pd.Timestamp(ANCHOR).date()])
    assert d["same_sigma"] == pytest.approx(2.0)
    assert d["same_absr"] == pytest.approx(2.0)
    assert d["score_same"] == pytest.approx(1.0)


def test_an_item_served_but_never_pooled_elsewhere_does_not_inflate_the_ratio():
    """An item whose only row IS the anchor has itself as its own median, so its
    relative sigma is exactly 1 and it can neither create nor hide a date effect.
    Worth pinning: the served cohort at a real anchor contains such items, and a
    NaN or a division by a foreign median would silently bias the median ratio.
    """
    items = [f"item-{n}" for n in range(9)]
    panel = _panel(
        _uniform(items, OTHER, 0.08, 4.0) + _uniform(items, [ANCHOR], 0.16, 4.0) + [("newcomer", ANCHOR, 0.40, 20.0)]
    )
    d = decompose(panel, [pd.Timestamp(ANCHOR).date()])
    assert d["n_served"] == 10
    assert d["same_sigma"] == pytest.approx(2.0)  # the newcomer's own is 1.0
    assert np.isfinite(d["score_same"])


def test_no_served_row_returns_empty_rather_than_a_ratio_of_nothing():
    panel = _panel(_uniform(["a", "b"], OTHER, 0.08, 4.0))
    assert decompose(panel, [pd.Timestamp(ANCHOR).date()]) == {}


# --------------------------------------------------------------------------- #
# the date-level rescaling arm
# `docs/research/2026-08-13-date-level-sigma-rescaling-preregistration.md`
# --------------------------------------------------------------------------- #


def _level_panel(levels, item_factors, resid_tracks_level, eps=None):
    """A panel whose date effect is known by construction.

    `sigma[i,t] = levels[t] * item_factors[i]`, and `|resid[i,t]|` either carries
    the same `levels[t]` (forward dispersion repays trailing volatility, so
    production is already right and `b = 1`) or does not (the defect the arm
    targets, `b = 0`). `eps` is a per-item multiplier and is deterministic, so
    the fitted slope is exact rather than approximate.
    """
    eps = eps if eps is not None else [1.0] * len(item_factors)
    rows = []
    for _t, (d, lev) in enumerate(sorted(levels.items())):
        for i, (f, e) in enumerate(zip(item_factors, eps)):
            r = f * e * (lev if resid_tracks_level else 1.0)
            rows.append({"item_id": f"item-{i}", "date": d, "sigma": lev * f, "resid": r * (1 if i % 2 else -1)})
    return pd.DataFrame(rows)


def _levels_over(n_dates, values, start="2026-01-01"):
    days = [pd.Timestamp(start).date() + datetime.timedelta(days=k) for k in range(n_dates)]
    return {d: values[k % len(values)] for k, d in enumerate(days)}


def test_the_date_level_is_the_cross_section_median_of_sigma_on_that_date():
    """`L[t]` is a median over the panel's whole cohort, not over the served rows.
    Computing it on two different item sets in calibration and serving is the
    void condition the pre-registration names; the instrument therefore derives
    it once, from the frame, and every arm divides by the same series.
    """
    levels = _levels_over(4, [0.05, 0.15])
    panel = _level_panel(levels, [1.0, 2.0, 3.0], resid_tracks_level=False)
    got = date_levels(panel)
    assert list(got.index) == sorted(levels)
    assert got.to_numpy() == pytest.approx([0.10, 0.30, 0.10, 0.30])


def test_dispersion_that_tracks_the_trailing_level_one_for_one_fits_b_of_one():
    """The null, and the arm's void condition. If the forward residual rises with
    trailing volatility exactly as `sigma` does, the conformal score is already
    level-free, `gamma = 0`, and production's pooled `q_hat` is right.
    """
    levels = _levels_over(24, [0.04, 0.06, 0.09, 0.13])
    panel = _level_panel(levels, [1.0, 1.5, 2.0, 2.5], resid_tracks_level=True)
    fit = fit_level_elasticity(panel, n_boot=200)
    assert fit["b"] == pytest.approx(1.0)
    assert fit["gamma"] == pytest.approx(0.0)
    assert fit["ci_lo"] <= fit["b"] <= fit["ci_hi"]


def test_dispersion_flat_in_the_trailing_level_fits_b_of_zero():
    """The mechanism the changelog measured, in its pure form: trailing `sigma`
    swings by date and the forward move does not follow, so the whole level is
    uninformative and `gamma = 1`.
    """
    levels = _levels_over(24, [0.04, 0.06, 0.09, 0.13])
    panel = _level_panel(levels, [1.0, 1.5, 2.0, 2.5], resid_tracks_level=False)
    fit = fit_level_elasticity(panel, n_boot=200)
    assert fit["b"] == pytest.approx(0.0, abs=1e-9)
    assert fit["gamma"] == pytest.approx(1.0)
    assert fit["ci_hi"] < 1.0


def test_the_bootstrap_resamples_dates_so_a_noisy_fit_reports_a_wide_interval():
    """The CI is the void check -- an interval containing both 0 and 1 means the
    arm has no defensible `gamma`. It must therefore widen with the DATE count,
    which is the unit of variation, not with the row count: a panel of four dates
    and 900 items each carries four observations of the level, not 3,600.
    """
    rng = np.random.default_rng(11)
    levels = _levels_over(24, [0.04, 0.06, 0.09, 0.13])
    panel = _level_panel(levels, [1.0, 1.5, 2.0, 2.5], resid_tracks_level=False)
    # Per-DATE noise: shifts each date's whole cross-section, which is exactly
    # what a date-level fit cannot average away.
    noise = {d: float(rng.lognormal(0.0, 0.5)) for d in levels}
    panel["resid"] = panel["resid"] * panel["date"].map(noise)
    wide = fit_level_elasticity(panel, n_boot=400)
    tight = fit_level_elasticity(_level_panel(levels, [1.0, 1.5, 2.0, 2.5], resid_tracks_level=False), n_boot=400)
    assert (wide["ci_hi"] - wide["ci_lo"]) > (tight["ci_hi"] - tight["ci_lo"])


def _defect_panel(n_dates=80, n_items=60, seed=3):
    """A panel with the measured defect: `sigma`'s level doubles between dates
    and the residual it normalises does not follow.
    """
    rng = np.random.default_rng(seed)
    levels = _levels_over(n_dates, [0.05, 0.05, 0.10, 0.10])
    factors = list(1.0 + rng.random(n_items))
    eps = list(rng.lognormal(0.0, 0.6, n_items))
    return _level_panel(levels, factors, resid_tracks_level=False, eps=eps), levels


def test_a_gamma_of_zero_is_production_exactly():
    """`gamma = 0` must reproduce the control band bit for bit. Without this the
    arm's effect cannot be read as an effect: any difference in plumbing between
    the two paths would show up as a coverage move.
    """
    panel, _ = _defect_panel()
    anchors = sorted(panel["date"].unique())[-3:]
    control = anchor_coverage(panel, anchors, horizon=3)
    arm = anchor_coverage(panel, anchors, horizon=3, gamma=0.0)
    assert control == arm


def test_dividing_out_the_level_flattens_coverage_across_dates():
    """The claim, on a panel built to carry it. Two date levels, one residual
    distribution: the control over-covers on high-`sigma` dates and under-covers
    on low ones, and dividing the level out must collapse that spread.
    """
    panel, _ = _defect_panel()
    dates = sorted(panel["date"].unique())
    # One anchor of each level, both far enough from the panel's start to have a
    # calibration set under the H+13 embargo.
    anchors = [dates[-4], dates[-2]]
    control = anchor_coverage(panel, anchors, horizon=3)
    arm = anchor_coverage(panel, anchors, horizon=3, gamma=1.0)
    spread_control = max(control.values()) - min(control.values())
    spread_arm = max(arm.values()) - min(arm.values())
    assert spread_control > 0.10  # the defect is present to begin with
    assert spread_arm < spread_control / 2


def test_a_shuffled_level_does_not_flatten_the_same_panel():
    """The placebo's discriminating power, asserted on a panel where the real arm
    works. If a level drawn from the wrong date flattened coverage too, the arm
    would be `q_hat` re-absorbing a constant and the read would be worthless --
    so this test is what makes a passing placebo mean anything.
    """
    panel, _ = _defect_panel()
    dates = sorted(panel["date"].unique())
    anchors = [dates[-4], dates[-2]]
    real = anchor_coverage(panel, anchors, horizon=3, gamma=1.0)
    real_spread = max(real.values()) - min(real.values())

    levels = date_levels(panel)
    spreads = []
    for perm in shuffled_levels(levels, seed=20260813, n_perm=25):
        cov = anchor_coverage(panel, anchors, horizon=3, gamma=1.0, levels=perm)
        spreads.append(max(cov.values()) - min(cov.values()))
    assert float(np.median(spreads)) > real_spread


def test_the_pooled_coverage_is_row_weighted_not_an_average_of_anchors():
    """The placebo differences a MARGINAL coverage, so the pooling rule is part of
    the bar. Anchors carry unequal row counts, and averaging their rates equally
    would let a thin date swing the number that decides the read -- the same
    composition trap that cost nine published figures on 2026-08-11.
    """
    panel, _ = _defect_panel()
    dates = sorted(panel["date"].unique())
    anchors = [dates[-4], dates[-2]]
    # Thin one anchor out so equal-weight and row-weight cannot coincide.
    thin = panel[(panel["date"] != anchors[0]) | (panel["item_id"].isin({f"item-{n}" for n in range(5)}))]
    per = anchor_coverage(thin, anchors, horizon=3)
    n = thin[thin["date"].isin(anchors)].groupby("date").size()
    expected = float(sum(per[a] * n[a] for a in anchors) / sum(n[a] for a in anchors))
    pooled = pooled_anchor_coverage(thin, anchors, horizon=3)
    assert pooled == pytest.approx(expected)
    assert pooled != pytest.approx(float(np.mean(list(per.values()))))


# --------------------------------------------------------------------------- #
# the low-level anchor selection rule
# `docs/research/2026-08-13-low-level-anchor-preregistration.md`
# --------------------------------------------------------------------------- #


def _levels(pairs):
    return pd.Series({pd.Timestamp(d).date(): v for d, v in pairs}).sort_index()


def _all_eligible(levels, horizons=(3, 7, 14, 30)):
    return {d: set(horizons) for d in levels.index}


def _rows(levels, n=500):
    return pd.Series({d: n for d in levels.index})


def test_the_low_set_is_taken_in_ascending_level():
    """The rule is "lowest `L[t]` first", because the arm's untested direction is
    the one where it must RAISE the band, and dose is what separates it from noise."""
    lv = _levels([("2024-01-01", 0.09), ("2024-03-01", 0.02), ("2024-05-01", 0.05), ("2024-07-01", 0.04)])
    got = select_low_level_anchors(lv, _all_eligible(lv), _rows(lv), n=3, spacing_days=30, quantile=1.0)
    assert got == [datetime.date(2024, 3, 1), datetime.date(2024, 7, 1), datetime.date(2024, 5, 1)], got


def test_two_anchors_closer_than_the_spacing_cannot_both_be_taken():
    """Adjacent dates share a cross-section and a calibration pool, and at 30d
    they share an outcome window. Two of those are a replication, not evidence."""
    lv = _levels([("2024-03-01", 0.02), ("2024-03-05", 0.021), ("2024-06-01", 0.03)])
    got = select_low_level_anchors(lv, _all_eligible(lv), _rows(lv), n=3, spacing_days=30, quantile=1.0)
    assert got == [datetime.date(2024, 3, 1), datetime.date(2024, 6, 1)], got


def test_a_date_failing_either_audit_at_any_horizon_is_not_selected():
    """The 2026-08-13 lesson, applied at selection rather than after the read:
    the collection audit and the label-voiding detector must both pass, at every
    horizon the read scores, before an anchor set is fixed."""
    lv = _levels([("2024-03-01", 0.02), ("2024-06-01", 0.03)])
    elig = _all_eligible(lv)
    elig[datetime.date(2024, 3, 1)] = {3, 7, 14}  # 30d refused
    got = select_low_level_anchors(lv, elig, _rows(lv), n=2, spacing_days=30, quantile=1.0)
    assert got == [datetime.date(2024, 6, 1)], got


def test_a_thin_date_is_not_selected():
    """A coverage rate on 40 rows is not a coverage rate."""
    lv = _levels([("2024-03-01", 0.02), ("2024-06-01", 0.03)])
    rows = pd.Series({datetime.date(2024, 3, 1): 40, datetime.date(2024, 6, 1): 500})
    got = select_low_level_anchors(lv, _all_eligible(lv), rows, n=2, spacing_days=30, quantile=1.0)
    assert got == [datetime.date(2024, 6, 1)], got


def test_only_the_bottom_quantile_of_the_level_is_eligible():
    """ "Low" is defined against the panel's own distribution, fixed in advance --
    not "the lowest six whatever they are", which would select a set from an
    ordinary regime if the panel happened to hold no calm dates."""
    lv = _levels([(f"2024-{m:02d}-01", 0.02 + 0.01 * m) for m in range(1, 13)])
    got = select_low_level_anchors(lv, _all_eligible(lv), _rows(lv), n=6, spacing_days=1, quantile=0.25)
    assert len(got) == 3, got  # 3 of 12 dates sit at or below the p25
    assert max(lv[d] for d in got) <= lv.quantile(0.25)


def test_an_unsatisfiable_set_is_returned_short_rather_than_relaxed():
    """If the frame cannot supply `n` spaced eligible anchors, the caller gets
    what exists. Relaxing the rule to reach a count is how a set stops being
    the set that was pre-registered."""
    lv = _levels([("2024-03-01", 0.02), ("2024-03-02", 0.021)])
    got = select_low_level_anchors(lv, _all_eligible(lv), _rows(lv), n=6, spacing_days=30, quantile=1.0)
    assert got == [datetime.date(2024, 3, 1)], got


def test_restricting_the_candidates_does_not_move_what_low_means():
    """Leg A reads a sub-period. If "low" were re-derived inside it, a period with
    no calm dates would have its quietest ordinary ones relabelled — which is the
    selection-on-the-arm's-own-axis trap. The cut stays the panel's."""
    lv = _levels([(f"2024-{m:02d}-01", 0.02 + 0.01 * m) for m in range(1, 13)])
    late = [d for d in lv.index if d.month >= 7]  # all above the panel p25
    assert select_low_level_anchors(lv, _all_eligible(lv), _rows(lv), candidates=late, n=6, spacing_days=1) == []


def test_the_trailing_windows_warm_up_is_not_a_calm_date():
    """`sigma` is `price_std_60d / price`, so the panel's first 60 days carry a
    window that is not yet 60 days long. On the real panel 2024-07-09 is 100%
    pinned at the sigma clip floor and has the lowest `L[t]` by a factor of two —
    it would have been the largest dose in the set, and it measures the warm-up."""
    lv = _levels([("2024-07-09", 0.002), ("2024-09-21", 0.05)])
    got = select_low_level_anchors(
        lv,
        _all_eligible(lv),
        _rows(lv),
        n=2,
        spacing_days=30,
        quantile=1.0,
        not_before=datetime.date(2024, 7, 9) + datetime.timedelta(days=SIGMA_TRAILING_WINDOW_DAYS),
    )
    assert got == [datetime.date(2024, 9, 21)], got


def test_the_warm_up_window_is_the_feature_s_own():
    """Pinned against the rolling feature it is derived from, so a change to the
    feature set cannot leave this constant silently describing nothing."""
    from models.forecaster import ItemForecaster

    assert f"price_std_{SIGMA_TRAILING_WINDOW_DAYS}d" in ItemForecaster._DOLLAR_SCALE_FEATURES


def test_the_miss_statistic_weights_each_date_once():
    """(L2) asks how far a TYPICAL DATE sits from target, so a date is one
    observation of it — unlike the marginal rate, which is row-weighted."""
    assert mean_abs_miss({1: 0.70, 2: 0.90}) == pytest.approx(10.0)
    assert mean_abs_miss({1: 0.80}) == pytest.approx(0.0)
    assert np.isnan(mean_abs_miss({}))


def test_an_overshoot_in_either_direction_is_a_breach():
    """The high-vol read overshot DOWNWARD at this gamma; on a calm date the same
    arithmetic inflates instead. A mean would net the two out, so (L3) is a
    per-anchor guard and symmetric."""
    control = {"a": 0.82, "b": 0.78, "c": 0.60}
    arm = {"a": 0.93, "b": 0.68, "c": 0.95}
    got = {a for a, _, _ in overshoot_breaches(control, arm)}
    assert got == {"a", "b"}, got  # "c" was already far out: not thrown there


def test_an_anchor_the_arm_never_scored_is_not_a_breach():
    assert overshoot_breaches({"a": 0.80}, {}) == []
