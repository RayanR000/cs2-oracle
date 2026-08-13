"""The serving replay: score predict() against outcomes the archive already has.

Walk-forward CV scores fold predictions. `_recenter_on_direction`, the tier
bias, the prior-day blend and the conformal band run inside `predict()` and
nowhere else, so nothing measured them without publishing a forecast and
waiting. REPLAY_ANCHOR rewinds the serving clock instead.

The tests that matter are the leak tests: a replay that can see past its own
anchor scores a forecast against data it already had.
"""
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster
from scripts.replay_serving import (COVERAGE_HEADER, DOLLAR_HEADER,
                                    PINNED_MAX_SPAN_DAYS,
                                    PINNED_SMOOTH_WINDOW,
                                    audit_anchor_feed, _feed_profile,
                                    cutovers_from_counts,
                                    cutovers_in_outcome_window,
                                    _basis_frame, _coverage_line,
                                    _coverage_by_sigma_rows, _sigma_line,
                                    SIGMA_HEADER, sigma_tilt_pp,
                                    _coverage_row, _dollar_error_rows,
                                    _dollar_line, _exact_day, _naive_baseline,
                                    _pin_matches_production, _pinned_anchor,
                                    _pinned_rank_ic, _rel_abs_error,
                                    _requested_horizons, _served_rows,
                                    _tied_mask)


def _fc():
    return ItemForecaster.__new__(ItemForecaster)


def _history(rows):
    """rows: (item_id, 'YYYY-MM-DD', price), in the `_outcomes` shape."""
    return pd.DataFrame(
        [{"item_id": i, "day": pd.Timestamp(d), "price": p} for i, d, p in rows]
    )


def test_no_anchor_is_the_live_path(monkeypatch):
    monkeypatch.delenv("REPLAY_ANCHOR", raising=False)
    assert ItemForecaster.replay_anchor() is None

    fc = ItemForecaster.__new__(ItemForecaster)
    before = datetime.now(timezone.utc)
    now = fc._now()
    assert (now - before).total_seconds() < 5
    assert now.tzinfo is not None


def test_anchor_rewinds_the_clock(monkeypatch):
    monkeypatch.setenv("REPLAY_ANCHOR", "2026-06-01")
    assert ItemForecaster.replay_anchor() == date(2026, 6, 1)

    fc = ItemForecaster.__new__(ItemForecaster)
    assert fc._now() == datetime(2026, 6, 1, tzinfo=timezone.utc)


def test_an_unparseable_anchor_raises_rather_than_falling_back(monkeypatch):
    """Silently ignoring it would replay against today and score the forecast
    on its own answer -- a pass, and a meaningless one."""
    monkeypatch.setenv("REPLAY_ANCHOR", "june 1st")
    with pytest.raises(ValueError, match="ISO date"):
        ItemForecaster.replay_anchor()


def test_the_archive_read_is_upper_bounded(monkeypatch):
    """The leak test. `WHERE day >= cutoff` alone lets a rewound run read every
    row after its anchor, so the model would forecast with the answer in hand.
    """
    import inspect
    src = inspect.getsource(ItemForecaster._fetch_voted_price_history)
    assert "day <= " in src, "the archive read has no upper bound"
    assert "{_upper_bound}" in src
    # No anchor must mean "no upper bound", not "no rows".
    assert 'if _anchor else ""' in src


def test_the_voted_cache_key_sees_the_anchor(monkeypatch):
    """Otherwise the replay restores the live frame -- which holds exactly the
    rows the upper bound exists to exclude."""
    fc = ItemForecaster.__new__(ItemForecaster)
    fc.VOTED_CACHE_VERSION = 6
    monkeypatch.setattr(ItemForecaster, "_archive_fingerprint",
                        lambda self: "fixed")

    monkeypatch.delenv("REPLAY_ANCHOR", raising=False)
    live = fc._voted_cache_key(730, True, None)
    monkeypatch.setenv("REPLAY_ANCHOR", "2026-06-01")
    replay = fc._voted_cache_key(730, True, None)
    assert live != replay


def test_the_prior_day_blend_cannot_read_past_the_anchor():
    import inspect
    src = inspect.getsource(ItemForecaster._fetch_prior_forecasts)
    # Comments mention date.today(); the code must not call it.
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    assert "self._now().date()" in code
    assert "date.today()" not in code, (
        "the blend would pull real forecasts dated after the anchor and mix "
        "the future into a backdated prediction"
    )


def test_no_disable_knob_is_the_full_serving_path(monkeypatch):
    monkeypatch.delenv("REPLAY_DISABLE", raising=False)
    assert ItemForecaster.replay_disabled() == frozenset()


def test_the_disable_knob_is_refused_without_an_anchor(monkeypatch):
    """It exists to attribute a replay's loss. Honoured on the live path it
    would change what production serves, and 'silently ignored' looks exactly
    like 'applied' in a log."""
    monkeypatch.delenv("REPLAY_ANCHOR", raising=False)
    monkeypatch.setenv("REPLAY_DISABLE", "recenter")
    with pytest.raises(ValueError, match="without REPLAY_ANCHOR"):
        ItemForecaster.replay_disabled()


def test_an_unknown_transform_name_raises(monkeypatch):
    """A typo would otherwise read as a clean control and measure nothing."""
    monkeypatch.setenv("REPLAY_ANCHOR", "2026-06-01")
    monkeypatch.setenv("REPLAY_DISABLE", "recentre")   # British spelling
    with pytest.raises(ValueError, match="not a serving transform"):
        ItemForecaster.replay_disabled()


def test_the_disable_knob_parses_under_an_anchor(monkeypatch):
    monkeypatch.setenv("REPLAY_ANCHOR", "2026-06-01")
    monkeypatch.setenv("REPLAY_DISABLE", "blend, recenter")
    assert ItemForecaster.replay_disabled() == frozenset({"blend", "recenter"})


def test_the_band_is_not_disablable():
    """It sets low and high around the mid, so it cannot move the mid's
    cross-sectional ranking. Offering the knob would imply otherwise."""
    assert "band" not in ItemForecaster.REPLAY_DISABLABLE
    assert ItemForecaster.REPLAY_DISABLABLE == {"blend", "bias", "recenter"}


def test_each_transform_is_actually_guarded():
    """The knob has to reach all three call sites, not just parse."""
    import inspect
    src = inspect.getsource(ItemForecaster.predict)
    assert '"blend" not in _disabled' in src
    assert '"bias" in _disabled' in src
    assert '"recenter" not in _disabled' in src


def test_no_horizon_filter_scores_everything_the_artifact_serves():
    assert _requested_horizons(["scripts/replay_serving.py"]) is None


def test_the_horizon_filter_exists_for_the_one_horizon_a_matrix_job_trained():
    """model-diagnostics.yml trains ONE horizon per job beside restored
    production boosters, and a feature transform applies to the whole frame --
    so the other three rows are boosters fitted on untransformed features being
    fed transformed ones. Scoring them would publish garbage in the same table.
    """
    assert _requested_horizons(["x", "--horizons", "3"]) == {3}
    assert _requested_horizons(["x", "--horizons", "3,30"]) == {3, 30}


def test_the_naive_baseline_is_built_on_the_served_basis():
    """It is scored against a return whose denominator is `predict`'s
    span-bounded median. Built from raw quotes instead, a one-day jump moves the
    baseline and barely moves the denominator, and the outcome carries the jump:
    at anchor 2026-06-01 that read -0.6187 rank IC at h=3 against a CV baseline
    of about +0.19. The median of (10, 10, 16) is 10, so a raw build would score
    this item at -0.60 and the served build at 0.0.
    """
    anchor = date(2026, 6, 1)
    hist = _history([
        ("jump", "2026-05-30", 10.0),
        ("jump", "2026-05-31", 10.0),
        ("jump", "2026-06-01", 16.0),
        # Its outcome, which the baseline must not read.
        ("jump", "2026-06-04", 30.0),
    ])
    out = _naive_baseline(_fc(), hist, anchor)
    assert out.loc[out["item_id"] == "jump", "naive"].iloc[0] == pytest.approx(0.0)


def test_the_naive_baseline_cannot_read_past_the_anchor():
    """`_smoothed_anchor_prices` falls back to `last()` over the WHOLE frame for
    an item with nothing inside the span window -- and the frame it is handed
    here spans the outcome. Truncating is what stops that."""
    anchor = date(2026, 6, 1)
    hist = _history([
        # Both quotes older than MAX_WINDOW_SPAN_DAYS, so both legs fall back.
        ("stale", "2026-04-01", 10.0),
        ("stale", "2026-04-02", 10.0),
        ("stale", "2026-07-15", 90.0),   # post-anchor: the leak
    ])
    out = _naive_baseline(_fc(), hist, anchor)
    assert out.loc[out["item_id"] == "stale", "naive"].iloc[0] == pytest.approx(0.0), (
        "the baseline saw the post-anchor quote through the staleness fallback"
    )


def test_the_naive_baseline_signs_a_move_as_reversal():
    """-return_1d: an item that rose is predicted to fall."""
    anchor = date(2026, 6, 1)
    hist = _history([
        ("up", "2026-05-31", 10.0),
        ("up", "2026-06-01", 11.0),
        ("down", "2026-05-31", 10.0),
        ("down", "2026-06-01", 9.0),
    ])
    out = _naive_baseline(_fc(), hist, anchor).set_index("item_id")["naive"]
    assert out["up"] < 0 < out["down"]


def test_the_cv_basis_drops_an_item_with_no_observation_that_day():
    """`prepare_targets` joins on `date + horizon` exactly. Applying the
    replay's tolerance here would erase the very difference being measured."""
    hist = _history([("gappy", "2026-06-01", 10.0),
                     ("gappy", "2026-06-05", 12.0)])   # nothing on 06-04
    assert _exact_day(hist, date(2026, 6, 4)).empty
    assert _exact_day(hist, date(2026, 6, 5))["px"].iloc[0] == pytest.approx(12.0)


def test_the_four_bases_cross_two_axes():
    """served = median outcome / smoothed anchor; cv = raw / raw. The mixed
    pair changes exactly one leg each, which is what makes a difference
    attributable to a numerator or a denominator rather than to 'the basis'."""
    anchor = date(2026, 6, 1)
    hist = _history([
        # Anchor window: raw price on 06-01 is 12, the 3-obs median is 10.
        ("a", "2026-05-30", 10.0),
        ("a", "2026-05-31", 10.0),
        ("a", "2026-06-01", 12.0),
        # Outcome: raw price on 06-04 is 20, the trailing median over
        # (06-01, 06-04] is 15.
        ("a", "2026-06-03", 15.0),
        ("a", "2026-06-04", 20.0),
    ])
    f = _basis_frame(_fc(), hist, anchor, horizon=3).set_index("item_id").loc["a"]
    assert f["anchor_raw"] == pytest.approx(12.0)
    assert f["anchor_smooth"] == pytest.approx(10.0)
    assert f["out_raw"] == pytest.approx(20.0)
    assert f["out_med"] == pytest.approx(17.5)   # median(15, 20)
    assert f["served"] == pytest.approx(17.5 / 10.0 - 1)
    assert f["cv"] == pytest.approx(20.0 / 12.0 - 1)
    assert f["num_only"] == pytest.approx(20.0 / 10.0 - 1)
    assert f["den_only"] == pytest.approx(17.5 / 12.0 - 1)
    assert not f["anchor_is_tied"]          # raw 12 vs median 10


def test_a_tied_anchor_makes_cv_and_served_share_a_denominator():
    """The decisive split: where the anchor quote equals its own local median
    there is no deviation for a feature to read, so the hypothesised
    shared-quote channel cannot operate and `served` differs from `cv` only by
    the outcome leg."""
    anchor = date(2026, 6, 1)
    hist = _history([
        ("flat", "2026-05-30", 10.0),
        ("flat", "2026-05-31", 10.0),
        ("flat", "2026-06-01", 10.0),      # raw == median == 10
        ("flat", "2026-06-04", 20.0),
    ])
    f = _basis_frame(_fc(), hist, anchor, horizon=3).set_index("item_id").loc["flat"]
    assert f["anchor_is_tied"]
    assert f["anchor_raw"] == pytest.approx(f["anchor_smooth"])
    assert f["served"] == pytest.approx(f["den_only"])


def test_a_replay_ignores_the_engineered_cache():
    """The other cache leak. Its key fingerprints forecaster.py, which a replay
    does not change, so a live cache is a valid hit carrying post-anchor rows.
    """
    import inspect
    src = inspect.getsource(ItemForecaster._load_engineered_cache)
    head = src.split("path = self._engineered_cache_path")[0]
    assert "replay_anchor() is not None" in head
    assert "return None" in head


# ---------------------------------------------------------------------------
# The pinned denominator (task 1 of the serving-anchor freshness plan).
#
# The next experiment changes WHAT PRICE predict() quotes from. `_basis_frame`
# built its `anchor_smooth` by calling `fc._smoothed_anchor_prices`, and the
# main loop divides both the prediction and the realised outcome by the served
# `current_price` -- so an arm that moves the serving anchor moves the referee
# with it, and a rank IC compared across arms measures nothing. These tests fix
# the referee in place. See docs/superpowers/plans/2026-08-11-serving-anchor-freshness.md.
# ---------------------------------------------------------------------------


def _anchor_window_history():
    """Raw 12 on the anchor, 3-observation median 10, one stale item."""
    return _history([
        ("a", "2026-05-30", 10.0),
        ("a", "2026-05-31", 10.0),
        ("a", "2026-06-01", 12.0),
        # Nothing within MAX_WINDOW_SPAN_DAYS of the anchor.
        ("old", "2026-04-01", 99.0),
    ])


def test_the_pinned_denominator_reproduces_the_shipped_one_today(monkeypatch):
    """It has to be the SAME number before it can be a fixed reference. If this
    fails, the pin changed the basis rather than freezing it, and every stored
    replay number becomes incomparable to the next one.

    The shipped constants are pinned to their defaults for the comparison, on
    purpose: `MAX_WINDOW_SPAN_DAYS` is `FALLBACK_MAX_AGE_DAYS`, which reads the
    environment at import, so without this the test would pass or fail
    according to a variable set outside the repo. The runtime divergence check
    in `main()` is what reports that case, not this test.
    """
    import backtest.price_resolution as pr
    monkeypatch.setattr(pr, "SMOOTH_WINDOW", 3, raising=False)
    monkeypatch.setattr(pr, "MAX_WINDOW_SPAN_DAYS", 7, raising=False)

    hist = _anchor_window_history()
    anchor = date(2026, 6, 1)

    pinned = _pinned_anchor(hist, anchor)
    shipped = ItemForecaster._smoothed_anchor_prices(
        hist.rename(columns={"day": "date"}), pd.Timestamp(anchor))

    assert set(pinned.index) == set(shipped)
    for item, value in shipped.items():
        assert pinned[item] == pytest.approx(value), item


def test_the_pinned_denominator_does_not_follow_the_serving_path(monkeypatch):
    """The point of the pin. An arm that changes what predict() quotes from
    must not be able to change the number it is scored against."""
    hist = _anchor_window_history()
    anchor = date(2026, 6, 1)
    before = _pinned_anchor(hist, anchor)

    monkeypatch.setattr(ItemForecaster, "_smoothed_anchor_prices",
                        staticmethod(lambda df, at: {"a": 1.0, "old": 1.0}))
    after = _pinned_anchor(hist, anchor)

    assert after["a"] == pytest.approx(before["a"])
    assert after["a"] == pytest.approx(10.0)


def test_the_pinned_denominator_does_not_follow_the_window_constants(monkeypatch):
    """Arm B of the plan moves SMOOTH_WINDOW / MAX_WINDOW_SPAN_DAYS, which are
    shared with the backtest resolver. The referee's constants are literals in
    replay_serving for exactly this reason."""
    import backtest.price_resolution as pr
    import models.forecaster as fcmod

    hist = _anchor_window_history()
    anchor = date(2026, 6, 1)
    before = _pinned_anchor(hist, anchor)

    monkeypatch.setattr(pr, "SMOOTH_WINDOW", 1, raising=False)
    monkeypatch.setattr(pr, "MAX_WINDOW_SPAN_DAYS", 1, raising=False)
    monkeypatch.setattr(fcmod, "SMOOTH_WINDOW", 1, raising=False)
    monkeypatch.setattr(fcmod, "MAX_WINDOW_SPAN_DAYS", 1, raising=False)

    assert _pinned_anchor(hist, anchor)["a"] == pytest.approx(before["a"])


def test_the_pinned_denominator_bounds_its_window_in_calendar_days():
    """An item with nothing near the anchor keeps its latest observation rather
    than anchoring on prices months apart -- the shipped fallback, which exists
    so serving never manufactures a price and never drops an item."""
    pinned = _pinned_anchor(_anchor_window_history(), date(2026, 6, 1))
    assert pinned["old"] == pytest.approx(99.0)


def test_the_pinned_denominator_reads_nothing_after_the_anchor():
    hist = _history([
        ("a", "2026-05-31", 10.0),
        ("a", "2026-06-01", 10.0),
        ("a", "2026-06-02", 500.0),      # after the anchor
    ])
    assert _pinned_anchor(hist, date(2026, 6, 1))["a"] == pytest.approx(10.0)


def test_the_basis_frame_uses_the_pinned_anchor(monkeypatch):
    """`_basis_frame` carries the tied/deviating split every 2026-08-11 result
    rests on. If an arm could move `anchor_smooth`, it would move the
    definition of `tied` along with it and the split would stop being a
    constant across arms."""
    monkeypatch.setattr(ItemForecaster, "_smoothed_anchor_prices",
                        staticmethod(lambda df, at: {"a": 1.0}))
    hist = _history([
        ("a", "2026-05-30", 10.0),
        ("a", "2026-05-31", 10.0),
        ("a", "2026-06-01", 12.0),
        ("a", "2026-06-04", 20.0),
    ])
    f = _basis_frame(_fc(), hist, date(2026, 6, 1), horizon=3) \
        .set_index("item_id").loc["a"]
    assert f["anchor_smooth"] == pytest.approx(10.0)
    assert not f["anchor_is_tied"]


def test_the_pinned_rank_ic_is_scored_on_one_denominator():
    """Both legs divide by the pinned anchor, so two arms quoting different
    prices are still measured against the same yardstick. Dividing the
    prediction by the arm's own `current` and the outcome by the pinned anchor
    would compare a return to a different return."""
    frame = pd.DataFrame({
        "item_id": ["a", "b", "c"],
        "current": [12.0, 20.0, 5.0],      # what the arm quoted
        "mid": [13.0, 19.0, 6.0],
        "realised": [14.0, 18.0, 7.0],
    })
    pinned = pd.Series({"a": 10.0, "b": 20.0, "c": 4.0}, name="d_fixed")

    ic = _pinned_rank_ic(frame, pinned)
    # Ranks of mid/d_fixed against realised/d_fixed: a 1.30/1.40, b 0.95/0.90,
    # c 1.50/1.75 -- concordant, so a perfect +1.
    assert ic == pytest.approx(1.0)


def test_the_pinned_rank_ic_ignores_items_it_has_no_denominator_for():
    frame = pd.DataFrame({
        "item_id": ["a", "b", "c", "d"],
        "current": [10.0, 10.0, 10.0, 10.0],
        "mid": [11.0, 9.0, 12.0, 99.0],
        "realised": [12.0, 8.0, 13.0, 1.0],
    })
    pinned = pd.Series({"a": 10.0, "b": 10.0, "c": 10.0}, name="d_fixed")
    assert _pinned_rank_ic(frame, pinned) == pytest.approx(1.0)


def test_a_divergence_between_the_pin_and_production_is_reported(monkeypatch, capsys):
    """FALLBACK_MAX_AGE_DAYS is read from the environment at import, so an
    operator can move production's smoothing span without touching this repo.
    The pin must NOT follow it -- that is the point -- but a run that scores
    against a denominator production no longer uses has to say so, or the
    numbers look citable and are not.
    """
    import backtest.price_resolution as pr
    monkeypatch.setattr(pr, "MAX_WINDOW_SPAN_DAYS", 14, raising=False)
    assert _pin_matches_production() is False

    monkeypatch.setattr(pr, "MAX_WINDOW_SPAN_DAYS", PINNED_MAX_SPAN_DAYS,
                        raising=False)
    monkeypatch.setattr(pr, "SMOOTH_WINDOW", PINNED_SMOOTH_WINDOW, raising=False)
    assert _pin_matches_production() is True


# ---------------------------------------------------------------------------
# Dollar error (task 2 of the serving-anchor freshness plan).
#
# Rank IC cannot referee an arm that changes what price predict() quotes from:
# `main()` divides both the prediction and the outcome by the served
# `current_price`, so the arm moves the label and the prediction together. The
# gate is the median absolute dollar error over the DEVIATING cohort, which is
# basis-free because its denominator is the realised price -- a quantity no arm
# touches. See docs/superpowers/plans/2026-08-11-serving-anchor-freshness.md.
# ---------------------------------------------------------------------------


def test_the_dollar_error_divides_by_the_realised_price():
    """The whole reason this metric can compare two arms. Dividing by the served
    quote instead would make it basis-relative, exactly like rank IC."""
    err = _rel_abs_error(np.array([11.0, 8.0, 11.0, 8.0]), np.full(4, 10.0))
    assert err[0] == pytest.approx(0.15)          # median(0.1, 0.2, 0.1, 0.2)


def test_the_dollar_error_reports_a_p90_beside_the_median():
    """A quoting change can leave the median alone and move the tail: the
    smoothed anchor and the raw quote diverge by a median 3.6% and a p90 35%."""
    pred = np.array([10.0] * 8 + [20.0, 20.0])
    med, p90 = _rel_abs_error(pred, np.full(10, 10.0))
    assert med == pytest.approx(0.0)
    assert p90 == pytest.approx(1.0)


def test_the_dollar_error_drops_a_non_positive_realised_price():
    """An unresolved or zero outcome divides to infinity, which would swallow
    the median of the whole cohort rather than dropping one item."""
    pred = np.array([11.0, 99.0, 12.0, 9.0])
    realised = np.array([10.0, 0.0, 10.0, 10.0])
    med, _ = _rel_abs_error(pred, realised)
    # median(0.1, 0.2, 0.1). Keeping the inf row would read 0.15 instead.
    assert med == pytest.approx(0.1)


def test_the_dollar_error_needs_three_items_to_report():
    med, p90 = _rel_abs_error(np.array([11.0, 12.0]), np.array([10.0, 10.0]))
    assert np.isnan(med) and np.isnan(p90)


def _error_frame():
    """Two tied items and two deviating ones, with a known error per item."""
    return pd.DataFrame({
        "item_id": ["t1", "t2", "t3", "d1", "d2", "d3"],
        "current": [10.0, 10.0, 10.0, 12.0, 12.0, 12.0],
        "mid": [11.0, 9.0, 11.0, 13.0, 11.0, 13.0],
        "realised": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "naive": [0.10, -0.10, 0.10, 0.20, -0.20, 0.20],
    })


def _error_pinned():
    return pd.Series({"t1": 10.0, "t2": 10.0, "t3": 10.0,
                      "d1": 10.0, "d2": 10.0, "d3": 10.0}, name="d_fixed")


def test_the_dollar_error_splits_on_the_tied_cohort():
    """The gate reads the deviating cohort alone. On the tied cohort an arm that
    only chooses between the raw quote and its own median serves an identical
    price, so a pooled number dilutes the effect with rows that cannot move."""
    tied = pd.Series({"t1": True, "t2": True, "t3": True,
                      "d1": False, "d2": False, "d3": False})
    rows = {r["subset"]: r for r in
            _dollar_error_rows(_error_frame(), _error_pinned(), tied)}
    assert [r for r in rows] == ["pooled", "tied", "deviating"]
    assert rows["pooled"]["n"] == 6
    assert rows["tied"]["n"] == 3
    assert rows["deviating"]["n"] == 3
    assert rows["tied"]["model"] == pytest.approx(0.10)
    assert rows["deviating"]["model"] == pytest.approx(0.30)


def test_an_item_with_no_anchor_observation_is_not_tied():
    """`np.isclose(nan, nan)` is False and the mask is a left join, so "never
    observed at the anchor" must fall on the deviating side rather than
    defaulting into the clean cohort the durable result rests on."""
    tied = pd.Series({"t1": True, "t2": True, "t3": True, "d1": False})
    rows = {r["subset"]: r for r in
            _dollar_error_rows(_error_frame(), _error_pinned(), tied)}
    assert rows["deviating"]["n"] == 3           # d1 plus the two unlabelled


def test_the_quote_control_is_the_no_change_prediction():
    """The reference an arm is actually claiming to beat: "the price we publish
    is closer to what happens". It is the served quote, not the model's mid."""
    tied = pd.Series({"t1": True, "t2": True, "t3": True,
                      "d1": False, "d2": False, "d3": False})
    rows = {r["subset"]: r for r in
            _dollar_error_rows(_error_frame(), _error_pinned(), tied)}
    assert rows["tied"]["quote"] == pytest.approx(0.0)     # current 10, realised 10
    assert rows["deviating"]["quote"] == pytest.approx(0.2)  # current 12


def test_the_naive_dollar_prediction_does_not_follow_the_served_quote():
    """`naive` is a return, so it needs a base to become dollars. Taking the
    arm's own `current_price` would move the baseline whenever the arm moved,
    and "the model beat naive" would shift for a reason that is not the model.
    """
    tied = pd.Series({"t1": True, "t2": True, "t3": True,
                      "d1": False, "d2": False, "d3": False})
    frame = _error_frame()
    before = {r["subset"]: r["naive"] for r in
              _dollar_error_rows(frame, _error_pinned(), tied)}

    moved = frame.copy()
    moved["current"] = moved["current"] * 1.5      # the arm quotes elsewhere
    after = {r["subset"]: r["naive"] for r in
             _dollar_error_rows(moved, _error_pinned(), tied)}

    assert after["deviating"] == pytest.approx(before["deviating"])
    # pinned 10 x (1 + 0.20) = 12 against realised 10.
    assert before["deviating"] == pytest.approx(0.2)


def test_the_naive_column_is_scored_on_the_items_that_have_one():
    """`_naive_baseline` drops an item with no prior observation, and the merge
    leaves NaN. Counting those as zero error would flatter the baseline."""
    tied = pd.Series({"t1": True, "t2": True, "t3": True,
                      "d1": False, "d2": False, "d3": False})
    frame = _error_frame()
    frame.loc[frame["item_id"] == "d2", "naive"] = np.nan
    rows = {r["subset"]: r for r in
            _dollar_error_rows(frame, _error_pinned(), tied)}
    assert rows["deviating"]["n_naive"] == 2
    assert rows["deviating"]["n"] == 3


def test_the_workflow_summary_captures_the_dollar_table():
    """`model-diagnostics.yml` greps the replay log into the step summary, and
    the gate metric has to survive that filter.

    Worse than absent: the dollar rows are `h` then a word, the same shape as the
    basis sweep's tied/deviating rows, so a pattern that takes the rows but not
    the `DOLLAR ERROR` heading interleaves two tables into one unlabelled block.
    """
    import re
    from pathlib import Path

    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "model-diagnostics.yml").read_text()
    m = re.search(r'grep -E "([^"]+)" \\\n\s*replay-', wf)
    assert m, "the replay summary grep moved; this test no longer reads it"
    pattern = m.group(1)

    row = {"subset": "deviating", "n": 300, "model": 0.0412, "p90": 0.183,
           "quote": 0.0405, "naive": 0.0431, "n_naive": 298}
    for line in (f"DOLLAR ERROR @ 2026-06-01   (|x - realised| / realised, %)",
                 DOLLAR_HEADER,
                 _dollar_line(3, row)):
        assert re.search(pattern, line), line


def test_a_cohort_too_thin_to_score_still_prints_a_row():
    """`_rel_abs_error` returns NaN below three items, and a `%` format on NaN
    is the kind of thing that raises at the end of an hour of CI. The row has to
    survive it: an anchor whose tied cell is 26 items is a real case
    (2026-07-09), and its `n` is what tells you to discount it."""
    row = {"subset": "tied", "n": 2, "model": float("nan"),
           "p90": float("nan"), "quote": 0.05, "naive": float("nan"),
           "n_naive": 0}
    line = _dollar_line(7, row)
    assert "tied" in line and "nan" in line
    assert "5.00" in line          # the one column that resolved


def test_the_dollar_line_reports_percentages():
    row = {"subset": "deviating", "n": 300, "model": 0.0412, "p90": 0.183,
           "quote": 0.0405, "naive": 0.0431, "n_naive": 298}
    line = _dollar_line(14, row)
    assert "4.12" in line and "18.30" in line and "4.05" in line


def test_the_tied_mask_is_the_basis_frames_own_split():
    """One definition of `tied`, so the cohort the dollar gate reads is the same
    population as the tied/deviating rows of the basis sweep. Two copies of the
    isclose test would drift and the two tables would silently disagree."""
    anchor = date(2026, 6, 1)
    hist = _history([
        ("flat", "2026-05-30", 10.0),
        ("flat", "2026-05-31", 10.0),
        ("flat", "2026-06-01", 10.0),
        ("spiky", "2026-05-30", 10.0),
        ("spiky", "2026-05-31", 10.0),
        ("spiky", "2026-06-01", 12.0),
        ("gappy", "2026-05-25", 10.0),      # no observation on the anchor
    ])
    mask = _tied_mask(hist, anchor)
    assert bool(mask["flat"]) is True
    assert bool(mask["spiky"]) is False
    assert bool(mask["gappy"]) is False

    bases = _basis_frame(_fc(), hist, anchor, horizon=3).set_index("item_id")
    for item in ("flat", "spiky", "gappy"):
        assert bool(bases.loc[item, "anchor_is_tied"]) == bool(mask[item]), item


# ---------------------------------------------------------------------------
# F1: the band's coverage, which nothing here measured
#
# `predict` publishes low/mid/high; the replay scored only the mid. q_hat is
# calibrated to cover 80% of residuals to the q50 mid, and then
# `_recenter_on_direction` moves the centre out from under it — so served
# coverage is an empirical question, and this is the only vehicle that can ask
# it without publishing a forecast and waiting.
# ---------------------------------------------------------------------------


def _served_payload(rows):
    """rows: (item_id, current, mid, low, high) for one horizon, h=7."""
    return pd.DataFrame([
        {"item_id": i, "current_price": c,
         "forecasts": {7: {"mid": m, "low": lo, "high": hi}}}
        for i, c, m, lo, hi in rows
    ])


def test_the_served_rows_carry_the_band_not_just_the_mid():
    """The not-inert pin. A coverage table fed a frame with no low/high column
    is a table of NaN that reads as 'nothing to see'."""
    served = _served_payload([("a", 10.0, 10.5, 9.0, 12.0)])
    frame = _served_rows(served, 7)
    assert frame.loc[0, "low"] == pytest.approx(9.0)
    assert frame.loc[0, "high"] == pytest.approx(12.0)
    assert frame.loc[0, "mid"] == pytest.approx(10.5)


def test_the_served_rows_keep_a_mid_whose_band_is_missing():
    """A row with no band still scores DA and rank IC, so dropping it here would
    silently shrink the cohort every other table in this script reports."""
    served = _served_payload([("a", 10.0, 10.5, None, None)])
    frame = _served_rows(served, 7)
    assert len(frame) == 1
    assert np.isnan(frame.loc[0, "low"])


def test_band_coverage_counts_the_boundary_as_covered_and_splits_the_misses():
    """`in_interval` in the backtest is `low <= actual <= high`; this must agree,
    or the two coverage figures are not the same measurement.

    The low/high split is the diagnostic that names the cause: q_hat calibrated
    around a centre the serving path moves misses asymmetrically, because the
    displacement has a sign.
    """
    frame = pd.DataFrame({
        "low":      [9.0, 9.0, 9.0, 9.0],
        "high":     [11.0, 11.0, 11.0, 11.0],
        "realised": [9.0, 11.0, 8.5, 11.5],   # both boundaries, then one each side
    })
    row = _coverage_row(frame)
    assert row["n"] == 4
    assert row["cov"] == pytest.approx(0.5)
    assert row["below"] == pytest.approx(0.25)
    assert row["above"] == pytest.approx(0.25)


def test_band_coverage_ignores_a_row_with_no_band():
    """A missing band is not a miss. Counting it as one would report
    under-coverage for a reason that is not the calibration."""
    frame = pd.DataFrame({
        "low":      [9.0, np.nan, 9.0],
        "high":     [11.0, 11.0, np.nan],
        "realised": [10.0, 10.0, 10.0],
    })
    row = _coverage_row(frame)
    assert row["n"] == 1
    assert row["cov"] == pytest.approx(1.0)


def test_band_coverage_reports_nan_rather_than_dividing_by_zero():
    frame = pd.DataFrame({"low": [np.nan], "high": [np.nan], "realised": [10.0]})
    row = _coverage_row(frame)
    assert row["n"] == 0
    assert np.isnan(row["cov"])


def test_the_coverage_line_survives_a_nan_cohort():
    """Same reason `_dollar_line` exists: a %-formatted NaN raising would take
    out the run at the point where the table prints."""
    line = _coverage_line(7, {"n": 0, "cov": float("nan"),
                              "below": float("nan"), "above": float("nan")})
    assert "7" in line
    # And it lines up under its own header, or the table is unreadable in a log.
    assert len(line) == len(COVERAGE_HEADER)


def test_the_sigma_strata_rank_by_width_which_is_monotone_in_sigma():
    """The stratifier is the served half-width, because
    `half = q_hat * sigma ** beta` is strictly increasing in sigma for any
    beta > 0 -- so its quantiles are sigma's quantiles, and the item ordering is
    identical under both arms of SIGMA_EXPONENT. That invariance is the whole
    reason the two arms' tables can be read stratum by stratum.
    """
    # Ten items, relative half-width increasing with the index, all centred at
    # 100 so the width is the only thing that varies.
    n = 10
    widths = np.linspace(1.0, 10.0, n)
    frame = pd.DataFrame({
        "low": 100.0 - widths,
        "high": 100.0 + widths,
        # Every band covers, so `cov` cannot drive the ordering.
        "realised": np.full(n, 100.0),
    })
    rows = _coverage_by_sigma_rows(frame, n_strata=5)
    assert [r["stratum"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["n"] for r in rows] == [2, 2, 2, 2, 2]
    # Stratum 1 is the NARROWEST band, i.e. the lowest sigma. A table that got
    # this backwards would report the tilt with the wrong sign.
    assert rows[0]["halfw"] < rows[-1]["halfw"]
    assert all(r["cov"] == pytest.approx(1.0) for r in rows)


def test_the_sigma_strata_find_a_tilt_the_marginal_number_cannot_see():
    """The defect this table exists for: a band at exactly nominal marginal
    coverage that is badly conditional on sigma. `BAND COVERAGE`'s `cov%` is
    blind to it by construction, which is how a 62->95% ramp survived every
    coverage check on record.
    """
    # 100 items in five width strata of 20. The narrow strata under-cover and
    # the wide ones over-cover, and it averages to exactly nominal. A miss is
    # placed at 5x the half-width, so it is outside the band whatever the width.
    lo, hi, real = [], [], []
    for s, covered_in_stratum in enumerate([8, 14, 18, 20, 20]):
        w = 1.0 + s
        for i in range(20):
            lo.append(100.0 - w)
            hi.append(100.0 + w)
            real.append(100.0 if i < covered_in_stratum else 100.0 + 5 * w)
    frame = pd.DataFrame({"low": lo, "high": hi, "realised": real})

    assert _coverage_row(frame)["cov"] == pytest.approx(0.80)  # nominal, and wrong

    rows = _coverage_by_sigma_rows(frame, n_strata=5)
    assert [r["cov"] for r in rows] == [pytest.approx(c) for c in
                                        (0.40, 0.70, 0.90, 1.00, 1.00)]
    # And the summary statistic sees it where the marginal one saw nothing.
    assert sigma_tilt_pp(rows) == pytest.approx(
        float(np.mean([40.0, 10.0, 10.0, 20.0, 20.0])))


def test_the_sigma_strata_decline_rather_than_stratify_a_tiny_cohort():
    """Fewer rows than strata is not an error, it is an anchor with no resolved
    outcomes. Returning [] prints no table; raising would take out every table
    after it."""
    frame = pd.DataFrame({"low": [9.0, 9.0], "high": [11.0, 11.0],
                          "realised": [10.0, 10.0]})
    assert _coverage_by_sigma_rows(frame, n_strata=5) == []
    assert np.isnan(sigma_tilt_pp([]))


def test_the_sigma_strata_drop_a_row_with_no_defined_width():
    """A missing band and a band straddling zero have no place on the sigma
    axis, and must not be swept into stratum 1 as if they were the narrowest
    bands -- that would fabricate exactly the under-coverage being measured."""
    frame = pd.DataFrame({
        "low": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, -12.0],
        "high": [9.0, 8.0, 7.0, 6.0, 5.5, 11.0, 10.0],
        "realised": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    })
    rows = _coverage_by_sigma_rows(frame, n_strata=5)
    assert sum(r["n"] for r in rows) == 5


def test_the_sigma_line_aligns_under_its_header():
    row = {"stratum": 3, "n": 207, "cov": 0.7431, "halfw": 0.0912}
    assert len(_sigma_line(7, row)) == len(SIGMA_HEADER)


def test_band_coverage_reports_the_width_because_q_hat_is_not_comparable():
    """The only quantity two SIGMA_EXPONENT arms may be differenced on besides
    coverage. `q_hat` is ~5.5x apart in units across that flag, so a paired read
    of it has nothing to compare unless the width is reported here.

    Taken against the band's own midpoint: a half-width of 1.0 on a band centred
    at 10.0 is 10%.
    """
    frame = pd.DataFrame({
        "low":      [9.0, 18.0, 45.0],
        "high":     [11.0, 22.0, 55.0],
        "realised": [10.0, 20.0, 50.0],
    })
    row = _coverage_row(frame)
    # Every band here is +/-10% of its centre, so the median is 10% whatever the
    # price level -- which is the invariance that makes it arm-comparable.
    assert row["halfw"] == pytest.approx(0.10)
    assert len(_coverage_line(7, row)) == len(COVERAGE_HEADER)


def test_the_width_survives_a_band_straddling_zero():
    """A band whose midpoint is <= 0 has no relative width, and dividing by it
    would report a negative or infinite one. It must drop out of the median
    rather than poison it -- the coverage columns still count the row."""
    frame = pd.DataFrame({
        "low":      [-12.0, 9.0],
        "high":     [10.0, 11.0],
        "realised": [5.0, 10.0],
    })
    row = _coverage_row(frame)
    assert row["n"] == 2
    assert row["halfw"] == pytest.approx(0.10)

    # And an all-degenerate cohort reports NaN rather than warning its way to it.
    only = pd.DataFrame({"low": [-12.0], "high": [10.0], "realised": [5.0]})
    assert np.isnan(_coverage_row(only)["halfw"])


# --- the anchor feed audit -------------------------------------------------
#
# 2026-07-09's usual feed is absent and a smaller one with a different basis
# stands in, quoting the served cohort 1.287x higher and reverting the next day.
# Four horizons reversed on that one anchor and the tied share flagged it twice
# before anyone diagnosed it, so the check is a gate rather than a note.
# docs/changelog/2026-08-12-july-09-anchor-is-a-feed-substitution.md

def _profile(rows):
    """rows: ('YYYY-MM-DD', items, [sources]), in `_feed_profile`'s shape."""
    return pd.DataFrame(
        [{"day": pd.Timestamp(d), "items": n, "sources": sorted(s)}
         for d, n, s in rows])


ORDINARY_WEEK = [(f"2026-07-{d:02d}", 26_170, ["aggregator_steam_17mafo"])
                 for d in (6, 7, 8, 10, 11, 12)]


def test_an_ordinary_anchor_passes_the_feed_audit():
    profile = _profile(ORDINARY_WEEK + [("2026-07-09", 26_190,
                                         ["aggregator_steam_17mafo"])])
    ok, lines = audit_anchor_feed(date(2026, 7, 9), profile)
    assert ok
    # It still reports: an audit that only speaks up on failure cannot be read
    # as evidence that the anchor was checked.
    assert any("day(s) before it" in ln for ln in lines)


def test_the_real_substitution_is_refused_on_both_sides():
    """The measured 2026-07-09: the 26k-item feed absent, a 5.5k-item one in.

    It fails BEFORE and AFTER at once -- its features read a synthetic +28.7%
    jump and its outcome resolves back on the ordinary feed -- which is why all
    four horizons reversed rather than tilting.
    """
    profile = _profile(ORDINARY_WEEK + [("2026-07-09", 5_502,
                                         ["aggregator_sync"])])
    ok, lines = audit_anchor_feed(date(2026, 7, 9), profile)
    assert not ok
    blob = "\n".join(lines)
    assert "the days BEFORE the anchor were collected differently" in blob
    assert "the days AFTER the anchor were collected differently" in blob
    assert "absent on the anchor: aggregator_steam_17mafo" in blob
    assert "only on the anchor:   aggregator_sync" in blob
    # And the count leg fires independently of the swap.
    assert "21%" in blob


def test_a_substitution_at_full_coverage_is_still_refused():
    """The item count alone would pass this. A feed that reprices the cohort at
    the same breadth is the harder case and the one a count check misses."""
    profile = _profile(ORDINARY_WEEK + [("2026-07-09", 26_100,
                                         ["aggregator_sync"])])
    ok, lines = audit_anchor_feed(date(2026, 7, 9), profile)
    assert not ok
    assert any("collected differently" in ln for ln in lines)
    assert not any("item count is" in ln for ln in lines)


def test_a_thin_day_on_the_right_feed_is_still_refused():
    """And the converse: the 07-11..07-13 transition days carry the full source
    set at a third of the breadth, so most items are served from a quote before
    the anchor -- stale rather than substituted."""
    profile = _profile(ORDINARY_WEEK + [("2026-07-09", 5_525,
                                        ["aggregator_steam_17mafo"])])
    ok, lines = audit_anchor_feed(date(2026, 7, 9), profile)
    assert not ok
    assert any("item count is 21%" in ln for ln in lines)
    assert not any("collected differently" in ln for ln in lines)


def test_a_feed_change_only_after_the_anchor_names_the_outcome_side():
    """The anchor's own day and history are fine; the feed set changes two days
    later, so its 3d outcome resolves on a basis its quote was never measured
    on. Refused, and the message must say AFTER -- the anchor is not the
    deficient day here and a 'missing feed' reading would send the reader to the
    wrong date."""
    profile = _profile(
        [(f"2026-07-{d:02d}", 26_170, ["aggregator_steam_17mafo"])
         for d in (6, 7, 8)]
        + [(f"2026-07-{d:02d}", 33_000,
            ["aggregator_steam_17mafo", "aggregator_buff163"])
           for d in (10, 11, 12)]
        + [("2026-07-09", 26_190, ["aggregator_steam_17mafo"])])
    ok, lines = audit_anchor_feed(date(2026, 7, 9), profile)
    assert not ok
    blob = "\n".join(lines)
    assert "the days AFTER the anchor" in blob
    assert "the days BEFORE the anchor" not in blob
    assert "absent on the anchor: aggregator_buff163" in blob


def test_a_feed_change_only_before_the_anchor_names_the_feature_side():
    """2026-03-22's shape: the anchor is collected like every day that follows
    it, and unlike the days it builds its features from."""
    profile = _profile(
        [(f"2026-03-{d:02d}", 3_478, ["aggregator_sync"]) for d in (19, 20, 21)]
        + [(f"2026-03-{d:02d}", 32_449,
            ["aggregator_sync", "aggregator_buff163", "aggregator_csfloat"])
           for d in (23, 24, 25)]
        + [("2026-03-22", 32_437, ["aggregator_sync", "aggregator_buff163",
                                   "aggregator_csfloat"])])
    ok, lines = audit_anchor_feed(date(2026, 3, 22), profile)
    assert not ok
    blob = "\n".join(lines)
    assert "the days BEFORE the anchor" in blob
    assert "synthetic jump" in blob
    # The count leg must not also fire: the anchor matches the side it belongs
    # to, so the reference is 32k rather than the pre-change 3.5k.
    assert not any("item count is" in ln for ln in lines)


def test_the_reference_is_the_modal_day_not_the_union():
    """One neighbour of three on the later side carries an extra feed. The union
    would call the ordinary anchor 'missing' it; the intersection would let a
    substitution through as a subset of the reference."""
    profile = _profile(
        [(f"2026-07-{d:02d}", 26_170, ["aggregator_steam_17mafo"])
         for d in (6, 7, 8, 10, 11)]
        + [("2026-07-12", 33_000,
            ["aggregator_steam_17mafo", "aggregator_buff163"])]
        + [("2026-07-09", 26_190, ["aggregator_steam_17mafo"])])
    ok, _ = audit_anchor_feed(date(2026, 7, 9), profile)
    assert ok


def test_a_dropped_calendar_day_is_named_as_one():
    """2026-07-27, 07-30, 08-02 and 08-03 hold no rows at all. That is a
    different failure from a substitution and the message must not confuse the
    two -- the anchor cannot be replayed, rather than replayed badly."""
    profile = _profile([("2026-07-26", 26_237, ["aggregator_steam_17mafo"]),
                        ("2026-07-28", 26_312, ["aggregator_steam_17mafo"])])
    ok, lines = audit_anchor_feed(date(2026, 7, 27), profile)
    assert not ok
    assert "not in the archive at all" in lines[0]


def test_an_anchor_with_no_comparable_neighbour_fails_closed():
    """No neighbour means no reference. Passing would make the audit vacuous on
    exactly the sparse stretches of the archive where collection is least
    uniform."""
    ok, lines = audit_anchor_feed(
        date(2026, 7, 9),
        _profile([("2026-07-09", 26_190, ["aggregator_steam_17mafo"])]))
    assert not ok
    assert "nothing to compare" in lines[0]

    ok, lines = audit_anchor_feed(date(2026, 7, 9), _profile([]))
    assert not ok
    assert "NO rows" in lines[0]


def test_the_pre_2026_series_compares_equal_rather_than_vacuous():
    """`source` is NULL for all 13 years before 2026, which `_feed_profile`
    COALESCEs to a label. Without that every day's source set would be equal and
    empty, and the audit would pass every historical anchor by construction --
    including one collected differently."""
    profile = _profile([(f"2024-06-{d:02d}", 900, ["<none>"])
                        for d in (12, 13, 14, 16, 17, 18)]
                       + [("2024-06-15", 900, ["<none>"])])
    ok, _ = audit_anchor_feed(date(2024, 6, 15), profile)
    assert ok


def test_the_profile_query_reads_the_archive_through_the_reader(tmp_path):
    """One end-to-end read, because the audit's inputs come from SQL: the two
    columns have to survive `prices_relation`'s projection and the universe
    filter, and a source substitution has to show up as one."""
    pd.DataFrame({
        "item_slug": (["AK-47 | Redline (Field-Tested)"] * 3 + ["AWP | Asiimov (Field-Tested)"] * 3
                      + ["Glock-18 | Fade (Factory New)"] * 2),
        "day": pd.to_datetime(["2026-07-08", "2026-07-09", "2026-07-10"] * 2
                              + ["2026-07-08", "2026-07-10"]),
        "source": (["aggregator_steam_17mafo", "aggregator_sync",
                    "aggregator_steam_17mafo"] * 2
                   + ["aggregator_steam_17mafo"] * 2),
        "mean_price": [10.0, 12.9, 10.1, 50.0, 64.0, 50.2, 5.0, 5.0],
        "volume": [1] * 8,
        "ingested_at": pd.to_datetime(["2026-07-10"] * 8),
    }).to_parquet(tmp_path / "prices-2026-07.parquet", index=False)

    profile = _feed_profile(date(2026, 7, 9), archive_dir=tmp_path)
    assert list(profile["day"].astype(str)) == ["2026-07-08", "2026-07-09",
                                               "2026-07-10"]
    assert list(profile["items"]) == [3, 2, 3]
    assert list(profile["sources"].iloc[1]) == ["aggregator_sync"]

    ok, lines = audit_anchor_feed(date(2026, 7, 9), profile)
    assert not ok
    assert any("absent on the anchor: aggregator_steam_17mafo" in ln
               for ln in lines)


def test_the_bid_source_cannot_make_a_day_look_differently_collected(tmp_path):
    """`archive_universe_sql_filter` drops BUFF's bid from the consensus, so it
    must not appear in a day's source set either. A day where only the bid
    arrived would otherwise read as a substitution -- and a day where it arrived
    beside the usual feed would read as the anchor 'missing' a feed on every
    other day."""
    pd.DataFrame({
        "item_slug": ["AK-47 | Redline (Field-Tested)"] * 4,
        "day": pd.to_datetime(["2026-07-08", "2026-07-09", "2026-07-09",
                               "2026-07-10"]),
        "source": ["aggregator_steam_17mafo", "aggregator_steam_17mafo",
                   "aggregator_buff163_buy", "aggregator_steam_17mafo"],
        "mean_price": [10.0, 10.1, 9.0, 10.2],
        "volume": [1] * 4,
        "ingested_at": pd.to_datetime(["2026-07-10"] * 4),
    }).to_parquet(tmp_path / "prices-2026-07.parquet", index=False)

    profile = _feed_profile(date(2026, 7, 9), archive_dir=tmp_path)
    assert all("aggregator_buff163_buy" not in s for s in profile["sources"])
    ok, _ = audit_anchor_feed(date(2026, 7, 9), profile)
    assert ok


# --------------------------------------------------------------------------- #
# the OUTCOME side of the anchor audit
# --------------------------------------------------------------------------- #

def _counts(rows):
    """rows: ('YYYY-MM-DD', item_count) -- the daily universe size."""
    return pd.Series({pd.Timestamp(d).date(): n for d, n in rows}).sort_index()


def test_a_cutover_is_detected_by_the_same_rule_the_label_path_uses():
    """`_collection_shift_dates` fires on an abrupt change in the collected
    universe size. The audit must use that rule and not a second one, or the
    replay and the training labels will disagree about which dates exist.
    """
    counts = _counts([("2026-03-19", 32_000), ("2026-03-20", 32_100),
                      ("2026-03-21", 31_900), ("2026-03-22", 25_000),
                      ("2026-03-23", 25_100)])
    assert cutovers_from_counts(counts) == [date(2026, 3, 22)]


def test_an_ordinary_wobble_in_the_universe_is_not_a_cutover():
    """The threshold is 20%. A collector that drops a few hundred items overnight
    is normal and must not void a horizon -- the check is worthless if it fires
    on ordinary days, because then every anchor is refused.
    """
    counts = _counts([("2026-05-01", 26_000), ("2026-05-02", 25_400),
                      ("2026-05-03", 26_300), ("2026-05-04", 25_900)])
    assert cutovers_from_counts(counts) == []


def test_a_cutover_after_the_anchor_voids_only_the_horizons_that_span_it():
    """The span rule, `anchor < s <= anchor + h`, copied from `prepare_targets`.
    This is the case that matters: 2026-03-10 passes the feed audit -- its own
    collection is ordinary -- and its 14d and 30d OUTCOMES land the far side of
    the 2026-03-22 consensus break, so the replay would score them against a
    synthetic market-wide crash while the training label path voids them.
    """
    spanned = cutovers_in_outcome_window(
        date(2026, 3, 10), [3, 7, 14, 30], [date(2026, 3, 22)])
    assert spanned == {14: [date(2026, 3, 22)], 30: [date(2026, 3, 22)]}


def test_a_cutover_on_the_anchor_itself_is_the_feed_audit_s_job_not_this_one():
    """Exclusive at the anchor, inclusive at the target -- `(anchor, anchor+h]`.
    A cutover ON the anchor is a different defect with a different remedy, and
    double-reporting it here would make the two audits disagree about who
    refused the date.
    """
    assert cutovers_in_outcome_window(
        date(2026, 3, 22), [3, 7], [date(2026, 3, 22)]) == {}
    assert cutovers_in_outcome_window(
        date(2026, 3, 19), [3], [date(2026, 3, 22)]) == {3: [date(2026, 3, 22)]}
