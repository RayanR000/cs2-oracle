"""The serving replay: score predict() against outcomes the archive already has.

Walk-forward CV scores fold predictions. `_recenter_on_direction`, the tier
bias, the prior-day blend and the conformal band run inside `predict()` and
nowhere else, so nothing measured them without publishing a forecast and
waiting. REPLAY_ANCHOR rewinds the serving clock instead.

The tests that matter are the leak tests: a replay that can see past its own
anchor scores a forecast against data it already had.
"""
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from models.forecaster import ItemForecaster
from scripts.replay_serving import (PINNED_MAX_SPAN_DAYS, PINNED_SMOOTH_WINDOW,
                                    _basis_frame, _exact_day, _naive_baseline,
                                    _pin_matches_production, _pinned_anchor,
                                    _pinned_rank_ic, _requested_horizons)


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
