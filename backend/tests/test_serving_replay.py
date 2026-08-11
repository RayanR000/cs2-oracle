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
from scripts.replay_serving import (_basis_frame, _exact_day, _naive_baseline,
                                    _requested_horizons)


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


def test_a_replay_ignores_the_engineered_cache():
    """The other cache leak. Its key fingerprints forecaster.py, which a replay
    does not change, so a live cache is a valid hit carrying post-anchor rows.
    """
    import inspect
    src = inspect.getsource(ItemForecaster._load_engineered_cache)
    head = src.split("path = self._engineered_cache_path")[0]
    assert "replay_anchor() is not None" in head
    assert "return None" in head
