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
from scripts.replay_serving import _naive_baseline


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


def test_a_replay_ignores_the_engineered_cache():
    """The other cache leak. Its key fingerprints forecaster.py, which a replay
    does not change, so a live cache is a valid hit carrying post-anchor rows.
    """
    import inspect
    src = inspect.getsource(ItemForecaster._load_engineered_cache)
    head = src.split("path = self._engineered_cache_path")[0]
    assert "replay_anchor() is not None" in head
    assert "return None" in head
