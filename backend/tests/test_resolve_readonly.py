"""The read-only resolver must be the PRODUCTION scorer, not a second one.

Its whole justification is that it reaches panel depth without a prod write. If
its guards or its verdict drift from `backtest_accuracy.resolve_outcomes`, it
silently reports a different number than the published figures -- which is the
`base_price` vs `current_price` wedge failure all over again
(docs/changelog/2026-08-25-centre-shrinks-to-zero-and-the-dollar-band-is-the-wedge.md).

So these pin the two guards that decide which rows exist, plus the fact that the
verdict is delegated rather than recomputed. The exact-reproduction check against
prod lives in `--validate`, which needs the DB and cannot run here.
"""

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from scripts.archive.resolve_readonly import MIN_PRICE, _resolve


def _forecast(fid=1, item_id=7, f_date="2026-08-01", horizon=3, mid=110.0, low=90.0, high=130.0, quote=100.0):
    return SimpleNamespace(
        id=fid,
        item_id=item_id,
        forecast_date=date.fromisoformat(f_date),
        horizon_days=horizon,
        price_low=low,
        price_mid=mid,
        price_high=high,
        current_price=quote,
        direction="up",
        model_version="v-test",
    )


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A voted-price frame stubbed in place of the Parquet archive read."""

    def _install(rows):
        frame = pd.DataFrame(rows)
        monkeypatch.setattr("scripts.archive.resolve_readonly.load_voted_prices", lambda *a, **k: frame)
        return tmp_path

    return _install


def _series(slug, start, n, price):
    """`n` daily observations from `start`, all at `price`."""
    d0 = date.fromisoformat(start)
    return [{"item_id": slug, "date": d0 + timedelta(days=i), "price": price} for i in range(n)]


def test_resolves_a_clean_pair(archive):
    # Base window entirely before the forecast, actual window entirely after.
    rows = _series("ak", "2026-07-30", 3, 100.0) + _series("ak", "2026-08-02", 3, 120.0)
    out, dropped = _resolve([_forecast()], {7: "ak"}, archive(rows))
    assert dropped == 0
    assert len(out) == 1
    assert out[0]["base_price"] == 100.0
    assert out[0]["actual_price"] == 120.0
    # Verdict delegated to _derive_verdict, not recomputed here.
    assert out[0]["direction_actual"] == "up"
    assert out[0]["direction_correct"] == 1


def test_drops_when_actual_window_reaches_before_the_forecast(archive):
    """The manufactured-flat guard: overlapping windows score a real move as 0."""
    # Only one observation post-dates the forecast, so the actual leg's median is
    # still decided by pre-forecast observations.
    rows = [*_series("ak", "2026-07-30", 3, 100.0), {"item_id": "ak", "date": date(2026, 8, 4), "price": 150.0}]
    out, dropped = _resolve([_forecast()], {7: "ak"}, archive(rows))
    assert out == []
    assert dropped == 1


def test_drops_when_an_anchor_cannot_resolve(archive):
    """No observation at or before the forecast date -- never a fallback."""
    rows = _series("ak", "2026-08-02", 3, 120.0)
    out, dropped = _resolve([_forecast()], {7: "ak"}, archive(rows))
    assert out == []
    assert dropped == 1


def test_excludes_below_the_served_cohort_without_counting_it_dropped(archive):
    """Out of scope is not a resolution failure, and must not inflate `dropped`."""
    cheap = MIN_PRICE / 2
    rows = _series("ak", "2026-07-30", 3, cheap) + _series("ak", "2026-08-02", 3, cheap)
    out, dropped = _resolve([_forecast()], {7: "ak"}, archive(rows))
    assert out == []
    assert dropped == 0


def test_missing_slug_mapping_drops_the_forecast(archive):
    out, dropped = _resolve([_forecast()], {}, archive([]))
    assert out == []
    assert dropped == 1
