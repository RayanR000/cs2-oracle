"""Tests for the market factor used by the market-relative label experiment.

See docs/superpowers/specs/2026-08-06-market-relative-labels-design.md.
"""

import numpy as np
import pandas as pd
import pytest
from models.market_factor import (
    INDEX_TOLERANCE_DAYS,
    MIN_INDEX_ITEMS,
    build_market_index,
    forecast_market_factor,
    market_factor_for_horizon,
)


def _frame(rows):
    """rows: list of (item_id, 'YYYY-MM-DD', price)."""
    return pd.DataFrame(rows, columns=["item_id", "date", "price"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def _flat_panel(n_items, dates, price=10.0, start_id=0):
    return _frame([(start_id + i, d, price) for i in range(n_items) for d in dates])


def test_index_is_flat_when_no_prices_move():
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    idx = build_market_index(_flat_panel(40, dates), min_items=5)
    assert list(idx.index) == list(pd.to_datetime(dates))
    # First date has no predecessor, so no return.
    assert not idx["valid"].iloc[0]
    assert idx["log_return"].iloc[1:].eq(0.0).all()
    assert np.allclose(idx["level"].iloc[1:], 1.0)


def test_index_tracks_a_uniform_ten_percent_move():
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 11.0))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["log_return"].iloc[1] == pytest.approx(np.log(1.1))
    assert idx["level"].iloc[1] == pytest.approx(1.1)


def test_median_ignores_a_single_outlier_item():
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 10.0 if i else 1000.0))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["log_return"].iloc[1] == pytest.approx(0.0)


def test_sub_dollar_items_do_not_move_the_index():
    """A penny item swinging 50% must not register. Its prior price is < $1."""
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 10.0))
    for j in range(100, 200):  # 100 penny items, all up 50%
        rows.append((j, "2026-01-01", 0.03))
        rows.append((j, "2026-01-02", 0.045))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["log_return"].iloc[1] == pytest.approx(0.0)
    assert idx["n_items"].iloc[1] == 40


def test_thin_days_are_marked_invalid_not_dropped():
    rows = []
    for i in range(3):  # only 3 pairs, below min_items
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 12.0))
    idx = build_market_index(_frame(rows), min_items=MIN_INDEX_ITEMS)
    assert pd.to_datetime("2026-01-02") in idx.index
    assert not idx.loc[pd.to_datetime("2026-01-02"), "valid"]
    assert np.isnan(idx.loc[pd.to_datetime("2026-01-02"), "log_return"])


def test_an_invalid_day_does_not_poison_later_levels():
    """One thin day must not NaN the whole tail of the index."""
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-03", 10.0))
    for i in range(2):  # 2026-01-02 is thin
        rows.append((i, "2026-01-02", 10.0))
    idx = build_market_index(_frame(rows), min_items=MIN_INDEX_ITEMS)
    assert np.isfinite(idx["level"].iloc[-1])
    # invalid_cum lets a consumer detect that a window spanned the bad day.
    assert idx["invalid_cum"].iloc[-1] > idx["invalid_cum"].iloc[0]


def test_items_entering_midway_do_not_create_a_jump():
    """Chain-linking: a new item joining at a different price level must not
    move the index, because only paired day-over-day returns contribute."""
    rows = []
    for i in range(40):
        for d in ["2026-01-01", "2026-01-02", "2026-01-03"]:
            rows.append((i, d, 10.0))
    for j in range(100, 140):  # join on day 2 at a much higher level
        rows.append((j, "2026-01-02", 500.0))
        rows.append((j, "2026-01-03", 500.0))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["log_return"].iloc[1] == pytest.approx(0.0)
    assert idx["log_return"].iloc[2] == pytest.approx(0.0)
    assert idx["n_items"].iloc[1] == 40  # entrant has no prior day
    assert idx["n_items"].iloc[2] == 80


def test_non_positive_prices_are_excluded():
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 10.0))
    rows.append((999, "2026-01-01", 0.0))
    rows.append((999, "2026-01-02", 5.0))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["n_items"].iloc[1] == 40


def test_empty_frame_returns_empty_index():
    idx = build_market_index(_frame([]), min_items=5)
    assert idx.empty
    assert list(idx.columns) == ["log_return", "n_items", "valid", "level", "invalid_cum"]


def _index_from_daily(returns, start="2026-01-01"):
    """Build an index directly from a list of daily log returns."""
    dates = pd.date_range(start, periods=len(returns), freq="D")
    out = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    out["log_return"] = returns
    out["n_items"] = 100
    out["valid"] = out["log_return"].notna()
    out["level"] = np.exp(out["log_return"].fillna(0.0).cumsum())
    out["invalid_cum"] = (~out["valid"]).cumsum().astype(int)
    return out


def test_factor_is_the_percent_move_over_the_window():
    idx = _index_from_daily([np.nan] + [np.log(1.01)] * 10)
    m = market_factor_for_horizon(idx, horizon=3)
    d0 = pd.Timestamp("2026-01-01")
    # 3 days of +1% compounding = 1.01^3 - 1
    assert m.loc[d0] == pytest.approx((1.01**3 - 1) * 100)


def test_factor_is_nan_past_the_end_of_the_index():
    idx = _index_from_daily([np.nan] + [0.0] * 4)
    m = market_factor_for_horizon(idx, horizon=3)
    assert np.isnan(m.iloc[-1])
    assert np.isnan(m.iloc[-2])


def test_factor_resolves_a_short_calendar_gap_within_tolerance():
    """The window end date is missing but a date 2 days later exists."""
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06"])
    idx = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    idx["log_return"] = [np.nan, 0.0, 0.0, np.log(1.05)]
    idx["n_items"] = 100
    idx["valid"] = idx["log_return"].notna()
    idx["level"] = np.exp(idx["log_return"].fillna(0.0).cumsum())
    idx["invalid_cum"] = (~idx["valid"]).cumsum().astype(int)
    # From 01-02, +3d lands on 01-05 which is absent; 01-06 is 1 day later.
    m = market_factor_for_horizon(idx, horizon=3, tolerance_days=3)
    assert m.loc[pd.Timestamp("2026-01-02")] == pytest.approx(5.0)


def test_factor_is_nan_when_the_gap_exceeds_tolerance():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-20"])
    idx = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    idx["log_return"] = [np.nan, 0.0, 0.0]
    idx["n_items"] = 100
    idx["valid"] = idx["log_return"].notna()
    idx["level"] = np.exp(idx["log_return"].fillna(0.0).cumsum())
    idx["invalid_cum"] = (~idx["valid"]).cumsum().astype(int)
    m = market_factor_for_horizon(idx, horizon=3, tolerance_days=INDEX_TOLERANCE_DAYS)
    assert np.isnan(m.loc[pd.Timestamp("2026-01-02")])


def test_factor_is_nan_when_the_window_spans_an_invalid_day():
    rets = [np.nan] + [0.0] * 3 + [np.nan] + [0.0] * 5  # index 4 is thin
    idx = _index_from_daily(rets)
    m = market_factor_for_horizon(idx, horizon=3)
    # 2026-01-02 (i=1) -> 2026-01-05 (i=4) spans the invalid day.
    assert np.isnan(m.loc[pd.Timestamp("2026-01-02")])
    # 2026-01-06 (i=5) -> 2026-01-09 (i=8) does not.
    assert m.loc[pd.Timestamp("2026-01-06")] == pytest.approx(0.0)


def test_degenerate_all_flat_market_gives_zero_not_nan():
    idx = _index_from_daily([np.nan] + [0.0] * 10)
    m = market_factor_for_horizon(idx, horizon=3)
    assert m.loc[pd.Timestamp("2026-01-02")] == pytest.approx(0.0)


# --- the leakage tests: the single most important thing in this file ---


def test_forecast_ignores_everything_after_as_of():
    """Truncating the index at `as_of` must not change the forecast. If it
    does, the estimator is reading the future and any measured gain is fake."""
    rng = np.random.RandomState(0)
    rets = [np.nan, *list(rng.normal(0, 0.01, 400))]
    idx = _index_from_daily(rets)
    as_of = idx.index[300]
    full = forecast_market_factor(idx, as_of, horizon=7)
    truncated = forecast_market_factor(idx.loc[:as_of], as_of, horizon=7)
    assert full == pytest.approx(truncated)


def test_forecast_is_unchanged_when_the_future_is_nulled():
    rng = np.random.RandomState(1)
    rets = [np.nan, *list(rng.normal(0, 0.01, 400))]
    idx = _index_from_daily(rets)
    as_of = idx.index[300]
    before = forecast_market_factor(idx, as_of, horizon=14)
    poisoned = idx.copy()
    poisoned.loc[poisoned.index > as_of, "log_return"] = 99.0
    poisoned["level"] = np.exp(poisoned["log_return"].fillna(0.0).cumsum())
    after = forecast_market_factor(poisoned, as_of, horizon=14)
    assert before == pytest.approx(after)


def test_forecast_scales_a_constant_drift_to_the_horizon():
    daily = np.log(1.001)
    idx = _index_from_daily([np.nan] + [daily] * 300)
    as_of = idx.index[-1]
    got = forecast_market_factor(idx, as_of, horizon=7)
    assert got == pytest.approx((np.exp(7 * daily) - 1) * 100)


def test_forecast_returns_zero_with_no_usable_history():
    idx = _index_from_daily([np.nan, np.nan])
    assert forecast_market_factor(idx, idx.index[-1], horizon=7) == 0.0


def test_forecast_before_the_index_starts_returns_zero():
    idx = _index_from_daily([np.nan] + [0.0] * 5)
    assert forecast_market_factor(idx, pd.Timestamp("2020-01-01"), horizon=7) == 0.0


# --- guards retained after the market-relative label experiment was removed ---
#
# The experiment that introduced this module was refuted and its forecaster
# wiring removed (2026-08-06). The module itself survives because
# scripts/ab_test_item_metadata.py depends on it, so these two guards stay.


def test_market_factor_columns_are_never_features():
    """`market_factor_*` is built from other items' FUTURE prices. Any frame
    carrying it must not hand it to the model as a feature."""
    from models.forecaster import ItemForecaster

    f = ItemForecaster(db_session=None)
    df = pd.DataFrame(
        {
            "item_id": [1, 2],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "price": [10.0, 11.0],
            "return_7d": [9.0, 10.0],
            "target_return_3d": [10.0, 9.0],
            "market_factor_3d": [1.0, 2.0],
            "market_factor_7d": [1.5, 2.5],
            "market_factor_14d": [2.0, 3.0],
            "market_factor_30d": [2.5, 3.5],
        }
    )
    cols = f._select_feature_cols(df, ItemForecaster.HORIZONS, ItemForecaster.SHELVED_FEATURES)
    for h in ItemForecaster.HORIZONS:
        assert f"market_factor_{h}d" not in cols
    assert "return_7d" in cols


def test_demean_returns_survives_for_the_item_metadata_ab():
    """scripts/ab_test_item_metadata.py calls this directly. Removing it with
    the rest of the refuted experiment would break that A/B and orphan the
    market-relative amendments in docs/research/accuracy-opportunities.md."""
    from models.forecaster import ItemForecaster

    got = ItemForecaster._demean_returns(np.array([5.0, -2.0]), np.array([1.0, np.nan]))
    assert got == pytest.approx([4.0, -2.0])
