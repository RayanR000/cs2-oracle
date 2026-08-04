"""Prediction must not engineer 1460 days of history to keep 3 rows per item.

Truncation has to be row-based, not calendar-based. The feature set mixes
calendar-date lag joins (LAGS up to 180 days, forecaster.py:937) with
positional row-count rollings (windows to 200, forecaster.py:1118) over an
archive that is only ~48% dense, so a 240-*day* cutoff can yield ~115 rows and
silently change every rolling feature at the serving edge.

Multi-source voting collapses the frame to one row per item-day, so the last N
rows always span >= N calendar days. That invariant is what makes a row-based
tail satisfy both requirements at once, and test_voting_yields_one_row_per_item_day
is what keeps it true.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(),
                          model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _price_frame(n_items=3, n_days=1460, start=date(2022, 1, 1)):
    """Dense daily series, one row per item-day, with a mild upward drift."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_items):
        price = 10.0 + i
        for d in range(n_days):
            price *= 1.0 + rng.normal(0.0005, 0.02)
            rows.append({
                "item_id": f"item-{i}",
                "date": start + timedelta(days=d),
                "price": round(max(price, 0.05), 4),
                "volume": float(rng.integers(1, 500)),
            })
    return pd.DataFrame(rows)


def test_tail_constant_covers_both_window_requirements():
    # 200-row positional rollings + PREDICT_TAIL_ROWS, and 180-day calendar
    # lags + PREDICT_TAIL_ROWS. The row-based tail must clear both.
    assert ItemForecaster.PREDICT_TAIL_ITEM_DAYS >= 200 + ItemForecaster.PREDICT_TAIL_ROWS
    assert ItemForecaster.PREDICT_TAIL_ITEM_DAYS >= 180 + ItemForecaster.PREDICT_TAIL_ROWS


def test_voting_yields_one_row_per_item_day(forecaster):
    """The invariant the row-based tail depends on.

    If voting ever emits two rows for one item-day, N rows would no longer span
    N calendar days and the 180-day calendar lags could silently fall short.
    """
    raw = _price_frame(n_items=2, n_days=30)
    # Same item-days observed by three sources, plus one outlier per day.
    multi = pd.concat([
        raw.assign(source="STEAMCOMMUNITY"),
        raw.assign(source="BUFF163", price=raw["price"] * 1.01),
        raw.assign(source="CSFLOAT", price=raw["price"] * 20),
    ], ignore_index=True)

    voted = forecaster._apply_multi_source_voting(multi)

    dupes = voted.groupby(["item_id", "date"]).size()
    assert (dupes == 1).all(), (
        f"Voting emitted duplicate item-days: "
        f"{dupes[dupes > 1].head().to_dict()}"
    )


def test_tail_is_a_noop_for_short_history_items(forecaster):
    """Items with <= PREDICT_TAIL_ITEM_DAYS rows must be untouched.

    PREDICT_MIN_HISTORY_DAYS is 14, so most eligible items hold far fewer than
    240 rows. Their features must be bit-identical after this change.
    """
    short = _price_frame(n_items=2, n_days=100)
    tailed = forecaster._tail_predict_frame(short)
    pd.testing.assert_frame_equal(
        short.sort_values(["item_id", "date"]).reset_index(drop=True),
        tailed.sort_values(["item_id", "date"]).reset_index(drop=True),
    )


def test_tail_keeps_exactly_the_last_n_rows_per_item(forecaster):
    long = _price_frame(n_items=3, n_days=1460)
    tailed = forecaster._tail_predict_frame(long)

    counts = tailed.groupby("item_id").size()
    assert (counts == ItemForecaster.PREDICT_TAIL_ITEM_DAYS).all()

    # It must be the *last* rows, not the first.
    for item, group in tailed.groupby("item_id"):
        expected_max = long[long["item_id"] == item]["date"].max()
        assert group["date"].max() == expected_max


def test_tail_selects_distinct_dates_not_row_positions(forecaster):
    """Duplicated item-days must not shorten the calendar span retained.

    Voting collapses duplicates today, so this guards the helper against a
    caller that has not voted yet: a row-position tail would keep only ~120
    calendar days from 240 duplicated rows.
    """
    dense = _price_frame(n_items=1, n_days=400)
    doubled = pd.concat([dense, dense], ignore_index=True)

    tailed = forecaster._tail_predict_frame(doubled)
    span = tailed["date"].nunique()
    assert span == ItemForecaster.PREDICT_TAIL_ITEM_DAYS, (
        f"Expected {ItemForecaster.PREDICT_TAIL_ITEM_DAYS} distinct dates, "
        f"got {span} — the tail is counting rows, not dates."
    )


def test_served_features_survive_truncation(forecaster):
    """THE LOAD-BEARING TEST.

    The served feature vector is the last row per item. It must be identical
    whether engineered from 1460 days or from the truncated tail. Anything that
    differs here is train/serve skew shipped to production.
    """
    full = _price_frame(n_items=3, n_days=1460)
    events = pd.DataFrame(columns=["date", "event_type", "name"])
    first_dates = full.groupby("item_id")["date"].min()

    full_feats = forecaster.engineer_features(full, events)
    tail_feats = forecaster.engineer_features(
        forecaster._tail_predict_frame(full), events,
        item_first_dates=first_dates,
    )

    def _last_rows(df):
        return (df.sort_values(["item_id", "date"])
                  .groupby("item_id").last()
                  .sort_index())

    a, b = _last_rows(full_feats), _last_rows(tail_feats)

    shared = [c for c in a.columns if c in b.columns
              and pd.api.types.is_numeric_dtype(a[c])]
    assert shared, "No numeric feature columns to compare"

    # MACD is the one exception, and it is arithmetic, not a window shortfall.
    # It is built from ewm(), which never fully forgets: the weight on a point
    # PREDICT_TAIL_ITEM_DAYS back in a span-26 EWM is (1-2/27)^240 ~ 2e-8, so a
    # truncated series reproduces it only to ~1e-5 relative. Measured worst case
    # on this fixture is 3.7e-6. Prices carry 4 decimals and split thresholds
    # are nowhere near that resolution, so this is numerically irrelevant --
    # but it is bounded here so a genuine regression cannot hide inside it.
    EWM_FAMILY = {"macd_line", "macd_signal", "macd_histogram",
                  "macd_hist_slope_7d"}
    EWM_RTOL = 1e-4

    def _close(col):
        rtol = EWM_RTOL if col in EWM_FAMILY else 1e-9
        return np.allclose(a[col].to_numpy(dtype=float),
                           b[col].to_numpy(dtype=float),
                           rtol=rtol, atol=1e-9, equal_nan=True)

    mismatched = [c for c in shared if not _close(c)]
    assert not mismatched, (
        f"Truncation changed {len(mismatched)} served feature(s): "
        f"{sorted(mismatched)[:12]}"
    )


def test_item_age_days_needs_the_true_first_seen_date(forecaster):
    """Truncation alone reports every item as PREDICT_TAIL_ITEM_DAYS old.

    item_age_days is derived from the frame's own min date, so it is the one
    feature a tail cannot reconstruct. It is not in the served allowlist today
    (FEATURE_GROUP_ALLOWLIST is price_technicals only), but relying on that
    would make widening the allowlist silently corrupt it.
    """
    full = _price_frame(n_items=2, n_days=1460)
    events = pd.DataFrame(columns=["date", "event_type", "name"])
    tail = forecaster._tail_predict_frame(full)

    def _age(feats):
        return (feats.sort_values(["item_id", "date"])
                     .groupby("item_id")["item_age_days"].last())

    truth = _age(forecaster.engineer_features(full, events))

    # Without the mapping: wrong, and wrong by exactly the tail length.
    naive = _age(forecaster.engineer_features(tail, events))
    assert (naive == ItemForecaster.PREDICT_TAIL_ITEM_DAYS - 1).all()

    # With it: correct.
    fixed = _age(forecaster.engineer_features(
        tail, events, item_first_dates=full.groupby("item_id")["date"].min()))
    pd.testing.assert_series_equal(truth, fixed)


def test_v1_engineered_cache_is_rejected(forecaster):
    """A pre-truncation cache holds full history, not a tail.

    Dated today so the existing 3-day staleness check cannot be what rejects
    it — this isolates the version guard.
    """
    df = _price_frame(n_items=2, n_days=20)
    df.attrs["_cache_date"] = str(date.today())      # fresh, but unversioned
    df.to_parquet(forecaster._engineered_cache_path, index=False)
    assert forecaster._load_engineered_cache() is None


def test_cache_written_by_this_version_roundtrips(forecaster):
    df = _price_frame(n_items=2, n_days=20)
    forecaster._save_engineered_cache(df)
    loaded = forecaster._load_engineered_cache()
    assert loaded is not None
    assert len(loaded) == len(df)
    # Carried in attrs, not as a column: the predict frame reaches ~2M rows and
    # copying it to append a constant column would double peak memory on the
    # path that already OOMs in CI.
    assert list(loaded.columns) == list(df.columns)


def test_a_future_cache_version_is_rejected(forecaster):
    df = _price_frame(n_items=2, n_days=20)
    df.attrs["_cache_date"] = str(date.today())
    df.attrs["_cache_version"] = ItemForecaster.ENGINEERED_CACHE_VERSION + 1
    df.to_parquet(forecaster._engineered_cache_path, index=False)
    assert forecaster._load_engineered_cache() is None
