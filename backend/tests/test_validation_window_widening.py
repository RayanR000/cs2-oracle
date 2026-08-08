"""Voided labels must not silently demote the production split to a positional one.

`prepare_targets` voids labels built across a fabricated archive day
(`test_degenerate_label_dates.py`), and `_train_horizon_inline` drops the voided
rows. The fabricated days — the 2026-07-16 / 07-22 re-published snapshots and the
2026-07-09/10 collector cutovers — sit *inside* the trailing
`VALIDATION_WINDOW_DAYS` window, and the span rule voids a horizon-wide band
around each cutover. So the calendar validation window loses rows in proportion
to the horizon:

| horizon | val rows before voiding | after |
|---|---|---|
| 3d | 2,445 | 1,683 |
| 7d | 2,445 | 1,386 |
| 14d | 2,445 | 990 |

Below the 2,000-row floor the split fell through to `tdf.iloc[:int(len*0.8)]`,
which is not a thinner version of the same thing: validation stops being the
recent 30 days and becomes the last 20% of the date-sorted frame — roughly ten
months. Early stopping, the Optuna objective and the classifier's stopping set
are all scored on that window. Measured at the then-production default of
`TRAIN_FEATURE_ROWS=100_000`, three of four horizons took that path; only a 6x
budget escaped it. **The default is now 1_200_000** (2026-08-08), a 12x budget,
so production should clear the floors on its own — the widening is the guard
for a thin frame, not the routine path it was.

The fix widens the calendar window backwards until it clears the floors, which
keeps validation a recent contiguous window. The positional split survives only
for a frame too small for widening to help.
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


def _panel(n_items=70, n_days=206, start=date(2026, 1, 1)):
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n_items):
        price = 10.0 + i
        for d in range(n_days):
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"item_id": f"i{i}",
                         "date": start + timedelta(days=d),
                         "price": price})
    return pd.DataFrame(rows)


def _republish(df, day):
    """Overwrite `day` with the previous day's prices, as the collector did."""
    df = df.copy()
    prev = df[df["date"] == day - timedelta(days=1)].set_index("item_id")["price"]
    mask = df["date"] == day
    df.loc[mask, "price"] = df.loc[mask, "item_id"].map(prev).values
    return df


def _drop_items(df, day, share=0.3):
    """Shrink the collected universe from `day` on — a source cutover."""
    keep = sorted(df["item_id"].unique())[int(len(df["item_id"].unique()) * share):]
    return df[(df["date"] < day) | (df["item_id"].isin(keep))]


def _window(tdf, split_date):
    d = pd.to_datetime(tdf["date"])
    val = tdf[d >= pd.to_datetime(split_date)]
    return len(val), pd.to_datetime(val["date"]).nunique()


class TestChooseValidationSplit:

    def test_a_healthy_frame_keeps_the_default_window(self, forecaster):
        tdf = _panel()
        max_date = pd.to_datetime(tdf["date"].max())

        split, ok = forecaster._choose_validation_split(tdf)

        assert ok
        assert split == max_date - timedelta(
            days=ItemForecaster.VALIDATION_WINDOW_DAYS)

    def test_a_row_starved_window_widens_until_it_clears_the_floor(self, forecaster):
        """The production case: enough dates, not enough rows.

        40 items puts the trailing 30 days under the floor (31 x 40 = 1,240) while
        the 90-day cap can clear it (91 x 40 = 3,640).
        """
        tdf = _panel(n_items=40)
        default = pd.to_datetime(tdf["date"].max()) - timedelta(
            days=ItemForecaster.VALIDATION_WINDOW_DAYS)
        assert _window(tdf, default)[0] < ItemForecaster.MIN_VAL_ROWS, "precondition"

        split, ok = forecaster._choose_validation_split(tdf)

        assert ok
        assert split < default
        rows, _ = _window(tdf, split)
        assert rows >= ItemForecaster.MIN_VAL_ROWS

    def test_a_date_starved_window_widens_until_it_clears_the_floor(self, forecaster):
        """Enough rows, too few distinct dates."""
        tdf = _panel(n_items=900, n_days=97)
        last = pd.to_datetime(tdf["date"].max())
        # Keep only 3 dates inside the trailing window.
        keep = pd.to_datetime(tdf["date"]).isin(
            [last, last - timedelta(days=1), last - timedelta(days=2)])
        inside = pd.to_datetime(tdf["date"]) >= (
            last - timedelta(days=ItemForecaster.VALIDATION_WINDOW_DAYS))
        tdf = tdf[keep | ~inside]
        default = last - timedelta(days=ItemForecaster.VALIDATION_WINDOW_DAYS)
        rows, dates = _window(tdf, default)
        assert rows >= ItemForecaster.MIN_VAL_ROWS, "precondition: rows are fine"
        assert dates < ItemForecaster.MIN_VAL_DATES, "precondition: dates are not"

        split, ok = forecaster._choose_validation_split(tdf)

        assert ok
        assert _window(tdf, split)[1] >= ItemForecaster.MIN_VAL_DATES

    def test_the_widened_window_still_ends_at_the_last_date(self, forecaster):
        """Widening must stay a *recent* window, never a positional slice."""
        tdf = _panel(n_items=40)
        split, _ = forecaster._choose_validation_split(tdf)
        val = tdf[pd.to_datetime(tdf["date"]) >= split]
        assert pd.to_datetime(val["date"]).max() == pd.to_datetime(tdf["date"].max())

    def test_widening_is_capped(self, forecaster):
        """A frame too sparse to ever clear the floor reports failure, and the
        window it reports is still bounded — the caller falls back from here."""
        tdf = _panel(n_items=1, n_days=400, start=date(2025, 6, 1))

        split, ok = forecaster._choose_validation_split(tdf)

        assert not ok
        span = (pd.to_datetime(tdf["date"].max()) - split).days
        assert span <= ItemForecaster.MAX_VALIDATION_WINDOW_DAYS

    def test_an_empty_frame_reports_failure(self, forecaster):
        empty = pd.DataFrame(columns=["item_id", "date", "price"])
        _, ok = forecaster._choose_validation_split(empty)
        assert not ok


class TestBuildProductionSplit:
    """The whole split decision, so `_train_horizon_inline` holds no logic.

    `_train_horizon_inline` is the full Optuna + ensemble path and cannot be
    driven from a test, so the split it uses lives here instead of inline.
    """

    def test_a_healthy_frame_validates_on_the_trailing_window(self, forecaster):
        tdf = _panel().sort_values("date")
        max_date = pd.to_datetime(tdf["date"].max())

        _, val = forecaster._build_production_split(tdf, horizon=7, max_rows=10**6)

        assert pd.to_datetime(val["date"]).min() == max_date - timedelta(
            days=ItemForecaster.VALIDATION_WINDOW_DAYS)
        assert pd.to_datetime(val["date"]).max() == max_date

    @pytest.mark.parametrize("horizon", [3, 7, 14, 30])
    def test_train_rows_labelled_inside_the_window_are_purged(self, forecaster,
                                                              horizon):
        tdf = _panel().sort_values("date")

        train, val = forecaster._build_production_split(
            tdf, horizon=horizon, max_rows=10**6)

        latest_label = pd.to_datetime(train["date"]).max() + timedelta(days=horizon)
        assert latest_label < pd.to_datetime(val["date"]).min()

    def test_a_starved_window_validates_on_a_widened_recent_window(self, forecaster):
        tdf = _panel(n_items=40).sort_values("date")
        max_date = pd.to_datetime(tdf["date"].max())

        _, val = forecaster._build_production_split(tdf, horizon=7, max_rows=10**6)

        span = (max_date - pd.to_datetime(val["date"]).min()).days
        assert span > ItemForecaster.VALIDATION_WINDOW_DAYS, "should have widened"
        assert span <= ItemForecaster.MAX_VALIDATION_WINDOW_DAYS
        assert pd.to_datetime(val["date"]).max() == max_date
        assert len(val) >= ItemForecaster.MIN_VAL_ROWS

    def test_a_frame_too_small_to_widen_still_falls_back_positionally(self,
                                                                      forecaster):
        """The safety net stays for frames widening cannot rescue."""
        tdf = _panel(n_items=1, n_days=400, start=date(2025, 6, 1)).sort_values("date")

        train, val = forecaster._build_production_split(
            tdf, horizon=7, max_rows=10**6)

        assert len(val) == len(tdf) - int(len(tdf) * 0.8)

    def test_the_positional_fallback_is_still_purged(self, forecaster):
        tdf = _panel(n_items=1, n_days=400, start=date(2025, 6, 1)).sort_values("date")

        train, val = forecaster._build_production_split(
            tdf, horizon=7, max_rows=10**6)

        latest_label = pd.to_datetime(train["date"]).max() + timedelta(days=7)
        assert latest_label < pd.to_datetime(val["date"]).min()

    def test_the_train_row_cap_is_respected(self, forecaster):
        tdf = _panel().sort_values("date")

        train, _ = forecaster._build_production_split(
            tdf, horizon=7, max_rows=500)

        assert len(train) == 500

    def test_the_capped_train_set_keeps_the_full_calendar_window(self, forecaster):
        """Sampling must be random, never `tail()` — truncating the calendar
        window silently disables expanding-window CV."""
        tdf = _panel().sort_values("date")
        uncapped, _ = forecaster._build_production_split(
            tdf, horizon=7, max_rows=10**6)

        capped, _ = forecaster._build_production_split(
            tdf, horizon=7, max_rows=500)

        assert pd.to_datetime(capped["date"]).min() == pd.to_datetime(
            uncapped["date"]).min()


class TestVoidedLabelsNoLongerCollapseTheSplit:
    """End-to-end over the real fabricated-day shapes, no LightGBM."""

    @pytest.fixture
    def voided(self):
        """A panel carrying both archive defects inside the trailing window."""
        df = _panel(n_items=70)
        df = _drop_items(df, date(2026, 7, 9))          # cutover
        df = _republish(df, date(2026, 7, 16))          # snapshot
        return df

    @pytest.mark.parametrize("horizon", [3, 7, 14])
    def test_the_trailing_window_is_thinned_by_voiding(self, forecaster, voided,
                                                       horizon):
        """Precondition for the fix: this is the bug's trigger."""
        tdf = forecaster.prepare_targets(voided, horizon)
        kept = tdf.dropna(subset=[f"target_return_{horizon}d"])
        default = pd.to_datetime(kept["date"].max()) - timedelta(
            days=ItemForecaster.VALIDATION_WINDOW_DAYS)

        before = _window(tdf, default)[0]
        after = _window(kept, default)[0]
        assert after < before, "voiding must thin the trailing window"

    @pytest.mark.parametrize("horizon", [3, 7, 14, 30])
    def test_voiding_never_forces_the_positional_fallback(self, forecaster,
                                                          voided, horizon):
        tdf = forecaster.prepare_targets(voided, horizon)
        tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).sort_values("date")

        split, ok = forecaster._choose_validation_split(tdf)

        assert ok, f"{horizon}d still falls through to the positional split"
        rows, dates = _window(tdf, split)
        assert rows >= ItemForecaster.MIN_VAL_ROWS
        assert dates >= ItemForecaster.MIN_VAL_DATES

    @pytest.mark.parametrize("horizon", [3, 7, 14, 30])
    def test_the_purge_is_applied_at_the_widened_split(self, forecaster, voided,
                                                       horizon):
        """Widening must not reintroduce the leak the purge closes."""
        tdf = forecaster.prepare_targets(voided, horizon)
        tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).sort_values("date")
        split, _ = forecaster._choose_validation_split(tdf)

        train = tdf[pd.to_datetime(tdf["date"]) < split]
        train = forecaster._purge_overlapping_train_rows(train, split, horizon)

        assert not train.empty
        latest = pd.to_datetime(train["date"]).max()
        assert latest + timedelta(days=horizon) < pd.to_datetime(split)
