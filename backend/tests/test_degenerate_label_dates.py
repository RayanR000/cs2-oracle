"""Labels must not be built across days the collector fabricated.

Two distinct archive defects, measured 2026-08-06 over all 4,735 archive days:

1. **Re-published snapshots.** 2026-07-16 and 2026-07-22 are byte copies of the
   preceding day across the whole 40k-item cross-section — 100.00% of items carry
   an identical price. Exactly two days fire archive-wide; the next-highest day
   sits at 69.01%, so the 99% threshold falls in a wide empty gap and is not a
   tuned parameter. 2026-07-16 is the day *before* 2026-07-17, the only genuine
   production forecast date of the current model lineage.

   A snapshot day is unusable as a label ENDPOINT (the price is stale) but is
   harmless in the middle of a window — it shifts no level.

2. **Collector cutovers.** The archive is a stitch of source regimes, not a
   continuously-collected panel. On 2026-03-22 the mean market return reads
   -31.6%, on 2026-07-09/10 +17.4% then -17.8%, against ±0.5% on a normal day.
   Those are basis changes between source sets, not price moves.

   A cutover is detected from the ITEM UNIVERSE SIZE, deliberately — prices
   moving cannot change how many items a collector returns, so this can never
   mask a real crash. It fires 12 times in 4,735 days (0.25%): 4 in 2013
   (archive startup), 1 in 2016, 7 in 2026.

   A cutover corrupts any label whose window SPANS it, not just one landing on
   it: the anchor is quoted on the old source basis and the target on the new.
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


def _panel(n_items=50, n_days=60, start=date(2026, 1, 1)):
    rng = np.random.default_rng(3)
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


class TestSnapshotDayDetection:

    def test_a_republished_day_is_flagged(self, forecaster):
        day = date(2026, 1, 20)
        df = _republish(_panel(), day)
        assert day in forecaster._snapshot_dates(df)

    def test_a_normal_day_is_not_flagged(self, forecaster):
        df = _panel()
        assert forecaster._snapshot_dates(df) == frozenset()

    def test_a_partially_flat_day_is_not_flagged(self, forecaster):
        """69.01% is the real archive's next-highest day. It must stay in."""
        day = date(2026, 1, 20)
        df = _panel(n_items=100)
        prev = df[df["date"] == day - timedelta(days=1)].set_index("item_id")["price"]
        mask = (df["date"] == day) & (df["item_id"].isin([f"i{k}" for k in range(69)]))
        df.loc[mask, "price"] = df.loc[mask, "item_id"].map(prev).values
        assert forecaster._snapshot_dates(df) == frozenset()

    def test_a_tiny_cross_section_is_ignored(self, forecaster):
        """Three items all unchanged is an ordinary quiet day, not a snapshot."""
        df = _panel(n_items=3)
        day = date(2026, 1, 20)
        df = _republish(df, day)
        assert forecaster._snapshot_dates(df) == frozenset()


class TestCollectionShiftDetection:

    def test_a_universe_expansion_is_flagged(self, forecaster):
        df = _panel(n_items=50)
        cut = date(2026, 1, 30)
        # Before `cut`, only the first 10 items were collected.
        df = df[(df["date"] >= cut) | (df["item_id"].isin([f"i{k}" for k in range(10)]))]
        assert cut in forecaster._collection_shift_dates(df)

    def test_a_stable_universe_is_not_flagged(self, forecaster):
        assert forecaster._collection_shift_dates(_panel()) == frozenset()

    def test_a_price_crash_is_never_flagged(self, forecaster):
        """THE LOAD-BEARING TEST.

        A real market-wide crash must survive. The detector reads the item
        universe, never prices, precisely so that it cannot delete the events
        the model exists to learn from.
        """
        df = _panel()
        crash = date(2026, 1, 25)
        df.loc[df["date"] >= crash, "price"] *= 0.4  # -60% market-wide, permanent
        assert forecaster._collection_shift_dates(df) == frozenset()
        assert forecaster._snapshot_dates(df) == frozenset()


class TestLabelsSkipDegenerateDates:

    def test_a_target_landing_on_a_snapshot_day_is_dropped(self, forecaster):
        snap = date(2026, 2, 10)
        df = _republish(_panel(n_days=90), snap)
        out = forecaster.prepare_targets(df, horizon=7)
        anchors = out.loc[out["target_return_7d"].notna(), "date"]
        assert snap - timedelta(days=7) not in set(anchors)

    def test_an_anchor_on_a_snapshot_day_is_dropped(self, forecaster):
        snap = date(2026, 2, 10)
        df = _republish(_panel(n_days=90), snap)
        out = forecaster.prepare_targets(df, horizon=7)
        anchors = set(out.loc[out["target_return_7d"].notna(), "date"])
        assert snap not in anchors

    def test_a_window_spanning_a_cutover_is_dropped(self, forecaster):
        """The whole horizon-wide band of anchors before a cutover must go —
        each of their targets is quoted on the other side of the basis change."""
        df = _panel(n_items=50, n_days=90)
        cut = date(2026, 2, 15)
        df = df[(df["date"] >= cut) | (df["item_id"].isin([f"i{k}" for k in range(10)]))]

        out = forecaster.prepare_targets(df, horizon=7)
        anchors = set(out.loc[out["target_return_7d"].notna(), "date"])
        spanning = {cut - timedelta(days=k) for k in range(1, 8)}
        assert not (anchors & spanning), sorted(anchors & spanning)

    def test_a_snapshot_day_inside_a_window_is_harmless(self, forecaster):
        """A republished day shifts no level, so a 30d label stepping over it is
        still valid. Only endpoints are affected."""
        snap = date(2026, 2, 10)
        df = _republish(_panel(n_days=120), snap)
        out = forecaster.prepare_targets(df, horizon=30)
        anchors = set(out.loc[out["target_return_30d"].notna(), "date"])
        assert snap - timedelta(days=15) in anchors

    def test_clean_data_keeps_every_label(self, forecaster):
        """No silent attrition on a well-formed panel."""
        df = _panel(n_days=90)
        out = forecaster.prepare_targets(df, horizon=7)
        n_expected = len(df[df["date"] <= df["date"].max() - timedelta(days=7)])
        assert out["target_return_7d"].notna().sum() == n_expected
