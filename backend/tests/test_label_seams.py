"""The pieces of `check_label_seams` that fail silently.

The detector exists to say whether the *label* fabricates market-wide moves at
a source cutover, so each of its parts has to be pinned against a frame whose
answer is known by hand. Each one produces a plausible number when it is wrong:
a return taken across a calendar gap fabricates exactly the move under test, a
"within-source" return that does not require the source on *both* legs is just
the voted return wearing a different name, and a threshold read against a
window that contains the seam is inflated by it.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_label_seams import (  # noqa: E402
    NULL_SOURCE_LABEL,
    daily_median_returns,
    flag_seams,
    within_source_index,
    within_source_returns,
)

START = date(2026, 7, 1)


def _voted(rows) -> pd.DataFrame:
    """`(item_id, day_offset, price)` tuples as a voted daily frame."""
    return pd.DataFrame(
        [{"item_id": i, "date": START + timedelta(days=d), "price": p}
         for i, d, p in rows]
    )


def _raw(rows) -> pd.DataFrame:
    """`(item_id, day_offset, source, price)` tuples as a per-source frame."""
    return pd.DataFrame(
        [{"item_id": i, "date": START + timedelta(days=d), "source": s,
          "price": p}
         for i, d, s, p in rows]
    )


class TestDailyMedianReturns:
    def test_return_needs_consecutive_calendar_days(self):
        """A gap is an unobserved day, not a one-day move.

        An as-of lookup would carry day 0's price to day 2 and report the whole
        two-day move as a daily return -- which is the artifact under test.
        """
        frame = _voted([("a", 0, 10.0), ("a", 2, 12.0),
                        ("b", 0, 10.0), ("b", 1, 11.0)])
        out = daily_median_returns(frame, min_price=1.0)
        assert list(out["date"]) == [pd.Timestamp(START + timedelta(days=1))]
        assert out.loc[0, "n_items"] == 1
        assert out.loc[0, "median_return"] == pytest.approx(0.10)

    def test_price_floor_reads_the_anchor_leg(self):
        """The floor selects the cohort at `t-1`, as the served cohort does.

        Filtering on the *later* price lets a sub-$1 item that mooned into the
        cohort join it, which is selection on the outcome.
        """
        frame = _voted([("cheap", 0, 0.50), ("cheap", 1, 5.00),
                        ("dear", 0, 10.0), ("dear", 1, 10.5)])
        out = daily_median_returns(frame, min_price=1.0)
        assert out.loc[0, "n_items"] == 1
        assert out.loc[0, "median_return"] == pytest.approx(0.05)


class TestWithinSourceReturns:
    def test_source_must_be_present_on_both_legs(self):
        """The seam case: one feed quotes day 0, a different one quotes day 1.

        The voted label reports the level difference between two feeds as a
        return. Within-source has no common source across the pair and so
        reports nothing, which is the whole point of the basis.
        """
        rows = _raw([("a", 0, "aggregator_sync", 10.0),
                     ("a", 1, "aggregator_steam_17mafo", 13.0)])
        assert within_source_returns(rows).empty

    def test_common_source_is_the_only_leg_used(self):
        """A second, one-sided feed must not move the answer.

        `sync` is on both days and is flat; `17mafo` appears only on day 1 at a
        different level. The return is the flat one.
        """
        rows = _raw([("a", 0, "aggregator_sync", 10.0),
                     ("a", 1, "aggregator_sync", 10.0),
                     ("a", 1, "aggregator_steam_17mafo", 13.0)])
        out = within_source_returns(rows)
        assert len(out) == 1
        assert out.loc[0, "return"] == pytest.approx(0.0)

    def test_null_source_is_one_named_source(self):
        """13 years of the archive carry `source IS NULL`.

        Treating NULL as "never equal to itself" would drop the entire
        pre-2026 series from the clean basis and silently turn the comparison
        into 2026-vs-history.
        """
        rows = _raw([("a", 0, None, 10.0), ("a", 1, None, 11.0)])
        out = within_source_returns(rows)
        assert len(out) == 1
        assert out.loc[0, "return"] == pytest.approx(0.10)
        assert out.loc[0, "sources"] == NULL_SOURCE_LABEL

    def test_bids_and_trailing_windows_never_vote(self):
        """The same exclusion the consensus applies, or this is a new label.

        A bid on both legs is a clean within-source return and still must not
        contribute: it is the wrong side of the book, and a trailing mean is
        the wrong time basis.
        """
        rows = _raw([("a", 0, "aggregator_buff163_buy", 10.0),
                     ("a", 1, "aggregator_buff163_buy", 11.0),
                     ("b", 0, "aggregator_steam_30d", 10.0),
                     ("b", 1, "aggregator_steam_30d", 11.0)])
        assert within_source_returns(rows).empty


class TestFlagSeams:
    def _daily(self, moves) -> pd.DataFrame:
        return pd.DataFrame(
            [{"date": START + timedelta(days=d), "n_items": 1000,
              "median_return": m} for d, m in enumerate(moves)]
        )

    def test_a_market_wide_move_is_flagged(self):
        daily = self._daily([0.001, -0.002, 0.2647, -0.1373, 0.000])
        out = flag_seams(daily, threshold=0.05, min_items=100)
        assert list(out["flagged"]) == [False, False, True, True, False]

    def test_ordinary_days_are_not_flagged(self):
        daily = self._daily([0.004, -0.010, 0.022, -0.031, 0.008])
        out = flag_seams(daily, threshold=0.05, min_items=100)
        assert not out["flagged"].any()

    def test_a_thin_date_cannot_flag(self):
        """A 30% median over 4 items is a small-sample artifact, not a market.

        Flagging it would spend the detector's credibility on the cells least
        able to support a claim.
        """
        daily = self._daily([0.30])
        daily.loc[0, "n_items"] = 4
        out = flag_seams(daily, threshold=0.05, min_items=100)
        assert not out["flagged"].any()

    def test_the_robust_scale_excludes_the_flagged_days(self):
        """A seam must not inflate the yardstick it is measured against.

        With the two seam days in the scale, their own z-scores collapse
        toward the bulk and the report understates how anomalous they are.
        """
        daily = self._daily([0.001, -0.002, 0.2647, -0.1373, 0.000, 0.002])
        out = flag_seams(daily, threshold=0.05, min_items=100)
        assert out.loc[2, "robust_z"] > 20


class TestWithinSourceIndex:
    def test_a_seam_moves_the_level_by_nothing(self):
        """The whole point: a source turnover is a zero step, not a jump.

        `sync` quotes day 0 at 10; on day 1 `sync` is gone and `17mafo` quotes
        at 13 (a +30% cross-source offset). The clean index must stay at 10 --
        the level is carried forward, not re-anchored to the other feed.
        """
        rows = _raw([("a", 0, "aggregator_sync", 10.0),
                     ("a", 1, "aggregator_steam_17mafo", 13.0)])
        voted = _voted([("a", 0, 10.0), ("a", 1, 13.0)])
        out = within_source_index(rows, voted).sort_values("date")
        assert list(out["price"].round(6)) == [10.0, 10.0]

    def test_a_within_source_move_compounds(self):
        """A real move -- same source both days -- must pass through.

        `sync` goes 10 -> 11 (+10%) then 11 -> 13.2 (+20%); the index compounds
        to 10, 11, 13.2 regardless of a one-sided second feed on day 1.
        """
        rows = _raw([("a", 0, "aggregator_sync", 10.0),
                     ("a", 1, "aggregator_sync", 11.0),
                     ("a", 1, "aggregator_steam_17mafo", 99.0),
                     ("a", 2, "aggregator_sync", 13.2)])
        voted = _voted([("a", 0, 10.0), ("a", 1, 11.0), ("a", 2, 13.2)])
        out = within_source_index(rows, voted).sort_values("date")
        assert list(out["price"].round(6)) == [10.0, 11.0, 13.2]

    def test_cohort_matches_the_voted_basis(self):
        """One index row per voted item-day, so the two bases stay paired."""
        rows = _raw([("a", 0, "aggregator_sync", 10.0),
                     ("a", 1, "aggregator_sync", 11.0),
                     ("b", 0, "aggregator_sync", 5.0)])
        voted = _voted([("a", 0, 10.0), ("a", 1, 11.0), ("b", 0, 5.0)])
        out = within_source_index(rows, voted)
        assert len(out) == len(voted)
        assert set(map(tuple, out[["item_id"]].assign(
            d=out["date"]).itertuples(index=False, name=None))) == {
            ("a", pd.Timestamp(START)), ("a", pd.Timestamp(START + timedelta(days=1))),
            ("b", pd.Timestamp(START))}
