"""The pieces of `measure_composition_stability` that fail silently.

Each of the three produces a plausible number when it is wrong: a stability
predicate that accepts a gapped window quietly measures a different partition,
a pooled Spearman quietly measures the market factor instead of the
cross-section, and an endpoint-only exclusion quietly leaves windows that span
a source cutover in every cell. All three are pinned here against frames whose
answer is known by hand.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.measure_composition_stability import (  # noqa: E402
    MIN_ITEMS_PER_DATE,
    build_windows,
    rank_ic,
    voided_anchors,
)

START = date(2026, 1, 1)


def _voted(rows) -> pd.DataFrame:
    """`(item_id, day_offset, price, n_ask_sources)` tuples as a voted frame."""
    return pd.DataFrame(
        [{"item_id": item, "date": START + timedelta(days=offset),
          "price": price, "n_ask_sources": n}
         for item, offset, price, n in rows]
    )


def _series(item: str, offsets, n_ask_sources, price: float = 10.0):
    """One item observed on `offsets`, each day carrying its own source count."""
    if isinstance(n_ask_sources, int):
        n_ask_sources = [n_ask_sources] * len(offsets)
    # A drifting price keeps every return non-zero and every rank distinct, so
    # a row is never dropped for a degenerate reason the test did not intend.
    return [(item, off, price + 0.1 * i, n)
            for i, (off, n) in enumerate(zip(offsets, n_ask_sources))]


# --------------------------------------------------------------------------
# The stability predicate: same count on every day of t-1 ... t+h, all present
# --------------------------------------------------------------------------

def test_an_unbroken_window_at_one_source_count_is_stable():
    voted = _voted(_series("a", range(0, 6), 2))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    # h=3 needs t-1 and t+3, so day 0 and days 3-5 cannot anchor: anchors are
    # days 1 and 2.
    assert sorted(windows["date"].dt.date) == [START + timedelta(days=1),
                                               START + timedelta(days=2)]
    assert windows["stable"].all()


def test_a_missing_day_inside_the_window_is_not_stable():
    # Day 3 absent. The anchor at day 1 still has both return legs (day 0 and
    # day 4), so it survives as a row -- but its window is not observed.
    voted = _voted(_series("a", [0, 1, 2, 4, 5], 2))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    at_day_1 = windows[windows["date"] == pd.Timestamp(START + timedelta(days=1))]
    assert len(at_day_1) == 1
    assert not at_day_1["stable"].iloc[0], (
        "an absent day is an unobserved composition, not a matching one")


def test_a_count_change_inside_the_window_is_not_stable():
    # Day 3 has three ask sources; every other day has two.
    voted = _voted(_series("a", range(0, 6), [2, 2, 2, 3, 2, 2]))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    # Anchor day 1 spans days 0-4 and anchor day 2 spans days 1-5; both
    # contain day 3.
    assert not windows["stable"].any()


def test_a_count_change_outside_the_window_leaves_it_stable():
    # The change is on day 6, past the last anchor's window (days 1-5).
    voted = _voted(_series("a", range(0, 8), [2, 2, 2, 2, 2, 2, 3, 3]))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    at_day_2 = windows[windows["date"] == pd.Timestamp(START + timedelta(days=2))]
    assert at_day_2["stable"].iloc[0]


def test_the_window_includes_the_day_before_the_anchor():
    # Only day 0 differs. If the window started at t rather than t-1, the
    # anchor at day 1 would read stable.
    voted = _voted(_series("a", range(0, 6), [3, 2, 2, 2, 2, 2]))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    at_day_1 = windows[windows["date"] == pd.Timestamp(START + timedelta(days=1))]
    assert not at_day_1["stable"].iloc[0]


def test_stability_is_per_item_not_per_date():
    voted = _voted(_series("a", range(0, 6), 2) + _series("b", range(0, 6),
                                                          [2, 2, 2, 5, 2, 2]))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    # Item a holds still, item b does not; both anchor on day 1, so the flag
    # cannot be a property of the date.
    at_day_1 = windows[windows["date"] == pd.Timestamp(START + timedelta(days=1))]
    assert sorted(at_day_1["stable"]) == [False, True]


def test_returns_use_exact_calendar_days():
    # Day 0 absent, so the anchor at day 1 has no t-1 leg and must not fall
    # back to an earlier price.
    voted = _voted(_series("a", [1, 2, 3, 4, 5], 2))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    assert (windows["date"] != pd.Timestamp(START + timedelta(days=1))).all()


def test_the_price_floor_applies_to_the_anchor_day():
    voted = _voted(_series("a", range(0, 6), 2, price=0.5)
                   + _series("b", range(0, 6), 2, price=10.0))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    assert len(windows) == 2  # both anchors of item b, none of item a


# --------------------------------------------------------------------------
# Rank IC: within date, then averaged across dates
# --------------------------------------------------------------------------

def _cell(per_date: dict[date, list[tuple[float, float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": pd.Timestamp(day), "x": x, "y": y}
         for day, pairs in per_date.items() for x, y in pairs]
    )


def test_rank_ic_is_within_date_not_pooled():
    # Two dates, each perfectly ANTI-correlated inside itself, but placed at
    # different levels so a pooled Spearman over all ten rows comes out
    # POSITIVE. The within-date answer is -1 on both dates, so the mean is -1.
    cell = _cell({
        date(2026, 1, 1): [(1.0, 30.0), (2.0, 20.0), (3.0, 10.0),
                           (4.0, 5.0), (5.0, 1.0)],
        date(2026, 1, 2): [(10.0, 300.0), (20.0, 200.0), (30.0, 100.0),
                           (40.0, 50.0), (50.0, 10.0)],
    })
    pooled = cell["x"].corr(cell["y"], method="spearman")
    assert pooled > 0, "the pooled read must get the sign wrong, or the test is empty"

    result = rank_ic(cell)
    assert result["n_dates"] == 2
    assert result["rank_ic"] == -1.0


def test_rank_ic_matches_scipy_per_date():
    rng = np.random.default_rng(0)
    per_date = {}
    for day in range(4):
        n = 12
        x = rng.normal(size=n)
        per_date[date(2026, 1, 1 + day)] = list(zip(x, 0.4 * x + rng.normal(size=n)))
    cell = _cell(per_date)

    from scipy.stats import spearmanr
    expected = [spearmanr(g["x"], g["y"]).statistic
                for _, g in cell.groupby("date")]

    result = rank_ic(cell)
    assert result["n_dates"] == 4
    assert result["rank_ic"] == pytest.approx(float(np.mean(expected)))
    sd = float(np.std(expected, ddof=1))
    assert result["t"] == pytest.approx(float(np.mean(expected) / (sd / np.sqrt(4))))


def test_a_date_with_too_few_items_does_not_contribute():
    cell = _cell({
        date(2026, 1, 1): [(float(i), float(-i)) for i in range(MIN_ITEMS_PER_DATE)],
        date(2026, 1, 2): [(1.0, 1.0), (2.0, 2.0)],
    })
    result = rank_ic(cell)

    assert result["n_dates"] == 1
    assert result["rank_ic"] == -1.0


def test_a_date_with_no_variation_does_not_contribute():
    # A date where every y is identical has no defined correlation; counting it
    # as zero would be an assertion the data does not make.
    flat = [(float(i), 7.0) for i in range(MIN_ITEMS_PER_DATE)]
    ranked = [(float(i), float(i)) for i in range(MIN_ITEMS_PER_DATE)]
    result = rank_ic(_cell({date(2026, 1, 1): flat, date(2026, 1, 2): ranked}))

    assert result["n_dates"] == 1
    assert result["rank_ic"] == 1.0


def test_an_empty_cell_reports_no_dates_and_no_number():
    result = rank_ic(pd.DataFrame(columns=["date", "x", "y"]))

    assert result["n_dates"] == 0
    assert result["rank_ic"] is None
    assert result["t"] is None


# --------------------------------------------------------------------------
# The label-voiding exclusion: an endpoint rule and a span rule
# --------------------------------------------------------------------------

def test_a_snapshot_voids_only_the_anchors_that_read_it_as_an_endpoint():
    snapshot = date(2026, 5, 10)
    void = voided_anchors(frozenset({snapshot}), frozenset(), horizon=3)

    # t-1 leg, the anchor itself, and the target leg.
    assert void == {date(2026, 5, 11), snapshot, date(2026, 5, 7)}
    # A copy sitting mid-window shifts no level, so 5-9 and 5-8 survive.
    assert date(2026, 5, 9) not in void
    assert date(2026, 5, 8) not in void


def test_a_collection_shift_voids_every_anchor_whose_window_spans_it():
    shift = date(2026, 5, 10)
    void = voided_anchors(frozenset(), frozenset({shift}), horizon=3)

    # Windows (t-1, t+3] containing 5-10: anchors 5-7 through 5-10.
    assert void == {date(2026, 5, 7), date(2026, 5, 8),
                    date(2026, 5, 9), date(2026, 5, 10)}
    # 5-11's window is (5-10, 5-14] -- the cutover is its t-1 leg, so both of
    # its legs are quoted on the post-cutover basis.
    assert date(2026, 5, 11) not in void
    # 5-6's window ends on 5-9, before the cutover.
    assert date(2026, 5, 6) not in void


def test_the_span_rule_widens_with_the_horizon():
    shift = date(2026, 5, 10)
    assert len(voided_anchors(frozenset(), frozenset({shift}), horizon=7)) == 8
