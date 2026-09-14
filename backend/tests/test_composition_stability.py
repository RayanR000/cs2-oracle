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

from models.item_parser import (
    STEAM_SPOT_SOURCES,
    TRAILING_WINDOW_SOURCES,
)
from scripts.measure_composition_stability import (
    MIN_DATES_TO_REPORT,
    MIN_ITEMS_PER_DATE,
    NULL_SOURCE_LABEL,
    build_windows,
    paired_difference,
    partitions,
    rank_ic,
    report,
    source_masks,
    verdict,
    voided_anchors,
)

START = date(2026, 1, 1)


def _voted(rows, masks=None) -> pd.DataFrame:
    """`(item_id, day_offset, price, n_ask_sources)` tuples as a voted frame.

    `masks` optionally overrides the source-set bitmask per row; it defaults to
    the lowest `n_ask_sources` bits, i.e. every day drawing on the same sources.
    """
    frame = pd.DataFrame(
        [
            {"item_id": item, "date": START + timedelta(days=offset), "price": price, "n_ask_sources": n}
            for item, offset, price, n in rows
        ]
    )
    frame["source_mask"] = masks if masks is not None else [(1 << n) - 1 for n in frame["n_ask_sources"]]
    return frame


def _series(item: str, offsets, n_ask_sources, price: float = 10.0):
    """One item observed on `offsets`, each day carrying its own source count."""
    if isinstance(n_ask_sources, int):
        n_ask_sources = [n_ask_sources] * len(offsets)
    # A drifting price keeps every return non-zero and every rank distinct, so
    # a row is never dropped for a degenerate reason the test did not intend.
    return [(item, off, price + 0.1 * i, n) for i, (off, n) in enumerate(zip(offsets, n_ask_sources))]


# --------------------------------------------------------------------------
# The stability predicate: same count on every day of t-1 ... t+h, all present
# --------------------------------------------------------------------------


def test_an_unbroken_window_at_one_source_count_is_stable():
    voted = _voted(_series("a", range(0, 6), 2))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    # h=3 needs t-1 and t+3, so day 0 and days 3-5 cannot anchor: anchors are
    # days 1 and 2.
    assert sorted(windows["date"].dt.date) == [START + timedelta(days=1), START + timedelta(days=2)]
    assert windows["stable"].all()


def test_a_missing_day_inside_the_window_is_not_stable():
    # Day 3 absent. The anchor at day 1 still has both return legs (day 0 and
    # day 4), so it survives as a row -- but its window is not observed.
    voted = _voted(_series("a", [0, 1, 2, 4, 5], 2))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    at_day_1 = windows[windows["date"] == pd.Timestamp(START + timedelta(days=1))]
    assert len(at_day_1) == 1
    assert not at_day_1["stable"].iloc[0], "an absent day is an unobserved composition, not a matching one"


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
    voted = _voted(_series("a", range(0, 6), 2) + _series("b", range(0, 6), [2, 2, 2, 5, 2, 2]))
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
    voted = _voted(_series("a", range(0, 6), 2, price=0.5) + _series("b", range(0, 6), 2, price=10.0))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    assert len(windows) == 2  # both anchors of item b, none of item a


# --------------------------------------------------------------------------
# Rank IC: within date, then averaged across dates
# --------------------------------------------------------------------------


def _cell(per_date: dict[date, list[tuple[float, float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": pd.Timestamp(day), "x": x, "y": y} for day, pairs in per_date.items() for x, y in pairs]
    )


def test_rank_ic_is_within_date_not_pooled():
    # Two dates, each perfectly ANTI-correlated inside itself, but placed at
    # different levels so a pooled Spearman over all ten rows comes out
    # POSITIVE. The within-date answer is -1 on both dates, so the mean is -1.
    cell = _cell(
        {
            date(2026, 1, 1): [(1.0, 30.0), (2.0, 20.0), (3.0, 10.0), (4.0, 5.0), (5.0, 1.0)],
            date(2026, 1, 2): [(10.0, 300.0), (20.0, 200.0), (30.0, 100.0), (40.0, 50.0), (50.0, 10.0)],
        }
    )
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

    expected = [spearmanr(g["x"], g["y"]).statistic for _, g in cell.groupby("date")]

    result = rank_ic(cell)
    assert result["n_dates"] == 4
    assert result["rank_ic"] == pytest.approx(float(np.mean(expected)))
    sd = float(np.std(expected, ddof=1))
    assert result["t"] == pytest.approx(float(np.mean(expected) / (sd / np.sqrt(4))))


def test_a_date_with_too_few_items_does_not_contribute():
    cell = _cell(
        {
            date(2026, 1, 1): [(float(i), float(-i)) for i in range(MIN_ITEMS_PER_DATE)],
            date(2026, 1, 2): [(1.0, 1.0), (2.0, 2.0)],
        }
    )
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
    assert void == {date(2026, 5, 7), date(2026, 5, 8), date(2026, 5, 9), date(2026, 5, 10)}
    # 5-11's window is (5-10, 5-14] -- the cutover is its t-1 leg, so both of
    # its legs are quoted on the post-cutover basis.
    assert date(2026, 5, 11) not in void
    # 5-6's window ends on 5-9, before the cutover.
    assert date(2026, 5, 6) not in void


def test_the_span_rule_widens_with_the_horizon():
    shift = date(2026, 5, 10)
    assert len(voided_anchors(frozenset(), frozenset({shift}), horizon=7)) == 8


# --------------------------------------------------------------------------
# The set basis: what the count basis cannot see
# --------------------------------------------------------------------------


def _two_sources_then_a_swap():
    """Six days at two ask sources, where day 3 swaps one source for another.

    `n_ask_sources` is 2 on every day, so the count basis sees nothing at all.
    """
    rows = _series("a", range(0, 6), 2)
    swapped = [0b011, 0b011, 0b011, 0b101, 0b011, 0b011]
    return _voted(rows, masks=swapped)


def test_a_source_swap_with_the_count_unchanged_reads_as_changed():
    windows = build_windows(_two_sources_then_a_swap(), horizon=3, min_price=1.0, basis="set")

    # Both anchors (days 1 and 2) have day 3 inside their window.
    assert len(windows) == 2
    assert not windows["stable"].any(), (
        "swapping one source for another is a composition change, and it is the case the count basis is blind to"
    )


def test_the_count_basis_is_blind_to_the_swap():
    # The same frame, read on the other basis. This is not a bug in the count
    # basis, it is its known limitation -- pinned so the difference between the
    # two published measurements stays visible.
    windows = build_windows(_two_sources_then_a_swap(), horizon=3, min_price=1.0, basis="count")

    assert windows["stable"].all()


def test_an_unchanged_set_is_stable_on_the_set_basis():
    rows = _series("a", range(0, 6), 2)
    windows = build_windows(_voted(rows, masks=[0b011] * 6), horizon=3, min_price=1.0, basis="set")

    assert windows["stable"].all()


# --------------------------------------------------------------------------
# The source set itself, built from the frame the vote consumed
# --------------------------------------------------------------------------


def _raw(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [{"item_id": item, "date": START + timedelta(days=offset), "source": source} for item, offset, source in rows]
    )


def test_the_mask_encodes_the_set_not_the_row_count():
    masks = source_masks(
        _raw(
            [
                ("a", 0, "skinport"),
                ("a", 0, "skinport"),
                ("a", 0, "buff163"),
                ("b", 0, "skinport"),
                ("b", 0, "buff163"),
            ]
        )
    )

    by_item = dict(zip(masks["item_id"], masks["source_mask"]))
    assert by_item["a"] == by_item["b"], "a duplicate row from a source already in the set must not change it"


def test_different_sets_of_the_same_size_get_different_masks():
    masks = source_masks(
        _raw(
            [
                ("a", 0, "skinport"),
                ("a", 0, "buff163"),
                ("b", 0, "skinport"),
                ("b", 0, "youpin"),
            ]
        )
    )

    by_item = dict(zip(masks["item_id"], masks["source_mask"]))
    assert by_item["a"] != by_item["b"]
    assert np.bitwise_count(by_item["a"]) == np.bitwise_count(by_item["b"]) == 2


def test_a_trailing_window_source_does_not_count_toward_the_mask():
    # `_apply_multi_source_voting` drops `TRAILING_WINDOW_SOURCES` before it
    # counts `n_ask_sources`, so the mask must drop them too, or
    # `load_voted_series`'s `bitwise_count(source_mask) == n_ask_sources`
    # guard fires on every item-day where one sat alongside a real ask.
    trailing = next(iter(TRAILING_WINDOW_SOURCES))
    masks = source_masks(
        _raw(
            [
                ("a", 0, "skinport"),
                ("a", 0, trailing),
                ("b", 0, "skinport"),
            ]
        )
    )

    by_item = dict(zip(masks["item_id"], masks["source_mask"]))
    assert by_item["a"] == by_item["b"], "a trailing-window source must not appear in the set at all"
    assert np.bitwise_count(by_item["a"]) == 1


def test_an_item_day_that_is_only_a_trailing_window_source_has_no_mask_row():
    # Mirrors the vote's own early-return: an item-day with nothing but a
    # trailing-window source produces no row in `voted` either, so the two
    # must agree that it does not exist rather than one reporting an empty set.
    trailing = next(iter(TRAILING_WINDOW_SOURCES))
    masks = source_masks(_raw([("a", 0, trailing)]))

    assert masks.empty


def test_a_null_source_is_a_name_like_any_other():
    # Two NULL-source days must read as the SAME composition. Treating NULL as
    # never equal to itself is what made every pre-2026 row look like a
    # composition change in the refuted 2026-08-08 measurement.
    masks = source_masks(_raw([("a", 0, None), ("b", 0, None), ("c", 0, NULL_SOURCE_LABEL)]))

    assert masks["source_mask"].nunique() == 1
    assert (np.bitwise_count(masks["source_mask"].to_numpy()) == 1).all()


# --------------------------------------------------------------------------
# The far end of the window, which an off-by-one leaves unpinned
# --------------------------------------------------------------------------


def test_a_change_on_exactly_t_plus_h_is_not_stable():
    # The change sits on day 5. For the anchor at day 2 that is exactly t+h;
    # for the anchor at day 1 it is t+h+1, one day past the window. A window
    # built as `range(-1, horizon)` instead of `range(-1, horizon + 1)` keeps
    # both stable and every other test in this file still passes.
    voted = _voted(_series("a", range(0, 7), [2, 2, 2, 2, 2, 3, 3]))
    windows = build_windows(voted, horizon=3, min_price=1.0)

    by_day = dict(zip(windows["date"].dt.date, windows["stable"]))
    assert not by_day[START + timedelta(days=2)], "t+h is inside the window"
    assert by_day[START + timedelta(days=1)], "t+h+1 is outside it"


# --------------------------------------------------------------------------
# Three-way split: "not stable" is two different claims
# --------------------------------------------------------------------------


def test_a_gapped_window_is_not_counted_as_a_composition_change():
    # Item a holds its composition but is missing day 3; item b is observed
    # every day and changes composition on day 3.
    voted = _voted(_series("a", [0, 1, 2, 4, 5], 2) + _series("b", range(0, 6), [2, 2, 2, 3, 2, 2]))
    windows = build_windows(voted, horizon=3, min_price=1.0)
    cells = dict(partitions(windows))

    changed = cells["composition changed (present)"]
    incomplete = cells["window incomplete"]

    assert len(changed) == 2 and len(incomplete) == 2
    assert cells["composition stable"].empty
    # Every row lands in exactly one of the three.
    assert len(cells["composition stable"]) + len(changed) + len(incomplete) == len(cells["all rows"])


def test_the_three_stability_cells_partition_the_rows():
    voted = _voted(
        _series("a", range(0, 8), 2)
        + _series("b", [0, 1, 2, 4, 5, 6, 7], 2)
        + _series("c", range(0, 8), [1, 1, 2, 2, 3, 3, 1, 1])
    )
    cells = dict(partitions(build_windows(voted, horizon=3, min_price=1.0)))

    total = (
        len(cells["composition stable"]) + len(cells["composition changed (present)"]) + len(cells["window incomplete"])
    )
    assert total == len(cells["all rows"])


# --------------------------------------------------------------------------
# The reporting floor, which is the brief's hardest constraint
# --------------------------------------------------------------------------


def test_a_cell_below_the_floor_is_underpowered():
    assert verdict(MIN_DATES_TO_REPORT - 1) == "underpowered"
    assert verdict(MIN_DATES_TO_REPORT) == "measured"
    assert verdict(0) == "underpowered"


def test_an_underpowered_cell_prints_no_number(capsys):
    results = [
        {
            "cell": "stable & >=3 sources",
            "n_dates": MIN_DATES_TO_REPORT - 1,
            "n_rows": 407_437,
            "rank_ic": 0.1234,
            "t": 9.87,
            "verdict": "underpowered",
        }
    ]
    paired = {"n_dates": 0, "difference": None, "t": None, "verdict": "underpowered"}

    report(results, paired, horizon=3, start=date(2026, 1, 1), min_price=1.0, basis="set")

    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("stable & >=3 sources"))
    assert "0.1234" not in line and "9.9" not in line and "9.87" not in line
    assert "--" in line and "underpowered" in line


def test_a_cell_above_the_floor_prints_its_number(capsys):
    results = [
        {
            "cell": "composition stable",
            "n_dates": MIN_DATES_TO_REPORT,
            "n_rows": 100,
            "rank_ic": 0.1027,
            "t": 14.1,
            "verdict": "measured",
        }
    ]
    paired = {"n_dates": 0, "difference": None, "t": None, "verdict": "underpowered"}

    report(results, paired, horizon=3, start=date(2026, 1, 1), min_price=1.0, basis="set")

    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("composition stable"))
    assert "+0.1027" in line and "14.1" in line


# --------------------------------------------------------------------------
# The paired difference the headline rests on
# --------------------------------------------------------------------------


def test_the_paired_difference_uses_only_the_common_dates():
    left = _cell(
        {
            date(2026, 1, 1): [(float(i), float(i)) for i in range(6)],
            date(2026, 1, 2): [(float(i), float(i)) for i in range(6)],
            date(2026, 1, 3): [(float(i), float(-i)) for i in range(6)],
        }
    )
    right = _cell(
        {
            date(2026, 1, 1): [(float(i), float(-i)) for i in range(6)],
            date(2026, 1, 2): [(float(i), float(-i)) for i in range(6)],
        }
    )

    result = paired_difference(left, right)

    # 1-3 is in `left` only and must not contribute; on the two shared dates
    # the difference is +1 - (-1) = +2 both times.
    assert result["n_dates"] == 2
    assert result["difference"] == 2.0
    assert result["t"] is None  # zero dispersion -> no t, not an infinite one


def test_a_paired_difference_of_zero_reports_zero_not_none():
    pairs = {date(2026, 1, d): [(float(i), float(i * (-1) ** d)) for i in range(6)] for d in range(1, 5)}
    result = paired_difference(_cell(pairs), _cell(pairs))

    assert result["n_dates"] == 4
    assert result["difference"] == 0.0
    assert result["t"] is None  # zero dispersion is not an infinite t


def test_a_date_too_thin_in_either_cell_drops_out_of_the_pair():
    # 1-2 has plenty of items in `left` and only two in `right`. It cannot
    # contribute an IC on the right, so it must not contribute a difference
    # either -- silently pairing it against nothing would compare a measured
    # date to an absent one.
    left = _cell(
        {
            date(2026, 1, 1): [(float(i), float(i)) for i in range(6)],
            date(2026, 1, 2): [(float(i), float(i)) for i in range(6)],
        }
    )
    right = _cell(
        {date(2026, 1, 1): [(float(i), float(-i)) for i in range(6)], date(2026, 1, 2): [(1.0, 1.0), (2.0, 2.0)]}
    )

    result = paired_difference(left, right)

    assert result["n_dates"] == 1
    assert result["difference"] == 2.0


def test_source_mask_drops_every_source_the_vote_drops():
    """The mask's exclusion set must be the vote's, not a copy of it.

    `STEAM_SPOT_SOURCES` joined `_apply_multi_source_voting`'s exclusion after
    `source_masks` was written, so the mask counted a source that never voted
    and `load_voted_series`'s `bitwise_count == n_ask_sources` guard raised on
    every window containing 2026-08-18 or later. The failure is loud, which is
    why it was survivable; the cost was that no measurement could run over the
    current archive at all.
    """
    day = date(2026, 8, 18)
    rows = pd.DataFrame(
        [
            {"item_id": "a", "date": day, "source": "aggregator_sync", "price": 10.0},
            {"item_id": "a", "date": day, "source": next(iter(STEAM_SPOT_SOURCES)), "price": 10.5},
            {"item_id": "a", "date": day, "source": next(iter(TRAILING_WINDOW_SOURCES)), "price": 9.0},
        ]
    )
    masks = source_masks(rows)
    assert len(masks) == 1
    # One voting source, so one bit -- the spot and trailing rows contribute
    # none, matching the `n_ask_sources = 1` the vote would report.
    assert int(masks.loc[0, "source_mask"]).bit_count() == 1
