from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from backtest.price_resolution import (
    MAX_WINDOW_SPAN_DAYS,
    SMOOTH_WINDOW,
    load_voted_prices,
    resolve_anchors,
)


def smoothed_prices(voted, anchors, **kwargs):
    """resolve_anchors projected to prices, for the tests that only care about
    the value. Deliberately test-local: production reads Resolution objects, and
    exporting a price-only wrapper from the module would put two things that
    each look like "the" estimator on the public surface."""
    return {k: r.price for k, r in resolve_anchors(voted, anchors, **kwargs).items()}


def _frame(rows):
    return pd.DataFrame(rows, columns=["item_id", "date", "price"])


def test_flat_series_gives_exactly_the_flat_price():
    """The symmetry property: a flat series must resolve to its own value on
    both legs, so actual_ret is exactly 0.0. The old code could not do this."""
    rows = [("ak", date(2026, 7, d), 2.0) for d in range(1, 11)]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 4)), ("ak", date(2026, 7, 10))})

    base = out[("ak", date(2026, 7, 4))]
    actual = out[("ak", date(2026, 7, 10))]
    assert base == 2.0
    assert actual == 2.0
    assert (actual - base) / base == 0.0


def test_uses_median_of_last_three_observations_at_or_before_anchor():
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 2), 10.0),  # spike
        ("ak", date(2026, 7, 3), 2.0),
        ("ak", date(2026, 7, 4), 3.0),  # after the anchor — must be ignored
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 3))})
    # median(1.0, 10.0, 2.0) == 2.0 — the spike is filtered, 07-04 excluded
    assert out[("ak", date(2026, 7, 3))] == 2.0


def test_lookback_is_row_based_not_calendar_based():
    """Observations on 07-01, 07-05, 07-09 are the last 3 rows even though
    they span 8 calendar days — but that exceeds the staleness cap."""
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 5), 2.0),
        ("ak", date(2026, 7, 9), 3.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 9))})
    assert ("ak", date(2026, 7, 9)) not in out  # span 8d > 7d cap


def test_staleness_cap_rejects_scattered_observations():
    rows = [
        ("ak", date(2026, 5, 1), 1.0),
        ("ak", date(2026, 6, 1), 2.0),
        ("ak", date(2026, 7, 1), 3.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 1))})
    assert out == {}


def test_fewer_than_three_observations_resolve_within_the_span():
    rows = [
        ("ak", date(2026, 7, 8), 4.0),
        ("ak", date(2026, 7, 9), 6.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 9))})
    assert out[("ak", date(2026, 7, 9))] == 5.0  # median(4.0, 6.0)


def test_anchor_before_any_observation_is_unresolvable():
    rows = [("ak", date(2026, 7, 9), 4.0)]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 1))})
    assert out == {}


def test_items_are_independent():
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 2), 1.0),
        ("m4", date(2026, 7, 1), 50.0),
        ("m4", date(2026, 7, 2), 50.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 2)), ("m4", date(2026, 7, 2))})
    assert out[("ak", date(2026, 7, 2))] == 1.0
    assert out[("m4", date(2026, 7, 2))] == 50.0


def test_span_is_measured_from_the_anchor_not_across_the_window():
    """Replaces test_span_is_measured_across_the_window_not_from_the_anchor,
    which asserted the defect Task 8d fixes.

    The span is measured from the ANCHOR to the oldest selected observation.
    Measuring only across the selected window leaves the gap from the newest
    observation to the anchor unbounded, so a tight cluster of observations
    resolves an anchor arbitrarily far in the future — a stale price re-stamped
    as the anchor's own value.

    Observations 07-01/02/03 span 2 days, so the old rule resolved an anchor on
    07-10. From the anchor the oldest observation is 9 days back, past the 7-day
    cap, so it must not resolve."""
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 2), 2.0),
        ("ak", date(2026, 7, 3), 3.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 10))})
    assert out == {}


def test_an_anchor_beyond_archive_coverage_does_not_carry_a_stale_price_forward():
    """Test 1 of the brief — the reproduction. Observations end on 07-25 at 3.0.
    Every one of these anchors used to resolve to a bit-identical 3.0; in the
    Task 9a dry run that was 25,317 of 75,195 targets.

    Asserted on the boundary rather than a single case: an anchor resolves iff
    the OLDEST selected observation is within the cap of it. With observations
    on 07-23/24/25 the oldest of the last three is 07-23, so anchors up to
    07-30 resolve and 07-31 onward do not."""
    rows = [("ak", date(2026, 7, d), 3.0) for d in (23, 24, 25)]

    resolved = {
        anchor: smoothed_prices(_frame(rows), {("ak", anchor)})
        for anchor in [
            date(2026, 7, 26),
            date(2026, 7, 29),
            date(2026, 8, 30),
            date(2027, 1, 1),
        ]
    }
    assert resolved[date(2026, 7, 26)] == {("ak", date(2026, 7, 26)): 3.0}
    assert resolved[date(2026, 7, 29)] == {("ak", date(2026, 7, 29)): 3.0}
    assert resolved[date(2026, 8, 30)] == {}
    assert resolved[date(2027, 1, 1)] == {}

    # The boundary itself: oldest selected observation is 07-23.
    last = date(2026, 7, 23) + timedelta(days=MAX_WINDOW_SPAN_DAYS)
    assert ("ak", last) in smoothed_prices(_frame(rows), {("ak", last)})
    past = last + timedelta(days=1)
    assert smoothed_prices(_frame(rows), {("ak", past)}) == {}


def test_the_anchor_gap_boundary_is_inclusive_at_exactly_max_span_days():
    """Test 2 of the brief. A single observation exactly MAX_WINDOW_SPAN_DAYS
    before the anchor resolves; one day older does not. Driven with one
    observation so the gap under test is unambiguously the anchor gap and not
    the window span, which is 0 either way."""
    obs_day = date(2026, 7, 10)
    rows = [("ak", obs_day, 4.0)]

    on_cap = obs_day + timedelta(days=MAX_WINDOW_SPAN_DAYS)
    assert smoothed_prices(_frame(rows), {("ak", on_cap)}) == {("ak", on_cap): 4.0}

    past_cap = on_cap + timedelta(days=1)
    assert smoothed_prices(_frame(rows), {("ak", past_cap)}) == {}


def test_an_anchor_with_no_observation_on_it_still_resolves_when_recent():
    """Test 3 of the brief, and the legitimate half of the test this change
    replaced. The rule bounds staleness; it does not require an observation on
    the anchor itself. Observations 07-06/07/08 with an anchor on 07-10 are 4
    days stale at the oldest — well inside the cap — so the anchor resolves at
    the median."""
    rows = [
        ("ak", date(2026, 7, 6), 1.0),
        ("ak", date(2026, 7, 7), 2.0),
        ("ak", date(2026, 7, 8), 3.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 10))})
    assert out[("ak", date(2026, 7, 10))] == 2.0  # median(1.0, 2.0, 3.0)


def test_the_anchor_rule_subsumes_the_between_observations_rule():
    """The brief claims the anchor-relative bound makes a separate
    between-observations bound redundant. Every selected observation is at or
    before the anchor, so selected[-1] <= anchor and therefore
    (selected[-1] - selected[0]) <= (anchor - selected[0]). Anything the window
    check would reject, the anchor check rejects too.

    Asserted rather than reasoned about alone: sweep every 3-observation shape
    up to a 12-day reach and confirm no case passes the anchor check while
    failing the window check."""
    start = date(2026, 7, 1)
    for gap_a in range(0, 13):
        for gap_b in range(gap_a, 13):
            for anchor_gap in range(gap_b, 13):
                days = sorted({0, gap_a, gap_b})
                rows = [("ak", start + timedelta(days=d), 1.0) for d in days]
                anchor = start + timedelta(days=anchor_gap)
                out = smoothed_prices(_frame(rows), {("ak", anchor)})

                # sorted({0, gap_a, gap_b}) dedupes, so this is 1-3 observations;
                # either way they all fit the 3-slot window and 0 is the oldest.
                selected_oldest = start
                anchor_ok = (anchor - selected_oldest).days <= MAX_WINDOW_SPAN_DAYS
                window_ok = ((start + timedelta(days=days[-1])) - selected_oldest).days <= MAX_WINDOW_SPAN_DAYS

                assert bool(out) == anchor_ok, (days, anchor_gap)
                # The subsumption itself: anchor-pass implies window-pass.
                assert not (anchor_ok and not window_ok), (days, anchor_gap)


def test_constants_match_the_codebase_staleness_convention():
    from collectors.pipeline import FALLBACK_MAX_AGE_DAYS

    assert SMOOTH_WINDOW == 3
    assert MAX_WINDOW_SPAN_DAYS == FALLBACK_MAX_AGE_DAYS


def test_missing_archive_raises_rather_than_returning_empty(tmp_path):
    """A green run with zero actuals is the failure shape 324cfff removed.
    Resolution must fail loudly instead."""
    with pytest.raises(FileNotFoundError, match="price archive"):
        load_voted_prices(tmp_path / "nope", ["ak"], date(2026, 7, 1), date(2026, 7, 9))


def test_empty_archive_dir_raises_rather_than_returning_empty(tmp_path):
    """A present-but-empty archive directory (no prices-*.parquet files) is a
    second silent-success shape: the directory exists, so a naive `.exists()`
    check alone would pass, and a scan-and-return-empty loader would report a
    green run over zero rows. Assert on the distinct message so this test
    verifies the *second* raise (no parquet files found), not the first
    (directory absent)."""
    archive = tmp_path / "price-archive"
    archive.mkdir()  # exists, but contains no prices-*.parquet
    with pytest.raises(FileNotFoundError, match=r"no prices-\*.parquet"):
        load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 9))


def test_loads_and_votes_multi_source_rows(tmp_path):
    archive = tmp_path / "price-archive"
    archive.mkdir()
    pd.DataFrame(
        {
            "item_slug": ["ak", "ak", "ak"],
            "day": pd.to_datetime([date(2026, 7, 1)] * 3),
            "mean_price": [2.0, 2.1, 90.0],  # 90.0 is the outlier source
            "volume": [10, 10, 10],
            "source": ["a", "b", "c"],
        }
    ).to_parquet(archive / "prices-2026.parquet")

    out = load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 1))

    assert list(out.columns) == ["item_id", "date", "price"]
    assert len(out) == 1  # one row per item-day after voting
    assert out.iloc[0]["price"] < 10.0  # the 90.0 source was voted out


def test_window_dates_before_the_range_are_loaded(tmp_path):
    """Resolving an anchor needs the observations *before* it, so the loader
    must reach back past min_date by the staleness cap."""
    archive = tmp_path / "price-archive"
    archive.mkdir()
    days = pd.to_datetime([date(2026, 6, 28), date(2026, 6, 29), date(2026, 7, 1)])
    pd.DataFrame(
        {
            "item_slug": ["ak"] * 3,
            "day": days,
            "mean_price": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
            "source": ["a", "a", "a"],
        }
    ).to_parquet(archive / "prices-2026.parquet")

    out = load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 1))
    assert len(out) == 3


def test_slug_with_apostrophe_round_trips(tmp_path):
    """The registered-DataFrame JOIN exists specifically because CS2 item
    names contain apostrophes, which the old f-string `IN (...)` list
    hand-escaped. Push a real-shaped apostrophe slug through the loader and
    confirm it comes back rather than being dropped or breaking the query."""
    archive = tmp_path / "price-archive"
    archive.mkdir()
    slug = "Charm | Lil' Buns"
    pd.DataFrame(
        {
            "item_slug": [slug],
            "day": pd.to_datetime([date(2026, 7, 1)]),
            "mean_price": [3.5],
            "volume": [5],
            "source": ["a"],
        }
    ).to_parquet(archive / "prices-2026.parquet")

    out = load_voted_prices(archive, [slug], date(2026, 7, 1), date(2026, 7, 1))

    assert len(out) == 1
    assert out.iloc[0]["item_id"] == slug
    assert out.iloc[0]["price"] == 3.5
