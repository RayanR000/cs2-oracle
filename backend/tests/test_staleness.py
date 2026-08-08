"""Run-length semantics for `models/staleness.py`.

The rules under test are the ones a caller can get wrong silently: what breaks
a run, what a NULL does, and that the result lines up with the caller's own row
order rather than the sorted order the scan needs internally.
"""

import numpy as np
import pandas as pd
import pytest

from models.staleness import (
    STALE_RUN_GAP_BREAK_DAYS,
    stale_run_days,
    stale_run_lookup,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["item_id", "date", "price"])


def _days(start, n):
    return list(pd.date_range(start, periods=n, freq="D").date)


def test_fresh_level_is_zero_and_repeats_count_up():
    d = _days("2026-01-01", 4)
    df = _frame([("a", d[0], 1.0), ("a", d[1], 1.0),
                 ("a", d[2], 1.0), ("a", d[3], 2.0)])
    assert list(stale_run_days(df)) == [0, 1, 2, 0]


def test_run_restarts_when_the_price_returns_to_an_earlier_level():
    """A->B->A is three fresh levels, not a continued run.

    The comparison is against the PREVIOUS observation only. A run is a
    republished number, so a price that moved away and came back was observed
    moving and is not stale.
    """
    d = _days("2026-01-01", 4)
    df = _frame([("a", d[0], 1.0), ("a", d[1], 2.0),
                 ("a", d[2], 1.0), ("a", d[3], 1.0)])
    assert list(stale_run_days(df)) == [0, 0, 0, 1]


def test_runs_do_not_cross_items():
    d = _days("2026-01-01", 2)
    df = _frame([("a", d[0], 5.0), ("a", d[1], 5.0),
                 ("b", d[0], 5.0), ("b", d[1], 5.0)])
    assert list(stale_run_days(df)) == [0, 1, 0, 1]


def test_a_gap_at_the_break_width_still_continues_the_run():
    """The bound is inclusive: <= STALE_RUN_GAP_BREAK_DAYS continues."""
    start = pd.Timestamp("2026-01-01").date()
    later = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        int(STALE_RUN_GAP_BREAK_DAYS), unit="D")
    df = _frame([("a", start, 3.0), ("a", later.date(), 3.0)])
    assert list(stale_run_days(df)) == [0, 1]


def test_a_gap_wider_than_the_break_width_starts_a_fresh_level():
    """The collector-outage case. Nothing was observed in between to be frozen."""
    start = pd.Timestamp("2026-01-01").date()
    later = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        int(STALE_RUN_GAP_BREAK_DAYS) + 1, unit="D")
    df = _frame([("a", start, 3.0), ("a", later.date(), 3.0)])
    assert list(stale_run_days(df)) == [0, 0]


def test_a_null_price_breaks_the_run_and_scores_zero():
    """A NULL must never be shown identical to anything.

    Continuing across it would let one collection hole manufacture a run
    spanning the whole series.
    """
    d = _days("2026-01-01", 4)
    df = _frame([("a", d[0], 1.0), ("a", d[1], np.nan),
                 ("a", d[2], 1.0), ("a", d[3], 1.0)])
    assert list(stale_run_days(df)) == [0, 0, 0, 1]


def test_a_null_date_scores_zero_and_does_not_break_a_run_around_it():
    """An undated row is not an observation, so it neither runs nor interrupts.

    Contrast with a null *price*, which breaks the run: that row was observed,
    it just cannot be shown identical to its neighbour. An undated row was
    never placed on the timeline at all, and letting it split two genuinely
    consecutive days would under-report staleness.
    """
    d = _days("2026-01-01", 3)
    df = _frame([("a", d[0], 1.0), ("a", None, 1.0), ("a", d[1], 1.0)])
    assert list(stale_run_days(df)) == [0, 0, 1]


def test_result_is_aligned_to_the_callers_row_order_not_the_sorted_one():
    """Assigning the result back onto an unsorted frame must not scramble it.

    The scan has to sort internally; a positional return would silently attach
    each row's run length to some other row.
    """
    d = _days("2026-01-01", 3)
    df = _frame([("a", d[2], 1.0), ("a", d[0], 1.0), ("a", d[1], 1.0)])
    out = stale_run_days(df)
    assert list(out.index) == list(df.index)
    # Chronologically the prices are 1.0, 1.0, 1.0 on d0/d1/d2 -> 0, 1, 2.
    assert list(out) == [2, 0, 1]


def test_a_non_default_index_is_preserved():
    d = _days("2026-01-01", 3)
    df = _frame([("a", d[0], 1.0), ("a", d[1], 1.0), ("a", d[2], 1.0)])
    df.index = [10, 20, 30]
    out = stale_run_days(df)
    assert list(out.index) == [10, 20, 30]
    assert list(out) == [0, 1, 2]


def test_empty_frame_returns_an_empty_int_series():
    out = stale_run_days(_frame([]))
    assert out.empty
    assert out.dtype == np.int32


def test_missing_column_raises_rather_than_returning_zeros():
    """Silent zeros would read as 'nothing is stale', the wrong direction."""
    df = pd.DataFrame({"item_id": ["a"], "date": _days("2026-01-01", 1)})
    with pytest.raises(KeyError, match="price"):
        stale_run_days(df)


def test_exact_equality_not_approximate():
    """A one-cent move is a move. The artifact is a republished number."""
    d = _days("2026-01-01", 2)
    df = _frame([("a", d[0], 1.00), ("a", d[1], 1.01)])
    assert list(stale_run_days(df)) == [0, 0]


def test_custom_column_names():
    d = _days("2026-01-01", 2)
    df = pd.DataFrame({"slug": ["a", "a"], "day": d, "mean_price": [2.0, 2.0]})
    out = stale_run_days(df, item_col="slug", date_col="day", price_col="mean_price")
    assert list(out) == [0, 1]


def test_lookup_keys_on_plain_dates():
    """Callers hold `datetime.date` anchors; a Timestamp key would never match."""
    d = _days("2026-01-01", 3)
    df = _frame([("a", d[0], 1.0), ("a", d[1], 1.0), ("a", d[2], 2.0)])
    lookup = stale_run_lookup(df)
    assert lookup[("a", d[0])] == 0
    assert lookup[("a", d[1])] == 1
    assert lookup[("a", d[2])] == 0


def test_lookup_drops_null_dated_rows_rather_than_keying_on_nat():
    d = _days("2026-01-01", 1)
    df = _frame([("a", d[0], 1.0), ("a", None, 1.0)])
    lookup = stale_run_lookup(df)
    assert list(lookup) == [("a", d[0])]


def test_lookup_of_an_empty_frame_is_empty():
    assert stale_run_lookup(_frame([])) == {}
