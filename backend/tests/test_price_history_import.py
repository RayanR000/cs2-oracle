"""Tests for the historical price-source import."""
from datetime import date, datetime

import pytest

from collectors.price_history_sources import cs2_prices_tracker as tracker
from collectors.price_history_import import (
    StalledSourceError,
    detect_stalled_days,
)


def test_source_label_is_exact():
    assert tracker.SOURCE == "tracker_steam_24h"


def test_day_url_uses_the_dated_price_file():
    url = tracker.day_url(date(2025, 6, 15))
    assert url == (
        "https://raw.githubusercontent.com/LukeX404/cs2-prices-tracker/"
        "main/static/prices/date/2025-06-15.json"
    )


def test_parse_day_reads_last_24h_only():
    payload = {
        "AK-47 | Redline (Field-Tested)": {
            "steam": {"last_24h": 12.5, "last_7d": 99.0, "last_30d": 98.0}
        }
    }
    assert tracker.parse_day(payload, date(2025, 6, 15)) == [
        ("AK-47 | Redline (Field-Tested)", date(2025, 6, 15), 12.5)
    ]


def test_parse_day_treats_null_last_24h_as_an_absent_row():
    payload = {
        "Sticker | Sherry": {"steam": {"last_24h": None, "last_7d": 4.0}},
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]


def test_parse_day_drops_non_positive_prices_rather_than_writing_zero():
    payload = {
        "Zero Item": {"steam": {"last_24h": 0.0}},
        "Negative Item": {"steam": {"last_24h": -1.0}},
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]


def test_parse_day_tolerates_a_missing_or_null_steam_object():
    payload = {
        "No Steam Key": {"buff": {"price": 3.0}},
        "Null Steam": {"steam": None},
        "Null Item": None,
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]


def test_detect_stalled_days_finds_consecutive_identical_files():
    digests = {
        date(2026, 7, 26): "aaa",
        date(2026, 7, 27): "bbb",
        date(2026, 7, 28): "bbb",
        date(2026, 7, 29): "bbb",
        date(2026, 7, 30): "ccc",
    }
    assert detect_stalled_days(digests) == [
        [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]
    ]


def test_detect_stalled_days_ignores_identical_files_that_are_not_adjacent():
    digests = {
        date(2025, 6, 1): "aaa",
        date(2025, 6, 2): "bbb",
        date(2025, 6, 3): "aaa",
    }
    assert detect_stalled_days(digests) == []


def test_detect_stalled_days_is_empty_for_all_distinct_files():
    digests = {date(2025, 6, d): f"h{d}" for d in range(1, 6)}
    assert detect_stalled_days(digests) == []


def test_stalled_source_error_is_raisable_with_the_groups():
    with pytest.raises(StalledSourceError):
        raise StalledSourceError([[date(2026, 7, 27), date(2026, 7, 28)]])


from datetime import timedelta

from collectors.price_history_import import (
    MAX_GAP_DAYS,
    MIN_DISTINCT_DAYS,
    apply_gap_gate,
)


def _series(name, days, price=1.0):
    """Records for *name* on the given day offsets from 2025-06-01."""
    return [(name, date(2025, 6, 1) + timedelta(days=d), price) for d in days]


# The gap-specific tests set min_distinct_days=1 so they isolate the gap
# condition; the sparse floor gets its own tests below.
def _gaps_only(records):
    return apply_gap_gate(records, min_distinct_days=1)


def test_max_gap_matches_the_archive_window_constant():
    from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS
    assert MAX_GAP_DAYS == MAX_WINDOW_SPAN_DAYS == 7


def test_min_distinct_days_floor_is_180():
    assert MIN_DISTINCT_DAYS == 180


def test_a_seven_day_gap_is_kept():
    kept, report = _gaps_only(_series("Item A", [0, 7, 14]))
    assert {r[0] for r in kept} == {"Item A"}
    assert report.rejected_gap_items == 0


def test_an_eight_day_gap_is_rejected():
    kept, report = _gaps_only(_series("Item A", [0, 8, 16]))
    assert kept == []
    assert report.rejected_gap_items == 1
    assert report.rejected_rows == 3
    assert report.worst_gap_days == 8


def test_the_gate_is_per_item_not_global():
    records = _series("Dense", [0, 1, 2]) + _series("Sparse", [0, 30])
    kept, report = _gaps_only(records)
    assert {r[0] for r in kept} == {"Dense"}
    assert report.kept_items == 1
    assert report.rejected_gap_items == 1


def test_edges_are_not_penalised_only_interior_gaps_count():
    # Starts late and ends early inside the range; interior spacing is daily.
    kept, _ = _gaps_only(_series("Late Starter", [40, 41, 42, 43]))
    assert {r[0] for r in kept} == {"Late Starter"}


def test_a_dense_item_spanning_180_days_passes_the_default_floor():
    kept, report = apply_gap_gate(_series("Dense", range(180)))
    assert {r[0] for r in kept} == {"Dense"}
    assert report.rejected_sparse_items == 0


def test_179_distinct_days_is_one_short_and_is_rejected():
    kept, report = apply_gap_gate(_series("Nearly", range(179)))
    assert kept == []
    assert report.rejected_sparse_items == 1


def test_two_adjacent_observations_do_not_pass_trivially():
    """Max-gap alone is necessary and NOT sufficient.

    Two consecutive days have a max interior gap of 1, so without the distinct-
    day floor this item sails through and flips is_backfilled on two rows —
    measured at 904 such items (611 at >=$1) on the 90-day block.
    """
    kept, report = apply_gap_gate(_series("Two Days", [0, 1]))
    assert kept == []
    assert report.rejected_sparse_items == 1
    assert report.rejected_gap_items == 0


def test_a_single_observation_item_is_rejected_by_the_sparse_floor():
    kept, report = apply_gap_gate(_series("Lonely", [5]))
    assert kept == []
    assert report.rejected_sparse_items == 1


def test_the_gap_condition_takes_precedence_in_the_report():
    """An item failing BOTH conditions counts once, as a gap rejection."""
    kept, report = apply_gap_gate(_series("Both", [0, 30]))
    assert kept == []
    assert report.rejected_gap_items == 1
    assert report.rejected_sparse_items == 0


def test_records_are_deduplicated_on_item_and_day_keeping_the_first():
    records = [
        ("Item A", date(2025, 6, 1), 10.0),
        ("Item A", date(2025, 6, 1), 99.0),
        ("Item A", date(2025, 6, 2), 11.0),
    ]
    kept, report = _gaps_only(records)
    assert sorted(kept) == [
        ("Item A", date(2025, 6, 1), 10.0),
        ("Item A", date(2025, 6, 2), 11.0),
    ]
    assert report.kept_rows == 2
