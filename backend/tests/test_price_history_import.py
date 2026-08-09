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
