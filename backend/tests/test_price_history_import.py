"""Tests for the historical price-source import."""
from datetime import date, datetime

import pytest

from collectors.price_history_sources import cs2_prices_tracker as tracker


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
