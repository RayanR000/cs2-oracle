import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.archive.ingest_supply_history import parse_supply_history

# 2023-06-01 and 2023-06-02 in unix seconds
T1, T2 = 1685577600, 1685664000


def test_extracts_third_array_as_listing_count():
    raw = {"AK-47 | Redline (Field-Tested)": [[T1, T2], [1000, 1100], [42, 40]]}
    out = parse_supply_history(raw)
    assert list(out.columns) == ["item_id", "date", "buff_listing_count"]
    assert out.iloc[0]["buff_listing_count"] == 42
    assert out.iloc[0]["date"] == date(2023, 6, 1)


def test_drops_items_without_listing_counts_not_zero_fills():
    # An item present before 2023-01-25 has only two arrays (price, no count).
    raw = {"Old | Item": [[T1], [500]]}
    out = parse_supply_history(raw)
    assert len(out) == 0  # dropped, never zero-filled


def test_length_mismatch_raises():
    raw = {"Bad | Item": [[T1, T2], [1000, 1100], [42]]}  # counts shorter than timestamps
    with pytest.raises(ValueError):
        parse_supply_history(raw)
