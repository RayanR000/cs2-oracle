import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ingest_volume_panel import build_volume_panel


def _src():
    return pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Redline (Field-Tested)", "AK-47 | Redline (Field-Tested)", "Sticker | X"],
            "date": [date(2026, 5, 1), date(2026, 5, 1), date(2026, 5, 1)],
            "price_median": [10.0, 12.0, 1.0],
            "volume": [5, 7, 0],
        }
    )


def test_maps_name_to_item_id_and_keeps_volume():
    out = build_volume_panel(_src())
    assert list(out.columns) == ["item_id", "date", "steam_volume", "steam_sale_median"]
    row = out[out["item_id"] == "AK-47 | Redline (Field-Tested)"]
    # duplicate (item_id, date) collapses to one row
    assert len(row) == 1


def test_volume_is_integer_not_zero_filled():
    out = build_volume_panel(_src())
    assert out["steam_volume"].dtype.kind == "i"
    # the zero-volume sticker row is kept as 0, never dropped or nan-filled
    assert (out["steam_volume"] == 0).sum() == 1


def test_rejects_missing_required_column():
    bad = _src().drop(columns=["volume"])
    with pytest.raises((KeyError, ValueError)):
        build_volume_panel(bad)
