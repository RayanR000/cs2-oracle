import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_bid_panel import build_bid_panel


def _rows():
    return pd.DataFrame(
        {
            "item_slug": ["AK | X", "AK | X", "AK | X"],
            "day": [date(2026, 8, 1), date(2026, 8, 1), date(2026, 8, 2)],
            "mean_price": [7.0, 9.0, 8.0],
        }
    )


def test_renames_keys_and_averages_within_day():
    out = build_bid_panel(_rows())
    assert list(out.columns) == ["item_id", "date", "buff_bid"]
    aug1 = out[(out["item_id"] == "AK | X") & (out["date"] == date(2026, 8, 1))]
    assert len(aug1) == 1
    assert aug1.iloc[0]["buff_bid"] == 8.0  # mean of 7 and 9


def test_drops_nonpositive_bids():
    rows = _rows()
    rows.loc[len(rows)] = ["AK | Y", date(2026, 8, 1), 0.0]
    out = build_bid_panel(rows)
    assert (out["item_id"] == "AK | Y").sum() == 0
