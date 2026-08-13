import sys
from pathlib import Path
from datetime import date
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_bid_panel import build_bid_panel
from scripts.build_stattrak_panel import build_stattrak_panel


def test_every_sidecar_has_the_join_key():
    # Contract: every sidecar exposes exactly (item_id, date) + its feed cols.
    bid = build_bid_panel(pd.DataFrame({
        "item_slug": ["AK | X"], "day": [date(2026, 8, 1)], "mean_price": [8.0]}))
    assert {"item_id", "date"} <= set(bid.columns)
    st = build_stattrak_panel(pd.DataFrame({
        "item_id": ["StatTrak™ AK | X", "AK | X"],
        "date": [date(2026, 8, 1), date(2026, 8, 1)], "price": [16.0, 8.0]}))
    assert {"item_id", "date"} <= set(st.columns)
