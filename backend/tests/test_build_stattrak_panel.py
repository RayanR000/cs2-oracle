import sys
from pathlib import Path
from datetime import date
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_stattrak_panel import base_name, build_stattrak_panel  # noqa: E402


def test_base_name_strips_stattrak_prefix():
    assert base_name("StatTrak™ AK-47 | Redline (Field-Tested)") == \
        "AK-47 | Redline (Field-Tested)"
    assert base_name("AK-47 | Redline (Field-Tested)") is None  # not ST


def test_premium_is_ratio_on_matched_day():
    prices = pd.DataFrame({
        "item_id": ["StatTrak™ AK-47 | Redline (Field-Tested)",
                    "AK-47 | Redline (Field-Tested)"],
        "date": [date(2026, 5, 1), date(2026, 5, 1)],
        "price": [21.0, 10.0],
    })
    out = build_stattrak_panel(prices)
    assert list(out.columns) == ["item_id", "date", "st_premium"]
    assert out.iloc[0]["item_id"].startswith("StatTrak")
    assert abs(out.iloc[0]["st_premium"] - 2.1) < 1e-9


def test_st_without_base_is_dropped():
    prices = pd.DataFrame({
        "item_id": ["StatTrak™ Only | Item"],
        "date": [date(2026, 5, 1)],
        "price": [50.0],
    })
    assert len(build_stattrak_panel(prices)) == 0
