import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_17mafo_gap as m


SAMPLE_DAY = {
    "AK-47 | Redline (Field-Tested)": {
        "steam": {"last_24h": 47.63418, "last_7d": 47.37, "last_30d": 46.6,
                  "last_90d": 48.5, "last_ever": 45.99}
    },
    "★ Karambit | Doppler (Factory New)": {  # null recent price -> skipped
        "steam": {"last_24h": None, "last_7d": None, "last_30d": None,
                  "last_90d": None, "last_ever": 1000.0}
    },
    "Broken Item": {"steam": {}},            # no last_24h -> skipped
    "Weird Item": "not-a-dict",              # non-dict value -> skipped
}


def test_transform_day_columns_and_values():
    df = m.transform_day(SAMPLE_DAY, "2026-04-16")

    assert list(df.columns) == m.PRICE_COLS
    assert len(df) == 1  # only the valid Redline row survives

    row = df.iloc[0]
    assert row["item_slug"] == "AK-47 | Redline (Field-Tested)"
    assert row["day"] == pd.Timestamp("2026-04-16")
    assert row["source"] == "aggregator_steam_17mafo"
    assert row["mean_price"] == 47.63418
    assert row["median_price"] == 47.63418
    assert row["min_price"] == 47.63418
    assert row["max_price"] == 47.63418
    assert row["volume"] == 0


def test_prices_to_snapshots():
    prices = m.transform_day(SAMPLE_DAY, "2026-04-16")
    snaps = m.prices_to_snapshots(prices)

    assert list(snaps.columns) == m.SNAP_COLS
    assert len(snaps) == 1
    assert snaps.iloc[0]["price"] == 47.63418
    assert snaps.iloc[0]["volume"] == 0
    assert snaps.iloc[0]["source"] == "aggregator_steam_17mafo"
