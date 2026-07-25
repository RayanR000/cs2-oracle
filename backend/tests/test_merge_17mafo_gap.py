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


import pytest


def _prices_for_dates(dates, n_items):
    frames = []
    for d in dates:
        obj = {f"Item {i}": {"steam": {"last_24h": 1.0 + i}} for i in range(n_items)}
        frames.append(m.transform_day(obj, d))
    return pd.concat(frames, ignore_index=True)


def test_gap_dates_inclusive():
    dates = m.gap_dates("2026-04-16", "2026-07-08")
    assert dates[0] == "2026-04-16"
    assert dates[-1] == "2026-07-08"
    assert len(dates) == 84


def test_validate_coverage_passes():
    dates = ["2026-04-16", "2026-04-17"]
    prices = _prices_for_dates(dates, n_items=25)
    m.validate_coverage(prices, dates, min_items=25)  # no raise


def test_validate_coverage_missing_day_raises():
    dates = ["2026-04-16", "2026-04-17"]
    prices = _prices_for_dates(["2026-04-16"], n_items=25)  # 04-17 missing
    with pytest.raises(AssertionError, match="2026-04-17"):
        m.validate_coverage(prices, dates, min_items=25)


def test_validate_coverage_low_count_raises():
    dates = ["2026-04-16"]
    prices = _prices_for_dates(dates, n_items=5)
    with pytest.raises(AssertionError, match="item count"):
        m.validate_coverage(prices, dates, min_items=25)


import json


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode()


class _FakeSession:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload or {}
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        return _FakeResp(self.status_code, self.payload)


def test_fetch_day_downloads_when_absent(tmp_path):
    sess = _FakeSession(200, {"Item A": {"steam": {"last_24h": 5.0}}})
    path = m.fetch_day("2026-04-16", tmp_path, session=sess)
    assert path.exists()
    assert sess.calls == 1
    assert m.load_day(path)["Item A"]["steam"]["last_24h"] == 5.0


def test_fetch_day_uses_cache(tmp_path):
    (tmp_path / "2026-04-16.json").write_text('{"Item A": {"steam": {"last_24h": 9.0}}}')
    sess = _FakeSession(200, {})
    path = m.fetch_day("2026-04-16", tmp_path, session=sess)
    assert sess.calls == 0                      # cache hit, no network
    assert m.load_day(path)["Item A"]["steam"]["last_24h"] == 9.0


def test_fetch_day_raises_on_error(tmp_path):
    sess = _FakeSession(404, {})
    with pytest.raises(RuntimeError, match="404"):
        m.fetch_day("2026-04-16", tmp_path, session=sess)
