"""DB-free snapshot export: CSV schema + raw-source coverage.

These exercise the pure writers with a hand-built `_raw_sources`, so no network
and no DB. They pin the schema `append_to_parquet.py --snapshot-csv` expects.
"""

import csv

from collectors.csgotrader_aggregator import CSGOTraderAggregator
from scripts.export_snapshot_dbfree import write_exchange_rates_csv, write_snapshot_csv


def _read(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_snapshot_csv_schema_and_rows(tmp_path):
    agg = CSGOTraderAggregator()
    agg._raw_sources = {
        "steam": {
            "AK-47 | Redline (FT)": {"last_24h": 10.0, "last_7d": 11.0, "last_30d": 12.0, "last_90d": 13.0},
        },
        "buff163": {
            "AK-47 | Redline (FT)": {"starting_at": {"price": 9.5}, "highest_order": {"price": 9.0}},
        },
        "csfloat": {
            "AK-47 | Redline (FT)": {"price": 10.2},
        },
    }
    out = str(tmp_path / "snap.csv")
    n = write_snapshot_csv(agg, "2026-08-23", out)

    rows = _read(out)
    assert rows[0] == ["item_slug", "day", "source", "price", "volume"]
    body = rows[1:]
    assert len(body) == n

    labels = {r[2] for r in body}
    # steam expands to sync + spot + 7d/30d/90d; buff163 to listing + buy; csfloat one.
    assert {
        "aggregator_sync",
        "aggregator_steam_spot",
        "aggregator_steam_7d",
        "aggregator_steam_30d",
        "aggregator_steam_90d",
        "aggregator_buff163",
        "aggregator_buff163_buy",
        "aggregator_csfloat",
    } <= labels

    # Every row: same slug + day, empty volume, positive price.
    for r in body:
        assert r[0] == "AK-47 | Redline (FT)"
        assert r[1] == "2026-08-23"
        assert r[4] == ""  # VOLUME_NOT_OBSERVED -> empty field
        assert float(r[3]) > 0


def test_snapshot_drops_zero_and_missing_prices(tmp_path):
    agg = CSGOTraderAggregator()
    agg._raw_sources = {
        "steam": {"Zero Item": {"last_24h": 0, "last_7d": None}},
        "csfloat": {"None Item": {"price": None}},
    }
    out = str(tmp_path / "snap.csv")
    n = write_snapshot_csv(agg, "2026-08-23", out)
    assert n == 0
    assert _read(out) == [["item_slug", "day", "source", "price", "volume"]]


def test_steam_spot_absent_when_no_24h(tmp_path):
    # aggregator_sync falls back to 7d, but aggregator_steam_spot must stay empty.
    agg = CSGOTraderAggregator()
    agg._raw_sources = {"steam": {"Illiquid": {"last_7d": 5.0}}}
    out = str(tmp_path / "snap.csv")
    write_snapshot_csv(agg, "2026-08-23", out)
    body = _read(out)[1:]
    labels = {r[2] for r in body}
    assert "aggregator_sync" in labels  # fell back to 7d
    assert "aggregator_steam_spot" not in labels  # no 24h -> no spot


def test_exchange_rates_csv(tmp_path):
    out = str(tmp_path / "rates.csv")
    n = write_exchange_rates_csv({"EUR": 0.92, "CNY": 7.1}, "2026-08-23", out)
    assert n == 2
    rows = _read(out)
    assert rows[0] == ["currency", "rate", "day"]
    assert ["EUR", "0.92", "2026-08-23"] in rows
