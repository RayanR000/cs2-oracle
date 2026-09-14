"""Tests for the Skinport sales/history volume collector.

The parser is pure, so the schema contract is tested without an egress Skinport
will answer — this machine's egress is a Cloudflare IP and gets a 403 WAF
challenge regardless of the endpoint.

The theme running through these: a schema change must *fail the run*, never
drain the feed quietly. A silently-empty collector behind a green badge is the
failure mode this repo keeps hitting.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from collectors.sales_volume import (  # noqa: E402
    SalesVolumeError,
    parse_sales_history,
    volume_parquet_path,
    write_volume_rows,
)

DAY = date(2026, 8, 8)
AT = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


def _entry(name, v24=1, v7=5, v30=20, v90=60, median=12.5):
    return {
        "market_hash_name": name,
        "currency": "USD",
        "last_24_hours": {"min": 10, "max": 14, "avg": 12, "median": 12, "volume": v24},
        "last_7_days": {"min": 9, "max": 15, "avg": 12, "median": 12, "volume": v7},
        "last_30_days": {"min": 8, "max": 16, "avg": 12, "median": median, "volume": v30},
        "last_90_days": {"min": 7, "max": 17, "avg": 12, "median": 12, "volume": v90},
    }


class TestParsing:
    def test_reduces_one_row_per_item(self):
        rows = parse_sales_history(
            [_entry("AK-47 | Redline (Field-Tested)"), _entry("AWP | Asiimov (FT)")],
            DAY,
            AT,
        )
        assert len(rows) == 2
        assert list(rows["source"].unique()) == ["skinport_sales"]
        assert rows["day"].unique().tolist() == [DAY]

    def test_carries_every_window(self):
        rows = parse_sales_history([_entry("X", v24=3, v7=9, v30=40, v90=120)], DAY, AT)
        row = rows.iloc[0]
        assert row["sales_24h"] == 3
        assert row["sales_7d"] == 9
        assert row["sales_30d"] == 40
        assert row["sales_90d"] == 120
        assert row["median_30d"] == 12.5

    def test_a_never_sold_item_is_zero_not_null(self):
        """Skinport reports a null window for an item it has never sold. That
        is a real 'no sales', unlike the aggregator's *unobserved* volume."""
        entry = _entry("Unsold")
        for key in ("last_24_hours", "last_7_days", "last_30_days", "last_90_days"):
            entry[key] = None
        rows = parse_sales_history([entry], DAY, AT)
        assert rows.iloc[0]["sales_30d"] == 0
        assert pd.isna(rows.iloc[0]["median_30d"])

    def test_counts_are_nullable_integers(self):
        rows = parse_sales_history([_entry("X")], DAY, AT)
        assert str(rows["sales_30d"].dtype) == "Int64"

    def test_entries_without_a_name_are_skipped(self):
        rows = parse_sales_history(
            [_entry("Real"), {"currency": "USD", "last_30_days": {"volume": 3}}],
            DAY,
            AT,
        )
        assert rows["item_slug"].tolist() == ["Real"]


class TestSchemaChangesFailLoudly:
    def test_a_non_list_payload_raises(self):
        with pytest.raises(SalesVolumeError, match="expected a JSON array"):
            parse_sales_history({"items": []}, DAY, AT)

    def test_zero_parsed_rows_raises(self):
        with pytest.raises(SalesVolumeError, match="schema has almost certainly changed"):
            parse_sales_history([{"name": "renamed", "sales": {}}], DAY, AT)

    def test_a_mostly_unparsed_payload_raises(self):
        """Half the catalogue vanishing is a renamed key, not a thin market."""
        payload = [_entry(f"item-{i}") for i in range(3)] + [{"nope": i} for i in range(20)]
        with pytest.raises(SalesVolumeError, match="Treating this as a schema change"):
            parse_sales_history(payload, DAY, AT)

    def test_a_renamed_window_key_does_not_silently_zero(self):
        entry = _entry("X")
        entry["last_month"] = entry.pop("last_30_days")
        rows = parse_sales_history([entry], DAY, AT)
        # The remaining windows still parse, but the renamed one is NULL --
        # never 0, which would read as "no sales happened".
        assert pd.isna(rows.iloc[0]["sales_30d"])
        assert rows.iloc[0]["sales_7d"] == 5


class TestWriting:
    def test_writes_a_monthly_partition(self, tmp_path):
        rows = parse_sales_history([_entry("X")], DAY, AT)
        n = write_volume_rows(rows, tmp_path, DAY)
        assert n == 1
        assert volume_parquet_path(tmp_path, DAY).name == "volume-2026-08.parquet"
        assert volume_parquet_path(tmp_path, DAY).exists()

    def test_a_second_day_accumulates(self, tmp_path):
        """The bug this guards is the file being replaced rather than appended,
        which would leave the panel one day deep forever."""
        write_volume_rows(parse_sales_history([_entry("X")], DAY, AT), tmp_path, DAY)
        day2 = date(2026, 8, 9)
        write_volume_rows(parse_sales_history([_entry("X")], day2, AT), tmp_path, day2)

        stored = pd.read_parquet(volume_parquet_path(tmp_path, DAY))
        assert sorted(stored["day"].unique()) == [DAY, day2]

    def test_rerunning_a_day_replaces_rather_than_duplicates(self, tmp_path):
        write_volume_rows(parse_sales_history([_entry("X", v30=20)], DAY, AT), tmp_path, DAY)
        write_volume_rows(parse_sales_history([_entry("X", v30=25)], DAY, AT), tmp_path, DAY)

        stored = pd.read_parquet(volume_parquet_path(tmp_path, DAY))
        assert len(stored) == 1
        assert stored.iloc[0]["sales_30d"] == 25
