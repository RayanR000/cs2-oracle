"""restore_archive_day: put one lost day back into an archive file from an older copy of it.

The 2026-09-20 Aggregator wrote day 2026-09-19 and a concurrent publish force-pushed over it
21 seconds later (docs/changelog/2026-09-29-archive-day-0919-restored.md). The restore must
append exactly that day's rows, keep the file's schema, compression and first-arrival
timestamps, and do nothing at all when the day is already present.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.restore_archive_day import day_column, restore_file

D = dt.date(2026, 9, 19)


def _prices(days, price=1.0):
    n = len(days)
    return pa.table(
        {
            "item_slug": pa.array([f"item-{i}" for i in range(n)]),
            "day": pa.array(days, type=pa.date32()),
            "source": pa.array(["aggregator_sync"] * n),
            "mean_price": pa.array([price] * n, type=pa.float64()),
            "volume": pa.array([1] * n, type=pa.int64()),
            "ingested_at": pa.array([dt.datetime(2026, 9, 20, 0, 57)] * n, type=pa.timestamp("us")),
        }
    )


def _write(table, path, compression="snappy"):
    pq.write_table(table, path, compression=compression)
    return path


def test_day_column_per_archive_file():
    assert day_column("prices-2026-09.parquet") == "day"
    assert day_column("volume-2026-09.parquet") == "day"
    assert day_column("exchange-rates-2026.parquet") == "day"
    assert day_column("supply-2026-09.parquet") == "snapshot_day"  # collected_at is tz-aware
    with pytest.raises(ValueError):
        day_column("item_forecasts.parquet")  # ops tables are not day-appended


def test_restores_only_the_missing_day(tmp_path):
    current = _write(_prices([dt.date(2026, 9, 18), dt.date(2026, 9, 20)]), tmp_path / "prices-2026-09.parquet")
    source = _write(_prices([dt.date(2026, 9, 18), D, D], price=2.0), tmp_path / "src.parquet")

    report = restore_file(current, source, D, apply=True)

    out = pq.read_table(current)
    assert report["restored"] == 2
    assert out.num_rows == 4
    days = out.column("day").to_pylist()
    assert days.count(D) == 2
    # The 09-18 row already in the file is the current one; the source's copy is never merged in.
    assert out.filter(pa.compute.equal(out["day"], pa.scalar(dt.date(2026, 9, 18)))).column(
        "mean_price"
    ).to_pylist() == [1.0]


def test_is_a_no_op_when_the_day_is_already_present(tmp_path):
    current = _write(_prices([D]), tmp_path / "prices-2026-09.parquet")
    before = current.read_bytes()
    source = _write(_prices([D, D], price=9.0), tmp_path / "src.parquet")

    report = restore_file(current, source, D, apply=True)

    assert report["restored"] == 0
    assert report["skipped"] == "day already present"
    assert current.read_bytes() == before


def test_dry_run_reports_without_writing(tmp_path):
    current = _write(_prices([dt.date(2026, 9, 18)]), tmp_path / "prices-2026-09.parquet")
    before = current.read_bytes()
    source = _write(_prices([D]), tmp_path / "src.parquet")

    report = restore_file(current, source, D, apply=False)

    assert report["restored"] == 1
    assert report["applied"] is False
    assert current.read_bytes() == before


def test_preserves_schema_compression_and_first_arrival(tmp_path):
    current = _write(_prices([dt.date(2026, 9, 18)]), tmp_path / "volume-2026-09.parquet", compression="zstd")
    source = _write(_prices([D]), tmp_path / "src.parquet", compression="snappy")
    schema_before = pq.read_schema(current)

    restore_file(current, source, D, apply=True)

    assert pq.read_schema(current).equals(schema_before)
    meta = pq.ParquetFile(current).metadata
    assert meta.row_group(meta.num_row_groups - 1).column(0).compression == "ZSTD"
    restored = pq.read_table(current).filter(pa.compute.equal(pq.read_table(current)["day"], pa.scalar(D)))
    assert restored.column("ingested_at").to_pylist() == [dt.datetime(2026, 9, 20, 0, 57)]


def test_refuses_a_source_with_a_different_schema(tmp_path):
    current = _write(_prices([dt.date(2026, 9, 18)]), tmp_path / "prices-2026-09.parquet")
    source = _write(_prices([D]).drop(["volume"]), tmp_path / "src.parquet")
    with pytest.raises(ValueError, match="schema"):
        restore_file(current, source, D, apply=True)


def test_refuses_a_source_without_the_day(tmp_path):
    current = _write(_prices([dt.date(2026, 9, 18)]), tmp_path / "prices-2026-09.parquet")
    source = _write(_prices([dt.date(2026, 9, 18)]), tmp_path / "src.parquet")
    with pytest.raises(ValueError, match="no rows"):
        restore_file(current, source, D, apply=True)
