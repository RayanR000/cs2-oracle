#!/usr/bin/env python3
"""Put one lost day back into archive files from an older copy of them.

The data repo keeps no history (each publish is an orphan commit + force-push), so a publish
built from a stale checkout silently deletes whatever landed after that checkout. That is how
day 2026-09-19 vanished: the 09-20 Aggregator pushed it at 00:57:54 UTC and an item_forecasts
publish force-pushed an older tree over it at 00:58:15. GitHub still serves the overwritten
commit by SHA, so the rows are recoverable from it.

For each file, the restore appends the source's rows for `--day` only, and only when the
current file holds no row for that day. Rows already in the current file are never touched,
the file's schema and compression are kept, and `ingested_at` / `collected_at` keep their
original first-arrival values, since the restored rows really did arrive then
(`.claude/rules/archive-reads.md`).

Usage (from backend/; dry run unless --apply):
    venv/bin/python -m scripts.restore_archive_day --day 2026-09-19 \\
        --source-dir /tmp/at-284f9d9 --archive-dir ../archive/price-archive \\
        prices-2026-09.parquet supply-2026-09.parquet volume-2026-09.parquet exchange-rates-2026.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# The column that says which day a row describes, per day-appended archive file family.
# supply-* also carries collected_at, a tz-aware wall clock that does not match the day.
DAY_COLUMNS = {
    "prices-": "day",
    "volume-": "day",
    "exchange-rates-": "day",
    "supply-": "snapshot_day",
}


def day_column(filename: str) -> str:
    name = Path(filename).name
    for prefix, col in DAY_COLUMNS.items():
        if name.startswith(prefix):
            return col
    raise ValueError(f"{name}: not a day-appended archive file ({sorted(DAY_COLUMNS)})")


def _on_day(table: pa.Table, col: str, day: dt.date) -> pa.Table:
    return table.filter(pc.equal(pc.cast(table[col], pa.date32()), pa.scalar(day, pa.date32())))


def _compression(path: Path) -> str:
    meta = pq.ParquetFile(path).metadata
    return meta.row_group(0).column(0).compression.lower() if meta.num_row_groups else "snappy"


def restore_file(current_path, source_path, day: dt.date, apply: bool = False) -> dict:
    """Append `source_path`'s rows for `day` to `current_path` if the day is absent there."""
    current_path, source_path = Path(current_path), Path(source_path)
    col = day_column(current_path.name)
    current = pq.read_table(current_path)
    source = pq.read_table(source_path)

    if source.schema.names != current.schema.names:
        raise ValueError(
            f"{current_path.name}: schema mismatch, source {source.schema.names} vs current {current.schema.names}"
        )
    report = {"file": current_path.name, "day": str(day), "rows_before": current.num_rows, "applied": False}

    if _on_day(current, col, day).num_rows:
        return {**report, "restored": 0, "skipped": "day already present"}

    rows = _on_day(source, col, day)
    if not rows.num_rows:
        raise ValueError(f"{current_path.name}: source has no rows for {day}")
    rows = rows.cast(current.schema)
    report["restored"] = rows.num_rows

    if apply:
        out = pa.concat_tables([current, rows])
        tmp = current_path.with_suffix(".restore.tmp")
        pq.write_table(out, tmp, compression=_compression(current_path))
        os.replace(tmp, current_path)  # atomic: an interrupted write never truncates the file
        report["rows_after"] = pq.read_metadata(current_path).num_rows
        report["applied"] = True
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--source-dir", required=True, type=Path, help="files as they were at the good commit")
    ap.add_argument("--archive-dir", required=True, type=Path, help="the checked-out price-archive/ to write")
    ap.add_argument("--apply", action="store_true", help="write; without it, report only")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)

    reports = [restore_file(args.archive_dir / f, args.source_dir / f, args.day, apply=args.apply) for f in args.files]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
