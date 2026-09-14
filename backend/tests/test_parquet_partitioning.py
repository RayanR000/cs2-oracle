"""Tests for db.parquet.append_monthly — the canonical month-partition writer."""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db.parquet import append_monthly

KEYS = ["item_slug", "day", "source"]


def _rows(day, items):
    return pd.DataFrame([{"item_slug": s, "day": pd.Timestamp(day), "source": "src", "price": p} for s, p in items])


def _count(pq):
    return duckdb.connect().sql(f"SELECT count(*) FROM read_parquet('{pq}')").fetchone()[0]


def test_routes_rows_to_per_month_files(tmp_path):
    df = pd.concat(
        [
            _rows("2026-07-31", [("a", 1.0)]),
            _rows("2026-08-01", [("b", 2.0)]),
            _rows("2026-08-15", [("c", 3.0)]),
        ],
        ignore_index=True,
    )
    append_monthly(tmp_path, "prices", df, KEYS)

    assert (tmp_path / "prices-2026-07.parquet").exists()
    assert (tmp_path / "prices-2026-08.parquet").exists()
    assert not (tmp_path / "prices-2026.parquet").exists()
    assert _count(tmp_path / "prices-2026-07.parquet") == 1
    assert _count(tmp_path / "prices-2026-08.parquet") == 2


def test_appends_and_dedups_within_month(tmp_path):
    append_monthly(tmp_path, "prices", _rows("2026-08-01", [("a", 1.0)]), KEYS)
    # new key same month -> append; repeat key -> dedup (keep last)
    append_monthly(tmp_path, "prices", _rows("2026-08-02", [("b", 2.0)]), KEYS)
    append_monthly(tmp_path, "prices", _rows("2026-08-01", [("a", 9.9)]), KEYS)

    pq = tmp_path / "prices-2026-08.parquet"
    assert _count(pq) == 2
    price_a = duckdb.connect().sql(f"SELECT price FROM read_parquet('{pq}') WHERE item_slug='a'").fetchone()[0]
    assert price_a == 9.9  # kept the last write


def test_empty_df_is_noop(tmp_path):
    append_monthly(tmp_path, "prices", pd.DataFrame(columns=["item_slug", "day", "source", "price"]), KEYS)
    assert list(tmp_path.glob("*.parquet")) == []


def test_glob_matches_monthly_files(tmp_path):
    """Readers use prices-*.parquet — confirm monthly files are picked up."""
    append_monthly(tmp_path, "prices", _rows("2026-08-01", [("a", 1.0)]), KEYS)
    append_monthly(tmp_path, "prices", _rows("2026-09-01", [("b", 2.0)]), KEYS)
    total = duckdb.connect().sql(f"SELECT count(*) FROM read_parquet('{tmp_path}/prices-*.parquet')").fetchone()[0]
    assert total == 2
