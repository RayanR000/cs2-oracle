"""Tests for db/archive.py — the single price-archive read path.

The fixture archive reproduces the real store's schema drift on purpose: a
legacy yearly file with no `source` and a `TIMESTAMP` day, a monthly file with
`source` and `TIMESTAMP_NS`, a range-carrying month, and a month whose columns
are in a different order. A reader that only works on a uniform archive is not
the reader this repo needs.
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.archive import (  # noqa: E402
    CANONICAL_PRICE_COLUMNS,
    present_columns,
    price_files,
    prices_relation,
    resolve_archive_dir,
)


def _legacy(n=3):
    """Pre-2026 shape: no source, day is TIMESTAMP."""
    return pd.DataFrame({
        "item_slug": [f"legacy-{i}" for i in range(n)],
        "day": pd.to_datetime(["2025-06-01"] * n),
        "mean_price": [10.0 + i for i in range(n)],
        "volume": [5] * n,
    })


def _modern(n=3):
    """2026 shape: source present, canonical order."""
    return pd.DataFrame({
        "item_slug": [f"modern-{i}" for i in range(n)],
        "day": pd.to_datetime(["2026-05-01"] * n),
        "source": ["aggregator_buff163"] * n,
        "mean_price": [20.0 + i for i in range(n)],
        "volume": [7] * n,
    })


def _with_range(n=2):
    """2026-03/04 shape: real intraday range columns."""
    return pd.DataFrame({
        "item_slug": [f"range-{i}" for i in range(n)],
        "day": pd.to_datetime(["2026-03-01"] * n),
        "mean_price": [30.0 + i for i in range(n)],
        "volume": [1] * n,
        "min_price": [1.0] * n,
        "max_price": [99.0] * n,
        "source": ["aggregator_sync"] * n,
    })


def _reordered(n=2):
    """prices-2026-08's shape: source in a different position."""
    frame = _modern(n)
    frame["item_slug"] = [f"reordered-{i}" for i in range(n)]
    frame["day"] = pd.to_datetime(["2026-08-01"] * n)
    return frame[["item_slug", "day", "mean_price", "volume", "source"]]


@pytest.fixture
def archive(tmp_path):
    _legacy().to_parquet(tmp_path / "prices-2025.parquet", index=False)
    _with_range().to_parquet(tmp_path / "prices-2026-03.parquet", index=False)
    _modern().to_parquet(tmp_path / "prices-2026-05.parquet", index=False)
    _reordered().to_parquet(tmp_path / "prices-2026-08.parquet", index=False)
    return tmp_path


@pytest.fixture
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


# ── the bug this module exists to prevent ────────────────────────────────────

def test_plain_glob_silently_drops_source(archive, con):
    """The failure mode being fixed. If DuckDB ever stops doing this, the
    workaround can go — but until then this is why prices_relation exists."""
    cols = {r[0] for r in con.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{archive}/prices-*.parquet')"
    ).fetchall()}
    assert "source" not in cols


def test_prices_relation_keeps_source(archive, con):
    rel = prices_relation(con, archive)
    cols = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {rel}").fetchall()}
    assert cols == set(CANONICAL_PRICE_COLUMNS)


# ── correctness across the drift ─────────────────────────────────────────────

def test_reads_every_row_from_every_file(archive, con):
    rel = prices_relation(con, archive)
    assert con.sql(f"SELECT count(*) FROM {rel}").fetchone()[0] == 10


def test_source_is_null_for_legacy_and_populated_for_modern(archive, con):
    rel = prices_relation(con, archive)
    rows = dict(con.sql(
        f"SELECT item_slug, source FROM {rel} WHERE item_slug IN "
        f"('legacy-0', 'modern-0', 'reordered-0')"
    ).fetchall())
    assert rows["legacy-0"] is None
    assert rows["modern-0"] == "aggregator_buff163"
    assert rows["reordered-0"] == "aggregator_buff163"


def test_reordered_columns_are_matched_by_name_not_position(archive, con):
    """prices-2026-08 puts source where mean_price sits in other months. A
    positional union would swap them and report a price of 'aggregator_*'."""
    rel = prices_relation(con, archive)
    price, volume = con.sql(
        f"SELECT mean_price, volume FROM {rel} WHERE item_slug = 'reordered-0'"
    ).fetchone()
    assert price == 20.0
    assert volume == 7


def test_day_is_one_type_across_mixed_timestamp_precisions(archive, con):
    rel = prices_relation(con, archive)
    day_type = [r[1] for r in con.sql(f"DESCRIBE SELECT * FROM {rel}").fetchall()
                if r[0] == "day"][0]
    assert day_type == "DATE"


def test_day_filter_matches_the_expected_rows(archive, con):
    rel = prices_relation(con, archive)
    n = con.sql(
        f"SELECT count(*) FROM {rel} WHERE day >= DATE '2026-01-01'").fetchone()[0]
    assert n == 7


# ── optional columns ─────────────────────────────────────────────────────────

def test_range_columns_are_readable_where_present(archive, con):
    rel = prices_relation(
        con, archive, columns=["item_slug", "min_price", "max_price"])
    assert con.sql(
        f"SELECT min_price, max_price FROM {rel} WHERE item_slug = 'range-0'"
    ).fetchone() == (1.0, 99.0)


def test_range_columns_are_null_where_absent(archive, con):
    rel = prices_relation(
        con, archive, columns=["item_slug", "min_price"])
    assert con.sql(
        f"SELECT min_price FROM {rel} WHERE item_slug = 'legacy-0'"
    ).fetchone()[0] is None


def test_column_absent_from_every_file_is_a_typed_null_not_an_error(tmp_path, con):
    """A pre-2026-only archive still answers a query that selects source."""
    _legacy().to_parquet(tmp_path / "prices-2025.parquet", index=False)
    rel = prices_relation(con, tmp_path)
    assert con.sql(f"SELECT count(*) FROM {rel} WHERE source IS NULL"
                   ).fetchone()[0] == 3


def test_unknown_column_is_rejected(archive, con):
    with pytest.raises(ValueError, match="unknown price column"):
        prices_relation(con, archive, columns=["item_slug", "nope"])


# ── where clause ─────────────────────────────────────────────────────────────

def test_where_is_applied(archive, con):
    rel = prices_relation(con, archive, where="source IS NOT NULL")
    assert con.sql(f"SELECT count(*) FROM {rel}").fetchone()[0] == 7


# ── path handling ────────────────────────────────────────────────────────────

def test_missing_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        price_files(tmp_path / "nope")


def test_empty_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="contains no"):
        price_files(tmp_path)


def test_slug_with_a_quote_in_the_path_is_escaped(tmp_path, con):
    weird = tmp_path / "it's an archive"
    weird.mkdir()
    _modern().to_parquet(weird / "prices-2026-05.parquet", index=False)
    rel = prices_relation(con, weird)
    assert con.sql(f"SELECT count(*) FROM {rel}").fetchone()[0] == 3


def test_resolve_defaults_to_repo_root(tmp_path):
    assert resolve_archive_dir().name == "price-archive"
    assert resolve_archive_dir(tmp_path) == tmp_path


def test_present_columns_unions_across_files(archive, con):
    assert present_columns(con, archive) == {
        "item_slug", "day", "source", "mean_price", "volume",
        "min_price", "max_price",
    }
