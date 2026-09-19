"""Tests for scripts/normalize_price_schema.py."""

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.archive import CANONICAL_PRICE_COLUMNS
from scripts.normalize_price_schema import (
    main,
    needs_rewrite,
    normalize_archive,
    target_columns,
    verify,
)

# Taken from the module rather than spelled out: this list is the thing under
# test, and a copy of it here only ever tests that the copy was updated.
CANONICAL = list(CANONICAL_PRICE_COLUMNS)


def _legacy(n=3, day="2025-06-01"):
    return pd.DataFrame(
        {
            "item_slug": [f"legacy-{i}" for i in range(n)],
            "day": pd.to_datetime([day] * n),
            "mean_price": [10.0 + i for i in range(n)],
            "volume": [5] * n,
        }
    )


def _with_range(n=2):
    return pd.DataFrame(
        {
            "item_slug": [f"range-{i}" for i in range(n)],
            "day": pd.to_datetime(["2026-03-01"] * n),
            "mean_price": [30.0 + i for i in range(n)],
            "volume": [1] * n,
            "min_price": [1.0] * n,
            "max_price": [99.0] * n,
            "source": ["aggregator_sync"] * n,
        }
    )


def _reordered(n=2):
    return pd.DataFrame(
        {
            "item_slug": [f"reordered-{i}" for i in range(n)],
            "day": pd.to_datetime(["2026-08-01"] * n),
            "mean_price": [20.0 + i for i in range(n)],
            "volume": [7] * n,
            "source": ["aggregator_buff163"] * n,
        }
    )[["item_slug", "day", "mean_price", "volume", "source"]]


def _schema(path):
    con = duckdb.connect()
    try:
        return [(r[0], r[1]) for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
    finally:
        con.close()


def _cols(path):
    return [c for c, _ in _schema(path)]


@pytest.fixture
def archive(tmp_path):
    _legacy().to_parquet(tmp_path / "prices-2025.parquet", index=False)
    _with_range().to_parquet(tmp_path / "prices-2026-03.parquet", index=False)
    _reordered().to_parquet(tmp_path / "prices-2026-08.parquet", index=False)
    return tmp_path


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_target_columns_puts_range_after_canonical():
    assert target_columns(
        ["item_slug", "day", "mean_price", "volume", "min_price", "max_price", "source"]
    ) == [*CANONICAL, "min_price", "max_price"]


def test_target_columns_preserves_an_unknown_column():
    assert target_columns([*CANONICAL, "mystery"]) == [*CANONICAL, "mystery"]


def test_needs_rewrite_is_false_for_canonical():
    assert not needs_rewrite([(c, "DATE" if c == "day" else "VARCHAR") for c in CANONICAL])


def test_needs_rewrite_is_true_for_a_timestamp_day():
    assert needs_rewrite([(c, "TIMESTAMP" if c == "day" else "VARCHAR") for c in CANONICAL])


# ── rewriting ────────────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(archive):
    before = {p.name: p.read_bytes() for p in archive.glob("*.parquet")}
    assert normalize_archive(archive, apply=False) == 3
    assert {p.name: p.read_bytes() for p in archive.glob("*.parquet")} == before


def test_legacy_file_gains_a_source_column(archive):
    normalize_archive(archive, apply=True)
    assert _cols(archive / "prices-2025.parquet") == CANONICAL


def test_added_source_is_null_not_a_label(archive):
    """`init_local_db.py` documents `source IS NULL` and `day < 2026` as
    selecting the same backfilled cohort. Stamping a label would break that."""
    normalize_archive(archive, apply=True)
    con = duckdb.connect()
    try:
        assert (
            con.sql(
                f"SELECT count(*) FROM read_parquet('{archive}/prices-2025.parquet') WHERE source IS NULL"
            ).fetchone()[0]
            == 3
        )
    finally:
        con.close()


def test_added_ingested_at_is_a_typed_null(archive):
    """The materialised column has to come back as the type
    `prices_relation` casts it to, or a migrated file disagrees with its own
    reader. It used to be NULLed to VARCHAR unconditionally, which was correct
    only while `source` was the one column ever missing.

    NULL means "arrival unknown". It must never be read as "arrived on `day`":
    every row in this file predates the column, and 13 years of it were written
    by backfill writers whose arrival was nothing like their `day`.
    """
    normalize_archive(archive, apply=True)
    path = archive / "prices-2025.parquet"

    assert dict(_schema(path))["ingested_at"] == "TIMESTAMP"
    con = duckdb.connect()
    try:
        assert con.sql(f"SELECT count(*) FROM read_parquet('{path}') WHERE ingested_at IS NULL").fetchone()[0] == 3
    finally:
        con.close()


def test_day_becomes_date(archive):
    normalize_archive(archive, apply=True)
    for name in ("prices-2025.parquet", "prices-2026-03.parquet", "prices-2026-08.parquet"):
        assert dict(_schema(archive / name))["day"] == "DATE"


def test_reordered_file_is_put_back_in_canonical_order(archive):
    normalize_archive(archive, apply=True)
    assert _cols(archive / "prices-2026-08.parquet") == CANONICAL


def test_range_columns_are_preserved_after_the_canonical_ones(archive):
    normalize_archive(archive, apply=True)
    assert _cols(archive / "prices-2026-03.parquet") == [*CANONICAL, "min_price", "max_price"]


def test_values_survive_the_rewrite(archive):
    normalize_archive(archive, apply=True)
    con = duckdb.connect()
    try:
        assert con.sql(
            f"SELECT mean_price, volume, source FROM read_parquet("
            f"'{archive}/prices-2026-08.parquet') WHERE item_slug='reordered-0'"
        ).fetchone() == (20.0, 7, "aggregator_buff163")
        assert con.sql(
            f"SELECT min_price, max_price FROM read_parquet("
            f"'{archive}/prices-2026-03.parquet') WHERE item_slug='range-0'"
        ).fetchone() == (1.0, 99.0)
        assert con.sql(f"SELECT count(*) FROM read_parquet('{archive}/prices-*.parquet')").fetchone()[0] == 7
    finally:
        con.close()


def test_no_temp_files_are_left_behind(archive):
    normalize_archive(archive, apply=True)
    assert list(archive.glob("*.tmp")) == []


# ── the point of the exercise ────────────────────────────────────────────────


def test_plain_glob_read_sees_source_afterwards(archive):
    con = duckdb.connect()
    try:
        before = {
            r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{archive}/prices-*.parquet')").fetchall()
        }
        assert "source" not in before
    finally:
        con.close()

    normalize_archive(archive, apply=True)
    assert verify(archive)


# ── idempotence ──────────────────────────────────────────────────────────────


def test_second_run_is_a_no_op(archive):
    assert normalize_archive(archive, apply=True) == 3
    assert normalize_archive(archive, apply=True) == 0


def test_second_run_leaves_bytes_identical(archive):
    normalize_archive(archive, apply=True)
    before = {p.name: p.read_bytes() for p in archive.glob("*.parquet")}
    normalize_archive(archive, apply=True)
    assert {p.name: p.read_bytes() for p in archive.glob("*.parquet")} == before


# ── safety ───────────────────────────────────────────────────────────────────


def test_non_midnight_day_is_refused(tmp_path):
    frame = _legacy()
    frame["day"] = pd.to_datetime(["2025-06-01 13:45:00"] * len(frame))
    frame.to_parquet(tmp_path / "prices-2025.parquet", index=False)
    with pytest.raises(RuntimeError, match="non-midnight"):
        normalize_archive(tmp_path, apply=True)


def test_refused_file_is_left_untouched(tmp_path):
    frame = _legacy()
    frame["day"] = pd.to_datetime(["2025-06-01 13:45:00"] * len(frame))
    path = tmp_path / "prices-2025.parquet"
    frame.to_parquet(path, index=False)
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        normalize_archive(tmp_path, apply=True)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_main_rejects_a_missing_dir(tmp_path):
    assert main(["--archive-dir", str(tmp_path / "nope")]) == 1


def test_main_rejects_a_dir_with_no_prices_files(tmp_path):
    assert main(["--archive-dir", str(tmp_path)]) == 1


def test_main_dry_run_succeeds(archive):
    assert main(["--archive-dir", str(archive)]) == 0
    assert _cols(archive / "prices-2025.parquet") == ["item_slug", "day", "mean_price", "volume"]


def test_main_apply_normalizes(archive):
    assert main(["--archive-dir", str(archive), "--apply"]) == 0
    assert _cols(archive / "prices-2025.parquet") == CANONICAL
