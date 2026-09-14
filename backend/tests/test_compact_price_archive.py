"""Tests for scripts/compact_price_archive.py."""

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compact_price_archive import (
    absorb_orphan_snapshot_rows,
    columns_to_drop,
    compact_columns,
    drop_snapshots,
    main,
)

EIGHT_COL = ["item_slug", "day", "mean_price", "median_price", "volume", "min_price", "max_price", "source"]


def _prices_frame(n=4, collapsed=True):
    return pd.DataFrame(
        {
            "item_slug": [f"AK-47 | Redline (Field-Tested) {i}" for i in range(n)],
            "day": pd.to_datetime(["2026-05-01"] * n),
            "mean_price": [10.0 + i for i in range(n)],
            "median_price": [10.0 + i for i in range(n)],
            "volume": [0] * n,
            "min_price": [10.0 + i if collapsed else 1.0 for i in range(n)],
            "max_price": [10.0 + i if collapsed else 99.0 for i in range(n)],
            "source": ["aggregator_sync"] * n,
        }
    )[EIGHT_COL]


def _write(archive, name, frame):
    frame.to_parquet(archive / name, index=False)


def _cols(path):
    con = duckdb.connect()
    try:
        return [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
    finally:
        con.close()


@pytest.fixture
def archive(tmp_path):
    d = tmp_path / "price-archive"
    d.mkdir()
    return d


# ── column policy ──────────────────────────────────────────────────────


def test_median_price_is_dropped_from_every_prices_file():
    assert columns_to_drop("prices-2019.parquet", ["item_slug", "day", "mean_price", "median_price", "volume"]) == [
        "median_price"
    ]


def test_march_and_april_2026_keep_their_range_columns():
    """These two files hold real min != max rows; only median_price goes."""
    for name in ("prices-2026-03.parquet", "prices-2026-04.parquet"):
        assert columns_to_drop(name, EIGHT_COL) == ["median_price"]


def test_other_2026_files_lose_the_collapsed_range_columns():
    assert columns_to_drop("prices-2026-07.parquet", EIGHT_COL) == ["median_price", "min_price", "max_price"]


def test_volume_is_never_dropped():
    """price_resolution.py selects volume even though it is all-zero."""
    for name in ("prices-2019.parquet", "prices-2026-07.parquet"):
        assert "volume" not in columns_to_drop(name, EIGHT_COL)


def test_item_slug_day_and_source_are_never_dropped():
    dropped = columns_to_drop("prices-2026-07.parquet", EIGHT_COL)
    for keeper in ("item_slug", "day", "mean_price", "source"):
        assert keeper not in dropped


# ── rewrite behaviour ──────────────────────────────────────────────────


def test_dry_run_leaves_files_untouched(archive):
    _write(archive, "prices-2026-07.parquet", _prices_frame())
    before = (archive / "prices-2026-07.parquet").read_bytes()
    compact_columns(archive, apply=False)
    assert (archive / "prices-2026-07.parquet").read_bytes() == before


def test_apply_drops_columns_and_preserves_rows(archive):
    frame = _prices_frame(n=5)
    _write(archive, "prices-2026-07.parquet", frame)

    compact_columns(archive, apply=True)

    path = archive / "prices-2026-07.parquet"
    assert _cols(path) == ["item_slug", "day", "mean_price", "volume", "source"]
    out = pd.read_parquet(path)
    assert len(out) == 5
    pd.testing.assert_series_equal(out["mean_price"], frame["mean_price"])
    pd.testing.assert_series_equal(out["item_slug"], frame["item_slug"])


def test_apply_preserves_row_order(archive):
    frame = _prices_frame(n=6)
    frame = frame.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
    _write(archive, "prices-2026-07.parquet", frame)

    compact_columns(archive, apply=True)

    out = pd.read_parquet(archive / "prices-2026-07.parquet")
    assert out["item_slug"].tolist() == frame["item_slug"].tolist()


def test_range_columns_survive_in_march(archive):
    _write(archive, "prices-2026-03.parquet", _prices_frame(collapsed=False))

    compact_columns(archive, apply=True)

    cols = _cols(archive / "prices-2026-03.parquet")
    assert "min_price" in cols and "max_price" in cols
    assert "median_price" not in cols
    out = pd.read_parquet(archive / "prices-2026-03.parquet")
    assert out["min_price"].tolist() == [1.0] * 4
    assert out["max_price"].tolist() == [99.0] * 4


def test_no_temp_file_is_left_behind(archive):
    _write(archive, "prices-2026-07.parquet", _prices_frame())
    compact_columns(archive, apply=True)
    assert list(archive.glob("*.tmp")) == []


def test_rerun_is_a_no_op(archive):
    _write(archive, "prices-2026-07.parquet", _prices_frame())
    compact_columns(archive, apply=True)
    first = (archive / "prices-2026-07.parquet").read_bytes()
    compact_columns(archive, apply=True)
    assert (archive / "prices-2026-07.parquet").read_bytes() == first


def test_missing_archive_dir_is_reported_not_crashed(tmp_path):
    assert main(["--archive-dir", str(tmp_path / "nope")]) == 1


def test_a_dir_with_no_prices_files_fails_loudly(archive):
    """A typo'd --archive-dir in CI must not exit 0 having done nothing."""
    assert main(["--archive-dir", str(archive), "--apply"]) == 1


def test_a_dir_with_prices_files_succeeds(archive):
    _write(archive, "prices-2026-07.parquet", _prices_frame())
    assert main(["--archive-dir", str(archive), "--apply"]) == 0


# ── snapshot retirement ────────────────────────────────────────────────


def _snapshot_frame(prices):
    return prices[["item_slug", "day", "source"]].assign(price=prices["mean_price"], volume=prices["volume"])


def test_derivable_snapshot_is_deleted(archive):
    prices = _prices_frame()
    _write(archive, "prices-2026-07.parquet", prices)
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(prices))

    freed = drop_snapshots(archive, apply=True)

    assert not (archive / "snapshots-2026-07.parquet").exists()
    assert freed > 0


def test_dry_run_keeps_the_snapshot(archive):
    prices = _prices_frame()
    _write(archive, "prices-2026-07.parquet", prices)
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(prices))

    assert drop_snapshots(archive, apply=False) == 0
    assert (archive / "snapshots-2026-07.parquet").exists()


def test_snapshot_with_a_divergent_price_is_kept(archive):
    """The safety check is what makes the delete defensible; it must bite."""
    prices = _prices_frame()
    _write(archive, "prices-2026-07.parquet", prices)
    snap = _snapshot_frame(prices)
    snap.loc[0, "price"] = 999.0
    _write(archive, "snapshots-2026-07.parquet", snap)

    drop_snapshots(archive, apply=True)

    assert (archive / "snapshots-2026-07.parquet").exists()


def test_snapshot_with_an_unmatched_row_is_kept(archive):
    prices = _prices_frame()
    _write(archive, "prices-2026-07.parquet", prices)
    snap = _snapshot_frame(prices)
    snap.loc[0, "item_slug"] = "Not In Prices (Factory New)"
    _write(archive, "snapshots-2026-07.parquet", snap)

    drop_snapshots(archive, apply=True)

    assert (archive / "snapshots-2026-07.parquet").exists()


def test_snapshot_without_a_prices_counterpart_is_kept(archive):
    prices = _prices_frame()
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(prices))

    drop_snapshots(archive, apply=True)

    assert (archive / "snapshots-2026-07.parquet").exists()


def test_fingerprint_is_stable_across_reads_of_the_same_data(archive):
    """Guards the DOUBLE-vs-DECIMAL trap: a float sum is not reproducible."""
    from scripts.compact_price_archive import _describe, _fingerprint

    frame = pd.DataFrame(
        {
            "item_slug": [f"item {i}" for i in range(20000)],
            "day": pd.to_datetime(["2026-05-01"] * 20000),
            "mean_price": [0.01 * (i + 1) for i in range(20000)],
            "volume": [i for i in range(20000)],
        }
    )
    _write(archive, "prices-2026-05.parquet", frame)
    path = archive / "prices-2026-05.parquet"

    con = duckdb.connect()
    try:
        cols = _describe(con, path)
        assert _fingerprint(con, path, cols) == _fingerprint(con, path, cols)
    finally:
        con.close()


def test_a_real_value_change_is_caught(archive):
    """The guard must still bite when data actually differs."""
    from scripts.compact_price_archive import _describe, _fingerprint

    a = _prices_frame(n=4)
    b = a.copy()
    b.loc[0, "mean_price"] = a.loc[0, "mean_price"] + 0.01
    _write(archive, "prices-2026-05.parquet", a)
    _write(archive, "prices-2026-06.parquet", b)

    con = duckdb.connect()
    try:
        cols = _describe(con, archive / "prices-2026-05.parquet")
        assert _fingerprint(con, archive / "prices-2026-05.parquet", cols) != _fingerprint(
            con, archive / "prices-2026-06.parquet", cols
        )
    finally:
        con.close()


# ── orphan absorption ──────────────────────────────────────────────────


def test_orphan_rows_are_absorbed_into_prices(archive):
    prices = _prices_frame(n=3)
    _write(archive, "prices-2026-07.parquet", prices)
    snap = _snapshot_frame(_prices_frame(n=4))  # 4th row has no prices match
    _write(archive, "snapshots-2026-07.parquet", snap)

    assert absorb_orphan_snapshot_rows(archive, apply=True) == 1

    out = pd.read_parquet(archive / "prices-2026-07.parquet")
    assert len(out) == 4
    orphan = out[out["item_slug"].str.endswith(" 3")].iloc[0]
    assert orphan["mean_price"] == 13.0


def test_orphan_absorption_dry_run_writes_nothing(archive):
    _write(archive, "prices-2026-07.parquet", _prices_frame(n=3))
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(_prices_frame(n=4)))
    before = (archive / "prices-2026-07.parquet").read_bytes()

    assert absorb_orphan_snapshot_rows(archive, apply=False) == 1

    assert (archive / "prices-2026-07.parquet").read_bytes() == before


def test_absorption_does_not_resurrect_dropped_columns(archive):
    _write(archive, "prices-2026-07.parquet", _prices_frame(n=3))
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(_prices_frame(n=4)))

    compact_columns(archive, apply=True)
    absorb_orphan_snapshot_rows(archive, apply=True)

    assert _cols(archive / "prices-2026-07.parquet") == ["item_slug", "day", "mean_price", "volume", "source"]


def test_absorption_is_a_no_op_when_nothing_is_orphaned(archive):
    prices = _prices_frame()
    _write(archive, "prices-2026-07.parquet", prices)
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(prices))

    assert absorb_orphan_snapshot_rows(archive, apply=True) == 0


def test_orphans_are_absorbed_then_the_snapshot_is_deleted(archive):
    """End-to-end: an orphan must not block retirement, nor be lost by it."""
    _write(archive, "prices-2026-07.parquet", _prices_frame(n=3))
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(_prices_frame(n=4)))

    main(["--archive-dir", str(archive), "--apply"])

    assert not (archive / "snapshots-2026-07.parquet").exists()
    assert len(pd.read_parquet(archive / "prices-2026-07.parquet")) == 4


def test_rerun_after_absorption_adds_no_duplicates(archive):
    _write(archive, "prices-2026-07.parquet", _prices_frame(n=3))
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(_prices_frame(n=4)))

    absorb_orphan_snapshot_rows(archive, apply=True)
    assert absorb_orphan_snapshot_rows(archive, apply=True) == 0
    assert len(pd.read_parquet(archive / "prices-2026-07.parquet")) == 4


def test_snapshot_check_runs_against_already_compacted_prices(archive):
    """Order matters: columns are compacted first, snapshots verified after."""
    prices = _prices_frame()
    _write(archive, "prices-2026-07.parquet", prices)
    _write(archive, "snapshots-2026-07.parquet", _snapshot_frame(prices))

    main(["--archive-dir", str(archive), "--apply"])

    assert not (archive / "snapshots-2026-07.parquet").exists()
    assert _cols(archive / "prices-2026-07.parquet") == ["item_slug", "day", "mean_price", "volume", "source"]
