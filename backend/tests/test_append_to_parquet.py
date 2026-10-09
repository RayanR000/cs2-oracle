"""Integration tests for scripts/append_to_parquet.py monthly partitioning.

Uses the --snapshot-csv path, which never touches the database, so these run
without a live DB (a dummy DATABASE_URL is injected only to satisfy the lazy
engine import).
"""

import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from db.archive import CANONICAL_PRICE_COLUMNS  # noqa: E402

SCRIPT = BACKEND / "scripts" / "append_to_parquet.py"


def _run(date, out_dir, csv_path):
    env = {**os.environ, "DATABASE_URL": "postgresql://u:p@localhost:5432/db", "ENVIRONMENT": "test"}
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--date", date, "--out-dir", str(out_dir), "--snapshot-csv", str(csv_path)],
        cwd=str(BACKEND),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"script failed:\n{res.stdout}\n{res.stderr}"
    return res


def _write_csv(path, day, rows):
    df = pd.DataFrame([{"item_slug": s, "day": day, "source": src, "price": p, "volume": v} for (s, src, p, v) in rows])
    df.to_csv(path, index=False)


def _count(pq):
    return duckdb.connect().sql(f"SELECT count(*) FROM read_parquet('{pq}')").fetchone()[0]


def test_writes_monthly_files_not_yearly(tmp_path):
    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-08-03", [("ak-redline", "aggregator_csfloat", 10.0, 5)])
    _run("2026-08-03", tmp_path, csv)

    arch = tmp_path / "price-archive"
    assert (arch / "prices-2026-08.parquet").exists()
    # the old yearly layout must NOT be produced
    assert not (arch / "prices-2026.parquet").exists()


def test_snapshots_parquet_is_not_written(tmp_path):
    """Retired as a pure projection of prices-*; see compact_price_archive.py."""
    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-08-03", [("ak-redline", "aggregator_csfloat", 10.0, 5)])
    _run("2026-08-03", tmp_path, csv)

    arch = tmp_path / "price-archive"
    assert list(arch.glob("snapshots-*.parquet")) == []


def test_redundant_price_columns_are_not_written(tmp_path):
    """median/min/max were exact copies of mean_price — 45% of archive bytes."""
    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-08-03", [("ak-redline", "aggregator_csfloat", 10.0, 5)])
    _run("2026-08-03", tmp_path, csv)

    pq = tmp_path / "price-archive" / "prices-2026-08.parquet"
    cols = [r[0] for r in duckdb.connect().sql(f"DESCRIBE SELECT * FROM read_parquet('{pq}')").fetchall()]
    assert set(cols) == set(CANONICAL_PRICE_COLUMNS)


def test_same_month_appends_and_dedups(tmp_path):
    arch = tmp_path / "price-archive"

    csv1 = tmp_path / "d1.csv"
    _write_csv(csv1, "2026-08-03", [("ak-redline", "aggregator_csfloat", 10.0, 5)])
    _run("2026-08-03", tmp_path, csv1)

    # next day, same month: one new key + a re-report of the same key (dedup keep=last)
    csv2 = tmp_path / "d2.csv"
    _write_csv(csv2, "2026-08-04", [("ak-redline", "aggregator_csfloat", 11.0, 6)])
    _run("2026-08-04", tmp_path, csv2)

    prices = arch / "prices-2026-08.parquet"
    # two distinct (slug, day, source) rows across the two days
    assert _count(prices) == 2


def test_different_months_go_to_separate_files(tmp_path):
    arch = tmp_path / "price-archive"

    csv_jul = tmp_path / "jul.csv"
    _write_csv(csv_jul, "2026-07-31", [("ak-redline", "aggregator_csfloat", 9.0, 4)])
    _run("2026-07-31", tmp_path, csv_jul)

    csv_aug = tmp_path / "aug.csv"
    _write_csv(csv_aug, "2026-08-01", [("ak-redline", "aggregator_csfloat", 10.0, 5)])
    _run("2026-08-01", tmp_path, csv_aug)

    assert (arch / "prices-2026-07.parquet").exists()
    assert (arch / "prices-2026-08.parquet").exists()
    assert _count(arch / "prices-2026-07.parquet") == 1
    assert _count(arch / "prices-2026-08.parquet") == 1


def _schema(pq):
    return [(r[0], r[1]) for r in duckdb.connect().sql(f"DESCRIBE SELECT * FROM read_parquet('{pq}')").fetchall()]


def test_day_is_written_as_date_not_timestamp(tmp_path):
    """A TIMESTAMP here is what made the yearly and monthly files disagree."""
    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-09-03", [("AK-47 | Redline (FT)", "aggregator_sync", 10.0, 3)])
    _run("2026-09-03", tmp_path, csv)
    pq = tmp_path / "price-archive" / "prices-2026-09.parquet"
    assert dict(_schema(pq))["day"] == "DATE"


def test_columns_are_written_in_canonical_order(tmp_path):
    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-09-03", [("AK-47 | Redline (FT)", "aggregator_sync", 10.0, 3)])
    _run("2026-09-03", tmp_path, csv)
    pq = tmp_path / "price-archive" / "prices-2026-09.parquet"
    assert [c for c, _ in _schema(pq)] == list(CANONICAL_PRICE_COLUMNS)


def test_appending_to_a_date_typed_file_does_not_duplicate_rows(tmp_path):
    """The dedup key includes `day`. Re-running a day must not leave two rows
    because the file round-tripped as `date` and the new frame holds Timestamps."""
    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-09-03", [("AK-47 | Redline (FT)", "aggregator_sync", 10.0, 3)])
    _run("2026-09-03", tmp_path, csv)
    _run("2026-09-03", tmp_path, csv)
    pq = tmp_path / "price-archive" / "prices-2026-09.parquet"
    assert _count(pq) == 1


def test_exchange_rates_day_is_also_a_date(tmp_path):
    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-09-03", [("AK-47 | Redline (FT)", "aggregator_sync", 10.0, 3)])
    fx = tmp_path / "fx.csv"
    pd.DataFrame([{"currency": "EUR", "rate": 0.92, "day": "2026-09-03"}]).to_csv(fx, index=False)
    env = {**os.environ, "DATABASE_URL": "postgresql://u:p@localhost:5432/db", "ENVIRONMENT": "test"}
    res = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--date",
            "2026-09-03",
            "--out-dir",
            str(tmp_path),
            "--snapshot-csv",
            str(csv),
            "--exchange-rates-csv",
            str(fx),
        ],
        cwd=str(BACKEND),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    pq = tmp_path / "price-archive" / "exchange-rates-2026.parquet"
    assert dict(_schema(pq))["day"] == "DATE"


def test_new_month_file_matches_what_the_migration_produces(tmp_path):
    """The writer and normalize_price_schema.py must agree, or every new month
    immediately drifts back out of canonical shape."""
    sys.path.insert(0, str(BACKEND))
    from scripts.normalize_price_schema import needs_rewrite

    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-09-03", [("AK-47 | Redline (FT)", "aggregator_sync", 10.0, 3)])
    _run("2026-09-03", tmp_path, csv)
    pq = tmp_path / "price-archive" / "prices-2026-09.parquet"
    assert not needs_rewrite(_schema(pq))


class TestIngestedAt:
    """`ingested_at` records ARRIVAL; `day` records what the row describes.

    Added 2026-08-08 for the embargo. A purge computed from `day` assumes a row
    dated `d` was knowable on `d`, which a backfill writer violates by
    definition — and the archive had no arrival timestamp anywhere to check it
    against. See `db/archive.py` and
    `docs/changelog/2026-08-08-embargo-and-harness-hygiene.md`.
    """

    def test_a_fresh_row_is_stamped_with_a_timestamp(self, tmp_path):
        csv = tmp_path / "snap.csv"
        _write_csv(csv, "2026-09-03", [("AK-47 | Redline (FT)", "aggregator_sync", 10.0, 3)])
        _run("2026-09-03", tmp_path, csv)
        pq = tmp_path / "price-archive" / "prices-2026-09.parquet"

        assert dict(_schema(pq))["ingested_at"] == "TIMESTAMP"
        stamped = (
            duckdb.connect()
            .sql(f"SELECT count(*) FROM read_parquet('{pq}') WHERE ingested_at IS NOT NULL")
            .fetchone()[0]
        )
        assert stamped == 1

    def test_arrival_is_not_the_day_being_exported(self, tmp_path):
        """Backdating `--date` must not backdate the arrival: re-exporting an
        old day is something that happens now, and that is the fact worth
        recording."""
        csv = tmp_path / "snap.csv"
        _write_csv(csv, "2026-01-05", [("AK-47 | Redline (FT)", "aggregator_sync", 10.0, 3)])
        _run("2026-01-05", tmp_path, csv)
        pq = tmp_path / "price-archive" / "prices-2026-01.parquet"

        day, arrived = duckdb.connect().sql(f"SELECT day, ingested_at FROM read_parquet('{pq}')").fetchone()
        assert str(day) == "2026-01-05"
        assert pd.Timestamp(arrived) > pd.Timestamp("2026-01-06")

    def test_a_re_append_keeps_the_first_arrival(self, tmp_path):
        """`keep="last"` replaces a row wholesale, which is right for a
        corrected price and wrong for the arrival time behind it. Re-stamping
        on every re-run would date the whole month forward and make the column
        useless as an embargo input."""
        csv = tmp_path / "snap.csv"
        _write_csv(csv, "2026-09-03", [("AK-47 | Redline (FT)", "aggregator_sync", 10.0, 3)])
        _run("2026-09-03", tmp_path, csv)
        pq = tmp_path / "price-archive" / "prices-2026-09.parquet"
        first = duckdb.connect().sql(f"SELECT ingested_at FROM read_parquet('{pq}')").fetchone()[0]

        corrected = tmp_path / "snap2.csv"
        _write_csv(corrected, "2026-09-03", [("AK-47 | Redline (FT)", "aggregator_sync", 11.0, 3)])
        _run("2026-09-03", tmp_path, corrected)

        price, arrived = duckdb.connect().sql(f"SELECT mean_price, ingested_at FROM read_parquet('{pq}')").fetchone()
        assert price == 11.0, "the corrected price should win"
        assert arrived == first, "the original arrival should not"


class TestVolumeAbsenceIsNull:
    """An unobserved volume must reach the archive as NULL, never as 0.

    The aggregator feeds carry no volume field at all, so every live row's
    volume is *unknown*. Writing 0 makes that indistinguishable from a real
    zero -- and a real zero never occurs, because a day with no sale produces
    an absent row rather than a zero one. The 0 also defeats every guard in
    `_compute_volume_features`: `has_volume` tests notna() so it stays True and
    `volume_missing` reports 0, i.e. "present". See
    `models/forecaster.py` and the shelved volume features.
    """

    def test_missing_volume_lands_as_null(self, tmp_path):
        csv = tmp_path / "snap.csv"
        pd.DataFrame(
            [
                {
                    "item_slug": "AK-47 | Redline (FT)",
                    "day": "2026-10-02",
                    "source": "aggregator_csgotrader",
                    "price": 10.0,
                    "volume": None,
                }
            ]
        ).to_csv(csv, index=False)
        _run("2026-10-02", tmp_path, csv)

        pq = tmp_path / "price-archive" / "prices-2026-10.parquet"
        vol = duckdb.connect().sql(f"SELECT volume FROM read_parquet('{pq}')").fetchone()[0]
        assert vol is None, f"absent volume must stay NULL, got {vol!r}"

    def test_a_real_volume_still_sums(self, tmp_path):
        """The NULL path must not cost the aggregation its real values."""
        csv = tmp_path / "snap.csv"
        pd.DataFrame(
            [
                {
                    "item_slug": "AWP | Asiimov (FT)",
                    "day": "2026-10-02",
                    "source": "aggregator_sync",
                    "price": 10.0,
                    "volume": 3,
                },
                {
                    "item_slug": "AWP | Asiimov (FT)",
                    "day": "2026-10-02",
                    "source": "aggregator_sync",
                    "price": 12.0,
                    "volume": 4,
                },
            ]
        ).to_csv(csv, index=False)
        _run("2026-10-02", tmp_path, csv)

        pq = tmp_path / "price-archive" / "prices-2026-10.parquet"
        vol = duckdb.connect().sql(f"SELECT volume FROM read_parquet('{pq}')").fetchone()[0]
        assert vol == 7, f"two observed volumes should sum, got {vol!r}"

    def test_a_partial_group_sums_only_what_was_observed(self, tmp_path):
        """One NULL among real values is a gap in the panel, not a zero."""
        csv = tmp_path / "snap.csv"
        pd.DataFrame(
            [
                {
                    "item_slug": "M4A4 | Howl (FN)",
                    "day": "2026-10-02",
                    "source": "aggregator_sync",
                    "price": 10.0,
                    "volume": 5,
                },
                {
                    "item_slug": "M4A4 | Howl (FN)",
                    "day": "2026-10-02",
                    "source": "aggregator_sync",
                    "price": 12.0,
                    "volume": None,
                },
            ]
        ).to_csv(csv, index=False)
        _run("2026-10-02", tmp_path, csv)

        pq = tmp_path / "price-archive" / "prices-2026-10.parquet"
        vol = duckdb.connect().sql(f"SELECT volume FROM read_parquet('{pq}')").fetchone()[0]
        assert vol == 5


class TestDuckDBMerge:
    """`_append_parquet` merges in DuckDB (2026-10-09). These pin the pandas
    `concat` + `drop_duplicates(keep="last")` semantics it replaced."""

    @staticmethod
    def _frame(rows, ingested="2026-10-09 12:00:00"):
        df = pd.DataFrame(rows, columns=["item_slug", "day", "source", "mean_price", "volume"])
        df["day"] = pd.to_datetime(df["day"])
        df["volume"] = df["volume"].astype("Int64")
        df["ingested_at"] = pd.Timestamp(ingested)
        return df

    @staticmethod
    def _rows(pq):
        return duckdb.connect().sql(f"SELECT * FROM read_parquet('{pq}')").fetchall()

    def test_last_duplicate_in_a_batch_wins(self, tmp_path):
        from scripts.append_to_parquet import PRICE_KEYS, _append_parquet

        pq = tmp_path / "p.parquet"
        batch = self._frame([("a", "2026-10-01", "s", 1.0, None), ("a", "2026-10-01", "s", 2.0, None)])
        _append_parquet(pq, batch, PRICE_KEYS)
        assert [r[3] for r in self._rows(pq)] == [2.0]

    def test_null_keys_match_each_other(self, tmp_path):
        """`dropna=False`: a NULL-source row is replaced by its re-run, not duplicated."""
        from scripts.append_to_parquet import PRICE_KEYS, _append_parquet

        pq = tmp_path / "p.parquet"
        _append_parquet(pq, self._frame([("a", "2025-12-01", None, 1.0, 3)], "2026-10-01"), PRICE_KEYS)
        _append_parquet(pq, self._frame([("a", "2025-12-01", None, 2.0, 3)], "2026-10-09"), PRICE_KEYS)
        rows = self._rows(pq)
        assert len(rows) == 1
        assert rows[0][3] == 2.0
        assert str(rows[0][5]) == "2026-10-01 00:00:00", "first arrival must survive the replacement"

    def test_a_file_predating_source_reads_it_as_null(self, tmp_path):
        from scripts.append_to_parquet import PRICE_KEYS, _append_parquet

        pq = tmp_path / "p.parquet"
        duckdb.connect().sql(
            f"COPY (SELECT 'a' AS item_slug, DATE '2025-12-01' AS day, 1.0 AS mean_price, "
            f"3::BIGINT AS volume, NULL::TIMESTAMP AS ingested_at) TO '{pq}' (FORMAT PARQUET)"
        )
        _append_parquet(
            pq, self._frame([("a", "2025-12-01", None, 2.0, 3), ("b", "2025-12-01", "s", 5.0, 1)]), PRICE_KEYS
        )
        rows = sorted(self._rows(pq), key=lambda r: r[0])
        assert [(r[0], r[2], r[3]) for r in rows] == [("a", None, 2.0), ("b", "s", 5.0)]
        assert str(rows[0][5]) == "2026-10-09 12:00:00", "an unknown arrival takes the new stamp"

    def test_written_zstd_and_sorted_on_the_key(self, tmp_path):
        from scripts.append_to_parquet import PRICE_KEYS, _append_parquet

        pq = tmp_path / "p.parquet"
        _append_parquet(
            pq, self._frame([("b", "2026-10-02", "y", 1.0, None), ("a", "2026-10-01", "z", 1.0, None)]), PRICE_KEYS
        )
        _append_parquet(pq, self._frame([("c", "2026-10-03", "x", 1.0, None)]), PRICE_KEYS)
        con = duckdb.connect()
        codecs = {r[0] for r in con.sql(f"SELECT DISTINCT compression FROM parquet_metadata('{pq}')").fetchall()}
        assert codecs == {"ZSTD"}
        assert [r[2] for r in self._rows(pq)] == ["x", "y", "z"]
