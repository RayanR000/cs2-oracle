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
SCRIPT = BACKEND / "scripts" / "append_to_parquet.py"


def _run(date, out_dir, csv_path):
    env = {**os.environ, "DATABASE_URL": "postgresql://u:p@localhost:5432/db",
           "ENVIRONMENT": "test"}
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--date", date,
         "--out-dir", str(out_dir), "--snapshot-csv", str(csv_path)],
        cwd=str(BACKEND), env=env, capture_output=True, text=True,
    )
    assert res.returncode == 0, f"script failed:\n{res.stdout}\n{res.stderr}"
    return res


def _write_csv(path, day, rows):
    df = pd.DataFrame([
        {"item_slug": s, "day": day, "source": src, "price": p, "volume": v}
        for (s, src, p, v) in rows
    ])
    df.to_csv(path, index=False)


def _count(pq):
    return duckdb.connect().sql(
        f"SELECT count(*) FROM read_parquet('{pq}')").fetchone()[0]


def test_writes_monthly_files_not_yearly(tmp_path):
    csv = tmp_path / "snap.csv"
    _write_csv(csv, "2026-08-03", [("ak-redline", "aggregator_csfloat", 10.0, 5)])
    _run("2026-08-03", tmp_path, csv)

    arch = tmp_path / "price-archive"
    assert (arch / "prices-2026-08.parquet").exists()
    assert (arch / "snapshots-2026-08.parquet").exists()
    # the old yearly layout must NOT be produced
    assert not (arch / "prices-2026.parquet").exists()
    assert not (arch / "snapshots-2026.parquet").exists()


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
