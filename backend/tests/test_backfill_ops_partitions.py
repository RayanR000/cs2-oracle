"""The ops partition backfill merges the Supabase backup with the single-file mirror.

Pins the merge rule measured on 2026-10-11: Postgres wins every shared row whole
(the mirror kept a re-run band and the retired dollar-basis verdict on ~38k rows),
the mirror adds only what Postgres lacks, deleted rows survive flagged, and the
single-file mirror is never written.
"""

import sys
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backfill_ops_partitions import main

CUTOFF = date(2026, 8, 5)


def _forecast(i, item, d, low, **kw):
    return dict(
        item_id=item,
        forecast_date=d,
        horizon_days=7,
        price_low=low,
        price_mid=low + 1,
        price_high=low + 2,
        created_at=datetime(2026, 8, 1),
        **({"id": i} if i else {}),
        **kw,
    )


def _outcome(fid, item, d, in_interval, evaluated, **kw):
    return dict(
        forecast_id=fid,
        item_id=item,
        forecast_date=d,
        horizon_days=7,
        actual_price=10.0,
        predicted_price_low=9.0,
        in_interval=in_interval,
        evaluated_at=evaluated,
        **kw,
    )


@pytest.fixture
def stores(tmp_path):
    backup, ops = tmp_path / "backup", tmp_path / "archive" / "ops"
    backup.mkdir()
    ops.mkdir(parents=True)
    d1, d2 = date(2026, 7, 31), date(2026, 8, 4)

    pd.DataFrame(
        [
            _forecast(1, 10, d1, 5.0),  # shared, Postgres band differs
            _forecast(2, 11, d1, 6.0),  # backup only: needs a slug from items
            _forecast(3, 10, CUTOFF, 7.0),
        ]
    ).to_parquet(backup / "item_forecasts.parquet")
    pd.DataFrame(
        [
            _outcome(1, 10, d1, 1, datetime(2026, 8, 11, 21, 10)),  # refreshed in Postgres only
            _outcome(2, 11, d1, 0, datetime(2026, 8, 5)),
        ]
    ).assign(id=[100, 101]).to_parquet(backup / "forecast_outcomes.parquet")
    pd.DataFrame(
        [
            dict(
                id=50,
                item_id=10,
                forecast_date=d1,
                horizon_days=7,
                component="c",
                candidate_name="n",
                candidate_version="v",
            )
        ]
    ).to_parquet(backup / "forecast_candidates.parquet")
    pd.DataFrame([dict(candidate_id=50, actual_price=1.0)]).to_parquet(backup / "forecast_candidate_outcomes.parquet")
    pd.DataFrame([dict(id=10, item_id="ak"), dict(id=11, item_id="awp")]).to_parquet(backup / "items.parquet")

    # Mirror: TIMESTAMP forecast dates, no id, carries slug.
    ts = lambda d: pd.Timestamp(d)  # noqa: E731
    pd.DataFrame(
        [
            {**_forecast(None, 10, ts(d1), 99.0), "item_slug": "ak"},  # the re-run band
            {**_forecast(None, 12, ts(d2), 8.0), "item_slug": None},  # deleted from Postgres
            {**_forecast(None, 10, ts(date(2026, 8, 6)), 9.0), "item_slug": "ak"},  # after the backup
        ]
    ).to_parquet(ops / "item_forecasts.parquet")
    pd.DataFrame(
        [
            {**_outcome(1, 10, ts(d1), 0, datetime(2026, 8, 5)), "item_slug": "ak"},  # dollar-basis verdict
            {**_outcome(77, 12, ts(d2), 1, datetime(2026, 8, 6)), "item_slug": None},  # orphan
        ]
    ).to_parquet(ops / "forecast_outcomes.parquet")
    return backup, ops


def _read(path):
    df = duckdb.sql(f"SELECT * FROM read_parquet('{path}') ORDER BY ALL").df()
    if "forecast_date" in df:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    return df


def _run(backup, ops, *extra):
    return main(["--backup-dir", str(backup), "--archive-dir", str(ops.parent), *extra])


def test_dry_run_writes_nothing(stores):
    backup, ops = stores
    before = sorted(p.name for p in ops.iterdir())
    assert _run(backup, ops) == 0
    assert sorted(p.name for p in ops.iterdir()) == before


def test_postgres_wins_shared_rows_and_mirror_fills_the_rest(stores):
    backup, ops = stores
    mirror_before = (ops / "item_forecasts.parquet").read_bytes()
    assert _run(backup, ops, "--apply") == 0

    fc = _read(ops / "item_forecasts" / "*.parquet").set_index(["item_id", "forecast_date"])
    shared = fc.loc[(10, date(2026, 7, 31))]
    assert shared["price_low"] == 5.0 and shared["id"] == 1  # Postgres band, not the re-run's 99
    assert shared["item_slug"] == "ak"  # mirror-only column still carried
    assert fc.loc[(11, date(2026, 7, 31)), "item_slug"] == "awp"  # filled from items

    deleted = fc.loc[(12, date(2026, 8, 4))]
    assert bool(deleted["pg_deleted"]) and deleted["id"] == 77  # id recovered from its outcome
    newer = fc.loc[(10, date(2026, 8, 6))]
    assert not bool(newer["pg_deleted"])  # after the backup: newer, not deleted

    fo = _read(ops / "forecast_outcomes" / "*.parquet").set_index("forecast_id")
    assert fo.loc[1, "in_interval"] == 1  # the refreshed verdict, not the mirror's 0
    assert bool(fo.loc[77, "pg_deleted"]) and not bool(fo.loc[1, "pg_deleted"])

    assert sorted(p.name for p in (ops / "item_forecasts").iterdir()) == ["2026-07.parquet", "2026-08.parquet"]
    assert len(_read(ops / "forecast_candidate_outcomes" / "*.parquet")) == 1
    assert "_part" not in _read(ops / "forecast_candidate_outcomes" / "*.parquet").columns
    assert len(_read(ops / "items.parquet")) == 2
    assert (ops / "item_forecasts.parquet").read_bytes() == mirror_before  # mirror never written


def test_refuses_to_overwrite_existing_targets(stores):
    backup, ops = stores
    assert _run(backup, ops, "--apply") == 0
    assert _run(backup, ops, "--apply") == 1
    assert _run(backup, ops, "--apply", "--overwrite") == 0


def test_orphan_candidate_outcome_fails(stores):
    backup, ops = stores
    pd.DataFrame([dict(candidate_id=999, actual_price=1.0)]).to_parquet(backup / "forecast_candidate_outcomes.parquet")
    assert _run(backup, ops, "--apply") == 1
    assert not (ops / "forecast_candidates").exists()
