# The daily chain survives a read-only Postgres

**Date:** 2026-10-11

## Why

Prod Supabase is on the free tier and measured **986 MB** against its **500 MB** cap on
2026-10-11 (`pg_database_size`, read-only). It grows about 170 MB a week. Past the cap,
Supabase documents read-only mode, where every write fails with SQLSTATE 25006. It was not
enforced on the day (`default_transaction_read_only = off`, and the 10-11 runs wrote), but
nothing says when it will be.

Had it flipped, four writes would have failed the chain:

| Write | Step | Effect |
|---|---|---|
| `alembic upgrade head` | Aggregator, first step (`run_task.py migrate`) | Aggregator fails, so Price Forecast never runs |
| `collection_runs` insert | Aggregator (`collectors/pipeline.py`) | Same, after the prices were already saved |
| `accuracy_alerts` commit | Price Forecast, drift check before predict | Run fails before any forecast |
| `item_forecasts` upsert | Price Forecast (`_write_forecasts_to_db`) | Raised after 3 retries, **before** the Parquet mirror write, so the day's forecasts landed nowhere |

A forecast that is never written cannot be recreated. Outcomes can be re-scored from archive
prices, but the served band of that day is gone.

## What changed

- `db/read_only.py`: `is_read_only_error` (SQLSTATE 25006 anywhere in the cause chain, or the
  Postgres message) and `report_read_only`, which logs an error and emits a GitHub
  `::warning` annotation so a green run still shows the skip.
- `_write_forecasts_to_db` writes the Parquet mirror **first**. A read-only error rolls back
  and skips the Postgres write without retrying. Every other error keeps the existing
  three-retry-then-raise behaviour.
- The shadow-candidate write, the drift-alert commits, the `collection_runs` insert and the
  migration step skip on read-only. The migration step checks
  `default_transaction_read_only` before running Alembic. Shadow candidates have no Parquet
  mirror, so on a read-only night they are lost. That is accepted: they are shadow only.
- Backtest Accuracy is unchanged and fails loudly on a read-only database. Its outcomes can
  be re-scored later.

Tests: `tests/test_read_only_postgres.py` (14). The fast gate passes (2,825).

## Not in this change

The storage move itself: `specs/2026-10-11-postgres-to-parquet-history-design.md`.
A full local backup of every table was taken on 2026-10-11 before any change
(`~/personal-projects/cs2-analyzer/supabase-backup-2026-10-11/`, 18 tables, 56 MB; row
counts, per-column non-null counts and numeric sums match Postgres on the five large tables).
