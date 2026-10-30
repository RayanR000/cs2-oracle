# Phase 2: move the forecast-record readers off Postgres

**Date:** 2026-10-13
**Status:** approved 2026-10-14 (gate night, gate length, line references, `compare` signature and PR 1's read-only writes amended on review)
**Parent:** `docs/specs/2026-10-11-postgres-to-parquet-history-design.md`, "Phase 2"
**Depends on:** #118 (dual-write and nightly parity, merged 2026-10-13) and the 2026-10-11 id
fill (`ops-sync.yml` run 38329783100)
**Ships in:** six PRs, in the order below

## Goal

With Postgres read-only or gone, the nightly chain still scores, checks freshness and serves
forecasts and bands, and no number changes. Every reader of `item_forecasts`, `forecast_outcomes`,
`forecast_candidates`, `forecast_candidate_outcomes` and `items` that no frozen instrument
depends on reads the monthly partitions instead. Phase 3 is then only the instrument switches,
and Phase 4 only deletion.

This is storage only, as in the parent spec. No change to forecasts, bands or scoring.

## Why risk order, not the parent's list order

Prod was 986 MB against the free tier's 500 MB cap on 2026-10-11, so Postgres can go read-only
on any night. Last night's chain logged no read-only error, so it has not happened yet. When it
does, writers on that night put their new rows only in the partitions, with ids from 2^40, and
every reader still on Postgres goes quietly stale: scoring stops seeing new forecasts, the
freshness check fails, and the feedback factors and prior-day blending run on an old panel. The API is not
deployed, so it cannot go stale for anyone. The PRs therefore go in the order of what goes
stale first.

## The read module: `db/ops_store.py`

`connect(root=None, tables=TABLES) -> duckdb.DuckDBPyConnection` returns an in-memory DuckDB
connection with one view per named table (default all five), named after the Postgres tables:

| View | Reads | Filter |
|---|---|---|
| `item_forecasts` | `ops/item_forecasts/*.parquet` | `pg_deleted IS NOT TRUE` |
| `forecast_outcomes` | `ops/forecast_outcomes/*.parquet` | `pg_deleted IS NOT TRUE` |
| `forecast_candidates` | `ops/forecast_candidates/*.parquet` | none (no such column) |
| `forecast_candidate_outcomes` | `ops/forecast_candidate_outcomes/*.parquet` | none |
| `items` | `ops/items.parquet` | none |

- **Files** come from `ops_partitions.month_files`, the function the writer and
  `scripts/ops_parity.py` already use, so all three agree on which files exist.
- **Columns** are read with `union_by_name = true` and projected explicitly from
  `ops_partitions.SPECS`, so a column an older month lacks comes back NULL instead of
  vanishing (the multi-file trap in `.claude/rules/archive-reads.md`). Names and types match
  the Postgres tables, so `FROM forecast_outcomes o JOIN items i ON i.id = o.item_id` runs as
  written.
- **A table with no files raises**, naming the table. An empty view would give an empty panel,
  and an empty feedback panel snaps the factor to 1.0 without an error. Only the tables a reader
  names are checked, so a reader of `items` does not fail on a missing candidate table.
- **The file list is fixed when the connection opens.** A reader that runs after a write in the
  same process (the scoring panel after tonight's outcomes) opens a fresh connection;
  `reader(tables)` returns a `sql -> rows` function that does exactly that per call.
- **Rows** come back as named tuples, so `r.base_price` works as it does on a SQLAlchemy row.
  A reader written against `read(sql)` runs unchanged on either store, which is how read twice
  computes its Postgres side.
- **Root** defaults to `parquet.OPS_DIR`, the writers' root. Price Forecast, Backtest Accuracy
  and Forecast Freshness Check already check out all of `cs2-oracle-data` and link
  `price-archive`, so no workflow changes. Closed months stay in the tree beside their
  `ops-YYYY-MM` releases, so history reads see every month.

### Read twice

`compare(name, pg, parquet_value, *, tol=0.0, db=None)` takes the Postgres side as a callable,
so a failed Postgres read (Postgres gone, or an aborted transaction) is caught there, logged as a
`::warning`, and rolls back *db*. It logs both values (for a dict, the key count and every
differing key), and a `::warning` naming the reader and both values when they differ beyond
`tol`. It never fails a run. A switched reader always uses the Parquet value; Postgres is shown
for comparison only. The warning says that on a read-only night Postgres is the stale store and
a difference is expected; on any other night it blocks the next PR. `compare` and its Postgres reads are removed in Phase 4.

## PR sequence and gates

**Before PR 1:** nightly parity (`scripts/ops_parity.py` in Backtest Accuracy) passes 3
consecutive nights for every table. The first clean night is 2026-10-13 (run 38498152350 on
10-14: 0 open-month and 0 closed-month mismatches); the 10-12 night failed with 29 mismatches
from the gap between the backup and #118, fixed by ops-sync run 38344129562. This replaces the
parent spec's "≥7 nights" for the start of Phase 2: Postgres can go read-only any night, and
read twice keeps comparing every switched reader after the switch, so a longer wait buys less
than it risks. `ops_parity.py`'s log line changes to match in PR 1. **Between PRs:**
each PR waits for its predecessor to go through one green nightly chain with no unexplained
read-twice difference.

**If Postgres goes read-only mid-sequence:** switched readers keep working; unswitched readers
serve a stale panel and say so. The remaining PRs then go in order without the 3-night wait,
because there is no longer a live Postgres to stay equal to.

### PR 1: `ops_store`, backtest scoring and verdict refresh

**Shipped** in #122 (2026-10-30).

`scripts/backtest_accuracy.py` reads from the store: the matured forecasts (the
`FROM item_forecasts f` query in `backtest_forecasts`), the frozen-id set beside it, the scoring
panel (`_records_from_frozen_outcomes`), the verdict refresh (`_iter_outcome_rows` under
`_refresh_verdict_columns`) and the id→slug map (`_id_to_slug`, and the inline copy of it in
`backtest_forecasts`).

The "already scored?" check (the `existing_ids` query in `_store_forecast_outcomes`) does two
jobs, and they split:

- **Which outcomes to compute** reads the partitions.
- **Which rows to insert into Postgres** keeps its Postgres query, so `bulk_insert_mappings`
  never inserts a duplicate when a past partition write failed and the partitions lack a row
  Postgres has. Those rows are then written to the partitions from Postgres (Postgres wins), so
  the gap closes that night instead of re-resolving every night.

The backtest's own writes also survive a read-only Postgres, which #114 did not cover (only
the forecast, run-record, migration, drift-alert and shadow writers): on SQLSTATE 25006 the
outcome insert rolls back and writes the partitions from the rows in hand with ids from
`READ_ONLY_ID_BASE`, the verdict refresh skips its `UPDATE` but still writes the partitions, the
`prediction_accuracy` upsert skips with the single-file Parquet copy already written, and
candidate resolution's swallowed failure rolls the session back. Each skip calls
`report_read_only`. Without this, switching the readers would not keep scoring alive on a
read-only night, which is PR 1's reason to go first.

Read twice: matured-forecast count per forecast date, and the `prediction_accuracy` headline
metrics per horizon.

### PR 2: candidate resolution

`backtest/candidate_resolution.py` reads candidates, their outcomes and the production
outcomes it pairs against from the store. It is its own PR because its output feeds the frozen
champion–challenger read. The PR shows that the candidate-outcome rows it writes are identical
from both stores on one night (same rows, same values), which is the parent spec's Phase 3
condition for a storage change under a frozen read. Read twice: rows written per date.

### PR 3: freshness check

`scripts/check_forecast_freshness.py` passes when the partitions carry the expected forecast
date (it passes today only when both stores do). Postgres's newest date is still reported; if
Postgres is behind the partitions, that is a `::notice`, not a failure.

### PR 4: feedback factors and prior-day blending

**Shipped** 2026-10-30, ahead of PR 2 and PR 3: Postgres went read-only after the 10-30 forecast
write, and these are the readers that move served numbers
(`changelog/2026-10-30-feedback-and-blend-read-ops-partitions.md`). One departure from § Errors:
a missing `item_forecasts` store skips the blend with a `::error` annotation instead of failing
the run, which would serve nothing.

Both change served numbers, so both carry the parent spec's proof: the PR description quotes
each value from both stores on the same night, equal to `1e-9`.

- `models/served_recalibration._load_panel` reads the store. The `band_multiplier` probe becomes
  a check of the view's columns. Read twice: factor per horizon, in the Price Forecast log
  every night.
- The prior-day forecast read in `models/forecaster.py` (the `FROM item_forecasts` query that
  blends yesterday's forecast into today's). Read twice: rows read per horizon and
  the sum of their `price_mid`.
- `ItemForecaster.update_bias_corrections_from_outcomes` reads `forecast_outcomes`.
  It is off by default since #113; it moves here so it cannot read a stale panel if turned on,
  with an equivalence test and no read twice.

`_load_panel`'s docstring says the ops mirrors are "stale/selected". That was the single-file
mirror (`ops/forecast_outcomes.parquet`), not the partitions; the line is rewritten.

The drift check (`check_concept_drift`) is **not** a Phase 2 reader: it reads
`prediction_accuracy`, not the forecast record. The parent's inventory row "Feedback factors,
drift" is corrected to say so.

### PR 5: catalog

`models/serve_universe.py` (`is_backfilled`, `release_date`), the existence check in
`collectors/new_item_discovery.py`, and `ItemForecaster._fetch_item_metadata` /
`_fetch_supply_metadata` (the second already tries Parquet first) read
`items.parquet`. Read twice: the size of the
established set and of the young set.

### PR 6: API

`api/routes/items.py`, `opportunities.py`, `accuracy.py` and `market.py` (its `Item` query,
missing from the parent's inventory) lose their Postgres fallback; they already read
Parquet first. No read twice; the API is not deployed. Route tests run on a hermetic `OPS_DIR`.

## Errors

- A switched reader whose store read cannot find a table, or finds a required column all NULL,
  fails loudly. The freshness check is the exception: missing data is its verdict, reported
  as stale, not a crash.
- The feedback factors keep today's behaviour on a failed panel read (factor 1.0, full-width
  band), with the store and the error named in the log.
- No reader falls back to Postgres silently once switched.
- A read-twice difference on a writable-Postgres night blocks the next PR until explained; it
  never fails the run that found it.

## Tests

Per PR:

- The reader's existing tests run against the store through a dual-store fixture: the same rows
  written to in-memory SQLite and through `ops_partitions.upsert` into a temp `OPS_DIR`.
- One equivalence test runs the reader on those rows from both stores and asserts identical
  output.
- A `pg_deleted` row is hidden from every view that has the column.
- PR 1 also: an outcome Postgres holds but the partitions lack is written to the partitions
  from Postgres and not inserted into Postgres again; and a read-only rejection of each backtest
  write leaves that night's outcomes in the partitions with ids above `READ_ONLY_ID_BASE`.
- `ops_store` itself: deleted rows hidden, a column an older month lacks reads NULL, a table
  with no files raises, a join across all five views.
- Each PR runs the CI gate's own command locally before pushing
  (`pytest backend/tests -m "not slow" -q -n 4`). #118 passed serially and failed under xdist.
- Nothing touches prod: hermetic `OPS_DIR` (conftest `_hermetic_ops_dir`) and in-memory SQLite.

## Docs

- A dated `docs/changelog/` note per PR.
- `.claude/rules/archive-reads.md` drops "Postgres is still the read source" and lists the
  readers switched so far, updated by each PR.
- The parent spec's Phase 2 section links here.

## Done

All six PRs merged; every read-twice comparison equal for 3 consecutive nights; and a grep for
Postgres reads of the five tables finds only the Phase 3 instruments
(`measure_conformal_pid.py`, `ranking_transfer_report.py`, `centre_promotion_report.py`,
`measure_direction_reopen.py`, `date_dependence_sensitivity.py`), `ops_parity.py` /
`ops_sync_from_postgres.py`, one-off backfills (`backfill_band_multiplier.py`,
`backfill_ops_item_slug.py`) and `scripts/archive/`.

## Not in scope

- The frozen instruments (Phase 3).
- Removing Postgres writes, `ops_parity`, `ops_sync`, migrations or `database.py`'s engine
  (Phase 4).
- `collectors/csmarketapi_backfill.py`: it reads its own local SQLite, not Postgres.
- `collection_runs`, `prediction_accuracy`, `accuracy_alerts` and the other non-ops tables,
  which includes the drift check (the parent spec's "Run records" and later work).
