# Phase 2: move the forecast-record readers off Postgres

**Date:** 2026-10-13
**Status:** draft, awaiting review
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

`connect(root=None) -> duckdb.DuckDBPyConnection` returns an in-memory DuckDB connection with
five views, named after the Postgres tables:

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
  and an empty feedback panel snaps the factor to 1.0 without an error.
- **Root** defaults to `parquet.OPS_DIR`, the writers' root. Price Forecast, Backtest Accuracy
  and Forecast Freshness Check already check out all of `cs2-oracle-data` and link
  `price-archive`, so no workflow changes. Closed months stay in the tree beside their
  `ops-YYYY-MM` releases, so history reads see every month.

### Read twice

`compare(name, pg_value, parquet_value, *, tol=0.0)` logs both values, and a `::warning`
naming the reader, the date and both values when they differ beyond `tol`. It never fails a
run. A switched reader always uses the Parquet value; Postgres is shown for comparison only.
On a night Postgres was read-only, Postgres is the stale store, so a difference is expected
and the message says so. `compare` and its Postgres reads are removed in Phase 4.

## PR sequence and gates

**Before PR 1:** nightly parity (`scripts/ops_parity.py` in Backtest Accuracy) passes 3
consecutive nights for every table, counting from the night of 2026-10-12. **Between PRs:**
each PR waits for its predecessor to go through one green nightly chain with no unexplained
read-twice difference.

**If Postgres goes read-only mid-sequence:** switched readers keep working; unswitched readers
serve a stale panel and say so. The remaining PRs then go in order without the 3-night wait,
because there is no longer a live Postgres to stay equal to.

### PR 1: `ops_store`, backtest scoring and verdict refresh

`scripts/backtest_accuracy.py` reads from the store: the matured forecasts (the
`FROM item_forecasts f` query, `:1177` on 7302ac0), the scoring panel (`:689`), the verdict
refresh (`_refresh_verdict_columns`) and the id→slug map (`:339`).

The "already scored?" check (`:577`) does two jobs, and they split:

- **Which outcomes to compute** reads the partitions.
- **Which rows to insert into Postgres** keeps its Postgres query, so `bulk_insert_mappings`
  never inserts a duplicate when a past partition write failed and the partitions lack a row
  Postgres has.

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

Both change served numbers, so both carry the parent spec's proof: the PR description quotes
each value from both stores on the same night, equal to `1e-9`.

- `models/served_recalibration._load_panel` reads the store. The `band_multiplier` probe becomes
  a check of the view's columns. Read twice: factor per horizon, in the Price Forecast log
  every night.
- The prior-day forecast read in `models/forecaster.py` (`FROM item_forecasts`, `:9453` on
  7302ac0) that blends yesterday's forecast into today's. Read twice: rows read per horizon and
  the sum of their `price_mid`.
- `ItemForecaster.update_bias_corrections_from_outcomes` (`:2042`) reads `forecast_outcomes`.
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
`_fetch_supply_metadata` (`:3176`, `:3530`; the second already tries Parquet first) read
`items.parquet`. Read twice: the size of the
established set and of the young set.

### PR 6: API

`api/routes/items.py`, `opportunities.py`, `accuracy.py` and `market.py` (its `Item` query,
`:58`, missing from the parent's inventory) lose their Postgres fallback; they already read
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
- PR 1 also: a forecast Postgres holds but the partitions lack is scored for the partitions and
  not inserted into Postgres again.
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
