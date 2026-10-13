# Every forecast-record writer also writes the monthly ops partitions

**Date:** 2026-10-12

Phase 1, step 2 (and the parity check of step 3) of
`docs/specs/2026-10-11-postgres-to-parquet-history-design.md`, section "Phase 1.2: dual-write".
Postgres stays the read source; no reader changes.

## What changed

- `db/ops_partitions.py` is the one writer of `ops/<table>/YYYY-MM.parquet` and
  `ops/items.parquet`. `upsert` replaces rows by natural key, `replace` also removes named keys
  (`--reresolve`), and a column the write does not carry keeps its stored value, so a Postgres
  read-back never blanks `model_config`, `item_slug` or `pg_deleted`. Guards raise: a closed
  month (calendar end + 43 days) without `allow_closed`, a column outside the schema the backfill
  wrote, NULL or repeated keys, and any month whose row count after the merge is not
  `before - removed + new`. `closes_on` moved here from `scripts/ops_releases.py`.
- `db/ops_readback.py` re-selects just-committed rows by natural key, so the partitions carry the
  ids Postgres issued. It selects only the columns the live table has (prod runs behind its
  migrations).
- Writers, each **after** its unchanged Postgres commit and inside `dual_write`, which logs ERROR
  plus a `::error` annotation and never fails the run:

  | Writer | Table |
  |---|---|
  | `scripts/forecast_prices.py::_write_forecasts_to_db` | `item_forecasts` (this batch's keys only: regime and global rows share a date) |
  | `db/candidate_store.py::write_candidate_batches` | `forecast_candidates` |
  | `scripts/backtest_accuracy.py` daily outcomes and `--reresolve` | `forecast_outcomes` |
  | `scripts/backtest_accuracy.py::_flush_verdict_refresh` | `forecast_outcomes`, from the rows already in hand (see below) |
  | `backtest/candidate_resolution.py` (resolve and `--rescore`) | `forecast_candidate_outcomes`, routed by the candidate's `forecast_date` |
  | Aggregator step "Sync item catalog" (`ops_sync_from_postgres.py --tables items`) | `items.parquet`, the whole table, after the data checkout. Discovery runs before that checkout exists, so it cannot write the file itself |

- **Read-only nights.** On SQLSTATE 25006 the forecast and shadow-candidate writers still land
  the partitions. Batches that committed before the error keep the ids Postgres gave them (read
  back; reads still work), a key already stored keeps its id, and the rest are issued from
  `READ_ONLY_ID_BASE = 2**40` up, a range the Postgres sequence never reaches. Outcomes are
  recomputable from prices and get no fallback.
- **A failed partition write rolls back the writer's session** (`dual_write(..., db=db)`): a
  failed read-back leaves a Postgres transaction aborted, and the next writer on that session
  would otherwise raise `InFailedSqlTransaction` and fail the run.
- **Closed months.** A batch that touches a closed month still writes its open months, then
  raises (logged by `dual_write`). Parity reports closed-month differences as a `::warning` and in
  `n_closed_mismatches`, not as a failure: a released month can never be synced (`ops-sync`
  refuses released months even with `allow_closed`), so failing on it would stall the
  clean-night count forever.
- **The verdict refresh does not read back.** `test_the_refresh_streams_in_bounded_flushes_and_never_re_reads_a_row`
  forbids write-side SELECTs, and the refresh never changes an id, so it upserts its mirror rows
  (no `id`) and the stored id is kept.
- `scripts/ops_parity.py`, in Backtest Accuracy before the archive publish: per table and
  `forecast_date`, Postgres and the partitions compute the same summary (count, `count(id)`,
  `sum(id)`, value-column sums). Postgres returns only the aggregates. Counts and id sums must be
  exact, float sums within 1e-9 relative; `pg_deleted` rows are excluded. Each night appends to
  `ops/parity_runs.parquet` and logs the clean-night streak. It also fails on any id repeated
  across a table, which no per-date aggregate can see. The step is `continue-on-error`, so
  the closed-month release still runs; "Fail on parity mismatch" turns the run red at the end.
- `scripts/ops_sync_from_postgres.py` via `ops-sync.yml` (dispatch only; dry run unless `apply`):
  copies named dates from Postgres into the partitions. It fills ids and repairs a failed night,
  and writes only rows Postgres holds.
- `tests/conftest.py` gains an autouse `_hermetic_ops_dir`: writer tests that never redirected
  `OPS_DIR` had started writing partition files into the local `price-archive/ops`.

## The id-less rows

The backfill wrote 44,550 forecasts with no id. 22,507 are dated 10-11: written to Postgres at
02:32 on 10-12, after the backfill's dry run, so they reached the partitions from the single-file
mirror only. Postgres has their ids, and `ops-sync.yml` with `dates=2026-10-11` fills them. The
other 22,043 are `pg_deleted` rows of 08-04 and 08-05 whose id no outcome names. No id exists to
recover; they stay NULL, and parity excludes them.

The gap is wider than 10-11. The 10-12 night's Backtest wrote outcomes for forecast dates 09-11,
09-27, 10-04 and 10-08 (5,536–5,899 each) after the backup was taken. They reached the partitions
through the mirror, which carries no outcome `id`. That night's shadow candidates (dated 10-11) and
candidate outcomes are absent outright, because candidates were never mirrored. Closed and
closing months are unaffected: August's newest `resolved_at` is 09-27 and its only id-less rows
are `pg_deleted`, so the August release on 10-13 freezes a complete month.

## Next

1. The first chain after merge writes partitions; its parity run fails and names every
   `(table, forecast_date)` that differs (expected: the dates above, plus the candidate tables).
2. Dispatch `ops-sync.yml` with exactly those dates, dry run, then `apply`. Syncing only the named
   dates keeps the Postgres egress to a few hundred thousand rows.
3. The next clean parity run is night one of seven; Phase 2 starts after seven in a row.
