# Move off Supabase: every table to Parquet

**Date:** 2026-10-11
**Status:** draft, awaiting review
**Source:** session of 2026-10-11 (migration 0030 review → Supabase storage check)
**Ships in:** Phase 0 now (`changelog/2026-10-11-survive-read-only-postgres.md`); Phases 1–4 as
separate PRs.

## Problem

Prod Supabase is on the **free tier (500 MB database cap)** and measured **986 MB** on
2026-10-11 (read-only `pg_database_size` / `pg_total_relation_size`):

| Table | Heap | Indexes | Total | Rows | Rows added, last 7 days |
|---|---|---|---|---|---|
| `item_forecasts` | 169 MB | 136 MB | 305 MB | 1,332,853 (60 forecast dates, 2025-12-01 → 2026-10-10) | 157,533 |
| `forecast_candidates` | 182 MB | 121 MB | 303 MB | 974,336 (23 dates from 09-08) | 310,016 |
| `forecast_outcomes` | 155 MB | 83 MB | 238 MB | 955,064 | 152,304 |
| `forecast_candidate_outcomes` | 35 MB | 15 MB | 49 MB | 470,560 | |
| `items` | 19 MB | 9 MB | 28 MB | 32,281 | |
| event tables (3) | | | ~40 MB | | |
| everything else | | | ~10 MB | | |

Growth is **~170 MB a week**: each table's rows added in the last 7 days × its average row size,
so an estimate. Past the cap Supabase documents read-only mode
([database-size docs](https://supabase.com/docs/guides/platform/database-size)). It was not
enforced on the day.

**Trimming does not fix it.** Everything safe to delete (forecasts and outcomes before 08-24,
which the PID priors need; three unused or duplicate indexes; the event tables) comes to
~310 MB, leaving ~675 MB, and the growth refills that within weeks. The rest is the shadow
tables, which the champion–challenger read needs into November. Paying for Pro was declined.

## Decision: no database

Nothing needs a live database. Every reader is a batch job in GitHub Actions or a local
script. The API is not deployed and there is no frontend. A future frontend can add a small
store of the latest forecasts when it exists. The hybrid in the first draft of this spec
(latest forecasts plus `items` kept in Postgres) is dropped.

Why Parquet fits:

- **One writer.** Aggregator → Forecast → Backtest run in sequence, and data-repo pushes are
  leased (#74).
- **It is ~20× smaller.** The 2026-10-11 backup holds all 18 tables in **56 MB** of ZSTD Parquet
  against 986 MB in Postgres. At the current row rate that is roughly **30–40 MB a month**
  (estimate).
- **One source of truth.** Today only Postgres holds the full scored panel, and the
  two-store split cost nine published figures on 2026-08-11 (`backend/AGENTS.md`).
- **The mirror exists.** `db/parquet.py` has `append_table`, `replace_rows` and atomic writes,
  the forecast and backtest steps already write it, and `outside_baseline.py` already reads it.

## Target layout (`cs2-oracle-data`)

```
price-archive/ops/
  item_forecasts/YYYY-MM.parquet               (by forecast_date)
  forecast_outcomes/YYYY-MM.parquet            (by forecast_date)
  forecast_candidates/YYYY-MM.parquet
  forecast_candidate_outcomes/YYYY-MM.parquet  (by the candidate's forecast_date)
  items.parquet                                (catalog + is_backfilled + release_date)
  accuracy_alerts.parquet, prediction_accuracy.parquet, collection_runs.parquet,
  events.parquet, event_impacts_denorm.parquet (unchanged)
```

Monthly files keep each one far under GitHub's 100 MB file limit and match `append_monthly`.

**Surrogate ids.** `item_forecasts.id`, `forecast_candidates.id` and `items.id` come from
Postgres sequences, and outcomes key on them. The writers must keep issuing them, as
`max(id) + 1` over the partitions under the push lease, so no existing key changes and the
frozen instruments join the same way.

### Mutability

The verdict refresh rewrites rows that are already scored (`backtest_accuracy.py`,
`replace_rows`). With Parquet that is a rewrite of the affected month. A row matures at most
h + 13 days after its forecast date (h ≤ 30), so only the current and previous two months are
written. **Older months are closed:** written once, never rewritten, and checksummed.

### Durability: the cost of losing the database

The data repo is a single force-pushed orphan commit, so a bad write replaces the only copy.
That is how archive day 09-19 was lost. For the forecast record:

1. **Closed months are immutable.** The writer refuses to touch one without an explicit flag.
2. **Second copy: GitHub Release assets on `cs2-oracle-data`** (decided 2026-10-11). See
   "Second and third copies" below.
3. **Row-count guard.** Before the push, each writer asserts that no partition lost rows against
   the checked-out copy, the same pattern as `run_task.py`'s zero-row guard.

### Second and third copies

**When a month closes.** Month M is closed once every forecast in it has matured: its
calendar end + 30 + 13 days (the h=30 horizon plus the embargo carry). Calendar end, not the
last forecast date, so a row added late cannot reopen a month. September 2026 closes
on 11-12. Nothing writes a closed month again.

**Release per closed month.** A final step in Backtest Accuracy (the last job in the chain,
after outcomes are written) does, for each closed month:

- **No release yet:** create tag `ops-YYYY-MM` on `cs2-oracle-data` with that month's file from
  each partitioned table plus a `SHA256SUMS`, then publish it.
- **Release exists:** compare the tree's files with its `SHA256SUMS`. A mismatch **fails the
  run**: a closed month was rewritten, which is the bad-write case this exists to catch. The
  step never replaces an asset.

So the release is both the backup and the immutability check. Restore one month with
`gh release download ops-YYYY-MM -R RayanR000/cs2-oracle-data`.

**Why Release assets.** They are free, need no new account, and live outside the git history,
so an orphan force-push cannot touch them. Creating a release needs `contents: write`, the
permission the data-repo pushes already use. **To verify in Phase 1:** that
`CS2_DATA_REPO_TOKEN` carries it for releases, and whether the repository setting for
immutable releases is available. If it is, enable it so published assets cannot be edited or
deleted.

**Known weakness.** The release lives on the same GitHub account, and the same token can
delete it. Separate storage (for example Cloudflare R2's free tier) would remove that, at the
cost of a new account, a new secret and new upload code. Not worth it for this project: the
real risk is a bad write, not a compromised account.

**Third copy, off GitHub.** Once a month, the local Mac pulls every `ops-*` release it doesn't
have into `~/personal-projects/cs2-analyzer/ops-releases/` and checks `SHA256SUMS`. It can run
as a launchd job or by hand. It is the one copy GitHub cannot delete, and it costs nothing.

**Phase 1 seeds it.** The backup of 2026-10-11 holds forecasts for 2025-12 (one date), 2026-07
(five) and 2026-08 (twenty; closes 10-13), plus open September and October. Those months are released as part of the Phase 1 backfill, and
the first local pull runs right after.

## What has to change

Inventory from grep on 2026-10-11 (excluding `tests/`, `scripts/archive/` and migrations):

| Area | Files | Change |
|---|---|---|
| Migrations | Aggregator step `run_task.py migrate` (runs `alembic upgrade head` **every night**), `migrations/`, `schema-drift-check.yml` | Removed in Phase 4 |
| Run records | `collectors/pipeline.py` (`collection_runs`) | Parquet only (already mirrored) |
| Catalog | `collectors/new_item_discovery.py` (writes), `models/serve_universe.py` (reads `is_backfilled`, `release_date`) | `items.parquet` |
| Forecast write | `scripts/forecast_prices.py` | Partitions only; issue ids |
| Shadow write | `db/candidate_store.py` | Partitions only |
| Outcome scoring and refresh | `scripts/backtest_accuracy.py`, `backtest/candidate_resolution.py` | Read and write partitions; refresh rewrites open months |
| Feedback factors, blending | `models/served_recalibration.py`, `models/forecaster.py` (prior-day blend, bias corrections, item metadata; the drift check reads `prediction_accuracy`, not this record) | DuckDB over partitions |
| Freshness check | `scripts/check_forecast_freshness.py` | Newest partition |
| Instruments | `measure_conformal_pid.py`, `ranking_transfer_report.py`, `centre_promotion_report.py`, `measure_direction_reopen.py`, `date_dependence_sensitivity.py` | Phase 3 rule below |
| API | `api/routes/items.py`, `opportunities.py`, `accuracy.py` | Parquet only (already its first read) |
| Events | `scripts/event_correlation_analysis.py` (workflow `disabled_manually`) | Parquet only |
| CI | `.github/workflows/*.yml` | Sparse-checkout the partitions a run needs; drop `SUPABASE_DATABASE_URL` |

## Phases

### Phase 0: done 2026-10-11

- Full local backup of every table (18 tables, 56 MB; counts, non-null counts and sums verified).
- The chain survives a read-only Postgres: the Parquet forecast mirror is written first, and
  migrations, run records, drift alerts and shadow writes skip loudly
  (`changelog/2026-10-11-survive-read-only-postgres.md`).

### Phase 1: backfill and dual-write

1. Write the backup into the target partitions in the data repo, so history the mirror never
   had becomes durable. The local mirror of 09-20 has `item_forecasts` only from 08-04 (31 dates
   against Postgres's 60), and the shadow tables and `items` were never mirrored.
   Release the closed months (2025-12, 2026-07, then 2026-08 from 10-13), add the release
   step to Backtest Accuracy, and run the first local pull. **Built 2026-10-12:**
   `scripts/ops_releases.py` (`publish` in Backtest Accuracy, `pull` on the Mac, by hand).
   **Seed built 2026-10-11** (`ops-partition-backfill.yml`, from release
   `supabase-backup-2026-10-11`). The mirror turned out not to be a copy: Postgres wins every
   shared row, the mirror adds only what Postgres lacks, and rows Postgres deleted are kept
   with `pg_deleted` (`changelog/2026-10-11-ops-partition-backfill.md`). Dual-write must
   compare against Postgres, not the single-file mirror.
2. Every writer writes both stores. Postgres stays the read source. Design below
   ("Phase 1.2: dual-write"). **Built 2026-10-12** (`changelog/2026-10-12-ops-dual-write.md`).
3. A nightly parity check (rows per date and table, and the verdict-refresh deltas) runs for
   ≥7 nights. Design below.

#### Phase 1.2: dual-write (designed 2026-10-12)

**Starting state.** Postgres is still writable: on 10-12 it took 22,507 forecasts (dated
10-11). Those are the difference between the backfill's dry run (22,043 id-less forecasts) and
its applied run (44,550). Measured on the applied partitions: the 22,507 dated 10-11 have an id
in Postgres, recoverable by `(item_id, forecast_date, horizon_days)`; the other 22,043 are
`pg_deleted` rows of 08-04 and 08-05 whose id no outcome names, so no id exists to recover and
they stay NULL. Parity excludes `pg_deleted` rows, so they do not count against it.

**Ids come from Postgres, read back after the commit.** Each writer keeps its Postgres write
exactly as it is, commits, then re-selects the ids it just wrote by natural key and writes the
partitions with them. Changing the inserts to `RETURNING` would touch the Postgres paths the
frozen reads depend on; the read-back is one night's keys per table. Rejected alternatives:
a nightly export from Postgres (makes parity circular, none of it survives into Phase 2, and
re-reading ~1.2M open-month rows twice a night exhausts the free tier's 5 GB/month egress), and
partition-issued ids written into Postgres explicitly (moves the Postgres sequences and collides
with the 10-11 id-less rows until they are filled).

**Read-only fallback.** On SQLSTATE 25006 (`db/read_only.py`), the writer issues ids as
`max(id) + 1` over the partitions, under the archive push lease the workflow already holds, and
still writes the partitions. The single-file mirror stays written first, unchanged: it remains
the copy that survives a crash before either store lands.

**One module: `db/ops_partitions.py`.**

- `upsert(table, rows, key)`: routes each row to `ops/<table>/YYYY-MM.parquet` by
  `forecast_date` (candidate outcomes by their candidate's `forecast_date`), replaces rows with
  the same key, and writes each month to a temp file swapped in with `os.replace`.
- `replace(table, key_col, keys, rows)`: the delete-and-reinsert the verdict refresh and
  `--reresolve` need, applied to the months the keys fall in.
- Guards, all raising: a closed month (`ops_releases.closes_on`) is refused without an explicit
  flag; a month never loses rows except through `replace` with named keys; a column the
  backfill's schema does not have fails instead of being dropped.
- The keys, sort orders and schema the backfill wrote live here, and `ops_releases.closes_on`
  moves here; the one-off backfill keeps its own (already-run) write path.
- A column the write does not carry keeps its stored value (`model_config`, `pg_deleted`), so a
  Postgres read-back never blanks a mirror-only column.

**Writers.**

| Writer | After its Postgres commit |
|---|---|
| `scripts/forecast_prices.py` | read back ids by `(item_id, forecast_date, horizon_days)`; `upsert` `item_forecasts` |
| `db/candidate_store.py` | read back ids by the candidate identity; `upsert` `forecast_candidates` |
| `scripts/backtest_accuracy.py`, daily outcomes | read back outcome ids by `forecast_id`; `upsert` `forecast_outcomes` |
| `scripts/backtest_accuracy.py`, verdict refresh and `--reresolve` | after the DB commit, read back by `forecast_id` and `upsert` / `replace` in the affected months; the single-file mirror keeps its mirror-first order |
| `backtest/candidate_resolution.py` | `upsert` `forecast_candidate_outcomes` |
| `collectors/new_item_discovery.py` | none: it runs inside the Aggregator task, before the data checkout. The Aggregator syncs `items.parquet` from Postgres after the checkout (amended 2026-10-12 at review) |

Not touched: readers, the single-file mirror, frozen instruments, and the workflows (their
archive pushes already include `price-archive/ops/`).

**A failed partition write does not fail the run.** It logs ERROR and the run continues: by
then Postgres and the single-file mirror hold the night, so nothing is lost, and failing Price
Forecast after its commit only endangers the Backtest chain. The parity check fails that night
instead.

**Fill and repair: `scripts/ops_sync_from_postgres.py --dates …`.** Copies the named dates from
Postgres into the partitions, ids included, and adds no row Postgres does not hold. Dry run
unless `--apply`; run only by dispatching `ops-sync.yml` (only CI writes the data repo), never scheduled. Its first use fills the dates the first parity run names: the 22,507 forecasts of 10-11, outcomes the 10-12 night wrote through the mirror (09-11, 09-27, 10-04, 10-08), and that night's shadow candidates and candidate outcomes.

**Parity: `scripts/ops_parity.py`, in Backtest Accuracy before the archive publish** (so its
record is pushed), with `continue-on-error`; a last step fails the run on a mismatch, so the
closed-month release still runs.

- Per table and `forecast_date`, both stores compute the same summary: row count, `count(id)`,
  `sum(id)`, and sums of the value columns (forecasts: `price_low`, `price_mid`, `price_high`,
  `current_price`, direction counts; outcomes: `actual_price`, `base_price`, `abs_error`,
  `pct_error`, counts of `direction_correct` and `in_interval`; candidates: `centre_price`,
  `predicted_price_low`, `predicted_price_high`, `score`; candidate outcomes, which have no
  `forecast_date` of their own and are grouped by their candidate's: `count(candidate_id)`,
  `sum(candidate_id)`, `actual_price`, `absolute_error`, `percentage_error`, `in_interval` count).
- Postgres returns only the aggregate rows (a few KB a night). The partitions run the same query
  in DuckDB, excluding `pg_deleted` rows.
- Pass: counts equal exactly; sums within 1e-9 relative (the float8 noise measured on 10-11).
  The verdict refresh's rewrites show up as verdict-sum deltas.
- A mismatch fails the run and names the table, date and summary that differ.
- Each night appends to `ops/parity_runs.parquet`, so "≥7 clean nights in a row" is counted,
  not remembered. Night one is the first run of the merged code after the fill is applied.

**Tests (written first).**

- `ops_partitions` on temp directories: month routing, closed-month refusal, the no-row-loss
  guard, unknown-column failure, and an interrupted write leaving the old file intact.
- Each writer against the SQLite fixture: partition rows equal the DB rows, ids included.
- Read-only fallback, using #114's 25006 simulation: ids come from the partitions and none
  collide.
- Parity: identical stores pass; a missing row, a wrong id and a changed verdict each fail with
  the right message.
- Sync: the dry run writes nothing; `--apply` fills ids and adds no row Postgres lacks.

### Phase 2: switch readers that no frozen read depends on

Designed 2026-10-13: `docs/specs/2026-10-13-ops-readers-switch-design.md` (six PRs, risk order).

The API, freshness check, drift check, backtest scoring and refresh, catalog and feedback
factors move one PR at a time. The feedback factors change served bands, so their PR must show
the same factor value from both stores on the same night.

### Phase 3: frozen instruments

Each preregistered instrument moves only after **either** its read is done, **or** its input
panel is shown identical from both stores (same rows, same values, same order where the
statistic depends on order). A storage swap that feeds a frozen read the same panel does not
change the read. Each switch gets a dated note beside its prereg; the prereg itself is not
edited. Reads in flight: PID ~10-23, h=7 naive-quantile band ~10-25, direction gate ~10-29 /
~11-02, champion–challenger at 20 shared dates.

### Phase 4: delete the project

Stop the dual-write, remove the migration step, Alembic, `database.py`'s engine and the
schema-drift workflow, and delete the Supabase project after a final backup.

## Not in scope

- Any change to forecasts, bands or scoring. This is storage only.
- Migration 0030. The Aggregator applies it automatically on the first run after #111; its net
  size change is small (drops a 47 MB duplicate index and adds two narrower ones).

## Open questions

1. ~~Second copy: GitHub Release assets or object storage.~~ **Decided 2026-10-11: Release
   assets, plus a monthly local pull** ("Second and third copies").
2. Should `forecast_candidates` keep full history once the champion–challenger read resolves,
   or only the shared-date window the gate uses?
3. ~~`outside_baseline.py` (item 22, read ~10-25) reads the Parquet mirror today. Confirm the
   durable mirror covers h=7 forecast dates from 09-27 before that read.~~ **Confirmed
   2026-10-11:** every h=7 date 09-20 to 10-10 holds 5,536 rows in both stores, with
   identical bands and direction (float8 noise <=5e-16 elsewhere). It keeps reading the
   single-file mirror until after its read.
