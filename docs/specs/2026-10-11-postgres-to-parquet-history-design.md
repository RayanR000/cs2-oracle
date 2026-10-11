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

**When a month closes.** Month M is closed once every forecast in it has matured: its last
forecast date + 30 + 13 days (the h=30 horizon plus the embargo carry). September 2026 closes
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

**Phase 1 seeds it.** The backup of 2026-10-11 covers December 2025 through August 2026, all
of which are already closed. Those months are released as part of the Phase 1 backfill, and
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
| Feedback factors, drift | `models/served_recalibration.py`, `models/forecaster.py` (live panel read, `accuracy_alerts`) | DuckDB over partitions |
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
   Release the already-closed months (December 2025 through August 2026), add the release
   step to Backtest Accuracy, and run the first local pull.
2. Every writer writes both stores. Postgres stays the read source.
3. A nightly parity check (rows per date and table, and the verdict-refresh deltas) runs for
   ≥7 nights.

### Phase 2: switch readers that no frozen read depends on

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
3. `outside_baseline.py` (item 22, read ~10-25) reads the Parquet mirror today. Confirm the
   durable mirror covers h=7 forecast dates from 09-27 before that read, independent of this
   migration.
