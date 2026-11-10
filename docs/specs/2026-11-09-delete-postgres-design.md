# Phase 4: delete Postgres

**Date:** 2026-11-09 · **Status:** approved 2026-11-09 (Q1 and Q4 decided as recommended; plan: `docs/plans/2026-11-09-delete-postgres.md`)
**Parent:** `2026-10-11-postgres-to-parquet-history-design.md` (Phase 4) · **Follows:**
`2026-10-13-ops-readers-switch-design.md` (Phase 2, done) and Phase 3 (frozen instruments, PR #133)

## Problem

Supabase has been read-only or out of disk since 2026-10-30. Phase 2 moved every reader of the
forecast record, the candidates, the catalog and the accuracy table onto the monthly ops
partitions. Phase 3 moved the instruments. What is left is plumbing that still opens a Postgres
connection, mostly to write a copy nobody reads, and each such write is a place a night can fail
on a dead database.

Nothing here changes a forecast, a band or a score. Storage and plumbing only.

## What still touches Postgres (grep, 2026-11-09, excluding tests, archive and migrations)

| Area | Where | Today | Phase 4 |
|---|---|---|---|
| Forecast write | `scripts/forecast_prices.py` (7 `SessionLocal`) | Writes Postgres, reads ids back, dual-writes partitions | Partitions only. Ids from `ops_partitions.assign_ids` (the read-only path already does this). |
| Shadow write | `db/candidate_store.py` | Postgres then partitions | Partitions only |
| Id read-back, parity | `db/ops_readback.py`, `scripts/ops_parity.py`, `ops_sync_from_postgres.py`, `backfill_*` | Exist to reconcile the two stores | Delete (nothing to reconcile) |
| Run records | `collectors/pipeline.py` (`CollectionRun`, a `PriceHistory` count at ~L653), `run_task.py` | Postgres insert, then `collection_runs.parquet` | Parquet only. The `PriceHistory` read is a stale check, see Q3. |
| Forecaster events | `models/forecaster.py` `fetch_events` (L2746), called at L5880 and L9804 | Reads the `events` table, builds a frame no served feature uses (Q2, answered) | Return an empty frame; delete the reconnect fallback |
| Forecaster other DB reads | `forecaster.py` L2325 (`price_history`, legacy path), L6374 and L8735 (`served_coverage_factors(self.db, ...)`), L9488 (`pg_reader` read-twice), L11218-11321 (`AccuracyAlert` write) | Postgres sessions inside the model class | Not yet audited: PR 4 starts by listing which are live and which are Postgres-only fallbacks |
| Events | `sync_events_from_news.py`, `event_correlation_analysis.py` (workflow disabled) | Postgres | Retire or Parquet, see Q2 |
| Replay | `scripts/replay_serving.py` (model-diagnostics) | Postgres | `ops_store` reader |
| Migrations | `run_task.py migrate`, `migrations/`, `alembic.ini`, `check_schema_drift.py`, `schema-drift-check.yml` | Nightly `alembic upgrade head` | Delete |
| Engine | `database.py` (engine, `SessionLocal`, ORM models), `main.py` startup | Builds an engine at import | See "ORM models" below |
| API | `main.py`; routes `items`, `opportunities`, `market`, `auth`, `events` | `accuracy` is on the ops store; the rest read `PriceHistory` (frozen since 2026-07-11), `users` or `events` | Retire, see Q1 |
| CI | 7 workflows carry `SUPABASE_DATABASE_URL` (17 steps) | | Drop the secret; `lint.yml` keeps its SQLite gate until tests no longer need the ORM |

## Design

### 1. Order: remove writes before readers of writes, then the engine

Each PR is independently mergeable and leaves nightly green:

1. **Forecast + shadow writes → partitions only.** The riskiest PR: it is the nightly forecast.
   Ids from `assign_ids` (partition space ≥ 2^40 for new rows, existing ids kept). Drop the
   read-back and the Postgres insert; keep the 25006/53100 handling only until PR 6 deletes it.
   Gate: the same night's forecast, written both ways in a dry run, is row-identical on every column
   except `id`.
2. **Run records, discovery and the aggregator steps → Parquet only.** `collection_runs`, the
   `items` sync step, `backfill_ops_item_slug`, `append_to_parquet`'s `DATABASE_URL`.
3. **Parity, sync and reconcile scripts deleted** with their workflow (`ops-sync.yml`) and the
   "Ops parity" step. Closes the failing evidence-report path.
4. **Events and the remaining scripts** per Q2; `replay_serving.py` to `ops_store`.
5. **API retired** per Q1.
6. **Delete the engine:** `run_task.py migrate` and the "Database Migration" step, Alembic,
   `migrations/`, `check_schema_drift.py`, `schema-drift-check.yml`, `SessionLocal`, the 25006/53100
   helpers (`db/read_only.py`), and every `SUPABASE_DATABASE_URL` reference.
7. **Final backup, then delete the Supabase project** (human step, see below).

### 2. ORM models

`database.py`'s models double as the test fixtures' schema (tests seed SQLite through the ORM and
`tests/_ops_mirror.py` mirrors it to partitions). Deleting them means rewriting about 40 test
files. Decision: **keep the models as a test-only schema module**, rename `database.py` →
`db/models_orm.py` is *not* done here. The engine, `SessionLocal`, `get_db` and the production
connection string go; `Base` and the table classes stay until a separate cleanup. Smaller blast
radius, and nothing in production imports them.

### 3. Durability replaces the database

The parent spec's durability plan (partitions in the data repo, release assets, monthly local pull)
is the only copy after this. Before PR 7:

- A fresh full backup of everything Postgres still holds that the partitions do not (users, supply,
  social, event tables, `price_history`): release `supabase-final-<date>` on the data repo, with a
  `SHA256SUMS`, restore-tested into a scratch SQLite.
- The partitions are confirmed complete for every table a reader needs (row counts by month against
  Postgres for the months Postgres still has, i.e. through 2026-10-29).

### 4. Rollback

PRs 1–6 are ordinary reverts; Postgres still exists until PR 7, so a revert restores the old writes
(from a read-only database, the dual-write degrades the way it does today). PR 7 is the only
irreversible step, and it waits at least 7 nightly cycles after PR 6.

## Open questions (need an answer before the plan)

1. ~~**API.**~~ **Decided 2026-11-09: retire everything.** Retire `main.py` and all routes (the API is not deployed, there is no frontend), or keep
   `accuracy` on the ops store and drop the rest? Recommended: retire everything; `accuracy` has no
   consumer. `auth.py` and the `users` table go with it.
2. ~~**Events.**~~ **Answered 2026-11-09: no served feature uses them.** `FEATURE_GROUP_ALLOWLIST` is
   `["price_technicals"]`, so `_skipped_feature_groups()` includes `events` and
   `_add_event_features` never runs on the production frame. `fetch_events()` is still called at
   L5880 (training) and L9804 (serving), and its frame only reaches `engineer_features`, where it is
   ignored. Making `fetch_events` return an empty frame cannot move a band. Caveat: the 14d horizon's
   `HORIZON_EXCLUDED_GROUPS` comment says events help 14d (+2.0pp), but the global allowlist overrides
   it, so that is history. Re-admitting the group would need the events table back; that is a
   modelling decision, not a Phase 4 one. The `sync_events_from_news.py` and
   `event_correlation_analysis.py` scripts feed nothing served and are retired with the API.
3. ~~**`PriceHistory` stale check in `pipeline.py`.**~~ **Answered 2026-11-09: it is a dead fallback.**
   `_load_latest_prices` and `_load_latest_non_aggregator_prices` recover an item missing from the
   day's scrape from `price_history`, but only if its newest row is within `FALLBACK_MAX_AGE_DAYS`
   (7). The table has been frozen since 2026-07-11, so every lookup is stale and recovers nothing; the
   item lands in `still_missing_names` and is counted as an error exactly as if the fallback were
   absent. Delete both loaders and the fallback block. The only visible change is the "declined N
   items" warning going away.
4. ~~**Supabase deletion.**~~ **Decided 2026-11-09: pause after PR 6, delete after the final backup verifies.** Delete the project, or pause it and let it lapse? Pausing is reversible for
   a while and costs nothing; deleting is not. Recommended: pause after PR 6, delete at the monthly
   pull after the final backup verifies.

## Not in scope

- Any change to forecasts, bands, scoring or a preregistered instrument.
- Removing the ORM models (see 2).
- A replacement API or frontend.
