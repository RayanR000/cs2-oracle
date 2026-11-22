# The Aggregator's catalog comes from items.parquet (Phase 4, PR 2b)

2026-11-22. Part of `specs/2026-11-09-delete-postgres-design.md`, plan step "PR 2b". Stacked on
PR 2a (`2026-11-22-run-records-and-fallback-deleted.md`).

## What changed

- **The catalog is `ops/items.parquet`.** `DataPipeline` read its item list, slug map and
  `is_backfilled` flags from Postgres `items`. It now reads them once from `items.parquet` in the
  ops dir it is given (`AGGREGATOR_OPS_DIR`, the data checkout, in CI). No `items.parquet` there
  fails the run (`ops_store.MissingTable` → status `failed` → `run_task.py` exits 1) instead of
  collecting nothing.
- **New-item discovery writes `items.parquet` only.** Its known set is the catalog the pipeline
  already read; new rows take ids from `max(id, 2**40) + 1`, the space read-only nights already
  used. The Postgres insert, its read-only fallback and the "no ops dir → insert nothing" branch
  are gone.
- **The `aggregate` task touches no database.** `run_task.py` builds the pipeline without a
  session and no longer imports `database` at module level. Verified: a full run on real dumps
  ends with `database` absent from `sys.modules`.
- **Deleted:** the `priority` task and `run_priority_collection` (its `repositories` import no
  longer resolved, so it already raised); the legacy Supabase read in `append_to_parquet.py` and its
  `--backfilled-csv` flag (CI always passes `--snapshot-csv`, now required);
  `scripts/backfill_ops_item_slug.py` and its one-time workflow input.
- **Workflow:** `DATABASE_URL` is gone from "Run Aggregator Task" and "Append to Parquet archive";
  the "Sync item catalog to ops/items.parquet" step is deleted (it copied Postgres `items`, which
  has not changed since it went read-only on 2026-10-30). "Database Migration" stays until PR 6.
- `run_daily_local.sh` clones the data repo before the aggregate step and exports
  `AGGREGATOR_OPS_DIR` to it.

## Evidence

- **`items.parquet` holds the whole Postgres catalog.** The durable copy (32,287 rows) contains all
  32,281 rows of the 2026-10-11 backup with identical `item_id`, `name`, `type` and
  `is_backfilled`; 5,542 established items in both. The other 6 are 4 rows Postgres took between
  10-11 and 10-29 and 2 that discovery wrote only to the partitions (ids >= 2**40).
- **Gate: same snapshot CSV as main.** One real download (2026-11-22, 7 sources) replayed through
  main (catalog = SQLite seeded with the Postgres-id rows) and through this branch (catalog =
  the durable `items.parquet`): 350,586 rows from main, 350,590 from the branch. After the
  `(item_slug, day, source)` reduction `append_to_parquet` applies, nothing is only in main;
  the 4 rows only in the branch are `Souvenir Glock-18 | Ifrit Lattice (Battle-Scarred)`'s
  four Steam labels. That item was discovered on a read-only night, so Postgres never had it and
  main never priced it; the branch does, which is the point. The backfilled CSV is byte-identical.
