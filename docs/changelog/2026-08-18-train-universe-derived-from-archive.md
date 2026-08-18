# Train universe is derived from the archive, not an `is_trainable` DB column

**Date:** 2026-08-18

## Problem

`is_trainable` was added to the ORM (`database.py`) and the fresh-DB init
(`scripts/init_local_db.py::populate_items`) but **no Alembic migration ever
added it** — migration head is `0021`, and `grep -r is_trainable migrations/`
is empty. The managed Supabase Postgres therefore has no such column.

`_resolve_backfilled_slugs("train")` read `SELECT item_id FROM items WHERE
is_trainable = 1`, which throws `UndefinedColumn` on that DB and fell back to a
raw `read_parquet('prices-*.parquet')` with **no `day < '2026-01-01'` predicate
and no universe filter** — returning all **41,885** archive slugs / ~19.7M rows.
That OOMed (exit 143) every cold train-universe resolution: `model-diagnostics`,
the recency/`train_universe` ab-harnesses, and any cold retrain. Production's
weekly retrain was masked by its warm voted-frame/model cache.

## Fix

Derive the train universe from the Parquet archive (the data repo), matching
`populate_items`' derivation exactly and honoring the invariant that training
data comes from Parquet, not the DB (`backend/AGENTS.md`):

- **train** = pre-2026 backfill (`day < '2026-01-01'`) minus iflow-only history
  (`source IS DISTINCT FROM 'buff_iflow'`), read through
  `db/archive.py::prices_relation` + `archive_universe_sql_filter()` so
  phase-collapsed and phantom keys never enter the cohort. **No DB read.**
- **serve** = `is_backfilled` from the DB (unchanged — the DB's one job here),
  with the same archive derivation (iflow included) as a fallback.

New helper `ItemForecaster._archive_universe_slugs(exclude_iflow)`. Measured on
the current archive: both universes resolve to **5,536** items (iflow is still
staging-only, so the sets coincide today). This removes the Supabase
schema-drift dependency for the train universe entirely.

## Tests

`tests/test_universe_routing.py` updated: train now routes to the archive helper
and must not touch the DB; serve still reads `is_backfilled`. 12 passed.
