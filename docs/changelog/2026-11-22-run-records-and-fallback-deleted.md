# Run records and the price_history fallback deleted (Phase 4, PR 2a)

2026-11-22. Part of `specs/2026-11-09-delete-postgres-design.md`, plan step "PR 2a".

## What changed

- **`collection_runs` is no longer written, to either store.** The Aggregator used to insert a
  `CollectionRun` row into Postgres and then append the same row to
  `price-archive/ops/collection_runs.parquet`. Both writes are gone. The run is recorded by the
  counts `run_full_aggregator_collection` returns (which drive `run_task.py`'s zero-row guard),
  the `aggregator-logs-<run id>` artifact and the Actions run itself.
- **The historical fallback is deleted.** An item missing from the day's dumps was recovered
  from its newest Postgres `price_history` row if that row was under `FALLBACK_MAX_AGE_DAYS`
  (7) old. The table has been frozen since 2026-07-11, so every lookup was stale and the
  fallback recovered nothing (spec Q3). A missing item now counts as an error directly. The
  "Historical fallback declined N items" warning and the `fallback_items_collected` /
  `fallback_stale_declined` result keys are gone.
- **`FALLBACK_MAX_AGE_DAYS` stays**, same value and env read: the backtest's
  `MAX_WINDOW_SPAN_DAYS` and staleness's `STALE_RUN_GAP_BREAK_DAYS` derive from it. Nothing
  scored or served moves.

## Why not keep the Parquet copy

The plan said to keep the Parquet write, "already first-class". It was not durable:

- In CI it wrote to `backend/../price-archive/ops/`, a runner directory that is thrown away.
  The data checkout is `archive/`, and the publish step commits only that. The durable
  `cs2-oracle-data` copy holds 194 rows and stops at **2026-08-09**; its last four rows
  (ids 1–4, 1–2 items each, within 30 ms) look like a local test run, not production.
- It deduplicated on `id` with `IS NOT DISTINCT FROM`, so once Postgres went read-only
  (2026-10-30) and every id was NULL, each night's row would have replaced the last.

Nothing reads `collection_runs`. The user chose deleting it over making the Parquet copy
durable (2026-11-22). The 194 historical rows stay in the data repo and the
2026-10-11 backup; Postgres's own rows go with the final backup before PR 7.
