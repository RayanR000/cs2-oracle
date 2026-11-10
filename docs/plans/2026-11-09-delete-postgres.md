# Plan: delete Postgres (Phase 4)

**Spec:** `docs/specs/2026-11-09-delete-postgres-design.md` (approved 2026-11-09)
**Rule for every PR:** one worktree, one PR, `pytest -n 4` as its own step before commit, one
commit per logical step, no attribution trailers. Merge order is the numbering. Nightly must be
green for **2 nights** between PR 1 and PR 2a, and between 2a and 2b, and **7 nights** between PR 6 and PR 7.

**Before PR 1:** #133, #134, #135 and #136 merged.

## PR 1: forecast and shadow writes go to the partitions only (the risky one)

Files: `scripts/forecast_prices.py`, `db/candidate_store.py`, `db/ops_readback.py` (delete),
`db/ops_partitions.py` (`assign_ids`), tests.

1. **Dry-run harness first.** A script that runs one night's forecast write twice into temp
   partition roots, once through today's path (Postgres stubbed from a SQLite fixture, ids read
   back) and once through the new one, then diffs every column except `id`. This is the merge gate.
2. Replace the insert-then-read-back in `forecast_prices.py` with `assign_ids("item_forecasts", ...)`
   (existing keys keep their id; new ones take `max(id, 2**40) + 1`). Remove the 25006/53100 branches
   there: with no Postgres write, no code path raises them.
3. Same for `candidate_store.py` (candidates and candidate outcomes).
4. Delete `db/ops_readback.py` and its tests; remove the id read-back from callers.
5. Gate: harness diff is empty on the last 3 nights' inputs; `test_ops_dual_write_*` rewritten to
   assert partitions-only; full suite.
6. First production night after merge: check `item_forecasts/<month>.parquet` row count equals the
   forecast count, ids unique, no id below the old max reused.

## PR 2a: run records and the dead fallback

Files: `collectors/pipeline.py`, `scripts/run_task.py`, tests.

1. `collection_runs`: drop the `CollectionRun` insert (success and failure paths, L506-530 and L597-612)
   and keep the Parquet write, which is already first-class. Remove the read-only branch there.
2. Delete `_load_latest_prices`, `_load_latest_non_aggregator_prices`, `_latest_per_item`,
   `_split_fresh_and_stale` and the fallback block (spec Q3). Drop `FALLBACK_MAX_AGE_DAYS`.
3. Tests: the pipeline tests that assert a `CollectionRun` row now assert the Parquet row.

## PR 2b: the aggregator's catalog comes from `items.parquet` (found 2026-11-09, not in the spec)

The pipeline still reads its item list from Postgres: `db_session.query(Item)` (L113) and
`query(Item.id, Item.item_id, Item.is_backfilled)` (L275), and `discover_new_items(session, ...)` still
inserts new releases into Postgres `items` (L180) and builds its known set from both stores. A dead
Postgres would stop the Aggregator, the head of the whole chain, so this is the real blocker for PR 6.

1. Load the catalog from `items.parquet` through `ops_store` / `serve_universe.served_items`-style
   reader (id, slug, `is_backfilled`). The Aggregator workflow already checks the data repo out before
   aggregating (`AGGREGATOR_OPS_DIR`, Phase 2 PR 5), so the file is present.
2. `discover_new_items`: known set from `items.parquet` only; new items get ids from `assign_ids`
   (>= 2**40, as on a read-only night) and are written to `items.parquet` only. Delete the Postgres insert.
3. `ItemRepository.get_top_items` (the `limit=` branch, used by the unscheduled `priority` task): the
   `repositories` module no longer exists, so that branch already raises. Delete it and the
   `priority` task.
4. `run_task.py`: build `DataPipeline` without a session.
5. Workflow: remove `DATABASE_URL` from "Run Aggregator Task", "Append to Parquet archive",
   "Sync item catalog to ops/items.parquet" (delete the step if it only syncs from Postgres) and the
   one-time slug backfill step (delete it and `scripts/backfill_ops_item_slug.py`). Keep
   "Database Migration" until PR 6.
6. Gate: a local Aggregator run against the scratch partitions (`--no-db` style, no `DATABASE_URL`)
   produces the same snapshot CSV as main for the same input.

## PR 3: parity, sync and reconcile deleted

Delete `scripts/ops_parity.py`, `ops_sync_from_postgres.py`, `ops-sync.yml`, the "Ops parity" step in
`backtest-accuracy.yml` and its `parity_runs.parquet` push, `tests/test_ops_parity.py`,
`tests/test_ops_sync_from_postgres.py`. Also `evidence-report` and `maturity-gate` steps' `DATABASE_URL`
(they read only the partitions now: confirm by running each against the scratch clone first).

## PR 4: the forecaster stops holding a session

1. **Audit, as a table in the PR body:** every use of `self.db` in `models/forecaster.py`
   (L2325 legacy `price_history` read, L6374/L8735 `served_coverage_factors(self.db, ...)`, L9488
   `pg_reader` read-twice, L11218-11321 drift check + `AccuracyAlert` write) marked live or
   fallback, with the partitions equivalent (`prediction_accuracy.parquet`, `accuracy_alerts.parquet`).
2. `fetch_events` returns an empty frame (spec Q2); delete the reconnect fallback. Test: a frame
   built with and without it is identical for the production allowlist.
3. Drift check reads `prediction_accuracy` through `ops_store`/Parquet; alerts append to
   `accuracy_alerts.parquet`. Delete `_commit_alert` and `db/read_only.py` uses here.
4. Remove `db_session` from `ItemForecaster.__init__`; every caller passes none.
5. Gate: a `--predict-only` run on the scratch partitions produces byte-identical `item_forecasts`
   rows to main for the same input, every column except `id` and `created_at`.
6. `replay_serving.py` and the model-diagnostics steps move to `ops_store` readers; drop their
   `DATABASE_URL`.

## PR 5: retire the API

Delete `main.py`, `api/routes/*`, `api/serving_policy.py` (check importers first), `api/` schemas,
the `users`/auth code, `sync_events_from_news.py`, `event_correlation_analysis.py`, and their tests.
Keep `serving_policy.MIN_SERVED_PRICE_USD` if `outside_baseline.py` or the backtest import it: move
the constant to `backtest/scoring.py` instead of deleting it. Add a changelog note: "no API until the
frontend rebuild".

## PR 6: delete the engine

1. Remove `run_task.py migrate`, `run_migrations`, the "Database Migration" step.
2. Delete `migrations/`, `alembic.ini`, `check_schema_drift.py`, `schema-drift-check.yml`.
3. `database.py`: delete the engine, `SessionLocal`, `get_db`, the production URL handling and
   `config.py`'s `DATABASE_URL`. Keep `Base` and the table classes as the test-only schema (spec 2);
   rename nothing.
4. Delete `db/read_only.py` and any remaining 25006/53100 handling.
5. Remove `SUPABASE_DATABASE_URL` from every workflow; `lint.yml` keeps its SQLite gate.
6. `grep -rn "SessionLocal\|DATABASE_URL\|SUPABASE" backend .github` must come back empty outside
   `tests/`, `docs/` and `database.py`'s ORM classes.
7. Update `AGENTS.md`/`CLAUDE.md` (the "backend/.env points at production" gotcha, the deploy-model
   paragraph) and `docs/architecture/`.

## Between PR 6 and PR 7 (human steps)

1. Pause the Supabase project (dashboard; the user does this).
2. Final backup of what the partitions do not hold (users, supply, social, event tables,
   `price_history`, `collection_runs` before the Parquet mirror): `supabase-final-<date>` release on the
   data repo with `SHA256SUMS`, restore-tested into scratch SQLite. Needs the project un-paused for the
   dump; do it before pausing.
3. Row counts per month, partitions vs Postgres, for every table a reader needed, through 2026-10-29.
4. 7 green nights.

## PR 7: record and close

A changelog entry (`2026-MM-DD-postgres-retired.md`) and the CLAUDE.md edits; the Supabase project is
deleted by the user at the monthly pull after the backup verifies. Not a code PR.

## Risks

- **PR 1 is the nightly forecast.** The dry-run harness is the gate, not the suite. A bad id scheme
  silently orphans outcomes from forecasts.
- **Hidden `SessionLocal` imports** (the `database.py` engine builds at import): PR 6 step 6's grep is
  the check; also `python -c "import main"` is gone, so run each script's `--help` in CI smoke.
- **Workflow secrets removed before the code stops needing them** fail a night. Remove a secret only in
  the PR that deletes its last reader.
