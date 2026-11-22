# The Postgres engine is deleted (Phase 4, PR 6)

2026-11-22. Part of `specs/2026-11-09-delete-postgres-design.md`, plan step "PR 6". Follows
PR 5b (`feat/backtest-no-postgres`, #144), after which nothing in the nightly chain opened a
session.

## Deleted

- `run_task.py migrate`, `run_migrations`, `_postgres_read_only`, and the Aggregator's
  "Database Migration" step.
- `backend/migrations/` (the Alembic chain), `alembic.ini`, `scripts/check_schema_drift.py`,
  `schema-drift-check.yml`.
- `database.py`'s engine, `SessionLocal`, `get_db`, `init_db`; `config.database_url`,
  `config.secret_key` and `check_secret_key`. `database.py` keeps `Base` and the table classes
  as the test-only SQL schema.
- `db/read_only.py` and its tests (25006 / 53100 handling: nothing writes Postgres now).
- `ops_store.pg_reader`, `measure_conformal_pid --prove-equal`, the freshness check's Postgres
  comparison (`_db_newest`, `postgres_note`), `scripts/backfill_band_multiplier.py`, and the
  tests whose only job was to prove the partitions equal Postgres.
- `DATABASE_URL` / `SUPABASE_DATABASE_URL` from every workflow; the `lint.yml` throwaway
  SQLite env guard (nothing reads it).
- Dependencies `alembic`, `psycopg2-binary`, `fastapi`, `uvicorn`, `itsdangerous`; `uv.lock`
  regenerated.
- 34 finished-experiment scripts in `scripts/archive/` that imported `SessionLocal` at module
  level, and the 26 test files (plus 8 tests in mixed files) that imported them. They could
  not run without Postgres. The scripts stay in git history; their results are in
  `docs/changelog/` and `experiment_log`.

## Checks

- Plan step 6 grep (`SessionLocal|DATABASE_URL|SUPABASE|get_db|alembic|25006|53100`) over
  `backend/` and `.github/` outside `tests/`: empty.
- Suite 2260 passed, ruff and `ruff format --check` clean, `mypy models backtest` clean.

## Next

PR 7 is not code: take the final backup of what the partitions do not hold, pause Supabase,
seven green nights, delete the project at the monthly pull.
