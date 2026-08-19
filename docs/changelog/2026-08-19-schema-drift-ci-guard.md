# CI guard: ORM vs migrations schema-drift check

`Item.is_trainable` was added to the ORM (`database.py`) and merged with no Alembic
migration (deep-model-review §2). Prod `items` never got the column, the aggregator
crashed on its first query, and the nightly chain skipped silently for days — review,
tests, and merge all passed it through. Migration `0023` fixed the instance; this guard
stops the class from recurring.

## What

- `backend/scripts/check_schema_drift.py` — applies to a DB already at `alembic upgrade
  head`, then runs `compare_metadata(ctx, Base.metadata)` (the same diff
  `alembic revision --autogenerate` runs).
- `.github/workflows/schema-drift-check.yml` — Postgres 16 service → `alembic upgrade
  head` → the check. Fires on PRs/pushes touching `database.py` or `migrations/`. It is a
  **pre-merge gate**, which is where `is_trainable` slipped through.

## Fail vs warn

Only structural presence drift fails the build — `add_table`, `add_column`,
`remove_column` — which is the missing-migration signature. Type / default / nullable /
index diffs are warnings: Postgres reflection compares them noisily (server defaults,
varchar length, index naming) and gating CI on them would flake. `SCHEMA_DRIFT_STRICT=1`
escalates every diff to a failure.

## Validation

The check must run against Postgres — the migrations use PG-only DDL (`ALTER COLUMN ...
TYPE`) that SQLite cannot execute, so the end-to-end path is validated in CI, not locally.
The drift-detection logic was validated locally against SQLite: a schema built from
`Base.metadata` reports no drift; an empty DB reports every table/column as structural
drift and exits non-zero; a partially-migrated DB flags `add_column: items.is_trainable` —
the exact defect this guards against.
