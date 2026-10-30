# The accuracy API reads the ops store only

**Date:** 2026-10-30
**Spec:** `specs/2026-10-13-ops-readers-switch-design.md`, PR 6 (amended 2026-10-30)

`api/routes/accuracy.py` loses every Postgres fallback; no route there takes a database.

- `/accuracy/`, `/latest`, `/headline` and `/summary` read `ops/prediction_accuracy.parquet`,
  which the backtest writes before Postgres. A missing file raises instead of serving nothing.
- `/accuracy/outcomes` and `/outcomes/stats` read the monthly `forecast_outcomes` partitions
  through `db/ops_store.py`, which hides `pg_deleted` rows. They used to read the single-file
  `ops/forecast_outcomes.parquet`, which on 2026-10-30 held 1,325,949 rows against the
  partitions' 1,386,686.

On the 10-30 store (data `77fbc28`), `/outcomes/stats` counts 1,377,239 outcomes, the same
panel the backtest scores, and `/headline` serves the 2026-10-30 evaluation at every horizon.

**Not moved:** `items.py`, `opportunities.py` and `market.py` are Postgres-native, not
Parquet-first with a fallback. They read `PriceHistory`, which no longer receives the daily
collection. They are left for Phase 4 to rewrite or retire, and they stop working when Postgres
is deleted. The API is not deployed.
