# The backtest reads the forecast record from the ops partitions

**Date:** 2026-10-30
**PR:** #122 (Phase 2, PR 1 of 6 — `docs/specs/2026-10-13-ops-readers-switch-design.md`)

Backtest Accuracy now reads matured forecasts, the frozen-id set, the scoring panel, the
verdict refresh and the id→slug map from the monthly partitions through `db/ops_store.py`.
Postgres is read only for comparison: the matured count per forecast date and the headline
per horizon are logged from both stores, and a difference is a `::warning`, never a failure.

What changed besides the read:

- **Partition gaps heal from Postgres.** "Already frozen?" now means frozen in the partitions.
  An outcome Postgres holds but the partitions lack is resolved again, not re-inserted, and its
  Postgres row is written to the partitions. Postgres wins.
- **The backtest survives a read-only Postgres.** #114 covered the forecast, run-record,
  migration, drift-alert and shadow writers, not the backtest. On SQLSTATE 25006 the outcome
  insert lands in the partitions with ids from `READ_ONLY_ID_BASE`, the verdict refresh still
  writes the partitions, and the `prediction_accuracy` upsert is skipped after its single-file
  Parquet copy.
- **Partition-only forecasts** (ids from a read-only night) never reach Postgres: its
  `forecast_id` is a 32-bit foreign key.
- **The verdict refresh diffs against the partitions.** If a Postgres UPDATE is lost, Postgres
  keeps the old verdict while the partitions carry the new one. Postgres is being retired, and
  `ops_parity.py` shows the delta.

Gate: merged after 3 clean parity nights from 2026-10-13. Not changed: forecasts, bands,
scoring, the frozen instruments, candidate resolution (PR 2).
