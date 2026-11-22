# The backtest writes the partitions only (Phase 4, PR 5b)

2026-11-22. Part of `specs/2026-11-09-delete-postgres-design.md`; not in the plan, added as
"PR 5b" (`plans/2026-11-09-delete-postgres.md`). Found while finishing PR 5: PR 4 left
`forecast_prices.py` holding a session for the in-run backtest, and no plan PR covered
`backtest_accuracy.py`.

## Audit

| Where | Before | Now |
|---|---|---|
| `_upsert_accuracy` | `prediction_accuracy.parquet`, then a Postgres upsert that failed read-only | Parquet only |
| `_store_forecast_outcomes`, daily path | Postgres insert, then the 25006 fallback to the partitions with ids from `READ_ONLY_ID_BASE` | The fallback is the path. The freeze check reads the partitions instead of Postgres |
| `_store_forecast_outcomes`, "Postgres holds it, the partitions lack it" heal | Postgres won and was copied over | Deleted: the partitions are the record since 2026-10-30 |
| `--reresolve` | Postgres delete + insert first, which a read-only Postgres rejected — **broken since 10-30** | Mirror `replace_rows`, then `ops_partitions.replace`; stored ids kept |
| `_flush_verdict_refresh` | Postgres UPDATE (rejected), then the partitions | Mirror, then the partitions |
| `_read_twice_matured`, `_read_twice_headlines`, candidate `_read_twice_per_date`, forecast's served-universe read-twice | Logged the partitions' value beside Postgres's | Deleted, with `ops_store.compare` and `serve_universe.universe_sizes` |
| `candidate_resolution._write_candidate_outcomes` | Postgres, fallback to the partitions on 25006/53100 | Partitions only |
| `db/ops_readback.py` | Read committed rows back from Postgres for the dual-write | Deleted |
| `run_backtest`, `forecast_prices.run_forecast` | Each opened a `SessionLocal` | No session; `DATABASE_URL` removed from Backtest Accuracy, Price Forecast's forecasting step and the model-diagnostics CV step |

## Gate

The 11-22 data snapshot (data commit `17eb899`) with that night's resolutions removed — 22,516
outcomes, 44,288 candidate outcomes, 37 accuracy rows — run once on main, with every
INSERT/UPDATE/DELETE rejected the way the read-only Supabase rejects it, and once on the branch.
All four outputs (outcome partitions, candidate outcome partitions, `prediction_accuracy.parquet`,
the single-file `forecast_outcomes.parquet`) are identical row for row, ids included, apart from
the run's own timestamps. Main reproduced the night's real counts; the branch ran 0 SQL statements.

## Decisions

- Partition writes in the backtest still log and continue (`dual_write`) instead of failing the
  run: an outcome that does not land stays unresolved and the next night resolves it again. The
  forecast writers raise, as before (`db/candidate_store.py`).
- The single-file `ops/forecast_outcomes.parquet` mirror is still written. It is 60k rows short of
  the partitions and nothing reads it, but dropping it is a separate change.
- Tests: conftest's `ops_mirror` copies the test's SQLite into the partitions before every read;
  `tests/_ops_mirror.write_back` now copies each partition write back into the session, so the
  existing ORM assertions read what reached the partitions.
