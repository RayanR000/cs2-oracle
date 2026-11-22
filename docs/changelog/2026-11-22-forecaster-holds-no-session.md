# The forecaster holds no database session (Phase 4, PR 4)

2026-11-22. Part of `specs/2026-11-09-delete-postgres-design.md`, plan step "PR 4". Stacked on
PR 3 (`2026-11-22-parity-and-sync-deleted.md`).

## Audit: every use of `self.db` in `models/forecaster.py`

| Use | Live or dead | Now |
|---|---|---|
| `fetch_price_history` Postgres branch (`price_history`) | Dead: taken only with no archive or `days_back <= 14`; no caller asks for that, and the table froze 2026-07-11 | Raises, naming the archive path. Before, a missing archive silently read frozen Postgres |
| `fetch_events` (`events` + reconnect fallback) | Read, but nothing served uses it: `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` skips the events group (spec Q2) | Returns an empty frame |
| `served_coverage_factors(self.db, ...)` at train and in the predict-time refresh | Live, reads the partitions since Phase 2 PR 4; the session was the on/off switch and the read-twice side | `serving_reads` is the switch; read-twice removed |
| `_fetch_prior_forecasts` guard + `pg_reader` read-twice | Live (the prior-day blend); same switch | `serving_reads`; read-twice removed |
| `check_concept_drift` (`prediction_accuracy` read, `AccuracyAlert` write, `_commit_alert`) | Effectively dead: Postgres froze 2026-10-30, so no rows in the window and it returned None every night | Reads `ops/prediction_accuracy.parquet`; drift is a `::warning` annotation and a log line |

## Decisions

- **`serving_reads` replaces `db_session`.** The session did two jobs: it was a handle, and its
  None-ness switched off the prior-day blend and the served-coverage feedback (harnesses pass None
  so their bands are not pulled toward what was served). The flag keeps the second job, default
  off. Every caller was mapped to what it did before: a session or a mock became
  `serving_reads=True`, `None` became the default. Production (`forecast_prices`, `replay_serving`)
  sets it.
- **No `accuracy_alerts` writes** (user's call, 2026-11-22, as with `collection_runs`): nothing
  read the table, its Parquet copy stopped in July with 13 alerts never resolved.
- `replay_serving.py` opens no session; the model-diagnostics replay step drops `DATABASE_URL`.
  `forecast_prices.py` keeps its own session for the in-run backtest (`backtest_forecasts(db)`)
  and the served-universe read-twice until the backtest moves; the Price Forecast workflow keeps
  `DATABASE_URL` for that.

## Gate: identical forecasts

`predict()` on main and on this branch, same local archive, same model artifacts, and the
durable `item_forecasts` / `forecast_outcomes` / `items` partitions as the ops dir so the blend
and the coverage feedback both run (factors 3d 0.538, 7d 0.600, 14d 0.642, 30d 0.667 on both).
Main read 177 real events from the 2026-10-11 backup; the branch read none and never imported
`database`. 5,903 items, every column identical (the nested per-horizon bands compared exactly),
except `generated_at`, the run's own timestamp.
