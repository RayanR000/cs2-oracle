# Parity and sync deleted; the maturity gate reads the partitions (Phase 4, PR 3)

2026-11-22. Part of `specs/2026-11-09-delete-postgres-design.md`, plan step "PR 3". Stacked on
PR 2b (`2026-11-22-aggregator-catalog-from-items-parquet.md`).

## Deleted

- `scripts/ops_parity.py`, the "Ops parity" and "Fail on parity mismatch" steps in
  `backtest-accuracy.yml`, and `tests/test_ops_parity.py`. Parity compared Postgres with the
  partitions per table and forecast date. Postgres has taken no forecast-record write since
  2026-10-30, and since #123 the step skipped itself every night, so it had nothing left to compare.
  `ops/parity_runs.parquet` stays in the data repo as history; nothing writes it now.
- `scripts/ops_sync_from_postgres.py`, `ops-sync.yml` and `tests/test_ops_sync_from_postgres.py`.
  Sync copied named dates from Postgres into the partitions. With Postgres frozen there is nothing
  newer to copy (PR 2b already removed its last caller, the nightly items sync).

## The maturity gate moved, and was not what the plan said

The plan said the evidence-report and maturity-gate steps "read only the partitions now". The
evidence reports did (they had no `DATABASE_URL`). The maturity gate did not:
`scripts/archive/centre_vs_lastprice.py --gate` read Postgres `forecast_outcomes`, so since
2026-10-30 it had been counting a frozen panel. Its `--archive-dir` option read the single-file
mirror the `archive-reads` rule says never to use for a panel.

It now reads the `forecast_outcomes` monthly partitions through `db/ops_store.py`
(`--archive-dir` points at another price-archive's partitions). Proof against the 2026-10-11
Postgres backup, on the gate's own filter (served cohort, resolved, base >= $1):

- all 160,230 backup rows are in the partitions, 0 with any differing gate column;
- the partitions hold 35,533 more rows for forecast dates the backup covers, all h=7/14/30
  outcomes resolved from 2026-10-12 on (after the backup);
- in all 306,252 rows over 98 forecast dates to 2026-11-18.

On that panel the gate exits 0: **every horizon has >= 20 clean dates, so the GBM-vs-last-price
verdict is now quotable** (h=3 skill −0.064 [−0.073, −0.056]; h=7 −0.010 [−0.026, +0.005];
h=14 +0.016 [−0.002, +0.032]; h=30 −0.031 [−0.068, 0.000]). This PR changes only where the gate
reads; the decision it opens (`docs/research/2026-08-25-gbm-retirement-decision.md`) is separate.
