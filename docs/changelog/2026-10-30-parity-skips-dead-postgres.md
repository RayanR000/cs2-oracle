# Parity skips a dead Postgres

**Date:** 2026-10-30

Supabase ran out of disk on 2026-10-29 and went read-only on 10-30. Since #122 the backtest
scores from the ops partitions and lands its outcomes there when Postgres refuses the insert, so
the partitions are the current store and Postgres falls further behind every night.

`scripts/ops_parity.py` now checks `transaction_read_only` first, and treats a `DiskFull`
(SQLSTATE 53100) on its summary query the same way: it logs a `::warning title=Ops parity
skipped::`, records no night in `ops/parity_runs.parquet` (a skipped night is not a clean one),
and exits 0. Any other error still fails. Run 41004689748 is the night that showed it: scoring
succeeded, and parity crashed on `No space left on device`.
