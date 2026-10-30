# The freshness check reads the ops partitions

**Date:** 2026-10-30
**Spec:** `specs/2026-10-13-ops-readers-switch-design.md`, PR 3

Postgres went read-only on 2026-10-30, and its newest forecast date stays at 2026-10-29.
`scripts/check_forecast_freshness.py` passed only when **both** stores carried the expected
date, so from the 10-31 run it would have failed every day on forecasts that landed.

- **Verdict:** the `item_forecasts` partitions (`db/ops_store.py`) carry the expected forecast
  date or later. Missing partitions are a stale verdict, not a crash.
- **Postgres:** its newest date is still logged. Behind the partitions, or unreadable
  (read-only, DiskFull, gone), is a `::notice`, never a failure. A current Postgres does not
  rescue stale partitions.
- **Shadow collection readiness** (warn-only) reads the production dates and candidate batches
  from the `item_forecasts` and `forecast_candidates` partitions too, so it does not go stale
  with Postgres. Not in the spec's PR 3 text; same reader, same rule.

Checked on the 10-30 partitions (data `77fbc28`): newest forecast date 2026-10-29, shadow
collection ready.
