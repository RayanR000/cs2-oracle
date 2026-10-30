# The feedback factors and the prior-day blend read the ops partitions

**Date:** 2026-10-30
**PR:** Phase 2, PR 4 of 6 — `docs/specs/2026-10-13-ops-readers-switch-design.md`

Three readers that change served numbers now read the monthly partitions through
`db/ops_store.py` instead of Postgres:

- **The served-coverage feedback panel** (`models/served_recalibration._load_panel`). The factor
  per horizon is logged from both stores (`read twice served-coverage factor per horizon`,
  tolerance 1e-9). A failed partition read still returns no factor (full-width band) and now
  names the store and the error.
- **The prior-day blend** (`ItemForecaster._fetch_prior_forecasts`). Rows read and their summed
  `price_mid` are logged from both stores per horizon (`read twice prior-day forecasts h=N`).
  Postgres's side is an aggregate query, so it is not asked for the whole forecast history twice.
- **The tier bias fit** (`update_bias_corrections_from_outcomes`). Off by default since #113;
  moved so it cannot fit a stale panel if turned on. Equivalence test, no read twice.

**Why this one went before PR 2 and PR 3.** Postgres went read-only after the 10-30 02:36 UTC
forecast write. From the 11-01 run, yesterday's forecast exists only in the partitions (ids from
`READ_ONLY_ID_BASE`), so the blend would have blended a two-day-old forecast, and the feedback
panel already lacked every outcome scored since. Both move served numbers; PR 2 and PR 3 do not.
The spec's "remaining PRs go in order without the 3-night wait" applies.

**The blend's missing-store behaviour departs from the spec's "fails loudly".** If the
partitions have no `item_forecasts` files, the blend logs an error, prints
`::error title=Prior-day blend::`, and serves unblended. Crashing would have served nothing.

Gone: the `band_multiplier` column probes in both readers. The view always projects the column
(NULL in a month that lacks it), and prod Postgres has migration 0028.

The `_load_panel` docstring's "the ops parquet mirrors are stale/selected" was about the
single-file `ops/forecast_outcomes.parquet`, not the partitions.

Not changed: the slug→id map the blend uses (`serve_universe.served_items`, PR 5), the drift
check (it reads `prediction_accuracy`), candidate resolution (PR 2), freshness (PR 3).
