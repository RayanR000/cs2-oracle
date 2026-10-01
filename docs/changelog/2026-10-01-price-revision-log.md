# Price revision log: next-steps item 21, decided

**Date:** 2026-10-01

## Decision

The data repo stays flat (one orphan commit, force-pushed with a lease). History is not
retained. Instead each Aggregator run logs the price rows it changed, removed or back-filled
into `ops/price_revisions.parquet`.

## Why not keep commits

The 08-25 anchor audit could not tell whether a day's prices had been revised, because no
earlier version of any day survives
(`2026-08-25-centre-shrinks-to-zero-and-the-dollar-band-is-the-wedge.md`, "What follows" 3).
Retaining history costs too much: Parquet does not delta-compress, and the files rewritten
each run are `prices-<month>` (25-45 MB), `supply-<month>` (17-26 MB), `item_forecasts`
(13 MB) and `forecast_outcomes` (11 MB). That is roughly 70-100 MB a day (estimated from file
sizes in a 09-20 local copy, not measured as a real daily delta) against a 489 MB repo.

## What ships

- `scripts/log_price_revisions.py` diffs the monthly `prices-YYYY-MM.parquet` files as checked
  out against the same files after the run's writers, over the trailing 14 days
  (`--window-days all` for a backfill).
- Kinds: `changed` (price or volume differs), `removed` (key gone, which is what a
  stale-checkout publish does), `late_added` (a new key for a day already published; the
  routine new-day rows are not logged).
- A run that would log more than 500,000 rows logs one aggregate row per (day, source, kind)
  with `item_slug` NULL and `n_items` set.
- Re-running a `run_date` replaces its own rows.
- `aggregator-update.yml` gains "Snapshot prices before writes" (before the append) and
  "Log price revisions" (before the publish, `continue-on-error`, so a failure here cannot take
  down the Price Forecast chain).

## Scope and limits

- Prices only. `supply-*`, `volume-*` and the forecast tables are not diffed.
- A revision made by a writer outside the Aggregator (a manual restore or backfill script run
  against a checkout) is invisible unless that run also snapshots.
- Nothing before 2026-10-01 is in the log, and it cannot be reconstructed.
- Checked on a scratch copy of the local archive (5 injected revisions, all found, about 2 s);
  not yet run in CI.
