# The catalog reads, and new-item discovery writes, the ops partitions

**Date:** 2026-10-30
**Spec:** `specs/2026-10-13-ops-readers-switch-design.md`, PR 5 (amended 2026-10-30)

Postgres went read-only on 2026-10-30. New-item discovery's `INSERT INTO items` would have
failed every night (caught and logged, so the collection still succeeded), and `ops/items.parquet`,
copied from Postgres, would have frozen with it. No new release would have entered the catalog,
and each night's rediscovery would have pushed its `release_date`, and so its 60-day serve gate,
one day later.

## Readers

- `models/serve_universe.served_items(as_of, read=None)` reads `items.parquet` by default. Its
  four callers (predict universe, prior-row blend slug map, young-release set, forecast writer)
  no longer pass a session.
- `ItemForecaster._fetch_item_metadata` and the fallback in `_fetch_supply_metadata` read
  `items.parquet`.
- Read twice in `scripts/forecast_prices.py`: the established and young set sizes.

## Discovery write

- The Aggregator checks out `cs2-oracle-data` before the aggregate step and passes
  `AGGREGATOR_OPS_DIR`. `price-archive` is not linked, so nothing else in that step starts
  writing to the archive.
- The known set is Postgres plus `items.parquet`, so an item only the partitions hold is not
  rediscovered and does not count against `MAX_NEW_PER_RUN`.
- Postgres first. When it cannot take the insert (25006 or 53100), the rows go to
  `items.parquet` only, with ids from `max(id, READ_ONLY_ID_BASE) + 1` and a `::warning`. The
  nightly items sync upserts by id, so it keeps them.
- Accepted: a partition-only item never reaches Postgres.

## Proof

On the 10-30 partitions (data `77fbc28`), `served_items(2026-10-30)` gives **5,542 established
and 372 young**, from 32,285 catalog rows. The 10-30 Price Forecast run (40992224883) read
5,542 and 372 from Postgres. Identical.
