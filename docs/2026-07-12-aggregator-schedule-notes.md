# Aggregator Schedule & Data Freshness Notes — 2026-07-12

## API Refresh Behavior

- All sources fetch from `prices.csgotrader.app/*.json`
- Confirmed: API dumps refresh **once daily around ~21:40 UTC**
- Extra same-day runs return identical raw data (verified over 5 weeks)

## Schedule Rationale

| Time (UTC) | Event |
|---|---|
| ~21:40 | API refresh |
| 23:00 | Scheduled aggregator run (catches fresh dump) |

The 23:00 UTC (7 PM EDT) run is intentionally placed post-refresh to get a clean, independent daily snapshot.

## Manual Run at 16:11 UTC — What Happened

The manual run on Jul 12 ran **before** the 21:40 UTC refresh. This means:

- It pulled the **same API dump** as the Jul 11 scheduled run at 23:45 UTC
- The `day` column was correctly stamped `2026-07-12` (when the pipeline ran)
- Rolling window fields (`last_24h`, `last_7d`, `last_30d`, `last_90d` from Steam) and live market prices (`buff163 starting_at`, `csfloat price`, etc.) produce slightly different values because the windows shift with time — but the underlying dump is the same

This is not incorrect data, but it is not a fully independent daily snapshot.

## Dedup Safety

No action needed. `append_to_parquet.py` deduplicates on `(item_slug, day, source)` with `keep="last"`. Tonight's scheduled run at 23:00 UTC will stamp the same `day = 2026-07-12` and automatically **replace** the manual run's rows with the post-refresh data.

## Collecting More Often

Not useful. The dump only refreshes once daily. More runs would return the same underlying data with only rolling-window shifts — no new information.
