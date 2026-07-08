# Session Notes — Full System Overhaul (2026-07-07 → 07-08)

Everything done in this session, in order. Companion docs:
`DB_SCHEMA_FIX_RECOMMENDATIONS.md`, `PLAN_STEAM_PRICE_BASIS_SWAP.md`.

## 1. Database schema fix (589 MB → 392 MB)

- Reviewed `SESSION_NOTES_DB_STRATEGY.md` / `DB_CLEANUP_AND_MIGRATION_PLAN.md`;
  concluded the problem was write patterns, not hosting.
- **Migration 0006**: dropped the surrogate `id` bigint PK on `price_history`
  (no FK referenced it) and promoted `(item_id, timestamp, source)` to primary
  key. Freed the 80 MB pkey index; the fresh PK index came out 138 MB vs the
  old bloated 256 MB unique index.
- `alembic upgrade` hit **DiskFull** (single transaction builds the new index
  before dropped-index space frees) — re-applied manually in separate
  autocommit transactions, then `alembic stamp 0006`. `VACUUM FULL` also
  didn't fit (needs ~370 MB temp); plain `VACUUM ANALYZE` ran instead.
- Updated all `PriceHistory.id` call sites (order-by tie-breaks → created_at/
  source; counts → count(item_id)). Guarded migration 0005 to skip missing
  tables. Verified `ON CONFLICT (item_id, timestamp, source)` still works.

## 2. Collection pipeline: found and fixed a silent 5-day outage

- Collection had written nothing since **Jul 2** while GitHub Actions showed
  green: pipeline methods return status dicts, `run_task.py` exited 0, and
  the failure recorder itself crashed (error text overflowed
  `error_message VARCHAR(1000)`; no rollback before insert).
- Fixes: `run_task.py` exits non-zero on `status: failed`; error messages
  truncated to 1000 chars; session rolled back before recording failures.
- Weekly downsampler (`prune_database.py`) had two landmines: its DELETE
  used the removed `price_history.id`, and it had **no source filter** — the
  next Sunday run would have collapsed the imported market_csgo daily
  history to weekly/monthly. Rewrote it on the composite key and scoped it
  to live snapshot sources only (verified: 845K protected rows excluded).

## 3. Collection schedule: 4×/day → 1×/day at 23:00 UTC

- Measured across 117,990 same-day run pairs: prices+volumes were identical
  **100.0%** of the time (day-over-day: 57% change). CSGOTrader's dumps
  regenerate once daily ~21:40 UTC (verified via Last-Modified headers +
  5 weeks of data; no official docs exist).
- New cron `0 23 * * *`: fresher data than any old run, quarter the writes.
- Deleted the existing **117,990 duplicate intraday rows** (kept latest per
  item/day/source).

## 4. Two-tier storage, then backfilled-only catalog

- Only 5,525 of 19,321 items had CSMarketAPI historical data; 12,328 had
  live-only snapshots that can't support analysis.
- First: archived their 247,203 price rows to
  `runtime/archived_live_prices_2026-07-08.db` + deleted from Supabase;
  collector became **two-tier** (historical items accumulate daily history,
  others keep one replaced snapshot; auto-promotion when monthly backfill
  imports an item).
- **Daily git archive**: aggregator workflow now dumps each day's collected
  prices to `data-archive` branch (`price-archive/YYYY/MM/*.csv.gz`,
  ~300 KB/day) — durable record independent of Supabase/laptop. Pre-Jul-8
  snapshot history also backed up there (1.7 MB gz).
- Later (user decision): **catalog reduced to backfilled items only** —
  13,796 items archived to `runtime/archived_items_2026-07-08.db` and
  deleted with their snapshot rows + 20,418 degenerate forecasts. Weekly
  Steam `discover-new-items` schedule disabled (would repopulate); new
  items enter only via monthly CSMarketAPI backfill runs.

## 5. Storage audit across all three stores

- Local `.git` was **1.3 GB** from an 8.3 GB DB blob committed once in May
  (never pushed; GitHub repo is 6 MB) — pruned to **6.6 MB**.
- Deleted stubs/debris: 0-byte `cs2_market.db` (symptom: something ran
  without DATABASE_URL on Jul 7), empty `ssr_history.db`, ~5 MB old logs.
- Kept: `csmarketapi.db` (4.2 GB master archive — catalog knows **31,417
  items**, so the 5.5K backfilled so far is progress, not a ceiling),
  `market_catalog.db`, reference DB, archive exports.

## 6. Analysis pipelines: first-ever successful runs

- All six analysis tables existed but were **empty** — every prior run had
  failed (forecast: `daily_analysis does not exist`; others silently).
- Accuracy fixes: source filters (synthetic/fallback exclusion) added to
  `long_term_trend_analyzer` and `event_analyzer` (incl. its top-1000
  ranking); forecaster collapses prices to daily in SQL; validation split
  now relative (last 21 days) instead of hardcoded 2026-06-15; prediction
  requires ≥14 days of history; `item_forecasts` pruned to 45 days (was
  unbounded ~31K rows/day).
- Scheduling: forecast now **chains off Daily Trend Analysis completion**
  (`workflow_run`) instead of racing it on a delayed cron. Created the
  0005 indexes that were skipped while tables were missing.
- Verified live: trends, forecast (auto-chained), event correlation, and
  long-term trends all ran green and populated; next morning's *scheduled*
  cycle also succeeded unattended.

## 7. Frontend fixes (verified with browser screenshots)

- **Item pages 404'd** for ids with spaces/pipes: `useParams()` returns the
  encoded segment and `api.ts` encoded again (double-encoding). Now decoded
  once.
- **Price chart was empty for every item since it was built**: it requested
  sources `steam,csfloat`, which never existed in `price_history`. The
  `/prices` endpoint now defaults to all real sources (excluding
  synthetic/fallback) and the chart renders one line per source.
- **Full historical depth**: `/prices` capped `days` at 365, hiding
  2013–2021 and most of 2024–2025; raised to 5000 — "ALL" shows the whole
  series.
- Listing endpoints filter to backfilled items (now moot after deletion,
  kept as safety net).

## 8. API performance + caching

- `market/summary` rebuilt the full 9,974-group summary (~3.5 s) per
  request **and ignored skip/limit** (returned everything; every page of
  the market table showed the same items). Now: in-process TTL cache
  (`api/cache.py`, replacing an unused broken decorator), paginated from
  memory — warm responses **3.6 s → ~2 ms**.
- TTL caching on items list / trending / opportunities; cache warming on
  startup; Cache-Control middleware fixed (GET+200 only, covers
  /opportunities, `max-age=300, stale-while-revalidate=600`).

## 9. Documented for later

- `PLAN_STEAM_PRICE_BASIS_SWAP.md`: fix for the ~13.5% price step between
  market_csgo (market.csgo.com basis) and aggregator_sync (Steam basis) —
  swap the serving series to the local STEAMCOMMUNITY data + refresh the
  backfill to close the Apr–May 2026 gap. Runs with the monthly key budget.

## End state (2026-07-08)

- **Supabase 436 MB / 500 MB**: 5,525 backfilled items; price_history 2.35M
  historical rows + growing live Steam-basis series; all analysis tables
  populated and pruned on schedule; alembic at 0006.
- **Schedules**: collect 23:00 UTC daily (+ git archive) → trends 03:00 →
  forecast chained → weekly event/long-term/prune Sundays (all with real
  failure reporting). Discovery disabled.
- **Growth**: ~5.5K price rows/day (~0.7 MB), bounded analysis tables.
- **Open items**: Steam price-basis swap (planned), monthly backfill run
  (adds items + closes gap), `ma_30/90` fill in as live series matures,
  trending ranking still uses `updated_at` (cosmetic).
