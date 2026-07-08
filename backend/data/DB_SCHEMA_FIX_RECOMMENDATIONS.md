# Database Fix Recommendations (2026-07-07)

Follow-up to `SESSION_NOTES_DB_STRATEGY.md` and `DB_CLEANUP_AND_MIGRATION_PLAN.md`.

## Core observation: the problem is write patterns, not hosting

The strategy notes frame the 500 MB overage as a hosting decision ("which database do
we buy/migrate to"), but two facts point at architecture instead:

1. **The analysis jobs don't use Postgres as a database.** `analyze_trends.py` and
   `forecast_prices.py` pull 1.3M–3M rows into pandas and do all work app-side.
   Supabase functions as a dumb archive that gets bulk-read and bulk-written.
2. **The full history already exists locally.** `runtime/csmarketapi.db` (4.2 GB,
   2013–2026, 7 markets) is the source of truth; the Supabase copy is a lossy subset.

The projected ~4.8 GB/year growth is mostly self-inflicted:
- ~7.4M rows/year is **recomputable derived data** (analysis tables — the same tables
  already dropped once in Phase 0 to free 194 MB; appending them forever recreates
  that problem monthly).
- ~10.9M rows/year is historical backfill for new items, which belongs in the local
  archive, not the serving DB.

## Recommended fixes

### 1. Short term — schema fixes, no data loss (IMPLEMENTED — see migration 0006)

`price_history` was 568 MB: 232 MB heap + 336 MB of indexes (256 MB unique index +
80 MB pkey). The surrogate `id` bigint PK does nothing — no table FKs into
`price_history`, and the natural key `(item_id, timestamp, source)` already has a
unique index.

- Drop the `id` column and its 80 MB pkey index.
- Promote `(item_id, timestamp, source)` to the primary key. The PK index is built
  fresh, which also discards accumulated bloat in the old 256 MB unique index.
- Drop redundant secondary index `idx_price_history_item_timestamp` (prefix of the PK).
- `VACUUM FULL price_history` afterwards to rewrite the heap (reclaims the dropped
  column plus dead space from the Phase 1a delete of 553K rows).

All existing writers are compatible: every `INSERT INTO price_history` uses an
explicit column list and `ON CONFLICT (item_id, timestamp, source)`, which the PK
now arbitrates.

**Applied 2026-07-07.** Result: **589 MB → 390 MB** (price_history: 231 MB heap +
138 MB PK index; was 231 + 336). Notes from the apply:

- `alembic upgrade head` failed with `DiskFull` — the single-transaction migration
  builds the new PK index before the dropped-index space is released at commit,
  and the free-tier disk can't hold both. Alembic rolled back cleanly.
- The steps were re-applied manually in **separate autocommit transactions**
  (drop `id` column → drop unique constraint → add PK), then
  `alembic stamp 0006_price_history_composite_pk`. Migration 0006 remains correct
  as written for any environment with normal disk headroom.
- `VACUUM FULL price_history` also failed with `DiskFull` (needs a ~370 MB
  temporary copy). Deferred — it would reclaim maybe another 30–60 MB of heap
  (dropped column + dead tuples). A plain `VACUUM ANALYZE` ran instead; dead
  space is reusable by future inserts. Retry VACUUM FULL if quota pressure
  returns, ideally right after a retention prune.
- Verified after apply: 2,812,516 rows intact, `ON CONFLICT` upsert path works
  against the new PK, `alembic current` = `0006` (head).
- Migration 0005 was also unapplied on Supabase and would have crashed on the
  missing analysis tables; it's now guarded to skip absent tables and ran clean.

### 2. Deferred — `source` as smallint lookup

A `smallint` source code (with a lookup table for labels) would shrink both the heap
and the PK index meaningfully — the source string ('steam_historical',
'aggregator_sync', …) is repeated in all 2.8M rows and in the index.

**Deferred because the blast radius is large**: `collectors/pipeline.py` builds
dynamic labels (`historical_fallback:{source}`) and multiple queries pattern-match
with `LIKE 'historical_fallback:%'` / `LIKE 'synthetic_demo'`. Converting to
smallint requires reworking the fallback-labeling scheme (e.g., a separate
`is_fallback` flag + source FK), touching raw SQL in `pipeline.py`,
`free_data_importer.py`, `migrate_historical_data.py`, and several scripts. Do this
as its own change if fix #1 + #3 aren't enough.

### 3. Medium term — cap derived-data growth

Make the analysis tables (`daily_analysis`, `item_forecasts`, `trend_indicators`,
event tables) **upsert-latest** (one row per item) instead of append-only history.
That bounds them at ~20K rows each, roughly forever. If analysis history is wanted,
keep it locally or in flat files — nothing in the serving path needs last month's
forecast.

### 4. Structural — hot/cold split

Treat Supabase as a **serving layer only**: live `aggregator_sync` data plus a
rolling window (12–18 months) of daily history, plus latest analysis results. Full
history lives locally (existing SQLite, or DuckDB/parquet — which pandas reads
faster anyway), and the nightly batch jobs run against *that* instead of the remote
DB. This respects the "don't downsample, don't delete years" decisions — nothing is
destroyed; it just doesn't all live in the 500 MB tier. It also dissolves the Aiven
performance concerns, since heavy scans never touch the hosted DB.

## Two-tier price collection (applied 2026-07-08)

Only 5,525 of 19,321 items have CSMarketAPI historical data; 12,328 more had
only live aggregator snapshots (247K rows) that can't support analysis.

- Their live rows were **archived** to `runtime/archived_live_prices_2026-07-08.db`
  (35.5 MB, includes item names/slugs for re-import) and deleted from Supabase.
- The aggregator collector is now **two-tier**: items with a historical series
  (`market_csgo`/`steam_historical`) accumulate daily history; all other items
  keep a single latest-snapshot row, replaced each run. Promotion is automatic —
  once a monthly CSMarketAPI run imports history for an item, the next
  collection starts accumulating for it, and CSMarketAPI backfills the
  un-collected gap.
- Also applied the same day: collection schedule cut from 4×/day to once at
  23:00 UTC (CSGOTrader refreshes ~21:40 UTC; the extra runs were 100%
  duplicates), and 117,990 existing intraday duplicates deleted.
- Daily write volume: ~71K rows → ~5.5K history rows + ~12K replaced snapshots.

**Open question**: local CSMarketAPI data covers only ~5.5K distinct items so
far. If that's CSMarketAPI's catalog ceiling (check `failed_items` and the next
monthly run) rather than backfill progress, most snapshot-tier items will never
be promoted — revisit whether the app should chart aggregator history for them
(the archive preserves what was collected through 2026-07-08).

## On the hosting options

- **Supabase Pro ($25/mo)**: right call only if zero engineering time is worth
  $300/yr — and at the projected ~400 MB/month, 8 GB buys ~2 years before this
  conversation happens again. Buying disk defers the problem; it doesn't fix it.
- **Aiven Free**: defers even less (5 GB ≈ 1 year) and adds real costs (N+1 refactor
  in `long_term_trend_analyzer.py`, two connection strings, 2–5× slower nightly
  runs). Skip — if the hot/cold split is done, Aiven isn't needed at all.
- CockroachDB / Neon / multiple free tiers: correctly ruled out in the strategy notes.

## Fix regardless of hosting

1. **Migration targeting**: Alembic previously ran against local SQLite instead of
   Supabase (why `daily_analysis` etc. never got created there). Pin the migration
   environment to the intended connection string explicitly, and check
   `alembic current` before/after every run.
2. **`long_term_trend_analyzer.py` N+1**: 14K individual queries; refactor to the
   bulk-query pattern `analyze_trends.py` already uses. Worth it even on Supabase.

## Bottom line

Do the free schema fixes now (composite PK — done; source smallint if needed later),
cap the analysis tables via upsert, and move batch analytics to read from the local
4.2 GB archive. That combination keeps Supabase Free viable indefinitely. Supabase
Pro is a reasonable "I don't want to do that work" escape hatch, but it's renting
time, not solving the problem.
