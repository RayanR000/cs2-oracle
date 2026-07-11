# 2026-07-11: Multi-Source Aggregator Storage & Chart Points Cleanup

## Context

The aggregator was collecting prices from all 4 CSGOTrader endpoints (steam, skinport, buff163, csfloat) — yielding 8 source variants — but **only storing 1** (`aggregator_sync` / Steam 24h). The other 7 sources were fetched over the network then discarded.

`chart_points` was a 289 MB Postgres cache of daily OHLCV closes, entirely reconstructable from the parquet files.

## Changes Made

### 1. Dropped `chart_points` table (freed 290 MB)

**Migration:** `backend/migrations/versions/0012_drop_chart_points.py`
**DB size:** 356 MB → 66 MB

Parquet files confirmed to have all historical data (9.8M rows across 14 year-split files, 2013-2026).

**Removed from code:**
- `backend/database.py` — Removed `ChartPoint` model class
- `backend/api/routes/items.py` — Removed all 15+ chart_points queries. API now queries `price_history` exclusively. No more chart_points fallback for deep history.
- `backend/scripts/backtest_accuracy.py` — Removed chart_points loading path; backtest uses `price_history` directly
- `backend/scripts/append_to_parquet.py` — Removed `--skip-chart-points` flag and sync subprocess call
- `backend/scripts/build_chart_points.py` — Kept as manual utility (can rebuild from parquet if ever needed)

### 2. Multi-source parquet storage

**Before:** Only `aggregator_sync` (Steam 24h) was written to the backfilled CSV → OHLCV parquet

**After:** All 8 sources are written to the backfilled CSV with a `source` column:

| Source | Endpoint | Label |
|---|---|---|
| Steam 24h | `steam.json` | `aggregator_sync` |
| Steam 7d | `steam.json` | `aggregator_steam_7d` |
| Steam 30d | `steam.json` | `aggregator_steam_30d` |
| Steam 90d | `steam.json` | `aggregator_steam_90d` |
| Skinport | `skinport.json` | `aggregator_skinport` |
| Buff163 sell | `buff163.json` | `aggregator_buff163` |
| Buff163 buy | `buff163.json` | `aggregator_buff163_buy` |
| CSFloat | `csfloat.json` | `aggregator_csfloat` |

**Files changed:**
- `backend/collectors/pipeline.py:249-261` — Backfilled CSV now includes all sources with a `source` column (schema: `item_slug, day, source, price, volume`)
- `backend/scripts/append_to_parquet.py:117-128` — OHLCV aggregation groups by `(item_slug, day, source)` instead of filtering to `aggregator_sync`
- `backend/scripts/append_to_parquet.py:158-176` — `_append_parquet` handles schema migration: old parquet rows without `source` column get `source = "aggregator_sync"` on append

### New parquet schema

```
prices-YYYY.parquet
├── item_slug          ← "AK-47 | Redline (Field-Tested)"
├── day                ← 2026-07-11
├── source             ← "aggregator_sync" | "aggregator_skinport" | ...
├── mean_price
├── min_price
├── max_price
├── median_price
└── volume
```

One OHLCV row per `(item_slug, day, source)`. Query example: `WHERE source = 'aggregator_skinport'`.

### 3. Supabase writes left unchanged

All 8 sources still only go to the parquet archives. Only `aggregator_sync` + `historical_fallback:*` continue to write to Supabase `price_history`.

## Supabase State After Changes

| Table | Size | Rows |
|---|---|---|
| event_correlations | 17 MB | 67,211 |
| event_impacts | 15 MB | 67,211 |
| items | 9.9 MB | 5,525 |
| item_forecasts | 8.4 MB | 10,970 |
| **price_history** | **2.3 MB** | **16,487** |
| daily_analysis | 1.8 MB | 4,313 |
| event_patterns | 1 MB | 4,000 |
| collection_runs | 168 KB | 173 |
| events | 104 KB | 79 |
| prediction_accuracy | 96 KB | 6 |
| *(chart_points dropped)* | — | — |
| **Total** | **66 MB** | |

## Storage Estimate (All Sources, Daily Collection)

| | Current (1 source) | All 8 sources |
|---|---|---|
| Rows per day | ~5,500 | ~37,000 |
| Rows per year | ~2M | ~13.5M |
| Parquet size/year | ~30 MB | ~200 MB |

## Data Coverage

CSGOTrader endpoints cover **34,076 items** across all categories:

- Skins/Knives/Gloves: 15,206
- Stickers: 14,849
- Graffiti: 1,825
- Agents/Collectibles: 1,003
- Cases/Capsules: 664
- Music Kits: 197
- Pins/Coins/Trophies: 130
- Patches: 124
- Other: 77

Not covered: individual pattern seeds, individual float values (only 5 wear tiers), non-marketable items, zero-volume items.

## Key Commands

```bash
# Run migration
DATABASE_URL='<supabase-url>' alembic upgrade head

# Run aggregator collection (all sources → parquet)
cd backend && python scripts/run_task.py aggregate

# Append today's data to parquet
python scripts/append_to_parquet.py --date $(date -u +%F) --out-dir ../archive \
  --snapshot-csv /tmp/aggregator-snapshots-$(date -u +%F).csv
```
