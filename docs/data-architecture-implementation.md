# Data Architecture Implementation

## What changed

### Motivation

Supabase has a 500 MB limit. The `price_history` table held 15.2M daily OHLCV rows (~4.2 GB originally, pruned to stay under limit). Analysis scripts needed full daily granularity for SMA-7, momentum, event impact, and 7-day forecasts, but Supabase couldn't hold it all. The local `csmarketapi.db` had the full data but wasn't backed up.

### Solution

Move full historical data to Parquet files on the `data-archive` branch. Supabase becomes a lean serving layer. Analysis scripts read training data from local Parquet via DuckDB instead of querying Supabase over the network.

---

## New tables

### `chart_points` (Supabase)

Daily close price per item for all-time charts. One row per item per day. ~4M rows, ~50 MB. This is what the API serves for chart data.

Columns: `item_id`, `day`, `close`

### `prices-YYYY.parquet` (archive)

Full daily OHLCV per item, split by year. Columns: `item_slug`, `day`, `mean_price`, `min_price`, `max_price`, `median_price`, `volume`. ~15M rows total, ~10-15 MB per year.

---

## Schema changes

### `items` table — added `is_backfilled` column

```python
is_backfilled = Column(Integer, default=0)
```

Boolean flag: 1 = item has CSMarketAPI historical series (backfilled). Replaces the old pattern of scanning `price_history` for `source IN ('market_csgo', 'steam_historical')` on every listing query.

### `backfilled_item_clause()` rewritten

Before: `EXISTS (SELECT 1 FROM price_history WHERE item_id=Item.id AND source IN ('market_csgo','steam_historical'))`

After: `Item.is_backfilled == True`

All listing endpoints (`list_items`, `search_items`, `trending_items`, `market_summary`) that use this clause now filter via the indexed column instead of a correlated subquery.

### Migration

New file: `backend/migrations/versions/0007_add_is_backfilled_and_chart_points.py`
- Adds `is_backfilled INTEGER DEFAULT 0` to `items`
- Creates `chart_points` table with composite PK `(item_id, day)`

---

## New scripts

### `scripts/export_historical_parquet.py`

**Purpose:** One-time export of `csmarketapi.db` → year-split Parquet files.

**What it does:** Reads the local SQLite database (15.2M OHLCV rows, 4.2 GB), groups by year, writes each year as a separate Parquet file to `archive/price-archive/prices-YYYY.parquet`.

**Usage:** `python scripts/export_historical_parquet.py --db-path csmarketapi.db --out-dir ../archive`

### `scripts/build_chart_points.py`

**Purpose:** Read Parquet archive → upsert daily close into Supabase `chart_points` table.

**What it does:** Queries `items WHERE is_backfilled = 1` for slug→ID mapping, reads Parquet files via DuckDB, batches upserts into `chart_points` with `ON CONFLICT (item_id, day) DO UPDATE`.

**Usage:** `python scripts/build_chart_points.py --parquet-dir ../archive/price-archive`

### `scripts/append_to_parquet.py`

**Purpose:** Daily: append today's aggregator rows to the current year's Parquet file, then sync to `chart_points`.

**What it does:** Reads snapshot-tier rows from Supabase `price_history` and backfilled rows from a CSV (written by the aggregator), collapses to daily OHLCV via groupby, deduplicates, appends to `prices-YYYY.parquet`, then calls `build_chart_points.py` for today's date.

**Usage:** `python scripts/append_to_parquet.py --date 2026-07-08 --out-dir ../archive --backfilled-csv /tmp/aggregator-backfilled-2026-07-08.csv`

---

## Modified files

### `backend/database.py`

- Added `Item.is_backfilled` column
- Added `ChartPoint` model (`item_id`, `day`, `close`)
- Rewrote `backfilled_item_clause()` to use `Item.is_backfilled == True`
- Kept `BACKFILLED_SOURCES` constant for backward compat

### `backend/api/routes/items.py`

- Import `ChartPoint`
- `get_price_history`: for `days >= 365`, reads from `chart_points` instead of `price_history`
- `get_multi_source_prices`: for `days >= 365` or `source=historical`, reads from `chart_points`
- Added `_latest_prices()` helper: reads latest price from `chart_points`, falls back to `price_history` for snapshot items
- `_build_trending`: uses `_latest_prices()` instead of querying `PriceHistory` directly
- `get_item_trends`: falls back to `ChartPoint` when `PriceHistory` returns nothing
- `get_item_prediction`: falls back to `ChartPoint` when `PriceHistory` returns nothing

### `backend/api/routes/market.py`

- No code change needed — `DailyAnalysis.current_price` is already the primary price source in `_build_market_summary`, and that still works

### `backend/collectors/pipeline.py`

**Removed dead code:**
- Entire scheduler (`start()`, `stop()`, `get_scheduled_jobs()`) — never called in production, relied on `BackgroundScheduler`/`CronTrigger` from APScheduler
- `PipelineMonitor` class — never instantiated
- `run_daily_collection()` — used `SteamMarketCollector`, never called by any workflow

**Changed price storage logic in `run_full_aggregator_collection()`:**
- Before: all items (backfilled + snapshot) written to `price_history`
- After: items with `is_backfilled == True` are written to `/tmp/aggregator-backfilled-YYYY-MM-DD.csv` instead of `price_history`
- Only snapshot-tier items get written to `price_history` (with old rows deleted first)
- The backfilled CSV path is returned in the result dict

### `backend/models/forecaster.py`

- `fetch_price_history()`: if Parquet archive exists and `days_back > 14`, reads from Parquet via DuckDB instead of Supabase
- Falls back to Supabase query for short windows or if archive is absent

### `backend/scripts/analyze_trends.py`

- Added module-level `_load_from_parquet()` helper using DuckDB
- `get_recent_price_history_bulk()`: tries Parquet for windows > 14 days
- `get_item_price_history()`: tries Parquet for windows > 14 days
- Falls back to Supabase for short windows or if archive missing

### `backend/scripts/event_analyzer.py`

- `batch_load_prices()`: tries Parquet for ranges > 60 days
- Reads slug→ID mapping from items table, queries Parquet via DuckDB
- Falls back to Supabase for short ranges or if archive missing

### `backend/scripts/long_term_trend_analyzer.py`

- `get_first_seen_bulk()`: reads from Parquet when archive exists, with slug→ID mapping
- `get_price_history_bulk()`: reads from Parquet when archive exists
- Falls back to Supabase queries

### `backend/scripts/run_task.py`

- Removed `prune` task (pruning scripts deleted)
- Updated task list

### `backend/requirements.txt`

- Added `pyarrow>=14.0.0`
- Added `duckdb>=0.10.0`

### `.github/workflows/aggregator-update.yml`

- Removed CSV export step (redundant with Parquet)
- Parquet step now passes `--backfilled-csv /tmp/aggregator-backfilled-$(date -u +%F).csv`

---

## Deleted files

| File | Reason |
|---|---|
| `backend/scripts/prune_database.py` | No longer needed — Supabase is under 70 MB, 500 MB limit won't be hit for years |
| `backend/tests/test_pruning.py` | Tested the deleted pruning script |
| `.github/workflows/db-maintenance.yml` | Triggered the deleted pruning task |

---

## Data flow (after)

### Daily aggregator run

```
GitHub Actions (23:00 UTC)
  └─ run_task.py aggregate
       ├─ Fetch prices for all 5,500 items from CSGOTraderAggregator
       ├─ Backfilled items → /tmp/aggregator-backfilled-YYYY-MM-DD.csv
       ├─ Snapshot items   → Supabase price_history (INSERT, old rows deleted)
       └─ Record CollectionRun

  └─ Checkout data-archive branch → archive/

  └─ append_to_parquet.py --backfilled-csv /tmp/...csv
       ├─ Read snapshot items from Supabase price_history
       ├─ Read backfilled items from CSV
       ├─ Merge, collapse to daily OHLCV
       ├─ Append to archive/price-archive/prices-YYYY.parquet
       └─ Run build_chart_points.py → upsert today's close into chart_points

  └─ Commit & push updated Parquet files to data-archive branch
```

### Analysis reads

```
Analysis scripts (trends, forecasts, events)
  └─ DuckDB + read_parquet('archive/price-archive/prices-*.parquet')
       └─ 90-day or 365-day or full history ← local, ~200ms
  └─ Compute results (trend_indicators, daily_analysis, item_forecasts, etc.)
  └─ Write results to Supabase
```

### API serving

```
GET /items/{id}/price-history
  ├─ days < 365  → Supabase price_history (last 7 days of raw data)
  └─ days >= 365 → Supabase chart_points (daily close, all time)

GET /items/{id}/prices
  ├─ days < 365  → Supabase price_history (raw multi-source)
  └─ days >= 365 → Supabase chart_points (historical daily close)

GET /items/{id}/trends → Supabase trend_indicators + daily_analysis (already computed)
GET /items/{id}/prediction → Supabase item_forecasts (already computed)
GET /items/list → Supabase items (filtered by is_backfilled)
GET /market/summary → Supabase items + daily_analysis + price_history (snapshot only)
```

---

## Supabase storage breakdown

| Table | Row count | Est. size | Notes |
|---|---|---|---|
| `items` | ~5,500 | ~2 MB | Static catalog |
| `price_history` | ~few hundred | ~1 MB | Snapshot-tier only, 7-day rolling |
| `chart_points` | ~4M | ~50 MB | Daily closes, never pruned |
| `daily_analysis` | ~5,500/day | ~5 MB/year | Computed, kept indefinitely |
| `trend_indicators` | ~5,500/day | ~3 MB/year | Computed, kept indefinitely |
| `item_forecasts` | ~5,500/day | ~2 MB/year | Computed, kept indefinitely |
| `events` + impacts | ~1,000 | ~5 MB | Static |
| **Total** | | **~70 MB** | |

Well within the 500 MB Supabase limit with room to grow for years.

---

## Performance

| Operation | Before | After |
|---|---|---|
| Analysis (Actions runner) | Supabase query over network (~2-5s) | DuckDB local Parquet (~200ms) |
| API chart response | `price_history` query (~100-500ms) | `chart_points` query (~50ms) |
| API listing filter | Correlated EXISTS subquery | `is_backfilled` column index |
| Aggregator workflow | Same + pruning | Same - pruning + ~10s Parquet append |

---

## Removed dead code

Removed ~320 lines across:
- Scheduler jobs in `pipeline.py` (`start()`, `stop()`, `get_scheduled_jobs()`) — the in-process APScheduler was never started in production; all tasks are triggered via GitHub Actions
- `PipelineMonitor` class — never instantiated anywhere
- `run_daily_collection()` — used `SteamMarketCollector`, never called by any workflow
- `prune` task in `run_task.py`
- Whole `prune_database.py` script and its test file
- Whole `db-maintenance.yml` workflow
