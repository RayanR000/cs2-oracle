# Database Cleanup & Historical Data Migration Plan

## Status

- ✅ Phase 0 completed: Removed 3 orphaned local DBs + dropped 6 computed Supabase tables (freed ~194 MB)
- ⏳ Phase 1: Clean Supabase (not started)
- ⏳ Phase 2: Import historical data (not started)
- ⏳ Phase 3: Recompute analysis tables (not started)

**Current Supabase usage: 189 MB / 500 MB**

---

## Phase 0: What Was Done (Completed)

### Removed Empty / Orphaned Local Databases

| File | Size | Reason |
|---|---|---|
| `backend/cs2_market.db` | 0 B | Empty stub — default dev SQLite path but never populated |
| `backend/data/market_catalog.db` | 0 B | Orphaned copy — real catalog at `runtime/market_catalog.db` |
| `backend/runtime/ssr_history.db` | 3.5 MB | Abandoned — SSR backfill never successfully ran; 0 rows in `price_history` |

### Removed Computed Tables from Supabase (freed ~194 MB)

| Table | Size | Script to Recompute |
|---|---|---|
| `daily_analysis` | 90 MB | `scripts/analyze_trends.py` |
| `item_forecasts` | 37 MB | `scripts/forecast_prices.py` |
| `event_impacts` | 25 MB | `scripts/event_analyzer.py` |
| `trend_indicators` | 23 MB | `scripts/long_term_trend_analyzer.py` |
| `event_correlations` | 15 MB | `scripts/event_analyzer.py` |
| `event_patterns` | 4 MB | `scripts/event_analyzer.py` |

---

## Phase 1: Clean Supabase

### 1a. Delete stale `price_history` sources (frees ~86 MB)

```sql
DELETE FROM price_history WHERE source IN (
  'kaggle_csgo',                                        -- 466K rows, ~72 MB
  'historical_fallback:kaggle_csgo',                    -- 50K rows, ~8 MB
  'historical_fallback:csgotrader',                     -- 19K rows, ~3 MB
  'historical_fallback:aggregator_sync',                -- 13K rows, ~2 MB
  'csgotrader'                                          -- 6K rows, ~1 MB
);
```

### 1b. Deduplicate `items` table (frees ~2 MB)

Delete 5,570 slug-format duplicates where `item_id` is a slug AND `classid` IS NULL.
Keep the non-slug entries that have `classid` populated.

```sql
DELETE FROM items
WHERE item_id ~ '^[a-z0-9-]+$'
  AND classid IS NULL
  AND name IN (
    SELECT name FROM items
    GROUP BY name HAVING COUNT(*) > 1
  );
```

### 1c. Enrich `items` with metadata from local DB

For each item matched by `name` ↔ `market_hash_name`:

| Supabase column | Source | Current state |
|---|---|---|
| `classid` | Local `items.classid` | 17,752 NULL → 100% populated |
| `type` | Local `category` + `type` + `sticker_type` | All "skin"/"case" → proper types (sticker, knife, glove, graffiti, etc.) |
| `icon_url` | — (not in local items) | Leave NULL |

---

## Phase 2: Import Historical Data

Both imports write to `price_history`. Item FK resolved by matching
local `market_hash_name` → Supabase `items.name`. Missing items are
created in `items` first (with auto-generated `item_id` slug and proper `type`).

### 2a. Import MARKETCSGO (daily, all 5 columns)

| Source | Period | Rows | Est. size |
|---|---|---|---|
| `sales_history WHERE market = 'MARKETCSGO'` | 2022–2026 | 2,126,719 | ~252 MB |
| Avg volume: 10/day | 100% complete (mean/min/max/median/volume) | | |

**Mapping:**

| Local field | Supabase field |
|---|---|
| `day` | `timestamp` |
| `mean_price` | `price` |
| `median_price` | `median_price` |
| `volume` | `volume` |
| — | `source` = `'market_csgo'` |

### 2b. Import STEAMCOMMUNITY pre-2022 (weekly downsampled)

| Source | Period | Rows | Est. size |
|---|---|---|---|
| `sales_history WHERE market = 'STEAMCOMMUNITY' AND day < '2022-01-01'` | 2013–2021 | ~3.9M daily → ~557K weekly | ~90 MB |
| Only median_price + volume | Downsampled to week-start averages | | |

**Mapping:**

| Local field | Supabase field |
|---|---|
| Week start date | `timestamp` |
| `AVG(median_price)` | `price` |
| `SUM(volume)` | `volume` |
| `AVG(median_price)` | `median_price` |
| — | `source` = `'steam_historical'` |

---

## Phase 3: Verify & Recompute

### 3a. Storage check

```
price_history:    kaggle (deleted) + MARKETCSGO (2.1M) +
                  steam_weekly (557K) + aggregator_sync (548K)
                = ~3.2M rows
items:           19,252 (deduplicated, enriched)
Total:          ~459 MB / 500 MB ✅ (41 MB free)
```

### 3b. Run computation pipelines (in order)

| Step | Script | Est. time |
|---|---|---|
| 1. `daily_analysis` + `trend_indicators` | `scripts/analyze_trends.py` + `scripts/long_term_trend_analyzer.py` | ~30–45 min |
| 2. `item_forecasts` | `scripts/forecast_prices.py` | ~60 min |
| 3. `event_impacts` / `event_patterns` / `event_correlations` | `scripts/event_analyzer.py` | ~30 min |

---

## Final Coverage Map

| Period | Steam data | MARKETCSGO data |
|---|---|---|
| **2013–2021** | ✅ Steam weekly (median+volume) | ❌ No data |
| **2022–2023** | ✅ Steam weekly | ✅ MARKETCSGO daily (all columns) |
| **2024** | ✅ Steam weekly | ✅ MARKETCSGO daily |
| **2025–Mar 2026** | ✅ Steam weekly | ✅ MARKETCSGO daily |
| **Mar–Jul 2026** | ❌ (local Steam stops Mar 2026) | ❌ (local MC stops Mar 2026) |
| **Jul 2026+** | ✅ Live via CSGOTrader aggregator | — |

---

## Local Databases Still Active

| DB | Size | Purpose | Used By |
|---|---|---|---|
| `runtime/csmarketapi.db` | 4.2 GB | Historical multi-market price data (7 markets, 2013–2026) | `collectors/csmarketapi_backfill.py` |
| `runtime/market_catalog.db` | 18 MB | Steam item catalog (32K items) | `build_market_catalog.py`, `repair_catalog_gaps.py` |
| `runtime/csmarketapi_reference.db` | 692 KB | Market definitions, FX rates, player counts | `csmarketapi_backfill.py` |

All 3 are gitignored via `backend/runtime/` in `.gitignore`.

---

## Market Data Comparison

| | STEAMCOMMUNITY | MARKETCSGO | WHITEMARKET | SKINPORT |
|---|---|---|---|---|
| **Rows** | 9.8M (64%) | 2.1M (14%) | 866K (6%) | 396K (3%) |
| **Items** | 5,542 | 5,527 | 5,537 | 5,466 |
| **Effective range** | 2013–2026 | 2022–2026 | 2022–2026 | 2025–2026 |
| **Columns** | median + volume only | **all 5** | mean + volume only | **all 5** |
| **Avg volume/day** | 590 | 10 | 5 | 13 |
| **Price vs Steam** | baseline | 0.865× (13.5% cheaper) | 0.914× (8.6% cheaper) | 1.129× (12.9% pricier) |
