# Database Optimization — 2026-07-08

## Problem
Supabase database was 775 MB, exceeding the 500 MB free tier limit. Historical data was supposed to live in Parquet files on the `data-archive` branch, but the database was bloated with duplicates and dead space.

## Root Causes

### 1. Redundant index on `chart_points` (~100 MB wasted)
The composite PK `(item_id, day)` already creates a unique B-tree index (`chart_points_pkey`). An explicit `Index('idx_chart_point_item_day', 'item_id', 'day')` on the same columns was identical, doubling the index size to 204 MB.

**Fix**: Dropped `idx_chart_point_item_day`. Removed the redundant `Index()` from `ChartPoint.__table_args__` in `database.py:96-98`.

### 2. Old `price_history` rows never cleaned (~350 MB)
When the Parquet architecture was introduced, the pipeline was changed to route backfilled items' daily prices to CSV → Parquet instead of `price_history`. But the old rows (124,623 `aggregator_sync` rows for all 5,525 backfilled items) were never deleted. The DELETE in `pipeline.py:220-230` only runs for snapshot-tier item IDs — since all items were backfilled, it never executed.

**Fix**: Deleted all `price_history` rows where `item_id IN (SELECT id FROM items WHERE is_backfilled = 1)`. Then `VACUUM FULL price_history` to reclaim the disk space.

### 3. `trend_indicators` table grew unbounded (~6M+ rows/year)
Two scripts wrote to `trend_indicators` using pure INSERT with a unique `datetime.utcnow()` timestamp each time:
- `pipeline.run_feature_computation()` — daily, ~17,000 rows
- `LongTermTrendAnalyzer.run_analysis()` — weekly, ~17,000 rows

Despite a `UniqueConstraint(item_id, timestamp)`, the unique constraint never fired because each run had a different timestamp. The API only ever reads the **latest row per item**, so historical rows were dead weight.

**Fix**: Removed `trend_indicators` table entirely. It was redundant with `daily_analysis` (same metrics, more fields, proper UPSERT).

## Changes Made

### database.py
- Removed `trend_indicators` relationship from `Item` model
- Removed `TrendIndicator` class
- Removed redundant `Index('idx_chart_point_item_day', ...)` from `ChartPoint`

### pipeline.py
- Removed `run_feature_computation()` method entirely (was daily writer to `trend_indicators`, redundant with `analyze_trends.py`)

### run_task.py
- Removed `pipeline.run_feature_computation()` call from the "trends" task

### long_term_trend_analyzer.py
- Rewrote to write to `daily_analysis` table via UPSERT instead of `trend_indicators`
- Uses `ON CONFLICT (item_id, analysis_date) DO UPDATE` — bounded at 365 rows/item
- Removed pruning code (no longer needed)
- Removed `data_points` and `age_days` from output (not in `daily_analysis` schema)

### items.py (API)
- Removed `TrendIndicator` import
- Changed trend endpoint to read from `DailyAnalysis` instead of `TrendIndicator`
- Derived confidence from `momentum_score` instead of a stored `confidence` column
- Removed duplicate `latest_analysis` query

### Migrations

| File | What it does |
|---|---|
| `0008_drop_redundant_chart_point_index_and_cleanup.py` | Drops redundant index, deletes stale price_history rows |
| `0009_prune_old_trend_indicators.py` | Prunes trend_indicators > 90 days (replaced by 0010) |
| `0010_drop_trend_indicators.py` | Drops trend_indicators table entirely |

## Final Database State

| Before | After |
|---|---|
| 775 MB | **301 MB** |

### Tables in Supabase
| Table | Size | Rows | Growth |
|---|---|---|---|
| `chart_points` | 238 MB | 3,158,100 | UPSERT, bounded |
| `event_correlations` | 17 MB | 67,211 | Full rebuild weekly |
| `event_impacts` | 15 MB | 67,211 | Full rebuild weekly |
| `items` | 9.9 MB | 5,525 | Static |
| `item_forecasts` | 8.4 MB | 10,970 | UPSERT, bounded |
| `daily_analysis` | 1.8 MB | 4,313 | UPSERT, bounded |
| Others | ~11 MB | — | Static or negligible |

### Data Flow
```
Aggregator → prices → CSV → Parquet (data-archive branch)
                          → chart_points (daily closes for API serving)

Daily Analysis → Parquet (90-day) → daily_analysis (trend metrics)
Weekly Long-term Analysis → Parquet (3-year) → daily_analysis (overwrites daily)

Event Correlation → Parquet (all-time) → event_* tables (full rebuild)
ML Forecast → Parquet (all-time) → item_forecasts (comparison date check)
```

### Workflows
| Workflow | Schedule | Writes to |
|---|---|---|
| `aggregator-update` | 23:00 UTC daily | Parquet, chart_points, collection_runs |
| `daily-trend-analysis` | 03:00 UTC daily | daily_analysis |
| `long-term-trend-analysis` | 06:00 UTC Sunday | daily_analysis (overwrites) |
| `price-forecast` | 04:00 UTC daily | item_forecasts |
| `event-correlation-analysis` | 04:00 UTC Sunday | event_* tables |
| `discover-new-items` | Various | items |
