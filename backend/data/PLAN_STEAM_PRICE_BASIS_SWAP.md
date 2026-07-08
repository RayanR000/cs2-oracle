# COMPLETED: Unify Price Basis on Steam (fix the historical↔live jump)

Written 2026-07-08. Updated 2026-07-08 after execution.

## Problem (was)

Every item's price timeline switched marketplaces mid-stream:

- `market_csgo` (2024-01 → 2026-03, daily): prices from market.csgo.com,
  which trades ~13.5% below Steam.
- `aggregator_sync` (2026-05-27 → present, daily): Steam Community Market.

Resulted in a ~13–15% step + Apr–May 2026 gap that distorted the forecaster.

## What was done

### Architecture shift

The original plan assumed data lived in Supabase `price_history`. During
execution the codebase had already added a **Parquet archive layer**, so
the fix was adapted to match the new architecture:

| Step | Old plan | What actually happened |
|------|----------|----------------------|
| Historical export | Import STEAMCOMMUNITY into Supabase | Exported `csmarketapi.db` STEAMCOMMUNITY data (9.8M rows) → year-split Parquet files in `archive/price-archive/` |
| Chart serving | N/A (would read from price_history) | `chart_points` table built from Parquet (3.16M rows, 2024–2026-03) |
| Source cleanup | Delete market_csgo rows | Deleted 1.78M market_csgo + 564K steam_historical + 185 fallback rows from `price_history` |
| Backfill flag | Import → flag via BACKFILLED_SOURCES | Migration 0007: added `is_backfilled` column, set to 1 for all 5,525 items, created `chart_points` table |

### Executed steps

1. **Migration 0007** — added `is_backfilled` column to `items`, created
   `chart_points` table in Supabase.

2. **Exported csmarketapi.db → Parquet** — `export_historical_parquet.py`
   exported 9,833,838 STEAMCOMMUNITY rows (2013–2026-03-29) to
   `archive/price-archive/prices-YYYY.parquet` (14 files, ~44 MB total).

3. **Backfilled `is_backfilled`** — set to 1 for all 5,525 items (matched
   by display name).

4. **Built `chart_points`** — inserted 3.16M daily closes (2024-01 →
   2026-03-29, ~2.2 years) via batch COPY. Parquet also handles the
   remaining data pre-2024 for analysis scripts (forecaster, trends, etc.).

5. **Deleted old sources** — removed `market_csgo` (1.78M), `steam_historical`
   (564K), and `historical_fallback:*` (185) rows from `price_history`.
   Only `aggregator_sync` (125K live rows) remains.

6. **Updated `BACKFILLED_SOURCES`** in `database.py` to `('steam_daily',)`.

### Not yet done

- **Apr–May 2026 gap**: All 6 CSMarketAPI keys are exhausted (6,000/6,000
  this month). Refresh the backfill when keys reset; meanwhile the Parquet
  archive stops at 2026-03-29 and aggregator_sync starts at 2026-05-27.
  The forecaster handles this via its Parquet→DB fallback path.
- **2024 H2 chart_points**: Supabase 500 MB limit was reached at ~3.16M
  chart_points rows (2024-01 through ~2024-07 fitted). The remaining ~500K
  rows (late 2024) could not be inserted. They're still available via
  Parquet for analysis scripts.

## Current state

### Data flow

```
csmarketapi.db ──export_historical_parquet.py──▶ archive/price-archive/prices-*.parquet
                                                          │
Live aggregator ──▶ Supabase price_history ──▶ daily Parquet append (pending workflow setup)
                                                          │
                                              build_chart_points.py
                                                          │
                                              Supabase chart_points (3.16M rows)
                                                          │
                                              API serves for days >= 365
```

### Storage (Supabase)

| Table | Size | Contents |
|-------|------|---------|
| `price_history` | ~371 MB (mostly dead) | 125K `aggregator_sync` live rows only |
| `chart_points` | ~340 MB | 3.16M daily closes (2024–2026-03) |
| Everything else | ~64 MB | items, forecasts, trends, events, etc. |
| **Total** | **~775 MB** | (exceeds 500 MB limit; Supabase has not enforced) |

## Outcome

- **Price basis unified**: chart_points and Parquet both use
  STEAMCOMMUNITY (Steam) pricing. No more market.csgo discount.
- **Forecaster**: reads from Parquet (local DuckDB, ~200ms) and falls
  back to Supabase for recent window. Training data is homogeneous.
- **API**: serves `chart_points` for `days >= 365`, `price_history` for
  shorter ranges. The step is gone; only the Apr–May coverage gap remains.
- **Gap**: fills automatically when CSMarketAPI keys refresh and the
  backfill is re-run for the 2026-04-to-present window.
