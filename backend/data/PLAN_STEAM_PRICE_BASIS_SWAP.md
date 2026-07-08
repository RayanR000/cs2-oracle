# Plan: Unify Price Basis on Steam (fix the historical↔live jump)

Written 2026-07-08. Follow-up to `DB_SCHEMA_FIX_RECOMMENDATIONS.md`.

## Problem

Every item's price timeline switches marketplaces mid-stream:

- `market_csgo` (2024-01 → 2026-03, daily): prices from market.csgo.com,
  which trades ~13.5% below Steam (measured 0.865× in the market data
  comparison — see `DB_CLEANUP_AND_MIGRATION_PLAN.md`).
- `aggregator_sync` (2026-05-27 → present, daily): CSGOTrader's
  **Steam Community Market** prices.

Result: a ~13–15% level step sitting on top of the Apr–May 2026 gap.
Cosmetic on the chart (separate lines per source), but the **forecaster
trains on a 365-day window that concatenates both sources into one
series**, so the step and gap distort its features and targets.

## Fix (recommended): swap the serving series to STEAMCOMMUNITY

The local archive `backend/runtime/csmarketapi.db` holds the
STEAMCOMMUNITY series (9.8M rows, 2013–2026) for the same 5,525 items —
the same price basis as the live feed.

### Steps

1. **Refresh the local backfill first** (closes the Apr–May 2026 gap).
   Local data stops 2026-03-29 only because that's when
   `collectors/csmarketapi_backfill.py` last ran — CSMarketAPI has the
   later data. Re-run the backfill for the existing 5,525 items as part of
   the monthly run. Cost: ~1 request/item against the 6 × 1,000/month key
   budget. This also creates an overlap window with `aggregator_sync` to
   validate the two Steam series agree.

2. **Import STEAMCOMMUNITY daily rows (2022-01 → present) into Supabase**
   for the 5,525 items, `source = 'steam_daily'` (or extend
   `steam_historical`). Columns: median_price → price/median_price,
   volume → volume (STEAMCOMMUNITY has no mean/min/max). Reuse the
   import path in `scripts/migrate_historical_data.py` (name-matched,
   `ON CONFLICT (item_id, timestamp, source) DO NOTHING`, COPY-based).

3. **Delete the 2024–2026 `market_csgo` rows** (1.78M rows) after
   verifying the Steam import row counts per item. Roughly size-neutral
   overall; VACUUM ANALYZE afterwards. market_csgo data remains in the
   local archive for offline cross-market analysis — nothing is lost.

4. **Update `BACKFILLED_SOURCES`** in `backend/database.py` (currently
   `('market_csgo', 'steam_historical')`) to the new source label(s) —
   the listing filter, two-tier collector, and downsample guard all key
   off it. Also update `SOURCE_CHART_META` in
   `frontend/app/items/[id]/page.tsx` and `SOURCE_META` in
   `frontend/components/PriceSourceFilter.tsx`.

5. **Verify**: chart shows one continuous Steam-basis line into the live
   feed with no step; forecaster trains next Monday on a homogeneous
   series; check the aggregator↔steam overlap agreement (should be ~1.0×).

### Storage accounting (do the math before step 2)

Current DB: ~436 MB. The swap should be roughly neutral (steam daily
2024–2026 ≈ market_csgo row count for the same items/period), plus
2022–2023 fills the current coverage hole (~2 years × 5.5K items daily ≈
+3–4M rows if imported daily — **that exceeds quota**). Options:
- Import 2022–2023 as weekly (like the pre-2022 data): ~575K rows, fits.
- Or keep 2022–2023 out of Supabase entirely for now.
Decide based on headroom at execution time; the hot/cold split
(recommendations doc) remains the long-term answer.

## Alternative (not recommended): rescale market_csgo in place

One-time `UPDATE` multiplying `market_csgo` prices by a **per-item ratio**
computed from the local steam↔market_csgo overlap (a global 1/0.865 is too
crude — the discount varies with item liquidity). No import work and keeps
mean/min/max columns, but permanently rewrites observed prices into
synthetic ones and the ratio drifts over time. Only pick this if the
5-column data must stay in Supabase.

## Status

- [ ] Backfill refresh (Apr → present) — runs with the monthly key budget
- [ ] STEAMCOMMUNITY daily import (2024 → present; decide 2022–2023 policy)
- [ ] Delete market_csgo rows after verification
- [ ] Update BACKFILLED_SOURCES + frontend source metadata
- [ ] Post-swap verification (chart continuity, forecast retrain, overlap check)
