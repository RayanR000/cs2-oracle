# CSMarketAPI backfill and retrain (2026-09-20)

## What

Backfilled 988 items from CSMarketAPI using the last 1,000 monthly API calls,
exported 3.2M rows to the price archive as 7 new `csmarketapi_*` sources, and
retrained. The new model is deployed.

## Data

| Metric | Value |
|--------|-------|
| Items fetched | 988 (275 top-volume + 669 high-value $10+ + 32 misc + 17 knives/gloves) |
| Price rows | 3,227,588 |
| Date range | 2013-08-15 to 2026-03-29 |
| Markets | 7 (Steam, CSFloat, MarketCSGO, WhiteMarket, Skinport, SkinBaron, CSDeals) |
| Sources added | `csmarketapi_steam`, `csmarketapi_csfloat`, `csmarketapi_marketcsgo`, `csmarketapi_whitemarket`, `csmarketapi_skinport`, `csmarketapi_skinbaron`, `csmarketapi_csdeals` |

Selection was iteratively refined:
1. All-time volume → wrong (dominated by sub-$1 stickers).
2. 2026 volume, sparse coverage → wrong (0% training overlap, items were new not sparse).
3. 2026 volume, trainable items with pre-2026 history, ≥$1 → correct.
4. Second pass: $10+ high-value items (knives, gloves, premium skins).

## Training impact

| | Before | After |
|---|---|---|
| Trainable items | 13,569 | 17,329 (+3,760) |
| Pre-2026 training rows | 1.0M | 11.7M (+10.7M) |
| Sources per item (pre-2026) | 1.0 avg | 7.6 avg |

Training time unchanged (~7 min) due to 300K row sampling cap.

## Model comparison (Sep 14 → Sep 20)

| Horizon | Rank IC (clean) | Edge vs naive | Notes |
|---------|----------------|---------------|-------|
| 3d | 0.1198 → 0.1269 | 0.0787 → 0.0747 | |
| 7d | 0.1047 → **0.1479** | 0.0659 → **0.0981** | "ML subtracts from best feature" warning gone |
| 14d | 0.1062 → **0.1486** | 0.0757 → **0.1157** | Warning gone |
| 30d | 0.0275 → **0.1022** | 0.0107 → **0.0824** | Warning gone; was near-useless, now has real skill |

Band widths slightly wider (+0.2-1.2pp) at matched 80% coverage — the model is more
honest about uncertainty with cleaner labels.

## Code changes

- `csmarketapi_backfill.py`: readonly SQLite fix, 0.1s delay, `--only-catalog` flag (PR #60).
- `VOTED_CACHE_VERSION` bumped v8 → v9.
- Raw data in `backend/runtime/csmarketapi.db` (988 items, 31K catalog).

## Remaining

305 knives/gloves checkpointed at `★ Ursus Knife | Marble Fade (Factory New)`.
Resume with `--only-catalog` when the monthly key resets.
